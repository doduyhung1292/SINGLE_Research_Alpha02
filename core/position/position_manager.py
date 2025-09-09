import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set, TypedDict

from models.model_position import get_open_positions_from_db
from models.model_trade_symbols import get_all_symbols
from core.config.constants import CONFIG, DEFAULT_CONFIG, Position
from core.data.market_data_manager import get_latest_price_from_ohlcv
from core.api.exchange_api import get_current_mark_price
from calculators import calculate_bollinger_bands, calculate_new_spread

logger = logging.getLogger(__name__)


class PositionData(TypedDict, total=False):
    """Type definition for position data dictionary"""

    id: str
    symbol: str
    quantity: float
    entryPrice: float
    positionSide: str
    status: str
    entry_spread: float
    exit_spread: Optional[float]
    entry_time: str
    exit_time: Optional[str]
    exit_reason: Optional[str]
    pnl: Optional[float]
    entry_window: int
    bb_entry_multiplier: float
    risk_parity_weight: float


def get_open_order_for_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    """Get open position for a specific trading pair."""
    try:
        positions = get_open_positions_from_db()
        for position in positions:
            if position["symbol"] == symbol:
                return position
        return None
    except Exception as e:
        logger.error(f"Error getting open order for pair {symbol}: {e}")
        logger.error(traceback.format_exc())
        return None


def calculate_position_pnl(
    position: Dict[str, Any], symbol: str, symbol_data: Dict[str, List]
) -> float:
    """
    Calculate the current PnL for a position.

    Args:
        position: Position data
        symbol: First symbol in pair
        symbol_data: Market data for all symbols

    Returns:
        float: Current PnL
    """
    try:
        trade_symbol = symbol
        if not trade_symbol:
            return 0

        # Get current price
        current_price_data = get_latest_price_from_ohlcv(trade_symbol, symbol_data)

        # Extract current price
        if isinstance(current_price_data, dict) and "close" in current_price_data:
            current_price = current_price_data["close"]
        else:
            current_price = float(current_price_data) if current_price_data else 0

        if current_price == 0:
            return 0

        # Get entry price and quantity
        if trade_symbol:
            entry_price = float(position.get("entryPrice", 0))
            quantity = float(position.get("quantity", 0))
            position_side = position.get("positionSide")
    
        # Calculate PnL
        if position_side == Position.LONG:
            pnl = (current_price - entry_price) * quantity
        else:  # SHORT
            pnl = (entry_price - current_price) * quantity

        return pnl

    except Exception as e:
        logger.error(f"Error calculating position PnL: {e}")
        logger.error(traceback.format_exc())
        return 0


def calculate_bands(
    open_position: Optional[Dict[str, Any]], spreads: List[float], multiplier: float
) -> Dict[str, Any]:
    """
    Calculate Bollinger Bands for a spread.

    Args:
        open_position: Open position data or None for new entries
        spreads: List of spread values
        multiplier: Multiplier for standard deviation

    Returns:
        Dict[str, Any]: Dictionary containing middle_band, upper_band, lower_band
    """
    try:
        # Ensure window is an integer
        if open_position and "entry_window" in open_position:
            window = int(open_position.get("entry_window"))
        else:
            # Get window from config if not in open_position
            window = int(CONFIG.get("window", DEFAULT_CONFIG["window"]))

        # Calculate bands
        middle_band, upper_band, lower_band = calculate_bollinger_bands(
            spreads, window, multiplier
        )

        # Return as dictionary - using .iloc[-1] for pandas Series
        return {
            "middle_band": middle_band.iloc[-1] if len(middle_band) > 0 else None,
            "upper_band": upper_band.iloc[-1] if len(upper_band) > 0 else None,
            "lower_band": lower_band.iloc[-1] if len(lower_band) > 0 else None,
        }
    except Exception as e:
        logger.error(f"Error calculating bands: {e}")
        logger.error(traceback.format_exc())
        return {"middle_band": None, "upper_band": None, "lower_band": None}


def check_stop_loss_for_open_positions(
    open_positions: List[Dict[str, Any]],
    symbol_data: Dict[str, List],
    current_equity: float,
) -> List[Dict[str, Any]]:
    """
    Check if any open positions have losses exceeding the maximum loss threshold.

    Args:
        open_positions (List[Dict[str, Any]]): List of open positions
        symbol_data (Dict[str, List]): OHLCV data for all symbols
        current_equity (float): Current account equity

    Returns:
        List[Dict[str, Any]]: List of positions that should be closed
    """
    try:
        # Get the current config
        current_config = CONFIG
        max_loss_pct = current_config.get(
            "max_loss_pct", DEFAULT_CONFIG["max_loss_pct"]
        )

        positions_to_close = []

        for position in open_positions:
            try:
                # Đảm bảo chỉ xử lý các vị thế OPEN
                if position.get("status") != "OPEN":
                    logger.info(
                        f"Skipping non-OPEN position {position.get('symbol', 'unknown')} with status {position.get('status')} in stop loss check"
                    )
                    continue

                symbol = position.get("symbol")

                # Skip if missing symbols
                if not symbol:
                    continue

                # Calculate current PnL
                pnl = calculate_position_pnl(position, symbol, symbol_data)

                # Calculate loss as percentage of equity (as a ratio, not percent)
                loss_ratio = abs(pnl) / current_equity if pnl < 0 else 0

                # Log for debugging (show as percentage in logs for readability)
                if pnl < 0:
                    logger.info(
                        f"Position {symbol} has loss of {pnl:.2f} USDT ({loss_ratio*100:.2f}% of equity)"
                    )

                # Check if loss exceeds threshold (max_loss_pct is stored as ratio, e.g., 0.05 for 5%)
                if pnl < 0 and loss_ratio > max_loss_pct:
                    logger.warning(
                        f"Stop loss triggered for {symbol}: Loss of {loss_ratio*100:.2f}% exceeds threshold of {max_loss_pct*100:.2f}%"
                    )

                    # Add to list of positions to close with all required fields for closure
                    position_copy = position.copy()
                    position_copy["status"] = "CLOSED"  # Mark for closure
                    position_copy["exit_reason"] = "stop_loss"
                    position_copy["loss_pct"] = (
                        loss_ratio  # Store as ratio to be consistent
                    )
                    position_copy["exit_time"] = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    # Get current prices for close prices
                    if symbol in symbol_data and len(symbol_data[symbol]) > 0:
                        latest_price = symbol_data[symbol][-1]
                        if (
                            isinstance(latest_price, dict)
                            and "close" in latest_price
                        ):
                            position_copy["closePrice"] = float(
                                latest_price["close"]
                            )
                        elif (
                            isinstance(latest_price, list) and len(latest_price) > 4
                        ):
                            position_copy["closePrice"] = float(latest_price[4])


                    # Calculate current spread for exit_spread
                    if (
                        "closePrice" in position_copy):
                        close = position_copy["closePrice"]
                        if close > 0:
                            position_copy["exit_spread"] = 0

                    # Set PNL
                    position_copy["pnl"] = pnl

                    # Initialize close_order_ids (will be populated after orders are executed)
                    position_copy["close_order_ids"] = []

                    positions_to_close.append(position_copy)

            except Exception as e:
                logger.error(
                    f"Error checking stop loss for position {position.get('id', 'unknown')}: {e}"
                )
                logger.error(traceback.format_exc())
                continue

        return positions_to_close
    except Exception as e:
        logger.error(f"Error in check_stop_loss_for_open_positions: {e}")
        logger.error(traceback.format_exc())
        return []


def check_open_position(
    open_position: Dict[str, Any], symbol_data: Dict[str, List]
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Check if an open position should be closed."""
    try:
        # Kiểm tra xem vị thế có thực sự ở trạng thái OPEN hay không
        if open_position.get("status") != "OPEN":
            logger.info(
                f"Position {open_position.get('symbol')} is not in OPEN status (current status: {open_position.get('status')}). Skipping take profit check."
            )
            return False, None

        # Get position details
        symbol = open_position.get("symbol")

        if not all([symbol]):
            logger.error(f"Missing required position data: {open_position}")
            return False, None

        # Determine which symbol is being traded and get its quantity and side
        if symbol:
            quantity = float(open_position.get("quantity", 0))
            side = open_position.get("side")
            position_side = open_position.get("positionSide")
        
        if not quantity > 0 or side == "none" or position_side == "none":
            logger.error(
                f"Invalid quantity or side for trade symbol {symbol}: quantity={quantity}, side={side}, position_side={position_side}"
            )
            return False, None

        # Get historical prices for both symbols
        if symbol not in symbol_data:
            logger.error(f"Missing price data for {symbol}")
            return False, None

        # Extract close prices
        close_prices = [float(candle["close"]) for candle in symbol_data[symbol]]
        open_prices = [float(candle["open"]) for candle in symbol_data[symbol]]
        high_prices = [float(candle["high"]) for candle in symbol_data[symbol]]
        low_prices = [float(candle["low"]) for candle in symbol_data[symbol]]
        volume_data = [float(candle["volume"]) for candle in symbol_data[symbol]]

        # Get latest price for PnL calculation
        latest_price_data = get_latest_price_from_ohlcv(symbol, symbol_data)

        # Extract close prices
        if isinstance(latest_price_data, dict) and "close" in latest_price_data:
            close_price = float(latest_price_data["close"])
            open_price = float(latest_price_data["open"])
            high_price = float(latest_price_data["high"])
            low_price = float(latest_price_data["low"])
            volume = float(latest_price_data["volume"])
        else:
            close_price = (
                float(latest_price_data) if latest_price_data else close_prices[-1]
            )
            open_price = (float(latest_price_data) if latest_price_data else open_prices[-1])
            high_price = (float(latest_price_data) if latest_price_data else high_prices[-1])
            low_price = (float(latest_price_data) if latest_price_data else low_prices[-1])
            volume = (float(latest_price_data) if latest_price_data else volume_data[-1])

        # Calculate spread using historical prices
        spreads = calculate_new_spread(close_price, open_price, high_price, low_price, volume)
        if not spreads or len(spreads) == 0:
            logger.error(f"Failed to calculate spread for {symbol}")
            return False, None

        # Get current spread (latest value)
        current_spread = spreads[-1]

        # Calculate bands
        window = open_position.get(
            "entry_window", CONFIG.get("window", DEFAULT_CONFIG["window"])
        )
        multiplier = open_position.get(
            "bb_entry_multiplier",
            CONFIG.get("bb_entry_multiplier", DEFAULT_CONFIG["bb_entry_multiplier"]),
        )

        if len(spreads) < window:
            logger.error(f"Not enough spread data for {symbol}")
            return False, None

        bands = calculate_bands(open_position, spreads, multiplier)

        # Check if we should close the position
        should_close = False
        exit_reason = "take_profit"

        # Take profit condition - Mean reversion: when spread crosses middle band
        if (position_side == "LONG" and current_spread >= bands['middle_band']) or (
                position_side == "SHORT"  and current_spread <= bands['middle_band']):
            should_close = True
            exit_reason = "take_profit"
            logger.info(
                f"Take profit triggered for {symbol} ({position_side}): spread={current_spread:.6f}, middle={bands['middle_band']:.6f}"
            )

            # Calculate PnL
            entry_price = float(
                open_position.get(
                    "entryPrice",
                    0,
                )
            )
            if position_side == "LONG":
                pnl = (close_price - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - close_price) * quantity

            logger.info(f"Position PnL for {symbol}: {pnl}")

            # Get both closing prices
            close_price = close_prices[-1]

            # Prepare position data for closing
            position_data = open_position.copy()
            position_data.update(
                {
                    "status": "CLOSED",
                    "exit_reason": exit_reason,
                    "exit_spread": current_spread,
                    "exit_middle_band": bands["middle_band"],
                    "exit_window": window,
                    "exit_multiplier": multiplier,
                    "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "close_order_ids": [],
                }
            )

            # Thiết lập giá đóng dựa vào tradeSymbol
            if symbol:
                position_data["closePrice"] = close_price
                
            # Tính và lưu PnL
            position_data["pnl"] = pnl

            return should_close, position_data

        return False, None

    except Exception as e:
        logger.error(f"Error in check_open_position: {e}")
        logger.error(traceback.format_exc())
        return False, None
