import logging
import numpy as np
import pandas as pd
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import namedtuple

# Import utility functions
from core.utils.common_utils import (
    convert_numpy_to_python,
    round_quantity,
    check_min_notional,
)
from core.config.config_manager import get_symbol_info

# Import position and market data functions
from core.position.position_manager import (
    calculate_position_pnl,
    calculate_bands,
    get_open_order_for_symbol,
)
from core.data.market_data_manager import (
    get_latest_price_from_ohlcv,
)

# Import calculators
from calculators import calculate_new_spread

# Initialize logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("bot.log"),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Define BatchOrder class for bundling order details
BatchOrder = namedtuple(
    "BatchOrder", ["symbol", "order_type", "side", "position_side", "quantity"]
)


def process_symbol(
    symbol: Dict[str, Any],
    symbol_data: Dict[str, List],
    current_equity: float,
    positions_by_symbol: Dict[str, List],
) -> Tuple[List, List, Optional[Dict]]:
    """
    Process a single trading symbol for signals and return orders if needed.

    Args:
        symbol: Dictionary containing symbol information
        symbol_data: Dictionary of market data for all symbols
        current_equity: Current account equity
        positions_by_symbol: Dictionary of positions organized by symbol

    Returns:
        Tuple containing three elements:
        - close_orders: List of orders to close existing positions
        - open_orders: List of orders to open new positions
        - position_data: Position data dictionary for database operations
          (May contain data for either opening or closing a position, depending on context)
    """
    try:
        # Import here to avoid circular imports
        from core.config.constants import CONFIG, DEFAULT_CONFIG
        from models.model_config import get_config_from_db
        from models.model_position import get_open_positions_from_db
        from calculators import calculate_bollinger_bands

        # Get the most current config values
        current_config = get_config_from_db() or CONFIG

        symbol = symbol["symbol"]

        # Check if this symbol has an open position in the database
        open_position = get_open_order_for_symbol(symbol)
        has_open_position = open_position is not None

        # Get OHLCV data from the pre-crawled data
        if symbol not in symbol_data:
            logger.warning(f"Missing data for {symbol}")
            return [], [], None

        data = symbol_data[symbol]

        # Extract close prices and timestamps
        close_prices = []
        open_prices = []
        timestamps = []
        for item in data:
            try:
                close_prices.append(float(item["close"]))
                open_prices.append(float(item["open"]))
                timestamps.append(int(item["time"]))
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing data for {symbol}: {e}")

        # Ensure data is sorted by timestamp (oldest to newest)
        if timestamps and close_prices:
            # Check if data is already in ascending order
            is_ascending = all(
                timestamps[i] <= timestamps[i + 1]
                for i in range(len(timestamps) - 1)
            )
            if not is_ascending:
                # Sort both arrays based on timestamps
                sorted_data = sorted(
                    zip(timestamps, close_prices), key=lambda x: x[0]
                )
                timestamps, close_prices = zip(*sorted_data)

        

        # Get window value from current_config
        window = current_config.get(
            "window", CONFIG.get("window", DEFAULT_CONFIG["window"])
        )
        multiplier = current_config.get(
            "bb_entry_multiplier",
            CONFIG.get("bb_entry_multiplier", DEFAULT_CONFIG["bb_entry_multiplier"]),
        )

        # Ensure we have enough data
        if len(close_prices) < window or len(close_prices) < window:
            logger.warning(f"Not enough data points for {symbol}")
            return [], [], None

        # Create pandas Series with timestamps as index
        close_series = pd.Series(close_prices, index=timestamps)
        open_series = pd.Series(open_prices, index=timestamps)

        # Find common timestamps
        common_index = close_series.index

        # Check if we have enough common timestamps
        if len(common_index) < window:
            logger.warning(
                f"Not enough common timestamps for {symbol}: {len(common_index)}/{window} required"
            )
            logger.warning(
                f"Original data points: {symbol}={len(close_series)}"
            )

        original_data_length = len(close_series)


        # Calculate the spread (logarithmic) using aligned series
        spread = calculate_new_spread(Close = close_series.values, Open = open_series.values)

        # Log data reduction statistics
        if len(spread) < original_data_length:
            logger.warning(
                f"Spread calculation reduced data points: original={original_data_length}, final={len(spread)}"
            )
            logger.warning(
                f"Data loss: {original_data_length - len(spread)} points ({((original_data_length - len(spread)) / original_data_length * 100):.1f}%)"
            )

        # Get current spread value
        current_spread = spread[-1] if len(spread) > 0 else None
        if current_spread is None:
            logger.warning(f"Could not calculate spread for {symbol}")
            return [], [], None

        # Calculate bands based on open position or for new entries
        if has_open_position:
            # Add entry_window to open_position if not present
            if "entry_window" not in open_position:
                open_position["entry_window"] = window
            current_bands = calculate_bands(open_position, spread, multiplier)
        else:
            current_bands = calculate_bands(None, spread, multiplier)

        # Check if bands calculation failed
        if current_bands["middle_band"] is None:
            logger.warning(
                f"Bollinger Bands calculation failed for {symbol}"
            )
            return [], [], None

        # Initialize variables for orders
        close_orders = []
        open_orders = []
        position_data = None

        # Check for exit conditions if there's an open position
        if has_open_position:
            # Get current price from OHLCV data for tradeSymbol
            current_price_data = get_latest_price_from_ohlcv(symbol, symbol_data)
            if not current_price_data:
                logger.error(
                    f"Failed to get current price for {symbol} from OHLCV data"
                )
                return [], [], None

            # Đảm bảo current_price_data là dict và có key "close"
            if isinstance(current_price_data, dict) and "close" in current_price_data:
                current_price = current_price_data["close"]
            else:
                current_price = float(current_price_data) if current_price_data else 0

            middle_band = current_bands["middle_band"]

            # Xác định đúng side và position_side dựa vào tradeSymbol
            if symbol:
                open_side = open_position.get("side")
                open_position_side = open_position.get("positionSide")
                quantity = float(open_position.get("quantity", 0))

            # Take profit condition (spread crosses middle band)
            if (open_position_side == "LONG"  and current_spread >= middle_band) or (
                open_position_side == "SHORT" and current_spread <= middle_band):
                logger.info(
                    f"Take profit triggered for {symbol}: spread={current_spread:.6f}, middle={middle_band:.6f}"
                )
                close_side = "SELL" if open_position_side == "LONG" else "BUY"

                close_orders.append(
                    BatchOrder(
                        symbol=f"{symbol}-USDT",
                        order_type="MARKET",
                        side=close_side,
                        position_side=open_position_side,
                        quantity=quantity,
                    )
                )
                # Record position data for database update (only 1 dict for the symbol)
                position_data = {
                    "id": open_position.get(
                        "id",
                        f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    ),
                    "symbol": symbol,
                    "orderType": "MARKET",
                    "side": open_position.get("side", "none"),
                    "positionSide": open_position.get("positionSide", "none"),
                    "entryPrice": open_position.get("entryPrice", 0),
                    "quantity": open_position.get("quantity", 0),
                    "status": "CLOSED",
                    "exit_reason": "take_profit",
                    "exit_spread": current_spread,
                    "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "exitPrice": current_price,
                    # Thêm các trường bắt buộc
                    "closePrice": 0,  # Sẽ được cập nhật từ exchange
                    "pnl": 0,  # Sẽ được tính toán sau khi đóng vị thế
                }

                # Tính toán PnL dựa trên giá hiện tại
                entry_price = float(
                    open_position.get(
                        "entryPrice", 0
                    )
                )
                quantity = float(
                    open_position.get(
                        "quantity", 0
                    )
                )

                if open_position_side == "LONG":
                    pnl = (current_price - entry_price) * quantity
                else:  # SHORT
                    pnl = (entry_price - current_price) * quantity

                position_data["closePrice"] = current_price


                position_data["pnl"] = pnl

                return close_orders, [], position_data

        # Generate entry signal if we don't have an open position for this symbol
        if not has_open_position:
            # Count open positions for the symbols in this symbol to avoid having too many positions per symbol
            open_positions_db = get_open_positions_from_db()
            symbol_position_counts = {}

            # Count open positions per symbol
            for pos in open_positions_db:
                symbol_ = pos["symbol"]

                symbol_position_counts[symbol_] = (
                    symbol_position_counts.get(symbol_, 0) + 1
                )

            # Check if either symbol in this symbol has reached the maximum number of open positions
            symbol_count = symbol_position_counts.get(symbol, 0)

            max_concurrent_positions = current_config.get(
                "max_concurrent_positions",
                CONFIG.get(
                    "max_concurrent_positions",
                    DEFAULT_CONFIG["max_concurrent_positions"],
                ),
            )

            can_open_position = (
                symbol_count < max_concurrent_positions
            )

            if can_open_position:
                entry_signal = 0

                # Get previous spread value if available (to detect crossovers)
                prev_spread = spread[-2] if len(spread) > 1 else None

                # Detect when spread crosses down through upper band
                if (
                    prev_spread is not None
                    and prev_spread > current_bands["upper_band"]
                    and current_spread <= current_bands["upper_band"]
                ):
                    entry_signal = -1  #Short 
                # Detect when spread crosses up through lower band
                elif (
                    prev_spread is not None
                    and prev_spread < current_bands["lower_band"]
                    and current_spread >= current_bands["lower_band"]
                ):
                    entry_signal = 1  # Long 

                # Open new position if we have an entry signal
                if entry_signal != 0:
                    logger.info(
                        f"Opening new position for {symbol}, signal: {entry_signal}"
                    )

                    # Get position size percentage from config
                    position_size_pct = current_config.get(
                        "position_size_pct",
                        CONFIG.get(
                            "position_size_pct", DEFAULT_CONFIG["position_size_pct"]
                        ),
                    )

                    # Get risk parity weight for this symbol
                    risk_parity_weight = symbol_data.get("risk_parity_weight", 1.0)
                    logger.info(
                        f"Using risk parity weight for {symbol}: {risk_parity_weight:.4f}"
                    )

                    # Calculate position size based on equity and risk parity weight
                    notional_per_leg = (
                        current_equity * position_size_pct * risk_parity_weight
                    )
                    logger.info(
                        f"Notional per leg for {symbol}: {notional_per_leg:.2f} USDT (equity: {current_equity:.2f}, position_size_pct: {position_size_pct:.4f}, weight: {risk_parity_weight:.4f}"
                    )

                    # Get current prices for both symbols from OHLCV data instead of API
                    try:
                        current_price_data = get_latest_price_from_ohlcv(
                            symbol, symbol_data
                        )

                        if not current_price_data:
                            logger.error(
                                f"Failed to get current prices for {symbol} from OHLCV data"
                            )
                            return [], [], None

                        current_price = float(current_price_data["close"])

                        # Initial quantity calculation
                        quantity = notional_per_leg / current_price

                        # Round quantities
                        quantity = round_quantity(quantity, symbol)

                        # Kiểm tra xem việc làm tròn có tăng notional do min_quantity không
                        actual_notional = quantity * current_price

                        # Luôn lấy notional cao hơn làm chuẩn và tính lại quantity cho position còn lại
                        logger.info(
                            f"Initial notional after rounding: {actual_notional:.4f}"
                        )


                        # Check and adjust for minimum notional
                        notional_check, adjusted_quantity = check_min_notional(
                            symbol, quantity, current_price
                        )


                        # If either fails the check, use the adjusted quantities
                        if not notional_check:
                            logger.warning(
                                f"Adjusting quantities to meet minimum requirements for {symbol}"
                            )

                            # Calculate notional values for both original and adjusted
                            notional = adjusted_quantity * current_price

                            quantity = adjusted_quantity

                            # Calculate notional values again after rounding to make sure they still meet requirements
                            final_notional = quantity * current_price

                            # Get minimum notional requirements
                            symbol_info = get_symbol_info(symbol)
                            min_notional = symbol_info.get("tradeMinUSDT", 2.0)

                            # Check if both meet requirements after rounding
                            max_iterations = 3  # Prevent infinite loop
                            iteration = 0

                            while (
                                final_notional < min_notional
                            ) and iteration < max_iterations:
                                logger.warning(
                                    f"After rounding, notional values still don't meet requirements: {final_notional}"
                                )

                                # Determine which one needs adjustment and use that as baseline
                                if final_notional < min_notional:
                                    # Recalculate quantities using min_notional_A as baseline
                                    quantity_= min_notional / current_price
                                    quantity = round_quantity(quantity, symbol)
                                    target_notional = quantity * current_price
    

                                # Recalculate notional values
                                final_notional = quantity * current_price

                                logger.info(
                                    f"Adjusted quantities: {quantity}"
                                )
                                logger.info(
                                    f"New notional values: {final_notional}"
                                )

                                iteration += 1

                           
                        # Check one more time if quantities are valid
                        if quantity <= 0:
                            logger.warning(
                                f"Adjusted quantities still invalid for {symbol}"
                            )
                            return [], [], None

                        # After calculating entry_signal, quantities, etc.
                        if entry_signal in [1, -1]:
                            # Xác định side/positionSide cho từng symbol
                            if entry_signal == 1:
                                side = "BUY"
                                positionSide = "LONG"
                            else:
                                side = "SELL"
                                positionSide = "SHORT"
                            entryPrice = current_price
                            quantity_db = quantity
                            
                            if quantity_db > 0:
                                open_orders.append(
                                    BatchOrder(
                                        symbol=f"{symbol}-USDT",
                                        order_type="MARKET",
                                        side=side,
                                        position_side=positionSide,
                                        quantity=quantity_db,
                                    )
                                )
                            
                            # Chuẩn bị dữ liệu position cho database
                            if not close_orders:
                                position_data = {
                                    "symbol": symbol,
                                    "orderType": "MARKET",
                                    "side": side,
                                    "positionSide": positionSide,
                                    "entryPrice": entryPrice,
                                    "quantity": quantity_db,
                                    "status": "OPEN",
                                    "entry_spread": float(current_spread),
                                    "entry_time": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    ),
                                }
                                position_data = convert_numpy_to_python(position_data)
                    except Exception as e:
                        logger.error(
                            f"Error getting prices or creating orders for {symbol}: {e}"
                        )
                        logger.error(traceback.format_exc())
                        return [], [], None
            else:
                logger.info(
                    f"Cannot open position for {symbol}: One or both symbols already have the maximum number of positions"
                )

        return close_orders, open_orders, position_data
    except Exception as e:
        logger.error(
            f"Error processing symbol {symbol_info.get('symbol', 'unknown')}: {e}"
        )
        logger.error(traceback.format_exc())
        return [], [], None
