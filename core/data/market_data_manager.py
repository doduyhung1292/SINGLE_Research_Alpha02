import logging
import traceback
import time
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.utils.common_utils import ensure_data_sort_oldest_to_newest
from core.config.constants import CONFIG, DEFAULT_CONFIG
from core.config.config_manager import get_unique_symbols
from core.api.exchange_api import get_current_mark_price

logger = logging.getLogger(__name__)


def get_ohclv(symbol, timeframe="4h", limit=900):
    """
    Fetch OHLCV (Open, High, Close, Low, Volume) data for a specific symbol.

    Args:
        symbol (str): The trading symbol (e.g., 'BTC')
        timeframe (str): The candle timeframe (e.g., '1m', '5m', '1h', '1d')
        limit (int): Number of candles to fetch

    Returns:
        list: List of OHLCV data points (dict format): [
            {
                "open": "0.7034",
                "close": "0.7065",
                "high": "0.7081",
                "low": "0.7033",
                "volume": "635494.00",
                "time": 1702717200000
            },
            ...
        ]

    Notes:
        Bingx API returns data sorted from newest to oldest (descending by time).
        The calling function should sort the data if ascending order is needed.
    """
    try:
        from core.api.exchange_api import (
            send_request,
            parseParam,
            get_sign,
            APIURL,
            SECRETKEY,
        )

        path = "/openApi/swap/v3/quote/klines"
        method = "GET"

        params = {
            "symbol": f"{symbol}-USDT",
            "interval": timeframe,
            "limit": limit,
            "timestamp": int(time.time() * 1000),
        }

        # Generate signature
        paramsStr = parseParam(params)

        # Send request
        response = send_request(method, path, paramsStr)

        # Process response
        if response and response.get("code") == 0 and "data" in response:
            # Return raw data directly, formatting will be handled in the calling function
            # Lưu ý: Dữ liệu từ Bingx thường được sắp xếp từ mới đến cũ (giảm dần theo thời gian)
            return response["data"]
        else:
            logger.error(f"API error: {response}")
            return []

    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return []


def fetch_historical_klines(
    symbol: str,
    timeframe: str,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Fetch historical klines data from REST API for a symbol and timeframe.
    This is useful for initializing the WebSocket data store or filling gaps.

    IMPORTANT: BingX REST API returns data from newest to oldest (newest first).
    This function will always sort the data to oldest-to-newest (ascending timestamp)
    for consistent calculations, which is what calculation functions expect.

    Args:
        symbol (str): The symbol to fetch data for (e.g., 'BTC')
        timeframe (str): The timeframe (e.g., '5m', '1h')
        limit (int): Number of candles to fetch.

    Returns:
        List[Dict[str, Any]]: List of formatted candle data sorted from oldest to newest
    """
    try:
        # Get data from REST API - now without startTime and endTime
        rest_data = get_ohclv(symbol, timeframe, limit)

        if not rest_data:
            logger.error(
                f"Failed to get historical data for {symbol} {timeframe} (latest {limit})"
            )
            return []

        # Format data to match WebSocket format
        formatted_data = []

        # Kiểm tra định dạng dữ liệu trả về
        if rest_data and isinstance(rest_data, list):
            if len(rest_data) == 0:
                logger.error(f"No data returned for {symbol}")
                return []

            # Kiểm tra cấu trúc của dữ liệu
            sample_item = rest_data[0]

            # Debug log to check if API returns newest-first
            if len(rest_data) >= 2:
                if isinstance(sample_item, dict) and "time" in sample_item:
                    first_time = rest_data[0]["time"]
                    last_time = rest_data[-1]["time"]
                    is_newest_first = first_time > last_time

                elif isinstance(sample_item, list) and len(sample_item) >= 6:
                    first_time = rest_data[0][0]
                    last_time = rest_data[-1][0]
                    is_newest_first = first_time > last_time


            # Định dạng API chuẩn: {'open': '0.7034', 'close': '0.7065', 'high': '0.7081', 'low': '0.7033', 'volume': '635494.00', 'time': 1702717200000}
            if (
                isinstance(sample_item, dict)
                and "open" in sample_item
                and "time" in sample_item
            ):

                for candle in rest_data:
                    formatted_candle = {
                        "time": int(candle["time"]),
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle["volume"]),
                    }
                    formatted_data.append(formatted_candle)

                # Always aggressively sort data from oldest to newest (thời gian tăng dần)
                formatted_data.sort(key=lambda x: x["time"])

            # Định dạng cũ dạng mảng: [timestamp, open, high, low, close, volume]
            elif isinstance(sample_item, list) and len(sample_item) >= 6:
                logger.info(f"Processing legacy array format for {symbol}")
                for candle in rest_data:
                    formatted_candle = {
                        "time": int(candle[0]),  # timestamp
                        "open": float(candle[1]),  # open
                        "high": float(candle[2]),  # high
                        "low": float(candle[3]),  # low
                        "close": float(candle[4]),  # close
                        "volume": float(candle[5]),  # volume
                    }
                    formatted_data.append(formatted_candle)

                # Always aggressively sort data from oldest to newest (thời gian tăng dần)
                formatted_data.sort(key=lambda x: x["time"])

            else:
                logger.error(f"Unknown data format for {symbol}: {sample_item}")
                return []
        else:
            logger.error(
                f"Unexpected data format from API for {symbol}: {type(rest_data)}"
            )
            return []

        # Verify data is properly sorted after processing
        if len(formatted_data) >= 2:
            is_sorted_ascending = all(
                formatted_data[i]["time"] <= formatted_data[i + 1]["time"]
                for i in range(len(formatted_data) - 1)
            )
            if not is_sorted_ascending:
                logger.error(
                    f"DATA ORDERING ERROR: Data for {symbol} is NOT sorted in ascending order after processing. Forcing re-sort."
                )
                formatted_data.sort(key=lambda x: x["time"])

            # Log first and last timestamps for verification
            first_ts = datetime.fromtimestamp(
                formatted_data[0]["time"] / 1000
            ).strftime("%Y-%m-%d %H:%M:%S")
            last_ts = datetime.fromtimestamp(
                formatted_data[-1]["time"] / 1000
            ).strftime("%Y-%m-%d %H:%M:%S")

        return formatted_data

    except Exception as e:
        logger.error(f"Error fetching historical data for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return []


def fetch_historical_data_parallel(
    symbols: List[str], timeframe: str, limit: int = 1000
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch historical data for multiple symbols in parallel to speed up initialization.

    Args:
        symbols: List of symbols to fetch data for
        timeframe: Timeframe to fetch data for
        limit: Number of candles to fetch per symbol

    Returns:
        Dictionary mapping symbols to their historical data
    """
    logger.info(f"Fetching historical data for {len(symbols)} symbols in parallel...")
    results = {}

    # Hàm xử lý cho mỗi symbol
    def process_symbol(symbol_arg):
        data_fetched = fetch_historical_klines(symbol_arg, timeframe, limit)
        return symbol_arg, data_fetched

    # Chia các symbol thành các batch nhỏ để tránh quá tải API
    batch_size = 5  # Tối đa 5 symbol mỗi batch
    symbol_batches = [
        symbols[i : i + batch_size] for i in range(0, len(symbols), batch_size)
    ]

    for i, batch in enumerate(symbol_batches):
        logger.info(
            f"Processing batch {i+1}/{len(symbol_batches)} with {len(batch)} symbols"
        )

        # Xử lý mỗi batch với các luồng song song
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            # Tạo và gửi các nhiệm vụ
            future_to_symbol = {
                executor.submit(process_symbol, sym_item): sym_item
                for sym_item in batch
            }

            # Xử lý kết quả khi hoàn thành
            for future in as_completed(future_to_symbol):
                try:
                    symbol_res, data_res = future.result()
                    if data_res:
                        results[symbol_res] = data_res
                    else:
                        logger.warning(f"Failed to fetch data for {symbol_res}")
                except Exception as e_inner:
                    symbol_exc = future_to_symbol[future]
                    logger.error(f"Error processing {symbol_exc}: {e_inner}")
                    logger.error(traceback.format_exc())

        # Đợi giữa các batch để tránh rate limit
        if i < len(symbol_batches) - 1:
            logger.info(f"Waiting 1.2 seconds before processing next batch...")
            time.sleep(1.2)

    logger.info(
        f"Completed fetching historical data for {len(results)}/{len(symbols)} symbols"
    )
    return results


def get_latest_price_from_ohlcv(
    symbol: str,
    symbol_data: Dict[str, List[Dict[str, Any]]],
) -> Union[Dict[str, Any], float, int]:
    """
    Get the latest price for a symbol from OHLCV data.

    Args:
        symbol: Symbol to get price for
        symbol_data: Dictionary of OHLCV data for all symbols

    Returns:
        Dict or float: Latest price data (as dict) or 0 if not found
    """
    try:
        # Check if symbol exists in the data and has data
        if not isinstance(symbol, str):
            logger.error(f"Invalid symbol type: {type(symbol)}, expected string")
            return 0

        if symbol not in symbol_data or not symbol_data[symbol]:
            logger.warning(f"No data found for symbol: {symbol}")
            return 0

        # Get the last candle
        last_candle = symbol_data[symbol][-1]

        # Return full candle data
        if isinstance(last_candle, dict) and "close" in last_candle:
            return last_candle
        elif isinstance(last_candle, list) and len(last_candle) > 4:
            # Convert array format to dictionary
            return {
                "open": float(last_candle[1]),
                "high": float(last_candle[2]),
                "low": float(last_candle[3]),
                "close": float(last_candle[4]),
                "volume": float(last_candle[5]) if len(last_candle) > 5 else 0,
                "time": last_candle[0],
            }

        logger.warning(f"Unexpected candle format for {symbol}: {last_candle}")
        return 0
    except Exception as e:
        logger.error(f"Error getting latest price for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return 0


def prepare_market_data(
    all_symbols: List[Dict[str, Any]], open_positions: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Prepare market data for all symbols needed for trading.

    Args:
        all_pairs: List of all trading pairs
        open_positions: List of open positions

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary of market data for all symbols
    """
    try:
        # Get unique symbols from pairs and positions
        unique_symbols = get_unique_symbols(all_symbols)

        # Add symbols from open positions
        for position in open_positions:
            if "symbol" in position and position["symbol"] not in unique_symbols:
                unique_symbols.append(position["symbol"])

        logger.info(f"Preparing market data for {len(unique_symbols)} symbols")

        # Fetch market data
        timeframe = CONFIG.get("timeframe", DEFAULT_CONFIG["timeframe"])
        limit = CONFIG.get("limit", DEFAULT_CONFIG["limit"])
        symbol_data = fetch_historical_data_parallel(unique_symbols, timeframe, limit)

        # Check for missing data
        missing_data_symbols = [
            s for s in unique_symbols if s not in symbol_data or not symbol_data[s]
        ]
        if missing_data_symbols:
            logger.warning(
                f"Data is missing for symbols: {', '.join(missing_data_symbols)}"
            )

        # Ensure all data is properly sorted
        for symbol, data in symbol_data.items():
            symbol_data[symbol] = ensure_data_sort_oldest_to_newest(data)

        return symbol_data

    except Exception as e:
        logger.error(f"Error preparing market data: {e}")
        logger.error(traceback.format_exc())
        return {}
