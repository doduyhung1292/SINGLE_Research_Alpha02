import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
from typing import Tuple, List, Dict, Any
import os
import logging
import requests
import json
import sys
import mplfinance as mpf
import ccxt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha.alpha import calculate_spread, list_available_alphas, ALPHA_DESCRIPTIONS

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("backtest.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PairTradingBacktest:
    def __init__(
        self,
        pairs: list,  # List of pairs [{symbolA, symbolB}]
        start_date: str,
        end_date: str,
        window: int = 500,
        timeframe: str = "5m",
        initial_equity: float = 100.0,  # Starting equity in USDT
        position_size_pct: float = 0.01,  # Position size as percentage of equity per leg
        bb_entry_multiplier: float = 2.0,  # Multiplier for entry Bollinger Bands
        maker_fee: float = 0.00025,  # 0.025%
        taker_fee: float = 0.00025,  # 0.025%
        slippage: float = 0.00001,  # 0.001%
        api_key: str = None,
        api_secret: str = None,
        alpha_name: str = "alpha1",  # Default alpha: LOG(CloseA) - LOG(CloseB)
        alpha_params: Dict[
            str, Any
        ] = None,  # Additional parameters for alpha calculation
    ):
        """
        Initialize the pair trading backtest.

        Args:
            pairs: List of pairs in format [{symbolA, symbolB}]
            start_date: Start date for the backtest in format 'YYYY-MM-DD'
            end_date: End date for the backtest in format 'YYYY-MM-DD'
            window: Window size for Bollinger Bands calculation
            timeframe: Timeframe for the data (e.g., '5m', '1h', '1d')
            initial_equity: Initial equity for the backtest
            position_size_pct: Position size as percentage of equity per leg
            bb_entry_multiplier: Multiplier for entry Bollinger Bands
            maker_fee: Maker fee as a decimal (e.g., 0.00025 for 0.025%)
            taker_fee: Taker fee as a decimal
            slippage: Slippage as a decimal
            api_key: API key for exchange
            api_secret: API secret for exchange
            alpha_name: Name of the alpha to use ('alpha1' through 'alpha7')
            alpha_params: Additional parameters for alpha calculation
        """
        self.pairs = pairs
        self.start_date = start_date
        self.end_date = end_date
        self.window = window
        self.timeframe = timeframe
        self.initial_equity = initial_equity
        self.position_size_pct = position_size_pct
        self.bb_entry_multiplier = bb_entry_multiplier
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.slippage = slippage
        self.alpha_name = alpha_name
        self.alpha_params = alpha_params or {}

        # Validate alpha_name
        if alpha_name not in ALPHA_DESCRIPTIONS:
            valid_alphas = list(ALPHA_DESCRIPTIONS.keys())
            raise ValueError(
                f"Invalid alpha_name: {alpha_name}. Available alphas: {valid_alphas}"
            )

        logger.info(f"Using {alpha_name}: {ALPHA_DESCRIPTIONS[alpha_name]}")

        # Initialize exchange
        self._initialize_exchange(api_key, api_secret)

        # Placeholder for data
        self.data_a = None
        self.data_b = None
        self.merged_data = None

        # Results storage
        self.results = {}

        # Convert dates to milliseconds for API calls
        self.start_timestamp = int(
            datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000
        )
        self.end_timestamp = int(
            datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000
        )

        # Increment for fetching data
        self.time_increment = self._calculate_time_increment(timeframe)

        logger.info("Pair Trading Backtest initialized")
        logger.info(
            f"Timeframe: {timeframe}, Window: {window}, BB Multiplier: {bb_entry_multiplier}"
        )
        logger.info(
            f"Initial Equity: {initial_equity} USDT, Position Size: {position_size_pct*100}%"
        )
        logger.info(
            f"Fees - Maker: {maker_fee*100}%, Taker: {taker_fee*100}%, Slippage: {slippage*100}%"
        )

        # Initialize data containers - will be populated for each pair
        self.current_pair = None
        self.symbol_a = None
        self.symbol_b = None
        self.signals = None
        self.trades = []
        self.equity_curve = None
        self.performance_metrics = {}
        self.all_results = {}  # Store results for all pairs

        # Trading variables
        self.current_position = (
            0  # 0: no position, 1: long A short B, -1: long B short A
        )
        self.entry_price_a = 0
        self.entry_price_b = 0
        self.position_size_a = 0
        self.position_size_b = 0

    def _initialize_exchange(self, api_key=None, api_secret=None):
        """Initialize connection to BingX API."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://open-api.bingx.com"
        self.rate_limit = 0.2  # Default rate limit in seconds
        return self

    def _calculate_time_increment(self, timeframe):
        """
        Calculate time increment in milliseconds based on the timeframe.
        This is used to advance the timestamp when no data is found.

        Args:
            timeframe: The timeframe string (e.g., '1m', '4h', '1d')

        Returns:
            Time increment in milliseconds
        """
        # Default increment (1 day in milliseconds)
        default_increment = 24 * 60 * 60 * 1000

        # Extract the number and unit from timeframe
        if len(timeframe) < 2:
            return default_increment

        try:
            # Parse the timeframe (e.g., '1m', '4h', '1d')
            number = int(timeframe[:-1])
            unit = timeframe[-1]

            # Calculate increment based on unit
            if unit == "m":  # minutes
                # For minute timeframes, advance by 1 day
                return 24 * 60 * 60 * 1000
            elif unit == "h":  # hours
                # For hour timeframes, advance by 7 days
                return 7 * 24 * 60 * 60 * 1000
            elif unit == "d":  # days
                # For day timeframes, advance by 30 days
                return 30 * 24 * 60 * 60 * 1000
            elif unit == "w":  # weeks
                # For week timeframes, advance by 90 days
                return 90 * 24 * 60 * 60 * 1000
            elif unit == "M":  # months
                # For month timeframes, advance by 180 days
                return 180 * 24 * 60 * 60 * 1000
            else:
                return default_increment

        except (ValueError, IndexError):
            return default_increment

    def _fetch_ohlcv_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch OHLCV data for the given symbol from BingX API, accounting for the window period.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')

        Returns:
            DataFrame with OHLCV data

        Raises:
            ValueError: If the symbol doesn't exist in the exchange or API error
        """
        # First, check if the symbol exists by making a simple API call
        try:
            logger.info(f"Checking if symbol {symbol} exists...")
            check_url = f"{self.base_url}/openApi/swap/v3/quote/klines"
            check_params = {"symbol": symbol, "interval": "5m", "limit": "1"}

            headers = {}
            if self.api_key:
                headers["X-BX-APIKEY"] = self.api_key

            response = requests.get(check_url, params=check_params, headers=headers)

            if response.status_code != 200:
                logger.error(
                    f"API error when checking symbol: {response.status_code} - {response.text}"
                )
                raise ValueError(
                    f"Symbol {symbol} check failed with API error: {response.status_code}"
                )

            result = response.json()

            if result.get("code") != 0 or not result.get("data"):
                logger.error(
                    f"Symbol {symbol} does not exist or is not available: {result.get('msg')}"
                )
                raise ValueError(
                    f"Symbol {symbol} does not exist or is not available: {result.get('msg')}"
                )

            logger.info(f"Symbol {symbol} exists and is available for trading")

        except Exception as e:
            logger.error(f"Error checking symbol {symbol}: {e}")
            raise ValueError(f"Failed to verify if symbol {symbol} exists: {e}")

        # Start fetching from the end date and work backward in time
        # This approach better matches BingX's data ordering (newest to oldest)
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        start_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_timestamp = int(end_date.timestamp() * 1000)
        target_start_timestamp = int(
            (start_date - timedelta(days=self.window * 5 // 288)).timestamp() * 1000
        )

        logger.info(
            f"Searching for {symbol} data from {end_date} backward to {start_date}"
        )
        logger.info(
            f"Note: For newly listed tokens, data may only be available from their listing date"
        )

        all_ohlcv = []
        current_end_timestamp = end_timestamp
        attempts = 0
        max_attempts = 150  # Limit the number of attempts to prevent infinite loops

        # For detecting when we're stuck on a single timestamp
        last_timestamp = None
        same_timestamp_count = 0

        # Track all processed timestamps to avoid loops
        processed_end_timestamps = set()

        # Keep track of the earliest timestamp we've seen
        earliest_timestamp_seen = None

        # Convert timeframe to BingX format if needed
        interval_map = {
            "1m": "1m",
            "3m": "3m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "2h",
            "4h": "4h",
            "6h": "6h",
            "12h": "12h",
            "1d": "1d",
            "1w": "1w",
            "1M": "1M",
        }

        bingx_interval = interval_map.get(self.timeframe, self.timeframe)

        # Fetch data in chunks to handle API limitations
        empty_responses = 0  # Track consecutive empty responses

        while (
            current_end_timestamp > target_start_timestamp and attempts < max_attempts
        ):
            try:
                # Check if we've already processed this timestamp to avoid loops
                if current_end_timestamp in processed_end_timestamps:
                    logger.warning(
                        f"Already processed timestamp {datetime.fromtimestamp(current_end_timestamp/1000)} for {symbol}. Breaking loop to avoid infinite processing."
                    )
                    break

                # Record that we're processing this timestamp
                processed_end_timestamps.add(current_end_timestamp)

                # Exit condition 1: If we've collected sufficient data but can't reach start date
                if len(all_ohlcv) >= 100 and empty_responses >= 3:
                    logger.info(
                        f"Collected {len(all_ohlcv)} data points but can't reach start date for {symbol}. Token may be newly listed. Stopping data collection."
                    )
                    break

                # Exit condition 2: If we're stuck on the same timestamp for multiple attempts
                if same_timestamp_count >= 3:
                    logger.info(
                        f"Detected we're stuck on timestamp {datetime.fromtimestamp(last_timestamp/1000)} for {symbol}. This is likely the earliest data available. Stopping data collection."
                    )
                    break

                logger.info(
                    f"Fetching {symbol} data ending at {datetime.fromtimestamp(current_end_timestamp/1000)}"
                )

                # Prepare request parameters - use endTime instead of startTime
                params = {
                    "symbol": symbol,
                    "interval": bingx_interval,
                    "limit": "1000",  # Maximum allowed
                    "endTime": str(current_end_timestamp),
                }

                # Make API request with retry mechanism
                url = f"{self.base_url}/openApi/swap/v3/quote/klines"

                # Add authentication headers if API key is provided
                headers = {}
                if self.api_key:
                    headers["X-BX-APIKEY"] = self.api_key

                # Implement retry mechanism
                max_retries = 3
                retry_count = 0
                retry_delay = 1  # Start with 1 second delay

                while retry_count < max_retries:
                    try:
                        response = requests.get(
                            url, params=params, headers=headers, timeout=10
                        )

                        if response.status_code == 200:
                            break  # Success, exit retry loop

                        # Handle rate limiting (429) with exponential backoff
                        if response.status_code == 429:
                            retry_count += 1
                            wait_time = retry_delay * (
                                2**retry_count
                            )  # Exponential backoff
                            logger.warning(
                                f"Rate limit hit for {symbol}. Retrying in {wait_time} seconds..."
                            )
                            time.sleep(wait_time)
                            continue

                        # For other errors, raise exception
                        logger.error(
                            f"API error for {symbol}: {response.status_code} - {response.text}"
                        )
                        raise ValueError(
                            f"API error for {symbol}: {response.status_code} - {response.text}"
                        )

                    except requests.exceptions.Timeout:
                        retry_count += 1
                        wait_time = retry_delay * (2**retry_count)
                        logger.warning(
                            f"Timeout when fetching {symbol}. Retrying in {wait_time} seconds..."
                        )
                        time.sleep(wait_time)
                    except requests.exceptions.RequestException as e:
                        retry_count += 1
                        wait_time = retry_delay * (2**retry_count)
                        logger.warning(
                            f"Request error for {symbol}: {e}. Retrying in {wait_time} seconds..."
                        )
                        time.sleep(wait_time)

                # If we've exhausted all retries
                if retry_count >= max_retries:
                    logger.error(
                        f"Failed to fetch data for {symbol} after {max_retries} retries"
                    )
                    raise ValueError(
                        f"Failed to fetch data for {symbol} after {max_retries} retries"
                    )

                # Check response status
                if response.status_code != 200:
                    logger.error(
                        f"API error for {symbol}: {response.status_code} - {response.text}"
                    )
                    raise ValueError(
                        f"API error for {symbol}: {response.status_code} - {response.text}"
                    )

                # Parse response
                try:
                    result = response.json()

                    # More detailed logging for debugging
                    logger.debug(f"API response for {symbol}: {result}")

                    if result.get("code") != 0:
                        logger.error(f"API error for {symbol}: {result.get('msg')}")
                        raise ValueError(f"API error for {symbol}: {result.get('msg')}")

                    data = result.get("data", [])

                    logger.info(f"Received {len(data)} data points for {symbol}")

                    if len(data) == 0:
                        empty_responses += 1
                        attempts += 1
                        logger.warning(
                            f"No data returned for {symbol} ending at {datetime.fromtimestamp(current_end_timestamp/1000)}. (Empty response #{empty_responses}, attempt #{attempts})"
                        )

                        # If we've had too many consecutive empty responses, we might have reached the earliest available data
                        time_increment = self._calculate_time_increment(bingx_interval)

                        # If we've had many consecutive empty responses, increase the increment to skip backward faster
                        if empty_responses > 5:
                            time_increment *= (
                                5  # Skip backward 5x faster after 5 empty responses
                            )
                            logger.info(
                                f"Multiple empty responses, skipping backward faster"
                            )

                        # Move the end timestamp back further in time
                        current_end_timestamp -= time_increment

                        # Ensure we don't go beyond target start timestamp
                        if current_end_timestamp < target_start_timestamp:
                            current_end_timestamp = target_start_timestamp

                        logger.info(
                            f"Moving back to {datetime.fromtimestamp(current_end_timestamp/1000)} to continue searching for data"
                        )
                        continue
                    else:
                        # Reset empty responses counter when we get data
                        empty_responses = 0

                    # BingX already returns data from newest to oldest, which is what we want

                except Exception as e:
                    logger.error(f"Error parsing response for {symbol}: {e}")
                    logger.error(
                        f"Response content: {response.text[:500]}..."
                    )  # Log first 500 chars
                    raise ValueError(f"Failed to parse API response for {symbol}: {e}")

                # Convert BingX data format to our format
                chunk_ohlcv = []
                for item in data:
                    timestamp = int(item["time"])
                    ohlcv_row = [
                        timestamp,
                        float(item["open"]),
                        float(item["high"]),
                        float(item["low"]),
                        float(item["close"]),
                        float(item["volume"]),
                    ]
                    chunk_ohlcv.append(ohlcv_row)

                # Ensure we don't add duplicate data points
                if chunk_ohlcv:
                    # Sort by timestamp
                    chunk_ohlcv.sort(key=lambda x: x[0])

                    # Track the earliest timestamp we've seen for this symbol
                    if (
                        earliest_timestamp_seen is None
                        or chunk_ohlcv[0][0] < earliest_timestamp_seen
                    ):
                        earliest_timestamp_seen = chunk_ohlcv[0][0]

                    # Add to our collection
                    all_ohlcv.extend(chunk_ohlcv)

                # Update current_end_timestamp for next request
                if chunk_ohlcv:
                    attempts += 1  # Count this as an attempt even when successful

                    # Get the oldest timestamp in this chunk
                    oldest_timestamp = chunk_ohlcv[0][0]

                    # Check if we're getting the same timestamp repeatedly (stuck in a loop)
                    if (
                        len(chunk_ohlcv) == 1
                        and last_timestamp is not None
                        and oldest_timestamp == last_timestamp
                    ):
                        same_timestamp_count += 1
                        logger.warning(
                            f"Received the same timestamp {datetime.fromtimestamp(oldest_timestamp/1000)} again. "
                            f"Count: {same_timestamp_count}. This may be the earliest data available."
                        )

                        # If we're stuck, we've likely reached the earliest data
                        # Instead of trying to back up more, just break the loop
                        if same_timestamp_count >= 2:
                            logger.info(
                                f"Detected earliest data at {datetime.fromtimestamp(oldest_timestamp/1000)} for {symbol}. Stopping data collection."
                            )
                            break

                        # Force a different timestamp that's definitely earlier and smaller
                        # Calculate a fixed increment that's guaranteed to be much earlier
                        # For 5m data, 1000 bars = about 3.5 days worth of data
                        safe_increment = (
                            3 * 24 * 60 * 60 * 1000
                        )  # 3 days in milliseconds
                        new_end_timestamp = oldest_timestamp - safe_increment

                        # Ensure the new timestamp is earlier than anything we've seen before
                        if (
                            earliest_timestamp_seen is not None
                            and new_end_timestamp >= earliest_timestamp_seen
                        ):
                            new_end_timestamp = earliest_timestamp_seen - safe_increment

                        # Ensure we don't go beyond target start timestamp
                        if new_end_timestamp < target_start_timestamp:
                            new_end_timestamp = target_start_timestamp

                        current_end_timestamp = new_end_timestamp
                        logger.info(
                            f"Jumping back to {datetime.fromtimestamp(current_end_timestamp/1000)} to break out of loop"
                        )
                    else:
                        # Not stuck, normal processing
                        same_timestamp_count = 0

                        # Just move back one millisecond from the oldest timestamp
                        new_end_timestamp = oldest_timestamp - 1

                        # Ensure the new timestamp is earlier than anything we've seen before
                        if (
                            earliest_timestamp_seen is not None
                            and new_end_timestamp >= earliest_timestamp_seen
                        ):
                            # Something is wrong - we should be getting older data
                            # Use a safe fallback
                            logger.warning(
                                f"Timestamp inconsistency detected for {symbol}. Using safe fallback."
                            )
                            new_end_timestamp = earliest_timestamp_seen - (
                                24 * 60 * 60 * 1000
                            )  # 1 day earlier

                        # Ensure we don't go beyond target start timestamp
                        if new_end_timestamp < target_start_timestamp:
                            new_end_timestamp = target_start_timestamp

                        current_end_timestamp = new_end_timestamp

                    # Update last_timestamp for next iteration
                    last_timestamp = oldest_timestamp

                    logger.info(
                        f"Got data, next request will end at {datetime.fromtimestamp(current_end_timestamp/1000)} (attempt #{attempts})"
                    )
                else:
                    # If no data in this chunk, move back by a calculated amount based on timeframe
                    attempts += 1
                    time_increment = self._calculate_time_increment(bingx_interval)

                    # Ensure the new timestamp is earlier than anything we've seen before
                    new_end_timestamp = current_end_timestamp - time_increment

                    # Ensure we don't go beyond target start timestamp
                    if new_end_timestamp < target_start_timestamp:
                        new_end_timestamp = target_start_timestamp

                    current_end_timestamp = new_end_timestamp
                    logger.info(
                        f"No data in chunk, moving back to {datetime.fromtimestamp(current_end_timestamp/1000)} (attempt #{attempts})"
                    )

                # Sleep to avoid hitting rate limits
                time.sleep(self.rate_limit)

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                attempts += 1
                time.sleep(2)  # Reduced wait time on error

        # Check if we reached max attempts
        if attempts >= max_attempts:
            logger.warning(
                f"Reached maximum number of attempts ({max_attempts}) for {symbol}. Stopping data collection."
            )

        # Convert to DataFrame
        if not all_ohlcv:
            logger.warning(
                f"No data found for symbol {symbol} in the entire date range from {self.start_date} to {self.end_date}"
            )
            logger.warning(
                f"This symbol may have been listed after your end date or delisted before your start date."
            )
            raise ValueError(
                f"No data returned for symbol {symbol} in the specified date range. This symbol may have been listed after your start date ({self.start_date}) or delisted before your end date ({self.end_date})."
            )

        # Log statistics about the data we collected
        logger.info(
            f"Collected a total of {len(all_ohlcv)} data points for {symbol} after {attempts} attempts"
        )

        # Remove duplicates before creating DataFrame
        unique_timestamps = set()
        unique_ohlcv = []

        for row in all_ohlcv:
            if row[0] not in unique_timestamps:
                unique_timestamps.add(row[0])
                unique_ohlcv.append(row)

        logger.info(
            f"After removing duplicates: {len(unique_ohlcv)} unique data points for {symbol}"
        )

        df = pd.DataFrame(
            unique_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        df.set_index("timestamp", inplace=True)

        # Sort index to ensure chronological order (ascending)
        df.sort_index(inplace=True)

        # Filter to our desired date range
        # Add a buffer for the window calculation
        buffer_start = datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(
            days=self.window * 5 // 288
        )
        end_date = datetime.strptime(self.end_date, "%Y-%m-%d")
        df = df[(df.index >= buffer_start) & (df.index <= end_date)]

        # Log how far back we were able to get data
        if len(df) > 0:
            oldest_data = df.index.min()
            newest_data = df.index.max()
            logger.info(
                f"Data for {symbol} spans from {oldest_data} to {newest_data} ({len(df)} data points)"
            )

            if oldest_data > buffer_start:
                logger.warning(
                    f"Could not get data back to requested start date with buffer ({buffer_start}). Earliest data is from {oldest_data}."
                )
                logger.warning(
                    f"This token may have been listed after your requested start date."
                )

        return df

    def fetch_and_prepare_data(self):
        """Fetch data for both symbols and prepare it for analysis."""
        logger.info(f"Fetching data for {self.symbol_a} and {self.symbol_b}")

        # Convert symbols to BingX format (e.g., BTC/USDT -> BTC-USDT)
        symbol_a_bingx = self.symbol_a.replace("/", "-")
        symbol_b_bingx = self.symbol_b.replace("/", "-")

        logger.info(f"Using BingX symbols: {symbol_a_bingx} and {symbol_b_bingx}")

        # Initialize data containers
        self.data_a = None
        self.data_b = None
        error_a = None
        error_b = None

        # Fetch data for both symbols sequentially to better identify issues
        # This helps us know exactly which symbol is causing problems
        try:
            logger.info(f"Fetching data for {symbol_a_bingx}...")
            self.data_a = self._fetch_ohlcv_data(symbol_a_bingx)
            logger.info(
                f"Successfully fetched {len(self.data_a)} data points for {symbol_a_bingx}"
            )
        except Exception as e:
            error_a = str(e)
            logger.error(f"Error fetching data for {symbol_a_bingx}: {e}")

        try:
            logger.info(f"Fetching data for {symbol_b_bingx}...")
            self.data_b = self._fetch_ohlcv_data(symbol_b_bingx)
            logger.info(
                f"Successfully fetched {len(self.data_b)} data points for {symbol_b_bingx}"
            )
        except Exception as e:
            error_b = str(e)
            logger.error(f"Error fetching data for {symbol_b_bingx}: {e}")

        # Check if we have data for both symbols
        if self.data_a is None and self.data_b is None:
            error_msg = f"Failed to fetch data for both symbols. "
            if error_a:
                error_msg += f"Error for {symbol_a_bingx}: {error_a}. "
            if error_b:
                error_msg += f"Error for {symbol_b_bingx}: {error_b}."
            raise ValueError(error_msg)

        if self.data_a is None:
            raise ValueError(f"Failed to fetch data for {symbol_a_bingx}: {error_a}")

        if self.data_b is None:
            raise ValueError(f"Failed to fetch data for {symbol_b_bingx}: {error_b}")

        # Double-check for any remaining duplicate indices
        if self.data_a.index.duplicated().any():
            dup_count = self.data_a.index.duplicated().sum()
            logger.warning(
                f"Found {dup_count} duplicate timestamps in data for {self.symbol_a} after processing. Removing duplicates."
            )
            self.data_a = self.data_a[~self.data_a.index.duplicated(keep="last")]

        if self.data_b.index.duplicated().any():
            dup_count = self.data_b.index.duplicated().sum()
            logger.warning(
                f"Found {dup_count} duplicate timestamps in data for {self.symbol_b} after processing. Removing duplicates."
            )
            self.data_b = self.data_b[~self.data_b.index.duplicated(keep="last")]

        # Find common timestamps where both symbols have data
        logger.info("Filtering for common timestamps between both symbols")
        common_timestamps = self.data_a.index.intersection(self.data_b.index)

        # Log detailed information about the data ranges to help diagnose issues
        logger.info(
            f"Symbol A ({self.symbol_a}) data range: {self.data_a.index.min()} to {self.data_a.index.max()}, {len(self.data_a)} points"
        )
        logger.info(
            f"Symbol B ({self.symbol_b}) data range: {self.data_b.index.min()} to {self.data_b.index.max()}, {len(self.data_b)} points"
        )

        # Filter data to only include common timestamps
        self.data_a = self.data_a.loc[common_timestamps]
        self.data_b = self.data_b.loc[common_timestamps]

        # Ensure both DataFrames have the same index order
        self.data_a = self.data_a.sort_index()
        self.data_b = self.data_b.sort_index()

        logger.info(f"After filtering: {len(common_timestamps)} common data points")

        # Check if we have enough data to proceed
        if len(common_timestamps) == 0:
            raise ValueError(
                f"No common data points found between {symbol_a_bingx} and {symbol_b_bingx}. "
                f"This could be because one or both symbols don't have data for the specified time period, "
                f"or the symbols trade on different schedules."
            )

        if len(common_timestamps) < self.window:
            raise ValueError(
                f"Not enough common data points ({len(common_timestamps)}) to proceed with analysis. Need at least {self.window} points."
            )

        # Merge data on timestamp
        logger.info("Merging and preparing data")
        self.merged_data = pd.DataFrame(
            {
                "open_a": self.data_a["open"],
                "high_a": self.data_a["high"],
                "low_a": self.data_a["low"],
                "close_a": self.data_a["close"],
                "volume_a": self.data_a["volume"],
                "open_b": self.data_b["open"],
                "high_b": self.data_b["high"],
                "low_b": self.data_b["low"],
                "close_b": self.data_b["close"],
                "volume_b": self.data_b["volume"],
            }
        )

        # Check for any remaining NaN values
        if self.merged_data.isna().any().any():
            logger.warning(
                "Still found NaN values after merging. Performing forward fill."
            )
            self.merged_data = self.merged_data.ffill()
            self.merged_data.dropna(inplace=True)

        # Use the alpha module to calculate spread
        logger.info(f"Calculating spread using {self.alpha_name}")

        # Create temporary DataFrames with required structure for alpha calculation
        df_a = pd.DataFrame(
            {
                "open": self.merged_data["open_a"],
                "high": self.merged_data["high_a"],
                "low": self.merged_data["low_a"],
                "close": self.merged_data["close_a"],
                "volume": self.merged_data["volume_a"],
            }
        )

        df_b = pd.DataFrame(
            {
                "open": self.merged_data["open_b"],
                "high": self.merged_data["high_b"],
                "low": self.merged_data["low_b"],
                "close": self.merged_data["close_b"],
                "volume": self.merged_data["volume_b"],
            }
        )

        # Calculate spread using the specified alpha
        self.merged_data["spread"] = calculate_spread(
            df_a, df_b, self.alpha_name, **self.alpha_params
        )

        # For backward compatibility with the original alpha1 (log price difference),
        # keep the log price calculations if using alpha1
        if self.alpha_name == "alpha1":
            self.merged_data["log_a"] = np.log(self.merged_data["close_a"])
            self.merged_data["log_b"] = np.log(self.merged_data["close_b"])

        # Calculate Bollinger Bands with shift(1) to avoid look-ahead bias
        self.merged_data["middle_band"] = (
            self.merged_data["spread"].rolling(window=self.window).mean().shift(1)
        )
        self.merged_data["std_dev"] = (
            self.merged_data["spread"].rolling(window=self.window).std().shift(1)
        )

        # Upper and lower Bollinger Bands for entry signals
        self.merged_data["upper_band"] = (
            self.merged_data["middle_band"]
            + self.bb_entry_multiplier * self.merged_data["std_dev"]
        )
        self.merged_data["lower_band"] = (
            self.merged_data["middle_band"]
            - self.bb_entry_multiplier * self.merged_data["std_dev"]
        )

        # Drop rows with NaN (initial window period)
        self.merged_data.dropna(inplace=True)

        # Add next bar's open price for execution (to avoid look-ahead bias)
        self.merged_data["next_open_a"] = self.merged_data["open_a"].shift(-1)
        self.merged_data["next_open_b"] = self.merged_data["open_b"].shift(-1)

        # Drop last row since it doesn't have next_open
        self.merged_data = self.merged_data[:-1]

        logger.info(f"Data prepared. Total bars: {len(self.merged_data)}")

    def generate_signals(self):
        """Generate trading signals based on Bollinger Bands."""
        logger.info("Generating trading signals")

        # Initialize signal column
        self.merged_data["signal"] = 0

        # Entry signals based on crossovers
        for i in range(1, len(self.merged_data)):
            current_spread = self.merged_data["spread"].iloc[i]
            prev_spread = self.merged_data["spread"].iloc[i - 1]
            current_upper = self.merged_data["upper_band"].iloc[i]
            current_lower = self.merged_data["lower_band"].iloc[i]

            # Detect when spread crosses down through upper band
            if prev_spread > current_upper and current_spread <= current_upper:
                self.merged_data.iloc[i, self.merged_data.columns.get_loc("signal")] = (
                    -1
                )  # Long B / Short A

            # Detect when spread crosses up through lower band
            elif prev_spread < current_lower and current_spread >= current_lower:
                self.merged_data.iloc[i, self.merged_data.columns.get_loc("signal")] = (
                    1  # Long A / Short B
                )

        # Exit signals will be handled during the backtest simulation (crossing middle band)

        logger.info(
            f"Signals generated. Long A/Short B entries: {(self.merged_data['signal'] == 1).sum()}, "
            f"Long B/Short A entries: {(self.merged_data['signal'] == -1).sum()}"
        )

    def run_backtest(self):
        """Run the backtest simulation."""
        logger.info("Starting backtest simulation")

        # Prepare for tracking equity curve and trade performance
        self.merged_data["equity"] = self.initial_equity  # Starting equity
        self.merged_data["cash"] = self.initial_equity  # Available cash
        self.merged_data["trade_pnl"] = 0.0
        self.merged_data["cum_pnl"] = 0.0
        self.merged_data["fees"] = 0.0
        self.merged_data["position_value"] = 0.0  # Track position value

        # Add tracking for individual symbol performances
        self.merged_data["equity_a"] = self.initial_equity  # Symbol A equity curve
        self.merged_data["equity_b"] = self.initial_equity  # Symbol B equity curve
        self.merged_data["pnl_a"] = 0.0  # P&L contribution from symbol A
        self.merged_data["pnl_b"] = 0.0  # P&L contribution from symbol B

        # Tracking variables
        current_position = 0  # 0: no position, 1: long A short B, -1: long B short A
        entry_price_a = 0.0
        entry_price_b = 0.0
        position_size_a = 0.0
        position_size_b = 0.0
        entry_date = None
        trade_count = 0
        current_equity = self.initial_equity
        current_cash = self.initial_equity

        # For individual symbol tracking
        cum_pnl_a = 0.0  # Cumulative P&L from symbol A
        cum_pnl_b = 0.0  # Cumulative P&L from symbol B

        # Trade details for blotter
        trades_list = []

        # Simulate trading bar by bar
        for i, (idx, row) in enumerate(self.merged_data.iterrows()):
            # Skip the last bar (we need next_open for execution)
            if i == len(self.merged_data) - 1:
                break

            # Current spread and signal
            spread = row["spread"]
            signal = row["signal"]

            # Current prices for MTM calculation
            current_close_a = row["close_a"]
            current_close_b = row["close_b"]

            # Next bar open prices for execution
            next_open_a = row["next_open_a"]
            next_open_b = row["next_open_b"]

            # Check for exit conditions if in a position
            exit_signal = False

            # Reset position value for this bar
            position_value = 0.0
            mtm_pnl = 0.0
            mtm_pnl_a = 0.0  # Mark-to-market P&L from symbol A
            mtm_pnl_b = 0.0  # Mark-to-market P&L from symbol B

            if current_position != 0:
                # Take profit condition (spread crosses middle band)
                # Use current middle band instead of entry middle band
                if (current_position == 1 and spread >= row["middle_band"]) or (
                    current_position == -1 and spread <= row["middle_band"]
                ):
                    exit_signal = True

                # Calculate current position value and MTM P&L
                current_value_a = position_size_a * current_close_a
                current_value_b = position_size_b * current_close_b

                if current_position == 1:  # Long A, Short B
                    # Net position value (long A, short B)
                    position_value = current_value_a - current_value_b

                    # MTM P&L compared to entry
                    entry_value_a = position_size_a * entry_price_a
                    entry_value_b = position_size_b * entry_price_b

                    # Calculate individual contributions to P&L
                    mtm_pnl_a = current_value_a - entry_value_a  # P&L from long A
                    mtm_pnl_b = entry_value_b - current_value_b  # P&L from short B
                    mtm_pnl = mtm_pnl_a + mtm_pnl_b
                else:  # Long B, Short A
                    # Net position value (long B, short A)
                    position_value = current_value_b - current_value_a

                    # MTM P&L compared to entry
                    entry_value_a = position_size_a * entry_price_a
                    entry_value_b = position_size_b * entry_price_b

                    # Calculate individual contributions to P&L
                    mtm_pnl_a = entry_value_a - current_value_a  # P&L from short A
                    mtm_pnl_b = current_value_b - entry_value_b  # P&L from long B
                    mtm_pnl = mtm_pnl_a + mtm_pnl_b

            # Update equity: cash + MTM P&L from open positions
            self.merged_data.at[idx, "cash"] = current_cash
            self.merged_data.at[idx, "position_value"] = position_value
            self.merged_data.at[idx, "equity"] = current_cash + mtm_pnl

            # Update individual symbol contributions
            self.merged_data.at[idx, "pnl_a"] = mtm_pnl_a
            self.merged_data.at[idx, "pnl_b"] = mtm_pnl_b

            # Update equity curves for individual symbols
            self.merged_data.at[idx, "equity_a"] = (
                self.initial_equity + cum_pnl_a + mtm_pnl_a
            )
            self.merged_data.at[idx, "equity_b"] = (
                self.initial_equity + cum_pnl_b + mtm_pnl_b
            )

            current_equity = current_cash + mtm_pnl

            # Determine if we need to take action
            if exit_signal or (
                current_position == 0 and signal != 0 and signal != current_position
            ):
                # Close existing position if any
                if current_position != 0:
                    # Calculate execution prices with slippage
                    exit_price_a = next_open_a * (
                        1 + self.slippage * (-1 if current_position == 1 else 1)
                    )
                    exit_price_b = next_open_b * (
                        1 + self.slippage * (1 if current_position == 1 else -1)
                    )

                    # Calculate transaction costs
                    cost_a = position_size_a * exit_price_a * self.taker_fee
                    cost_b = position_size_b * exit_price_b * self.taker_fee
                    total_fees = cost_a + cost_b

                    # Calculate final P&L for the trade
                    if current_position == 1:  # Long A, Short B
                        exit_value_a = position_size_a * exit_price_a
                        exit_value_b = position_size_b * exit_price_b
                        entry_value_a = position_size_a * entry_price_a
                        entry_value_b = position_size_b * entry_price_b

                        # Calculate individual contributions to P&L
                        pnl_a = exit_value_a - entry_value_a - cost_a  # P&L from long A
                        pnl_b = (
                            entry_value_b - exit_value_b - cost_b
                        )  # P&L from short B
                        trade_pnl = pnl_a + pnl_b

                        # Update cumulative P&L for each symbol
                        cum_pnl_a += pnl_a
                        cum_pnl_b += pnl_b
                    else:  # Long B, Short A
                        exit_value_a = position_size_a * exit_price_a
                        exit_value_b = position_size_b * exit_price_b
                        entry_value_a = position_size_a * entry_price_a
                        entry_value_b = position_size_b * entry_price_b

                        # Calculate individual contributions to P&L
                        pnl_a = (
                            entry_value_a - exit_value_a - cost_a
                        )  # P&L from short A
                        pnl_b = exit_value_b - entry_value_b - cost_b  # P&L from long B
                        trade_pnl = pnl_a + pnl_b

                        # Update cumulative P&L for each symbol
                        cum_pnl_a += pnl_a
                        cum_pnl_b += pnl_b

                    # Update cash and equity
                    current_cash += trade_pnl
                    current_equity = current_cash  # No open positions now

                    # Record trade
                    trade_data = {
                        "entry_date": entry_date,
                        "exit_date": idx,
                        "position": (
                            "Long A/Short B"
                            if current_position == 1
                            else "Long B/Short A"
                        ),
                        "entry_price_a": entry_price_a,
                        "entry_price_b": entry_price_b,
                        "exit_price_a": exit_price_a,
                        "exit_price_b": exit_price_b,
                        "size_a": position_size_a,
                        "size_b": position_size_b,
                        "pnl": trade_pnl,
                        "pnl_a": pnl_a,
                        "pnl_b": pnl_b,
                        "costs": total_fees,
                        "holding_bars": i - trade_count,
                    }
                    trades_list.append(trade_data)

                    # Update trade P&L and fees
                    self.merged_data.at[idx, "trade_pnl"] = trade_pnl
                    self.merged_data.at[idx, "fees"] = total_fees
                    self.merged_data.at[idx, "cash"] = current_cash
                    self.merged_data.at[idx, "equity"] = current_equity

                    # Update individual symbol equity curves after trade
                    self.merged_data.at[idx, "equity_a"] = (
                        self.initial_equity + cum_pnl_a
                    )
                    self.merged_data.at[idx, "equity_b"] = (
                        self.initial_equity + cum_pnl_b
                    )

                    # Reset position
                    current_position = 0
                    position_size_a = 0
                    position_size_b = 0
                    trade_count = i
                    mtm_pnl_a = 0
                    mtm_pnl_b = 0

                # Open new position if there's a signal
                if signal != 0:
                    current_position = signal

                    # Calculate execution prices with slippage
                    entry_price_a = next_open_a * (
                        1 + self.slippage * (1 if current_position == 1 else -1)
                    )
                    entry_price_b = next_open_b * (
                        1 + self.slippage * (-1 if current_position == 1 else 1)
                    )

                    # Calculate notional sizes based on current equity (10% per leg)
                    notional_per_leg = current_equity * self.position_size_pct
                    position_size_a = notional_per_leg / entry_price_a
                    position_size_b = notional_per_leg / entry_price_b

                    # Calculate transaction costs
                    cost_a = position_size_a * entry_price_a * self.taker_fee
                    cost_b = position_size_b * entry_price_b * self.taker_fee
                    total_fees = cost_a + cost_b

                    # Update cash (reduced by fees)
                    current_cash -= total_fees

                    # Update equity
                    current_equity = current_cash  # No unrealized P&L yet

                    # Record entry
                    entry_date = idx
                    trade_count = i

                    # Update fees
                    self.merged_data.at[idx, "fees"] = total_fees
                    self.merged_data.at[idx, "cash"] = current_cash
                    self.merged_data.at[idx, "equity"] = current_equity

            # Update cumulative P&L
            self.merged_data.at[idx, "cum_pnl"] = self.merged_data.iloc[: i + 1][
                "trade_pnl"
            ].sum()

        # Store trades for analysis
        self.trades = pd.DataFrame(trades_list)

        # Calculate performance metrics
        self._calculate_performance_metrics()

        logger.info(f"Backtest completed. Total trades: {len(trades_list)}")

    def _calculate_performance_metrics(self):
        """Calculate performance metrics from backtest results."""
        logger.info("Calculating performance metrics")

        # Calculate equity curve and returns
        self.equity_curve = self.merged_data["equity"].copy()
        self.merged_data["returns"] = self.equity_curve.pct_change()

        # Calculate basic metrics
        total_trades = len(self.trades) if not self.trades.empty else 0
        winning_trades = (
            len(self.trades[self.trades["pnl"] > 0]) if not self.trades.empty else 0
        )

        # Avoid division by zero
        if total_trades > 0:
            win_rate = winning_trades / total_trades
        else:
            win_rate = 0

        # Calculate Sharpe ratio (annualized)
        if len(self.merged_data) > 1:
            # For 5-minute data, multiply by sqrt(288*252) for annualization
            sharpe = (
                np.sqrt(288 * 252)
                * self.merged_data["returns"].mean()
                / self.merged_data["returns"].std()
            )
        else:
            sharpe = 0

        # Calculate drawdown
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # Calculate average P&L per trade
        avg_pnl = self.trades["pnl"].mean() if not self.trades.empty else 0

        # Final equity
        final_equity = (
            self.equity_curve.iloc[-1]
            if len(self.equity_curve) > 0
            else self.initial_equity
        )

        # Calculate total return
        total_return = (final_equity / self.initial_equity - 1) * 100

        # Store metrics
        self.performance_metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_pnl_per_trade": avg_pnl,
            "final_equity": final_equity,
            "total_return": total_return,
        }

        logger.info(
            f"Performance metrics calculated. Sharpe: {sharpe:.2f}, Max Drawdown: {max_drawdown:.2%}"
        )

    def plot_results(self):
        """
        Create and save equity curve plots to files.
        """
        if self.equity_curve is None or len(self.equity_curve) == 0:
            logger.warning("No equity curve data to plot")
            return

        # Create output directory if it doesn't exist
        os.makedirs("equity_curves", exist_ok=True)

        # Create a pair identifier for the filename
        pair_id = f"{self.symbol_a.replace('/', '_')}-{self.symbol_b.replace('/', '_')}"

        # Create figure with 4 subplots
        plt.figure(figsize=(12, 16))

        # Plot pair trading equity curve
        plt.subplot(4, 1, 1)
        plt.plot(self.equity_curve, color="purple", linewidth=2)
        plt.title(f"Pair Trading Equity Curve: {self.symbol_a} - {self.symbol_b}")
        plt.grid(True)
        plt.ylabel("Equity (USDT)")

        # Plot equity curve for symbol A
        plt.subplot(4, 1, 2)
        plt.plot(self.merged_data["equity_a"], color="blue", linewidth=2)
        plt.title(f"Symbol A Contribution: {self.symbol_a}")
        plt.grid(True)
        plt.ylabel("Equity (USDT)")

        # Plot equity curve for symbol B
        plt.subplot(4, 1, 3)
        plt.plot(self.merged_data["equity_b"], color="green", linewidth=2)
        plt.title(f"Symbol B Contribution: {self.symbol_b}")
        plt.grid(True)
        plt.ylabel("Equity (USDT)")

        # Plot drawdown
        plt.subplot(4, 1, 4)
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak
        plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color="r")
        plt.title("Drawdown")
        plt.grid(True)
        plt.ylabel("Drawdown (%)")
        plt.xlabel("Date")

        plt.tight_layout()

        # Save the plot to a file
        filename = f"equity_curves/{pair_id}.png"
        plt.savefig(filename)
        logger.info(f"Equity curve saved to {filename}")

        # Close the plot
        plt.close()

    def export_results(self):
        """
        Export backtest results as equity curve images.
        """
        # Save equity curve image
        self.plot_results()
        logger.info(f"Equity curve image exported for {self.symbol_a}-{self.symbol_b}")

    def print_summary(self):
        """Print summary of backtest results."""
        if not self.performance_metrics:
            logger.warning("No performance metrics available")
            return

        # Create a pair identifier for reference
        pair_id = f"{self.symbol_a.replace('/', '_')}-{self.symbol_b.replace('/', '_')}"

        print("\n" + "=" * 50)
        print(f"PAIR TRADING BACKTEST RESULTS: {self.symbol_a} - {self.symbol_b}")
        print("=" * 50)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Timeframe: {self.timeframe}")
        print(f"Initial equity: ${self.initial_equity:.2f}")
        print(f"Position size per leg: {self.position_size_pct:.1%} of equity")
        print(f"Window size: {self.window}")
        print(f"Bollinger Band entry multiplier: {self.bb_entry_multiplier}")
        print("-" * 50)
        print(f"Total trades: {self.performance_metrics['total_trades']}")
        print(f"Win rate: {self.performance_metrics['win_rate']:.2%}")
        print(
            f"Average P&L per trade: ${self.performance_metrics['avg_pnl_per_trade']:.2f}"
        )
        print(f"Sharpe ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
        print(f"Maximum drawdown: {self.performance_metrics['max_drawdown']:.2%}")
        print(f"Total return: {self.performance_metrics['total_return']:.2f}%")
        print(f"Final equity: ${self.performance_metrics['final_equity']:.2f}")
        print("=" * 50)
        print(f"\nEquity curve image saved to: equity_curves/{pair_id}.png")
        print("=" * 50 + "\n")

    def run_for_pair(self, pair):
        """Run the backtest for a specific pair."""
        logger.info(f"Running backtest for pair: {pair['symbolA']} - {pair['symbolB']}")

        # Set current pair
        self.current_pair = pair
        self.symbol_a = pair["symbolA"]
        self.symbol_b = pair["symbolB"]

        # Reset data containers for this pair
        self.data_a = None
        self.data_b = None
        self.merged_data = None
        self.signals = None
        self.trades = []
        self.equity_curve = None
        self.performance_metrics = {}

        try:
            # Fetch and prepare data
            self.fetch_and_prepare_data()

            # Check for duplicate indices one more time before proceeding
            if (
                self.data_a.index.duplicated().any()
                or self.data_b.index.duplicated().any()
            ):
                logger.error(
                    f"Found duplicate timestamps after data preparation for {self.symbol_a} or {self.symbol_b}"
                )
                logger.error(
                    f"This would cause 'cannot reindex on an axis with duplicate labels' error"
                )

                # Remove duplicates one more time
                self.data_a = self.data_a[~self.data_a.index.duplicated(keep="last")]
                self.data_b = self.data_b[~self.data_b.index.duplicated(keep="last")]

                # Find common timestamps again
                common_timestamps = self.data_a.index.intersection(self.data_b.index)
                self.data_a = self.data_a.loc[common_timestamps]
                self.data_b = self.data_b.loc[common_timestamps]

                # Sort indices
                self.data_a = self.data_a.sort_index()
                self.data_b = self.data_b.sort_index()

                logger.info(
                    f"Removed duplicates. Now have {len(self.data_a)} data points for analysis."
                )

            # Generate trading signals
            self.generate_signals()

            # Run backtest simulation
            self.run_backtest()

            # Save equity curve image
            self.export_results()

            # Print summary
            self.print_summary()

            # Store results for this pair
            pair_key = f"{self.symbol_a}-{self.symbol_b}"
            self.all_results[pair_key] = {
                "performance_metrics": self.performance_metrics,
                "trades": (
                    self.trades.copy()
                    if not isinstance(self.trades, list) and not self.trades.empty
                    else None
                ),
                "equity_curve": (
                    self.equity_curve.copy() if self.equity_curve is not None else None
                ),
            }

            return self.performance_metrics

        except ValueError as e:
            # Specific handling for invalid symbols
            if "does not exist in the exchange" in str(e):
                logger.warning(f"Skipping pair {self.symbol_a} - {self.symbol_b}: {e}")
            else:
                logger.error(
                    f"Error running backtest for pair {self.symbol_a} - {self.symbol_b}: {e}"
                )
            return None
        except Exception as e:
            logger.error(
                f"Error running backtest for pair {self.symbol_a} - {self.symbol_b}: {e}"
            )
            return None

    def run(self):
        """Run the complete backtest process for all pairs."""
        logger.info(f"Starting backtest for {len(self.pairs)} pairs")

        all_metrics = []

        for pair in self.pairs:
            metrics = self.run_for_pair(pair)
            if metrics:
                pair_result = {
                    "symbolA": pair["symbolA"],
                    "symbolB": pair["symbolB"],
                    **metrics,
                }
                all_metrics.append(pair_result)

        # Create a summary dataframe of all results
        if all_metrics:
            summary_df = pd.DataFrame(all_metrics)

            # Display summary table
            print("\n" + "=" * 50)
            print("SUMMARY OF ALL PAIRS")
            print("=" * 50)
            print(
                summary_df[
                    [
                        "symbolA",
                        "symbolB",
                        "total_trades",
                        "win_rate",
                        "sharpe_ratio",
                        "max_drawdown",
                        "total_return",
                    ]
                ]
            )
            print("=" * 50 + "\n")
            print(
                "All equity curve images have been saved to the 'equity_curves' directory"
            )

        return self.all_results


if __name__ == "__main__":
    # Example of running backtest

    # Print available alpha strategies
    list_available_alphas()

    # Configuration
    TIMEFRAME = "15m"
    START_DATE = "2023-04-01"
    END_DATE = "2025-07-15"
    WINDOW = 500
    ALPHA_NAME = "alpha3"  # Choose which alpha to use for spread calculation
    ALPHA_PARAMS = {}  # Additional parameters for alpha calculation

    # Example: Using alpha2 with a custom window
    # ALPHA_NAME = "alpha2"
    # ALPHA_PARAMS = {"window": 15}

    # Define pairs
    PAIRS = [
      { "symbolA": "1INCH/USDT", "symbolB": "1000BONK/USDT" },
{ "symbolA": "AAVE/USDT", "symbolB": "ENA/USDT" },
{ "symbolA": "ADA/USDT", "symbolB": "RSR/USDT" },
{ "symbolA": "ATOM/USDT", "symbolB": "ID/USDT" },
{ "symbolA": "ATOM/USDT", "symbolB": "NFP/USDT" },
{ "symbolA": "ATOM/USDT", "symbolB": "XAI/USDT" },
{ "symbolA": "AVAX/USDT", "symbolB": "ARK/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "FET/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "GRT/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "LUNC/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "MINA/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "OM/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "SHIB/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "SOL/USDT" },
{ "symbolA": "AXS/USDT", "symbolB": "YFI/USDT" },
{ "symbolA": "CELO/USDT", "symbolB": "ASTR/USDT" },
{ "symbolA": "CELO/USDT", "symbolB": "CFX/USDT" },
{ "symbolA": "CELO/USDT", "symbolB": "IOTA/USDT" },
{ "symbolA": "CELO/USDT", "symbolB": "LUNC/USDT" },
{ "symbolA": "CELO/USDT", "symbolB": "MINA/USDT" },
{ "symbolA": "CELO/USDT", "symbolB": "RDNT/USDT" },
{ "symbolA": "CELO/USDT", "symbolB": "SLP/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "1000000BABYDOGE/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "BLUR/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "ENA/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "ID/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "NFP/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "TLM/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "XAI/USDT" },
{ "symbolA": "CHR/USDT", "symbolB": "ZIL/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "1000000BABYDOGE/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "BLUR/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "ENA/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "MTL/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "REZ/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "SSV/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "TLM/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "XAI/USDT" },
{ "symbolA": "CHZ/USDT", "symbolB": "ZIL/USDT" },
{ "symbolA": "DOGE/USDT", "symbolB": "ETHFI/USDT" },
{ "symbolA": "DYDX/USDT", "symbolB": "AEVO/USDT" },
{ "symbolA": "EGLD/USDT", "symbolB": "STRK/USDT" },
{ "symbolA": "ENJ/USDT", "symbolB": "XAI/USDT" },
{ "symbolA": "ENS/USDT", "symbolB": "ALTCOIN/USDT" },
{ "symbolA": "ENS/USDT", "symbolB": "GOAT/USDT" },
{ "symbolA": "ENS/USDT", "symbolB": "IOST/USDT" },
{ "symbolA": "ENS/USDT", "symbolB": "NEO/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "APE/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "CAKE/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "CHR/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "FLUX/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "GALA/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "ID/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "KDA/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "LUNA/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "ORDI/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "RUNE/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "TLM/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "WOO/USDT" },
{ "symbolA": "GRT/USDT", "symbolB": "XAI/USDT" },
{ "symbolA": "LINK/USDT", "symbolB": "BAKE/USDT" },
{ "symbolA": "LINK/USDT", "symbolB": "BEAM/USDT" },
{ "symbolA": "LINK/USDT", "symbolB": "CTC/USDT" },
{ "symbolA": "LINK/USDT", "symbolB": "GALA/USDT" },
{ "symbolA": "LINK/USDT", "symbolB": "HBAR/USDT" },
{ "symbolA": "LINK/USDT", "symbolB": "ID/USDT" },
{ "symbolA": "LINK/USDT", "symbolB": "SSV/USDT" },
{ "symbolA": "STORJ/USDT", "symbolB": "CELO/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "10000SATS/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "ANKR/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "BEAM/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "FIO/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "ID/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "MTL/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "PNUT/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "SNX/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "TLM/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "VTHO/USDT" },
{ "symbolA": "SUSHI/USDT", "symbolB": "XAI/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "10000SATS/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "BEAM/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "FIO/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "ID/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "LUNA/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "METIS/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "PNUT/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "SSV/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "SUI/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "TLM/USDT" },
{ "symbolA": "THETA/USDT", "symbolB": "ZRX/USDT" },
{ "symbolA": "VET/USDT", "symbolB": "BIO/USDT" },
{ "symbolA": "VET/USDT", "symbolB": "TOKEN/USDT" },
{ "symbolA": "VET/USDT", "symbolB": "ZETA/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "1000PEPE/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "ETHFI/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "HIVE/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "HOT/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "ME/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "NOT/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "PEOPLE/USDT" },
{ "symbolA": "YGG/USDT", "symbolB": "VELODROME/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "CETUS/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "ENA/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "HBAR/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "KDA/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "MEW/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "MTL/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "ROSE/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "SNX/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "SUI/USDT" },
{ "symbolA": "ZRX/USDT", "symbolB": "TLM/USDT" },

    ]

    # Run backtest
    backtest = PairTradingBacktest(
        pairs=PAIRS,
        start_date=START_DATE,
        end_date=END_DATE,
        timeframe=TIMEFRAME,
        window=WINDOW,
        initial_equity=100.0,
        position_size_pct=0.1,
        bb_entry_multiplier=2.0,
        alpha_name=ALPHA_NAME,
        alpha_params=ALPHA_PARAMS,
    )

    backtest.run()
    backtest.print_summary()
    backtest.export_results()
    backtest.plot_results()
