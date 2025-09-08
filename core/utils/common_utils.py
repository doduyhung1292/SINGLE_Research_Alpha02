import logging
import traceback
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Import from config module
from core.config.config_manager import get_symbol_info

logger = logging.getLogger(__name__)


def convert_numpy_to_python(obj):
    """Convert any NumPy types in a dict to native Python types."""
    if isinstance(obj, dict):
        # Create a new dict to avoid modifying the input
        result = {}
        for key, value in obj.items():
            result[key] = convert_numpy_to_python(value)
        return result
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif str(type(obj)).startswith("<class 'numpy."):  # Catch any other NumPy types
        try:
            return obj.item()  # This converts most NumPy scalar types to native Python
        except:
            return str(obj)
    else:
        return obj


def round_quantity(quantity: float, symbol: str) -> float:
    """Round quantity to appropriate precision for the symbol."""
    try:
        # Get symbol information from cache/database
        symbol_info = get_symbol_info(symbol)

        # Get precision from symbol info
        precision = symbol_info.get("quantityPrecision", 4)

        # Round to the appropriate precision
        rounded_quantity = round(quantity, precision)

        # Ensure it meets minimum quantity requirement
        min_quantity = symbol_info.get("tradeMinQuantity", 0.001)
        if rounded_quantity < min_quantity:
            logger.warning(
                f"Rounded quantity {rounded_quantity} is below minimum {min_quantity} for {symbol}"
            )
            return min_quantity

        return rounded_quantity
    except Exception as e:
        logger.error(f"Error rounding quantity for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return quantity  # Return original quantity if rounding fails


def check_min_notional(
    symbol: str, quantity: float, price: float
) -> Tuple[bool, float]:
    """Check if the trade meets minimum notional value requirement and adjust if needed."""
    try:
        # Get symbol information
        symbol_info = get_symbol_info(symbol)

        # Calculate notional value
        notional = quantity * price

        # Check against minimum USDT requirement
        min_notional = symbol_info.get("tradeMinUSDT", 2.0)

        if notional < min_notional:
            logger.warning(
                f"Notional value {notional} is below minimum {min_notional} for {symbol}"
            )
            # Calculate adjusted quantity to meet minimum notional
            adjusted_quantity = min_notional / price
            # Round the adjusted quantity
            adjusted_quantity = round_quantity(adjusted_quantity, symbol)
            return False, adjusted_quantity

        return True, quantity
    except Exception as e:
        logger.error(f"Error checking minimum notional for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return False, quantity  # Be conservative and return False on error


def ensure_data_sort_oldest_to_newest(data):
    """
    Ensures market data is properly sorted from oldest to newest timestamp.
    This is critical for correct technical indicator calculations.

    Args:
        data: List of candle data dictionaries, each containing a 'time' key
             Can also be a deque

    Returns:
        The same data sorted by time in ascending order (oldest first)
    """
    if not data or len(data) < 2:
        return data

    # Convert to list if it's a deque or other iterable
    data_list = list(data)

    # Check if already sorted
    is_sorted = all(
        data_list[i]["time"] <= data_list[i + 1]["time"]
        for i in range(len(data_list) - 1)
    )
    if not is_sorted:
        logger.info(f"Sorting data from oldest to newest for calculation")
        data_list.sort(key=lambda x: x["time"])

    # Verify sort was successful
    verify_sort = all(
        data_list[i]["time"] <= data_list[i + 1]["time"]
        for i in range(len(data_list) - 1)
    )
    if not verify_sort:
        logger.error(
            "Data sort verification failed - still not properly sorted. Check data structure."
        )

    return data_list
