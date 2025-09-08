import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

from models.model_config import get_config_from_db, save_config_to_db
from models.model_trade_symbols import get_all_symbols, save_symbol_to_db
from models.model_symbol import save_symbol_info_to_db, get_symbol_info_from_db

from core.config.constants import DEFAULT_CONFIG, DEFAULT_PAIRS, CONFIG

logger = logging.getLogger(__name__)

# Global cache for symbol information
SYMBOL_INFO_CACHE = {}


def initialize_database():
    """Initialize database with default configuration, pairs, and symbol info."""
    initialize_config()
    initialize_symbols()
    initialize_symbol_info()


def initialize_config():
    """Initialize configuration in the database if it doesn't exist."""
    try:
        existing_config = get_config_from_db()
        if not existing_config:
            logger.info(
                "No configuration found in database. Adding default configuration."
            )
            save_config_to_db(DEFAULT_CONFIG)
            logger.info("Default configuration added to database successfully.")
        else:
            logger.info("Configuration already exists in database.")

            # Check if any new config keys need to be added to existing config
            updated = False
            for key, value in DEFAULT_CONFIG.items():
                if key not in existing_config:
                    existing_config[key] = value
                    updated = True

            if updated:
                save_config_to_db(existing_config)
                logger.info("Updated existing configuration with new fields.")
    except Exception as e:
        logger.error(f"Error initializing configuration: {e}")
        logger.error(traceback.format_exc())


def initialize_symbols():
    """Initialize default trading symbols in the database if they don't exist."""
    try:
        existing_symbols = get_all_symbols()
        if not existing_symbols:
            logger.info("No trading symbols found in database. Adding default symbols.")
            for symbol in existing_symbols:
                save_symbol_to_db(symbol)
                logger.info(
                    f"Added default symbol {symbol['symbol']} to database."
                )
        else:
            logger.info(f"Found {len(existing_symbols)} symbols in database.")
    except Exception as e:
        logger.error(f"Error initializing trading symbols: {e}")
        logger.error(traceback.format_exc())


def initialize_symbol_info():
    """Initialize symbol information in the database."""
    try:
        logger.info("Initializing symbol information...")
        symbols = get_all_symbols()
        symbols = get_unique_symbols(symbols)
        needs_update = False
        for symbol in symbols:
            symbol_info = get_symbol_info_from_db(symbol)
            if not symbol_info:
                needs_update = True
                break
        if needs_update:
            logger.info("Fetching symbol contract information from API...")
            contracts = fetch_and_save_symbol_contracts()
            if contracts:
                logger.info(
                    f"Successfully initialized information for {len(contracts)} symbols"
                )
            else:
                logger.error("Failed to initialize symbol information")
        else:
            logger.info("Symbol information already exists in database")
        load_symbol_info_cache()
    except Exception as e:
        logger.error(f"Error initializing symbol information: {e}")
        logger.error(traceback.format_exc())


def fetch_and_save_symbol_contracts() -> List[Dict[str, Any]]:
    """Fetch contract information from exchange API and save to database."""
    try:
        from core.api.exchange_api import get_symbol_contracts

        response = get_symbol_contracts()
        if not response or response.get("code") != 0 or "data" not in response:
            logger.error(f"Failed to get contract information: {response}")
            return []
        contracts = response["data"]
        saved_count = 0
        for contract in contracts:
            try:
                symbol = contract.get("symbol", "").split("-")[0]
                if not symbol:
                    continue
                if contract.get("currency", "") != "USDT":
                    continue
                symbol_info = {
                    "symbol": symbol,
                    "tradeMinQuantity": float(contract.get("tradeMinQuantity", 0)),
                    "tradeMinUSDT": 2.2,
                    "pricePrecision": int(contract.get("pricePrecision", 0)),
                    "quantityPrecision": int(contract.get("quantityPrecision", 4)),
                    "leverage": int(contract.get("leverage", 20)),
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_symbol_info_to_db(symbol_info)
                saved_count += 1
            except Exception as e:
                logger.error(
                    f"Error processing contract {contract.get('symbol', 'unknown')}: {e}"
                )
                continue
        logger.info(f"Saved information for {saved_count} symbols to database")
        return contracts
    except Exception as e:
        logger.error(f"Error fetching symbol contracts: {e}")
        logger.error(traceback.format_exc())
        return []


def load_symbol_info_cache():
    """Load symbol information from database into cache."""
    global SYMBOL_INFO_CACHE
    try:
        SYMBOL_INFO_CACHE.clear()
        list_symbols = get_all_symbols()
        symbols = get_unique_symbols(list_symbols)
        for symbol in symbols:
            symbol_info = get_symbol_info_from_db(symbol)
            if symbol_info:
                SYMBOL_INFO_CACHE[symbol] = symbol_info
        logger.info(
            f"Loaded information for {len(SYMBOL_INFO_CACHE)} symbols into cache"
        )
    except Exception as e:
        logger.error(f"Error loading symbol info cache: {e}")
        logger.error(traceback.format_exc())


def get_symbol_info(symbol: str) -> Dict[str, Any]:
    """Get symbol information from cache or database."""
    global SYMBOL_INFO_CACHE

    if symbol in SYMBOL_INFO_CACHE:
        return SYMBOL_INFO_CACHE[symbol]

    symbol_info = get_symbol_info_from_db(symbol)
    if symbol_info:
        SYMBOL_INFO_CACHE[symbol] = symbol_info
        return symbol_info

    logger.warning(f"Symbol information for {symbol} not found. Attempting to fetch...")
    fetch_and_save_symbol_contracts()

    symbol_info = get_symbol_info_from_db(symbol)
    if symbol_info:
        SYMBOL_INFO_CACHE[symbol] = symbol_info
        return symbol_info

    # Default values if all else fails
    logger.error(f"Could not get information for symbol {symbol}")
    return {
        "symbol": symbol,
        "tradeMinQuantity": 0.001,
        "tradeMinUSDT": 5.0,
        "pricePrecision": 4,
        "quantityPrecision": 4,
        "leverage": 20,
    }


def load_config_from_db() -> Dict[str, Any]:
    """Load configuration from database, using defaults for missing values."""
    try:
        db_config = get_config_from_db()
        if not db_config:
            logger.warning("Failed to load config from database, using default values")
            return DEFAULT_CONFIG

        # Merge with defaults to ensure all required keys exist
        for key, value in DEFAULT_CONFIG.items():
            if key not in db_config:
                db_config[key] = value

        logger.info(f"Loaded configuration from database")
        return db_config
    except Exception as e:
        logger.error(f"Error loading config from database: {e}")
        logger.error(traceback.format_exc())
        return DEFAULT_CONFIG


def get_unique_symbols(list_symbols: List[Dict[str, Any]]) -> List[str]:
    """Extract unique symbols from all list_symbols."""
    try:
        symbols = set()
        for pair in list_symbols:
            symbols.add(pair["symbol"])
        return list(symbols)
    except Exception as e:
        logger.error(f"Error extracting unique symbols: {e}")
        logger.error(traceback.format_exc())
        return []


def refresh_symbol_info():
    """Refresh symbol information from the exchange periodically."""
    try:
        logger.info("Refreshing symbol contract information...")
        contracts = fetch_and_save_symbol_contracts()
        if contracts:
            logger.info(
                f"Successfully refreshed information for {len(contracts)} symbols"
            )
            load_symbol_info_cache()
        else:
            logger.error("Failed to refresh symbol information")
    except Exception as e:
        logger.error(f"Error refreshing symbol information: {e}")
        logger.error(traceback.format_exc())


def set_bot_active_status(is_active: bool) -> bool:
    """
    Set the bot's active status to enable or disable opening new positions.
    When disabled, the bot will continue to monitor and close existing positions,
    but will not open any new positions.

    Args:
        is_active (bool): True to enable opening new positions, False to disable

    Returns:
        bool: True if the update was successful, False otherwise
    """
    try:
        # Get current config
        current_config = get_config_from_db()

        if not current_config:
            logger.error("Failed to get current configuration from database")
            return False

        # Update the is_active flag
        current_config["is_active"] = is_active

        # Save updated config back to database
        save_config_to_db(current_config)

        # Log the status change
        status_text = "ACTIVE" if is_active else "INACTIVE"
        logger.info(f"Bot status changed to {status_text}")

        # Import here to avoid circular imports
        from core.notification.notifier import send_telegram_message

        # Send notification via Telegram
        message = f"🤖 Bot status changed to {status_text}. " + (
            f"Bot will now open new positions when signals are detected."
            if is_active
            else f"Bot will NOT open new positions but will continue monitoring existing ones."
        )
        send_telegram_message(message)

        return True
    except Exception as e:
        logger.error(f"Error setting bot active status: {e}")
        logger.error(traceback.format_exc())
        return False
