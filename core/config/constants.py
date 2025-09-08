from datetime import datetime

# Default configuration for the bot
DEFAULT_CONFIG = {
    "window": 500,  # Bollinger Band window period
    "bb_entry_multiplier": 2.0,  # Multiplier for entry Bollinger Bands
    "timeframe": "4h",  # Default timeframe
    "limit": 900,  # Number of candles to fetch
    "position_size_pct": 0.1,  # Position size as percentage of equity (0.01 = 1%)
    "max_concurrent_positions": 3,  # Maximum number of concurrent positions
    "max_retry_attempts": 3,  # Maximum API retry attempts
    "retry_delay": 2,  # Delay between retries in seconds
    "symbol_info_refresh_hours": 24,  # Hours before refreshing symbol info
    "risk_parity_lookback": 100,  # Number of recent spread returns to use for risk parity calculation
    "risk_parity_adjustment": 0.0,  # Adjustment percentage for risk parity weights (0.05 = 5%)
    "is_active": True,  # Whether the bot is active (allowed to open new positions)
    "max_loss_pct": 1.0
}

# Default trading symbols for initialization
DEFAULT_SYMBOLS = [
    {
        "symbol": "VANRY",
        "is_active": True,
        "timeframe": "4h",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
]

# Initialize global CONFIG variable
CONFIG = DEFAULT_CONFIG.copy()

# Simulation mode constants
TRADING_MODE = "simulation"  # or "live" or "simulation"
SIMULATION_INITIAL_EQUITY = 1000.0
TRADING_FEE = 0.0005  # 0.05%
SLIPPAGE = 0.0002  # 0.02%


# Position constants
class Position:
    LONG = "LONG"
    SHORT = "SHORT"


# Default paths for log file and cache directory
DEFAULT_DATA_ROOT = "C:/bot_temp_data_solo_trading"
DEFAULT_LOG_FILE_PATH = f"{DEFAULT_DATA_ROOT}/bot.log"
DEFAULT_CACHE_DIR_PATH = f"{DEFAULT_DATA_ROOT}/data_cache"

# Default API credentials
DEFAULT_TELEGRAM_BOT_TOKEN = "7695614330:AAEWMeYdP41_iXz6K6XxsZoXucRBkYETEJg"
DEFAULT_TELEGRAM_CHAT_ID = "676022032"
