import traceback
import schedule
import time
import logging
import os
import sys
import codecs
from datetime import datetime

# Import core modules
from core.config.config_manager import (
    initialize_database,
    load_config_from_db,
    refresh_symbol_info,
)
from core.notification.notifier import send_telegram_message, send_error_message
from core.config.constants import (
    CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_LOG_FILE_PATH,
    DEFAULT_CACHE_DIR_PATH,
    TRADING_MODE,
    SIMULATION_INITIAL_EQUITY,
    TRADING_FEE,
    SLIPPAGE,
    Position,
)

# Import models
from models.model_trade_symbols import get_all_symbols
from models.model_position import get_open_positions_from_db

# Import trading modules
from core.data.market_data_manager import prepare_market_data
from core.api.exchange_api import (
    check_margin_available,
    get_open_positions,
)
from core.api.get_equity_multi_accounts import (
    get_current_equity,
)
from core.position.position_manager import (
    check_stop_loss_for_open_positions,
    check_open_position,
)

from order_execution import execute_batch_orders

# Import from trading module
from trading import process_symbol, BatchOrder

# Import for simulation
from simulation import get_simulation_equity


def setup_logging():
    """Configure logging for the application."""
    # Define paths for log file and cache directory
    log_file_path = os.getenv("BOT_LOG_FILE_PATH", DEFAULT_LOG_FILE_PATH)
    cache_dir_path = os.getenv("BOT_CACHE_DIR_PATH", DEFAULT_CACHE_DIR_PATH)

    # Ensure the directories for log file and cache exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    os.makedirs(cache_dir_path, exist_ok=True)

    # Initialize logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),
            logging.StreamHandler(codecs.getwriter("utf-8")(sys.stdout.buffer)),
        ],
    )


def schedule_tasks():
    """Schedule regular tasks for the bot."""
    # Schedule to run at 15-minute intervals
    for hour in [0, 4, 8, 12, 16, 20]:
        schedule.every().day.at(f"{hour:02d}:00").do(main_trading_logic)
        logging.debug(f"Scheduled main trading logic to run at {hour:02d}:00")

    # Schedule symbol info refresh daily at midnight
    schedule.every().day.at("00:00").do(refresh_symbol_info)
    logging.info("Tasks scheduled successfully")


def main_trading_logic():
    """Main trading logic that processes symbols and executes orders."""
    start_time = time.time()
    logger = logging.getLogger(__name__)
    logger.info(f"============= Starting main_trading_logic =============")

    try:
        # Step 1: Get all symbols from database
        all_symbols = get_all_symbols()
        logger.info(f"Found {len(all_symbols)} symbols to evaluate")

        if not all_symbols:
            return

        # Step 2: Get open positions
        open_positions = get_open_positions_from_db()
        logger.info(f"Found {len(open_positions)} open positions")

        # Step 4: Prepare market data
        symbol_data = prepare_market_data(all_symbols, open_positions)

        # Step 5: Get current equity
        if TRADING_MODE == "simulation":
            current_equity = get_simulation_equity(symbol_data)
        else:
            current_equity = get_current_equity()

        # Step 6: Get all open positions from exchange
        all_exchange_positions = {}
        try:
            exchange_positions = get_open_positions()
            logger.info(
                f"Retrieved {len(exchange_positions)} open positions from exchange"
            )

            # Organize by symbol
            for pos in exchange_positions:
                symbol = pos["symbol"]
                if symbol not in all_exchange_positions:
                    all_exchange_positions[symbol] = []
                all_exchange_positions[symbol].append(pos)
        except Exception as e:
            logger.error(f"Error getting positions from exchange: {e}")
            logger.error(traceback.format_exc())

        # Step 7: Check stop loss for open positions
        stop_loss_positions = check_stop_loss_for_open_positions(
            open_positions, symbol_data, current_equity
        )

        # Step 8: Process stop loss positions
        close_orders = []
        close_position_data = []

        if stop_loss_positions:
            logger.warning(
                f"Found {len(stop_loss_positions)} positions that triggered stop loss"
            )
            for sl_position in stop_loss_positions:
                symbol = sl_position.get("symbol")

                # Get the correct quantity and side based on which symbol is the trade symbol
                if symbol:
                    quantity = float(sl_position.get("quantity", 0))
                    position_side = sl_position.get("positionSide")

                if quantity > 0 and position_side != "none":
                    # Create close order for stop loss - opposite of original position
                    if position_side == "LONG":
                        close_orders.append(
                            BatchOrder(
                                symbol=f"{symbol}-USDT",
                                order_type="MARKET",
                                side="SELL",
                                position_side="LONG",
                                quantity=quantity,
                            )
                        )
                    else:  # SHORT
                        close_orders.append(
                            BatchOrder(
                                symbol=f"{symbol}-USDT",
                                order_type="MARKET",
                                side="BUY",
                                position_side="SHORT",
                                quantity=quantity,
                            )
                        )
                    close_position_data.append(sl_position)

        # Step 10: Process all symbols for new positions
        open_orders = []
        open_position_data = []

        # Process symbols one by one
        for symbol_ in all_symbols:
            try:
                # Get positions for symbol
                symbol = symbol_.get("symbol")
                positions_for_symbol = {}

                # Find positions for symbol
                symbol_key = f"{symbol}-USDT"
                if symbol_key in all_exchange_positions:
                    positions_for_symbol[symbol] = all_exchange_positions[symbol_key]


                # Process the symbol
                result = process_symbol(
                    symbol_, symbol_data, current_equity, positions_for_symbol
                )

                # Handle the result
                if isinstance(result, tuple) and len(result) == 3:
                    close_orders_symbol, open_orders_symbol, position_data = result

                    # Handle close orders if any
                    if close_orders_symbol:
                        close_orders.extend(close_orders_symbol)
                        if position_data:  # Position data for closing
                            close_position_data.append(position_data)

                    # Handle open orders if any
                    elif open_orders_symbol and position_data:
                        open_orders.extend(open_orders_symbol)
                        open_position_data.append(position_data)
                        logger.info(f"Added open order for {symbol}")

            except Exception as e:
                logger.error(
                    f"Error processing symbol {symbol_.get('symbol')}: {e}"
                )
                logger.error(traceback.format_exc())

        # Step 11: Execute close orders first
        if close_orders and close_position_data:
            logger.info(f"Found {len(close_orders)} close orders (TP/SL) to execute")
            execute_batch_orders(close_orders, "CLOSE", close_position_data)

        # Step 12: Execute open orders if bot is active and has enough margin
        if open_orders and open_position_data:
            is_bot_active = CONFIG.get("is_active", True)
            has_enough_margin = check_margin_available()

            if is_bot_active and has_enough_margin:
                logger.info(f"Found {len(open_orders)} open orders to execute")
                execute_batch_orders(open_orders, "OPEN", open_position_data)
            else:
                reason = []
                if not is_bot_active:
                    reason.append("bot is INACTIVE")
                if not has_enough_margin:
                    reason.append("insufficient margin")
                reason_text = " and ".join(reason)
                logger.info(
                    f"Skipping {len(open_orders)} open orders due to {reason_text}. Only monitoring existing positions."
                )
                send_telegram_message(
                    f"Skipped {len(open_orders)} potential new positions due to {reason_text}."
                )

        # Log execution time
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Main trading logic completed in {execution_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Error in main_trading_logic: {e}")
        logger.error(traceback.format_exc())
        send_error_message(f"Trading cycle failed with error: {e}", e)


def run_main_loop(sleep_seconds):
    """Main loop to run scheduled tasks."""
    logger = logging.getLogger(__name__)

    while True:
        try:
            schedule.run_pending()
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            send_telegram_message("Bot stopped by user")
            break
        except Exception as e:
            error_msg = f"Unexpected error in main loop: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            send_error_message(error_msg, e)
            # Wait a bit longer after an error to avoid rapid error loops
            time.sleep(max(sleep_seconds, 30))


def run():
    """Main function to run the trading bot with scheduled tasks."""
    try:
        # Configure logging
        setup_logging()
        logger = logging.getLogger(__name__)

        # Print startup banner
        logger.info("========================================================")
        logger.info("Starting Alpha Trading Bot")
        logger.info("========================================================")

        # Initialize database with default values if needed
        logger.info("Checking database initialization...")
        initialize_database()

        # Load configuration
        config = load_config_from_db()
        logger.info(f"Configuration loaded successfully")

        # Schedule trading logic to run at specified intervals
        schedule_tasks()

        # Send startup notification
        start_msg = "Alpha Trading Bot has started successfully!"
        logger.info(start_msg)
        send_telegram_message(start_msg)

        logger.info("Bot initialized. Waiting for next scheduled run time...")

        # Main loop
        MAIN_LOOP_SLEEP_SECONDS = int(os.environ.get("MAIN_LOOP_SLEEP", "5"))
        logger.info(
            f"Main loop will check for scheduled tasks every {MAIN_LOOP_SLEEP_SECONDS} seconds."
        )

        run_main_loop(MAIN_LOOP_SLEEP_SECONDS)

    except Exception as e:
        # Handle any exceptions during startup
        error_msg = f"Critical error during bot startup: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        send_error_message(error_msg, e)


if __name__ == "__main__":
    run()
