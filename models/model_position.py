import os
import requests
import logging
import traceback
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from typing import List, Dict, Any, Tuple, Optional

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler("bot.log"),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)

# Kết nối tới MongoDB
MONGO_URI = os.environ.get("MONGO_HOST_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["solo_alpha01"]  # Tên database
collection = db["positions"]  # Tên collection


def save_position_to_db(data):
    if data is None:
        logger.error("Lỗi dữ liệu")
        return
    position = {
        "symbol": data["symbol"],
        "orderType": data["orderType"],
        "side": data["side"],
        "positionSide": data["positionSide"],
        "entryPrice": data["entryPrice"],
        "quantity": data["quantity"],
        "dca": data["dca"],
        "status": data["status"],
        "timestamp": datetime.now().timestamp(),
    }

    # Chèn document vào collection pairs
    collection.update_one(
        {"symbol": data["symbol"]},
        {"$set": position},
        upsert=True,
    )
    logger.info("Data has been saved to the database successfully.")


def get_position_dca(symbol):
    """Lấy position"""
    return collection.find_one({"symbol": symbol})


def get_open_positions_from_db():
    """Get all open positions from the database"""
    return list(collection.find({"status": "OPEN"}))


def save_order_to_db(order_data):
    """Save a new order to the database"""
    if order_data is None:
        logger.error("Error: No data provided")
        return None

    # Add timestamp if not present
    if "entry_time" not in order_data:
        order_data["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Insert the order into the collection
    try:
        result = collection.insert_one(order_data)
        return result.inserted_id
    except Exception as e:
        logger.error(f"Error saving order to database: {e}")
        logger.error(f"Order data: {order_data}")
        logger.error(traceback.format_exc())

        # Attempt to recover by checking if identical order already exists
        try:
            if "symbol" in order_data:
                existing = collection.find_one(
                    {
                        "symbol": order_data["symbol"],
                        "status": "OPEN",
                    }
                )

                if existing:
                    logger.info(
                        f"Found existing order for {order_data['symbol']}, returning its ID"
                    )
                    return existing["_id"]
        except Exception as e2:
            logger.error(f"Error in recovery attempt: {e2}")

        return None


def update_order_status(position_data: Dict[str, Any]) -> bool:
    """Update the status of an order in the database."""
    try:
        # Logs full position data for debugging
        logger.debug(f"Received position data for update: {position_data}")

        # Check for required fields
        required_fields = [
            "status",
            "exit_reason",
            "exit_spread",
            "exit_time",
            "closePriceA",
            "closePriceB",
            "pnl",
            "close_order_id",
        ]

        # Check for symbol
        if "symbol" not in position_data:
            logger.error("Missing symbol in position data")
            return False

        # Check for other required fields
        missing_fields = [
            field for field in required_fields if field not in position_data
        ]
        if missing_fields:
            logger.error(f"Missing required fields in position data: {missing_fields}")
            logger.error(f"Position data received: {position_data}")
            return False

        # Prepare the update query
        update_query = {
            "symbol": position_data["symbol"],
            "status": "OPEN",  # Ensure we are updating an open position
        }

        # Prepare the update data
        update_data = {
            "$set": {
                "status": position_data["status"],
                "exit_reason": position_data["exit_reason"],
                "exit_spread": position_data["exit_spread"],
                "exit_time": position_data["exit_time"],
                "closePrice": position_data["closePrice"],
                "pnl": position_data["pnl"],
                "close_order_id": position_data.get("close_order_id", []),
            }
        }

        # Execute the update
        result = collection.update_one(update_query, update_data)

        if result.matched_count > 0:
            logger.info(
                f"Successfully updated position {position_data['symbol']} as CLOSED"
            )
            return True

        # Kiểm tra lại một lần nữa xem vị thế đã được đóng chưa (có thể đã được đóng bởi một tiến trình khác)
        existing_closed = collection.find_one(
            {
                "symbol": position_data["symbol"],
                "status": "OPEN",
            }
        )

        if not existing_closed:
            logger.info(
                f"Position {position_data['symbol']} was closed by another process. No update needed."
            )
            return True

        logger.error(
            f"No matching OPEN position found for {position_data['symbol']}"
        )
        return False

    except Exception as e:
        logger.error(f"Error updating position: {e}")
        logger.error(traceback.format_exc())
        return False
