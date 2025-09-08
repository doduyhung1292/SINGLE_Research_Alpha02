import logging
import traceback
import time
import json
import os
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.config.constants import CONFIG, DEFAULT_CONFIG
from core.notification.notifier import send_telegram_message

logger = logging.getLogger(__name__)

# Load environment variables
APIKEY = os.environ.get("BINGX_API_KEY")
SECRETKEY = os.environ.get("BINGX_SECRET_KEY")

# Base URL
APIURL = "https://open-api.bingx.com"


def parseParam(paramsMap):
    """Convert parameters to sorted URL query string."""
    sortedKeys = sorted(paramsMap)
    params = []
    for key in sortedKeys:
        params.append(key + "=" + str(paramsMap[key]))

    paramsStr = "&".join(params)

    # Add timestamp at the end
    if paramsStr:
        return paramsStr + "&timestamp=" + str(int(time.time() * 1000))
    else:
        return "timestamp=" + str(int(time.time() * 1000))


def get_sign(secretKey, data):
    """Generate HMAC-SHA256 signature."""
    if not secretKey:
        logger.error("ERROR: Secret key is missing!")
        return ""
    signature = hmac.new(secretKey.encode(), data.encode(), hashlib.sha256).hexdigest()
    return signature


def send_request(method, path, urlpa, payload={}):
    """Send request to BingX API with proper headers."""
    if not APIKEY:
        logger.error("ERROR: API key is missing!")
        return {"code": -1, "msg": "API key is missing"}

    if not SECRETKEY:
        logger.error("ERROR: Secret key is missing!")
        return {"code": -1, "msg": "Secret key is missing"}

    url = "%s%s?%s&signature=%s" % (APIURL, path, urlpa, get_sign(SECRETKEY, urlpa))

    headers = {"X-BX-APIKEY": APIKEY}

    # Only add Content-Type for JSON payload
    if method == "POST" and payload and isinstance(payload, dict) and payload:
        headers["Content-Type"] = "application/json"
        payload = json.dumps(payload)

    try:
        response = requests.request(method, url, headers=headers, data=payload)

        logger.debug(f"Response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Response error: {response.text}")
            return {"code": -1, "msg": f"HTTP error: {response.status_code}"}

        return response.json()
    except Exception as e:
        logger.error(f"Request error: {e}")
        return {"code": -1, "msg": f"Request error: {e}"}


def get_symbols():
    """Get all available trading symbols."""
    payload = {}
    path = "/openApi/swap/v2/quote/contracts"
    method = "GET"
    paramsMap = {"timestamp": int(time.time() * 1000)}
    paramsStr = parseParam(paramsMap)
    response = send_request(method, path, paramsStr, payload)

    if "data" in response:
        datas = response["data"]
        for data in datas:
            data["symbol"] = data["symbol"].replace("-USDT", "").replace("-USD", "")
        return datas
    return []


def get_current_equity() -> float:
    """Get current equity from Bingx exchange."""
    try:
        # Get the most current config values
        current_config = CONFIG
        max_retry_attempts = current_config.get(
            "max_retry_attempts",
            DEFAULT_CONFIG["max_retry_attempts"],
        )
        retry_delay = current_config.get("retry_delay", DEFAULT_CONFIG["retry_delay"])
        initial_equity = current_config.get("initial_equity", 100.0)

        retry_count = 0

        while retry_count < max_retry_attempts:
            try:
                account_info = get_account_balance()

                if account_info and "data" in account_info:
                    # Extract the equity value from the response
                    equity = float(account_info["data"][0]["balance"])
                    logger.info(f"Current equity: {equity}")
                    return equity
                else:
                    logger.error(
                        f"Failed to get account balance from Bingx: {account_info}"
                    )
                    retry_count += 1
                    if retry_count < max_retry_attempts:
                        logger.info(
                            f"Retrying in {retry_delay} seconds... (attempt {retry_count}/{max_retry_attempts})"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error("Max retries reached for getting equity")
                        return initial_equity  # Default fallback
            except Exception as e:
                logger.error(f"Error getting equity from Bingx: {e}")
                logger.error(traceback.format_exc())
                retry_count += 1
                if retry_count < max_retry_attempts:
                    logger.info(
                        f"Retrying in {retry_delay} seconds... (attempt {retry_count}/{max_retry_attempts})"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retries reached for getting equity")
                    return initial_equity  # Default fallback
    except Exception as e:
        logger.error(f"Unexpected error getting equity: {e}")
        logger.error(traceback.format_exc())
        return CONFIG.get("initial_equity", 100.0)  # Default fallback


def check_margin_available() -> bool:
    """
    Kiểm tra xem margin khả dụng có đủ để mở vị thế mới không.
    Dừng mở vị thế mới khi availableMargin < 3% balance.

    Returns:
        bool: True nếu đủ margin, False nếu không
    """
    try:
        account_info = get_account_balance()

        if not account_info or "data" not in account_info or not account_info["data"]:
            logger.error("Failed to get account information")
            return False

        # Lấy thông tin tài khoản
        account_data = account_info["data"][0]
        balance = float(account_data.get("balance", 0))
        available_margin = float(account_data.get("availableMargin", 0))

        # Tính toán margin tối thiểu (3% của balance)
        min_margin_threshold = balance * 0.03

        if available_margin < min_margin_threshold:
            logger.warning(
                f"Available margin ({available_margin:.2f} USDT) is less than 3% of balance ({min_margin_threshold:.2f} USDT)"
            )
            send_telegram_message(
                f"Cảnh báo: Margin khả dụng ({available_margin:.2f} USDT) thấp hơn ngưỡng 3% ({min_margin_threshold:.2f} USDT). Tạm dừng mở vị thế mới.",
                is_error=True,
            )
            return False

        logger.info(
            f"Margin check passed: Available margin ({available_margin:.2f} USDT) >= {min_margin_threshold:.2f} USDT (3% of balance)"
        )
        return True

    except Exception as e:
        logger.error(f"Error checking margin: {e}")
        logger.error(traceback.format_exc())
        return False


def execute_batch_orders(
    orders: List[Any], action_type: str, position_data: List[Dict[str, Any]]
) -> bool:
    """
    Execute a batch of orders on the exchange.

    Args:
        orders: List of order objects to execute
        action_type: Type of action (OPEN or CLOSE)
        position_data: List of position data associated with the orders

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not orders:
            logger.info(f"No {action_type} orders to execute")
            return True

        # Execute orders through API
        logger.info(f"Executing {len(orders)} {action_type} orders")

        # Format orders for API
        formatted_orders = []
        for order in orders:
            formatted_order = {
                "symbol": order.symbol,
                "side": order.side,
                "positionSide": order.position_side,
                "type": order.order_type,
                "quantity": order.quantity,
            }
            formatted_orders.append(formatted_order)

        # Call API
        response = batch_orders(formatted_orders)

        # Process response
        if response and response.get("code") == 0:
            logger.info(f"Successfully executed {len(orders)} {action_type} orders")

            # Update position data with order IDs
            if "data" in response and response["data"]:
                for i, order_response in enumerate(response["data"]):
                    if i < len(position_data):
                        order_id = order_response.get("orderId")
                        if order_id:
                            if action_type == "OPEN":
                                position_data[i]["order_id"] = order_id
                            else:  # CLOSE
                                if "close_order_ids" not in position_data[i]:
                                    position_data[i]["close_order_ids"] = []
                                position_data[i]["close_order_ids"].append(order_id)

            # Save position data to database
            from models.model_position import save_order_to_db, update_order_status

            for position in position_data:
                if action_type == "OPEN":
                    save_order_to_db(position)
                else:  # CLOSE
                    update_order_status(position)

            return True
        else:
            logger.error(f"Failed to execute {action_type} orders: {response}")
            return False

    except Exception as e:
        logger.error(f"Error executing {action_type} orders: {e}")
        logger.error(traceback.format_exc())
        return False


def batch_orders(batch_orders_json):
    """Send batch orders to the exchange."""
    path = "/openApi/swap/v2/trade/batchOrders"
    method = "POST"

    # Format the batchOrders as a string representation of JSON array
    if isinstance(batch_orders_json, dict) and "batchOrders" in batch_orders_json:
        orders_str = batch_orders_json["batchOrders"]
    elif isinstance(batch_orders_json, list):
        orders_str = json.dumps(batch_orders_json)
    else:
        return {"code": -1, "msg": "Invalid batch orders format"}

    params = {"batchOrders": orders_str}

    paramsStr = parseParam(params)

    # Send request with empty payload (the data is in the URL)
    return send_request(method, path, paramsStr, {})


def get_open_positions():
    """Get all open positions."""
    path = "/openApi/swap/v2/user/positions"
    method = "GET"
    params = {"timestamp": int(time.time() * 1000)}

    paramsStr = parseParam(params)
    response = send_request(method, path, paramsStr)

    if response.get("code") == 0 and "data" in response:
        positions = response["data"]
        for position in positions:
            # Extract base symbol from symbol-USDT format
            position["symbol"] = position["symbol"].split("-")[0]
        return positions
    else:
        logger.error(f"API error: {response}")
        return []


def get_positions_for_symbol(symbol):
    """Get positions for a specific symbol."""
    path = "/openApi/swap/v2/user/positions"
    method = "GET"
    params = {"symbol": f"{symbol}-USDT", "timestamp": int(time.time() * 1000)}

    paramsStr = parseParam(params)
    response = send_request(method, path, paramsStr)

    if response.get("code") == 0 and "data" in response:
        return response["data"]
    else:
        logger.error(f"API error getting positions for {symbol}: {response}")
        return []


def get_current_mark_price(symbol):
    """Get current mark price of a symbol."""
    path = "/openApi/swap/v1/ticker/price"
    method = "GET"

    params = {"symbol": f"{symbol}-USDT", "timestamp": int(time.time() * 1000)}

    paramsStr = parseParam(params)
    response = send_request(method, path, paramsStr)

    if response.get("code") == 0:
        return response.get("data", [])
    else:
        logger.error(f"API error: {response}")
        return []


def get_account_balance():
    """Get account balance."""
    path = "/openApi/swap/v3/user/balance"
    method = "GET"

    params = {"timestamp": int(time.time() * 1000)}

    paramsStr = parseParam(params)
    response = send_request(method, path, paramsStr)

    return response


def get_symbol_contracts():
    """Get contract information for all symbols."""
    path = "/openApi/swap/v2/quote/contracts"
    method = "GET"

    params = {"timestamp": int(time.time() * 1000)}

    paramsStr = parseParam(params)
    response = send_request(method, path, paramsStr)

    return response


def calculate_long_short_ratio():
    """Calculate the ratio of long to short positions based on notional value.

    Returns:
        dict: A dictionary containing long_sum, short_sum, and ratio
    """
    positions = get_open_positions()

    if not positions:
        return {
            "long_sum": 0,
            "short_sum": 0,
            "ratio": 0,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    long_sum = 0
    short_sum = 0

    for position in positions:
        # Calculate notional value (availableAmt * avgPrice)
        notional_value = float(position.get("availableAmt", 0)) * float(
            position.get("avgPrice", 0)
        )

        # Add to appropriate sum based on position side
        if position.get("positionSide") == "LONG":
            long_sum += notional_value
        elif position.get("positionSide") == "SHORT":
            short_sum += notional_value

    # Calculate ratio (avoid division by zero)
    ratio = long_sum / short_sum if short_sum > 0 else 0

    return {
        "long_sum": long_sum,
        "short_sum": short_sum,
        "ratio": ratio,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
