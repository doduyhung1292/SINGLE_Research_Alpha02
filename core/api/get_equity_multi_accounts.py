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


def send_request(method, path, urlpa, payload={}, api_key=None, secret_key=None):
    """Send request to BingX API with proper headers."""
    # Use provided credentials or fall back to default
    current_api_key = api_key if api_key else APIKEY
    current_secret_key = secret_key if secret_key else SECRETKEY
    
    if not current_api_key:
        logger.error("ERROR: API key is missing!")
        return {"code": -1, "msg": "API key is missing"}

    if not current_secret_key:
        logger.error("ERROR: Secret key is missing!")
        return {"code": -1, "msg": "Secret key is missing"}

    url = "%s%s?%s&signature=%s" % (APIURL, path, urlpa, get_sign(current_secret_key, urlpa))

    headers = {"X-BX-APIKEY": current_api_key}

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


def get_account_credentials():
    """Get all available BingX API credentials from environment variables."""
    credentials = []
    index = 1
    
    # First check if there are indexed credentials
    while True:
        api_key = os.environ.get(f"BINGX_API_KEY_{index}")
        secret_key = os.environ.get(f"BINGX_SECRET_KEY_{index}")
        
        if not api_key or not secret_key:
            break
            
        credentials.append({
            "api_key": api_key,
            "secret_key": secret_key
        })
        index += 1
    
    # If no indexed credentials found, use the default ones
    if not credentials and APIKEY and SECRETKEY:
        credentials.append({
            "api_key": APIKEY,
            "secret_key": SECRETKEY
        })
        
    return credentials


def get_account_balance(api_key=None, secret_key=None):
    """Get account balance."""
    path = "/openApi/swap/v3/user/balance"
    method = "GET"

    params = {"timestamp": int(time.time() * 1000)}

    paramsStr = parseParam(params)
    response = send_request(method, path, paramsStr, api_key=api_key, secret_key=secret_key)

    return response

def get_current_equity() -> float:
    """Get current equity from all Bingx exchange accounts."""
    try:
        # Get the most current config values
        current_config = CONFIG
        max_retry_attempts = current_config.get(
            "max_retry_attempts",
            DEFAULT_CONFIG["max_retry_attempts"],
        )
        retry_delay = current_config.get("retry_delay", DEFAULT_CONFIG["retry_delay"])
        initial_equity = current_config.get("initial_equity", 100.0)

        # Get all account credentials
        credentials = get_account_credentials()
        
        if not credentials:
            logger.error("No valid API credentials found")
            return initial_equity
            
        total_equity = 0.0
        accounts_checked = 0

        # Loop through each account
        for cred in credentials:
            api_key = cred["api_key"]
            secret_key = cred["secret_key"]
            
            retry_count = 0
            while retry_count < max_retry_attempts:
                try:
                    account_info = get_account_balance(api_key=api_key, secret_key=secret_key)

                    if account_info and "data" in account_info:
                        # Extract the equity value from the response
                        equity = float(account_info["data"][0]["balance"])
                        logger.info(f"Account {accounts_checked + 1} equity: {equity}")
                        total_equity += equity
                        accounts_checked += 1
                        break  # Success, exit retry loop
                    else:
                        logger.error(
                            f"Failed to get account balance from Bingx account {accounts_checked + 1}: {account_info}"
                        )
                        retry_count += 1
                        if retry_count < max_retry_attempts:
                            logger.info(
                                f"Retrying in {retry_delay} seconds... (attempt {retry_count}/{max_retry_attempts})"
                            )
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"Max retries reached for getting equity from account {accounts_checked + 1}")
                except Exception as e:
                    logger.error(f"Error getting equity from Bingx account {accounts_checked + 1}: {e}")
                    logger.error(traceback.format_exc())
                    retry_count += 1
                    if retry_count < max_retry_attempts:
                        logger.info(
                            f"Retrying in {retry_delay} seconds... (attempt {retry_count}/{max_retry_attempts})"
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"Max retries reached for getting equity from account {accounts_checked + 1}")
        
        if accounts_checked > 0:
            logger.info(f"Total equity from {accounts_checked} accounts: {total_equity}")
            return total_equity
        else:
            logger.error("Failed to get equity from any account")
            return initial_equity  # Default fallback
            
    except Exception as e:
        logger.error(f"Unexpected error getting equity: {e}")
        logger.error(traceback.format_exc())
        return CONFIG.get("initial_equity", 100.0)  # Default fallback


def check_margin_available() -> bool:
    """Kiểm tra xem margin khả dụng có đủ để mở vị thế mới không."""
    try:
        # Get all account credentials
        credentials = get_account_credentials()
        
        if not credentials:
            logger.error("No valid API credentials found")
            return False
            
        # Chỉ kiểm tra tài khoản đầu tiên cho margin
        api_key = credentials[0]["api_key"]
        secret_key = credentials[0]["secret_key"]
        
        account_info = get_account_balance(api_key=api_key, secret_key=secret_key)

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

