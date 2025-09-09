import time
import hmac
import hashlib
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
import pymongo
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
import os
from pymongo import MongoClient
from bson import ObjectId
import logging
import sys
from dotenv import load_dotenv
import traceback
import statistics
import math

# Import config từ ứng dụng chính
sys.path.append("..")
load_dotenv()

from core.config.config_manager import set_bot_active_status

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ----------------- Cấu hình MongoDB -----------------
client_mongo = pymongo.MongoClient(os.getenv("MONGO_HOST_URI"))
db = client_mongo["solo_alpha01"]
collection = db["trade_symbols"]
deleted_symbols_collection = db["symbols_deleted"]
positions_collection = db["positions"]
config_collection = db["config"]
equity_collection = db["equity_history"]

# ----------------- Thông tin API BingX -----------------
# Load API keys directly from environment variables
BINGX_API_KEY = os.environ.get("BINGX_API_KEY", "")
BINGX_API_SECRET = os.environ.get("BINGX_SECRET_KEY", "")

# Log the environment variables for debugging (masked for security)
if BINGX_API_KEY:
    logging.info(
        f"BINGX_API_KEY found in env: {BINGX_API_KEY[:5]}...{BINGX_API_KEY[-5:] if len(BINGX_API_KEY) > 10 else ''}"
    )
else:
    logging.error("BINGX_API_KEY not found in environment variables!")

if BINGX_API_SECRET:
    logging.info(
        f"BINGX_API_SECRET found in env: {BINGX_API_SECRET[:5]}...{BINGX_API_SECRET[-5:] if len(BINGX_API_SECRET) > 10 else ''}"
    )
else:
    logging.error("BINGX_API_SECRET not found in environment variables!")


# ----------------- Định nghĩa lớp BingXClient -----------------
class BingXClient:
    """BingX Futures API Client dựa trên tài liệu API."""

    BASE_URL = "https://open-api.bingx.com"
    ENDPOINTS = {
        "account": "/openApi/swap/v3/user/balance",
        "position": "/openApi/swap/v2/user/positions",
        "order": "/openApi/swap/v2/trade/order",
        "batch_orders": "/openApi/swap/v2/trade/batchOrders",
        "close_position": "/openApi/swap/v1/trade/closePosition",
        "klines": "/openApi/swap/v3/quote/klines",
    }

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()
        self.last_request_time = 0

        # Validate API keys
        if not api_key:
            logging.error("API key is missing or empty!")
        if not api_secret:
            logging.error("API secret is missing or empty!")

    def _generate_signature(self, payload_str: str) -> str:
        """Generate HMAC-SHA256 signature for API request."""
        if not self.api_secret:
            logging.error("Cannot generate signature: API secret is missing or empty!")
            return ""

        try:
            signature = hmac.new(
                self.api_secret.encode("utf-8"),
                payload_str.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).hexdigest()

            # Log the signature creation (for debugging)
            logging.debug(
                f"Generated signature for payload: {payload_str[:50]}... Signature: {signature[:10]}..."
            )

            return signature
        except Exception as e:
            logging.error(f"Error generating signature: {str(e)}")
            return ""

    def parseParam(self, paramsMap: Dict) -> str:
        """Convert parameters to sorted URL query string and add timestamp."""
        try:
            # Ensure all values are strings
            for key in paramsMap:
                paramsMap[key] = str(paramsMap[key])

            sortedKeys = sorted(paramsMap)
            params = []
            for key in sortedKeys:
                params.append(f"{key}={paramsMap[key]}")

            paramsStr = "&".join(params)

            # Add timestamp at the end
            timestamp = str(int(time.time() * 1000))
            if paramsStr:
                return paramsStr + "&timestamp=" + timestamp
            else:
                return "timestamp=" + timestamp
        except Exception as e:
            logging.error(f"Error parsing parameters: {str(e)}")
            return "timestamp=" + str(int(time.time() * 1000))

    def _make_request(
        self, method: str, endpoint: str, params: Dict = None, payload: dict = None
    ) -> Optional[dict]:
        """Make request to BingX API with proper headers and signature."""
        if not self.api_key:
            logging.error("API key is missing!")
            return {"code": -1, "msg": "API key is missing"}

        params = params or {}
        payload = payload or {}

        # Generate params string and signature
        paramsStr = self.parseParam(params)
        signature = self._generate_signature(paramsStr)

        if not signature:
            logging.error("Empty signature generated. Cannot proceed with API request.")
            return {"code": -1, "msg": "Empty signature generated"}

        url = f"{self.BASE_URL}{endpoint}?{paramsStr}&signature={signature}"

        headers = {
            "X-BX-APIKEY": self.api_key,
        }

        # Add Content-Type header for POST requests with non-empty payload
        if method.upper() == "POST" and payload:
            headers["Content-Type"] = "application/json"
            if isinstance(payload, dict):
                payload = json.dumps(payload)

        # Log request details (for debugging)
        logging.debug(f"Making {method} request to: {url}")
        logging.debug(f"Headers: {headers}")

        # Rate limiting
        RETRY_DELAY = 0.5
        if time.time() - self.last_request_time < RETRY_DELAY:
            time.sleep(RETRY_DELAY)

        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers)
            else:
                response = self.session.post(url, headers=headers, data=payload)

            self.last_request_time = time.time()

            # Check for HTTP errors
            if response.status_code != 200:
                logging.error(
                    f"BingX API HTTP error: {response.status_code} - {response.text}"
                )
                return {"code": -1, "msg": f"HTTP error: {response.status_code}"}

            # Log the response for debugging
            logging.debug(f"API Response: {response.text[:200]}...")

            return response.json()
        except Exception as e:
            logging.error(f"BingX API request failed: {str(e)}")
            return {"code": -1, "msg": f"Request error: {str(e)}"}

    def get_account_info(self) -> dict:
        """Get account balance and information."""
        path = self.ENDPOINTS["account"]
        paramsMap = {}
        result = self._make_request("GET", path, paramsMap)
        return result

    def get_positions(self) -> List[dict]:
        """Get all open positions."""
        path = self.ENDPOINTS["position"]
        paramsMap = {}
        result = self._make_request("GET", path, paramsMap)

        if result and result.get("code") == 0:
            positions = result.get("data", [])
            # Process position data to extract base symbol from symbol-USDT format
            for position in positions:
                if "symbol" in position and "-" in position["symbol"]:
                    position["symbol"] = position["symbol"].split("-")[0]
            return positions

        logging.error(f"Error in get_positions: {result}")
        return []

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Lấy giá đóng (close) của nến mới nhất cho symbol từ endpoint klines.
        """
        # Ensure symbol has proper format: BASE-USDT
        formatted_symbol = symbol
        if not symbol.endswith("-USDT") and not symbol.endswith("/USDT"):
            if "/" in symbol:
                base = symbol.split("/")[0]
                formatted_symbol = f"{base}-USDT"
            else:
                formatted_symbol = f"{symbol}-USDT"

        # Remove any "/" and ensure format is symbol-USDT
        formatted_symbol = formatted_symbol.replace("/", "-")

        params = {"symbol": formatted_symbol, "interval": "1m", "limit": 1}

        result = self._make_request("GET", self.ENDPOINTS["klines"], params)

        if result and result.get("code") == 0:
            data = result.get("data", [])
            if isinstance(data, list) and len(data) > 0:
                try:
                    price = float(data[0]["close"])
                    return price
                except Exception as e:
                    logging.error(f"Error parsing latest price for {symbol}: {str(e)}")
        return None


# Function to verify BingX API credentials
def verify_bingx_credentials():
    """Verify that BingX API credentials are valid by making a simple request."""
    if not BINGX_API_KEY or not BINGX_API_SECRET:
        logging.error(
            "BingX API credentials missing. Please check environment variables."
        )
        return False

    try:
        client = BingXClient(BINGX_API_KEY, BINGX_API_SECRET)
        result = client.get_account_info()

        if result and result.get("code") == 0:
            logging.info("BingX API credentials verified successfully.")
            return True
        else:
            logging.error(f"BingX API credentials verification failed: {result}")
            return False
    except Exception as e:
        logging.error(f"Error verifying BingX API credentials: {str(e)}")
        return False


# ----------------- Hàm crawl và lưu dữ liệu -----------------
def crawl_and_store_data():
    try:
        bingx = BingXClient(BINGX_API_KEY, BINGX_API_SECRET)

        # Lấy thông tin account
        account_info = bingx.get_account_info()
        if account_info and account_info.get("code") == 0:
            equity = float(account_info["data"][0].get("equity", 0))
        else:
            logging.error(f"Error getting account info: {account_info}")
            equity = None

        # Lấy thông tin position
        positions = bingx.get_positions()
        long_usdt = 0.0
        short_usdt = 0.0
        for pos in positions:
            positionSide = pos.get("positionSide")
            qty = float(pos.get("availableAmt", 0))
            symbol = pos.get("symbol")  # Ví dụ: "BTC/USDT" hoặc "BTC-USDT"

            # Lấy giá mới nhất của symbol
            latest_price = bingx.get_latest_price(symbol)
            if latest_price is None:
                logging.warning(f"Không lấy được giá cho symbol: {symbol}")
                continue

            usdt_value = qty * latest_price
            if positionSide == "LONG":
                long_usdt += usdt_value
            elif positionSide == "SHORT":
                short_usdt += usdt_value

        # Tính long/short ratio
        long_short_ratio = (long_usdt / short_usdt) if short_usdt > 0 else None

        # Lấy giá BTC
        btc_price = bingx.get_latest_price("BTC")

        # Lưu kết quả vào MongoDB - sử dụng equity_collection thay vì collection
        record = {
            "timestamp": datetime.now(timezone.utc),
            "equity": equity,
            "long_usdt": long_usdt,
            "short_usdt": short_usdt,
            "long_short_ratio": long_short_ratio,
            "btc_price": btc_price,
        }
        equity_collection.insert_one(record)
        logging.info(f"Data crawled and stored at {record['timestamp']}")
    except Exception as e:
        logging.error(f"Error in crawl_and_store_data: {e}")


# Hàm crawl và lưu dữ liệu equity cho 2 account
def crawl_equity_data_multi_account():
    try:
        # Lấy API key/secret cho 2 account
        api_key_1 = os.environ.get("BINGX_API_KEY_1", "")
        api_secret_1 = os.environ.get("BINGX_SECRET_KEY_1", "")
        api_key_2 = os.environ.get("BINGX_API_KEY_2", "")
        api_secret_2 = os.environ.get("BINGX_SECRET_KEY_2", "")

        # Lấy dữ liệu từ account 1
        bingx1 = BingXClient(api_key_1, api_secret_1)
        account_info_1 = bingx1.get_account_info()
        if account_info_1 and account_info_1.get("code") == 0:
            equity_1 = float(account_info_1["data"][0].get("equity", 0))
        else:
            logging.error(
                f"Không thể lấy thông tin account 1 từ BingX API: {account_info_1}"
            )
            equity_1 = None
        btc_price_1 = bingx1.get_latest_price("BTC")

        # Lấy dữ liệu từ account 2
        bingx2 = BingXClient(api_key_2, api_secret_2)
        account_info_2 = bingx2.get_account_info()
        if account_info_2 and account_info_2.get("code") == 0:
            equity_2 = float(account_info_2["data"][0].get("equity", 0))
        else:
            logging.error(
                f"Không thể lấy thông tin account 2 từ BingX API: {account_info_2}"
            )
            equity_2 = None
        btc_price_2 = bingx2.get_latest_price("BTC")

        # Tính tổng equity
        total_equity = None
        if equity_1 is not None and equity_2 is not None:
            total_equity = equity_1 + equity_2

        # Lưu vào MongoDB
        db = get_db()
        if "equity_history" not in db.list_collection_names():
            db.create_collection("equity_history")
        equity_collection = db["equity_history"]
        record = {
            "timestamp": datetime.now(timezone.utc),
            "equity_1": equity_1,
            "equity_2": equity_2,
            "total_equity": total_equity,
            "btc_price_1": btc_price_1,
            "btc_price_2": btc_price_2,
        }
        equity_collection.insert_one(record)
        logging.info(
            f"Đã lưu dữ liệu equity accounts: equity_1={equity_1}, equity_2={equity_2}, total={total_equity} tại {record['timestamp']}"
        )
    except Exception as e:
        logging.error(f"Lỗi khi crawl equity data multi account: {str(e)}")

# ----------------- Tạo ứng dụng Flask -----------------
app = Flask(__name__, static_folder="static", static_url_path="/static")
import json

app.jinja_env.filters["fromjson"] = lambda s: json.loads(s)


# Kết nối đến MongoDB
def get_db():
    client = MongoClient(os.getenv("MONGO_HOST_URI"))
    return client["solo_alpha01"]


def get_config_collection():
    """Lấy collection cấu hình chung từ MongoDB"""
    db = get_db()
    return db["config"]


def get_symbols_collection():
    """Lấy collection thông tin từ MongoDB"""
    db = get_db()
    return db["trade_symbols"]


def get_positions_collection():
    """Lấy collection thông tin positions từ MongoDB"""
    db = get_db()
    return db["positions"]


# Hàm hỗ trợ chuyển ObjectId thành chuỗi để serialize JSON
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return json.JSONEncoder.default(self, o)


def calculate_symbol_performance():
    # Lấy tất cả các positions đã đóng từ MongoDB
    closed_positions = list(positions_collection.find({"status": "CLOSED"}))

    # Tạo DataFrame từ các positions
    df = pd.DataFrame(closed_positions)
    if df.empty:
        return [], {}

    # Kiểm tra các thông tin cần thiết có tồn tại không
    required_cols = [
        "symbol",
        "entryPrice",
        "closePrice",
        "timestamp",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        logging.warning(f"Missing columns in positions data: {missing_cols}")
        return [], {}

    # Sắp xếp theo thời gian
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("timestamp")

    # Tính toán equity curves cho từng symbol
    symbol_equity_curves = {}
    for i, symbol_row in df.drop_duplicates(subset=["symbol"]).iterrows():
        symbol_name = f"{symbol_row['symbol']}"
        symbol_trades = df[
            (df["symbol"] ==symbol_row["symbol"])
        ].copy()

        # Tính lợi nhuận cho mỗi giao dịch
        symbol_trades["profit"] = symbol_trades.apply(
            lambda row: calculate_profit(row), axis=1
        )

        # Tạo time series cho equity curve
        dates = pd.date_range(
            start=symbol_trades["timestamp"].min(),
            end=symbol_trades["timestamp"].max(),
            freq="D",
        )
        equity_curve = pd.Series(index=dates, data=0.0)
        cumulative_return = 0.0

        for _, trade in symbol_trades.iterrows():
            cumulative_return += trade["profit"]
            equity_curve[trade["timestamp"].date() :] = (
                cumulative_return * 100
            )  # Convert to percentage

        # Fill forward và backward để không có gaps
        equity_curve = equity_curve.fillna(method="ffill").fillna(0)

        # Convert to list of [timestamp, value] for JSON serialization
        symbol_equity_curves[symbol_name] = [
            [int(date.timestamp() * 1000), float(value)]
            for date, value in equity_curve.items()
        ]

    # Tính toán lợi nhuận cho mỗi giao dịch symbol
    df["profit"] = df.apply(lambda row: calculate_profit(row), axis=1)

    # Nhóm theo symbol và tính các chỉ số
    df["symbol_name"] = df["symbol"]
    symbol_stats = (
        df.groupby("symbol_name")
        .agg({"profit": ["count", "mean", "sum", "std"], "_id": "count"})
        .reset_index()
    )

    # Đổi tên cột và tính thêm các chỉ số
    symbol_stats.columns = [
        "symbol",
        "total_trades",
        "avg_return",
        "total_return",
        "return_std",
        "trade_count",
    ]
    symbol_stats["win_rate"] = (
        df.groupby("symbol_name")["profit"].apply(lambda x: (x > 0).mean()).values
    )

    # Tính Sharpe Ratio (giả sử risk-free rate = 0)
    symbol_stats["sharpe_ratio"] = np.where(
        symbol_stats["return_std"] != 0,
        symbol_stats["avg_return"] / symbol_stats["return_std"],
        0,
    )

    # Chuyển về dạng list of dict và format số
    result = symbol_stats.round(4).to_dict("records")
    return result, symbol_equity_curves


def calculate_profit(position):
    """Tính lợi nhuận từ một position"""
    try:
        # Kiểm tra xem tất cả các trường cần thiết có tồn tại không
        required_fields = [
            "entryPrice",
            "closePrice",
            "quantity",
            "side"
        ]

        # Nếu thiếu trường, trả về 0
        if not all(field in position for field in required_fields):
            return 0

        # Nếu đã có trường pnl được tính sẵn, dùng luôn
        if "pnl" in position and position["pnl"] is not None:
            # Ensure it's a float
            try:
                return float(position["pnl"])
            except (ValueError, TypeError):
                pass  # If conversion fails, continue with manual calculation

        # Lấy thông tin từ position
        try:
            entry_price = float(position["entryPrice"])
            close_price = float(position["closePrice"])
            quantity = float(position["quantity"])
            side = position["side"]

            # Nếu side là 'none' thì không tính PnL cho symbol đó
            pnl = 0
            if side == "BUY":
                pnl = (close_price - entry_price) * quantity
            elif side == "SELL":
                pnl = (entry_price - close_price) * quantity
            # Nếu side_a là 'none' thì pnl = 0


            # Tổng PnL
            total_pnl = pnl

            # Phí giao dịch 0.025% mỗi giao dịch (entry và exit)
            fee_rate = 0.00025  # 0.025%
            fee = (
                (entry_price * quantity + close_price * quantity) * fee_rate
                if side != "none"
                else 0
            )


            # Thêm chi phí slippage 0.02% cho mỗi giao dịch (entry và exit)
            slippage_rate = 0.0002  # 0.02%
            slippage = (
                (entry_price * quantity + close_price * quantity)
                * slippage_rate
                if side != "none"
                else 0
            )


            # PnL ròng sau khi trừ phí và slippage
            net_pnl = total_pnl - fee - slippage

            # Ensure we don't return NaN or infinite values
            if pd.isna(net_pnl) or np.isinf(net_pnl):
                return 0

            return net_pnl
        except (ValueError, TypeError) as e:
            logging.error(f"Error converting values in position: {str(e)}")
            return 0

    except Exception as e:
        logging.error(f"Error calculating profit: {str(e)}")
        return 0


@app.route("/")
def index():
    # Lấy dữ liệu symbols
    symbols_collection = get_symbols_collection()
    symbols_data = list(symbols_collection.find().sort("timestamp", -1))
    for d in symbols_data:
        d["_id"] = str(d["_id"])

    # Lấy dữ liệu hiệu suất symbol và equity curves
    symbol_performance, symbol_equity_curves = calculate_symbol_performance()

    # Lấy dữ liệu tương quan BTC-Equity
    equity_records = list(equity_collection.find().sort("timestamp", 1))
    correlation = calculate_btc_equity_correlation(equity_records)

    return render_template(
        "index.html",
        data=json.dumps(symbols_data, cls=JSONEncoder),
        symbol_performance=symbol_performance,
        symbol_equity_curves=symbol_equity_curves,
        btc_equity_correlation=correlation,
    )


# API lấy danh sách cấu hình
@app.route("/api/configs", methods=["GET"])
def get_configs():
    config_type = request.args.get("type", "all")

    if config_type == "general":
        # Lấy cấu hình chung
        configs_coll = get_config_collection()
        configs = list(configs_coll.find())
    else:
        # Lấy thông tin symbol
        symbols_coll = get_symbols_collection()
        configs = list(symbols_coll.find())

    return jsonify(json.loads(JSONEncoder().encode(configs)))


# API lấy thông tin cấu hình chi tiết
@app.route("/api/config/<config_id>", methods=["GET"])
def get_config(config_id):
    config_type = request.args.get("type", "symbol")

    if config_type == "general":
        collection = get_config_collection()
    else:
        collection = get_symbols_collection()

    config = collection.find_one({"_id": ObjectId(config_id)})
    if config:
        return jsonify(json.loads(JSONEncoder().encode(config)))
    return jsonify({"error": "Config not found"}), 404


# API cập nhật cấu hình
@app.route("/api/config/<config_id>", methods=["PUT"])
def update_config(config_id):
    data = request.json
    config_type = request.args.get("type", "symbol")

    if config_type == "general":
        collection = get_config_collection()
    else:
        collection = get_symbols_collection()

    # Kiểm tra dữ liệu đầu vào
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Xóa _id nếu có vì MongoDB không cho phép cập nhật _id
    if "_id" in data:
        del data["_id"]

    # Thêm thời gian cập nhật
    data["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Cập nhật cấu hình
    result = collection.update_one({"_id": ObjectId(config_id)}, {"$set": data})

    if result.modified_count > 0:
        return jsonify({"message": "Config updated successfully"})
    return jsonify({"error": "Failed to update config"}), 500


# API tạo cấu hình mới
@app.route("/api/config", methods=["POST"])
def create_config():
    data = request.json
    config_type = request.args.get("type", "symbol")

    if config_type == "general":
        collection = get_config_collection()
    else:
        collection = get_symbols_collection()

    # Kiểm tra dữ liệu đầu vào
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Thêm thời gian
    data["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Kiểm tra và cập nhật trạng thái mặc định nếu chưa có
    if "is_active" not in data:
        data["is_active"] = True

    # Tạo cấu hình mới
    result = collection.insert_one(data)

    if result.inserted_id:
        return jsonify(
            {"message": "Config created successfully", "id": str(result.inserted_id)}
        )
    return jsonify({"error": "Failed to create config"}), 500


# API xóa cấu hình
@app.route("/api/config/<config_id>", methods=["DELETE"])
def delete_config(config_id):
    config_type = request.args.get("type", "symbol")

    if config_type == "general":
        collection = get_config_collection()
    else:
        collection = get_symbols_collection()

    result = collection.delete_one({"_id": ObjectId(config_id)})

    if result.deleted_count > 0:
        return jsonify({"message": "Config deleted successfully"})
    return jsonify({"error": "Failed to delete config"}), 500


# API lấy số dư tài khoản
@app.route("/api/balance", methods=["GET"])
def get_balance():
    # Lấy dữ liệu equity gần nhất
    equity_data = equity_collection.find_one({}, sort=[("timestamp", -1)])

    if equity_data:
        return jsonify(
            {
                "balance": equity_data.get("equity", 0),
                "updated_at": equity_data.get("timestamp"),
            }
        )
    return jsonify({"error": "Balance information not found"})




# API lấy danh sách lệnh đang mở
@app.route("/api/open_positions", methods=["GET"])
def get_open_positions():
    positions_coll = get_positions_collection()

    # Get all OPEN positions
    open_positions = list(positions_coll.find({"status": "OPEN"}))

    # Sort positions by entry_time (oldest to newest) if the field exists
    if open_positions and "entry_time" in open_positions[0]:
        try:
            # Sort by entry_time
            open_positions.sort(
                key=lambda x: pd.to_datetime(x.get("entry_time", "1970-01-01 00:00:00"))
            )
        except Exception as e:
            logging.error(f"Error sorting open positions by entry_time: {str(e)}")

    return jsonify(json.loads(JSONEncoder().encode(open_positions)))


# API lấy lịch sử giao dịch
@app.route("/api/trade_history", methods=["GET"])
def get_trade_history():
    positions_coll = get_positions_collection()

    # Lấy các tham số lọc từ query string
    symbol = request.args.get("symbol")
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    # Xây dựng điều kiện truy vấn - tìm kiếm cả CLOSED và closed
    query = {"$or": [{"status": "CLOSED"}, {"status": "closed"}]}
    if symbol:
        query["symbol"] = symbol

    # Điều kiện thời gian
    date_query = {}
    if from_date:
        try:
            # Timestamp in seconds
            from_timestamp = datetime.fromisoformat(from_date).timestamp()
            date_query["$gte"] = from_timestamp
        except:
            pass

    if to_date:
        try:
            to_timestamp = datetime.fromisoformat(to_date).timestamp()
            date_query["$lte"] = to_timestamp
        except:
            pass

    if date_query:
        query["timestamp"] = date_query

    # Thực hiện truy vấn
    trade_history = list(positions_coll.find(query).sort("timestamp", -1).limit(100))

    # Xử lý dữ liệu trước khi trả về
    for trade in trade_history:
        # Tính profit nếu chưa có
        if "profit" not in trade or trade["profit"] is None:
            trade["profit"] = calculate_profit(trade)

        # Calculate notional value (assuming quantity fields exist)
        # IMPORTANT: Verify the actual field names for quantity/size in your database
        quantity = float(trade.get("quantity", 0))  # Replace 'quantityA' if needed
        entry_price = float(trade.get("entryPrice", 0))


        # Simple notional calculation: sum of (price * quantity) for both legs
        # Adjust this calculation based on your specific definition of notional value
        trade["notional"] = (entry_price * quantity)

        # Đổi _id thành string
        if "_id" in trade:
            trade["_id"] = str(trade["_id"])

    return jsonify(json.loads(JSONEncoder().encode(trade_history)))


# API kích hoạt/vô hiệu hóa trading cho một symbol
@app.route("/api/toggle_active/<config_id>", methods=["POST"])
def toggle_active(config_id):
    collection = get_symbols_collection()

    config = collection.find_one({"_id": ObjectId(config_id)})

    if not config:
        return jsonify({"error": "Config not found"}), 404

    # Đảo ngược trạng thái is_active
    new_status = not config.get("is_active", True)

    result = collection.update_one(
        {"_id": ObjectId(config_id)},
        {
            "$set": {
                "is_active": new_status,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            }
        },
    )

    if result.modified_count > 0:
        return jsonify(
            {
                "message": f"Trading {'activated' if new_status else 'deactivated'} successfully",
                "is_active": new_status,
            }
        )
    return jsonify({"error": "Failed to update status"}), 500


# API lấy danh sách active symbols
@app.route("/api/active_symbols", methods=["GET"])
def get_active_symbols():
    # Lấy danh sách symbols active
    symbols_coll = get_symbols_collection()
    active_symbols = list(symbols_coll.find({"is_active": True}))

    return jsonify(json.loads(JSONEncoder().encode(active_symbols)))


# API lấy dữ liệu biểu đồ symbol performance
@app.route("/api/symbol_performance_chart", methods=["GET"])
def get_symbol_performance_chart():
    try:
        positions_coll = get_positions_collection()

        # Lấy tất cả các giao dịch đã đóng
        closed_positions = list(positions_coll.find({"status": "close"}))

        # Trả về dữ liệu trống nếu không có giao dịch
        if not closed_positions:
            return jsonify({"symbol_performance": []})

        # Chuyển đổi sang DataFrame để dễ phân tích
        df = pd.DataFrame(closed_positions)

        # Tạo column symbol_name
        df["symbol_name"] = df.apply(
            lambda row: f"{row.get('symbol', '')}", axis=1
        )

        # Đảm bảo có cột thời gian để sắp xếp
        if "timestamp" in df.columns:
            df["time"] = pd.to_datetime(df["timestamp"], unit="s")
        else:
            # Fallback nếu không có timestamp
            return jsonify({"symbol_performance": []})

        # Sắp xếp theo thời gian
        df = df.sort_values("time")

        # Tính lợi nhuận cho mỗi giao dịch
        df["profit"] = df.apply(lambda row: calculate_profit(row), axis=1)

        # Tính equity curve cho từng symbol theo số thứ tự giao dịch
        symbol_equity_curves = []

        for symbol in df["symbol_name"].unique():
            # Lọc giao dịch cho symbol này
            symbol_df = df[df["symbol_name"] == symbol].copy()

            # Tính lợi nhuận tích lũy
            symbol_df["cumulative_profit"] = symbol_df["profit"].cumsum()

            # Tạo dữ liệu theo số thứ tự giao dịch
            data_points = []
            for i, (_, row) in enumerate(symbol_df.iterrows(), 1):
                data_points.append([i, float(row["cumulative_profit"])])

            # Add initial point if needed
            if not data_points or data_points[0][0] != 0:
                data_points.insert(0, [0, 0.0])

            # Create series for the chart
            symbol_equity_curves.append({"name": symbol, "data": data_points})

        return jsonify({"symbol_performance": symbol_equity_curves})

    except Exception as e:
        app.logger.error(f"Error in get_symbol_performance_chart: {str(e)}")
        return jsonify({"error": str(e), "symbol_performance": []}), 200


# API lấy dữ liệu biểu đồ equity
@app.route("/api/equity_chart", methods=["GET"])
def get_equity_chart():
    try:
        # Lấy tất cả bản ghi, sắp xếp theo timestamp
        equity_records = list(equity_collection.find().sort("timestamp", 1))
        app.logger.info(f"Tìm thấy {len(equity_records)} bản ghi equity")

        # Xử lý dữ liệu
        equity_data_1 = []
        equity_data_2 = []
        total_equity_data = []
        btc_data_1 = []

        for record in equity_records:
            # MongoDB timestamp là datetime object
            if isinstance(record["timestamp"], datetime):
                timestamp = int(
                    record["timestamp"].timestamp() * 1000
                )  # Convert to milliseconds
            else:
                timestamp = int(record["timestamp"] * 1000)

            # Thêm giá trị equity vào dữ liệu
            if (
                "equity_1" in record
                and record["equity_1"] is not None
            ):
                equity_data_1.append([timestamp, record.get("equity_1", 0)])
            
            if (
                "equity_2" in record
                and record["equity_2"] is not None
            ):
                equity_data_2.append([timestamp, record.get("equity_2", 0)])
            
            if (
                "total_equity" in record
                and record["total_equity"] is not None
            ):
                total_equity_data.append([timestamp, record.get("total_equity", 0)])
            # Nếu không có total_equity nhưng có cả equity_1 và equity_2, tính tổng
            elif (
                "equity_1" in record
                and record["equity_1"] is not None
                and "equity_2" in record
                and record["equity_2"] is not None
            ):
                total = record.get("equity_1", 0) + record.get("equity_2", 0)
                total_equity_data.append([timestamp, total])

            # BTC price nếu muốn vẽ thêm
            if "btc_price_1" in record and record["btc_price_1"] is not None:
                btc_data_1.append([timestamp, record["btc_price_1"]])

        app.logger.info(
            f"Đã xử lý {len(equity_data_1)} điểm dữ liệu equity_1, {len(equity_data_2)} điểm dữ liệu equity_2, {len(total_equity_data)} điểm dữ liệu total_equity"
        )

        # Tính toán tương quan giữa giá BTC và equity (có thể dùng btc_price_1 hoặc btc_price_2)
        correlation = calculate_btc_equity_correlation(equity_records)

        # Trả về dữ liệu
        return jsonify(
            {
                "equity_data": [
                    {"name": "Alpha01", "data": equity_data_1},
                    {"name": "Alpha02", "data": equity_data_2},
                    {"name": "Total Equity", "data": total_equity_data},
                ],
                "correlation": correlation,
            }
        )

    except Exception as e:
        app.logger.error(f"Lỗi khi truy vấn dữ liệu equity: {str(e)}")
        return jsonify(
            {
                "error": str(e),
                "equity_data": [
                    {"name": "Alpha01", "data": []},
                    {"name": "Alpha02", "data": []},
                    {"name": "Total Equity", "data": []},
                ],
                "correlation": None,
            }
        )


# API lấy tỷ lệ long/short
@app.route("/api/long_short_ratio", methods=["GET"])
def get_long_short_ratio():
    try:
        from core.api.exchange_api import calculate_long_short_ratio

        long_short_ratio = calculate_long_short_ratio()
        return jsonify(long_short_ratio)
    except Exception as e:
        app.logger.error(f"Lỗi khi lấy tỷ lệ long/short: {str(e)}")
        return jsonify(
            {
                "error": str(e),
                "ratio": 0,
                "long_sum": 0,
                "short_sum": 0,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )


# Thêm API test đơn giản
@app.route("/api/test", methods=["GET"])
def test_api():
    return jsonify(
        {
            "status": "ok",
            "message": "API test successful",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "random_data": [np.random.randint(1, 100) for _ in range(5)],
        }
    )


# API lấy cấu hình bot
@app.route("/api/bot_config", methods=["GET"])
def get_bot_config():
    """
    Get the bot configuration settings from the database.
    """
    try:
        # Lấy cấu hình bot - không chỉ định _id cụ thể, chỉ cần lấy bản ghi đầu tiên
        logging.info("Fetching bot config from database")
        bot_config = config_collection.find_one()

        # Log thông tin về kết quả tìm kiếm
        if bot_config:
            logging.info(f"Found bot config with ID: {bot_config.get('_id')}")

            # Clone config to avoid modifying the original
            config_to_return = bot_config.copy()

            # Convert the _id field to string for serialization
            if "_id" in config_to_return:
                config_to_return["_id"] = str(config_to_return["_id"])

            return jsonify(config_to_return)
        else:
            logging.info("No bot config found, returning default values")

            # Import DEFAULT_CONFIG values from bot_main
            sys.path.append("..")
            from core.config.config_manager import DEFAULT_CONFIG

            return jsonify(DEFAULT_CONFIG)

    except Exception as e:
        logging.error(f"Error getting bot config: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/bot_config", methods=["POST"])
def update_bot_config():
    """
    Update the bot configuration settings in the database.
    """
    try:
        # Get request data
        config_data = request.json

        if not config_data:
            return jsonify({"error": "No data provided"}), 400

        # Validate key configuration fields
        required_fields = [
            "window",
            "bb_entry_multiplier",
            "bb_stop_multiplier",
            "timeframe",
            "limit",
            "position_size_pct",
            "max_concurrent_positions",
        ]

        for field in required_fields:
            if field not in config_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Xóa _id khỏi data nếu người dùng vô tình gửi nó
        if "_id" in config_data:
            logging.info(f"Removing _id from request data: {config_data['_id']}")
            del config_data["_id"]

        # Thêm timestamp
        config_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Đảm bảo các trường mới được xử lý đúng kiểu dữ liệu
        if "risk_parity_lookback" in config_data:
            config_data["risk_parity_lookback"] = int(
                config_data["risk_parity_lookback"]
            )

        if "risk_parity_adjustment" in config_data:
            config_data["risk_parity_adjustment"] = float(
                config_data["risk_parity_adjustment"]
            )

        if "max_loss_pct" in config_data:
            config_data["max_loss_pct"] = float(config_data["max_loss_pct"])

        if "blacklist_days" in config_data:
            config_data["blacklist_days"] = int(config_data["blacklist_days"])

        # Kiểm tra xem đã có config chưa
        existing_config = config_collection.find_one()

        if existing_config:
            # Cập nhật bản ghi hiện có, giữ nguyên _id
            logging.info(
                f"Updating existing bot config with ID: {existing_config['_id']}"
            )

            result = config_collection.update_one(
                {"_id": existing_config["_id"]}, {"$set": config_data}
            )

            if result.modified_count > 0:
                logging.info(f"Bot config updated successfully")
                # If bot is running, reload the config
                try:
                    sys.path.append("..")
                    from core.config.config_manager import load_config_from_db

                    # Reload config in memory if possible
                    load_config_from_db()
                    logging.info("Bot configuration reloaded in memory")
                except Exception as reload_error:
                    logging.warning(
                        f"Could not reload bot config in memory: {str(reload_error)}"
                    )

                return jsonify(
                    {
                        "success": True,
                        "message": "Bot configuration updated successfully",
                    }
                )
            else:
                return (
                    jsonify(
                        {
                            "success": True,
                            "message": "No changes made to bot configuration",
                        }
                    ),
                    200,
                )
        else:
            # Không có bản ghi nào, tạo mới
            logging.info("No existing config found, creating new bot config")
            result = config_collection.insert_one(config_data)

            if result.inserted_id:
                logging.info(f"Bot config created with ID: {result.inserted_id}")

                # If bot is running, reload the config
                try:
                    sys.path.append("..")
                    from core.config.config_manager import load_config_from_db

                    # Reload config in memory if possible
                    load_config_from_db()
                    logging.info("Bot configuration reloaded in memory")
                except Exception as reload_error:
                    logging.warning(
                        f"Could not reload bot config in memory: {str(reload_error)}"
                    )

                return jsonify(
                    {
                        "success": True,
                        "message": "Bot configuration created successfully",
                    }
                )
            else:
                return jsonify({"error": "Failed to create bot configuration"}), 500
    except Exception as e:
        logging.error(f"Error updating bot config: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/symbol_performance", methods=["GET"])
def get_symbol_performance():
    """
    Get symbols performance data for charting.
    Alternative endpoint that ensures valid data format for the chart.
    """
    try:
        # Get closed positions from database, check both uppercase and lowercase status
        closed_positions = list(
            positions_collection.find(
                {"$or": [{"status": "CLOSED"}, {"status": "closed"}]}
            )
        )

        if not closed_positions:
            # Return dummy data if no positions exist
            dummy_data = [{"name": "Chưa có giao dịch", "data": [[0, 0], [1, 0]]}]
            return jsonify({"symbol_performance": dummy_data})

        # Filter out positions that don't have complete data for profit calculation
        valid_positions = []
        for position in closed_positions:
            # Calculate profit if missing
            if "profit" not in position or position["profit"] is None:
                position["profit"] = calculate_profit(position)

            # Add to valid positions if it has profit calculated
            if position.get("profit") is not None:
                valid_positions.append(position)

        # If no valid positions after filtering, return dummy data
        if not valid_positions:
            dummy_data = [
                {"name": "Chưa có giao dịch hoàn chỉnh", "data": [[0, 0], [1, 0]]}
            ]
            return jsonify({"symbol_performance": dummy_data})

        # Convert positions to DataFrame for easier processing
        df = pd.DataFrame(valid_positions)

        # Check for entry_time field and use it for sorting if available
        if "entry_time" in df.columns:
            # Try to convert entry_time to datetime for proper sorting
            try:
                df["entry_time"] = pd.to_datetime(df["entry_time"])
                # Sort positions by entry_time (oldest to newest)
                df = df.sort_values("entry_time")
            except Exception as e:
                app.logger.warning(
                    f"Error converting entry_time, falling back to timestamp: {str(e)}"
                )
                # Fall back to timestamp sorting if entry_time conversion fails
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
                elif "exitTimeStamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["exitTimeStamp"], unit="s")
                elif "exitTime" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["exitTime"])
                else:
                    # If no timestamp, use current time (not ideal)
                    df["timestamp"] = datetime.now()
                # Sort by timestamp
                df = df.sort_values("timestamp")
        else:
            # Use timestamp fields if entry_time is not available
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            elif "exitTimeStamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["exitTimeStamp"], unit="s")
            elif "exitTime" in df.columns:
                df["timestamp"] = pd.to_datetime(df["exitTime"])
            else:
                # If no timestamp, use current time (not ideal)
                df["timestamp"] = datetime.now()
            # Sort positions by timestamp
            df = df.sort_values("timestamp")

        # Make sure profit values are numerical
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

        # Filter out rows with NaN profit
        df = df.dropna(subset=["profit"])

        # Create symbol identifiers
        df["symbol_name"] = df["symbol"]

        # Process data by symbol
        symbol_performance = []

        for symbol_name in df["symbol_name"].unique():
            # Filter data for this symbol
            symbol_df = df[df["symbol_name"] == symbol_name].copy()

            # Ensure the symbol data maintains the same sort order as the main dataframe
            if "entry_time" in symbol_df.columns:
                symbol_df = symbol_df.sort_values("entry_time")
            elif "timestamp" in symbol_df.columns:
                symbol_df = symbol_df.sort_values("timestamp")

            # Calculate cumulative profit
            symbol_df["cumulative_profit"] = symbol_df["profit"].cumsum()

            # Create data points with trade number and cumulative profit
            data_points = []
            for i, (_, row) in enumerate(symbol_df.iterrows(), 1):
                data_points.append([i, float(row["cumulative_profit"])])

            # Add initial point if needed
            if not data_points or data_points[0][0] != 0:
                data_points.insert(0, [0, 0.0])

            # Create series for the chart
            symbol_performance.append({"name": symbol_name, "data": data_points})

        return jsonify({"symbol_performance": symbol_performance})
    except Exception as e:
        app.logger.error(f"Error in get_symbol_performance: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "symbol_performance": []}), 500


# API to get symbol frequency data for the histogram
@app.route("/api/symbol_frequency", methods=["GET"])
def get_symbol_frequency():
    try:
        # Connect to lambda_symbol_trading database to get all available symbols
        lambda_client = pymongo.MongoClient(os.getenv("MONGO_HOST_URI"))
        lambda_db = lambda_client["solo_alpha01"]
        symbols_collection = lambda_db["symbols"]  # Correct collection name

        # Get all symbols from solo_alpha01
        all_symbols = list(symbols_collection.find({}, {"symbol": 1}))
        symbols_dict = {doc["symbol"]: 0 for doc in all_symbols if "symbol" in doc}

        # If no symbols found, handle gracefully
        if not symbols_dict:
            logging.warning("No symbols found in solo_alpha01 database")
            # Try to get unique symbols just from symbols data as fallback
            symbols = list(collection.find())
            unique_symbols = set()
            for symbol in symbols:
                if "symbol" in symbol:
                    unique_symbols.add(symbol["symbol"])
            symbols_dict = {symbol: 0 for symbol in unique_symbols}

        # Count occurrences of each symbol in symbols
        symbols = list(collection.find())

        # Count frequency of each symbol
        for symbol in symbols:
            symbol = symbol.get("symbol")

            if symbol:
                if symbol in symbols_dict:
                    symbols_dict[symbol] += 1

        # Convert to format for chart display
        result = []
        for symbol, count in symbols_dict.items():
            result.append({"symbol": symbol, "count": count})

        # Sort by count (descending)
        result = sorted(result, key=lambda x: x["count"], reverse=True)

        # Return with the correct key expected by frontend: 'symbol_frequency' instead of 'data'
        return jsonify({"symbol_frequency": result})
    except Exception as e:
        logging.error(f"Error getting symbol frequency data: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e), "symbol_frequency": []}), 500


# Add missing API endpoints
@app.route("/api/system_snapshot", methods=["GET"])
def get_system_snapshot():
    """Get system snapshot information including account balance, active configs, etc."""
    try:
        # Get account balance from BingX API
        bingx = BingXClient(BINGX_API_KEY, BINGX_API_SECRET)
        account_info = bingx.get_account_info()
        account_balance = 0

        if account_info and account_info.get("code") == 0:
            account_balance = float(account_info["data"][0].get("equity", 0))

        # Count active configs
        active_configs = collection.count_documents({"is_active": True})

        # Count open orders (positions)
        open_orders = positions_collection.count_documents({"status": "open"})

        # Count closed orders
        closed_orders = positions_collection.count_documents({"status": "close"})

        # Get symbol count from solo_alpha01 database
        lambda_client = pymongo.MongoClient(os.getenv("MONGO_HOST_URI"))
        lambda_db = lambda_client["solo_alpha01"]
        symbols_collection = lambda_db["symbols"]
        symbol_count = symbols_collection.count_documents({})

        # If symbols collection is empty, count unique symbols from symbols
        if symbol_count == 0:
            unique_symbols = set()
            symbols = list(collection.find())
            for symbol in symbols:
                if "symbol" in symbol:
                    unique_symbols.add(symbol["symbol"])

            symbol_count = len(unique_symbols)

        return jsonify(
            {
                "account_balance": account_balance,
                "active_configs": active_configs,
                "open_orders": open_orders,
                "closed_orders": closed_orders,
                "symbol_count": symbol_count,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as e:
        logging.error(f"Error getting system snapshot: {str(e)}")
        return jsonify(
            {
                "account_balance": 0,
                "active_configs": 0,
                "open_orders": 0,
                "closed_orders": 0,
                "symbol_count": 0,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )


@app.route("/api/symbols", methods=["GET"])
def get_symbols():
    """Get list of available symbols."""
    try:
        # Try to get symbols from solo_alpha01 database
        lambda_client = pymongo.MongoClient(os.getenv("MONGO_HOST_URI"))
        lambda_db = lambda_client["solo_alpha01"]
        symbols_collection = lambda_db["trade_symbols"]
        symbols = list(symbols_collection.find())

        # If no symbols found, extract unique symbols from symbols
        if not symbols:
            symbols = list(collection.find())
            unique_symbols = set()
            for symbol in symbols:
                if "symbol" in symbol:
                    unique_symbols.add(symbol["symbol"])

            symbols = [{"symbol": symbol} for symbol in unique_symbols]

        # Convert ObjectId to string for JSON serialization
        for symbol in symbols:
            if "_id" in symbol:
                symbol["_id"] = str(symbol["_id"])

        return jsonify(symbols)
    except Exception as e:
        logging.error(f"Error getting symbols: {str(e)}")
        return jsonify([])



@app.route("/api/default_config", methods=["PUT"])
def update_default_config():
    """Update default trading configuration."""
    try:
        data = request.json

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Mark as default configuration
        data["is_default"] = True

        # Add timestamp
        data["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # Check if default config already exists
        existing_config = config_collection.find_one({"is_default": True})

        if existing_config:
            # Update existing default config
            config_collection.update_one(
                {"_id": existing_config["_id"]}, {"$set": data}
            )
        else:
            # Create new default config
            config_collection.insert_one(data)

        return jsonify({"message": "Default configuration updated successfully"})
    except Exception as e:
        logging.error(f"Error updating default config: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Hàm tính tương quan giữa giá BTC và equity
def calculate_btc_equity_correlation(equity_records):
    try:
        # Lọc ra các bản ghi có cả equity và btc_price
        valid_records = [
            r
            for r in equity_records
            if r.get("equity") is not None and r.get("btc_price") is not None
        ]

        # Nếu không có đủ dữ liệu, trả về None
        if len(valid_records) < 2:
            return {"value": None, "window": 0, "sample_size": len(valid_records)}

        # Chuyển đổi dữ liệu sang DataFrame để dễ xử lý
        data = []
        for record in valid_records:
            data.append(
                {
                    "timestamp": record["timestamp"],
                    "equity": record["equity"],
                    "btc_price": record["btc_price"],
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("timestamp")

        # Xác định kích thước cửa sổ
        window_size = min(2000, len(df))

        # Tính tương quan cho window gần nhất
        recent_df = df.tail(window_size)
        correlation = recent_df["equity"].corr(recent_df["btc_price"])

        return {
            "value": round(float(correlation), 4) if not pd.isna(correlation) else None,
            "window": window_size,
            "sample_size": len(df),
        }
    except Exception as e:
        logging.error(f"Error calculating BTC-Equity correlation: {str(e)}")
        return {"value": None, "window": 0, "sample_size": 0, "error": str(e)}


@app.route("/api/blacklisted_symbols", methods=["GET"])
def get_blacklisted_symbols():
    """Get all currently blacklisted symbols."""
    return jsonify([])


@app.route("/api/blacklisted_symbols/<symbol_id>", methods=["DELETE"])
def remove_from_blacklist(symbol_id):
    """Remove a symbol from the blacklist."""
    try:
        # Connect to solo_alpha01 database
        lambda_client = pymongo.MongoClient(os.getenv("MONGO_HOST_URI"))
        lambda_db = lambda_client["solo_alpha01"]
        blacklist_collection = lambda_db["symbol_blacklist"]

        # Try to find and delete the blacklisted symbol by ID
        result = blacklist_collection.delete_one({"_id": ObjectId(symbol_id)})

        if result.deleted_count > 0:
            logging.info(f"Successfully removed symbol with ID {symbol_id} from blacklist")
            return jsonify(
                {"success": True, "message": "symbol successfully removed from blacklist"}
            )
        else:
            logging.warning(f"symbol with ID {symbol_id} not found in blacklist")
            return (
                jsonify({"success": False, "message": "symbol not found in blacklist"}),
                404,
            )

    except Exception as e:
        logging.error(f"Error removing symbol from blacklist: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


# API lấy cấu hình bot
@app.route("/api/toggle_bot_status", methods=["POST"])
def toggle_bot_status():
    """
    Toggle the bot's active status to enable/disable opening new positions.
    When inactive, the bot will continue to monitor and close existing positions,
    but will not open any new positions.
    """
    try:
        # Get the current configuration
        config = config_collection.find_one()
        if not config:
            return jsonify({"error": "Bot configuration not found"}), 404

        # Get the current status or default to True
        current_status = config.get("is_active", True)

        # Toggle the status
        new_status = not current_status

        # Update the status in the database and in the running bot
        success = set_bot_active_status(new_status)

        if success:
            return jsonify(
                {
                    "success": True,
                    "message": f"Bot trading status {'activated' if new_status else 'deactivated'} successfully",
                    "is_active": new_status,
                }
            )
        else:
            return jsonify({"error": "Failed to update bot status"}), 500

    except Exception as e:
        logging.error(f"Error toggling bot status: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# API lấy dữ liệu equity của simulation mode
@app.route("/api/simulation_equity_chart", methods=["GET"])
def get_simulation_equity_chart():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017")
        db = client["solo_alpha01"]
        equity_col = db["simulation_equity_curve"]
        equity_records = list(equity_col.find().sort("timestamp", 1))
        equity_data = []
        for record in equity_records:
            ts = record["timestamp"]
            if isinstance(ts, datetime):
                timestamp = int(ts.timestamp() * 1000)
            else:
                timestamp = int(ts * 1000)
            equity_data.append([timestamp, record.get("equity", 0)])
        return jsonify(
            {"equity_data": [{"name": "Simulation Equity", "data": equity_data}]}
        )
    except Exception as e:
        app.logger.error(f"Lỗi khi truy vấn dữ liệu simulation equity: {str(e)}")
        return jsonify(
            {
                "error": str(e),
                "equity_data": [{"name": "Simulation Equity", "data": []}],
            }
        )


# ----------------- Khởi chạy Background Scheduler -----------------
from apscheduler.schedulers.background import BackgroundScheduler

# Check API credentials before setting up scheduler
api_credentials_valid = verify_bingx_credentials()

# ----------------- Chạy ứng dụng Flask -----------------
if __name__ == "__main__":
    # Log the API keys (masked) for debugging
    if BINGX_API_KEY:
        logging.info(
            f"BingX API Key loaded: {BINGX_API_KEY[:5]}...{BINGX_API_KEY[-5:]}"
        )
    else:
        logging.warning("BingX API Key not found in environment variables")

    if BINGX_API_SECRET:
        logging.info(
            f"BingX API Secret loaded: {BINGX_API_SECRET[:5]}...{BINGX_API_SECRET[-5:]}"
        )
    else:
        logging.warning("BingX API Secret not found in environment variables")

    # Initialize and start scheduler only when running as main application
    scheduler = BackgroundScheduler()

    if api_credentials_valid:
        # Only schedule data collection if credentials are valid
        # scheduler.add_job(crawl_and_store_data, 'interval', minutes=15, next_run_time=datetime.now(timezone.utc))
        scheduler.add_job(
            crawl_equity_data_multi_account, "cron", hour="0,4,8,12,16,20",
    minute=1
        )  # Removed immediate execution
        logging.info("Scheduled data collection jobs successfully")
    else:
        logging.warning(
            "Data collection jobs NOT scheduled due to invalid API credentials"
        )

    scheduler.start()

    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
