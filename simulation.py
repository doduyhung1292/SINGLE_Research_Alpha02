import pymongo
from datetime import datetime
from core.config.constants import SIMULATION_INITIAL_EQUITY, TRADING_FEE, SLIPPAGE
from models.model_position import get_open_positions_from_db
from models.model_position import collection as position_collection
from core.data.market_data_manager import get_latest_price_from_ohlcv
from server.main import calculate_profit

MONGO_URI = "mongodb://localhost:27017"  # Sửa lại nếu cần
DB_NAME = "solo_alpha01"
EQUITY_COLLECTION = "simulation_equity_curve"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
equity_col = db[EQUITY_COLLECTION]


def get_last_equity():
    last = equity_col.find_one(sort=[("timestamp", -1)])
    if last and "equity" in last:
        return last["equity"]
    return SIMULATION_INITIAL_EQUITY


def get_simulation_equity(symbol_data=None):
    now = datetime.now()
    # Lấy các lệnh đã đóng
    closed_positions = list(
        position_collection.find({"status": "CLOSED", "simulated": True})
    )
    pnl_closed = 0
    for pos in closed_positions:
        # Dùng đúng hàm tính PnL đã trừ phí/slippage
        pnl_closed += calculate_profit(pos)
    # Lấy các lệnh đang mở
    open_positions = list(
        position_collection.find({"status": "OPEN", "simulated": True})
    )
    pnl_open = 0
    for pos in open_positions:
        entry = pos.get("entryPrice", 0)
        qty = pos.get("quantity", 0)
        symbol = pos.get("symbol")
        side = pos.get("side", "none")
        # Lấy giá close hiện tại (giả lập)
        price = None
        if symbol_data is not None:
            price = get_latest_price_from_ohlcv(symbol, symbol_data)
        if price:
            close = float(price["close"])
            # Tính PnL mark-to-market, đã trừ phí/slippage
            # Dùng lại logic như calculate_profit nhưng closePriceA/B là giá hiện tại
            pos_tmp = pos.copy()
            pos_tmp["closePrice"] = close
            pnl_open += calculate_profit(pos_tmp)
    new_equity = SIMULATION_INITIAL_EQUITY + pnl_closed + pnl_open
    # Lưu vào DB
    equity_col.insert_one(
        {
            "timestamp": now,
            "equity": new_equity,
            "pnl_closed": pnl_closed,
            "pnl_open": pnl_open,
        }
    )
    return new_equity
