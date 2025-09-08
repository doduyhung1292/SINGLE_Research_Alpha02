import json
import time
import logging
import random
import string
import traceback
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Import BatchOrder from trading module
from core.api.exchange_api import batch_orders
from models.model_position import update_order_status, save_order_to_db, save_position_to_db
from core.notification.notifier import send_telegram_message
from trading import BatchOrder
from core.utils.common_utils import convert_numpy_to_python
from core.config.config_manager import CONFIG, DEFAULT_CONFIG
from core.config.constants import TRADING_MODE, TRADING_FEE, SLIPPAGE

# Initialize logging
logger = logging.getLogger(__name__)


def check_existing_open_position(symbol: str) -> bool:
    """
    Kiểm tra xem đã có vị thế đang mở cho cặp symbol trong database hay chưa
    
    Args:
        symbol
        
    Returns:
        bool: True nếu đã có vị thế đang mở, False nếu chưa có
    """
    from models.model_position import get_open_positions_from_db
    
    try:
        open_positions = get_open_positions_from_db()
        for position in open_positions:
            if position["symbol"] == symbol:
                logger.info(f"Found existing OPEN position for {symbol}")
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking existing open position for {symbol}: {e}")
        logger.error(traceback.format_exc())
        return False  # Trả về False để an toàn


def create_client_order_id(symbol: str) -> str:
    """
    Tạo client_order_id duy nhất cho một lệnh.
    
    Args:
        symbol
        
    Returns:
        str: Client order ID duy nhất
    """
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{symbol}_{timestamp}_{random_str}"


def process_simulation_orders(orders_batch: List, positions_batch: List[Dict[str, Any]], order_type: str) -> bool:
    """
    Xử lý các lệnh trong chế độ mô phỏng.
    
    Args:
        orders_batch: Danh sách các lệnh cần xử lý
        positions_batch: Danh sách dữ liệu vị thế tương ứng
        order_type: Loại lệnh (OPEN hoặc CLOSE)
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    logger.info(f"Processing {len(orders_batch)} {order_type} orders in simulation mode")
    
    try:
        for position_data in positions_batch:
            # Thêm cờ mô phỏng và phí
            position_data["simulated"] = True
            position_data["trading_fee"] = TRADING_FEE
            position_data["slippage"] = SLIPPAGE
            
            if order_type == "OPEN":
                # Xử lý lệnh mở vị thế
                position_data["status"] = "OPEN"
                position_data["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                position_data["order_id"] = [
                    f"sim_open_{position_data['symbol']}_{int(time.time())}",
                ]
                
                # Chuyển đổi kiểu dữ liệu NumPy sang Python gốc
                position_data = convert_numpy_to_python(position_data)
                
                # Lưu vị thế mới vào database
                save_result = save_order_to_db(position_data)
                if save_result:
                    logger.info(f"Simulation: Successfully opened position for {position_data['symbol']}")
                else:
                    logger.error(f"Simulation: Failed to save OPEN position for {position_data['symbol']}")
            
            elif order_type == "CLOSE":
                # Xử lý lệnh đóng vị thế
                position_data["status"] = "CLOSED"
                position_data["exit_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Tạo close_order_ids nếu chưa có
                if "close_order_id" not in position_data:
                    position_data["close_order_id"] = [
                        f"sim_close_{position_data['symbol']}_{int(time.time())}"
                    ]
                
                # Chuyển đổi kiểu dữ liệu NumPy sang Python gốc
                position_data = convert_numpy_to_python(position_data)
                
                # Cập nhật trạng thái vị thế
                update_result = update_order_status(position_data)
                if update_result:
                    logger.info(f"Simulation: Successfully closed position for {position_data['symbol']} with PnL: {position_data.get('pnl', 0)}")
                    
                    # Gửi thông báo Telegram
                    pnl_value = position_data.get('pnl', 0)
                    close_msg = (
                        f"[SIM] Đã đóng vị thế {position_data['symbol']}\n"
                        f"PnL: {pnl_value:.2f} USDT\n"
                        f"Giá đóng: {position_data.get('closePrice', 'N/A')}\n"
                        f"Spread: {position_data.get('exit_alpha', 'N/A')}"
                    )
                    send_telegram_message(close_msg)
                else:
                    logger.error(f"Simulation: Failed to update position {position_data['symbol']}")
        
        return True
    
    except Exception as e:
        error_msg = f"Error in simulation order execution: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        send_telegram_message(error_msg, is_error=True)
        return False


def create_position_order_map(positions_batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Tạo ánh xạ giữa vị thế và lệnh.
    
    Args:
        positions_batch: Danh sách dữ liệu vị thế
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary ánh xạ vị thế với lệnh
    """
    position_order_map = {}
    
    for position_data in positions_batch:
        symbol = position_data.get("symbol", "")
        
        # Tạo position_id
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        position_id = str(position_id)  # Đảm bảo là chuỗi
        
        # Tạo mapping
        position_order_map[position_id] = {
            "orders": [],
            "position_data": position_data,
            "symbol_orders": {},
            "success_status": {},
        }
    
    return position_order_map


def prepare_batch_orders(orders_batch: List[BatchOrder], positions_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chuẩn bị danh sách lệnh cho API.
    
    Args:
        orders_batch: Danh sách các lệnh cần xử lý
        positions_batch: Danh sách dữ liệu vị thế tương ứng
        
    Returns:
        List[Dict[str, Any]]: Danh sách lệnh đã được chuẩn bị
    """
    batch_orders_array = []
    
    for i, (order, position) in enumerate(zip(orders_batch, positions_batch)):
        symbol = position.get("symbol", "")
        
        # Tạo client_order_id
        client_order_id = None
        if symbol:
            client_order_id = create_client_order_id(symbol)
        
        # Thêm lệnh vào danh sách
        batch_orders_array.append({
            "symbol": order.symbol,
            "type": order.order_type,
            "side": order.side,
            "positionSide": order.position_side,
            "quantity": order.quantity,
            "clientOrderId": client_order_id,
        })
    
    return batch_orders_array


def process_successful_orders(response: Dict[str, Any], orders_batch: List[BatchOrder], 
                             position_order_map: Dict[str, Dict[str, Any]], order_type: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Xử lý các lệnh thành công từ phản hồi API.
    
    Args:
        response: Phản hồi từ API
        orders_batch: Danh sách các lệnh đã gửi
        position_order_map: Ánh xạ giữa vị thế và lệnh
        order_type: Loại lệnh (OPEN hoặc CLOSE)
        
    Returns:
        Tuple[List[str], Dict[str, Dict[str, Any]]]: Danh sách order_id thành công và client_order_id_to_filled_data
    """
    successful_orders = []
    successful_client_order_ids = set()
    client_order_id_to_filled_data = {}
    
    # Lấy danh sách các lệnh thành công từ response
    order_data_list = []
    if "data" in response:
        if "orders" in response["data"]:
            order_data_list = response["data"]["orders"]
        elif "order" in response["data"]:
            order_data_list = [response["data"]["order"]]
    
    # Nếu order_data_list có độ dài = 0 nghĩa là không có order nào được thực thi thành công
    if len(order_data_list) == 0:
        return successful_orders, client_order_id_to_filled_data
        
    for order_data in order_data_list:
        if "orderId" in order_data and ("status" not in order_data or order_data["status"] == "FILLED"):
            client_order_id = order_data.get("clientOrderId", order_data.get("clientOrderID", ""))
            if client_order_id:
                successful_orders.append(order_data["orderId"])
                successful_client_order_ids.add(client_order_id.lower())  # Chuyển về chữ thường để so sánh
                client_order_id_to_filled_data[client_order_id.lower()] = order_data
    
    # Mapping dựa trên clientOrderId
    for order_data in order_data_list:
        if "clientOrderId" in order_data or "clientOrderID" in order_data:
            client_order_id = order_data.get("clientOrderId", order_data.get("clientOrderID", ""))
            client_order_id_lower = client_order_id.lower()  # Chuyển về chữ thường để so sánh
            order_symbol = order_data["symbol"]
            matched = False
            
            # Tìm position tương ứng với clientOrderId
            for position_id, map_data in position_order_map.items():
                symbol = map_data["position_data"].get("symbol", "")
                
                # Tạo các biến so sánh ở dạng chữ thường
                symbol_lower = symbol.lower()
                position_id_lower = position_id.lower()
                
                # Kiểm tra các điều kiện khớp
                if (client_order_id_lower.startswith(f"{symbol_lower}_") or 
                    (symbol_lower in client_order_id_lower) or
                    position_id_lower in client_order_id_lower):
                    
                    # Cập nhật trạng thái thành công
                    new_client_order_id = create_client_order_id(symbol)
                    map_data["success_status"][new_client_order_id] = True
                    
                    # Thêm order_id vào position_data
                    if order_type == "OPEN":
                        if "order_ids" not in map_data:
                            map_data["order_ids"] = []
                        map_data["order_ids"].append(order_data["orderId"])
                    elif order_type == "CLOSE":
                        if "close_order_ids" not in map_data:
                            map_data["close_order_ids"] = []
                        map_data["close_order_ids"].append(order_data["orderId"])
                    
                    # Thêm order vào danh sách orders của position
                    for order in orders_batch:
                        if order.symbol.lower() == order_symbol.lower():
                            map_data["orders"].append(order)
                            map_data["symbol_orders"][order_symbol] = order
                            break
                    
                    logger.info(f"Successfully mapped order {order_data['orderId']} to position {position_id}")
                    matched = True
                    break
    
    return successful_orders, client_order_id_to_filled_data


def process_open_position(position_id: str, map_data: Dict[str, Any], client_order_id_to_filled_data: Dict[str, Dict[str, Any]]) -> bool:
    """
    Xử lý vị thế mở thành công.
    
    Args:
        position_id: ID của vị thế
        map_data: Dữ liệu ánh xạ của vị thế
        client_order_id_to_filled_data: Ánh xạ giữa client_order_id và dữ liệu lệnh đã thực hiện
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    position_data = map_data["position_data"]
    symbol = position_data.get("symbol", "")
    
    # Kiểm tra tất cả các client_order_id trong success_status
    trade_success = False
    
    # Kiểm tra nếu có bất kỳ client_order_id nào bắt đầu bằng symbol đã thành công
    for success_client_id in map_data["success_status"].keys():
        if success_client_id.lower().startswith(f"{symbol.lower()}_") and map_data["success_status"][success_client_id]:
            trade_success = True
            logger.info(f"Found successful trade with client_order_id: {success_client_id}")
            break
    
    # Lấy order ids
    order_ids = map_data.get("order_ids", [])
    
    # Kiểm tra lệnh thành công dựa trên clientOrderId
    if trade_success:
        logger.info(f"Orders successful for opening position {position_id} - of symbol {symbol}, saving to database")

        # Thêm order IDs vào position data
        position_data["order_ids"] = order_ids

        # Thêm thời gian entry
        if "entry_time" not in position_data:
            position_data["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert any NumPy types to native Python types
        position_data = convert_numpy_to_python(position_data)

        # Lấy giá entry từ kết quả API nếu có
        price_data = None
        # Tìm kiếm bất kỳ client_order_id nào bắt đầu bằng symbol_
        for filled_client_id, filled_data in client_order_id_to_filled_data.items():
            if filled_client_id.startswith(f"{symbol.lower()}_"):
                price_data = filled_data
                logger.info(f"Found price data with client_order_id: {filled_client_id}")
                break
                
        if price_data:
            if "avgPrice" in price_data:
                position_data["entryPrice"] = float(price_data["avgPrice"])
            elif "price" in price_data:
                position_data["entryPrice"] = float(price_data["price"])
                    
        # Đảm bảo chỉ giữ các trường mới
        for old_field in ["side", "positionSide", "entryPrice", "quantity"]:
            if old_field in position_data:
                del position_data[old_field]

        # Đảm bảo trạng thái là OPEN
        position_data["status"] = "OPEN"

        # Lưu vào database
        try:
            # Lưu thông tin vị thế mới vào database
            save_result = save_order_to_db(position_data)
            if save_result:
                logger.info(f"Successfully saved new position {position_id} for {symbol} to database, ID: {save_result}")
                # Thêm thông báo Telegram
                epA = position_data.get("entryPrice", "N/A")
                prev_s = position_data.get("prev_entry_spread", "N/A")
                s = position_data.get("entry_spread", "N/A")
                mb = position_data.get("entry_middle_band", "N/A")
                ub = position_data.get("entry_upper_band", "N/A")
                lb = position_data.get("entry_lower_band", "N/A")
                w = position_data.get("entry_window", "N/A")
                m = position_data.get("entry_multiplier", "N/A")

                open_msg = f"Đã mở vị thế mới cho cặp {symbol}.\n"
                open_msg += f"Giá entry: {epA} \n"
                open_msg += f"Prev Spread: {f'{prev_s:.6f}' if isinstance(prev_s, float) else str(prev_s)}, Spread: {f'{s:.6f}' if isinstance(s, float) else str(s)}\n"
                open_msg += f"Bands (W:{w}, M:{m}): L={f'{lb:.6f}' if isinstance(lb, float) else str(lb)}, Mid={f'{mb:.6f}' if isinstance(mb, float) else str(mb)}, U={f'{ub:.6f}' if isinstance(ub, float) else str(ub)}"
                send_telegram_message(open_msg)
                return True
            else:
                logger.error(f"Failed to save position {position_id} to database")
                # Thử cách khác: save_symbol_position_to_db
                try:
                    save_result = save_symbol_position_to_db(position_data)
                    logger.info(f"Tried alternative save method for {symbol}")
                    if save_result is not None:
                        logger.info(f"Successfully saved position {position_id} using save_symbol_position_to_db")
                        # Thêm thông báo Telegram
                        send_telegram_message(f"Đã mở vị thế mới cho {symbol}-. Giá entry: {position_data.get('entryPrice', 0)}")
                        return True
                    else:
                        logger.error(f"Failed to save position {position_id} using save_symbol_position_to_db")
                        # Thông báo lỗi đến Telegram
                        send_telegram_message(f"Cảnh báo: Lệnh mở vị thế {symbol} thành công trên sàn nhưng không thể lưu vào database!", is_error=True)
                        return False
                except Exception as e2:
                    logger.error(f"Error using save_symbol_position_to_db: {e2}")
                    logger.error(traceback.format_exc())
                    return False
        except Exception as e:
            logger.error(f"Error saving position {position_id} to database: {e}")
            logger.error(traceback.format_exc())
            # Thông báo lỗi đến Telegram
            send_telegram_message(f"Cảnh báo: Lệnh mở vị thế {symbol} thành công trên sàn nhưng lỗi khi lưu vào database: {str(e)}", is_error=True)
            return False
    
    return False


def process_close_position(position_id: str, map_data: Dict[str, Any], client_order_id_to_filled_data: Dict[str, Dict[str, Any]]) -> bool:
    """
    Xử lý vị thế đóng thành công.
    
    Args:
        position_id: ID của vị thế
        map_data: Dữ liệu ánh xạ của vị thế
        client_order_id_to_filled_data: Ánh xạ giữa client_order_id và dữ liệu lệnh đã thực hiện
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    position_data = map_data["position_data"]
    symbol = position_data.get("symbol", "")

    if not symbol:
        logger.error(f"Position missing symbol: {position_data}")
        return False
    
    symbol_full = f"{symbol}-USDT"
    
    # Kiểm tra tất cả các client_order_id trong success_status
    trade_success = False
    close_order_ids = map_data.get("close_order_ids", [])
    
    # Kiểm tra nếu có bất kỳ client_order_id nào bắt đầu bằng symbol_ đã thành công
    for success_client_id in map_data["success_status"].keys():
        if success_client_id.startswith(f"{symbol}_") and map_data["success_status"][success_client_id]:
            trade_success = True
            logger.info(f"Found successful trade with client_order_id: {success_client_id}")
            break
    
    if trade_success:
        logger.info(f"Order for position {position_id} {symbol} successful, updating database")
        position_data["close_order_ids"] = close_order_ids
        
        # Lấy avgPrice từ kết quả API
        # Tìm price_data từ bất kỳ client_order_id nào khớp với symbol
        price_data = None
        for filled_client_id, filled_data in client_order_id_to_filled_data.items():
            if filled_data["symbol"] == f"{symbol}-USDT":
                price_data = filled_data
                break
        
        if price_data and "avgPrice" in price_data:
            avg_price = float(price_data["avgPrice"])
            position_data["closePrice"] = avg_price
           
                    
            # Tính lại PnL dựa trên giá đóng thực tế
            entry_price = float(position_data.get("entryPrice", 0))
            quantity = float(position_data.get("quantity", 0))
            position_side = position_data.get("positionSide", "LONG")

            if position_side == "LONG":
                pnl = (avg_price - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - avg_price) * quantity

            position_data["pnl"] = pnl

        # Đảm bảo các trường bắt buộc tồn tại
        if "closePrice" not in position_data:
            position_data["closePrice"] = 0
        if "pnl" not in position_data:
            position_data["pnl"] = 0

        position_data = convert_numpy_to_python(position_data)
        update_result = update_order_status(position_data)
        if update_result:
            logger.info(f"Successfully updated position {position_id} as CLOSED")
            pnl_value = position_data.get("pnl", 0)
            close_price = position_data.get("closePrice", "N/A")

            send_telegram_message(f"Đã đóng vị thế {position_data.get('symbol')} (ID: {position_id}) thành công.\nGiá đóng: A={close_price}. PnL: {pnl_value:.2f} USDT")
            return True
        else:
            logger.error(f"Failed to update position {position_id} in database")
            return False
    else:
        send_telegram_message(f"Cảnh báo: Lệnh đóng tradeSymbol {symbol_full} không thành công cho cặp {position_data.get('symbol')} (Position ID: {position_id})", is_error=True)
        return False


def process_missed_position(position_id: str, map_data: Dict[str, Any], order_type: str) -> bool:
    """
    Xử lý vị thế bị bỏ lỡ.
    
    Args:
        position_id: ID của vị thế
        map_data: Dữ liệu ánh xạ của vị thế
        order_type: Loại lệnh (OPEN hoặc CLOSE)
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    position_data = map_data["position_data"]
    symbol = position_data.get("symbol", "")
    
    # Đánh dấu client order ID là thành công
    new_client_order_id = create_client_order_id(symbol)
    map_data["success_status"][new_client_order_id] = True
    
    if order_type == "CLOSE":
        logger.info(f"Processing missed CLOSE position: {position_id} ({symbol})")

        # Chuyển đổi kiểu dữ liệu NumPy sang Python gốc
        position_data = convert_numpy_to_python(position_data)

        # Cập nhật trạng thái vị thế
        update_result = update_order_status(position_data)
        if update_result:
            logger.info(f"Successfully updated missed position {position_id} as CLOSED")
            # Gửi thông báo Telegram về việc đóng cặp thành công
            pnl_value = position_data.get("pnl", 0)
            avg_price = position_data.get("avgPriceClose", position_data.get("closePrice", "N/A"))

            # Chuẩn bị dữ liệu cho thông báo Telegram
            s = position_data.get("exit_spread", "N/A")
            mb = position_data.get("exit_middle_band", "N/A")
            w = position_data.get("exit_window", "N/A")
            m = position_data.get("exit_multiplier", "N/A")
            
            close_msg = f"Đã đóng vị thế {symbol} (ID: {position_id}) thành công.\n"
            close_msg += f"Giá đóng TB: {avg_price}. PnL: {pnl_value:.2f} USDT\n"
            close_msg += f"Spread: {f'{s:.6f}' if isinstance(s, float) else str(s)}, MiddleBand: {f'{mb:.6f}' if isinstance(mb, float) else str(mb)}\n"
            close_msg += f"Params: Window={w}, Multiplier={m}"
            send_telegram_message(close_msg)
            return True
        else:
            logger.error(f"Failed to update missed position {position_id} in database")
            # Cập nhật trực tiếp bằng symbols
            from models.model_position import collection

            manual_update = collection.update_one(
                {
                    "symbol": symbol,
                    "status": "OPEN",
                },
                {
                    "$set": {
                        "status": "CLOSED",
                        "exit_reason": position_data.get("exit_reason", "manual_close"),
                        "exit_spread": position_data.get("exit_spread"),
                        "exit_time": position_data.get("exit_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "closePrice": position_data.get("closePrice"),
                        "pnl": position_data.get("pnl", 0),
                        "close_order_id": position_data.get("close_order_ids", []),
                    }
                },
            )
            if manual_update.matched_count > 0:
                logger.info(f"Manually updated missed position {position_id} for {symbol} as CLOSED")
                send_telegram_message(f"Đã đóng vị thế {symbol} (after manual remediation). PnL: {position_data.get('pnl', 0):.2f} USDT")
                return True
            else:
                logger.error(f"Manual update also failed for missed position {position_id} ({symbol})")
                return False
    
    elif order_type == "OPEN":
        logger.info(f"Processing missed OPEN position: {position_id} ({symbol})")

        # Thêm thời gian entry
        if "entry_time" not in position_data:
            position_data["entry_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Đảm bảo trạng thái là OPEN
        position_data["status"] = "OPEN"

        # Order IDs might be missing for missed positions, use placeholders
        if "order_ids" not in position_data or not position_data["order_ids"]:
            position_data["order_ids"] = [
                f"assumed_successful_{int(time.time())}_{symbol}",
            ]

        # Chuyển đổi kiểu dữ liệu NumPy sang Python gốc
        position_data = convert_numpy_to_python(position_data)

        # Lưu vào database
        try:
            save_result = save_order_to_db(position_data)
            if save_result:
                logger.info(f"Successfully saved missed new position {position_id} for {symbol} to database, ID: {save_result}")
                # Thêm thông báo Telegram
                send_telegram_message(f"Đã mở vị thế mới cho {symbol} (after remediation). Giá entry: {position_data.get('entryPrice', 0)}")
                return True
            else:
                logger.error(f"Failed to save missed position {position_id} to database")
                # Thử cách khác: save_symbol_position_to_db
                try:
                    save_result = save_symbol_position_to_db(position_data)
                    if save_result is not None:
                        logger.info(f"Successfully saved missed position {position_id} using save_symbol_position_to_db")
                        # Thêm thông báo Telegram
                        send_telegram_message(f"Đã mở vị thế mới cho cặp {symbol} (after alternate method). Giá entry: {position_data.get('entryPrice', 0)}")
                        return True
                    else:
                        logger.error(f"Failed to save missed position {position_id} using save_symbol_position_to_db")
                        # Thông báo lỗi đến Telegram
                        send_telegram_message(f"Cảnh báo: Lệnh mở vị thế {symbol} thành công trên sàn nhưng không thể lưu vào database sau nhiều lần thử!", is_error=True)
                        return False
                except Exception as e2:
                    logger.error(f"Error using save_symbol_position_to_db for missed position: {e2}")
                    logger.error(traceback.format_exc())
                    return False
        except Exception as e:
            logger.error(f"Error saving missed position {position_id} to database: {e}")
            logger.error(traceback.format_exc())
            # Thông báo lỗi đến Telegram
            send_telegram_message(f"Cảnh báo: Lệnh mở vị thế {symbol} thành công trên sàn nhưng lỗi khi lưu vào database: {str(e)}", is_error=True)
            return False
    
    return False


def execute_batch_orders(orders: List[BatchOrder], order_type: str, position_data_list: List[Dict[str, Any]]) -> bool:
    """
    Thực thi một loạt lệnh trên sàn giao dịch.
    
    Args:
        orders: Danh sách các lệnh cần thực thi
        order_type: Loại lệnh (OPEN hoặc CLOSE)
        position_data_list: Danh sách dữ liệu vị thế tương ứng với các lệnh
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    # Xử lý chế độ mô phỏng
    if TRADING_MODE == "simulation":
        return process_simulation_orders(orders, position_data_list, order_type)
    
    # Kiểm tra nếu không có lệnh nào để thực thi
    if not orders:
        logger.warning(f"No {order_type} orders to execute")
        return True
    
    # Nếu là lệnh OPEN, kiểm tra xem đã có vị thế đang mở cho các cặp này chưa
    if order_type == "OPEN":
        # Lọc ra các vị thế không có vị thế đang mở
        filtered_positions = []
        filtered_orders = []
        
        for i, position_data in enumerate(position_data_list):
            symbol = position_data.get("symbol", "")
            
            if not check_existing_open_position(symbol):
                filtered_positions.append(position_data)
                if i < len(orders):
                    filtered_orders.append(orders[i])
            else:
                logger.warning(f"Skipping OPEN order for {symbol} as there is already an open position")
        
        # Cập nhật lại danh sách orders và position_data
        orders = filtered_orders
        position_data_list = filtered_positions
        
        # Nếu không còn orders nào sau khi lọc, trả về True
        if not orders:
            logger.warning("No new positions to open after filtering existing open positions")
            return True
    
    logger.info(f"Executing {len(orders)} {order_type} orders")
    
    # Chia lệnh thành các lô với kích thước tối đa là 15 lệnh
    batch_size = 15
    batches = [orders[i : i + batch_size] for i in range(0, len(orders), batch_size)]
    position_batches = [position_data_list[i : i + batch_size] for i in range(0, len(position_data_list), batch_size)]
    
    if len(batches) > 1:
        logger.info(f"Split orders into {len(batches)} batches with maximum {batch_size} orders per batch")
    
    all_success = True
    all_order_ids = []
    
    # Xử lý từng lô
    for batch_index, (orders_batch, positions_batch) in enumerate(zip(batches, position_batches)):
        logger.info(f"Processing batch {batch_index+1}/{len(batches)} with {len(orders_batch)} orders")
        
        # Tạo ánh xạ giữa vị thế và lệnh
        position_order_map = create_position_order_map(positions_batch)
        
        # Chuẩn bị lệnh cho API
        batch_orders_array = prepare_batch_orders(orders_batch, positions_batch)
        batch_orders_json = {"batchOrders": json.dumps(batch_orders_array)}
        
        try:
            # Gửi lệnh đến sàn giao dịch
            response = batch_orders(batch_orders_json)
            logger.info(f"API Response for batch: {json.dumps(response, default=str)}")
            
            # Xử lý các lệnh thành công
            successful_orders, client_order_id_to_filled_data = process_successful_orders(
                response, orders_batch, position_order_map, order_type
            )
            # Kiểm tra nếu có lệnh thành công
            if successful_orders:
                # Thêm order IDs vào danh sách tổng
                all_order_ids.extend(successful_orders)
                
                # Xử lý các vị thế dựa trên loại lệnh
                if order_type == "CLOSE":
                    # Xử lý đóng vị thế
                    for position_id, map_data in position_order_map.items():
                        process_close_position(position_id, map_data, client_order_id_to_filled_data)
                
                elif order_type == "OPEN":
                    # Xử lý mở vị thế
                    for position_id, map_data in position_order_map.items():
                        process_open_position(position_id, map_data, client_order_id_to_filled_data)
                
                # Thêm độ trễ giữa các lô
                if batch_index < len(batches) - 1:
                    delay_time = 1.0  # 1 giây
                    logger.info(f"Waiting {delay_time} seconds before processing next batch...")
                    time.sleep(delay_time)
                
                # Xử lý các vị thế bị bỏ lỡ
                processed_positions = set()
                for position_id, map_data in position_order_map.items():
                    # Kiểm tra xem vị thế đã được xử lý chưa
                    trade_success = False
                    for success_client_id, success_status in map_data["success_status"].items():
                        if success_status:
                            trade_success = True
                            break
                    
                    if trade_success:
                        processed_positions.add(position_id)
                    else:
                        # Nếu API báo cáo thành công chung (code=0), xử lý vị thế bị bỏ lỡ
                        if response.get("code") == 0:
                            process_missed_position(position_id, map_data, order_type)
            else:
                # API trả về lỗi
                error_msg = f"API error executing {order_type} batch {batch_index+1}: {response}"
                logger.error(error_msg)
                
                # Lấy thông điệp lỗi từ API
                api_error_msg = response.get("msg", "Unknown error")
                
                # Lấy danh sách các symbols trong batch này
                symbols_in_batch = [order.symbol for order in orders_batch]
                symbols_str = ", ".join(symbols_in_batch)
                
                # Gửi thông báo lỗi đến Telegram
                send_telegram_message(
                    f'Lỗi khi thực thi batch {batch_index+1}/{len(batches)} của {len(orders_batch)} lệnh {order_type} cho symbols [{symbols_str}]: "{api_error_msg}"',
                    is_error=True
                )
                
                all_success = False
        
        except Exception as e:
            # Xử lý ngoại lệ
            error_msg = f"Exception executing {order_type} batch {batch_index+1}: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Gửi thông báo lỗi đến Telegram
            send_telegram_message(error_msg, is_error=True)
            
            all_success = False
    
    return all_success