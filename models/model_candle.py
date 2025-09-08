import os
import requests
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Kết nối tới MongoDB
client = MongoClient(os.getenv("MONGO_HOST_URI"))
db = client['solo_alpha01']  # Tên database
collection = db['ohlcv']  # Tên collection


def save_data_candle_to_db(data, symbol, timeframe):
    for item in data:
        document = {
            'symbol': symbol.replace('USDT', ''),
            'timeframe': timeframe,
            'open_time': int(item[0]),
            'open': float(item[1]),
            'high': float(item[2]),
            'low': float(item[3]),
            'close': float(item[4]),
            'volume': float(item[5]),
            'close_time': int(item[6]),
            'quote_asset_volume': float(item[7]),
            'number_of_trades': int(item[8]),
            'taker_buy_base_asset_volume': float(item[9]),
            'taker_buy_quote_asset_volume': float(item[10])
        }

        # Chèn document vào collection
        collection.update_one(
            {'symbol': symbol, 'timeframe': timeframe, 'open_time': document['open_time']},
            {'$set': document},
            upsert=True
        )
    print("Data has been saved to the database successfully.")

def get_symbol_data(symbol):
    """Lấy dữ liệu của một symbol và sắp xếp theo thời gian."""
    try:
        # Truy vấn dữ liệu của symbol cụ thể
        data = list(collection.find({'symbol': symbol}).sort('open_time', 1))  # 1 để sắp xếp tăng dần

        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
