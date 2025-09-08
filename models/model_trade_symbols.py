import os
import requests
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Kết nối tới MongoDB
MONGO_URI = os.environ.get('MONGO_HOST_URI', 'mongodb://localhost:27017')
client = MongoClient(MONGO_URI)
db = client['solo_alpha01']  # Tên database
collection = db['trade_symbols']  # Tên collection
deleted_collection = db['symbols_deleted']  # Collection for deleted symbols


def save_symbol_to_db(data):
    if data is None:
        print('Lỗi dữ liệu')
        return
    
    
    symbols_document = {
        'symbol': data['symbol'],
        'timeframe': data.get('timeframe', '4h'),  # Default timeframe if not provided
        'timestamp': data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    }

    # Chèn document vào collection symbols
    collection.update_one(
            {'symbol': data['symbol'], 'timeframe': symbols_document['timeframe']},
            {'$set': symbols_document},
            upsert=True
        )
    print("Data has been saved to the database successfully.")


def find_symbols(symbols):
    query = {
        "$or": [
            {"symbol": {"$in": symbols}},
        ]
    }
    return list(collection.find(query))

def delete_symbol_add_to_deleted(symbol, reason="take_profit"):
    """
    Delete a symbol from the symbols collection and add it to the symbols_deleted collection.
    
    Args:
        symbol (str): Second symbol of the symbol
        reason (str): Reason for deletion (default: 'take_profit')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Find the symbol in the symbols collection
        symbol = collection.find_one({"symbol": symbol})
        
        if not symbol:
            print(f"symbol not found in database")
            return False
        
        # Add deletion information
        symbol['deleted_reason'] = reason
        symbol['deleted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Insert into deleted_symbols collection
        deleted_collection.insert_one(symbol)
        
        # Delete from symbols collection
        result = collection.delete_one({"symbol": symbol})
        
        if result.deleted_count > 0:
            print(f"symbol {symbol} moved to symbols_deleted collection. Reason: {reason}")
            return True
        else:
            print(f"Failed to delete symbol {symbol} from symbols collection")
            return False
            
    except Exception as e:
        print(f"Error deleting symbol {symbol}: {e}")
        return False


def get_all_symbols():
    """Get all symbols from the database."""
    return list(collection.find({}))
