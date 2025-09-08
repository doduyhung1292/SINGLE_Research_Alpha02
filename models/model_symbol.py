#!/usr/bin/env python3
"""
Model for handling symbol information in the database.
"""
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = os.environ.get('MONGO_HOST_URI', 'mongodb://localhost:27017')
client = MongoClient(MONGO_URI)
db = client['solo_alpha01']
symbol_collection = db['symbols']

def ensure_symbol_index():
    """Ensure index on symbol field for faster queries."""
    try:
        # Create unique index on symbol field
        symbol_collection.create_index('symbol', unique=True)
        logger.info("Ensured index on symbol field")
    except Exception as e:
        logger.error(f"Error creating index on symbol collection: {e}")

def save_symbol_info_to_db(symbol_info: Dict[str, Any]) -> bool:
    """Save symbol information to the database."""
    # Ensure index exists
    ensure_symbol_index()
    
    try:
        # Prepare the document
        symbol_doc = {
            'symbol': symbol_info['symbol'],
            'tradeMinQuantity': symbol_info['tradeMinQuantity'],
            'tradeMinUSDT': symbol_info['tradeMinUSDT'],
            'pricePrecision': symbol_info['pricePrecision'],
            'quantityPrecision': symbol_info['quantityPrecision'],
            'leverage': symbol_info['leverage'],
            'updated_at': symbol_info.get('updated_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        }
        
        # Use upsert to update if exists or insert if not
        result = symbol_collection.update_one(
            {'symbol': symbol_info['symbol']},
            {'$set': symbol_doc},
            upsert=True
        )
        
        if result.modified_count > 0:
            logger.debug(f"Updated symbol information for {symbol_info['symbol']}")
        elif result.upserted_id:
            logger.debug(f"Inserted new symbol information for {symbol_info['symbol']}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving symbol information for {symbol_info.get('symbol', 'unknown')}: {e}")
        return False

def get_symbol_info_from_db(symbol: str) -> Optional[Dict[str, Any]]:
    """Get symbol information from the database."""
    try:
        # Find the symbol document
        symbol_doc = symbol_collection.find_one({'symbol': symbol})
        
        if symbol_doc:
            # MongoDB adds _id field automatically, remove it from result
            if '_id' in symbol_doc:
                del symbol_doc['_id']
            return symbol_doc
        else:
            logger.warning(f"No information found for symbol {symbol}")
            return None
    except Exception as e:
        logger.error(f"Error getting symbol information for {symbol}: {e}")
        return None

def get_all_symbols_from_db() -> List[Dict[str, Any]]:
    """Get all symbols from the database."""
    try:
        # Find all symbol documents
        symbols_cursor = symbol_collection.find({})
        
        # Convert cursor to list and remove _id field
        symbols = []
        for doc in symbols_cursor:
            if '_id' in doc:
                del doc['_id']
            symbols.append(doc)
        
        return symbols
    except Exception as e:
        logger.error(f"Error getting all symbols: {e}")
        return []

def delete_symbol_from_db(symbol: str) -> bool:
    """Delete a symbol from the database."""
    try:
        # Delete the symbol document
        result = symbol_collection.delete_one({'symbol': symbol})
        
        if result.deleted_count > 0:
            logger.info(f"Symbol {symbol} deleted from database")
            return True
        else:
            logger.warning(f"Symbol {symbol} not found in database")
            return False
    except Exception as e:
        logger.error(f"Error deleting symbol {symbol}: {e}")
        return False

if __name__ == "__main__":
    # Ensure index exists
    ensure_symbol_index()
    
    # Example usage
    test_symbol = {
        'symbol': 'BTC',
        'tradeMinQuantity': 0.001,
        'tradeMinUSDT': 5.0,
        'pricePrecision': 2,
        'quantityPrecision': 4,
        'leverage': 20,
        'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    save_symbol_info_to_db(test_symbol)
    
    # Test retrieving the symbol
    retrieved_symbol = get_symbol_info_from_db('BTC')
    if retrieved_symbol:
        print(f"Retrieved symbol: {retrieved_symbol}")
    
    # Test getting all symbols
    all_symbols = get_all_symbols_from_db()
    print(f"Total symbols in database: {len(all_symbols)}")