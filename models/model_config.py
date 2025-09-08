import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Connect to MongoDB
MONGO_URI = os.environ.get('MONGO_HOST_URI', 'mongodb://localhost:27017')
client = MongoClient(MONGO_URI)
db = client['solo_alpha01']  # Database name
collection = db['config']  # Collection name

def get_config_from_db():
    """Get configuration from the database"""
    config = collection.find_one({})
    if config:
        # Remove MongoDB _id field for clean return
        if '_id' in config:
            del config['_id']
    return config

def save_config_to_db(config_data):
    """Save configuration to the database"""
    if config_data is None:
        print('Error: No data provided')
        return False
    
    # Add timestamp
    config_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Update or insert configuration
    result = collection.update_one(
        {},  # Empty filter to match any document
        {'$set': config_data},
        upsert=True  # Create if it doesn't exist
    )
    
    return result.acknowledged 