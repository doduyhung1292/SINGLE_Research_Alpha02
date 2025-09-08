import json

def create_batch_orders_json(orders):
    return json.dumps([order.to_dict() for order in orders])

def convert_to_json(obj):
    """Convert an object to JSON string."""
    return json.dumps(obj, default=lambda o: o.__dict__, sort_keys=True, indent=4)