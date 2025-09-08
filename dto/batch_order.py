class BatchOrder:
    """Class representing a batch order to be sent to the exchange."""
    
    def __init__(self, symbol, order_type, side, position_side, quantity):
        self.symbol = symbol
        self.order_type = order_type  # MARKET, LIMIT, etc.
        self.side = side  # BUY or SELL
        self.position_side = position_side  # LONG or SHORT
        self.quantity = quantity
    
    def to_dict(self):
        """Convert the order to a dictionary format for API requests."""
        return {
            "symbol": self.symbol,
            "type": self.order_type,
            "side": self.side,
            "positionSide": self.position_side,
            "quantity": self.quantity
        }
