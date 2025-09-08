import requests
import logging
import traceback
import os
from typing import Optional

from core.config.constants import DEFAULT_TELEGRAM_BOT_TOKEN, DEFAULT_TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


def send_telegram_message(message: str, is_error: bool = False):
    """Send a message to Telegram using the bot API.

    Args:
        message (str): The message to send
        is_error (bool): Whether this is an error message (will add emoji)
    """
    try:
        # Get credentials from environment variables or use defaults
        telegram_bot_token = os.environ.get(
            "TELEGRAM_BOT_TOKEN", DEFAULT_TELEGRAM_BOT_TOKEN
        )
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", DEFAULT_TELEGRAM_CHAT_ID)

        # Check if telegram notifications are enabled
        telegram_enabled = os.environ.get("TELEGRAM_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        if not telegram_enabled:
            logger.info("Telegram notifications are disabled. Message not sent.")
            return

        # Format message with emoji based on type
        if is_error:
            formatted_message = f"🚨 ERROR: {message}"
        else:
            formatted_message = f"✅ {message}"

        # Send message to Telegram
        url = f"https://api.telegram.org/bot{telegram_bot_token}/sendMessage"
        data = {"chat_id": chat_id, "text": formatted_message, "parse_mode": "HTML"}

        response = requests.post(url, data=data)

        if response.status_code != 200:
            logger.error(f"Failed to send Telegram message: {response.text}")
        else:
            logger.info(f"Telegram message sent successfully")

    except Exception as e:
        logger.error(f"Error sending Telegram message: {e}")
        logger.error(traceback.format_exc())


def send_error_message(error_message: str, exception: Optional[Exception] = None):
    """Send an error message to Telegram with formatted exception details.

    Args:
        error_message (str): The error message to send
        exception (Exception, optional): The exception object, if available
    """
    try:
        full_message = error_message

        if exception:
            exc_details = str(exception)
            # Truncate if too long
            if len(exc_details) > 500:
                exc_details = exc_details[:497] + "..."
            full_message += f"\n\nDetails: {exc_details}"

        send_telegram_message(full_message, is_error=True)

    except Exception as e:
        logger.error(f"Error sending error message to Telegram: {e}")
        logger.error(traceback.format_exc())
