"""Logging configuration and utilities for A4S evaluation.

This module provides custom logging configuration for the A4S evaluation system,
including JSON formatting, filtering capabilities, and queue-based logging setup.
It supports structured logging with custom attributes and asynchronous logging
through a queue handler.
"""

import atexit
import datetime as dt
import json
import logging
import logging.config
import logging.handlers
import pathlib
from logging import Logger
from typing import override

import yaml

# Main application logger instances
app_logger = logging.getLogger("a4s_eval")
root_logger = logging.getLogger()


# Set of built-in attributes in LogRecord objects
# Used to identify custom attributes when formatting log records
LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

COLORS = {
    "DEBUG": "\033[36m",  # Cyan
    "INFO": "\033[32m",  # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",  # Red
    "CRITICAL": "\033[41m",  # Red background
}
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if levelname in COLORS:
            record.colored_levelname = f"{COLORS[levelname]}{levelname}{RESET}"
        else:
            record.colored_levelname = levelname
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON objects.

    This formatter converts log records into JSON format, allowing for structured
    logging with custom fields. It handles both standard LogRecord attributes
    and custom attributes added during logging.
    """

    def __init__(
        self,
        *,
        fmt_keys: dict[str, str] | None = None,
    ):
        """Initialize the JSON formatter.

        Args:
            fmt_keys (dict[str, str] | None): Mapping of output keys to LogRecord
                attributes. If None, uses default mapping.
        """
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record (logging.LogRecord): The log record to format

        Returns:
            str: JSON-formatted log message
        """
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord) -> dict[str, str]:
        """Prepare a dictionary representation of the log record.

        Args:
            record (logging.LogRecord): The log record to convert

        Returns:
            dict[str, str]: Dictionary containing all log record fields
        """
        # Add standard fields that should always be present
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(
                record.created, tz=dt.timezone.utc
            ).isoformat(),
        }
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        # Map custom keys according to fmt_keys configuration
        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        message.update(always_fields)

        # Add any custom attributes from the log record
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message


class NonErrorFilter(logging.Filter):
    """Filter that only allows non-error log records to pass.

    This filter is used to separate error logs from non-error logs,
    typically used to route different severity levels to different handlers.
    """

    @override
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        """Check if the record should be logged.

        Args:
            record (logging.LogRecord): The log record to check

        Returns:
            bool | logging.LogRecord: True if the record's level is INFO or lower
        """
        return record.levelno <= logging.INFO


def setup_logging() -> None:
    """Set up logging configuration from the YAML config file.

    This function reads the logging configuration from config/logging.yaml
    and sets up the logging system accordingly. It also handles the setup
    of queue-based logging if configured.
    """
    config_file = pathlib.Path("config/logging.yaml")
    with open(config_file) as f_in:
        logger_config = yaml.safe_load(f_in)

    logging.config.dictConfig(logger_config)

    # Start the queue handler's listener if it exists
    queue_handler = logging.getHandlerByName("queue_handler")
    if (
        (queue_handler is not None)
        and isinstance(queue_handler, logging.handlers.QueueHandler)
        and (queue_handler.listener is not None)
    ):
        queue_handler.listener.start()
        atexit.register(queue_handler.listener.stop)


def get_logger() -> Logger:
    """Get the application logger, setting it up if necessary.

    Returns:
        Logger: The configured application logger instance
    """
    if not root_logger.handlers:
        setup_logging()

    return app_logger
