import logging
from logging import Handler, LogRecord
from typing_extensions import override
from bson import ObjectId
from typing import Any, Optional


class MongoDBHandler(Handler):
    """Logging handler for MongoDB-related events.
    
    NOTE: MongoDB write operations have been disabled.
    All DB persistence is now handled through state_manager.
    This handler is kept for backwards compatibility but does not save to MongoDB.
    """

    def __init__(
        self,
        uri: Optional[str],
        db_name: str,
        level: int = logging.INFO,
    ) -> None:
        super().__init__(level)
        # MongoDB connections disabled - all saves go through state_manager
        self.enabled = False
        self.client = None
        self.db = None
        self.executions = None
        self.agents = None
        self.exploits = None

    def _ensure_oid(self, value: Any) -> Optional[ObjectId]:
        """Ensure value is an ObjectId if it's a non-empty string."""
        if isinstance(value, str) and value:
            return ObjectId(value)
        return value

    @override
    def emit(self, record: LogRecord) -> None:
        # MongoDB writes disabled - all persistence goes through state_manager
        # This handler is now a no-op for database operations
        pass


__all__ = ["MongoDBHandler"]
