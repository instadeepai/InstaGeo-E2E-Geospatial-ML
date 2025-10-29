"""Redis client module for InstaGeo backend.

This module provides a centralized interface for all Redis operations,
including connection management, task data storage, and job queue operations.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import redis  # type: ignore
from rq import Queue

from instageo.new_apps.backend.app.settings import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Centralized Redis client for InstaGeo application."""

    def __init__(self):
        """Initialize Redis connection."""
        self._connection = None
        self._queues = {}
        self._connect()
        self.ttl = settings.redis_ttl

    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            self._connection = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                password=os.getenv("REDIS_PASSWORD", ""),
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            self._connection.ping()
            logger.info("Connected to Redis successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    @property
    def connection(self) -> redis.Redis:
        """Get Redis connection."""
        if self._connection is None:
            self._connect()
        return self._connection

    def get_queue(self, queue_name: str = "default") -> Queue:
        """Get RQ queue by name."""
        if queue_name not in self._queues:
            self._queues[queue_name] = Queue(queue_name, connection=self.connection)
        return self._queues[queue_name]

    # Task Operations
    def save_task(self, task_id: str, task_data: Dict[str, Any]) -> bool:
        """Save task data to Redis.

        Args:
            task_id: Task identifier
            task_data: Task data dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"task:{task_id}"
            self.connection.hset(key, mapping=task_data)
            self.connection.expire(key, self.ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to save task {task_id}: {e}")
            return False

    def load_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load task data from Redis.

        Args:
            task_id: Task identifier

        Returns:
            Task data dictionary or None if not found
        """
        try:
            key = f"task:{task_id}"
            data = self.connection.hgetall(key)
            return data if data else None
        except Exception as e:
            logger.error(f"Failed to load task {task_id}: {e}")
            return None

    def delete_task(self, task_id: str) -> bool:
        """Delete task data from Redis.

        Args:
            task_id: Task identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"task:{task_id}"
            return bool(self.connection.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False

    def save_task_stage(
        self, task_id: str, stage: str, stage_data: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Save task stage data to Redis.

        Args:
            task_id: Task identifier
            stage: Stage name
            stage_data: Stage data dictionary
            ttl: Time to live in seconds (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"task:{task_id}:stage:{stage}"
            self.connection.hset(key, mapping=stage_data)
            self.connection.expire(key, self.ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to save task stage {task_id}:{stage}: {e}")
            return False

    def load_task_stage(self, task_id: str, stage: str) -> Optional[Dict[str, Any]]:
        """Load task stage data from Redis.

        Args:
            task_id: Task identifier
            stage: Stage name

        Returns:
            Stage data dictionary or None if not found
        """
        try:
            key = f"task:{task_id}:stage:{stage}"
            data = self.connection.hgetall(key)
            return data if data else None
        except Exception as e:
            logger.error(f"Failed to load task stage {task_id}:{stage}: {e}")
            return None

    def delete_task_stage(self, task_id: str, stage: str) -> bool:
        """Delete task stage data from Redis.

        Args:
            task_id: Task identifier
            stage: Stage name

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"task:{task_id}:stage:{stage}"
            return bool(self.connection.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete task stage {task_id}:{stage}: {e}")
            return False

    # Job Operations
    def save_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """Save job data to Redis.

        Args:
            job_id: Job identifier
            job_data: Job data dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"job:{job_id}"
            self.connection.hset(key, mapping=job_data)
            self.connection.expire(key, self.ttl)
            return True
        except Exception as e:
            logger.error(f"Failed to save job {job_id}: {e}")
            return False

    def load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load job data from Redis.

        Args:
            job_id: Job identifier

        Returns:
            Job data dictionary or None if not found
        """
        try:
            key = f"job:{job_id}"
            data = self.connection.hgetall(key)
            return data if data else None
        except Exception as e:
            logger.error(f"Failed to load job {job_id}: {e}")
            return None

    def delete_job(self, job_id: str) -> bool:
        """Delete job data from Redis.

        Args:
            job_id: Job identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            key = f"job:{job_id}"
            return bool(self.connection.delete(key))
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    # Sorted Set Operations
    def add_task_to_sorted_set(self, task_id: str, score: Optional[float] = None) -> bool:
        """Add task to sorted set by creation date.

        Args:
            task_id: Task identifier
            score: Timestamp score (defaults to current time)

        Returns:
            True if successful, False otherwise
        """
        try:
            if score is None:
                score = datetime.now(timezone.utc).timestamp()
            return bool(self.connection.zadd("tasks_by_created", {task_id: score}))
        except Exception as e:
            logger.error(f"Failed to add task {task_id} to sorted set: {e}")
            return False

    def get_tasks_by_creation_date(
        self, start: int = 0, end: int = -1, with_scores: bool = False
    ) -> List[Union[str, tuple]]:
        """Get tasks sorted by creation date.

        Args:
            start: Start index
            end: End index (-1 for all)
            with_scores: Include scores in results

        Returns:
            List of task IDs or (task_id, score) tuples
        """
        try:
            return self.connection.zrange("tasks_by_created", start, end, withscores=with_scores)
        except Exception as e:
            logger.error(f"Failed to get tasks by creation date: {e}")
            return []

    def remove_task_from_sorted_set(self, task_id: str) -> bool:
        """Remove task from sorted set.

        Args:
            task_id: Task identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            return bool(self.connection.zrem("tasks_by_created", task_id))
        except Exception as e:
            logger.error(f"Failed to remove task {task_id} from sorted set: {e}")
            return False

    # Failure Recovery Operations
    def store_persistence_failure(
        self, task_id: str, task_data: Dict[str, Any], ttl: int = 7 * 24 * 3600
    ) -> bool:
        """Store task data for manual retry if database persistence fails.

        Args:
            task_id: Task identifier
            task_data: Complete task data
            ttl: Time to live in seconds (default: 7 days)

        Returns:
            True if successful, False otherwise
        """
        try:
            failure_key = f"task_persistence_failure:{task_id}"
            failure_data = {
                "task_id": task_id,
                "status": task_data.get("status"),
                "stages": json.dumps(task_data.get("stages", {})),
                "parameters": json.dumps(task_data.get("parameters", {})),
                "bboxes": json.dumps(task_data.get("bboxes", [])),
                "created_at": task_data.get("created_at"),
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "retry_count": 0,
            }

            self.connection.hset(failure_key, mapping=failure_data)
            self.connection.expire(failure_key, ttl)
            logger.info(f"Stored task {task_id} persistence failure data in Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to store persistence failure data for task {task_id}: {e}")
            return False

    def get_persistence_failures(self) -> List[Dict[str, Any]]:
        """Get all failed persistence operations.

        Returns:
            List of failure data dictionaries
        """
        try:
            failure_keys = self.connection.keys("task_persistence_failure:*")
            failures = []

            for failure_key in failure_keys:
                try:
                    failure_data = self.connection.hgetall(failure_key)
                    if failure_data:
                        failures.append(
                            {
                                "task_id": failure_data.get("task_id"),
                                "status": failure_data.get("status"),
                                "failed_at": failure_data.get("failed_at"),
                                "retry_count": int(failure_data.get("retry_count", 0)),
                                "created_at": failure_data.get("created_at"),
                            }
                        )
                except Exception as e:
                    logger.error(f"Error reading failure data for {failure_key}: {e}")
                    continue

            return failures
        except Exception as e:
            logger.error(f"Failed to get persistence failures: {e}")
            return []

    def retry_persistence_failure(self, task_id: str) -> bool:
        """Retry a specific persistence failure.

        Args:
            task_id: Task identifier

        Returns:
            True if retry was successful, False otherwise
        """
        try:
            failure_key = f"task_persistence_failure:{task_id}"
            failure_data = self.connection.hgetall(failure_key)

            if not failure_data:
                return False

            # Increment retry count
            retry_count = int(failure_data.get("retry_count", 0)) + 1
            self.connection.hset(failure_key, "retry_count", retry_count)

            # Remove if too many retries
            if retry_count >= 5:
                self.connection.delete(failure_key)
                logger.warning(f"Removed task {task_id} from retry list after 5 attempts")
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to retry persistence failure for task {task_id}: {e}")
            return False

    def remove_persistence_failure(self, task_id: str) -> bool:
        """Remove persistence failure record.

        Args:
            task_id: Task identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            failure_key = f"task_persistence_failure:{task_id}"
            return bool(self.connection.delete(failure_key))
        except Exception as e:
            logger.error(f"Failed to remove persistence failure for task {task_id}: {e}")
            return False

    # Utility Operations
    def get_all_keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern.

        Args:
            pattern: Key pattern to match

        Returns:
            List of matching keys
        """
        try:
            return self.connection.keys(pattern)
        except Exception as e:
            logger.error(f"Failed to get keys with pattern {pattern}: {e}")
            return []

    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information.

        Returns:
            Redis info dictionary
        """
        try:
            return self.connection.info()
        except Exception as e:
            logger.error(f"Failed to get Redis info: {e}")
            return {}

    def ping(self) -> bool:
        """Test Redis connection.

        Returns:
            True if connection is alive, False otherwise
        """
        try:
            return self.connection.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


# Global Redis client instance
redis_client = RedisClient()
