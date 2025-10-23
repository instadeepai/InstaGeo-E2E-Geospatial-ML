"""CRUD operations for InstaGeo backend."""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from instageo.new_apps.backend.app.models import Task as DBTask
from instageo.new_apps.backend.app.models import User

logger = logging.getLogger(__name__)


def add_user_to_db(db: Session, user_sub: str, user_info: Dict) -> User:
    """Add a user to the database.

    Args:
        db: Database session
        user_sub: User subject identifier
        user_info: User information dictionary

    Returns:
        User object
    """
    try:
        user = db.get(User, user_sub)
        if not user:
            user = User(sub=user_sub)
        user.email = user_info.get("email") if user_info.get("email") else None
        user.name = user_info.get("name") if user_info.get("name") else None
        user.last_seen_at = datetime.now(timezone.utc)

        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"Added user {user_sub} to database")
        return user
    except Exception as e:
        logger.error(f"Failed to add user {user_sub} to database: {e}")
        db.rollback()
        raise


# TODO: Add pagination handling
def get_user_tasks(db: Session, user_sub: str, stage: str | None = None) -> List[DBTask]:
    """Get all tasks for a specific user filtered by stage.

    Args:
        db: Database session
        user_sub: User subject identifier
        stage: Stage to filter by

    Returns:
        List of Task objects for the user
    """
    query = db.query(DBTask).filter(DBTask.user_sub == user_sub)

    if stage == "terminated":
        query = query.filter(DBTask.status.in_(["completed", "failed"]))
    elif stage == "in_progress":
        query = query.filter(DBTask.status.not_in(["completed", "failed"]))

    return query.order_by(DBTask.created_at.desc()).all()


def get_task_by_id(db: Session, task_id: str) -> Optional[DBTask]:
    """Get a task by its task_id.

    Args:
        db: Database session
        task_id: Task identifier

    Returns:
        Task object if found, None otherwise
    """
    return db.query(DBTask).filter(DBTask.task_id == task_id).first()


def update_task_metadata(
    db: Session,
    task_id: str,
    status: str,
    stages: Dict,
    model_short_name: Optional[str] = None,
    model_type: Optional[str] = None,
    model_name: Optional[str] = None,
    model_size: Optional[str] = None,
) -> bool:
    """Update task metadata in the database.

    Args:
        db: Database session
        task_id: Task identifier
        status: Task status
        stages: Complete stages information
        model_short_name: Model short name
        model_type: Model type
        model_name: Model name
        model_size: Model size

    Returns:
        True if update successful, False otherwise
    """
    try:
        db_task = get_task_by_id(db, task_id)
        if not db_task:
            logger.warning(f"Task {task_id} not found in database")
            return False

        db_task.status = status
        db_task.stages = json.dumps(stages)
        db_task.model_short_name = model_short_name
        db_task.model_type = model_type
        db_task.model_name = model_name
        db_task.model_size = model_size
        db_task.completed_at = datetime.now(timezone.utc)

        db.commit()
        logger.info(f"Updated task {task_id} metadata in database")
        return True

    except Exception as e:
        logger.error(f"Failed to update task {task_id} metadata: {e}")
        db.rollback()
        return False


def create_task_in_db(
    db: Session,
    user_sub: str,
    task_id: str,
    bboxes: List,
    parameters: Dict,
    status: str = "data_processing",
) -> DBTask:
    """Create a new task in the database.

    Args:
        db: Database session
        user_sub: User subject identifier
        task_id: Task identifier
        bboxes: Bounding boxes data
        parameters: Task parameters
        status: Initial task status

    Returns:
        Created Task object
    """
    try:
        db_task = DBTask(
            user_sub=user_sub,
            task_id=task_id,
            bboxes=json.dumps(bboxes) if bboxes else None,
            parameters=json.dumps(parameters) if parameters else None,
            status=status,
        )

        db.add(db_task)
        db.commit()
        db.refresh(db_task)

        logger.info(f"Created task {task_id} in database")
        return db_task
    except Exception as e:
        logger.error(f"Failed to create task {task_id} in database: {e}")
        db.rollback()
        raise


def task_to_dict(db_task: DBTask) -> Dict:
    """Convert a database task to dictionary format for API response.

    Args:
        db_task: Database Task object

    Returns:
        Dictionary representation of the task
    """
    try:
        # Parse JSON fields safely
        bboxes = json.loads(db_task.bboxes) if db_task.bboxes else []
        parameters = json.loads(db_task.parameters) if db_task.parameters else {}
        stages = json.loads(db_task.stages) if db_task.stages else {}

        return {
            "task_id": db_task.task_id,
            "status": db_task.status,
            "created_at": db_task.created_at.isoformat() + "Z",
            "bboxes": bboxes,
            "model_short_name": db_task.model_short_name,
            "model_type": db_task.model_type,
            "model_name": db_task.model_name,
            "model_size": db_task.model_size,
            "classes_mapping": parameters.get("classes_mapping"),
            "stages": stages,
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON for task {db_task.task_id}: {e}")
        raise


def get_completed_failed_tasks(db: Session, user_sub: str) -> List[Dict]:
    """Get completed and failed tasks for a user from database.

    Args:
        db: Database session
        user_sub: User subject identifier

    Returns:
        List of task dictionaries
    """
    db_tasks = get_user_tasks(db, user_sub, stage="terminated")
    tasks = []

    for db_task in db_tasks:
        try:
            if db_task.stages:
                tasks.append(task_to_dict(db_task))
        except Exception as e:
            logger.warning(f"Failed to process completed task {db_task.task_id}: {e}")
            continue

    return tasks


def get_in_progress_tasks(db: Session, user_sub: str) -> List[Dict]:
    """Get in-progress tasks for a user (requires Redis fallback).

    Args:
        db: Database session
        user_sub: User subject identifier

    Returns:
        List of task dictionaries
    """
    from instageo.new_apps.backend.app.tasks import Task

    db_tasks = get_user_tasks(db, user_sub, stage="in_progress")
    tasks = []

    for db_task in db_tasks:
        try:
            task = Task.get(db_task.task_id)
            tasks.append(
                {
                    "task_id": task.task_id,
                    "status": task.status,
                    "created_at": task.created_at,
                    "bboxes": task.bboxes,
                    "model_short_name": task.parameters.get("model_short_name"),
                    "model_type": task.parameters.get("model_type"),
                    "model_name": task.parameters.get("model_name"),
                    "model_size": task.parameters.get("model_size"),
                    "classes_mapping": task.parameters.get("classes_mapping"),
                    "stages": task.stages,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to load in-progress task {db_task.task_id}: {e}")
            continue

    return tasks
