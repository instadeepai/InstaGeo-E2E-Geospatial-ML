"""Authentication module for the InstaGeo backend."""
from functools import lru_cache, wraps
from typing import Any, Callable, Dict

import requests  # type: ignore
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from instageo.new_apps.backend.app.crud import add_user_to_db, get_task_by_id
from instageo.new_apps.backend.app.db import get_db
from instageo.new_apps.backend.app.models import User
from instageo.new_apps.backend.app.settings import settings

security = HTTPBearer()


@lru_cache(maxsize=1)
def get_jwks() -> Dict[str, Any]:
    """Get JSON Web Key Set (JWKS) from Auth0.

    Returns:
        Dict of the JSON Web Key Set (JWKS) from Auth0.
    """
    url = f"https://{settings.auth0_domain}/.well-known/jwks.json"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def verify_access_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    """Verifies the access token using the JSON Web Key Set (JWKS) from Auth0.

    Args:
        credentials: The HTTP Authorization Credentials.

    Returns:
        Dict of the verified access token.
    """
    try:
        jwks = get_jwks()
        unverified_header = jwt.get_unverified_header(credentials.credentials)

        rsa_key = {}
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"],
                }
                break

        if not rsa_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalid")

        payload = jwt.decode(
            credentials.credentials,
            rsa_key,
            algorithms=["RS256"],
            audience=settings.api_audience,
            issuer=f"https://{settings.auth0_domain}/",
        )
        payload["access_token"] = credentials.credentials
        return payload

    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalid")


def is_task_owner(func: Callable):
    """Decorator to ensure the current user is the owner of the task.

    Expects the wrapped endpoint to receive:
      - task_id: str
      - db: Session
      - claims: Dict[str, Any] containing key "sub"
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        task_id = kwargs.get("task_id")
        db = kwargs.get("db")
        claims = kwargs.get("claims")

        if not task_id or not db or not claims:
            raise HTTPException(status_code=400, detail="Missing required parameters")

        db_task = get_task_by_id(db, task_id)

        if not db_task or db_task.user_sub != claims.get("sub"):
            raise HTTPException(status_code=403, detail="Forbidden")

        return await func(*args, **kwargs) if callable(func) else func

    return wrapper


def get_current_user(
    db: Session = Depends(get_db), claims: Dict[str, Any] = Depends(verify_access_token)
) -> User:
    """Gets the current user from the database.

    Args:
        db: The database session.
        claims: The claims of the access token.

    Returns:
        User: The current user.
    """
    sub = claims.get("sub")

    if not sub:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalid")

    access_token = claims.get("access_token")
    user_info: Any = get_user_info(access_token) if access_token else None
    user = add_user_to_db(db, sub, user_info)
    return user


def get_user_info(access_token: str) -> Dict[str, Any]:
    """Fetch user information from Auth0.

    Args:
        access_token: The access token to use for authentication.

    Returns:
        Dict containing user information from Auth0.
    """
    import time

    url = f"https://{settings.auth0_domain}/userinfo"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {access_token}"}

    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to fetch user information",
                )
    return {}
