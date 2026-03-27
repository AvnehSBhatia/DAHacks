"""Validate Auth0 RS256 access tokens (JWKS)."""

from __future__ import annotations

import os
from typing import Any

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN", "").strip()
AUTH0_AUDIENCE = os.environ.get("AUTH0_AUDIENCE", "").strip()
AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "false").lower() in (
    "1",
    "true",
    "yes",
)

security = HTTPBearer(auto_error=False)

_jwks: PyJWKClient | None = None


def _jwks_client() -> PyJWKClient:
    global _jwks
    if not AUTH0_DOMAIN:
        raise HTTPException(
            status_code=500,
            detail="AUTH0_DOMAIN is not set but AUTH_REQUIRED is enabled",
        )
    if _jwks is None:
        _jwks = PyJWKClient(f"https://{AUTH0_DOMAIN}/.well-known/jwks.json")
    return _jwks


def verify_bearer_token(token: str) -> dict[str, Any]:
    if not AUTH0_AUDIENCE:
        raise HTTPException(
            status_code=500,
            detail="AUTH0_AUDIENCE is not set but AUTH_REQUIRED is enabled",
        )
    jwks = _jwks_client()
    key = jwks.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        key.key,
        algorithms=["RS256"],
        audience=AUTH0_AUDIENCE,
        issuer=f"https://{AUTH0_DOMAIN}/",
    )


async def get_demo_caller(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, Any]:
    """JWT claims when AUTH_REQUIRED=true; anonymous dict when false."""
    if not AUTH_REQUIRED:
        return {"sub": "anonymous"}
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        return verify_bearer_token(credentials.credentials)
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}") from e
