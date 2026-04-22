import base64

from fastapi import HTTPException, Request
from jose import JWTError, jwt
from jose.exceptions import ExpiredSignatureError

from core.settings import get_settings


def verify_jwt_token(request: Request) -> str:
    settings = get_settings()
    if not settings.secret_key or not settings.algorithm:
        raise HTTPException(status_code=500, detail="Server auth configuration missing")

    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing or malformed",
        )

    token = auth_header.split(" ", 1)[1]

    try:
        padded_key = settings.secret_key + "=" * (-len(settings.secret_key) % 4)
        signing_key = base64.urlsafe_b64decode(padded_key)
        payload = jwt.decode(token, signing_key, algorithms=[settings.algorithm])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: no subject")
        return username
    except ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Token has expired") from exc
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
