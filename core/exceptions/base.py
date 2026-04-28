from __future__ import annotations


class AppError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "application_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class BadRequestError(AppError):
    def __init__(self, message: str, *, code: str = "bad_request") -> None:
        super().__init__(message, status_code=400, code=code)


class UnauthorizedError(AppError):
    def __init__(self, message: str, *, code: str = "unauthorized") -> None:
        super().__init__(message, status_code=401, code=code)


class ForbiddenError(AppError):
    def __init__(self, message: str, *, code: str = "forbidden") -> None:
        super().__init__(message, status_code=403, code=code)


class NotFoundError(AppError):
    def __init__(self, message: str, *, code: str = "not_found") -> None:
        super().__init__(message, status_code=404, code=code)


class ConflictError(AppError):
    def __init__(self, message: str, *, code: str = "conflict") -> None:
        super().__init__(message, status_code=409, code=code)


class ServiceUnavailableError(AppError):
    def __init__(self, message: str, *, code: str = "service_unavailable") -> None:
        super().__init__(message, status_code=503, code=code)


class GatewayTimeoutError(AppError):
    def __init__(self, message: str, *, code: str = "gateway_timeout") -> None:
        super().__init__(message, status_code=504, code=code)


class ConfigurationError(AppError):
    def __init__(self, message: str, *, code: str = "configuration_error") -> None:
        super().__init__(message, status_code=500, code=code)
