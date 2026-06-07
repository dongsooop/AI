from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from psycopg2 import pool as pg_pool
from sshtunnel import SSHTunnelForwarder

from core.exceptions import ConfigurationError
from core.logging import get_logger
from core.settings import get_settings


settings = get_settings()
logger = get_logger(__name__)

_ssh_tunnel: Optional[SSHTunnelForwarder] = None
_db_pool: Optional[pg_pool.ThreadedConnectionPool] = None
_log_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="chatbot_log")


def init_db_pool() -> None:
    global _ssh_tunnel, _db_pool
    ssh_host = settings.ssh_host
    db_kwargs = dict(
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        connect_timeout=3,
        options="-c statement_timeout=5000",
    )
    connection_errors: list[Exception] = []
    if ssh_host:
        try:
            if not settings.ssh_user:
                raise ConfigurationError("SSH_USER is required when SSH_HOST is set")
            if not settings.ssh_key_path:
                raise ConfigurationError("SSH_KEY_PATH is required when SSH_HOST is set")
            ssh_key_path = Path(settings.ssh_key_path).expanduser()
            if not ssh_key_path.exists():
                raise ConfigurationError(f"SSH key file does not exist: {ssh_key_path}")
            _ssh_tunnel = SSHTunnelForwarder(
                (ssh_host, 22),
                ssh_username=settings.ssh_user,
                ssh_pkey=str(ssh_key_path),
                remote_bind_address=(settings.ssh_db_host, settings.ssh_db_port),
            )
            _ssh_tunnel.start()
            tunnel_db_kwargs = dict(db_kwargs, host="localhost", port=_ssh_tunnel.local_bind_port)
            _db_pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=5, **tunnel_db_kwargs)
            logger.info(
                "chatbot_db_pool_initialized ssh_tunnel=%s db_host=%s db_port=%s",
                True,
                tunnel_db_kwargs["host"],
                tunnel_db_kwargs["port"],
            )
            return
        except ConfigurationError as exc:
            connection_errors.append(exc)
            if settings.db_host is None:
                raise
            logger.warning("chatbot_ssh_db_config_invalid fallback_to_direct=true error=%s", exc)
        except Exception as exc:
            connection_errors.append(exc)
            if _ssh_tunnel:
                _ssh_tunnel.stop()
                _ssh_tunnel = None
            if settings.db_host is None:
                raise ConfigurationError(f"SSH database connection failed: {exc}") from exc
            logger.warning("chatbot_ssh_db_connect_failed fallback_to_direct=true error=%s", exc)

    hosts = (settings.db_host,) if settings.db_host else ("localhost",)
    for host in hosts:
        if not host:
            continue
        direct_db_kwargs = dict(db_kwargs, host=host, port=settings.db_port)
        try:
            _db_pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=5, **direct_db_kwargs)
            logger.info(
                "chatbot_db_pool_initialized ssh_tunnel=%s db_host=%s db_port=%s",
                False,
                direct_db_kwargs["host"],
                direct_db_kwargs["port"],
            )
            return
        except Exception as exc:
            connection_errors.append(exc)
            logger.warning(
                "chatbot_direct_db_connect_failed db_host=%s db_port=%s error=%s",
                host,
                settings.db_port,
                exc,
            )

    raise ConfigurationError(f"Database connection failed: {connection_errors[-1]}")


def shutdown_db_pool() -> None:
    if _db_pool:
        _db_pool.closeall()
    if _ssh_tunnel:
        _ssh_tunnel.stop()
    logger.info("chatbot_db_pool_shutdown")


def _write_chatbot_log(
    query: str,
    mode: str,
    response: str,
    url: Optional[str],
    cache_hit: bool,
    latency_ms: int,
) -> None:
    if _db_pool is None:
        return
    conn = None
    try:
        conn = _db_pool.getconn()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chatbot_logs (query, mode, response, url, cache_hit, latency_ms)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (query, mode, response, url, cache_hit, latency_ms),
                )
    except Exception as exc:
        logger.warning("chatbot_log_write_failed: %s", exc, exc_info=True)
    finally:
        if conn and _db_pool:
            _db_pool.putconn(conn)


def log_chatbot(
    query: str,
    mode: str,
    response: str,
    url: Optional[str],
    cache_hit: bool,
    latency_ms: int,
) -> None:
    _log_executor.submit(_write_chatbot_log, query, mode, response, url, cache_hit, latency_ms)
