import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Settings:
    secret_key: str | None
    algorithm: str | None
    spring_timetable_url: str | None
    org_name: str
    bot_name: str
    service_name: str
    org_homepage_label: str
    org_homepage_url: str
    grad_page_url: str
    staff_url_pattern: str
    bot_aliases: tuple[str, ...]
    oss_base_url: str | None
    oss_api_key: str | None
    oss_model: str | None
    json_only_mode: bool
    chatbot_profanity_filter_enabled: bool
    text_filter_api_url: str | None
    text_filter_api_timeout: float
    root_base_path: str | None
    dept_map_path: str | None
    ssh_host: str | None
    ssh_user: str | None
    ssh_key_path: str | None
    ssh_db_host: str
    ssh_db_port: int
    db_host: str | None
    db_port: int
    db_name: str | None
    db_user: str | None
    db_password: str | None

    @property
    def repo_root(self) -> Path:
        default_root = Path(__file__).resolve().parents[1]
        if self.root_base_path:
            path = Path(os.path.expanduser(self.root_base_path))
            if not path.is_absolute():
                path = default_root / path
            return path.resolve()
        return default_root

    @property
    def resolved_dept_map_path(self) -> Path:
        if self.dept_map_path:
            path = Path(os.path.expanduser(self.dept_map_path))
            if not path.is_absolute():
                path = self.repo_root / path
            return path.resolve()
        return (self.repo_root / "data" / "department.txt").resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    org_homepage_url = os.getenv("ORG_HOMEPAGE_URL", "https://example.com")
    bot_aliases = tuple(
        alias.strip()
        for alias in os.getenv("BOT_ALIASES", "").split(",")
        if alias.strip()
    )

    return Settings(
        secret_key=os.getenv("SECRET_KEY"),
        algorithm=os.getenv("ALGORITHM"),
        spring_timetable_url=os.getenv("SPRING_TIMETABLE_URL"),
        org_name=os.getenv("ORG_NAME", "해당 기관"),
        bot_name=os.getenv("BOT_NAME", "챗봇"),
        service_name=os.getenv("SERVICE_NAME", "챗봇 서비스"),
        org_homepage_label=os.getenv("ORG_HOMEPAGE_LABEL", "공식 홈페이지"),
        org_homepage_url=org_homepage_url,
        grad_page_url=os.getenv("GRAD_PAGE_URL", org_homepage_url),
        staff_url_pattern=os.getenv("STAFF_URL_PATTERN", r"/staff|/contact"),
        bot_aliases=bot_aliases,
        oss_base_url=os.getenv("OSS_BASE_URL"),
        oss_api_key=os.getenv("OSS_API_KEY"),
        oss_model=os.getenv("OSS_MODEL"),
        json_only_mode=os.getenv("JSON_ONLY_MODE", "0").strip() == "1",
        chatbot_profanity_filter_enabled=os.getenv("CHATBOT_PROFANITY_FILTER_ENABLED", "0").strip() == "1",
        text_filter_api_url=os.getenv("TEXT_FILTER_API_URL"),
        text_filter_api_timeout=_get_float_env("TEXT_FILTER_API_TIMEOUT", 10.0),
        root_base_path=os.getenv("ROOT_BASE_PATH"),
        dept_map_path=os.getenv("DEPT_MAP_PATH"),
        ssh_host=os.getenv("SSH_HOST"),
        ssh_user=os.getenv("SSH_USER"),
        ssh_key_path=os.getenv("SSH_KEY_PATH"),
        ssh_db_host=os.getenv("SSH_DB_HOST", "localhost"),
        ssh_db_port=int(os.getenv("SSH_DB_PORT", "5433")),
        db_host=os.getenv("DB_HOST"),
        db_port=int(os.getenv("DB_PORT", "5432")),
        db_name=os.getenv("DB_NAME"),
        db_user=os.getenv("DB_USER"),
        db_password=os.getenv("DB_PASSWORD"),
    )
