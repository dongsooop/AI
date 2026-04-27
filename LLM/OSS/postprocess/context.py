from dataclasses import dataclass
from typing import Optional


@dataclass
class PostProcessContext:
    user_text: str
    mode: str
    sub_answer: str
    label: Optional[str] = None
    phone: Optional[str] = None
    url: Optional[str] = None
    hint: Optional[dict[str, object]] = None
    first_line: str = ""
    contact_intent: bool = False

    @property
    def has_label(self) -> bool:
        return bool(self.label)

    @property
    def has_phone(self) -> bool:
        return bool(self.phone)

    @property
    def has_url(self) -> bool:
        return bool(self.url)

    @property
    def has_sub_answer(self) -> bool:
        return bool((self.sub_answer or "").strip())
