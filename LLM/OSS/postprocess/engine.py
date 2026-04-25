from typing import Optional

from core.settings import get_settings
from LLM.OSS import formatter
from LLM.OSS.postprocess.context import PostProcessContext
from LLM.OSS.postprocess.message_table import MESSAGES
from LLM.OSS.postprocess.registry import MODE_PIPELINES
from LLM.OSS.postprocess.rules_table import CONTACT_RESPONSE_RULES


settings = get_settings()


def _format_message(template_key: str, **values: object) -> str:
    template = MESSAGES[template_key]
    payload = {
        "org_homepage_label": settings.org_homepage_label,
        "org_homepage_url": settings.org_homepage_url,
        "grad_page_url": settings.grad_page_url,
        **values,
    }
    return template.format(**payload)


def _first_line(sub_answer: str) -> str:
    first = (sub_answer or "").strip().splitlines()[0].lstrip("- ").strip() if (sub_answer or "").strip() else ""
    if not first:
        return MESSAGES["not_found"]
    return first if first.endswith(("다.", "요.")) else first + "."


def _build_sub_answer_context(mode: str, user_text: str, sub_answer: str) -> PostProcessContext:
    ctx = formatter.build_contact_context(user_text, sub_answer, mode=mode)
    if not ctx.first_line:
        ctx.first_line = _first_line(sub_answer)
    return ctx


def _conditions(ctx: PostProcessContext) -> dict[str, bool]:
    return {
        "contact_intent": ctx.contact_intent,
        "has_label": ctx.has_label,
        "has_phone": ctx.has_phone,
        "has_url": ctx.has_url,
        "has_sub_answer": ctx.has_sub_answer,
        "not_has_phone": not ctx.has_phone,
    }


def _apply_contact_rules(ctx: PostProcessContext) -> Optional[str]:
    flags = _conditions(ctx)
    rules = sorted(CONTACT_RESPONSE_RULES, key=lambda item: item["priority"])
    for rule in rules:
        if ctx.mode not in rule["modes"]:
            continue
        if all(flags.get(condition, False) for condition in rule["conditions"]):
            return _format_message(rule["template_key"], label=ctx.label, phone=ctx.phone, url=ctx.url, user_text=ctx.user_text)
    return None


def _run_sub_answer(mode: str, user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    if not sub_answer:
        return MESSAGES["not_found"], None
    ctx = _build_sub_answer_context(mode, user_text, sub_answer)
    text = _apply_contact_rules(ctx)
    if text:
        return text, ctx.url
    return ctx.first_line, ctx.url


def _run_topic(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    title, url, first_line = formatter.extract_topic_candidate(user_text, sub_answer)
    if not sub_answer:
        return _format_message("topic_default", user_text=user_text), settings.org_homepage_url
    if title and url:
        url = formatter.ensure_layout_unknown(url)
        return _format_message("topic_page", user_text=user_text, title=title, url=url), url
    return first_line, None


def _run_policy(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    title, url, first_line = formatter.extract_policy_candidate(user_text, sub_answer)
    if not sub_answer:
        return _format_message("policy_default", user_text=user_text), settings.org_homepage_url
    if title and url:
        url = formatter.ensure_layout_unknown(url)
        return _format_message("official_page", user_text=user_text, title=title, url=url), url
    return first_line, None


def _run_dorm(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    title, url, first_line = formatter.extract_dorm_candidate(user_text, sub_answer)
    if not sub_answer:
        return _format_message("dorm_default", user_text=user_text), settings.org_homepage_url
    if title and url:
        url = formatter.ensure_layout_unknown(url)
        return _format_message("official_page", user_text=user_text, title=title, url=url), url
    return first_line, None


def _run_grad(user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    summary, url = formatter.extract_grad_summary(user_text, sub_answer)
    if not sub_answer:
        return _format_message("grad_default"), settings.grad_page_url
    if "확인할 수 있습니다." in summary:
        return summary, url
    return _format_message("grad_with_url", summary=summary, url=url), url


PROCESSORS = {
    "sub_answer": _run_sub_answer,
    "topic": _run_topic,
    "policy": _run_policy,
    "dorm": _run_dorm,
    "grad": _run_grad,
}


def run_postprocess(mode: str, user_text: str, sub_answer: str) -> tuple[str, Optional[str]]:
    config = MODE_PIPELINES.get(mode)
    if not config:
        return formatter.one_sentence_from_sub_answer(user_text, sub_answer)
    processor = PROCESSORS[config["processor"]]
    if config["processor"] == "sub_answer":
        return processor(mode, user_text, sub_answer)
    return processor(user_text, sub_answer)
