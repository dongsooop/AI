#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib import request, error
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_PATH = ROOT_DIR / "tests" / "reports" / "chatbot" / "chatbot_regression_report.json"


def load_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("cases must be a non-empty JSON array")
    ids = set()
    list_fields = {'engine_in', 'all_of_text_contains', 'any_of_text_contains', 'none_of_text_contains',
                   'all_of_text_regex', 'any_of_text_regex', 'none_of_text_regex', 'expected_urls', 'expected_tools'}
    allowed_fields = list_fields | {'id', 'text', 'behavior', 'url_policy', 'single_source', 'allowed_credit_values'}
    for case in data:
        if not isinstance(case, dict) or not case.get("id") or not isinstance(case.get("text"), str):
            raise ValueError("each case needs an id and text")
        if case['id'] in ids:
            raise ValueError(f"duplicate case id: {case['id']}")
        ids.add(case['id'])
        if set(case) - allowed_fields:
            raise ValueError(f"unknown case fields: {case['id']}")
        for key in list_fields & case.keys():
            if not isinstance(case[key], list) or not all(isinstance(v, str) for v in case[key]):
                raise ValueError(f"invalid list field {key}: {case['id']}")
        if 'allowed_credit_values' in case and (not isinstance(case['allowed_credit_values'], list) or
                any(type(v) is not int or v < 0 for v in case['allowed_credit_values'])):
            raise ValueError(f"invalid credit values: {case['id']}")
        if case.get('url_policy', 'optional') not in ('required', 'forbidden', 'optional'):
            raise ValueError(f"invalid url_policy: {case['id']}")
        if any(not canonical_url(u) for u in case.get('expected_urls', [])):
            raise ValueError(f"invalid expected URL: {case['id']}")
        for key in ('all_of_text_regex', 'any_of_text_regex', 'none_of_text_regex'):
            for pattern in case.get(key, []):
                re.compile(pattern)
    return data


def canonical_url(value):
    try:
        parsed = urlparse(value.rstrip(").,]〉」』>…"))
    except (AttributeError, ValueError):
        return ""
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        return ""
    query = sorted((k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != 'layout')
    return urlunparse((parsed.scheme, parsed.netloc.lower(), parsed.path, parsed.params, urlencode(query), ''))


def post_chatbot(url: str, text: str, token: str | None, timeout: float):
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"https", "http"}:
        return 0, {"error": f"unsupported_url_scheme:{scheme or '<empty>'}"}
    body = json.dumps({"text": text}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            try:
                return resp.status, json.loads(raw)
            except json.JSONDecodeError:
                return resp.status, {"error": "invalid_json_response", "raw": raw}
    except error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="ignore")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": raw}
    except (error.URLError, TimeoutError, ValueError) as e:
        return 0, {"error": str(e)}


def check_case(case: dict, status_code: int, response: dict):
    if not isinstance(response, dict):
        response = {}
    result = {
        "id": case.get("id"),
        "text": case.get("text", ""),
        "status_code": status_code,
        "engine": response.get("engine"),
        "response_text": response.get("text", ""),
        "passed": True,
        "reasons": [],
        "behavior": case.get("behavior", "unspecified"),
        "response_url": response.get("url"),
    }

    if status_code != 200:
        result["passed"] = False
        result["reasons"].append(f"http_{status_code}")
        return result

    text = str(response.get("text", ""))
    engine = str(response.get("engine", ""))
    if not isinstance(response.get('text'), str) or not response['text'].strip():
        result['reasons'].append('missing_response_text')
    raw_url = response.get('url')
    url = canonical_url(raw_url) if isinstance(raw_url, str) else ''
    body_urls = {canonical_url(u) for u in re.findall(r'https?://[^\s)<>]+', text)} - {''}
    policy = case.get('url_policy', 'optional')
    if policy == 'required' and not url:
        result['reasons'].append('source_url_required')
    if policy == 'forbidden' and (raw_url or body_urls):
        result['reasons'].append('unexpected_source_url')
    expected_urls = {canonical_url(u) for u in case.get('expected_urls', [])}
    if expected_urls and url not in expected_urls:
        result['reasons'].append('wrong_source_url')
    if case.get('single_source') and (not url or body_urls != {url}):
        result['reasons'].append('body_button_source_mismatch')
    for pattern in case.get('all_of_text_regex', []):
        if not re.search(pattern, text):
            result['reasons'].append(f'missing_regex:{pattern}')
    for pattern in case.get('none_of_text_regex', []):
        if re.search(pattern, text):
            result['reasons'].append(f'forbidden_regex:{pattern}')
    if 'allowed_credit_values' in case:
        found_credits = {int(n) for n in re.findall(r'(\d+)\s*학점', text)}
        if found_credits - set(case['allowed_credit_values']):
            result['reasons'].append('unexpected_credit_value')

    engines = case.get("engine_in")
    if engines and engine not in engines:
        result["passed"] = False
        result["reasons"].append(f"engine_not_in:{engines}")

    for needle in case.get("all_of_text_contains", []):
        if needle not in text:
            result["passed"] = False
            result["reasons"].append(f"missing_all:{needle}")

    any_needles = case.get("any_of_text_contains", [])
    if any_needles and not any(n in text for n in any_needles):
        result["passed"] = False
        result["reasons"].append(f"missing_any:{any_needles}")

    for needle in case.get("none_of_text_contains", []):
        if needle in text:
            result["passed"] = False
            result["reasons"].append(f"contains_forbidden:{needle}")

    any_regex = case.get("any_of_text_regex", [])
    if any_regex :
        try:
            matched = any(re.search(pat, text) for pat in any_regex)
        except re.error as e:
            result["passed"] = False
            result["reasons"].append(f"missing_any_regex:{e}")
        else:
            if not matched:
                result["passed"] = False
                result["reasons"].append(f"missing_any_regex:{any_regex}")
    result['passed'] = not result['reasons']
    return result


def summarize(results: list[dict]):
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    pass_rate = round((passed / total) * 100, 2) if total else 0.0
    fail_rate = round((failed / total) * 100, 2) if total else 0.0
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "fail_rate": fail_rate,
    }


def compare_with_baseline(current: list[dict], baseline_path: Path):
    if not baseline_path.exists():
        return {"baseline_found": False}
    with baseline_path.open("r", encoding="utf-8") as f:
        base = json.load(f)
    base_map = {r.get("id"): r for r in base.get("results", [])}
    improved = []
    worsened = []
    unchanged = []

    for cur in current:
        bid = cur.get("id")
        b = base_map.get(bid)
        if not b:
            continue
        b_pass = bool(b.get("passed"))
        c_pass = bool(cur.get("passed"))
        if (not b_pass) and c_pass:
            improved.append(bid)
        elif b_pass and (not c_pass):
            worsened.append(bid)
        else:
            unchanged.append(bid)

    return {
        "baseline_found": True,
        "baseline_file": str(baseline_path),
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
    }


def main():
    ap = argparse.ArgumentParser(description="Chatbot regression evaluator")
    ap.add_argument("--url", default="http://127.0.0.1:8010/chatbot", help="chatbot endpoint URL")
    ap.add_argument("--token", default=None, help="Bearer token")
    ap.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("chatbot_regression_cases.json")),
        help="path to cases json",
    )
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--sleep-ms", type=int, default=0, help="sleep between requests")
    ap.add_argument("--out", default="", help="output report path")
    ap.add_argument("--baseline", default="", help="baseline report path for comparison")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    results = []
    for case in cases:
        status, resp = post_chatbot(args.url, case.get("text", ""), args.token, args.timeout)
        results.append(check_case(case, status, resp))
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    summary = summarize(results)
    report = {
        "url": args.url,
        "cases_file": str(Path(args.cases)),
        "summary": summary,
        "results": results,
    }

    if args.baseline:
        report["comparison"] = compare_with_baseline(results, Path(args.baseline))

    out_path = Path(args.out) if args.out else DEFAULT_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False))
    print(f"report={out_path}")
    if "comparison" in report:
        print(json.dumps(report["comparison"], ensure_ascii=False))

    if summary["failed"] > 0:
        print("failed_cases:")
        for r in results:
            if not r["passed"]:
                print(f"- {r['id']}: {', '.join(r['reasons'])}")
        sys.exit(2)


if __name__ == "__main__":
    main()

