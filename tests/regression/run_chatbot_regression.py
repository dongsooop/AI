#!/usr/bin/env python3
import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib import request, error, urlparse


def load_cases(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("cases must be a JSON array")
    return data


def post_chatbot(url: str, text: str, token: str | None, timeout: float):
    scheme = urlparse(url).scheme.lower()
    if scheme not in {"https"}:
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
        except Exception:
            return e.code, {"error": raw}
    except (error.HTTPError, TimeoutError, ValueError) as e:
        return 0, {"error": str(e)}


def check_case(case: dict, status_code: int, response: dict):
    result = {
        "id": case.get("id"),
        "text": case.get("text", ""),
        "status_code": status_code,
        "engine": response.get("engine"),
        "response_text": response.get("text", ""),
        "passed": True,
        "reasons": [],
    }

    if status_code != 200:
        result["passed"] = False
        result["reasons"].append(f"http_{status_code}")
        return result

    text = str(response.get("text", ""))
    engine = str(response.get("engine", ""))

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

    out_path = Path(args.out) if args.out else Path("/tmp/chatbot_regression_report.json")
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

