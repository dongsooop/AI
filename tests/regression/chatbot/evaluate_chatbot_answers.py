#!/usr/bin/env python3
"""Golden final-response checks using the real service and local RAG artifacts.

Only profanity HTTP, chat logging and LLM I/O are replaced. Unexpected LLM use
fails the case; this is not an authenticated API or generative-model evaluation.
"""
import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from run_chatbot_regression import check_case, load_cases, summarize

DEFAULT_CASES = Path(__file__).with_name('answer_quality_cases.json')
DEFAULT_REPORT = ROOT / 'tests/reports/chatbot/answer_quality_report.json'


async def evaluate(cases):
    from LLM.OSS import service
    results = []
    with patch.object(service, 'should_block_profanity', AsyncMock(return_value=False)), \
         patch.object(service, 'log_chatbot'), patch.object(service, '_log_chatbot_summary') as summary_log, \
         patch.object(service, 'call_oss_async', AsyncMock(side_effect=RuntimeError('LLM not allowed in deterministic gate'))) as llm:
        for case in cases:
            service._CACHE_GENERAL.clear()
            service._CACHE_RULE_BOOK.clear()
            summary_log.reset_mock()
            llm.reset_mock()
            try:
                response = await service.chat_with_oss(service.ChatReq(text=case['text']))
                result = check_case(case, 200, response)
                route = summary_log.call_args.args[3] if summary_log.call_args else None
                result['tool'] = route.tool if route else None
                expected_tools = case.get('expected_tools')
                if expected_tools and result['tool'] not in expected_tools:
                    result['reasons'].append('unexpected_route')
            except Exception as exc:
                result = check_case(case, 500, {})
                result['error_type'] = type(exc).__name__
            if llm.await_count:
                result['reasons'].append('unexpected_llm_call')
            result['passed'] = not result['reasons']
            results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--out', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--strict', action='store_true', help='exit 2 if any answer fails')
    args = parser.parse_args()
    cases = load_cases(args.cases)
    if args.validate_only:
        print(json.dumps({'validated_cases': len(cases)}))
        return 0
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    results = asyncio.run(evaluate(cases))
    groups = defaultdict(list)
    for result in results:
        groups[result['behavior']].append(result)
    summary = {**summarize(results), 'by_behavior': {key: summarize(value) for key, value in groups.items()}}
    report = {'suite': 'chatbot_final_answers', 'summary': summary, 'results': results,
              'execution': 'real_service_and_local_index_with_external_io_mocked',
              'cases_file': str(args.cases)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))
    return 2 if args.strict and summary['failed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
