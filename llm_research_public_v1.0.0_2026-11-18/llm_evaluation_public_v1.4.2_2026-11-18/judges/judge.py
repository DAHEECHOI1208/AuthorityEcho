#!/usr/bin/env python3
"""Batch runner for the four supported judge profiles."""
import argparse
from pathlib import Path
import sys

from suite.common import ROOT, dumps, load_env
from suite.inputs import inspect_file, load_inputs
from suite.adapters import definitions
from suite.runner import prepare, execute, export_only, validate_config, review_strongreject


def main(argv=None):
    parser = argparse.ArgumentParser(description='LLM judge integration (no automatic retry)')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('profiles', help='List four fixed judge profiles; no API call')
    inspect = sub.add_parser('inspect', help='Read headers/row count without sending data')
    inspect.add_argument('file'); inspect.add_argument('--sheet')
    validate = sub.add_parser('validate', help='Validate input identity/text/metadata alignment offline')
    validate.add_argument('--input', required=True)
    setup = sub.add_parser('prepare', help='Freeze inputs, config, code, prompts; no inference/download')
    setup.add_argument('--input', required=True)
    setup.add_argument('--config', default=str(ROOT / 'configs/judges.json'))
    setup.add_argument('--run-dir', required=True)
    for command in ('smoke', 'run', 'resume'):
        cmd = sub.add_parser(command, help='One paid smoke request per judge' if command == 'smoke' else 'Execute/resume missing rows')
        cmd.add_argument('--run-dir', required=True)
        cmd.add_argument('--env', default='.env')
        cmd.add_argument('--judges', nargs='+', help='Subset of frozen enabled profiles')
        cmd.add_argument('--retry-errors', action='store_true', help='Manually retry previously recorded judge/HTTP errors once')
        cmd.add_argument('--retry-unknown', action='store_true', help='Explicitly resend UNKNOWN_DELIVERY (may duplicate provider billing)')
        cmd.add_argument('--max-calls', type=int, help='Limit new POST calls in this invocation')
    out = sub.add_parser('export', help='Rebuild Excel/JSONL/cohorts locally; no API call')
    out.add_argument('--run-dir', required=True)
    review = sub.add_parser('review-strongreject', help='Import raw-text-bound diagnostic reviews and export; no API call')
    review.add_argument('--run-dir', required=True)
    review.add_argument('--review-file', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'profiles':
            rows = [{k: v for k, v in d.items() if k not in {'config', 'model_spec'}} for d in definitions().values()]
            print(dumps(rows))
        elif args.command == 'inspect':
            print(dumps(inspect_file(args.file, args.sheet)))
        elif args.command == 'validate':
            rows, info = load_inputs(args.input)
            print(dumps({'task_count': len(rows), 'group_count': info['group_count'], 'conditions': info['conditions'],
                         'target_eligible_n': sum(r['target_eligible'] for r in rows),
                         'target_pending_n': sum(r['target_state'] == 'TARGET_PENDING' for r in rows)}))
        elif args.command == 'prepare':
            print(prepare(args.input, args.config, args.run_dir))
        elif args.command == 'review-strongreject':
            print(dumps(review_strongreject(args.run_dir, args.review_file)))
        elif args.command == 'export':
            print(export_only(args.run_dir))
        else:
            if args.max_calls is not None and args.max_calls < 1:
                raise ValueError('--max-calls must be positive')
            load_env(args.env)
            result = execute(args.run_dir, args.judges, args.retry_errors, args.retry_unknown, args.max_calls,
                             phase='smoke' if args.command == 'smoke' else 'judge')
            print(dumps(result))
            if args.command == 'smoke' and not result.get('smoke_ok', False):
                return 1
        return 0
    except KeyboardInterrupt:
        print('Interrupted. SQLite saved; unfinished delivery is recorded separately. Use resume.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
