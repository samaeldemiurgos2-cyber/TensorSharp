"""Prepare a llama.cpp reference template matching TensorSharp request rendering.

Requires Jinja2 3.1.6. Optionally verify each rendered prompt with the unchanged
llama.cpp engine via the render-llama-jinja executable built from this directory.
The generated template is experiment configuration, not a llama.cpp source patch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import jinja2

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'InferenceWeb.Tests/Fixtures/MuseSpark'
SUFFIX = ' to=user<|message|>'


def reference_template(source):
    # TensorSharp's context has no strftime_now builtin, so the release omits date.
    # Preserve the complete released source verbatim between these two additions.
    return ("{%- set current_date = '' -%}\n" + source +
            "\n{%- if add_generation_prompt and enable_thinking is defined and not enable_thinking and not tools -%}"
            "{{- ' to=user<|message|>' -}}{%- endif -%}\n")


def reference_request_options(thinking, has_tools):
    options = {'chat_template_kwargs': {'enable_thinking': thinking, 'current_date': ''}}
    # llama's specialized Spark parser hardcodes the shorter generation prefix.
    # On direct-answer requests it must accept the generated body as plain text.
    # Tool requests retain the routed parser and reasoning extraction.
    if not thinking and not has_tools:
        options['reasoning_format'] = 'none'
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--renderer', type=Path)
    parser.add_argument('--llama-checkout', type=Path)
    parser.add_argument('--fixture', type=Path,
                        help='Write verified render cases to a required test fixture (requires --renderer)')
    args = parser.parse_args()
    if args.fixture and not args.renderer:
        parser.error('--fixture requires --renderer')
    args.out.mkdir(parents=True, exist_ok=True)
    source = (FIXTURES / 'unsloth_chat_template.jinja').read_text(encoding='utf-8')
    wrapper = reference_template(source)
    path = args.out / 'llama-matched-template.jinja'
    path.write_text(wrapper, encoding='utf-8', newline='\n')
    environment = jinja2.Environment()
    environment.policies['json.dumps_kwargs'] = {'sort_keys': False, 'ensure_ascii': False}
    def compile_python(value):
        # HF generation tags annotate token spans without changing rendered text.
        return environment.from_string(re.sub(r'\{%[-+]?\s*(?:end)?generation\s*[-+]?%\}', '', value))
    original = compile_python(source)
    wrapped = compile_python(wrapper)
    cases = []
    for fixture in json.loads((FIXTURES / 'jinja_reference.json').read_text(encoding='utf-8')):
        if fixture['template'] != 'unsloth_chat_template.jinja':
            continue
        for thinking in (False, True):
            context = dict(fixture['context'], current_date='', enable_thinking=thinking)
            has_tools = bool(context.get('tools'))
            expected = original.render(**context).rstrip()
            if not thinking and not has_tools:
                expected += SUFFIX
            python_prompt = wrapped.render(**context)
            assert python_prompt == expected
            case = dict(case=fixture['case'], thinking=thinking, has_tools=has_tools,
                        context=context, expected=expected,
                        request_options=reference_request_options(thinking, has_tools))
            if args.renderer:
                input_path = args.out / f'input-{len(cases)}.json'
                input_path.write_text(json.dumps(context, ensure_ascii=False), encoding='utf-8')
                result = subprocess.run([str(args.renderer.resolve()), str(path.resolve()),
                                         str(input_path.resolve())], capture_output=True, check=True)
                actual = result.stdout.decode('utf-8').replace('\r\n', '\n')
                case['llama_jinja_matches'] = actual == expected
                case['llama_jinja_actual'] = actual
            cases.append(case)
    report = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  wrapper_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
                  jinja_version=jinja2.__version__, cases=cases)
    if args.llama_checkout:
        report['llama_revision'] = subprocess.check_output(
            ['git', '-C', str(args.llama_checkout), 'rev-parse', 'HEAD'], text=True).strip()
    (args.out / 'matched-template-report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    matches = sum(case.get('llama_jinja_matches', False) for case in cases)
    print(f'Prepared {path}; {len(cases)} Python render checks; {matches}/{len(cases)} llama render matches.')
    if args.renderer and matches != len(cases):
        raise SystemExit(1)
    if args.fixture:
        fixture = dict(report, cases=[{k: v for k, v in case.items() if k != 'llama_jinja_actual'} for case in cases])
        args.fixture.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
