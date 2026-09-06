#!/usr/bin/env python3
"""Harden every chat_template.jinja under a models dir against clients that send
tool-call arguments as a JSON array instead of an object.

Some clients (JetBrains AI Assistant in particular) emit tool_calls like

    {"function": {"name": "get_weather", "arguments": "[\"Moscow\", \"C\"]"}}

i.e. the arguments JSON string is an *array*, not an object. Qwen3 couples the
chat template's tool-call rendering to `tool_call.arguments|items` (it iterates
parameter name/value pairs), and jinja2's `|items` raises
"Can only get item pairs from a mapping." for anything that is not a mapping.
The pre-existing check `tool_call.arguments == ''` does not catch arrays, so the
request 400s and vLLM logs:

    TypeError: Can only get item pairs from a mapping.

This rewrites the `|items` loop to first branch on `is mapping` and falls back
to rendering non-mapping arguments as a single positional `arguments` parameter.
Idempotent: a template that already contains the branch is left untouched.

Usage:
    venv/bin/python prepare/harden_chat_template.py [models_dir | jinja_file] [...]

    # default: every models/*/chat_template.jinja under this repo
"""
import os
import sys

OLD = r"""                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}"""

NEW = r"""                {%- if tool_call.arguments is defined and tool_call.arguments != '' %}
                    {%- if tool_call.arguments is mapping %}
                        {%- for args_name, args_value in tool_call.arguments|items %}
                            {{- '<parameter=' + args_name + '>\n' }}
                            {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
                            {{- args_value }}
                            {{- '\n</parameter>\n' }}
                        {%- endfor %}
                    {%- else %}
                        {{- '<parameter=arguments>\n' }}
                        {%- set args_value = tool_call.arguments | tojson | safe %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endif %}
                {%- endif %}"""

MARKER = "tool_call.arguments is mapping"


def harden(path: str) -> str:
    """Return 'hardened', 'ok' or 'unchanged'."""
    with open(path) as f:
        src = f.read()
    if MARKER in src:
        return "unchanged"
    if OLD not in src:
        return "unknown"  # template differs; leave it alone, do not guess
    with open(path, "w") as f:
        f.write(src.replace(OLD, NEW))
    return "hardened"


def targets():
    for arg in sys.argv[1:]:
        if arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        yield arg
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = os.path.join(base, "models")
    if os.path.isdir(models):
        for d in sorted(os.listdir(models)):
            p = os.path.join(models, d, "chat_template.jinja")
            if os.path.isfile(p):
                yield p


def main() -> int:
    rc = 0
    seen = set()
    for t in targets():
        p = os.path.realpath(t)
        if p in seen:
            continue
        seen.add(p)
        if os.path.isdir(t):
            p = os.path.join(t, "chat_template.jinja")
        if not os.path.isfile(p):
            continue
        try:
            res = harden(p)
        except OSError as e:
            print(f"harden_chat_template: {p}: {e}", file=sys.stderr)
            rc = 1
            continue
        print(f"{res}: {os.path.relpath(p, os.getcwd())}")
        if res == "unknown":
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())