"""Syntax checks for the file tools, picked by language.

``edit`` and ``write`` take an optional ``lang`` argument (default: taken
from the file extension). The new content is checked before it is saved, so
a change that breaks the file is rejected and the file stays untouched.
Add a language by writing a checker and listing it in ``CHECKERS``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import Callable

# A checker returns None when the text is fine, else a short error message
Checker = Callable[[str, str], "str | None"]

# Language values that switch the check off
SKIP = {"text", "plain", "none", "raw"}

ALIASES = {"py": "python", "python3": "python", "yml": "yaml"}
SUFFIXES = {
    ".py": "python",
    ".pyi": "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}


def _python(text: str, name: str) -> str | None:
    """Parse into an AST, then compile it (catches e.g. return outside a function)."""
    try:
        tree = ast.parse(text, filename=name)
        compile(tree, name, "exec", dont_inherit=True)
    except SyntaxError as exc:
        where = f"line {exc.lineno}, col {exc.offset}" if exc.lineno else "unknown line"
        line = (exc.text or "").strip()
        return f"{where}: {exc.msg}" + (f"\n> {line}" if line else "")
    except (ValueError, RecursionError) as exc:  # null bytes, too deep nesting
        return str(exc)
    return None


def _json(text: str, name: str) -> str | None:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f"line {exc.lineno}, col {exc.colno}: {exc.msg}"
    return None


def _yaml(text: str, name: str) -> str | None:
    try:
        import yaml
    except ImportError:
        return None  # no parser available: do not block the edit
    try:
        # compose_all checks the structure without building objects, so
        # custom tags (!Ref, !vault) are not rejected
        for _ in yaml.compose_all(text):
            pass
    except yaml.YAMLError as exc:
        return " ".join(str(exc).split())
    return None


def _toml(text: str, name: str) -> str | None:
    try:
        import tomllib
    except ImportError:
        return None
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return str(exc)
    return None


CHECKERS: dict[str, Checker] = {
    "python": _python,
    "json": _json,
    "yaml": _yaml,
    "toml": _toml,
}


def resolve_lang(lang: object, name: str) -> str | None:
    """Canonical language with a checker, or None when nothing should run."""
    key = str(lang or "").strip().lower()
    if key in SKIP:
        return None
    if not key:
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        key = SUFFIXES.get(suffix, "")
    key = ALIASES.get(key, key)
    return key if key in CHECKERS else None


@dataclass(frozen=True, slots=True)
class CheckResult:
    lang: str | None = None  # language that was checked, None = no check ran
    error: str | None = None  # the new text does not pass
    preexisting: bool = False  # the old text did not pass either


def verify(lang: object, name: str, old: str | None, new: str) -> CheckResult:
    """Check ``new``; an error that ``old`` already had does not block the change."""
    key = resolve_lang(lang, name)
    if key is None:
        return CheckResult()
    error = CHECKERS[key](new, name)
    if error is None:
        return CheckResult(lang=key)
    broken_before = old is not None and CHECKERS[key](old, name) is not None
    return CheckResult(lang=key, error=error, preexisting=broken_before)


def apply_result(check: CheckResult, payload: dict) -> None:
    """Raise when the change must be rejected, else add the outcome to ``payload``."""
    if check.error and not check.preexisting:
        raise ValueError(
            f"{check.lang} check failed, the file was NOT changed: {check.error}"
        )
    if check.error:
        payload["warning"] = (
            f"{check.lang} check fails, but the file was already broken: {check.error}"
        )
    elif check.lang:
        payload["checked"] = check.lang


LANG_PARAM = {
    "type": "string",
    "description": (
        "Language of the file for the syntax check before saving: python "
        "(parse + compile), json, yaml, toml. Optional: taken from the file "
        "extension. Use text to skip the check."
    ),
}
