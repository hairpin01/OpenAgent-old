from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml


LOCALES = ("en", "ru", "uk")


class LocaleValidationError(ValueError):
    pass


def _project_dir() -> Path:
    configured = os.environ.get("CUBKIT_PROJECT_DIR")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


def _load_locale(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise LocaleValidationError(f"missing locale file: {path}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise LocaleValidationError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise LocaleValidationError(f"locale must contain a YAML mapping: {path}")
    return data


def _leaf_keys(data: Mapping[str, object], prefix: tuple[str, ...] = ()) -> set[str]:
    keys: set[str] = set()
    for key, value in data.items():
        if not isinstance(key, str):
            path = ".".join(prefix) or "<root>"
            raise LocaleValidationError(f"non-string key at {path}: {key!r}")

        current = (*prefix, key)
        if isinstance(value, Mapping):
            keys.update(_leaf_keys(value, current))
        elif isinstance(value, str):
            keys.add(".".join(current))
        else:
            path = ".".join(current)
            raise LocaleValidationError(
                f"locale value must be a string or mapping: {path}"
            )
    return keys


def validate_locales(locale_dir: Path) -> int:
    locale_keys = {
        locale: _leaf_keys(_load_locale(locale_dir / f"{locale}.yaml"))
        for locale in LOCALES
    }
    reference = locale_keys["en"]

    errors: list[str] = []
    for locale in LOCALES[1:]:
        keys = locale_keys[locale]
        missing = sorted(reference - keys)
        extra = sorted(keys - reference)
        if missing:
            errors.append(f"{locale}: missing keys: {', '.join(missing)}")
        if extra:
            errors.append(f"{locale}: extra keys: {', '.join(extra)}")

    if errors:
        raise LocaleValidationError("\n".join(errors))
    return len(reference)


def main() -> int:
    try:
        count = validate_locales(_project_dir() / "locales")
    except LocaleValidationError as exc:
        print(f"locale check failed: {exc}", file=sys.stderr)
        return 1

    print(f"locale check passed: {', '.join(LOCALES)} ({count} strings each)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
