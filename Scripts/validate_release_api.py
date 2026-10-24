from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import re
import sys
import types
import uuid
from pathlib import Path


class ReleaseAPIError(ValueError):
    """Raised when a release artifact cannot satisfy the MCUB API contract."""


_ARTIFACT_MODULE_ROOTS = {
    "OpenAgentLib",
    "Settings",
    "MCUBEvent",
    "openagent_system_tool_api",
}


def _artifact_owned_module_names(artifact: Path) -> set[str]:
    """Return bundle dependencies recorded in CubKit's generated source map."""
    names = set(_ARTIFACT_MODULE_ROOTS)
    source = artifact.read_text(encoding="utf-8")
    for match in re.finditer(r"^#\s+-\s+(.+?)\s+->\s+", source, re.MULTILINE):
        bundle_path = match.group(1)
        if not bundle_path.endswith(".py"):
            continue
        module_name = bundle_path[:-3].replace("/", ".")
        if module_name.endswith(".__init__"):
            module_name = module_name[: -len(".__init__")]
        if module_name == "OpenAgentLib" or module_name.startswith("OpenAgentLib."):
            names.add(module_name)
        elif module_name in _ARTIFACT_MODULE_ROOTS:
            names.add(module_name)
    return names


def _seed_stale_artifact_modules(names: set[str]) -> dict[str, types.ModuleType]:
    """Install non-package sentinels so an artifact must evict global dependencies."""
    sentinels = {name: types.ModuleType(f"_stale_{name}") for name in names}
    sys.modules.update(sentinels)
    return sentinels


def _absolute_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _has_symlink_component(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            return True
    return False


def _validate_paths(
    artifact_value: str | Path, core_root_value: str | Path
) -> tuple[Path, Path]:
    artifact = _absolute_path(artifact_value)
    core_root = _absolute_path(core_root_value)
    if artifact.is_symlink() or _has_symlink_component(artifact):
        raise ReleaseAPIError(f"artifact must not be a symlink: {artifact}")
    if not artifact.is_file():
        raise ReleaseAPIError(f"artifact does not exist: {artifact}")
    if _has_symlink_component(core_root):
        raise ReleaseAPIError(f"MCUB core root must not contain symlinks: {core_root}")
    core_root = core_root.resolve()
    if core_root == Path(core_root.anchor):
        raise ReleaseAPIError(f"refusing unsafe MCUB core root: {core_root}")
    if not core_root.is_dir():
        raise ReleaseAPIError(f"MCUB core root does not exist: {core_root}")
    return artifact, core_root


def _import_artifact(artifact: Path, core_root: Path):
    module_name = f"_openagent_release_{uuid.uuid4().hex}"
    previous_path = list(sys.path)
    previous_modules = set(sys.modules)
    artifact_module_names = _artifact_owned_module_names(artifact)
    missing = object()
    previous_artifact_modules = {
        name: sys.modules.get(name, missing) for name in artifact_module_names
    }
    sys.path.insert(0, str(core_root))
    try:
        core_module = importlib.import_module("core.lib.loader.module_base")
        stale_modules = _seed_stale_artifact_modules(artifact_module_names)
        module_spec = importlib.util.spec_from_file_location(module_name, artifact)
        if module_spec is None or module_spec.loader is None:
            raise ReleaseAPIError(f"unable to create import spec: {artifact}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        retained = sorted(
            name
            for name, sentinel in stale_modules.items()
            if sys.modules.get(name) is sentinel
        )
        if retained:
            raise ReleaseAPIError(
                "artifact reused stale dependency modules: " + ", ".join(retained)
            )
        return module, core_module.ModuleBase
    except ReleaseAPIError:
        raise
    except Exception as exc:
        raise ReleaseAPIError(
            f"artifact import failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        for name in set(sys.modules) - previous_modules:
            sys.modules.pop(name, None)
        for name, previous_module in previous_artifact_modules.items():
            if previous_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module
        sys.path[:] = previous_path


def validate_release_api(
    artifact_value: str | Path, core_root_value: str | Path
) -> None:
    artifact, core_root = _validate_paths(artifact_value, core_root_value)
    module, module_base = _import_artifact(artifact, core_root)
    agent = getattr(module, "OpenAgent", None)
    if agent is None or not inspect.isclass(agent):
        raise ReleaseAPIError("artifact does not export class OpenAgent")
    if module_base not in getattr(agent, "__mro__", ()):
        raise ReleaseAPIError("OpenAgent MRO does not include MCUB ModuleBase")

    snapshot = getattr(agent, "_registry_catalog_snapshot", None)
    if not callable(snapshot):
        raise ReleaseAPIError("OpenAgent lacks callable _registry_catalog_snapshot")

    final_buttons = getattr(agent, "_final_buttons", None)
    if not callable(final_buttons):
        raise ReleaseAPIError("OpenAgent lacks callable _final_buttons")
    try:
        parameters = inspect.signature(final_buttons).parameters.values()
    except (TypeError, ValueError) as exc:
        raise ReleaseAPIError(
            f"cannot inspect _final_buttons signature: {exc}"
        ) from exc
    agent_log = next(
        (parameter for parameter in parameters if parameter.name == "agent_log"), None
    )
    if agent_log is None or agent_log.kind not in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }:
        raise ReleaseAPIError("_final_buttons must accept keyword-capable agent_log")

    registry_getter = getattr(agent, "_get_installed_plugin_registry", None)
    if registry_getter is not None and not callable(registry_getter):
        raise ReleaseAPIError(
            "OpenAgent has a non-callable registry initialization path"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate an OpenAgent release API")
    parser.add_argument("artifact")
    parser.add_argument("mcub_core_root")
    args = parser.parse_args(argv)
    try:
        validate_release_api(args.artifact, args.mcub_core_root)
    except ReleaseAPIError as exc:
        print(f"release API validation failed: {exc}", file=sys.stderr)
        return 1
    print("release API validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
