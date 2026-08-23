from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = ROOT.parent / "MCUB-fork"


def test_built_artifact_replaces_stale_openagent_dependencies(tmp_path: Path) -> None:
    artifact = tmp_path / "OpenAgent.py"
    subprocess.run(
        [
            "cubkit",
            "build",
            ".",
            "--release",
            "--reproducible",
            "--quiet",
            "--skip-hook",
            "-o",
            str(artifact),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    smoke = textwrap.dedent(
        """
        import importlib.util
        import inspect
        import json
        import sys
        import types
        from pathlib import Path

        artifact = Path(sys.argv[1])
        cache_root = Path(sys.argv[2]).resolve()
        sentinel_path = "/sentinel-cache/OpenAgentLib"

        stale_package = types.ModuleType("OpenAgentLib")
        stale_package.__file__ = sentinel_path + "/__init__.py"
        stale_mixins = types.ModuleType("OpenAgentLib.OpenAgentMixins")
        stale_mixins.__file__ = sentinel_path + "/OpenAgentMixins.py"
        stale_response = types.ModuleType("OpenAgentLib.ResponseAgent")
        stale_response.__file__ = sentinel_path + "/ResponseAgent.py"
        sys.modules.update(
            {
                "OpenAgentLib": stale_package,
                "OpenAgentLib.OpenAgentMixins": stale_mixins,
                "OpenAgentLib.ResponseAgent": stale_response,
            }
        )

        spec = importlib.util.spec_from_file_location("_openagent_reload_smoke", artifact)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        agent = module.OpenAgent
        assert callable(agent._registry_catalog_snapshot)
        assert "agent_log" in inspect.signature(agent._final_buttons).parameters

        def module_origins(loaded_module):
            if loaded_module.__file__ is not None:
                return [str(Path(loaded_module.__file__).resolve())]
            namespace_paths = getattr(loaded_module, "__path__", ())
            origins = [str(Path(path).resolve()) for path in namespace_paths]
            assert origins
            return origins

        loaded = {
            name: module_origins(loaded_module)
            for name, loaded_module in sys.modules.items()
            if name == "OpenAgentLib" or name.startswith("OpenAgentLib.")
        }
        assert "OpenAgentLib.OpenAgentMixins" in loaded
        assert "OpenAgentLib.ResponseAgent" in loaded
        assert all(
            origin == cache_root
            or cache_root in Path(origin).parents
            for origins in loaded.values()
            for origin in origins
        )
        assert all(
            sentinel_path not in origin
            for origins in loaded.values()
            for origin in origins
        )
        print(json.dumps({"loaded": loaded}, sort_keys=True))
        """
    )
    environment = os.environ | {
        "CUBKIT_CACHE_DIR": str(tmp_path / "cubkit-cache"),
        "PYTHONPATH": str(CORE_ROOT),
    }
    result = subprocess.run(
        [sys.executable, "-c", smoke, str(artifact), environment["CUBKIT_CACHE_DIR"]],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = json.loads(result.stdout)["loaded"]
    cache_root = (tmp_path / "cubkit-cache").resolve()
    assert all(
        origin == str(cache_root) or cache_root in Path(origin).parents
        for origins in loaded.values()
        for origin in origins
    )
