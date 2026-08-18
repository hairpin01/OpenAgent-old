# OpenAgent for MCUB

OpenAgent is a tool-oriented AI agent implemented as a multi-file MCUB module
and packaged into one artifact by CubKit.

## Runtime architecture

- `OpenAgentMain.py` contains MCUB registration, commands and configuration.
- `OpenAgentLib/AgentRuntime.py` owns pure model-call, retry, prompt-routing and
  token-budget policies.
- `OpenAgentLib/HttpClient.py` owns the reusable HTTP session.
- `OpenAgentLib/Plugin/PluginsEngine.py` integrates plugins, tools and providers.
- `OpenAgentLib/SystemPlugins/` contains dynamically discovered tool descriptors.

The default runtime skips the separate progress-note model call when
`reasoning_effort=off`, limits a request to six tool rounds/eight provider
attempts in the main agent loop (including retries),
and applies a 180-second overall deadline. Read-only tools marked
`parallel_safe=True` may run concurrently when emitted in one batch. Context is
budgeted with conservative token estimates instead of character counts.

Important tuning options:

- `agent_max_steps`, `agent_max_model_calls`, `agent_deadline`;
- `context_window_tokens`, `context_reserve_tokens`;
- `context_compaction_tokens`, `context_compaction_max_tokens`;
- `provider_reconnect_attempts`.

## Build
```bash
pip install cubkit -U
git clone https://github.com/hairpin01/OpenAgent-old &&
cd OpenAgent-old
```
```bash
cubkit build . --skip-hook &&
ls dist/
```

## Checks

```bash
python -m pytest -q
cubkit check . --release
cubkit lint . --release --no-cache
cubkit build . --release --reproducible --quiet --skip-hook -o dist/module.py
python -m py_compile dist/module.py
```

## GitHub release hook

`cubkit build . --release` copies the repository artifact, creates or updates a
GitHub release through `gh`, and then sends the configured build notification.
The release tag defaults to the module version, for example
`v0.8.0-main.build-1045`. Existing releases receive the rebuilt asset with
`--clobber`.

Requirements:

```bash
gh auth login
```

Optional environment variables:

- `OPENAGENT_GITHUB_REPO=owner/repository` — override the current repository;
- `OPENAGENT_RELEASE_TAG=v1.2.3` and `OPENAGENT_RELEASE_TITLE=...`;
- `OPENAGENT_RELEASE_NOTES_FILE=CHANGELOG.md`;
- `OPENAGENT_GITHUB_PRERELEASE=1`;
- `OPENAGENT_GITHUB_RELEASE_DRY_RUN=1` — print the `gh` command without publishing.

Use CubKit's `--skip-hook`/`--skip-hooks` option for verification builds that
must not copy, publish or notify.
