#!/usr/bin/env bash
set -Eeuo pipefail

readonly PROJECT_DIR="${CUBKIT_PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly ARTIFACT_INPUT="${1:-}"
readonly DRY_RUN="${OPENAGENT_GITHUB_RELEASE_DRY_RUN:-0}"

fail() {
    printf 'github release error: %s\n' "$*" >&2
    exit 1
}

enabled() {
    case "${1,,}" in
        1 | true | yes | on) return 0 ;;
        *) return 1 ;;
    esac
}

print_command() {
    printf 'dry-run:'
    printf ' %q' "$@"
    printf '\n'
}

[[ -n "$ARTIFACT_INPUT" ]] || fail "usage: release_github.sh <artifact>"

if [[ "$ARTIFACT_INPUT" = /* ]]; then
    artifact="$ARTIFACT_INPUT"
else
    artifact="$PROJECT_DIR/$ARTIFACT_INPUT"
fi
[[ -f "$artifact" ]] || fail "artifact does not exist: $artifact"

command -v gh >/dev/null 2>&1 || fail "GitHub CLI (gh) is not installed"

version="$(
    sed -nE 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*["'\'']([^"'\'']+)["'\''].*/\1/p' \
        "$PROJECT_DIR/Src/OpenAgentMain.py" | head -n 1
)"
[[ -n "$version" ]] || fail "cannot read OpenAgent version"

tag="${OPENAGENT_RELEASE_TAG:-v${version//:/-}}"
title="${OPENAGENT_RELEASE_TITLE:-OpenAgent ${version}}"
repo="${OPENAGENT_GITHUB_REPO:-}"
asset_name="${OPENAGENT_RELEASE_ASSET_NAME:-OpenAgent-MCUB-repo.py}"
notes_file="${OPENAGENT_RELEASE_NOTES_FILE:-}"

[[ "$tag" != -* && "$tag" != *$'\n'* ]] || fail "invalid release tag: $tag"
[[ "$asset_name" != */* && -n "$asset_name" ]] || fail "invalid asset name"

if enabled "$DRY_RUN"; then
    [[ -n "$repo" ]] || repo="hairpin01/OpenAgent-old"
else
    gh auth status >/dev/null
    if [[ -z "$repo" ]]; then
        repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
    fi
fi
[[ "$repo" == */* ]] || fail "invalid GitHub repository: $repo"

asset="${artifact}#${asset_name}"

if ! enabled "$DRY_RUN" && gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
    gh release upload "$tag" "$asset" --repo "$repo" --clobber
    printf 'GitHub release asset updated: %s %s\n' "$repo" "$tag"
    exit 0
fi

create_args=(release create "$tag" "$asset" --repo "$repo" --title "$title")
if [[ -n "$notes_file" ]]; then
    [[ -f "$notes_file" ]] || fail "release notes file does not exist: $notes_file"
    create_args+=(--notes-file "$notes_file")
else
    create_args+=(--generate-notes)
fi
if enabled "${OPENAGENT_GITHUB_PRERELEASE:-0}"; then
    create_args+=(--prerelease)
fi

if enabled "$DRY_RUN"; then
    print_command gh "${create_args[@]}"
else
    gh "${create_args[@]}"
    printf 'GitHub release created: %s %s\n' "$repo" "$tag"
fi
