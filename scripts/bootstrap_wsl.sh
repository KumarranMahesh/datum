#!/usr/bin/env bash
# Bootstrap the Datum dev environment on WSL2 / Ubuntu 22.04.
# Refuses to run from /mnt/* paths because broadcast video on a Windows drive
# is a performance disaster we've already lived through once.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# 1. Refuse the bad path. This is the #1 first-timer mistake.
case "$REPO_ROOT" in
    /mnt/*)
        echo "error: repo is on a Windows-mounted path ($REPO_ROOT)."
        echo "       move the clone into the WSL filesystem (e.g. ~/datum)"
        echo "       and re-run. frame decode is ~25x slower across /mnt."
        exit 2
        ;;
esac

# 2. OS sanity. We don't pretend to support macOS or native Windows in this script.
if [[ "$(uname -s)" != "Linux" ]]; then
    echo "error: bootstrap_wsl.sh expects Linux. on macOS install deps manually."
    exit 2
fi

# 3. uv. The whole toolchain assumes it.
if ! command -v uv >/dev/null 2>&1; then
    echo "[1/5] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv present: $(uv --version)"
fi

# 4. Python pin. If the user has 3.12 floating around, uv will fetch 3.11.
echo "[2/5] resolving python 3.11"
uv python install 3.11 >/dev/null

# 5. venv + deps.
echo "[3/5] creating .venv and installing pinned deps"
uv venv --python 3.11
uv sync --extra dev --extra api --extra index

# 6. Sample match. Small enough to download on a coffee break.
SAMPLE="data/samples/sample_match.mp4"
if [[ ! -f "$SAMPLE" ]]; then
    echo "[4/5] downloading sample match (≈1.2 GB)"
    mkdir -p "$(dirname "$SAMPLE")"
    # Placeholder URL — replace with the actual hosted sample before release.
    # We intentionally don't pull from YouTube here; that's a contributor's problem.
    echo "    (skipping — wire up SAMPLE_URL before release)"
else
    echo "[4/5] sample match already present"
fi

# 7. Smoke test. Best to know the wheels turn before the user does anything else.
echo "[5/5] smoke check"
uv run python -c "import datum; print(f'    datum {datum.__name__} importable')"

echo
echo "bootstrap ok"
echo "next: source .venv/bin/activate    (or use 'uv run ...')"
