#!/usr/bin/env bash
# Build the standalone GUI bundle with PyInstaller.
# Output: dist/cv8000_compiler/cv8000_compiler
set -euo pipefail

cd "$(dirname "$0")/.."

# Ensure pyinstaller is available in the venv.
uv pip install --quiet pyinstaller

# Build (one-folder).
uv run pyinstaller gui/cv8000_compiler.spec --noconfirm --distpath dist --workpath build

echo
echo "Build complete: $(pwd)/dist/cv8000_compiler/cv8000_compiler"
