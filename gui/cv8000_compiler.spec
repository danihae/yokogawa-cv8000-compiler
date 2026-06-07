# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Yokogawa CV8000 compiler GUI.

Build (one-folder):
    uv run pyinstaller gui/cv8000_compiler.spec --noconfirm

Output: dist/cv8000_compiler/cv8000_compiler
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH)
REPO_ROOT = SPEC_DIR.parent
SRC_DIR = REPO_ROOT / "src"

# The compiler package is dynamically discovered by submodule, plus a few
# scientific stacks that PyInstaller's static analysis routinely under-
# counts. Better to over-collect than chase missing-module errors at runtime.
hiddenimports = []
for pkg in ("compiler", "skimage", "scipy", "imagecodecs", "tifffile",
            "dask", "xarray", "pandas", "xmltodict", "pydantic", "tqdm"):
    hiddenimports += collect_submodules(pkg)

datas = []
for pkg in ("skimage", "scipy", "imagecodecs", "xarray", "tifffile", "dask"):
    datas += collect_data_files(pkg)

a = Analysis(
    [str(SPEC_DIR / "app.py")],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib.tests", "scipy.tests", "skimage.tests", "pandas.tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="cv8000_compiler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="cv8000_compiler",
)
