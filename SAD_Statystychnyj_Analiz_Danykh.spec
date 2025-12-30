# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# --- hidden imports (часто потрібні для scipy/matplotlib) ---
hiddenimports = []
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("scipy")
hiddenimports += collect_submodules("matplotlib")

# TkAgg backend (якщо використовуєш matplotlib у Tkinter)
hiddenimports += [
    "matplotlib.backends.backend_tkagg",
    "matplotlib.backends.backend_agg",
]

# --- data files (matplotlib шрифти/стилі/конфіги) ---
datas = []
datas += collect_data_files("matplotlib")

# Якщо є іконка — можна додати як data (не обов’язково)
# datas += [("icon.ico", ".")]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="SAD_Statystychnyj_Analiz_Danykh",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,   # windowed
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
