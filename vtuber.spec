# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec file for VTuber 3D Avatar Application.
Packages PyQt6 + PyQt6-WebEngine + MediaPipe + OpenCV into a standalone executable.
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Repo root directory
ROOT_DIR = os.path.abspath(SPECPATH)

# Static web files, 3D models, and landmarks
datas = [
    (os.path.join(ROOT_DIR, 'viewer'), 'viewer'),
    (os.path.join(ROOT_DIR, 'assets'), 'assets'),
    (os.path.join(ROOT_DIR, 'models'), 'models'),
]

# Collect mediapipe internal data files
try:
    datas += collect_data_files('mediapipe')
except Exception:
    pass

hiddenimports = [
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebEngineCore',
    'cv2',
    'numpy',
    'mediapipe',
    'mediapipe.tasks',
    'mediapipe.tasks.python',
    'mediapipe.tasks.python.vision',
    'pygltflib',
    'configparser',
    'matplotlib',
]

try:
    hiddenimports += collect_submodules('mediapipe')
except Exception:
    pass

a = Analysis(
    [os.path.join(ROOT_DIR, 'src', 'main.py')],
    pathex=[os.path.join(ROOT_DIR, 'src'), ROOT_DIR],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'notebook'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Tạo 1 file .exe duy nhất (OneFile)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VTuberAvatar',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Tắt cửa sổ dòng lệnh đen khi chạy
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
