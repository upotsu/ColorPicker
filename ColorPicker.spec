# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# collect_all は (datas, binaries, hiddenimports) のタプルを返す
pyside6_datas, pyside6_binaries, pyside6_hiddenimports = collect_all("PySide6")
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all("numpy")

datas = []
binaries = []
hiddenimports = []

datas += pyside6_datas + numpy_datas
binaries += pyside6_binaries + numpy_binaries
hiddenimports += pyside6_hiddenimports + numpy_hiddenimports

# 追加の保険（環境差吸収）
hiddenimports += [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

a = Analysis(
    ["color_picker_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # ここでテスト系/不要系を除外しておくと事故りにくい
    excludes=[
        "numpy.f2py.tests",
        "PySide6.scripts.deploy_lib",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="ColorPicker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUIなのでコンソールなし
    disable_windowed_traceback=False,
    # icon="assets/app.ico",  # 使うならコメント解除
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ColorPicker",
)