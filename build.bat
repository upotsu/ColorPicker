@echo off
setlocal
cd /d %~dp0

if not exist .venv\Scripts\activate.bat (
  echo [ERROR] .venv が見つかりません。先に venv を作ってください。
  exit /b 1
)

call .venv\Scripts\activate

python -m pip install -U pip
python -m pip install -r requirements.txt

pyinstaller --noconfirm --clean ColorPicker.spec

echo.
echo [OK] 出力先:
echo %cd%\dist\ColorPicker\ColorPicker.exe
pause