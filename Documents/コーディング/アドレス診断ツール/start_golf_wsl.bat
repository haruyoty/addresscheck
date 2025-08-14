@echo off
chcp 65001 >nul
title ゴルフアドレス診断ツール (WSL版)

echo ====================================
echo    ⛳ ゴルフアドレス診断ツール ⛳
echo    (WSL環境版)
echo ====================================
echo.
echo 初期化中...

REM 現在のディレクトリをWindowsパスに設定
cd /d "%~dp0"

REM WSLでPythonコマンドを実行
echo 📋 WSL環境でPython環境を確認中...

REM WSLコマンドでPythonの確認
wsl python3 --version >nul 2>&1
if errorlevel 1 (
    echo ❌ エラー: WSL内でPython3がインストールされていません
    echo.
    echo 以下のコマンドをWSLで実行してください:
    echo sudo apt update
    echo sudo apt install python3 python3-pip
    echo.
    pause
    exit /b 1
)

echo ✅ WSL Python環境を確認しました
echo.

REM WSLパスを取得（/mnt/c/... 形式）
for %%I in (.) do set "WIN_PATH=%%~fI"
set "WSL_PATH=/mnt/c%WIN_PATH:~2%"
set "WSL_PATH=%WSL_PATH:\=/%"

echo Windowsパス: %WIN_PATH%
echo WSLパス: %WSL_PATH%
echo.

REM WSL環境での依存関係インストール確認
if not exist ".dependencies_installed_wsl" (
    echo 📦 初回実行: WSL環境で必要なライブラリをインストール中...
    echo （少し時間がかかります）
    echo.
    
    wsl bash -c "cd '%WSL_PATH%' && pip3 install -r requirements.txt"
    if errorlevel 1 (
        echo ❌ エラー: ライブラリのインストールに失敗しました
        echo.
        echo 以下を確認してください:
        echo - WSL内でのインターネット接続
        echo - requirements.txtファイルの存在
        echo - pip3の権限設定
        echo.
        echo 手動で以下を実行してください:
        echo wsl bash -c "cd '%WSL_PATH%' && pip3 install -r requirements.txt"
        echo.
        pause
        exit /b 1
    )
    
    REM インストール完了フラグファイルを作成
    echo. > .dependencies_installed_wsl
    echo ✅ WSL環境でのライブラリインストールが完了しました
    echo.
)

echo 🚀 WSL環境でアプリを起動中...
echo.
echo ブラウザが自動で開かない場合は、以下のURLにアクセスしてください:
echo http://localhost:8501
echo.
echo ⚠️  アプリを終了するには、このウィンドウでCtrl+Cを押してください
echo.

REM WSL環境でStreamlitアプリを起動
wsl bash -c "cd '%WSL_PATH%' && python3 -m streamlit run app.py --browser.gatherUsageStats false"

echo.
echo アプリが終了しました
pause