@echo off
chcp 65001 >nul
title ゴルフアドレス診断ツール

echo ====================================
echo    ⛳ ゴルフアドレス診断ツール ⛳
echo ====================================
echo.
echo 初期化中...

REM 現在のディレクトリをバッチファイルの場所に設定
cd /d "%~dp0"

REM Pythonがインストールされているかチェック
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ エラー: Pythonがインストールされていません
    echo.
    echo Pythonをインストールしてください:
    echo https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

REM pipがインストールされているかチェック
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ エラー: pipがインストールされていません
    echo.
    pause
    exit /b 1
)

echo ✅ Python環境を確認しました
echo.

REM 初回実行時のみ依存関係をインストール
if not exist ".dependencies_installed" (
    echo 📦 初回実行: 必要なライブラリをインストール中...
    echo （少し時間がかかります）
    echo.
    
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ エラー: ライブラリのインストールに失敗しました
        echo.
        echo 以下を確認してください:
        echo - インターネット接続
        echo - requirements.txtファイルの存在
        echo - Pythonの権限設定
        echo.
        pause
        exit /b 1
    )
    
    REM インストール完了フラグファイルを作成
    echo. > .dependencies_installed
    echo ✅ ライブラリのインストールが完了しました
    echo.
)

echo 🚀 アプリを起動中...
echo.
echo ブラウザが自動で開かない場合は、以下のURLにアクセスしてください:
echo http://localhost:8501
echo.
echo ⚠️  アプリを終了するには、このウィンドウでCtrl+Cを押してください
echo.

REM Streamlitアプリを起動（ブラウザを自動で開く）
streamlit run app.py --browser.gatherUsageStats false

echo.
echo アプリが終了しました
pause