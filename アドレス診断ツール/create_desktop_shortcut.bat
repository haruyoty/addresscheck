@echo off
chcp 65001 >nul
title デスクトップショートカット作成

echo ====================================
echo   デスクトップショートカット作成
echo   (WSL環境対応版)
echo ====================================
echo.

REM WSL環境でのWindowsパスを取得
set "CURRENT_DIR=%~dp0"

REM WSLでのWindowsデスクトップパスを取得 
set "DESKTOP_PATH=%USERPROFILE%\Desktop"

REM WSL環境の場合、/mnt/c/ パスを C:\ に変換
if "%CURRENT_DIR:~0,7%"=="/mnt/c/" (
    set "CURRENT_DIR=C:\%CURRENT_DIR:~7%"
)

REM パス内の / を \ に置換
set "CURRENT_DIR=%CURRENT_DIR:/=\%"

echo デスクトップパス: %DESKTOP_PATH%
echo アプリパス: %CURRENT_DIR%

REM ショートカット作成用のVBScriptファイルを一時作成
set "VBS_FILE=%TEMP%\create_shortcut.vbs"

echo Set WshShell = WScript.CreateObject("WScript.Shell") > "%VBS_FILE%"
echo Set oShellLink = WshShell.CreateShortcut("%DESKTOP_PATH%\⛳ ゴルフアドレス診断ツール.lnk") >> "%VBS_FILE%"
echo oShellLink.TargetPath = "%CURRENT_DIR%start_golf_address_tool.bat" >> "%VBS_FILE%"
echo oShellLink.WorkingDirectory = "%CURRENT_DIR%" >> "%VBS_FILE%"
echo oShellLink.Description = "ゴルフアドレス診断ツール - MediaPipeを使用したゴルフ姿勢解析アプリ" >> "%VBS_FILE%"
echo oShellLink.IconLocation = "%SystemRoot%\System32\shell32.dll,137" >> "%VBS_FILE%"
echo oShellLink.Save >> "%VBS_FILE%"

REM VBScriptを実行してショートカットを作成
cscript //nologo "%VBS_FILE%"

REM 一時ファイルを削除
del "%VBS_FILE%"

if exist "%DESKTOP_PATH%\⛳ ゴルフアドレス診断ツール.lnk" (
    echo.
    echo ✅ デスクトップショートカットを作成しました！
    echo.
    echo ショートカット名: ⛳ ゴルフアドレス診断ツール
    echo 場所: %DESKTOP_PATH%
    echo.
    echo デスクトップのアイコンをダブルクリックするとアプリが起動します。
) else (
    echo.
    echo ❌ ショートカットの作成に失敗しました
    echo 手動で以下の手順を実行してください:
    echo.
    echo 1. デスクトップを右クリック
    echo 2. 「新規作成」→「ショートカット」を選択
    echo 3. 参照ボタンで start_golf_address_tool.bat を選択
    echo 4. 名前を「⛳ ゴルフアドレス診断ツール」に設定
)

echo.
pause