#!/bin/bash

echo "===================================="
echo "  ⛳ ゴルフアドレス診断ツール"
echo "  デスクトップショートカット作成"
echo "  (WSL環境用)"
echo "===================================="
echo

# 現在のWSLパスを取得
CURRENT_WSL_PATH=$(pwd)
echo "現在のWSLパス: $CURRENT_WSL_PATH"

# WindowsパスをWSLパスから変換
WINDOWS_PATH=$(wslpath -w "$CURRENT_WSL_PATH")
echo "Windowsパス: $WINDOWS_PATH"

# バッチファイルのフルパス
BAT_FILE="$WINDOWS_PATH\\start_golf_wsl.bat"
echo "バッチファイル: $BAT_FILE"

# デスクトップパス取得（環境変数から）
DESKTOP_PATH="/mnt/c/Users/$(whoami)/Desktop"
if [ ! -d "$DESKTOP_PATH" ]; then
    DESKTOP_PATH="/mnt/c/Users/$USER/Desktop"
fi

if [ ! -d "$DESKTOP_PATH" ]; then
    # PowerShellで取得
    DESKTOP_WIN=$(powershell.exe -Command '[Environment]::GetFolderPath("Desktop")' 2>/dev/null | tr -d '\r')
    DESKTOP_PATH=$(wslpath "$DESKTOP_WIN" 2>/dev/null)
fi

echo "デスクトップパス: $DESKTOP_PATH"

# デスクトップディレクトリの存在確認
if [ ! -d "$DESKTOP_PATH" ]; then
    echo "❌ エラー: デスクトップフォルダが見つかりません"
    echo "手動でショートカットを作成してください"
    exit 1
fi

# ショートカット作成用のPowerShellコマンドを実行
SHORTCUT_PATH="$DESKTOP_PATH/⛳ ゴルフアドレス診断ツール.lnk"
SHORTCUT_WIN=$(wslpath -w "$SHORTCUT_PATH")

echo "ショートカットを作成中..."
echo

# PowerShellでショートカット作成
powershell.exe -Command "
\$WshShell = New-Object -ComObject WScript.Shell;
\$Shortcut = \$WshShell.CreateShortcut('$SHORTCUT_WIN');
\$Shortcut.TargetPath = '$BAT_FILE';
\$Shortcut.WorkingDirectory = '$WINDOWS_PATH';
\$Shortcut.Description = 'ゴルフアドレス診断ツール - MediaPipeを使用したゴルフ姿勢解析アプリ';
\$Shortcut.IconLocation = '%SystemRoot%\\System32\\shell32.dll,137';
\$Shortcut.Save();
Write-Host '✅ ショートカットを作成しました!' -ForegroundColor Green
"

# 結果確認
if [ -f "$SHORTCUT_PATH" ]; then
    echo
    echo "✅ デスクトップショートカットを作成しました！"
    echo
    echo "ショートカット名: ⛳ ゴルフアドレス診断ツール"
    echo "場所: $DESKTOP_PATH"
    echo
    echo "🚀 使用方法:"
    echo "デスクトップの「⛳ ゴルフアドレス診断ツール」アイコンをダブルクリック"
else
    echo
    echo "❌ ショートカット作成に失敗しました"
    echo
    echo "手動でショートカットを作成してください:"
    echo "1. デスクトップを右クリック"
    echo "2. 新規作成 → ショートカット"
    echo "3. 参照で以下のファイルを選択:"
    echo "   $BAT_FILE"
    echo "4. 名前を「⛳ ゴルフアドレス診断ツール」に設定"
fi

echo
read -p "Enterキーを押して終了..."