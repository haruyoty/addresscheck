#!/bin/bash

echo "===================================="
echo "  ゴルフアドレス診断ツール"
echo "  デスクトップショートカット作成"
echo "===================================="
echo

# 現在のディレクトリのWindowsパス
WINDOWS_PATH=$(wslpath -w "$(pwd)")
BAT_FILE="$WINDOWS_PATH\\start_golf_wsl.bat"

# デスクトップパス
DESKTOP_PATH="/mnt/c/Users/ym-no/Desktop"
SHORTCUT_PATH="$DESKTOP_PATH/ゴルフアドレス診断ツール.lnk"
SHORTCUT_WIN=$(wslpath -w "$SHORTCUT_PATH")

echo "バッチファイル: $BAT_FILE"
echo "ショートカット作成場所: $SHORTCUT_PATH"
echo

# PowerShellでショートカット作成（シンプルな名前で）
powershell.exe -Command "
\$WshShell = New-Object -ComObject WScript.Shell;
\$Shortcut = \$WshShell.CreateShortcut('$SHORTCUT_WIN');
\$Shortcut.TargetPath = '$BAT_FILE';
\$Shortcut.WorkingDirectory = '$WINDOWS_PATH';
\$Shortcut.Description = 'ゴルフアドレス診断ツール';
\$Shortcut.IconLocation = '%SystemRoot%\\System32\\shell32.dll,137';
\$Shortcut.Save()
"

# 結果確認
if [ -f "$SHORTCUT_PATH" ]; then
    echo "✅ デスクトップショートカットを作成しました！"
    echo
    echo "ショートカット名: ゴルフアドレス診断ツール.lnk"
    echo "場所: $DESKTOP_PATH"
    echo
    echo "使用方法:"
    echo "デスクトップの「ゴルフアドレス診断ツール」アイコンをダブルクリック"
else
    echo "❌ 自動作成に失敗しました"
    echo
    echo "手動でショートカットを作成してください:"
    echo "1. デスクトップを右クリック"
    echo "2. 新規作成 → ショートカット" 
    echo "3. 参照で以下のファイルを選択:"
    echo "   $BAT_FILE"
    echo "4. 名前を「ゴルフアドレス診断ツール」に設定"
fi

echo