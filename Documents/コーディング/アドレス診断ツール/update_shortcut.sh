#!/bin/bash

echo "========================================="
echo "  Golf Address Analysis Tool"
echo "  Update Desktop Shortcut"
echo "========================================="
echo

# Get current directory Windows path
WINDOWS_PATH=$(wslpath -w "$(pwd)")
BAT_FILE="$WINDOWS_PATH\\golf_app.bat"

# Desktop path
DESKTOP_PATH="/mnt/c/Users/ym-no/Desktop"

# Remove old shortcut if exists
OLD_SHORTCUT="$DESKTOP_PATH/ゴルフアドレス診断ツール.lnk"
if [ -f "$OLD_SHORTCUT" ]; then
    rm "$OLD_SHORTCUT"
    echo "Removed old shortcut"
fi

# Create new shortcut
NEW_SHORTCUT="$DESKTOP_PATH/Golf Address Tool.lnk"
SHORTCUT_WIN=$(wslpath -w "$NEW_SHORTCUT")

echo "Batch file: $BAT_FILE"
echo "Creating shortcut: $NEW_SHORTCUT"
echo

# Create shortcut using PowerShell
powershell.exe -Command "
\$WshShell = New-Object -ComObject WScript.Shell;
\$Shortcut = \$WshShell.CreateShortcut('$SHORTCUT_WIN');
\$Shortcut.TargetPath = '$BAT_FILE';
\$Shortcut.WorkingDirectory = '$WINDOWS_PATH';
\$Shortcut.Description = 'Golf Address Analysis Tool - Powered by MediaPipe';
\$Shortcut.IconLocation = '%SystemRoot%\\System32\\shell32.dll,137';
\$Shortcut.Save()
"

# Check result
if [ -f "$NEW_SHORTCUT" ]; then
    echo "✓ Desktop shortcut created successfully!"
    echo
    echo "Shortcut name: Golf Address Tool.lnk"
    echo "Location: $DESKTOP_PATH"
    echo
    echo "How to use:"
    echo "Double-click the 'Golf Address Tool' icon on desktop"
else
    echo "✗ Failed to create shortcut automatically"
    echo
    echo "Manual steps:"
    echo "1. Right-click on desktop"
    echo "2. New -> Shortcut"
    echo "3. Browse and select: $BAT_FILE"
    echo "4. Name it 'Golf Address Tool'"
fi

echo