# WSL環境用 PowerShellショートカット作成スクリプト

Write-Host "====================================" -ForegroundColor Green
Write-Host "  ⛳ ゴルフアドレス診断ツール" -ForegroundColor Green  
Write-Host "  デスクトップショートカット作成" -ForegroundColor Green
Write-Host "====================================" -ForegroundColor Green
Write-Host ""

# 現在のスクリプトのパスを取得
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatFile = Join-Path $ScriptDir "start_golf_address_tool.bat"

# デスクトップパスを取得
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "⛳ ゴルフアドレス診断ツール.lnk"

Write-Host "スクリプト場所: $ScriptDir" -ForegroundColor Yellow
Write-Host "バッチファイル: $BatFile" -ForegroundColor Yellow
Write-Host "デスクトップ: $DesktopPath" -ForegroundColor Yellow
Write-Host ""

# バッチファイルの存在確認
if (!(Test-Path $BatFile)) {
    Write-Host "❌ エラー: start_golf_address_tool.bat が見つかりません" -ForegroundColor Red
    Write-Host "場所: $BatFile" -ForegroundColor Red
    Read-Host "Enterキーを押して終了..."
    exit 1
}

# ショートカットを作成
try {
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = $BatFile
    $Shortcut.WorkingDirectory = $ScriptDir
    $Shortcut.Description = "ゴルフアドレス診断ツール - MediaPipeを使用したゴルフ姿勢解析アプリ"
    $Shortcut.IconLocation = "%SystemRoot%\System32\shell32.dll,137"
    $Shortcut.Save()
    
    Write-Host "✅ デスクトップショートカットを作成しました！" -ForegroundColor Green
    Write-Host ""
    Write-Host "ショートカット名: ⛳ ゴルフアドレス診断ツール" -ForegroundColor Cyan
    Write-Host "場所: $DesktopPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "🚀 使用方法:" -ForegroundColor Yellow
    Write-Host "デスクトップの「⛳ ゴルフアドレス診断ツール」アイコンをダブルクリック" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host "❌ ショートカット作成に失敗しました" -ForegroundColor Red
    Write-Host "エラー: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
    Write-Host "手動でショートカットを作成してください:" -ForegroundColor Yellow
    Write-Host "1. デスクトップを右クリック" -ForegroundColor White
    Write-Host "2. 新規作成 → ショートカット" -ForegroundColor White
    Write-Host "3. 参照で以下のファイルを選択:" -ForegroundColor White
    Write-Host "   $BatFile" -ForegroundColor Cyan
    Write-Host "4. 名前を「⛳ ゴルフアドレス診断ツール」に設定" -ForegroundColor White
}

Write-Host ""
Read-Host "Enterキーを押して終了..."