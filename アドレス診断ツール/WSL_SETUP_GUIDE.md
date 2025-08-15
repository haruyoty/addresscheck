# 🏌️ ゴルフアドレス診断ツール - WSL環境セットアップガイド

## 📋 WSL環境での準備

### 1. WSL内でPythonのセットアップ
WSLターミナルで以下を実行：
```bash
# Pythonとpipのインストール
sudo apt update
sudo apt install python3 python3-pip python3-venv

# バージョン確認
python3 --version
pip3 --version
```

## 🚀 デスクトップアプリ化手順（WSL版）

### ステップ1: PowerShellでデスクトップショートカット作成

**Windows側で**以下のいずれかの方法を実行：

#### 方法1: PowerShell を管理者として実行
1. スタートメニュー → PowerShell を右クリック → **管理者として実行**
2. 以下のコマンドを実行：
```powershell
cd "C:\Users\ym-no\Documents\コーディング\アドレス診断ツール"
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\create_desktop_shortcut.ps1
```

#### 方法2: ファイルエクスプローラーから実行
1. `C:\Users\ym-no\Documents\コーディング\アドレス診断ツール` フォルダを開く
2. `create_desktop_shortcut.ps1` を右クリック
3. **PowerShell で実行** を選択

### ステップ2: デスクトップショートカットの作成を確認
デスクトップに「⛳ ゴルフアドレス診断ツール」のショートカットが作成されているか確認

### ステップ3: アプリ起動
デスクトップアイコンをダブルクリックすると：
1. **初回**: WSL環境で必要なライブラリを自動インストール
2. **2回目以降**: すぐにアプリが起動
3. ブラウザが自動で開きます

## 📁 WSL版ファイル構成

```
アドレス診断ツール/
│
├── app.py                          # メインアプリ
├── requirements.txt                # 必要ライブラリ一覧
├── start_golf_wsl.bat              # WSL用起動バッチファイル ★
├── create_desktop_shortcut.ps1     # PowerShell ショートカット作成 ★
├── WSL_SETUP_GUIDE.md              # このファイル
└── .dependencies_installed_wsl     # WSL用インストール済みフラグ
```

## 🔧 手動でのショートカット作成

PowerShellが使えない場合：

1. デスクトップを右クリック
2. **新規作成** → **ショートカット**
3. **参照**ボタンで `start_golf_wsl.bat` を選択
4. 名前を「⛳ ゴルフアドレス診断ツール」に設定

## ⚠️ WSL環境でのトラブルシューティング

### WSL内でPython3が見つからない場合
```bash
# WSLターミナルで実行
sudo apt update
sudo apt install python3 python3-pip
```

### ライブラリインストールエラー
```bash
# WSLターミナルで手動実行
cd /mnt/c/Users/ym-no/Documents/コーディング/アドレス診断ツール
pip3 install -r requirements.txt
```

### PowerShellの実行ポリシーエラー
```powershell
# PowerShellで実行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### WSLとWindows間でファイルが見えない場合
- WSL側: `/mnt/c/Users/ym-no/Documents/コーディング/アドレス診断ツール`
- Windows側: `C:\Users\ym-no\Documents\コーディング\アドレス診断ツール`

### ブラウザが開かない場合
手動で以下のURLにアクセス：
```
http://localhost:8501
```

## 🎯 使用方法

1. デスクトップの「⛳ ゴルフアドレス診断ツール」をダブルクリック
2. 初回は WSL 環境でライブラリをインストール（数分）
3. ブラウザでアプリが自動起動
4. アドレス写真をアップロード
5. 設定を選択（右打ち/左打ち、撮影方向、クラブ）
6. 自動解析結果を確認
7. CSV/PNG でデータダウンロード

## 📞 WSL環境での注意点

- **ネットワーク**: WSL内からインターネット接続が必要
- **ファイルパス**: Windows ⇔ WSL 間のパス変換を自動実行
- **権限**: 一部操作で管理者権限が必要な場合あり
- **パフォーマンス**: 初回のライブラリインストールに時間がかかります

WSLでの実行に問題がある場合は、WSLターミナルで直接以下を実行することも可能：
```bash
cd /mnt/c/Users/ym-no/Documents/コーディング/アドレス診断ツール
python3 -m streamlit run app.py
```