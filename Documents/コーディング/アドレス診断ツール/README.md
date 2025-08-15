# ⛳ ゴルフアドレス診断ツール

AIを使ったゴルフアドレス姿勢の自動診断ツールです。

## 🎯 機能
- 📸 写真から自動姿勢解析
- 🤖 MediaPipeによる骨格検出
- 📊 8項目の詳細評価
- 📈 レーダーチャート表示
- 📋 診断履歴管理

## 🚀 使い方
1. アドレス時の写真をアップロード
2. 設定を選択（右打ち/左打ち、撮影方向、クラブ）
3. 診断結果を確認

## 💻 ローカル実行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ 技術スタック
- Streamlit
- MediaPipe
- OpenCV
- NumPy
- Pandas
- Plotly

---
Made with ❤️ for golf improvement