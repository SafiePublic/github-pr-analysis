## 概要
期間を指定してpull request の authorとreviewer の一覧を取得します。取得した結果を可視化して png 画像で出力します。

* 積み上げ縦棒グラフ
* ヒートマップ
* ネットワークグラフ
    * ノードの大きさ: PR作成数とレビューした数の合計
    * エッジの太さ: レビューした数と、レビューされた数の合計
* サンキーダイアグラム

## 使い方
Python 3.10.12 で動作確認済み。Linux で利用することを想定したコマンドを示します。

1. config.sample.py を config.py にコピーして、自身で発行した GitHub PAT に置換してください。Tokens (classic) で repo スコープにチェックを入れた場合の動作を確認しています。
```bash
cp config.sample.py config.py
```
2. 仮想環境を作成して必要なモジュールをインストールします（optional）
```bash
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
3. スクリプトを実行します
    * 詳細は `--help` オプションを参照してください
    * 100件までのPR検索結果に対応しています
```bash
python analyze.py  # Aggregate latest 1 month
python analyze.py -wn  # Aggregate latest 1 week
python analyze.py --from_date 2023-12-01 --to_date 2023-12-31  # Specify period
```
