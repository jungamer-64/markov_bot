# markov_bot

Discord 上の日本語メッセージを学習し、Markov 連鎖で返信する Rust ワークスペースです。学習済みモデルは独自バイナリ形式 `.mkv3` として保存し、bot 本体とは分離した `markov-storage` CLI で検査・変換できます。

## ワークスペース構成

| package / crate | 役割 |
| --- | --- |
| `markov_bot` | Discord Gateway を受け取り、学習・返信・永続化を行う bot 本体 |
| `markov-core` | `MarkovChain`、学習ロジック、生成ロジック、`ngram_order` の妥当性検証 |
| `markov-storage` | v8 `.mkv3` の encode/decode、`StorageSnapshot` JSON 変換 |
| `markov-storage-cli` | `markov-storage` バイナリを提供し、保存ファイルの inspect / export / import / migrate を行う |

## 最短セットアップ

1. `.env` を作成します。

   ```bash
   cp .env.example .env
   ```

2. `.env` の `DISCORD_TOKEN` を bot token に置き換えます。必要なら `MARKOV_DATA_PATH` や返信・保存パラメータも調整します。
3. bot を起動します。

   ```bash
   cargo run
   ```

4. Discord 側で `/set_channel` を実行し、学習と返信の対象チャンネルを指定します。

bot は起動時点では対象チャンネル未設定です。対象チャンネル以外のメッセージ、bot 自身のメッセージ、他 bot のメッセージは学習対象にも返信対象にもなりません。

## `markov-storage` CLI

保存ファイルの中身を確認する最短コマンドです。

```bash
cargo run -p markov-storage-cli -- inspect --input data/markov_chain.mkv3
```

より詳しい運用フローは [docs/operations.md](docs/operations.md) を参照してください。

## 詳細ドキュメント

- [docs/architecture.md](docs/architecture.md): ワークスペース構成、データフロー、`ngram_order` と保存層の関係
- [docs/operations.md](docs/operations.md): 環境変数、起動手順、保存ファイル運用、`markov-storage` CLI 実用例
- [docs/storage-format.md](docs/storage-format.md): v8 `.mkv3` バイナリ形式仕様
