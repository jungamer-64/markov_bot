# 運用ガイド

この文書は bot の起動・設定・保存ファイル運用をまとめた開発者向けの実務メモです。環境変数の挙動は `.env.example` と `src/config.rs` を正本とします。

## 起動手順

1. `.env` を作成します。

   ```bash
   cp .env.example .env
   ```

2. `DISCORD_TOKEN` を設定します。必要に応じて他の変数も調整します。
3. bot を起動します。

   ```bash
   cargo run
   ```

4. 対象チャンネルを Discord 上で設定します。

   ```text
   /set_channel
   ```

5. 対象チャンネルに投稿された通常ユーザーメッセージから学習と返信が始まります。

## 環境変数

| 変数 | 必須 | 既定値 | 制約 / canonical 値 | 用途 |
| --- | --- | --- | --- | --- |
| `DISCORD_TOKEN` | 必須 | なし | 空文字不可 | Discord bot token |
| `MARKOV_DATA_PATH` | 任意 | `data/markov_chain.mkv3` | `PathBuf` として解釈できること | 学習済みモデルの保存先 |
| `MARKOV_NGRAM_ORDER` | 任意 | `6` | `>= 1` かつ `u32` に収まること | 学習・生成・保存に使う n-gram 次数 |
| `STORAGE_COMPRESSION` | 任意 | `auto` | `auto`, `none`, `rle`, `zstd`, `lz4_flex` | `VocabBlob` 圧縮方式 |
| `STORAGE_MIN_EDGE_COUNT` | 任意 | `1` | `>= 1` | 保存時に残す最小 edge count |
| `REPLY_MAX_WORDS` | 任意 | `20` | `>= 1` | 1 返信あたりの最大 token 数 |
| `REPLY_TEMPERATURE` | 任意 | `1.0` | 有限かつ `> 0` | 返信生成時の温度 |
| `REPLY_MIN_WORDS_BEFORE_EOS` | 任意 | `0` | `<= REPLY_MAX_WORDS` | EOS を許可するまでの最小 token 数 |
| `REPLY_COOLDOWN_SECS` | 任意 | `5` | `u64` | 返信クールダウン秒数 |

`STORAGE_COMPRESSION` は parser 側で `off`, `uncompressed`, `vocab_rle`, `lz4` などの別名も受理しますが、設定ファイルでは表の canonical 値を使うのが前提です。

## 起動時と保存時の挙動

- `.env` は起動時に自動読込されます。shell で `export` しなくても、リポジトリ root の `.env` があれば反映されます。
- `MARKOV_DATA_PATH` が存在しない場合、空の `MarkovChain` が作られます。
- 保存先ディレクトリが存在しない場合は自動作成されます。
- 保存済みファイルの `ngram_order` が `MARKOV_NGRAM_ORDER` と違う場合、起動は失敗します。
- 保存は空でない token 列を学習した直後に毎回実行されます。
- `STORAGE_MIN_EDGE_COUNT` は永続化時のフィルタです。閾値未満の edge は保存されず、再起動後のモデルにも戻りません。
- 対象チャンネル ID は保存されません。プロセスを再起動したら `/set_channel` を再実行してください。

## `markov-storage` CLI の使い方

リポジトリ内から使う canonical な呼び出し方は次のとおりです。

### inspect

保存ファイルを自動判別して summary を表示します。v6 / v8 の両方を受理します。

```bash
cargo run -p markov-storage-cli -- inspect --input data/markov_chain.mkv3
```

### export

保存ファイルを `StorageSnapshot` JSON に書き出します。v6 / v8 の両方を受理します。

```bash
cargo run -p markov-storage-cli -- export \
  --input data/markov_chain.mkv3 \
  --output /tmp/markov_snapshot.json
```

### import

`StorageSnapshot` JSON を v8 `.mkv3` に変換します。出力 format は常に v8 です。

```bash
cargo run -p markov-storage-cli -- import \
  --input /tmp/markov_snapshot.json \
  --output data/markov_chain.mkv3
```

### migrate

v6 `.mkv3` を v8 `.mkv3` に変換します。v8 入力は拒否されます。

```bash
cargo run -p markov-storage-cli -- migrate \
  --input ./legacy.mkv3 \
  --output ./migrated.mkv3
```

`import` と `migrate` は in-place 更新を禁止しており、`--input` と `--output` は必ず別パスである必要があります。

## 典型的な作業フロー

### 保存内容を確認したい

```bash
cargo run -p markov-storage-cli -- inspect --input data/markov_chain.mkv3
```

### JSON で差分確認したい

```bash
cargo run -p markov-storage-cli -- export \
  --input data/markov_chain.mkv3 \
  --output /tmp/markov_snapshot.json
```

### `STORAGE_MIN_EDGE_COUNT` を変えて再保存したい

1. `.env` の `STORAGE_MIN_EDGE_COUNT` を変更します。
2. bot を再起動します。
3. 対象チャンネルで新しい学習イベントを 1 回発生させると、新設定で保存されます。

保存 format 自体を理解したい場合は [storage-format.md](storage-format.md) を参照してください。
