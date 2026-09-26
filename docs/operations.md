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
| `STORAGE_COMPRESSION` | 任意 | `auto` | `auto`, `none`, `rle`, `zstd` | `VocabBlob` 圧縮方式 |
| `STORAGE_MIN_EDGE_COUNT` | 任意 | `1` | `>= 1` | 保存時に残す最小 edge count |
| `REPLY_MAX_WORDS` | 任意 | `20` | `>= 1` | 1 返信あたりの最大 token 数 |
| `REPLY_TEMPERATURE` | 任意 | `1.0` | 有限かつ `> 0` | 返信生成時の温度 |
| `REPLY_MIN_WORDS_BEFORE_EOS` | 任意 | `0` | `<= REPLY_MAX_WORDS` | EOS を許可するまでの最小 token 数 |
| `REPLY_COOLDOWN_SECS` | 任意 | `5` | `u64` | 返信クールダウン秒数 |

保存方針は `STORAGE_FAILURE_POLICY=exit|retry` で指定します。既定値・byte 上限・再試行間隔の設定例は `.env.example`、制約は `src/config.rs` と `StorageLimits` が正本です。

- `STORAGE_RETRY_INTERVAL_SECS`: 正の秒数。`retry` は通信がなくてもこの間隔で I/O 失敗を再試行します。
- `STORAGE_MAX_FILE_BYTES`: binary/JSON ファイルの入出力 byte 上限。
- `STORAGE_MAX_VOCAB_BYTES`: 展開済み語彙 blob の byte 上限。ファイル上限とは独立です。

上限は正の整数で、そのプロセスの `usize` に収まる必要があります。モデル全体のメモリ使用量を制限する設定ではありません。不正な設定は起動時に拒否します。

`STORAGE_COMPRESSION` は parser 側で `off`, `uncompressed`, `vocab_rle` などの別名も受理しますが、設定ファイルでは表の canonical 値を使うのが前提です。

## 起動時と保存時の挙動

- `.env` は起動時に自動読込されます。shell で `export` しなくても、リポジトリ root の `.env` があれば反映されます。
- `MARKOV_DATA_PATH` が存在しない場合、空の `MarkovChain` が作られます。
- 保存先ディレクトリが存在しない場合は自動作成されます。
- 保存済みファイルの `ngram_order` が `MARKOV_NGRAM_ORDER` と違う場合、起動は失敗します。
- 通常は空でない token 列を学習した直後に保存します。retry 待ちの間は追加学習をまとめ、設定した間隔で最新モデル全体を保存します。
- `STORAGE_MIN_EDGE_COUNT` は永続化時のフィルタです。閾値未満の edge は保存されず、再起動後のモデルにも戻りません。
- 対象チャンネル ID は保存されません。プロセスを再起動したら `/set_channel` を再実行してください。

## `markov-storage` CLI の使い方

リポジトリ内から使う canonical な呼び出し方は次のとおりです。

### inspect

保存ファイルを自動判別して summary を表示します。v8 を受理します。

```bash
cargo run -p markov-storage-cli -- inspect --input data/markov_chain.mkv3
```

### export

保存ファイルを `StorageSnapshot` JSON に書き出します。v8 を受理します。

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

`import` と `export` は入力と出力に別ファイルを要求します。同じパス、symlink、hard link による同一ファイル指定を拒否します。出力先を bot が使用中なら writer 競合で失敗します。稼働中モデルを import で置き換える場合は先に bot を停止してください。

全 command に `--max-file-bytes` と `--max-vocab-bytes` を指定できます。binary/JSON の入力と出力、展開後語彙に適用されます。JSON export は全文の生成とサイズ確認を完了してから出力を公開するため、失敗しても既存ファイルを途中の JSON に置き換えません。

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

## 保存保証と復旧

Linux・macOS・Windows のローカルファイルシステムを対象とします。OS と機器が同期要求を正しく実装すること、既存の親ディレクトリが永続化済みであること、保存先の階層を実行中に移動・入れ替えないことが前提です。ネットワーク FS、同期要求を無視するドライブ、媒体故障に対する保証ではありません。

Linux は一時ファイルの `sync_all`、同一ディレクトリでの rename、親ディレクトリの同期を使用します。macOS はファイルとディレクトリの [`F_FULLFSYNC`](https://docs.rs/rustix/latest/rustix/fs/fn.fsync.html) を要求します。Windows はファイル同期後に [write-through 置換](https://learn.microsoft.com/ja-jp/windows/win32/api/winbase/nf-winbase-movefileexw)を使用します。新規親ディレクトリも同期付きで公開します。同期機能が失敗した場合、保証を弱めて成功扱いにはしません。

保存先は一時ファイルで置換されます。symlink は開く時点の参照先へ解決し、そこを更新します。hard link の別名は置換後も古い inode を参照するため、ライブモデルの共有方法には使わないでください。新しいファイルは一時ファイルの権限を持ち、元ファイルの ACL・拡張属性等を引き継ぐ契約はありません。

- 公開前の失敗では旧ファイルが正本です。
- 公開済みで同期に失敗した場合、新版は見えますが電源断耐性は未確認です。
- 置換結果が不明な場合も、失敗したという理由だけで「旧版のまま」と判断しません。
- `exit` は失敗を報告して停止します。原因を解消し、CLI の inspect で正本を検査してから再起動してください。破損ファイルを自動で空モデルに置き換えません。
- `retry` は未保存をログへ出し、排他保持下で最新モデルを全量置換します。未保存中の学習は強制終了・電源断で失われ得ます。
- 正常終了・Ctrl-C・Unix の SIGTERM は受付停止後に受理済み処理を待ち、未保存なら最後の保存を試みます。最後の保存失敗は非ゼロ終了です。

`<保存先>.lock` は停止後も残ります。ロックの有効性はファイルの存在でなく OS の lease で決まり、プロセス停止時に解放されます。稼働中に sidecar を削除しないでください。`.lock` 拡張子は予約され、CLI 出力にも使用できません。

異常終了で残った `.markov-*` / `.markov-dir-*` は一時ファイルであり、起動時に復元元として採用しません。bot と CLI が停止していることを確認した上で不要な残骸を削除できます。保存済みモデルの世代バックアップは自動作成しません。

## 開発時の検証

workspace の検証手順は次のとおりです。Linux の CLI 障害テストには `strace` と子プロセスを trace できる実行環境が必要です。

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --all-features --locked
cargo build --workspace --all-targets --all-features --locked
```

CI では Linux・macOS・Windows で同じ手順を実行します。Linux は実際の CLI に syscall failure と強制終了を注入し、旧版／完全な新版の残存、失敗段階、再起動後の lease 再取得を検証します。通常のテストや SIGKILL は物理的な電源断の再現ではありません。電源断耐性の実測には、対象 FS・機器での電源断／VM ストレージ障害試験が別途必要です。
