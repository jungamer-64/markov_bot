# アーキテクチャ

このワークスペースは、Discord bot の実行系と Markov 連鎖の中核ロジック、保存形式、保守用 CLI を分離しています。実装変更時は責務の境界を崩さず、公開面を最小限に保つ前提です。

## 責務分担

| package / crate | 主な責務 |
| --- | --- |
| `markov_bot` | 環境変数読込、Discord 接続、メッセージ受信、学習トリガ、返信、保存 |
| `markov-core` | `MarkovChain`、学習 (`train_tokens`)、生成 (`generate_sentence_with_options`) |
| `markov-storage` | `.mkv3` v8 reader / writer、`StorageSnapshot` の JSON 変換、保存形式 validation |
| `markov-storage-cli` | v6 / v8 ファイルの inspect、JSON export / import、v6 から v8 への migrate |

## 起動シーケンス

1. `markov_bot` は起動時に `.env` を自動読込し、`BotConfig` を構築します。
2. Discord API から現在の bot user と application を取得し、`/set_channel` コマンドを登録します。
3. `DiscordHandler::new` が `MARKOV_DATA_PATH` を読み込みます。
   - ファイルが存在しない場合は空の `MarkovChain` を新規生成します。
   - ファイルは存在するが保存済み `ngram_order` が実行時設定と不一致の場合、起動は失敗します。
4. Gateway の `MessageCreate` と `InteractionCreate` を購読し、以後の学習と返信をイベント駆動で処理します。

## メッセージ処理フロー

1. `/set_channel` が実行されるまで、bot は対象チャンネルを持ちません。
2. 対象チャンネルに届いたユーザーメッセージだけが処理対象になります。bot 自身と他 bot の投稿は無視します。
3. `Tokenizer` はまず URL と Discord mention 風トークンを除去し、Lindera (`ipadic`) で形態素分割を試みます。
4. Lindera が使えない場合は Unicode word segmentation にフォールバックします。
5. 空でない token 列だけを `MarkovChain::train_tokens` に流し込み、学習後に `.mkv3` へ保存します。
6. 返信クールダウンが切れていれば、現在の chain を snapshot として複製し、`GenerationOptions` を使って返信文を生成します。生成できない場合は固定フォールバック文を返します。

## ランタイム状態

`DiscordHandler` が保持する共有状態は次の 3 つです。

- 学習済み `MarkovChain`
- 最後に返信した時刻
- 対象チャンネル ID

対象チャンネル ID はメモリ上のみで保持され、保存ファイルには書き込まれません。プロセス再起動後は毎回 `/set_channel` を再実行する必要があります。

## `ngram_order` と保存層の整合性

- `markov-core` は `ngram_order >= 1` を必須条件として扱い、学習時には `order = 1..=ngram_order` の全モデルを更新します。
- `markov-storage` は保存時に header へ `ngram_order` を明記し、`Starts` と各 `Model(order)` section をその値に応じて可変個数で出力します。
- 読込時は保存ファイルの `ngram_order` と runtime の `MARKOV_NGRAM_ORDER` が一致しない限り復元しません。

このため、`ngram_order` は単なる生成パラメータではなく、学習・保存・復元の全レイヤーを貫く構成値です。

## 永続化の考え方

- bot 本体は `markov-storage::encode_v8_chain` / `decode_v8_chain` だけを使い、過去 format を直接扱いません。
- v6 の読込互換は `markov-storage-cli` に閉じ込められており、bot 本体と `markov-storage` crate は v8 のみを前提にしています。
- `STORAGE_MIN_EDGE_COUNT` は保存時フィルタです。閾値未満の edge と、それに依存する start prefix は保存されないため、再起動後のモデルはこのフィルタ後の状態から再開します。

format の詳細は [storage-format.md](storage-format.md)、日常運用は [operations.md](operations.md) を参照してください。
