# Discord Markov Bot 保存フォーマット仕様 v8

## 概要

`.mkv3` は学習済み `MarkovChain` を保存するための custom binary format である。  
v8 では固定 6 次前提を廃止し、`MARKOV_NGRAM_ORDER >= 1` を runtime 指定できる形へ全面抽象化した。

v8 の要点は次のとおり。

- `ngram_order` は header に保存し、reader は runtime 設定と一致しないファイルを拒否する
- model は `order = ngram_order .. 1` の可変個数 section として保存する
- `Starts` は固定 6-token 参照ではなく、`ngram_order` 長の prefix を直接保存する
- v7 以前との互換性は持たない。reader は v8 のみ受理する

## 実装配置

- bot 本体は root package `markov_bot` の binary のみを持つ
- `markov-core` crate が `MarkovChain` と生成・学習ロジックを持つ
- `markov-storage` crate が v8 の encode/decode と JSON 用 `StorageSnapshot` を持つ
- `markov-storage-cli` package が `markov-storage` バイナリを提供し、`inspect` / `export` / `import` / `migrate` を実装する
- v6 reader は `markov-storage-cli` の private module に閉じ込め、本体と `markov-storage` crate は過去 version を知らない

## CLI

`markov-storage` は次のサブコマンドを持つ。

- `inspect --input <path>`: v6 / v8 を自動判別して summary を表示する
- `export --input <path> --output <path>`: v6 / v8 を canonical JSON snapshot へ書き出す
- `import --input <json> --output <path>`: JSON snapshot を v8 `.mkv3` に変換する
- `migrate --input <v6> --output <path>`: v6 を v8 `.mkv3` へ変換する

`import` と `migrate` は in-place を許可しない。入力パスと出力パスは必ず分ける。

## 全体構造

ファイルは次の順で並ぶ。

1. `Header`
2. `SectionDescriptor[section_count]`
3. 8-byte aligned metadata padding
4. section bodies

section body の canonical order は常に次のとおり。

1. `VocabOffsets`
2. `VocabBlob`
3. `Starts`
4. `Model(order = ngram_order)`
5. `Model(order = ngram_order - 1)`
6. `...`
7. `Model(order = 1)`

したがって `section_count = 3 + ngram_order` である。

## Header

header は little-endian の固定長 52 bytes。

```rust
struct Header {
    magic: [u8; 8],              // b"MKV3BIN\0"
    version: u32,                // 8
    flags: u32,                  // vocab blob compression flags
    tokenizer_version: u32,      // 1
    normalization_flags: u32,    // 0
    ngram_order: u32,            // >= 1
    section_count: u64,          // 3 + ngram_order
    file_size: u64,              // ファイル全体サイズ
    checksum: u64,               // FNV-1a 64-bit
}
```

### Header flags

`flags` は `VocabBlob` 圧縮方式のみを表す。

- `0x0000_0001`: RLE
- `0x0000_0002`: Zstd
- `0x0000_0004`: LZ4 (`lz4_flex` block format)

0 または上記いずれか 1 つのみ許可する。

## SectionDescriptor

descriptor は little-endian の固定長 24 bytes。

```rust
struct SectionDescriptor {
    kind: u32,   // SectionKind
    flags: u32,  // Model のとき order、それ以外は 0
    offset: u64, // section body start
    size: u64,   // section byte length
}
```

### SectionKind

```text
1 = VocabOffsets
2 = VocabBlob
3 = Starts
4 = Model
```

- `VocabOffsets` / `VocabBlob` / `Starts` の `flags` は常に 0
- `Model` の `flags` はその section が表す order
- descriptor は canonical order で 1 回ずつのみ出現する

## 語彙セクション

### VocabOffsets

- `u64` 配列
- 長さは `token_count + 1`
- 先頭値は必ず `0`
- 非減少列
- 最終要素が `VocabBlob` 復号後サイズ

### VocabBlob

- token UTF-8 bytes を連結した blob
- `Header.flags` に従って plain / RLE / Zstd / LZ4 で保存される
- 復号後サイズは `VocabOffsets.last()` と一致しなければならない

語彙復元後は次を満たす必要がある。

- token id `0` は `<BOS>`
- token id `1` は `<EOS>`

## Starts section

body の先頭に `record_count: u32` を持ち、その後に `ngram_order` 長の固定レコードを並べる。

```rust
struct StartRecord {
    prefix: [u32; ngram_order],
    cumulative: u64,
}
```

- `prefix` は start prefix を直接保持する
- `cumulative` は strictly increasing
- record size は `ngram_order * 4 + 8`

## Model section

各 order ごとに 1 つの `Model` section を持つ。body 先頭は次の header。

```rust
struct ModelSectionHeader {
    record_count: u32,
    edge_count: u32,
}
```

その後に `ModelRecord[record_count]`、続けて `EdgeRecord[edge_count]` を並べる。

```rust
struct ModelRecord {
    prefix: [u32; order],
    edge_start: u32,
    edge_len: u32,
    total: u64,
}

struct EdgeRecord {
    next: u32,
    cumulative: u64,
}
```

- `prefix` はその order の prefix を直接保持する
- `edge_start` / `edge_len` は同一 section 内の `EdgeRecord` 配列を指す
- `EdgeRecord.cumulative` は各 prefix 内で strictly increasing
- `ModelRecord.total` はその prefix の最後の cumulative と一致しなければならない
- `ModelRecord` size は `order * 4 + 16`
- `EdgeRecord` size は 12 bytes

## Validation

reader は少なくとも次を検証する。

- `magic`, `version`, `tokenizer_version`, `normalization_flags`
- `ngram_order >= 1`
- header `section_count == 3 + ngram_order`
- descriptor canonical order と `Model.flags == order`
- 各 section の `offset` / `size` が file 範囲内で overlap しないこと
- `Starts` / `Model` body size が `record_count` / `edge_count` と一致すること
- prefix / edge が token range 内であること
- cumulative counts が strictly increasing であること
- `edge_start` / `edge_len` が有効範囲で連続していること
- checksum 一致

## 互換性

- v8 reader は v8 のみ受理する
- v7 以前の `.mkv3` は破棄して再学習する前提
- runtime の `MARKOV_NGRAM_ORDER` と保存済み `ngram_order` が違う場合、起動時に明示エラーで停止する
