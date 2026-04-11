# `.mkv3` 保存形式仕様 v8

この文書は `markov-storage` crate が読み書きする v8 `.mkv3` の単一正本です。対象は binary layout と validation ルールであり、bot の運用手順や CLI 利用法は扱いません。

## 概要

v8 の要点は次のとおりです。

- `ngram_order` を header に保存し、reader は runtime 設定と一致しないファイルを拒否する
- model は `order = ngram_order .. 1` の可変個数 section として保存する
- `Starts` は固定長 prefix 参照ではなく、`ngram_order` 長の prefix 自体を保存する
- reader が受理する storage version は v8 のみで、v7 以前とは互換性を持たない

## ファイル全体構造

ファイルは次の順で並びます。

1. `Header`
2. `SectionDescriptor[section_count]`
3. 8-byte aligned metadata padding
4. section bodies

section body の canonical order は常に次のとおりです。

1. `VocabOffsets`
2. `VocabBlob`
3. `Starts`
4. `Model(order = ngram_order)`
5. `Model(order = ngram_order - 1)`
6. `...`
7. `Model(order = 1)`

したがって `section_count = 3 + ngram_order` です。

## Header

header は little-endian の固定長 52 bytes です。

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

`flags` は `VocabBlob` 圧縮方式だけを表します。

- `0x0000_0001`: RLE
- `0x0000_0002`: Zstd
- `0x0000_0004`: LZ4 (`lz4_flex` block format)

`0` または上記いずれか 1 つのみ許可します。

## SectionDescriptor

descriptor は little-endian の固定長 24 bytes です。

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

- `VocabOffsets` / `VocabBlob` / `Starts` の `flags` は常に `0`
- `Model` の `flags` はその section が表す order
- descriptor は canonical order で 1 回ずつのみ出現する

## 語彙 section

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

語彙復元後は次を満たす必要があります。

- token id `0` は `<BOS>`
- token id `1` は `<EOS>`

## Starts section

body の先頭に `record_count: u32` を持ち、その後に `ngram_order` 長の固定レコードを並べます。

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

各 order ごとに 1 つの `Model` section を持ちます。body 先頭は次の header です。

```rust
struct ModelSectionHeader {
    record_count: u32,
    edge_count: u32,
}
```

その後に `ModelRecord[record_count]`、続けて `EdgeRecord[edge_count]` を並べます。

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

reader は少なくとも次を検証します。

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
- v7 以前の `.mkv3` はこの format と互換ではない
- runtime の `MARKOV_NGRAM_ORDER` と保存済み `ngram_order` が違う場合、復元は明示エラーで停止する

日常的な inspect / export / import / migrate の使い方は [operations.md](operations.md) を参照してください。
