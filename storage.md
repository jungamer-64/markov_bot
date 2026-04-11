# Discord Markov Bot 保存フォーマット仕様 v6

## 概要

このドキュメントは `markov_bot` の学習済み Markov 連鎖を保存する `.mkv3` バイナリ形式の実装仕様を定義する。

v6 の目的は次の 3 点である。

- order-6 / order-5 / order-4 / order-3 / order-2 / order-1 の固定 backoff モデルを同梱すること
- reader / writer / validator が同じ section 定義を共有できること
- 読み取り専用の完成形データとして高速に検証・復元できること

旧 v5 以前との互換性は持たない。reader は v6 のみを受理する。

---

## 基本方針

- 学習中の `MarkovChain` は更新効率を優先した構造を使う
- 保存時に cumulative 形式の読み取り専用レコード群へ変換する
- 保存ファイルには order-6 から order-1 までの 6 モデルを同梱する
- section 配置は `Header + SectionDescriptor[] + aligned section bodies` とする
- section body は canonical order で 1 回ずつのみ出現する

---

## 前処理互換性

保存ファイルは token 化前の前処理仕様にも依存する。

v6 では header に以下を保持し、reader / writer の一致を強制する。

- `tokenizer_version`
- `normalization_flags`

現在の実装値は以下で固定である。

- `tokenizer_version = 1`
- `normalization_flags = 0`

---

## ファイル全体構造

ファイルは以下の順で並ぶ。

1. `Header`
2. `SectionDescriptor[20]`
3. 8-byte aligned metadata padding
4. section bodies

section bodies 自体は descriptor が指す位置に配置される。各 section の `offset` は必ず 8-byte aligned でなければならない。

セクション種別は固定で、descriptor は次の canonical order だけを許可する。

1. `VocabOffsets`
2. `VocabBlob`
3. `Starts`
4. `Model6Pairs`
5. `Model6Prefixes`
6. `Model6Edges`
7. `Model5Pairs`
8. `Model5Prefixes`
9. `Model5Edges`
10. `Model4Pairs`
11. `Model4Prefixes`
12. `Model4Edges`
13. `Model3Pairs`
14. `Model3Prefixes`
15. `Model3Edges`
16. `Model2Pairs`
17. `Model2Prefixes`
18. `Model2Edges`
19. `Model1Prefixes`
20. `Model1Edges`

---

## Header

header は固定長 44 bytes で、little-endian とする。

```rust
struct Header {
    magic: [u8; 8],              // b"MKV3BIN\0"
    version: u32,                // 6
    flags: u32,                  // vocab blob compression flags
    tokenizer_version: u32,      // 1
    normalization_flags: u32,    // 0
    section_count: u32,          // 20
    file_size: u64,              // ファイル全体サイズ
    checksum: u64,               // FNV-1a 64-bit
}
```

### Header flags

`flags` は現在 `VocabBlob` の圧縮方式のみを表す。

- `0x0000_0001`: RLE
- `0x0000_0002`: Zstd
- `0x0000_0004`: LZ4 (lz4_flex block format)

0 または上記いずれか 1 つのみ許可する。複数同時指定は不正。

---

## SectionDescriptor

descriptor は固定長 24 bytes で、little-endian とする。

```rust
struct SectionDescriptor {
    kind: u32,   // SectionKind
    flags: u32,  // 現在は常に 0
    offset: u64, // section body start
    size: u64,   // section byte length
}
```

### SectionKind

```text
1  = VocabOffsets
2  = VocabBlob
3  = Starts
4  = Model6Pairs
5  = Model6Prefixes
6  = Model6Edges
7  = Model5Pairs
8  = Model5Prefixes
9  = Model5Edges
10 = Model4Pairs
11 = Model4Prefixes
12 = Model4Edges
13 = Model3Pairs
14 = Model3Prefixes
15 = Model3Edges
16 = Model2Pairs
17 = Model2Prefixes
18 = Model2Edges
19 = Model1Prefixes
20 = Model1Edges
```

`flags` は v6 では常に 0 でなければならない。

---

## 語彙セクション

### VocabOffsets

- `u64` 配列
- 長さは `token_count + 1`
- 先頭値は必ず `0`
- 非減少列でなければならない
- 最終要素が `VocabBlob` の復号後サイズを表す

### VocabBlob

- token UTF-8 bytes を連結した blob
- `Header.flags` に従って plain / RLE / Zstd / LZ4 で格納される
- 復号後サイズは `VocabOffsets.last()` と一致しなければならない

語彙復元後、以下を満たす必要がある。

- token id 0 は `<BOS>`
- token id 1 は `<EOS>`
- 重複 token は禁止

---

## レコードセクション

すべて little-endian。固定サイズ section は `descriptor.size % record_size == 0` を満たさなければならない。

### Starts (12 bytes)

```rust
struct StartRecord {
    prefix_id: u32,
    cumulative: u64,
}
```

- `prefix_id` は `Model6Prefixes` の index
- `cumulative` は strictly increasing

### Model6Pairs (28 bytes)

```rust
struct Pair6Record {
    w1: u32,
    w2: u32,
    w3: u32,
    w4: u32,
    w5: u32,
    prefix_start: u32,
    prefix_len: u32,
}
```

### Model6Prefixes (20 bytes)

```rust
struct Prefix6Record {
    w6: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}
```

### Model6Edges / Model5Edges / Model4Edges / Model3Edges / Model2Edges / Model1Edges

```rust
struct EdgeRecord {
    next: u32,
    cumulative: u64,
}
```

### Model5Pairs (24 bytes)

```rust
struct Pair5Record {
    w1: u32,
    w2: u32,
    w3: u32,
    w4: u32,
    prefix_start: u32,
    prefix_len: u32,
}
```

### Model5Prefixes (20 bytes)

```rust
struct Prefix5Record {
    w5: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}
```

### Model4Pairs (20 bytes)

```rust
struct Pair4Record {
    w1: u32,
    w2: u32,
    w3: u32,
    prefix_start: u32,
    prefix_len: u32,
}
```

### Model4Prefixes (20 bytes)

```rust
struct Prefix4Record {
    w4: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}
```

### Model3Pairs (16 bytes)

```rust
struct Pair3Record {
    w1: u32,
    w2: u32,
    prefix_start: u32,
    prefix_len: u32,
}
```

### Model3Prefixes (20 bytes)

```rust
struct Prefix3Record {
    w3: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}
```

### Model2Pairs (12 bytes)

```rust
struct Pair2Record {
    w1: u32,
    prefix_start: u32,
    prefix_len: u32,
}
```

### Model2Prefixes (24 bytes)

```rust
struct Prefix2Record {
    w1: u32,
    w2: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}
```

### Model1Prefixes (20 bytes)

```rust
struct Prefix1Record {
    w1: u32,
    edge_start: u32,
    edge_len: u32,
    total: u64,
}
```

---

## 検証規則

reader は少なくとも以下を検証する。

- `magic == b"MKV3BIN\0"`
- `version == 6`
- `section_count == 20`
- `tokenizer_version == 1`
- `normalization_flags == 0`
- `file_size` が実サイズと一致する
- `checksum` が一致する
- descriptor の `kind` が canonical order で 1 回ずつ現れる
- descriptor `flags == 0`
- section `offset` が 8-byte aligned
- metadata end より前に section が始まらない
- sections が overlap しない
- gap にある padding bytes はすべて 0
- 最終 section end が `file_size` と一致する
- fixed-size section が record size の整数倍
- token id / prefix range / edge range が妥当
- cumulative 値が strictly increasing
- prefix `total` が edge の最終 cumulative と一致する

---

## writer の要件

- writer は canonical order で descriptor を生成する
- metadata end と各 section start は 8-byte aligned にする
- section 間の gap は 0 埋めする
- checksum は checksum field 自体を 0 とみなした FNV-1a 64-bit で計算する
- `save_chain` は encode 後に `load_chain` 相当の decode を実行し、自己検証に成功した payload のみ書き出す

---

## 互換性ポリシー

- v6 reader は v5 以前を読まない
- 旧 `.mkv3` データは再生成前提とする
- 互換 shim / 移行用 dual reader / one-shot converter は持たない
