# Discord Markov Bot 保存フォーマット仕様 v1 草案

## 概要

このドキュメントは、Discord Bot 用のマルコフ連鎖学習データを保存するためのバイナリフォーマット仕様案を定義する。

本フォーマットは、主に以下を目的とする。

- 学習時に更新しやすい内部表現を別に持つこと
- 生成時に高速に読み取れる読み取り専用形式へ変換すること
- 短文チャットでも生成不能に陥りにくい backoff 構造を持つこと
- mmap に適した連続配置を採用すること
- 将来のバージョン拡張に耐えられること

基本方針として、**学習用データ** と **生成用データ** は分離する。

- 学習中: count ベースの更新しやすい構造
- 保存時: cumulative ベースの読み取り専用構造へ変換
- 生成時: cumulative 化済みファイルのみを使用

また、チャット用途では 1 語返信や短文が多いため、単一の高次モデルのみではなく、**複数次数のモデルを併存**させる。

---

## 設計方針

### 1. 学習用と生成用を分ける

学習中の構造は更新効率を優先し、保存時に生成向けへ変換する。

理由:

- cumulative 形式は生成時に高速
- cumulative 形式は部分更新に弱い
- count 形式は学習時の更新に向く
- 学習中と生成中で求められる最適化方向が異なる

そのため、本保存フォーマットは **読み取り専用の完成品** として設計する。

---

### 2. 高次モデルと低次モデルを併存させる

チャットでは以下のような短い応答が多い。

- 草
- 了解
- なるほど
- え？
- はい

このため、高次モデルだけでは文脈不足で遷移が見つからないケースが増える。

したがって、保存ファイルには以下の複数モデルを同梱する。

- order-3 モデル
- order-2 モデル
- order-1 モデル

生成時は以下の順で backoff する。

1. order-3 を試す
2. 見つからなければ order-2
3. さらに見つからなければ order-1
4. 最後は `<EOS>` または文頭分布へフォールバックする

---

### 3. 次数の定義を明確にする

本仕様では、**order-N モデル** を「直近 N 語を prefix として次語を予測するモデル」と定義する。

- order-3: `(t-2, t-1, t) -> next`
- order-2: `(t-1, t) -> next`
- order-1: `(t) -> next`

この定義により、backoff の説明と検索手順を一貫して記述できる。

---

## 用語

- **Token**: 分かち書き後の 1 単位
- **TokenId**: 語彙表中の token を指す整数 ID
- **Prefix**: 次語選択時の文脈
- **Edge**: ある prefix から遷移可能な次 token 候補
- **Count**: 学習中の生頻度
- **Cumulative**: 保存時の累積和
- **Total**: ある prefix に属する edge 群の最終 cumulative 値
- **Start distribution**: 文頭生成時に使用する開始分布

---

## トークン化方針

文字列はそのまま保持せず、すべて `TokenId` に変換する。

- `TokenId = u32`
- 語彙表は `TokenId <-> String` の対応を持つ

例:

- `0 = <BOS>`
- `1 = <EOS>`
- `2 = 今日は`
- `3 = 雨`
- `4 = です`

これにより、prefix や edge に文字列を重複して持たずに済む。

---

## 前処理互換性

保存ファイルは token 化前の前処理仕様にも依存するため、reader / writer 間の互換性を保つには、以下の差異を識別可能にする必要がある。

- tokenizer の種類
- Unicode 正規化方式
- メンションや URL の正規化有無
- 改行や句読点を token として保持するか
- 絵文字や顔文字の扱い
- repeated character の圧縮有無

そのため、ヘッダには少なくとも以下に相当する情報を持てるようにする。

- `tokenizer_version`
- `normalization_flags`

v1 では詳細仕様を固定しないが、**writer と reader は同一前処理系を前提とする**。

---

## 論理構造

保存ファイルは以下の論理要素から構成される。

- Header
- VocabOffsets
- VocabBlob
- StartRecords
- Model3PairRecords
- Model3PrefixRecords
- Model3EdgeRecords
- Model2PrefixRecords
- Model2EdgeRecords
- Model1PrefixRecords
- Model1EdgeRecords

### 各要素の役割

- Header: ファイル全体情報
- VocabOffsets: トークン文字列のオフセット表
- VocabBlob: UTF-8 文字列本体
- StartRecords: 文頭候補分布
- Model3PairRecords: order-3 の第 1 段索引
- Model3PrefixRecords: order-3 の完全 prefix 一覧
- Model3EdgeRecords: order-3 の遷移候補
- Model2PrefixRecords: order-2 の prefix 一覧
- Model2EdgeRecords: order-2 の遷移候補
- Model1PrefixRecords: order-1 の prefix 一覧
- Model1EdgeRecords: order-1 の遷移候補

---

## バイナリエンコーディング方針

### エンディアン

v1 では **little-endian 固定** とする。

### アラインメント

- すべてのセクション開始 offset は 8 バイト境界に揃えることを推奨する
- v1 reader は非整列でも読めてもよいが、writer は可能な限り 8 バイト境界へ揃えるべきである

### レコード表現

このドキュメント中の `struct` は **論理構造** を示すものであり、そのまま Rust のメモリ表現と一致することを必須とはしない。

実装では以下のどちらでもよい。

- バイト列を明示 decode する
- `#[repr(C)]` 等で ABI を固定し、十分な検証後に読む

ただし、移植性の観点からは前者が推奨である。

---

## Header

ファイル先頭には固定長ヘッダを置く。

```rust
struct Header {
    magic: [u8; 8],   // 例: b"MKV3BIN\0"
    version: u32,     // 破壊的変更時に更新
    flags: u32,       // 後方互換拡張用

    tokenizer_version: u32,
    normalization_flags: u32,

    token_count: u32,
    start_count: u32,

    model3_pair_count: u32,
    model3_prefix_count: u32,
    model3_edge_count: u32,

    model2_prefix_count: u32,
    model2_edge_count: u32,

    model1_prefix_count: u32,
    model1_edge_count: u32,

    vocab_offsets_offset: u64,
    vocab_blob_offset: u64,
    start_offset: u64,

    model3_pair_offset: u64,
    model3_prefix_offset: u64,
    model3_edge_offset: u64,

    model2_prefix_offset: u64,
    model2_edge_offset: u64,

    model1_prefix_offset: u64,
    model1_edge_offset: u64,

    file_size: u64,
    checksum: u64,    // v1 では 0 可。将来用予約でもよい
}
````

### 補足

`flags` は将来拡張用とする。

例:

- 64bit cumulative 使用フラグ
- 圧縮有無
- 予約領域利用有無

v1 では未知の `flags` を見た reader は **拒否** することを推奨する。

---

## 語彙表

語彙表は以下の 2 セクションで構成する。

### VocabOffsets

`u64 * (token_count + 1)` の配列。

各トークン文字列の開始位置を保持する。

### VocabBlob

すべてのトークン文字列を UTF-8 の連結バイト列として保持する。

`offsets[i]..offsets[i+1]` が token `i` の文字列本体となる。

---

## 文頭分布

文頭候補は cumulative 分布として保持する。

v1 では、開始候補は **order-3 prefix への参照** とする。

```rust
struct StartRecord {
    prefix_id: u32,
    cumulative: u32,
}
```

### 意図

- 開始時点で order-3 状態へ直接入れる
- triplet を重複保存しない
- 生成処理を統一できる

### 前提

短文開始を含め、文頭は `<BOS>` により埋められた正規化済み prefix として扱う。

例:

- 1 語目開始前: `(<BOS>, <BOS>, <BOS>)`
- 2 語目開始前: `(<BOS>, <BOS>, token1)`
- 3 語目開始前: `(<BOS>, token1, token2)`

---

## Model3

## 概要

order-3 モデルは以下の 2 段階索引で管理する。

1. `(w1, w2)` を引く
2. その範囲内で `w3` を引く

これにより、全 prefix を毎回二分探索するより高速に検索できる。

---

## Model3 PairRecords

```rust
struct Pair3Record {
    w1: u32,
    w2: u32,
    prefix_start: u32,
    prefix_len: u32,
}
```

### 役割

`(w1, w2)` に対応する `Prefix3Record` の範囲を示す。

---

## Model3 PrefixRecords

```rust
struct Prefix3Record {
    w3: u32,
    edge_start: u32,
    edge_len: u32,
    total: u32,
}
```

### 役割

- `w3` をキーに完全な 3 語 prefix を表現する
- 遷移候補の edge 範囲を持つ
- `total` は累積和の最終値を保持する

### 備考

`w1, w2` は `Pair3Record` 側に持たせるため、ここでは重複保持しない。

---

## Model3 EdgeRecords

```rust
struct EdgeRecord {
    next: u32,
    cumulative: u32,
}
```

### 役割

ある prefix に対する次トークン候補を cumulative 分布で保持する。

例:

生カウント:

- A: 3
- B: 5
- C: 2

保存時:

- A -> 3
- B -> 8
- C -> 10

この場合、合計重みは最後の `cumulative = 10` となる。

---

## Model2

## 概要

order-2 モデルは **直近 2 語** を prefix とする。

すなわち、order-3 が存在しない場合は `(t-1, t)` を用いて検索する。

---

## Model2 PrefixRecords

```rust
struct Prefix2Record {
    w1: u32,
    w2: u32,
    edge_start: u32,
    edge_len: u32,
    total: u32,
}
```

### 役割

- 2 語 prefix に対応する遷移候補の範囲を持つ
- `total` により乱択範囲を即時取得できる

---

## Model2 EdgeRecords

```rust
struct EdgeRecord {
    next: u32,
    cumulative: u32,
}
```

order-3 と同じ構造を用いる。

---

## Model1

## 概要

order-1 モデルは **直近 1 語** を prefix とする。

order-2 も失敗した場合の最終 backoff 先となる。

---

## Model1 PrefixRecords

```rust
struct Prefix1Record {
    w1: u32,
    edge_start: u32,
    edge_len: u32,
    total: u32,
}
```

---

## Model1 EdgeRecords

```rust
struct EdgeRecord {
    next: u32,
    cumulative: u32,
}
```

---

## ソート方針

### prefix の並び

- Model3 PairRecords: `(w1, w2)` の辞書順
- Model3 PrefixRecords: 各 pair 範囲内で `w3` 昇順
- Model2 PrefixRecords: `(w1, w2)` の辞書順
- Model1 PrefixRecords: `w1` 昇順

### edge の並び

v1 では `next` 昇順を採用する。

理由:

- 同じ学習結果から常に同じ保存結果が得られる
- デバッグしやすい
- cumulative + 二分探索では頻度順の利得が比較的小さい
- writer の実装が単純になる

---

## 学習時の内部表現

学習時は更新しやすい count ベース構造を持つ。

概念例:

```rust
HashMap<[u32; 3], HashMap<u32, u32>>
HashMap<[u32; 2], HashMap<u32, u32>>
HashMap<[u32; 1], HashMap<u32, u32>>

HashMap<String, u32>  // string -> token
Vec<String>          // token -> string

HashMap<[u32; 3], u32> // start count
```

ただし、これはあくまで概念説明であり、実装はこれに限定されない。

例えば以下でもよい。

- `HashMap<prefix, SmallVec<...>>`
- arena / slab ベース実装
- 一時ファイルへフラッシュしつつ後段でマージする方式
- ソート済み `Vec` を用いた集約方式

---

## 学習処理

1 文を読むたびに以下を更新する。

- order-3
- order-2
- order-1
- start distribution

例:

`<BOS> <BOS> <BOS> なるほど <EOS>`

から以下を積む。

- order-3: `(<BOS>, <BOS>, <BOS>) -> なるほど`
- order-3: `(<BOS>, <BOS>, なるほど) -> <EOS>`
- order-2: `(<BOS>, <BOS>) -> なるほど`
- order-2: `(<BOS>, なるほど) -> <EOS>`
- order-1: `(<BOS>) -> なるほど`
- order-1: `(なるほど) -> <EOS>`

start distribution には、文頭時点の正規化済み order-3 prefix を積む。

---

## 保存時の変換手順

count ベースから cumulative ベースへ変換する。

### 基本手順

1. 語彙表を確定する
2. 各次数の prefix を列挙する
3. prefix をソートする
4. 各 prefix の遷移 `next -> count` をソートする
5. 累積和を計算する
6. `EdgeRecord` を出力する
7. 最後の cumulative を `total` に保存する
8. 文頭候補も cumulative 化する
9. 各セクションの offset と count をヘッダへ記録する
10. 最後に整合性検査を行う

---

## Reader / Writer の責務

### Writer の責務

- count ベース構造から cumulative ベースへ変換する
- 各レコードを所定順序でソートする
- 重複 prefix / edge を統合する
- Header の count / offset / file_size を正しく埋める
- 書き出し後に整合性検査を行う

### Reader の責務

- Header を検証する
- 未対応 version / flags を拒否する
- 各 offset / count / 範囲整合性を確認する
- 不正ファイルを安全に拒否する
- 検証完了後のみ検索・サンプリング処理を行う

---

## 整合性制約

reader は少なくとも以下を検査すべきである。

### Header 全体

- `magic` が正しい
- `version` が対応範囲内
- `file_size` が実ファイル長と一致するか、または超過しない
- 各 `*_offset` が `file_size` 以下
- 各セクション範囲が相互に矛盾しない

### 語彙表

- `VocabOffsets.len() == token_count + 1`
- `offsets[0] == 0`
- `offsets[i] <= offsets[i + 1]`
- `offsets[token_count] == vocab_blob_size`
- すべての token 文字列範囲が `VocabBlob` 内に収まる

### StartRecords

- `start_count == StartRecords.len()`
- `start_count == 0` なら文頭生成不能として拒否してよい
- `prefix_id < model3_prefix_count`
- `cumulative` は strictly increasing
- 最終 `cumulative > 0`

### Prefix / Edge

- `edge_start + edge_len <= edge_count`
- `prefix_start + prefix_len <= prefix_count`
- `edge_len == 0` なら `total == 0`
- `edge_len > 0` なら最後の edge の `cumulative == total`
- `cumulative` は strictly increasing
- `total > 0` なら少なくとも 1 edge 存在
- 各 prefix キー列がソート済み
- 各 edge の `next` がソート済み

---

## 読み出し手順

### 文頭選択

1. `StartRecords` の最後の `cumulative` を total とする
2. `1..=total` の乱数を引く
3. `cumulative >= r` を満たす最初の要素を二分探索する
4. `prefix_id` から開始 prefix を取得する

### 1 ステップ生成

現在の文脈を `(a, b, c)` とする。

#### order-3

1. `(a, b)` を `Pair3Record` から二分探索
2. 該当範囲内で `c` を `Prefix3Record` から二分探索
3. edge 範囲を取得
4. `1..=total` で乱数を引く
5. edge 範囲内で `cumulative >= r` を二分探索
6. `next` を出力する

#### order-2

order-3 が見つからなければ `(b, c)` を `Prefix2Record` から探す。

#### order-1

order-2 も見つからなければ `c` を `Prefix1Record` から探す。

#### 最終フォールバック

order-1 も見つからなければ以下のいずれかを行う。

- `<EOS>` を返す
- 文頭分布へ戻す
- 外部の fail-safe 処理へ移譲する

v1 では挙動を実装側ポリシーに委ねるが、保存フォーマットとは独立に固定しておくのが望ましい。

---

## cumulative のビット幅

v1 では `cumulative` および `total` を `u32` とする。

### 利点

- 実装が単純
- 保存サイズが小さい
- 個人 Discord Bot 規模では十分なことが多い

### 注意点

- 長期運用や大規模学習では overflow の可能性がある
- writer は累積前に overflow を検査する必要がある

### 将来拡張

将来版では以下を検討する。

- `u64 cumulative`
- flag による可変ビット幅切り替え
- writer 側の rescale / prune

---

## 3 次だけ 2 段索引にする理由

v1 では order-3 のみ `(w1, w2)` による第 1 段索引を持つ。

理由:

- order-3 は prefix 数が最も増えやすい
- 全 order-3 prefix を直接二分探索するより高速になりやすい
- order-2 / order-1 は件数が比較的少なく、単純二分探索で十分な場合が多い
- 構造の複雑化を最小限に抑えられる

必要になれば将来版で order-2 にも補助索引を追加できる。

---

## 長所

- 学習用と生成用の責務が明確に分離される
- 保存ファイルを読み取り専用に最適化できる
- cumulative により生成時の選択が高速
- low-order backoff により短文チャットに強い
- mmap と相性が良い
- 語彙表と prefix / edge の分離で重複を抑えられる
- 検証項目を定義しやすく、壊れたファイルを拒否しやすい

---

## 短所

- 保存時に変換処理が必要
- 部分更新や追加学習には向かない
- 複数次数を持つ分、保存サイズは増える
- writer / reader 実装が多少長くなる
- tokenizer 仕様が変わると語彙互換性が崩れる

---

## versioning 方針

- `version` は破壊的変更時に更新する
- `flags` は後方互換拡張用に限定する
- 未知の `version` は reader が拒否する
- 未知の必須 `flags` は reader が拒否する
- 互換性が曖昧になる変更は無理に `flags` で吸収せず、明示的に `version` を上げる

---

## v1 で必須のもの

- count ベース学習 + cumulative ベース保存
- order-3 / order-2 / order-1 の併存
- `TokenId` 化された語彙表
- StartRecords
- order-3 の 2 段索引
- 各種 count / offset を持つ Header
- reader 側の基本整合性検査

---

## v1 で推奨のもの

- 8 バイト境界へのセクション整列
- checksum の導入
- tokenizer / normalization 識別情報の保持
- writer 完了後の自己検査
- ソート済み保証の検証

---

## 将来の改善候補

- checksum の厳密化
- セクションごとの整合性検査強化
- 64bit cumulative 対応
- 低頻度語の pruning
- 温度付きサンプリング
- alias method 導入の検討
- 文頭専用モデルの追加
- `<EOS>` 制御の改善
- 圧縮形式の追加
- order-2 への補助索引追加
- tokenizer 仕様の完全固定

---

## 結論

Discord のチャット用途を想定する場合、本保存フォーマットは以下の方針が適している。

- 学習時は count ベース
- 保存時は cumulative ベース
- 生成時は読み取り専用ファイルを使用
- order-3 を主モデルとし、order-2・order-1 を backoff 用に併存させる
- reader は検証後のみ生成に入る

この構成により、

- 長文では高次文脈の自然さ
- 短文では低次モデルの生存性
- 生成時の単純かつ高速な処理
- 仕様としての明確さと将来拡張性

を両立できる。
