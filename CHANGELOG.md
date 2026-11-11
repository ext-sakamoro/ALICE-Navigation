# Changelog

この crate の主な変更を記録する
書式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に沿い、版は [Semantic Versioning](https://semver.org/lang/ja/) に従う

## [Unreleased]

### Changed

- 依存 `alice-det-math` を rev 固定 (0.5.2) から `branch = "main"` + `version = "0.6"` に、`alice-physics` (feature `sdf-field`) を rev 固定から `branch = "main"` + `version = "4"` に (alice-sdf と同じ出所で、link する det-math は 1 つ) velocity の候補の hash (`tests/determinism_oracle.rs` の golden) は不変

### Added

- `graph_search`: 位置を持つ任意の graph (`SearchGraph`) に 1 つの決定論の A* (`astar` / `astar_filtered` / `hierarchical_astar` / `path_cost` / `cluster_grid` / `build_occupancy_navmesh`) 辺の費用は直線距離以上が契約 (debug build で検査)、算術は `+ − × ÷ √` だけ
- `sdf_reach` (feature `sdf-field`): 符号付き距離場 (`alice_physics::SdfField`) の上で縦向きの capsule が届くかを保守的に判定し、証拠の経路を返す
- `clippy.toml`: platform の libm の超越関数と `hypot` の呼び出しを禁止 (`mul_add` は正しく丸められるので許可)
- 依存 `alice-det-math` (角度の計算)

### Changed
- CI: wasm32 の job の wasmtime を v49.0.2 に固定し、配布物を SHA-256 (a4d6e9e3…、release の digest と手元の再計算で一致) で照合して retry 付きで取り、binary を cache する 配布元が落ち続けた時は同じ版の wasmtime-cli を crates.io から `cargo install --locked` で作る `wasmtime --version` が動かなければ fail (setup action は latest を選び、配布元の HTTP 504 で他の repository の job が落ちた) `scripts/preflight.sh` を生成器で生成し直した (main の版は公開文面の検査など後から足した CI の step を含んでいなかった)

- **Breaking:** `NavMesh::find_path` は三角形の重心の最短の列を `graph_search::astar` で求める (以前は別の Dijkstra) 長さの同じ経路が複数ある時は同点の規則 (f、g、三角形の番号の順、同じ g なら小さい番号の親) で 1 つに決まり、以前と違う列 (同じ長さの別の経路) を返すことがある
- **Breaking:** `Vec2::length` / `Vec2::distance_to` は `+ × √` だけで計算する (以前は `f64::hypot`、platform の libm で target により最終 bit が違いうる) どの platform でも同じ bit になる代わりに、値が最終 bit で以前と違うことがあり、経路の長さ・RRT・PRM・平滑化など距離を使う結果に及ぶ
- `Vec2` の距離と長さ (と `graph_search::distance`) は途中で overflow も underflow もしない 最大成分が `[2^-500, 2^500]` の内では `(x * x + y * y).sqrt()` と同じ bit、外では全成分を同じ 2 の冪で正確に拡大縮小してから同じ式で求める 誤差は通常の大きさでの式と同じで、指数の全域 (2D / 3D、両端) を掃いた測定で double-double の参照から 1 ulp 以内、subnormal の長さは subnormal の格子で 1 ulp 以内 真の長さが `f64::MAX` を超える時だけ無限大 NaN の成分は NaN、無限大の成分は +∞
- **Behavior change:** `velocity_obstacles::generate_velocity_candidates` の角度を `alice-det-math` で計算する (以前は platform の `sin` / `cos`) 値が最終 bit で以前と違うことがある
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る
