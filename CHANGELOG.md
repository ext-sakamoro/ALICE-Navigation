# Changelog

この crate の主な変更を記録する
書式は [Keep a Changelog](https://keepachangelog.com/ja/1.1.0/) に沿い、版は [Semantic Versioning](https://semver.org/lang/ja/) に従う

## [Unreleased]

### Added

- `graph_search`: 位置を持つ任意の graph (`SearchGraph`) に 1 つの決定論の A* (`astar` / `astar_filtered` / `hierarchical_astar` / `path_cost` / `cluster_grid` / `build_occupancy_navmesh`) 辺の費用は直線距離以上が契約 (debug build で検査)、算術は `+ − × ÷ √` だけ
- `sdf_reach` (feature `sdf-field`): 符号付き距離場 (`alice_physics::SdfField`) の上で縦向きの capsule が届くかを保守的に判定し、証拠の経路を返す
- `clippy.toml`: platform の libm の超越関数と `hypot` の呼び出しを禁止 (`mul_add` は正しく丸められるので許可)
- 依存 `alice-det-math` (角度の計算)

### Changed

- **Breaking:** `NavMesh::find_path` は三角形の重心の最短の列を `graph_search::astar` で求める (以前は別の Dijkstra) 長さの同じ経路が複数ある時は同点の規則 (f、g、三角形の番号の順、同じ g なら小さい番号の親) で 1 つに決まり、以前と違う列 (同じ長さの別の経路) を返すことがある
- **Breaking:** `Vec2::length` / `Vec2::distance_to` は `+ × √` だけで計算する (以前は `f64::hypot`、platform の libm で target により最終 bit が違いうる) どの platform でも同じ bit になる代わりに、値が最終 bit で以前と違うことがあり、経路の長さ・RRT・PRM・平滑化など距離を使う結果に及ぶ
- **Breaking:** `Vec2` の距離と長さの有効な範囲は、成分 (差) の大きさが `sqrt(f64::MAX / 2)` (約 9.48e153) まで 超えると無限大になり、debug build では panic する (以前の `hypot` は overflow を避けていた)
- **Behavior change:** `velocity_obstacles::generate_velocity_candidates` の角度を `alice-det-math` で計算する (以前は platform の `sin` / `cos`) 値が最終 bit で以前と違うことがある
