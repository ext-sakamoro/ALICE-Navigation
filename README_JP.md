[English](README.md) | **日本語**

# ALICE-Navigation

**ALICE 自律ナビゲーション** — RRT、PRM、ポテンシャルフィールド、障害物回避、経路平滑化、ウェイポイント追従、速度障害物、ナビゲーションメッシュ、動的再計画。

[Project A.L.I.C.E.](https://github.com/anthropics/alice) エコシステムの一部。

## 機能

- **RRT（急速探索ランダムツリー）** — 障害物の多い環境でのサンプリングベース経路計画
- **PRM（確率的ロードマップ）** — 事前構築ロードマップによる複数クエリ経路計画
- **ポテンシャルフィールド** — 勾配ベースのリアクティブナビゲーション
- **障害物回避** — リアルタイム衝突防止
- **経路平滑化** — ジグザグウェイポイントの除去後処理
- **ウェイポイント追従** — 逐次目標追跡コントローラー
- **速度障害物** — 速度空間における動的障害物回避
- **ナビゲーションメッシュ** — ポリゴンベースの歩行可能領域表現
- **動的再計画** — 環境変化時のオンザフライ経路再計算

## アーキテクチャ

```
Vec2（2Dベクトル演算）
 ├── length, distance, normalize
 ├── add, sub, scale, dot, cross
 └── lerp

Bounds2D（軸整列バウンディングボックス）

Planners（計画器）
 ├── RRT → ツリー展開 → 経路抽出
 ├── PRM → ロードマップ構築 → A*クエリ
 └── PotentialField → 引力 + 斥力

PathProcessor（経路処理）
 ├── 平滑化（反復平均化）
 └── ウェイポイント追従（先読み）

VelocityObstacle
 └── 動的エージェント回避

NavMesh
 └── ポリゴンベースナビゲーション
```

## クイックスタート

```rust
use alice_navigation::{Vec2, Bounds2D};

let start = Vec2::new(0.0, 0.0);
let goal = Vec2::new(10.0, 10.0);
let dist = start.distance_to(goal);
```

## ライセンス

MIT OR Apache-2.0
