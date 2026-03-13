**English** | [日本語](README_JP.md)

# ALICE-Navigation

**ALICE Autonomous Navigation** — RRT, PRM, potential field, obstacle avoidance, path smoothing, waypoint following, velocity obstacles, navigation mesh, and dynamic replanning.

Part of [Project A.L.I.C.E.](https://github.com/anthropics/alice) ecosystem.

## Features

- **RRT (Rapidly-exploring Random Trees)** — Sampling-based path planning in obstacle-rich environments
- **PRM (Probabilistic Roadmap)** — Multi-query path planning with pre-built roadmaps
- **Potential Field** — Gradient-based reactive navigation
- **Obstacle Avoidance** — Real-time collision prevention
- **Path Smoothing** — Post-processing to remove jagged waypoints
- **Waypoint Following** — Sequential target tracking controller
- **Velocity Obstacles** — Dynamic obstacle avoidance in velocity space
- **Navigation Mesh** — Polygon-based walkable area representation
- **Dynamic Replanning** — On-the-fly path recalculation on environment changes

## Architecture

```
Vec2 (core 2D math)
 ├── length, distance, normalize
 ├── add, sub, scale, dot, cross
 └── lerp

Bounds2D (axis-aligned bounding box)

Planners
 ├── RRT → Tree expansion → Path extraction
 ├── PRM → Roadmap construction → A* query
 └── PotentialField → Attractive + Repulsive forces

PathProcessor
 ├── Smoothing (iterative averaging)
 └── Waypoint following (lookahead)

VelocityObstacle
 └── Dynamic agent avoidance

NavMesh
 └── Polygon-based navigation
```

## Quick Start

```rust
use alice_navigation::{Vec2, Bounds2D};

let start = Vec2::new(0.0, 0.0);
let goal = Vec2::new(10.0, 10.0);
let dist = start.distance_to(goal);
```

## License

MIT OR Apache-2.0
