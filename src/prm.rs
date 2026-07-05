//! PRM (Probabilistic Roadmap) + Dijkstra helper.

use crate::collision::{point_free, segment_free};
use crate::core_types::{Bounds2D, CircleObstacle, Rng, Vec2};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// 2. PRM (Probabilistic Roadmap)
// ============================================================

/// Configuration for the PRM planner.
#[derive(Debug, Clone)]
pub struct PrmConfig {
    pub num_samples: usize,
    pub connection_radius: f64,
    pub seed: u64,
}

impl Default for PrmConfig {
    fn default() -> Self {
        Self {
            num_samples: 200,
            connection_radius: 2.0,
            seed: 123,
        }
    }
}

/// Plan a path with PRM.
#[must_use]
pub fn prm(
    start: Vec2,
    goal: Vec2,
    bounds: Bounds2D,
    obstacles: &[CircleObstacle],
    config: &PrmConfig,
) -> Option<Vec<Vec2>> {
    let mut rng = Rng::new(config.seed);
    let mut nodes = vec![start, goal];

    // Sample free-space points.
    for _ in 0..config.num_samples {
        let p = Vec2::new(
            rng.range(bounds.min.x, bounds.max.x),
            rng.range(bounds.min.y, bounds.max.y),
        );
        if point_free(p, obstacles) {
            nodes.push(p);
        }
    }

    let n = nodes.len();
    // Build adjacency with distances.
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = nodes[i].distance_to(nodes[j]);
            if d <= config.connection_radius && segment_free(nodes[i], nodes[j], obstacles) {
                adj[i].push((j, d));
                adj[j].push((i, d));
            }
        }
    }

    // Dijkstra from node 0 (start) to node 1 (goal).
    dijkstra_path(&nodes, &adj, 0, 1)
}

// ============================================================
// Dijkstra helper
// ============================================================

#[derive(PartialEq)]
pub(crate) struct DijkNode {
    pub(crate) cost: f64,
    pub(crate) index: usize,
}

impl Eq for DijkNode {}

impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

fn dijkstra_path(
    nodes: &[Vec2],
    adj: &[Vec<(usize, f64)>],
    start: usize,
    goal: usize,
) -> Option<Vec<Vec2>> {
    let n = nodes.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    dist[start] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(DijkNode {
        cost: 0.0,
        index: start,
    });

    while let Some(DijkNode { cost, index }) = heap.pop() {
        if index == goal {
            break;
        }
        if cost > dist[index] {
            continue;
        }
        for &(next, w) in &adj[index] {
            let new_cost = cost + w;
            if new_cost < dist[next] {
                dist[next] = new_cost;
                prev[next] = index;
                heap.push(DijkNode {
                    cost: new_cost,
                    index: next,
                });
            }
        }
    }

    if dist[goal].is_infinite() {
        return None;
    }

    let mut path = Vec::new();
    let mut cur = goal;
    while cur != usize::MAX {
        path.push(nodes[cur]);
        cur = prev[cur];
    }
    path.reverse();
    Some(path)
}
