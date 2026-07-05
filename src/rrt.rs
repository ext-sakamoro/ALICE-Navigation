//! RRT (Rapidly-exploring Random Trees).

use crate::collision::segment_free;
use crate::core_types::{Bounds2D, CircleObstacle, Rng, Vec2};
use std::cmp::Ordering;

// 1. RRT (Rapidly-exploring Random Trees)
// ============================================================

/// Configuration for the RRT planner.
#[derive(Debug, Clone)]
pub struct RrtConfig {
    pub step_size: f64,
    pub max_iterations: usize,
    pub goal_threshold: f64,
    pub seed: u64,
}

impl Default for RrtConfig {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            max_iterations: 5000,
            goal_threshold: 0.5,
            seed: 42,
        }
    }
}

/// Plan a path with RRT.
///
/// Returns `None` if no path is found within the iteration limit.
///
/// # Panics
///
/// Will not panic in practice because the tree always has at least one node (the start).
#[must_use]
pub fn rrt(
    start: Vec2,
    goal: Vec2,
    bounds: Bounds2D,
    obstacles: &[CircleObstacle],
    config: &RrtConfig,
) -> Option<Vec<Vec2>> {
    let mut nodes: Vec<Vec2> = vec![start];
    let mut parents: Vec<usize> = vec![0];
    let mut rng = Rng::new(config.seed);

    for _ in 0..config.max_iterations {
        // Bias toward goal 10% of the time.
        let sample = if rng.next_f64() < 0.1 {
            goal
        } else {
            Vec2::new(
                rng.range(bounds.min.x, bounds.max.x),
                rng.range(bounds.min.y, bounds.max.y),
            )
        };

        // Find nearest node.
        let (nearest_idx, nearest) = nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.distance_to(sample)
                    .partial_cmp(&b.distance_to(sample))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();

        let dir = sample.sub(*nearest).normalized();
        let new_point = nearest.add(dir.scale(config.step_size));

        if !bounds.contains(new_point) {
            continue;
        }
        if !segment_free(*nearest, new_point, obstacles) {
            continue;
        }

        nodes.push(new_point);
        parents.push(nearest_idx);

        if new_point.distance_to(goal) <= config.goal_threshold {
            // Trace back path.
            let mut path = vec![goal, new_point];
            let mut idx = nodes.len() - 1;
            while idx != 0 {
                idx = parents[idx];
                path.push(nodes[idx]);
            }
            path.reverse();
            return Some(path);
        }
    }
    None
}
