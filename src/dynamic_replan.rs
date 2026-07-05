//! Dynamic replanning (`DynamicPlanner`).

use crate::collision::segment_free;
use crate::core_types::{Bounds2D, CircleObstacle, Vec2};
use crate::rrt::{rrt, RrtConfig};

// 9. Dynamic Replanning
// ============================================================

/// A dynamic planner that replans when obstacles change.
#[derive(Debug, Clone)]
pub struct DynamicPlanner {
    pub start: Vec2,
    pub goal: Vec2,
    pub bounds: Bounds2D,
    pub obstacles: Vec<CircleObstacle>,
    pub current_path: Option<Vec<Vec2>>,
    rrt_config: RrtConfig,
}

impl DynamicPlanner {
    #[must_use]
    pub fn new(start: Vec2, goal: Vec2, bounds: Bounds2D, obstacles: Vec<CircleObstacle>) -> Self {
        Self {
            start,
            goal,
            bounds,
            obstacles,
            current_path: None,
            rrt_config: RrtConfig::default(),
        }
    }

    /// Set custom RRT config.
    pub const fn set_rrt_config(&mut self, config: RrtConfig) {
        self.rrt_config = config;
    }

    /// Plan (or replan) a path.
    pub fn plan(&mut self) -> bool {
        self.current_path = rrt(
            self.start,
            self.goal,
            self.bounds,
            &self.obstacles,
            &self.rrt_config,
        );
        self.current_path.is_some()
    }

    /// Add a new obstacle and replan if the current path is invalidated.
    pub fn add_obstacle(&mut self, obstacle: CircleObstacle) -> bool {
        self.obstacles.push(obstacle);
        if let Some(ref path) = self.current_path {
            let invalidated = path
                .windows(2)
                .any(|w| obstacle.intersects_segment(w[0], w[1]));
            if invalidated {
                return self.plan();
            }
            return true;
        }
        self.plan()
    }

    /// Remove obstacles that match a predicate and optionally replan.
    pub fn remove_obstacles_where<F: Fn(&CircleObstacle) -> bool>(&mut self, pred: F) {
        self.obstacles.retain(|o| !pred(o));
    }

    /// Update the agent position (for replanning from a new start).
    pub const fn update_start(&mut self, new_start: Vec2) {
        self.start = new_start;
    }

    /// Check if the current path is still valid.
    #[must_use]
    pub fn is_path_valid(&self) -> bool {
        self.current_path.as_ref().is_some_and(|path| {
            path.windows(2)
                .all(|w| segment_free(w[0], w[1], &self.obstacles))
        })
    }
}
