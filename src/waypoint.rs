//! Waypoint following (`WaypointFollower`).

use crate::core_types::Vec2;

// 6. Waypoint Following
// ============================================================

/// State for a waypoint follower.
#[derive(Debug, Clone)]
pub struct WaypointFollower {
    waypoints: Vec<Vec2>,
    current_index: usize,
    reach_threshold: f64,
    pub looping: bool,
}

impl WaypointFollower {
    #[must_use]
    pub const fn new(waypoints: Vec<Vec2>, reach_threshold: f64) -> Self {
        Self {
            waypoints,
            current_index: 0,
            reach_threshold,
            looping: false,
        }
    }

    /// Get the current target waypoint.
    #[must_use]
    pub fn current_target(&self) -> Option<Vec2> {
        self.waypoints.get(self.current_index).copied()
    }

    /// Advance the follower given the agent's current position.
    /// Returns the steering direction or `None` if finished.
    pub fn update(&mut self, pos: Vec2) -> Option<Vec2> {
        let target = self.current_target()?;
        if pos.distance_to(target) <= self.reach_threshold {
            self.current_index += 1;
            if self.current_index >= self.waypoints.len() {
                if self.looping {
                    self.current_index = 0;
                } else {
                    return None;
                }
            }
        }
        let target = self.waypoints[self.current_index];
        Some(target.sub(pos).normalized())
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        !self.looping && self.current_index >= self.waypoints.len()
    }

    #[must_use]
    pub const fn current_index(&self) -> usize {
        self.current_index
    }

    pub const fn reset(&mut self) {
        self.current_index = 0;
    }

    #[must_use]
    pub const fn waypoint_count(&self) -> usize {
        self.waypoints.len()
    }
}
