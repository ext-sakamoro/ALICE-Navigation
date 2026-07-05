//! Potential field navigation (`PotentialFieldConfig` / `attractive_force` / `repulsive_force` / `potential_field`).

use crate::core_types::{CircleObstacle, Vec2};

// 3. Potential Field
// ============================================================

/// Configuration for the potential field planner.
#[derive(Debug, Clone)]
pub struct PotentialFieldConfig {
    pub attractive_gain: f64,
    pub repulsive_gain: f64,
    pub repulsive_range: f64,
    pub step_size: f64,
    pub max_iterations: usize,
    pub goal_threshold: f64,
}

impl Default for PotentialFieldConfig {
    fn default() -> Self {
        Self {
            attractive_gain: 1.0,
            repulsive_gain: 100.0,
            repulsive_range: 2.0,
            step_size: 0.1,
            max_iterations: 5000,
            goal_threshold: 0.3,
        }
    }
}

/// Compute attractive force toward goal.
#[must_use]
pub fn attractive_force(pos: Vec2, goal: Vec2, gain: f64) -> Vec2 {
    goal.sub(pos).scale(gain)
}

/// Compute repulsive force from a single obstacle.
#[must_use]
pub fn repulsive_force(pos: Vec2, obstacle: &CircleObstacle, gain: f64, range: f64) -> Vec2 {
    let d = pos.distance_to(obstacle.center) - obstacle.radius;
    if d <= 0.0 {
        // Inside obstacle: strong push away.
        let dir = pos.sub(obstacle.center).normalized();
        return dir.scale(gain * 10.0);
    }
    if d > range {
        return Vec2::new(0.0, 0.0);
    }
    let dir = pos.sub(obstacle.center).normalized();
    let magnitude = gain * (1.0 / d - 1.0 / range) * (1.0 / (d * d));
    dir.scale(magnitude)
}

/// Plan a path using the artificial potential field method.
#[must_use]
pub fn potential_field(
    start: Vec2,
    goal: Vec2,
    obstacles: &[CircleObstacle],
    config: &PotentialFieldConfig,
) -> Vec<Vec2> {
    let mut path = vec![start];
    let mut pos = start;

    for _ in 0..config.max_iterations {
        if pos.distance_to(goal) <= config.goal_threshold {
            path.push(goal);
            break;
        }

        let f_att = attractive_force(pos, goal, config.attractive_gain);
        let mut f_rep = Vec2::new(0.0, 0.0);
        for obs in obstacles {
            let fr = repulsive_force(pos, obs, config.repulsive_gain, config.repulsive_range);
            f_rep = f_rep.add(fr);
        }

        let total = f_att.add(f_rep);
        let dir = total.normalized();
        pos = pos.add(dir.scale(config.step_size));
        path.push(pos);
    }
    path
}
