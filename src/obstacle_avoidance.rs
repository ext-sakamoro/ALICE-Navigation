//! Obstacle avoidance (`avoid_obstacles` / `is_near_obstacle`).

use crate::core_types::{CircleObstacle, Vec2};

// 4. Obstacle Avoidance
// ============================================================

/// Steer a velocity vector to avoid obstacles.
///
/// Returns the adjusted velocity. If no avoidance is needed, returns the input.
#[must_use]
pub fn avoid_obstacles(
    pos: Vec2,
    velocity: Vec2,
    obstacles: &[CircleObstacle],
    lookahead: f64,
    avoidance_strength: f64,
) -> Vec2 {
    let future_pos = pos.add(velocity.normalized().scale(lookahead));
    let mut steer = Vec2::new(0.0, 0.0);

    for obs in obstacles {
        let d = future_pos.distance_to(obs.center) - obs.radius;
        if d < lookahead {
            let away = future_pos.sub(obs.center).normalized();
            let strength = avoidance_strength * (1.0 - d / lookahead).max(0.0);
            steer = steer.add(away.scale(strength));
        }
    }

    velocity.add(steer)
}

/// Check if a position is within a certain margin of any obstacle.
#[must_use]
pub fn is_near_obstacle(pos: Vec2, obstacles: &[CircleObstacle], margin: f64) -> bool {
    obstacles
        .iter()
        .any(|o| pos.distance_to(o.center) - o.radius < margin)
}
