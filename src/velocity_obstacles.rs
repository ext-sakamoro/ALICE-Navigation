//! Velocity Obstacles (`VoAgent` / `is_in_velocity_obstacle` / `select_velocity_outside_vo` / `generate_velocity_candidates`).

use crate::core_types::Vec2;

// 7. Velocity Obstacles (VO)
// ============================================================

/// Agent for velocity-obstacle computation.
#[derive(Debug, Clone, Copy)]
pub struct VoAgent {
    pub pos: Vec2,
    pub vel: Vec2,
    pub radius: f64,
}

impl VoAgent {
    #[must_use]
    pub const fn new(pos: Vec2, vel: Vec2, radius: f64) -> Self {
        Self { pos, vel, radius }
    }
}

/// Check if a candidate velocity is inside the velocity obstacle
/// induced by `other` on `agent`.
#[must_use]
pub fn is_in_velocity_obstacle(agent: &VoAgent, other: &VoAgent, candidate_vel: Vec2) -> bool {
    let rel_pos = other.pos.sub(agent.pos);
    let combined_radius = agent.radius + other.radius;
    let dist = rel_pos.length();
    if dist < combined_radius {
        return true; // Already colliding.
    }

    let rel_vel = candidate_vel.sub(other.vel);
    // Check if relative velocity points toward the VO cone.
    let proj = rel_vel.dot(rel_pos.normalized());
    if proj <= 0.0 {
        return false; // Moving away.
    }

    // Perpendicular distance from relative velocity to the line connecting agents.
    let perp = (rel_vel.cross(rel_pos)).abs() / dist;
    perp < combined_radius
}

/// Select the best velocity from candidates that avoids all velocity obstacles.
/// Falls back to zero velocity if all are blocked.
#[must_use]
pub fn select_velocity_outside_vo(
    agent: &VoAgent,
    others: &[VoAgent],
    preferred_vel: Vec2,
    candidates: &[Vec2],
) -> Vec2 {
    let mut best = Vec2::new(0.0, 0.0);
    let mut best_cost = f64::INFINITY;

    for &cand in candidates {
        let blocked = others
            .iter()
            .any(|other| is_in_velocity_obstacle(agent, other, cand));
        if blocked {
            continue;
        }
        let cost = cand.sub(preferred_vel).length();
        if cost < best_cost {
            best_cost = cost;
            best = cand;
        }
    }
    best
}

/// Generate candidate velocities in a disc pattern.
#[must_use]
pub fn generate_velocity_candidates(
    max_speed: f64,
    num_rings: usize,
    num_angles: usize,
) -> Vec<Vec2> {
    let mut candidates = vec![Vec2::new(0.0, 0.0)];
    for ring in 1..=num_rings {
        let speed = max_speed * (ring as f64) / (num_rings as f64);
        for a in 0..num_angles {
            let angle = 2.0 * std::f64::consts::PI * (a as f64) / (num_angles as f64);
            candidates.push(Vec2::new(speed * angle.cos(), speed * angle.sin()));
        }
    }
    candidates
}
