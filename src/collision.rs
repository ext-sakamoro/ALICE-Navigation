//! Collision helpers (`point_free` / `segment_free`).

use crate::core_types::{CircleObstacle, Vec2};

// Collision helpers
// ============================================================

/// Returns `true` if a point is collision-free.
#[must_use]
pub fn point_free(p: Vec2, obstacles: &[CircleObstacle]) -> bool {
    obstacles.iter().all(|o| !o.contains(p))
}

/// Returns `true` if a segment is collision-free.
#[must_use]
pub fn segment_free(a: Vec2, b: Vec2, obstacles: &[CircleObstacle]) -> bool {
    obstacles.iter().all(|o| !o.intersects_segment(a, b))
}
