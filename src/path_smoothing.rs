//! Path smoothing (`smooth_path` / `chaikin_smooth`).

use crate::collision::segment_free;
use crate::core_types::{CircleObstacle, Rng, Vec2};

// 5. Path Smoothing
// ============================================================

/// Smooth a path by iteratively removing redundant waypoints (shortcutting).
#[must_use]
pub fn smooth_path(path: &[Vec2], obstacles: &[CircleObstacle], iterations: usize) -> Vec<Vec2> {
    if path.len() < 3 {
        return path.to_vec();
    }

    let mut result = path.to_vec();
    let mut rng = Rng::new(777);

    for _ in 0..iterations {
        if result.len() < 3 {
            break;
        }
        let i = (rng.next_u64() as usize) % (result.len() - 2);
        let j = i + 2 + (rng.next_u64() as usize) % (result.len() - i - 2).max(1);
        let j = j.min(result.len() - 1);
        if j <= i + 1 {
            continue;
        }
        if segment_free(result[i], result[j], obstacles) {
            // Remove intermediate points.
            let mut new_path = Vec::with_capacity(result.len());
            new_path.extend_from_slice(&result[..=i]);
            new_path.extend_from_slice(&result[j..]);
            result = new_path;
        }
    }
    result
}

/// Smooth a path using Chaikin's corner-cutting subdivision.
#[must_use]
pub fn chaikin_smooth(path: &[Vec2], iterations: usize) -> Vec<Vec2> {
    if path.len() < 2 {
        return path.to_vec();
    }

    let mut result = path.to_vec();
    for _ in 0..iterations {
        let mut new_path = Vec::with_capacity(result.len() * 2);
        new_path.push(result[0]);
        for w in result.windows(2) {
            let q = w[0].lerp(w[1], 0.25);
            let r = w[0].lerp(w[1], 0.75);
            new_path.push(q);
            new_path.push(r);
        }
        new_path.push(result[result.len() - 1]);
        result = new_path;
    }
    result
}
