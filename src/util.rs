//! Additional utilities (`path_length` / `resample_path` / `closest_point_on_segment` / `point_segment_distance`).

use crate::core_types::Vec2;

// Additional utilities
// ============================================================

/// Compute the total length of a path.
#[must_use]
pub fn path_length(path: &[Vec2]) -> f64 {
    path.windows(2).map(|w| w[0].distance_to(w[1])).sum()
}

/// Resample a path so that waypoints are approximately `spacing` apart.
#[must_use]
pub fn resample_path(path: &[Vec2], spacing: f64) -> Vec<Vec2> {
    if path.is_empty() {
        return Vec::new();
    }
    let mut result = vec![path[0]];
    let mut accumulated = 0.0;

    for w in path.windows(2) {
        let seg_len = w[0].distance_to(w[1]);
        accumulated += seg_len;
        while accumulated >= spacing {
            accumulated -= spacing;
            let t = 1.0 - accumulated / seg_len;
            result.push(w[0].lerp(w[1], t));
        }
    }

    if let Some(&last) = path.last() {
        if result.last().is_none_or(|r| r.distance_to(last) > 1e-9) {
            result.push(last);
        }
    }
    result
}

/// Compute the closest point on a line segment to a given point.
#[must_use]
pub fn closest_point_on_segment(p: Vec2, a: Vec2, b: Vec2) -> Vec2 {
    let ab = b.sub(a);
    let ap = p.sub(a);
    let t = ap.dot(ab) / ab.dot(ab);
    let t = t.clamp(0.0, 1.0);
    a.add(ab.scale(t))
}

/// Distance from a point to a line segment.
#[must_use]
pub fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f64 {
    p.distance_to(closest_point_on_segment(p, a, b))
}
