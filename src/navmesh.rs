//! Navigation Mesh (`NavTriangle` / `NavMesh` / `build_grid_navmesh`).

use crate::core_types::{Bounds2D, Vec2};
use crate::prm::DijkNode;
use std::collections::BinaryHeap;

// 8. Navigation Mesh
// ============================================================

/// A triangle in the navigation mesh.
#[derive(Debug, Clone, Copy)]
pub struct NavTriangle {
    pub vertices: [Vec2; 3],
    pub neighbors: [Option<usize>; 3],
}

impl NavTriangle {
    #[must_use]
    pub const fn new(v0: Vec2, v1: Vec2, v2: Vec2) -> Self {
        Self {
            vertices: [v0, v1, v2],
            neighbors: [None, None, None],
        }
    }

    #[must_use]
    pub fn centroid(&self) -> Vec2 {
        Vec2::new(
            (self.vertices[0].x + self.vertices[1].x + self.vertices[2].x) / 3.0,
            (self.vertices[0].y + self.vertices[1].y + self.vertices[2].y) / 3.0,
        )
    }

    /// Check if a point is inside the triangle using barycentric coordinates.
    #[must_use]
    pub fn contains(&self, p: Vec2) -> bool {
        let v0 = self.vertices[2].sub(self.vertices[0]);
        let v1 = self.vertices[1].sub(self.vertices[0]);
        let v2 = p.sub(self.vertices[0]);

        let dot00 = v0.dot(v0);
        let dot01 = v0.dot(v1);
        let dot02 = v0.dot(v2);
        let dot11 = v1.dot(v1);
        let dot12 = v1.dot(v2);

        let inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;

        u >= -1e-9 && v >= -1e-9 && (u + v) <= 1.0 + 1e-9
    }

    /// Compute the area of the triangle.
    #[must_use]
    pub fn area(&self) -> f64 {
        let ab = self.vertices[1].sub(self.vertices[0]);
        let ac = self.vertices[2].sub(self.vertices[0]);
        ab.cross(ac).abs() * 0.5
    }
}

/// Navigation mesh.
#[derive(Debug, Clone)]
pub struct NavMesh {
    pub triangles: Vec<NavTriangle>,
}

impl NavMesh {
    #[must_use]
    pub const fn new(triangles: Vec<NavTriangle>) -> Self {
        Self { triangles }
    }

    /// Find which triangle contains the point.
    #[must_use]
    pub fn find_triangle(&self, p: Vec2) -> Option<usize> {
        self.triangles.iter().position(|t| t.contains(p))
    }

    /// Plan a path through the navmesh using A* on triangle centroids.
    #[must_use]
    pub fn find_path(&self, start: Vec2, goal: Vec2) -> Option<Vec<Vec2>> {
        let start_tri = self.find_triangle(start)?;
        let goal_tri = self.find_triangle(goal)?;

        if start_tri == goal_tri {
            return Some(vec![start, goal]);
        }

        // Dijkstra on triangle graph.
        let n = self.triangles.len();
        let mut dist = vec![f64::INFINITY; n];
        let mut prev = vec![usize::MAX; n];
        dist[start_tri] = 0.0;

        let mut heap = BinaryHeap::new();
        heap.push(DijkNode {
            cost: 0.0,
            index: start_tri,
        });

        while let Some(DijkNode { cost, index }) = heap.pop() {
            if index == goal_tri {
                break;
            }
            if cost > dist[index] {
                continue;
            }
            for &neighbor_opt in &self.triangles[index].neighbors {
                if let Some(neighbor) = neighbor_opt {
                    let edge_cost = self.triangles[index]
                        .centroid()
                        .distance_to(self.triangles[neighbor].centroid());
                    let new_cost = dist[index] + edge_cost;
                    if new_cost < dist[neighbor] {
                        dist[neighbor] = new_cost;
                        prev[neighbor] = index;
                        heap.push(DijkNode {
                            cost: new_cost,
                            index: neighbor,
                        });
                    }
                }
            }
        }

        if dist[goal_tri].is_infinite() {
            return None;
        }

        let mut tri_path = Vec::new();
        let mut cur = goal_tri;
        while cur != usize::MAX {
            tri_path.push(cur);
            cur = prev[cur];
        }
        tri_path.reverse();

        // Convert to waypoints through centroids.
        let mut path = vec![start];
        for &tri_idx in &tri_path[1..tri_path.len().saturating_sub(1)] {
            path.push(self.triangles[tri_idx].centroid());
        }
        path.push(goal);
        Some(path)
    }

    /// Check if a point is within the navigable area.
    #[must_use]
    pub fn is_navigable(&self, p: Vec2) -> bool {
        self.find_triangle(p).is_some()
    }

    /// Return total navigable area.
    #[must_use]
    pub fn total_area(&self) -> f64 {
        self.triangles.iter().map(NavTriangle::area).sum()
    }
}

/// Build a simple grid-based navmesh from bounds, subdivided into `cols` x `rows` cells.
///
/// Each cell is split into two triangles (lower-left and upper-right).
/// Neighbor connectivity is established between adjacent triangles.
#[must_use]
pub fn build_grid_navmesh(bounds: Bounds2D, cols: usize, rows: usize) -> NavMesh {
    let dx = bounds.width() / cols as f64;
    let dy = bounds.height() / rows as f64;
    let mut triangles = Vec::with_capacity(cols * rows * 2);

    // First pass: create all triangles with diagonal neighbor only.
    for r in 0..rows {
        for c in 0..cols {
            let x0 = bounds.min.x + c as f64 * dx;
            let y0 = bounds.min.y + r as f64 * dy;
            let x1 = x0 + dx;
            let y1 = y0 + dy;

            let bl = Vec2::new(x0, y0);
            let br = Vec2::new(x1, y0);
            let tl = Vec2::new(x0, y1);
            let tr = Vec2::new(x1, y1);

            // t0: lower-left triangle (bl, br, tl) — edges: bottom, right-diag, left
            let t0 = NavTriangle::new(bl, br, tl);
            // t1: upper-right triangle (br, tr, tl) — edges: right, top, left-diag
            let t1 = NavTriangle::new(br, tr, tl);

            triangles.push(t0);
            triangles.push(t1);
        }
    }

    // Second pass: wire up neighbors.
    for r in 0..rows {
        for c in 0..cols {
            let idx0 = (r * cols + c) * 2;
            let idx1 = idx0 + 1;

            // Diagonal neighbors within the cell.
            triangles[idx0].neighbors[0] = Some(idx1);
            triangles[idx1].neighbors[0] = Some(idx0);

            // t0's left edge neighbor: right triangle of left cell.
            if c > 0 {
                let left_idx1 = (r * cols + c - 1) * 2 + 1;
                triangles[idx0].neighbors[1] = Some(left_idx1);
                triangles[left_idx1].neighbors[1] = Some(idx0);
            }
            // t0's bottom edge neighbor: upper triangle of cell below.
            if r > 0 {
                let below_idx1 = ((r - 1) * cols + c) * 2 + 1;
                triangles[idx0].neighbors[2] = Some(below_idx1);
                triangles[below_idx1].neighbors[2] = Some(idx0);
            }
        }
    }

    NavMesh::new(triangles)
}
