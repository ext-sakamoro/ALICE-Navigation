//! Reachability over a distance field: a room with a wall and a doorway, walked by a capsule.
//!
//! `cargo run --example sdf_reach --features sdf-field`
use alice_navigation::sdf_reach::{reach_map, Clearance, ReachGrid};
use alice_physics::SdfField;

/// The floor (y <= 0) and a wall at z = 4 with a 1.8 m doorway around x = 5.
struct Room;

fn slab(p: [f32; 3], c: [f32; 3], h: [f32; 3]) -> f32 {
    let q = [
        (p[0] - c[0]).abs() - h[0],
        (p[1] - c[1]).abs() - h[1],
        (p[2] - c[2]).abs() - h[2],
    ];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
}

impl SdfField for Room {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        let p = [x, y, z];
        let left = slab(p, [2.05, 1.5, 4.0], [2.05, 1.5, 0.15]);
        let right = slab(p, [7.95, 1.5, 4.0], [2.05, 1.5, 0.15]);
        y.min(left).min(right)
    }
    fn normal(&self, _: f32, _: f32, _: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

fn main() {
    let grid = ReachGrid::floor([0.0, 0.0, 0.0], 0.2, 50, 40);
    let agent = Clearance::capsule(0.4, &[0.65, 1.0, 1.35]);
    let map = reach_map(&Room, &grid, [1.0, 0.0, 1.0], &agent).expect("valid grid and agent");
    println!(
        "free cells: {} of {}",
        map.free_cells(),
        map.checked_cells()
    );
    match map.witness([5.0, 0.0, 7.0]) {
        Some(path) => println!("reached through the doorway in {} cells", path.len()),
        None => println!("not reached (the doorway is narrower than the guaranteed width)"),
    }
}
