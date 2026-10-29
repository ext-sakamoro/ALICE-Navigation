//! One distance definition, a stated valid range, and pinned outputs that must be the same on every
//! platform (CI: Linux, macOS, Windows, wasm32-wasip1).

use alice_navigation::core_types::{Bounds2D, Vec2};
use alice_navigation::graph_search::{astar, distance, SearchGraph};
use alice_navigation::navmesh::build_grid_navmesh;
use alice_navigation::velocity_obstacles::generate_velocity_candidates;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[test]
fn vec2_distance_and_graph_distance_are_the_same_bits() {
    let mut g = Lcg(1);
    let mut compared = 0;
    for _ in 0..20_000 {
        let a = Vec2::new(g.next() * 200.0 - 100.0, g.next() * 200.0 - 100.0);
        let b = Vec2::new(g.next() * 200.0 - 100.0, g.next() * 200.0 - 100.0);
        assert_eq!(
            a.distance_to(b).to_bits(),
            distance([a.x, a.y, 0.0], [b.x, b.y, 0.0]).to_bits()
        );
        compared += 1;
    }
    // and on a navmesh, edge by edge
    let mesh = build_grid_navmesh(
        Bounds2D::new(Vec2::new(-3.3, 7.1), Vec2::new(11.9, 15.4)),
        7,
        5,
    );
    let mut nb = Vec::new();
    for a in 0..mesh.node_count() {
        nb.clear();
        mesh.neighbors(a, &mut nb);
        for &b in &nb {
            let d2 = mesh.triangles[a]
                .centroid()
                .distance_to(mesh.triangles[b].centroid());
            assert_eq!(d2.to_bits(), mesh.edge_cost(a, b).to_bits());
            compared += 1;
        }
    }
    assert!(compared > 20_000, "compared {compared}");
}

#[test]
fn vec2_length_is_exact_where_the_square_root_is() {
    // 3-4-5 and friends: x² + y² exact and a perfect square
    for (x, y, l) in [
        (3.0, 4.0, 5.0),
        (5.0, 12.0, 13.0),
        (-8.0, 15.0, 17.0),
        (0.0, -2.5, 2.5),
    ] {
        assert_eq!(Vec2::new(x, y).length(), l);
    }
    assert_eq!(Vec2::new(1.0, 1.0).length(), std::f64::consts::SQRT_2);
}

#[test]
fn vec2_length_is_finite_up_to_the_stated_range() {
    let edge = (f64::MAX / 2.0).sqrt() * 0.999_999;
    let l = Vec2::new(edge, -edge).length();
    assert!(l.is_finite(), "{l}");
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "overflows")]
fn vec2_length_beyond_the_range_is_rejected_in_debug() {
    let _ = Vec2::new(1e154, 1e154).length();
}

/// Exact triangle chains of find_path on fixed meshes (the stated tie rule picks these among
/// equal-length routes).
#[test]
fn find_path_chains_are_pinned() {
    let mesh = build_grid_navmesh(
        Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(6.0, 4.0)),
        6,
        4,
    );
    let chain = |a: Vec2, b: Vec2| {
        let (s, t) = (
            mesh.find_triangle(a).unwrap(),
            mesh.find_triangle(b).unwrap(),
        );
        astar(&mesh, s, t).unwrap()
    };
    assert_eq!(
        chain(Vec2::new(0.2, 0.3), Vec2::new(5.8, 3.6)),
        GOLDEN_CHAIN_A
    );
    assert_eq!(
        chain(Vec2::new(5.7, 0.2), Vec2::new(0.4, 3.9)),
        GOLDEN_CHAIN_B
    );
    assert_eq!(
        chain(Vec2::new(2.5, 1.5), Vec2::new(3.5, 2.5)),
        GOLDEN_CHAIN_C
    );
    // waypoints: start, the inner centroids, goal
    let p = mesh
        .find_path(Vec2::new(0.2, 0.3), Vec2::new(5.8, 3.6))
        .unwrap();
    assert_eq!(p.len(), GOLDEN_CHAIN_A.len());
    for (k, &t) in GOLDEN_CHAIN_A
        .iter()
        .enumerate()
        .skip(1)
        .take(GOLDEN_CHAIN_A.len() - 2)
    {
        let c = mesh.triangles[t].centroid();
        assert_eq!(
            (p[k].x.to_bits(), p[k].y.to_bits()),
            (c.x.to_bits(), c.y.to_bits())
        );
    }
}

const GOLDEN_CHAIN_A: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 18, 19, 30, 31, 32, 33, 34, 35, 46, 47,
];
const GOLDEN_CHAIN_B: &[usize] = &[10, 9, 20, 19, 30, 29, 40, 39, 38, 37];
const GOLDEN_CHAIN_C: &[usize] = &[16, 17, 28, 29, 30];

/// FNV-1a over find_path waypoints on many queries of an offset mesh with non-integer cells.
const GOLDEN_FIND_PATH_HASH: u64 = 0xcbed_26fa_b48e_ce23;

#[test]
fn find_path_waypoints_hash_is_platform_independent() {
    let mesh = build_grid_navmesh(
        Bounds2D::new(Vec2::new(-3.3, 7.1), Vec2::new(11.9, 15.4)),
        7,
        5,
    );
    let mut g = Lcg(9);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    };
    let mut compared = 0;
    for _ in 0..400 {
        let a = Vec2::new(-3.3 + g.next() * 15.2, 7.1 + g.next() * 8.3);
        let b = Vec2::new(-3.3 + g.next() * 15.2, 7.1 + g.next() * 8.3);
        if let Some(p) = mesh.find_path(a, b) {
            for w in p {
                eat(w.x.to_bits());
                eat(w.y.to_bits());
                compared += 1;
            }
        }
    }
    eprintln!("[determinism golden] find_path hash {h:#018x} over {compared} waypoints");
    assert!(compared > 0, "0 waypoints compared");
    assert_eq!(h, GOLDEN_FIND_PATH_HASH);
}

/// Velocity candidates use alice-det-math for the angles: the same bits on every platform.
const GOLDEN_VELOCITY_HASH: u64 = 0xe5f7_573e_94c9_dfd2;

#[test]
fn velocity_candidates_hash_is_platform_independent() {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut n = 0;
    for v in generate_velocity_candidates(2.5, 4, 37) {
        for b in
            v.x.to_bits()
                .to_le_bytes()
                .into_iter()
                .chain(v.y.to_bits().to_le_bytes())
        {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
        n += 1;
    }
    eprintln!("[determinism golden] velocity candidates hash {h:#018x} over {n}");
    assert_eq!(n, 1 + 4 * 37);
    assert_eq!(h, GOLDEN_VELOCITY_HASH);
}
