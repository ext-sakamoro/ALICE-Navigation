//! Oracles for `sdf_reach`. Expected values come from closed forms (exact plane and box distances,
//! the clearance bound of the module doc), never from the implementation under test.
#![cfg(feature = "sdf-field")]

use alice_navigation::sdf_reach::{reach, reach_map, Clearance, ReachError, ReachGrid, MAX_CELLS};
use alice_physics::SdfField;

const R: f32 = 0.4;
const HEIGHTS: [f32; 3] = [0.65, 1.0, 1.35];
const S: f32 = 0.35;
const C: f32 = 0.2;

/// sqrt(c²/2 + s²/4): the farthest a capsule-axis point over a cell is from a sample
fn offset(c: f32, s: f32) -> f32 {
    (c * c * 0.5 + s * s * 0.25).sqrt()
}

fn central_normal<F: Fn(f32, f32, f32) -> f32>(f: F, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
    let e = 1e-3;
    let g = [
        f(x + e, y, z) - f(x - e, y, z),
        f(x, y + e, z) - f(x, y - e, z),
        f(x, y, z + e) - f(x, y, z - e),
    ];
    let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt().max(1e-12);
    (g[0] / l, g[1] / l, g[2] / l)
}

/// Exact distance to the union of a floor (y <= 0) and axis-aligned boxes.
struct Boxes {
    boxes: Vec<([f32; 3], [f32; 3])>, // centre, half extents
}

fn box_distance(p: [f32; 3], c: [f32; 3], h: [f32; 3]) -> f32 {
    let q = [
        (p[0] - c[0]).abs() - h[0],
        (p[1] - c[1]).abs() - h[1],
        (p[2] - c[2]).abs() - h[2],
    ];
    let o = [q[0].max(0.0), q[1].max(0.0), q[2].max(0.0)];
    (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt() + q[0].max(q[1]).max(q[2]).min(0.0)
}

impl Boxes {
    fn d(&self, x: f32, y: f32, z: f32) -> f32 {
        self.boxes
            .iter()
            .fold(y, |m, (c, h)| m.min(box_distance([x, y, z], *c, *h)))
    }
}

impl SdfField for Boxes {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.d(x, y, z)
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        central_normal(|a, b, c| self.d(a, b, c), x, y, z)
    }
}

/// A straight corridor of clear width `w` whose walls are the planes `|n·p| = w/2` (n in the
/// floor plane), plus the floor. Exact: the distance to a union is the minimum distance.
struct Corridor {
    n: [f32; 2],
    w: f32,
}

impl Corridor {
    fn d(&self, x: f32, y: f32, z: f32) -> f32 {
        (self.w * 0.5 - (self.n[0] * x + self.n[1] * z).abs()).min(y)
    }
}

impl SdfField for Corridor {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        self.d(x, y, z)
    }
    fn normal(&self, x: f32, y: f32, z: f32) -> (f32, f32, f32) {
        central_normal(|a, b, c| self.d(a, b, c), x, y, z)
    }
}

/// The floor alone, raised to `floor`.
struct Floor(f32);
impl SdfField for Floor {
    fn distance(&self, _: f32, y: f32, _: f32) -> f32 {
        y - self.0
    }
    fn normal(&self, _: f32, _: f32, _: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

struct NanAt([f32; 2]);
impl SdfField for NanAt {
    fn distance(&self, x: f32, y: f32, z: f32) -> f32 {
        if (x - self.0[0]).abs() < 0.3 && (z - self.0[1]).abs() < 0.3 {
            f32::NAN
        } else {
            y
        }
    }
    fn normal(&self, _: f32, _: f32, _: f32) -> (f32, f32, f32) {
        (0.0, 1.0, 0.0)
    }
}

/// A small deterministic generator for the corpus (test data only).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// A 10 m × 10 m room with walls, a few pillars and one partial wall.
fn room(seed: u64) -> Boxes {
    let mut g = Lcg(seed);
    let mut boxes = vec![
        ([5.0, 1.5, -0.25], [5.5, 1.5, 0.25]),
        ([5.0, 1.5, 10.25], [5.5, 1.5, 0.25]),
        ([-0.25, 1.5, 5.0], [0.25, 1.5, 5.5]),
        ([10.25, 1.5, 5.0], [0.25, 1.5, 5.5]),
    ];
    for _ in 0..6 {
        // away from the start (1, 1) and the goal (9, 9) so both are free in every scene
        let (x, z) = (2.5 + 5.0 * g.next(), 2.5 + 5.0 * g.next());
        let (hx, hz) = (0.2 + 0.6 * g.next(), 0.2 + 0.6 * g.next());
        boxes.push(([x, 1.5, z], [hx, 1.5, hz]));
    }
    let gap = 3.0 * g.next() - 1.5; // where the partial wall leaves an opening
    boxes.push(([gap - 3.5, 1.5, 5.0], [2.6, 1.5, 0.1]));
    boxes.push(([gap + 13.5 - 3.5, 1.5, 5.0], [8.5, 1.5, 0.1]));
    Boxes { boxes }
}

fn room_grid() -> ReachGrid {
    // not square, so swapping x and z in the cell index is visible
    ReachGrid::floor([0.0, 0.0, 0.0], C, 51, 50)
}

#[test]
fn sound_capsule_clears_obstacles_along_dense_witness() {
    let agent = Clearance::capsule(R, &HEIGHTS);
    let (mut found, mut missed, mut samples) = (0, 0, 0usize);
    for seed in 0..40 {
        let f = room(seed);
        match reach(&f, &room_grid(), [1.0, 0.0, 1.0], [9.0, 0.0, 9.0], &agent) {
            Ok(Some(w)) => {
                found += 1;
                for s in w.windows(2) {
                    for k in 0..=16 {
                        let t = k as f32 / 16.0;
                        let x = s[0][0] + (s[1][0] - s[0][0]) * t;
                        let z = s[0][2] + (s[1][2] - s[0][2]) * t;
                        let mut y = HEIGHTS[0];
                        while y <= HEIGHTS[2] + 1e-6 {
                            let d = f.distance(x, y, z);
                            assert!(
                                d >= R - 1e-5,
                                "seed {seed}: capsule hits at ({x}, {y}, {z}), d = {d}"
                            );
                            samples += 1;
                            y += 0.01;
                        }
                    }
                }
            }
            Ok(None) => missed += 1,
            Err(e) => panic!("seed {seed}: {e:?}"),
        }
    }
    eprintln!("[sdf_reach sound] reached {found}, not reached {missed}, samples {samples}");
    assert!(found > 0 && samples > 0, "no witness was checked");
}

/// The bound of the module doc: `w >= 2r + 2·offset + c` (axis-aligned) is always found; for every
/// grid offset across the corridor.
fn corridor_reached(n: [f32; 2], w: f32, offsets: usize) -> (usize, usize) {
    let agent = Clearance::capsule(R, &HEIGHTS);
    let f = Corridor { n, w };
    let dir = [-n[1], n[0]]; // along the corridor
    let (mut yes, mut no) = (0, 0);
    for k in 0..offsets {
        let shift = C * k as f32 / offsets as f32;
        let grid = ReachGrid::floor([-8.0 + shift, 0.0, -8.0 + shift], C, 80, 80);
        let a = [dir[0] * -5.0, 0.0, dir[1] * -5.0];
        let b = [dir[0] * 5.0, 0.0, dir[1] * 5.0];
        match reach_map(&f, &grid, a, &agent) {
            Ok(m) if m.reached(b) => yes += 1,
            Ok(_) | Err(ReachError::StartNotFree) => no += 1,
            Err(e) => panic!("{e:?}"),
        }
    }
    (yes, no)
}

#[test]
fn complete_at_the_axis_aligned_bound() {
    let bound = 2.0 * R + 2.0 * offset(C, S) + C;
    let (yes, no) = corridor_reached([0.0, 1.0], bound + 1e-3, 16);
    assert_eq!((yes, no), (16, 0), "bound {bound}");
    // just below the bound some grid offset misses the passage (the bound is tight, not loose)
    let (_, no) = corridor_reached([0.0, 1.0], bound - 0.02, 16);
    assert!(
        no > 0,
        "nothing missed below the bound: the test has no teeth"
    );
}

#[test]
fn complete_at_the_any_direction_bound() {
    let bound = 2.0 * R + 2.0 * offset(C, S) + std::f32::consts::SQRT_2 * C;
    let h = std::f32::consts::FRAC_1_SQRT_2;
    let (yes, no) = corridor_reached([h, h], bound + 1e-3, 16);
    assert_eq!((yes, no), (16, 0), "bound {bound}");
}

#[test]
fn floor_just_above_and_below_the_margin() {
    let m = R + offset(C, S);
    let grid = ReachGrid::floor([0.0, 0.0, 0.0], C, 10, 10);
    // two heights `s` apart, so the margin is exactly r + offset(c, s)
    let lift = |h0: f32| [h0, h0 + S];
    // the lowest sample `h0` above the floor: free iff h0 > m (the floor is the only obstacle)
    let above = Clearance::capsule(R, &lift(m + 1e-3));
    assert!(
        reach(&Floor(0.0), &grid, [1.0, 0.0, 1.0], [1.5, 0.0, 1.0], &above)
            .unwrap()
            .is_some()
    );
    let below = Clearance::capsule(R, &lift(m - 1e-3));
    assert_eq!(
        reach_map(&Floor(0.0), &grid, [1.0, 0.0, 1.0], &below).unwrap_err(),
        ReachError::StartNotFree
    );
}

#[test]
fn nan_is_an_error_never_blocked() {
    let grid = ReachGrid::floor([0.0, 0.0, 0.0], C, 30, 30);
    let agent = Clearance::capsule(R, &[2.0, 2.2]);
    let e = reach_map(&NanAt([3.0, 3.0]), &grid, [0.5, 0.0, 0.5], &agent).unwrap_err();
    assert!(matches!(e, ReachError::NonFiniteDistance { .. }), "{e:?}");
}

#[test]
fn invalid_inputs_are_errors() {
    let f = Floor(-10.0);
    let a = Clearance::capsule(R, &HEIGHTS);
    let p = [0.5, 0.0, 0.5];
    assert_eq!(
        reach_map(&f, &ReachGrid::floor([0.0; 3], C, 0, 5), p, &a).unwrap_err(),
        ReachError::InvalidGrid
    );
    assert_eq!(
        reach_map(&f, &ReachGrid::floor([0.0; 3], 0.0, 5, 5), p, &a).unwrap_err(),
        ReachError::InvalidGrid
    );
    assert_eq!(
        reach_map(&f, &ReachGrid::floor([0.0; 3], f32::NAN, 5, 5), p, &a).unwrap_err(),
        ReachError::InvalidGrid
    );
    let g = ReachGrid::floor([0.0; 3], C, 5, 5);
    for bad in [
        Clearance::capsule(R, &[]),
        Clearance::capsule(R, &[1.0, 0.5]),
        Clearance::capsule(0.0, &HEIGHTS),
        Clearance::capsule(f32::NAN, &HEIGHTS),
        Clearance::capsule(R, &HEIGHTS).with_lipschitz(0.0),
    ] {
        assert_eq!(
            reach_map(&f, &g, p, &bad).unwrap_err(),
            ReachError::InvalidClearance,
            "{bad:?}"
        );
    }
    assert_eq!(
        reach_map(&f, &g, [-1.0, 0.0, 0.5], &a).unwrap_err(),
        ReachError::StartOutOfGrid
    );
    assert_eq!(
        reach_map(&f, &g, [0.5, 0.0, 99.0], &a).unwrap_err(),
        ReachError::StartOutOfGrid
    );
}

#[test]
fn witness_is_a_four_connected_chain_from_start_to_goal() {
    let agent = Clearance::capsule(R, &HEIGHTS);
    let mut checked = 0;
    for seed in 0..10 {
        let (from, to) = ([1.0, 0.0, 1.0], [9.0, 0.0, 9.0]);
        if let Some(w) = reach(&room(seed), &room_grid(), from, to, &agent).unwrap() {
            for s in w.windows(2) {
                let d = (s[1][0] - s[0][0]).abs() + (s[1][2] - s[0][2]).abs();
                assert!((d - C).abs() < 1e-4, "not 4-adjacent: {s:?}");
            }
            let (a, b) = (w[0], w[w.len() - 1]);
            assert!(
                (a[0] - from[0]).abs() <= C * 0.5 + 1e-5
                    && (a[2] - from[2]).abs() <= C * 0.5 + 1e-5
            );
            assert!(
                (b[0] - to[0]).abs() <= C * 0.5 + 1e-5 && (b[2] - to[2]).abs() <= C * 0.5 + 1e-5
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no witness was checked");
}

/// FNV-1a over the witness bits of a fixed corpus. The same value on every platform is the
/// determinism claim of the module doc (CI runs this on Linux, macOS, Windows and wasm32-wasip1).
const GOLDEN_WITNESS_HASH: u64 = 0xb926_9c54_f10e_9e07;

#[test]
fn witness_hash_is_platform_independent() {
    let agent = Clearance::capsule(R, &HEIGHTS);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u32| {
        for b in v.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    };
    let mut compared = 0;
    for seed in 100..112 {
        let m = reach_map(&room(seed), &room_grid(), [1.0, 0.0, 1.0], &agent).unwrap();
        eat(m.free_cells() as u32);
        if let Some(w) = m.witness([9.0, 0.0, 9.0]) {
            for p in w {
                eat(p[0].to_bits());
                eat(p[2].to_bits());
                compared += 1;
            }
        }
    }
    eprintln!("[sdf_reach golden] hash {h:#018x} over {compared} witness points");
    assert!(compared > 0, "0 witness points compared");
    assert_eq!(h, GOLDEN_WITNESS_HASH);
}

#[test]
fn reachability_is_symmetric() {
    // a path is a path both ways: going from the goal back to the start needs -x and -z moves
    let agent = Clearance::capsule(R, &HEIGHTS);
    let (a, b) = ([1.0, 0.0, 1.0], [9.0, 0.0, 9.0]);
    let (mut both, mut neither) = (0, 0);
    for seed in 0..40 {
        let f = room(seed);
        let ab = reach(&f, &room_grid(), a, b, &agent).unwrap().is_some();
        let ba = reach(&f, &room_grid(), b, a, &agent).unwrap().is_some();
        assert_eq!(ab, ba, "seed {seed}: a→b {ab}, b→a {ba}");
        if ab {
            both += 1
        } else {
            neither += 1
        }
    }
    assert!(
        both > 0 && neither > 0,
        "the corpus has no variety: {both} / {neither}"
    );
}

#[test]
fn reached_and_counts_report_the_map() {
    // a wall across the room with no opening: the far side is not reached
    let f = Boxes {
        boxes: vec![([5.0, 1.5, 5.0], [6.0, 1.5, 0.2])],
    };
    let g = ReachGrid::floor([0.0, 0.0, 0.0], C, 51, 50);
    assert_eq!(g.cell(), C);
    let m = reach_map(&f, &g, [1.0, 0.0, 1.0], &Clearance::capsule(R, &HEIGHTS)).unwrap();
    assert!(m.reached([1.0, 0.0, 2.0]));
    assert!(
        !m.reached([1.0, 0.0, 9.0]),
        "the far side of a closed wall is reached"
    );
    assert!(m.witness([1.0, 0.0, 9.0]).is_none());
    assert!(
        !m.reached([-5.0, 0.0, 1.0]),
        "a point outside the grid is reached"
    );
    assert_eq!(m.checked_cells(), 51 * 50);
    assert!(m.free_cells() > 0 && m.free_cells() < m.checked_cells());
}

#[test]
fn a_sample_exactly_at_the_margin_is_not_free() {
    // one height, so s = 0 and the margin is r + c/√2 (closed form); `free` is strict (`>`)
    let m = R + C * std::f32::consts::FRAC_1_SQRT_2;
    let agent = Clearance::capsule(R, &[m]);
    assert!(
        (agent.margin(C) - m).abs() < 1e-6,
        "margin {} vs closed form {m}",
        agent.margin(C)
    );
    let exact = Clearance::capsule(R, &[agent.margin(C)]);
    let g = ReachGrid::floor([0.0, 0.0, 0.0], C, 10, 10);
    assert_eq!(
        reach_map(&Floor(0.0), &g, [1.0, 0.0, 1.0], &exact).unwrap_err(),
        ReachError::StartNotFree
    );
}

#[test]
fn grid_edges_and_size_limit() {
    let f = Floor(-10.0);
    let a = Clearance::capsule(R, &HEIGHTS);
    // cell 0.25 is exact in binary, so the far edge is exactly origin + n·cell
    let g = ReachGrid::floor([0.0, 0.0, 0.0], 0.25, 8, 6);
    assert!(reach_map(&f, &g, [1.999, 0.0, 1.499], &a).is_ok());
    assert_eq!(
        reach_map(&f, &g, [2.0, 0.0, 0.5], &a).unwrap_err(),
        ReachError::StartOutOfGrid
    );
    assert_eq!(
        reach_map(&f, &g, [0.5, 0.0, 1.5], &a).unwrap_err(),
        ReachError::StartOutOfGrid
    );
    assert_eq!(
        reach_map(
            &f,
            &ReachGrid::floor([0.0; 3], C, 5, 0),
            [0.5, 0.0, 0.5],
            &a
        )
        .unwrap_err(),
        ReachError::InvalidGrid
    );
    assert_eq!(
        reach_map(
            &f,
            &ReachGrid::floor([f32::NAN, 0.0, 0.0], C, 5, 5),
            [0.5, 0.0, 0.5],
            &a
        )
        .unwrap_err(),
        ReachError::InvalidGrid
    );
    assert_eq!(
        Clearance::capsule(R, &[1.0, 1.0]).margin(C),
        Clearance::capsule(R, &[1.0]).margin(C)
    );
    assert_eq!(
        reach_map(
            &f,
            &ReachGrid::floor([0.0; 3], C, 5, 5),
            [0.5, 0.0, 0.5],
            &Clearance::capsule(R, &[1.0, 1.0])
        )
        .unwrap_err(),
        ReachError::InvalidClearance,
        "equal heights are not ascending"
    );
    // the largest grid is accepted, one more column is not
    let one = Clearance::capsule(R, &[1.0]);
    assert!(reach_map(
        &f,
        &ReachGrid::floor([0.0; 3], 1.0, 4096, MAX_CELLS / 4096),
        [0.5, 0.0, 0.5],
        &one
    )
    .is_ok());
    assert_eq!(
        reach_map(
            &f,
            &ReachGrid::floor([0.0; 3], 1.0, 4097, MAX_CELLS / 4096),
            [0.5, 0.0, 0.5],
            &one
        )
        .unwrap_err(),
        ReachError::InvalidGrid
    );
}
