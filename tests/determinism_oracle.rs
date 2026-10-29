//! One distance definition, a measured accuracy bound, and pinned outputs that must be the same
//! on every platform (CI: Linux, macOS, Windows, wasm32-wasip1).

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

// --- a high-precision reference for the length (double-double, no fused operation) -------------

/// `a·b` exactly as `hi + lo` (Dekker's split; exact while the parts stay normal).
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let split = |x: f64| {
        let c = 134_217_729.0 * x; // 2^27 + 1
        let hi = c - (c - x);
        (hi, x - hi)
    };
    let p = a * b;
    let ((ah, al), (bh, bl)) = (split(a), split(b));
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// `a + b` exactly as `hi + lo`.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `2^e` for `e` in the normal range, from its bits.
fn pow2(e: i32) -> f64 {
    assert!((-1022..=1023).contains(&e), "2^{e} is not a normal f64");
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// `x · 2^e` exactly when the result is normal (two normal factors, so `e` may span the whole
/// range of exponents).
fn scale2(x: f64, e: i32) -> f64 {
    let a = e / 2;
    x * pow2(a) * pow2(e - a)
}

/// The binary exponent of a finite non-zero `m` (subnormals included).
fn exponent(m: f64) -> i32 {
    let raw = ((m.to_bits() >> 52) & 0x7ff) as i32;
    if raw == 0 {
        exponent(m * pow2(64)) - 64
    } else {
        raw - 1023
    }
}

/// `sqrt(x² + y² + z²)` to about 1e-30 relative, then rounded: the components are scaled by an
/// exact power of two so the largest lies in `[1, 2)`, the squares summed in double-double and the
/// square root refined by one Newton step in double-double. For a normal result this is the
/// correctly rounded length (bar ties within 1e-30); for a subnormal result it is rounded twice
/// (to 53 bits, then to the subnormal grid).
fn reference_norm(v: [f64; 3]) -> f64 {
    let m = v
        .iter()
        .fold(0.0f64, |a, b| if b.abs() > a { b.abs() } else { a });
    if m == 0.0 {
        return 0.0;
    }
    let k = exponent(m);
    let (mut hi, mut lo) = (0.0, 0.0);
    for c in v {
        // components below 2^-1022 after scaling lose bits, at most 2^-1074 against a sum ≥ 1
        let s = scale2(c, -k);
        let (p, e) = two_product(s, s);
        let (h, l) = two_sum(hi, p);
        hi = h;
        lo += l + e;
    }
    let (h, l) = two_sum(hi, lo);
    let r = h.sqrt();
    let (rr, rre) = two_product(r, r);
    let corr = ((h - rr) - rre + l) / (2.0 * r);
    scale2(r + corr, k)
}

fn ulps(a: f64, b: f64) -> u64 {
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
}

/// A random vector whose largest component has binary exponent `e`, the others up to 70 binades
/// smaller (or zero), random signs; `z = 0` when `dims == 2`.
fn random_vector(g: &mut Lcg, e: i32, dims: usize) -> [f64; 3] {
    let mut v = [0.0; 3];
    let lead = (g.next() * 3.0) as usize % dims;
    for (i, c) in v.iter_mut().enumerate().take(dims) {
        let sign = if g.next() < 0.5 { -1.0 } else { 1.0 };
        let mantissa = 1.0 + g.next(); // [1, 2)
        let ei = if i == lead {
            e
        } else if g.next() < 0.1 {
            continue;
        } else {
            e - (g.next() * 70.0) as i32
        };
        if ei < -1074 {
            continue;
        }
        // the mantissa is rounded to the subnormal grid there; still an exact input
        *c = sign
            * if ei >= -1022 {
                scale2(mantissa, ei)
            } else {
                mantissa * pow2(-1022) * pow2(ei + 1022)
            };
    }
    v
}

/// Two vectors where the division-based fallback was 2 ulps off and where the squares underflowed
/// to nothing.
#[test]
fn the_reported_vectors_are_within_one_ulp() {
    use alice_navigation::graph_search::norm;
    for v in [
        [6.067_314_953_608_391e307, -5.460_867_120_410_906e307, 0.0],
        [1.33e-165, 1.546e-162, 0.0],
    ] {
        let (n, r) = (norm(v), reference_norm(v));
        assert!(
            ulps(n, r) <= 1,
            "{v:?}: {n:e} vs reference {r:e} ({} ulps)",
            ulps(n, r)
        );
        let w = Vec2::new(v[0], v[1]).length();
        assert_eq!(w.to_bits(), n.to_bits());
    }
}

/// A seeded sweep over the whole exponent range of the largest component (2^-1022 .. 2^1023, both
/// the overflow and the underflow ends), 2D and 3D: the measured bound for every normal length is
/// 1 ulp against the reference.
#[test]
fn a_sweep_across_the_exponent_range_stays_within_one_ulp() {
    use alice_navigation::graph_search::norm;
    let mut g = Lcg(23);
    let (mut cases, mut worst, mut low, mut high) = (0u32, 0u64, 0u32, 0u32);
    for dims in [2, 3] {
        for _ in 0..20_000 {
            let e = -1022 + (g.next() * 2046.0) as i32;
            let v = random_vector(&mut g, e, dims);
            let r = reference_norm(v);
            if !r.is_finite() || r < f64::MIN_POSITIVE {
                continue;
            }
            let d = ulps(norm(v), r);
            assert!(d <= 1, "{v:?}: {:e} vs reference {r:e} ({d} ulps)", norm(v));
            worst = worst.max(d);
            cases += 1;
            low += u32::from(e < -511);
            high += u32::from(e > 511);
        }
    }
    // not a vacuous sweep: both ends where the plain form fails are well represented
    assert!(
        cases >= 39_000 && low >= 9_000 && high >= 9_000,
        "{cases} {low} {high}"
    );
    println!("normal lengths: {cases} cases, worst {worst} ulp ({low} low-end, {high} high-end)");
}

/// Lengths below `f64::MIN_POSITIVE` (largest component subnormal or just above): within 1 ulp of
/// the subnormal grid.
#[test]
fn subnormal_lengths_stay_within_one_ulp_of_the_grid() {
    use alice_navigation::graph_search::norm;
    let mut g = Lcg(29);
    let mut cases = 0u32;
    for dims in [2, 3] {
        for _ in 0..5_000 {
            let e = -1074 + (g.next() * 52.0) as i32;
            let v = random_vector(&mut g, e, dims);
            let r = reference_norm(v);
            if r >= f64::MIN_POSITIVE {
                continue;
            }
            assert!(ulps(norm(v), r) <= 1, "{v:?}: {:e} vs {r:e}", norm(v));
            cases += 1;
        }
    }
    assert!(cases >= 5_000, "{cases}");
    assert_eq!(norm([f64::from_bits(1), 0.0, 0.0]), f64::from_bits(1));
    assert_eq!(
        norm([0.0, -f64::from_bits(3), f64::from_bits(4)]),
        f64::from_bits(5)
    );
}

#[test]
fn the_extremes() {
    use alice_navigation::graph_search::norm;
    for v in [
        [f64::MAX, 0.0, 0.0],
        [0.0, 0.0, f64::MAX],
        [1.0, -1e-300, 1.7e308],
    ] {
        assert_eq!(norm(v).to_bits(), reference_norm(v).to_bits(), "{v:?}");
    }
    assert!(Vec2::new(f64::MAX / 2.0, -f64::MAX / 2.0)
        .distance_to(Vec2::new(0.0, 0.0))
        .is_finite());
    // the true length √2·MAX is not representable: +∞ is the stated answer, not an overflow
    assert_eq!(Vec2::new(f64::MAX, -f64::MAX).length(), f64::INFINITY);
}

/// Where every non-zero square is normal and the sum is finite, the plain form is as accurate as
/// any scaled one, and its bits are kept (the pinned paths and hashes depend on them).
#[test]
fn the_plain_form_keeps_its_bits_where_it_is_accurate() {
    use alice_navigation::graph_search::norm;
    let mut g = Lcg(11);
    let mut compared = 0u32;
    for _ in 0..20_000 {
        let e = (g.next() * 1200.0 - 600.0).round() as i32;
        let v = random_vector(&mut g, e, 3);
        let plain = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let accurate = v.iter().all(|c| *c == 0.0 || c * c >= f64::MIN_POSITIVE);
        if plain.is_finite() && accurate {
            assert_eq!(norm(v).to_bits(), plain.to_bits(), "{v:?}");
            compared += 1;
        }
    }
    assert!(compared >= 5_000, "{compared}");
}

#[test]
fn lengths_never_decrease_across_the_switch_point() {
    use alice_navigation::graph_search::norm;
    // directions whose plain form overflows near different thresholds
    for (dir, threshold) in [
        ([1.0, 0.0, 0.0], f64::MAX.sqrt()),
        ([1.0, 1.0, 0.0], (f64::MAX / 2.0).sqrt()),
        ([1.0, 1.0, 1.0], (f64::MAX / 3.0).sqrt()),
        ([1.0, 0.5, -0.25], (f64::MAX / 1.3125).sqrt()),
    ] {
        let mut x = f64::from_bits(threshold.to_bits() - 5000);
        let mut prev = 0.0f64;
        let mut switched = false;
        for _ in 0..10_000 {
            let v = [dir[0] * x, dir[1] * x, dir[2] * x];
            let n = norm(v);
            assert!(n.is_finite() && n >= prev, "{dir:?} at {x}: {n} < {prev}");
            switched |= !(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().is_finite();
            prev = n;
            x = f64::from_bits(x.to_bits() + 1);
        }
        assert!(switched, "{dir:?}: the sweep never reached the fallback");
    }
}

/// The plain form is used for a largest component in `[2^-500, 2^500]` (the documented switch
/// points): ulp-step sweeps across both, and a geometric sweep over the whole range.
#[test]
fn lengths_never_decrease_across_the_plain_range_ends() {
    use alice_navigation::graph_search::norm;
    for edge in [pow2(500), pow2(-500)] {
        for dir in [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, -0.75, 0.5]] {
            let mut x = f64::from_bits(edge.to_bits() - 5000);
            let mut prev = 0.0f64;
            for _ in 0..10_000 {
                let n = norm([dir[0] * x, dir[1] * x, dir[2] * x]);
                assert!(n >= prev, "{dir:?} at {x:e}: {n:e} < {prev:e}");
                prev = n;
                x = f64::from_bits(x.to_bits() + 1);
            }
            assert!(x > edge, "the sweep crossed the switch point");
        }
    }
    // 1e-320 .. 1e308 in steps of 0.5 %: far wider than 1 ulp, so a 1-ulp bound keeps it monotonic
    let mut x = 1e-320f64;
    let mut prev = 0.0f64;
    while x < 1e308 {
        let n = norm([x, -0.5 * x, 0.25 * x]);
        assert!(n >= prev, "{x:e}: {n:e} < {prev:e}");
        prev = n;
        x *= 1.005;
    }
}

#[test]
fn non_finite_components() {
    use alice_navigation::graph_search::norm;
    // NaN in, NaN out; otherwise an infinite component gives +∞
    assert!(norm([f64::NAN, 1.0, 2.0]).is_nan());
    assert!(norm([f64::INFINITY, f64::NAN, 0.0]).is_nan());
    assert_eq!(norm([f64::INFINITY, 1.0, 0.0]), f64::INFINITY);
    assert_eq!(norm([0.0, f64::NEG_INFINITY, 1e300]), f64::INFINITY);
    assert!(Vec2::new(f64::NAN, 0.0).length().is_nan());
    assert_eq!(Vec2::new(f64::NEG_INFINITY, 3.0).length(), f64::INFINITY);
    assert_eq!(norm([0.0, 0.0, 0.0]), 0.0);
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
