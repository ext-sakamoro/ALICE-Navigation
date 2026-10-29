//! Deterministic shortest-path search on any graph whose nodes have positions.
//!
//! One A* serves every mesh in this crate and its users: a graph gives node positions in
//! `[x, y, z]` (f64) and neighbour lists ([`SearchGraph`]).
//!
//! # Contract
//!
//! * The cost of an edge is [`SearchGraph::edge_cost`] (default: the Euclidean distance between
//!   the two positions). A custom cost must be at least that distance: the heuristic is the
//!   Euclidean distance to the goal, which is then admissible and consistent, so the path found
//!   is a shortest one. Debug builds assert the contract on every edge the search relaxes.
//! * Ties are broken by a stated rule: the open set is ordered by `(f, g, node index)`, and when
//!   two parents give a node the same `g`, the smaller parent index wins. The result therefore
//!   does not depend on the order in which a graph lists neighbours.
//! * Arithmetic is `+ − × ÷` and `sqrt` on f64 only (all correctly rounded by IEEE 754; no
//!   transcendental and no fused operation), so the same graph gives the same path on every
//!   platform.

use crate::core_types::Vec2;
use crate::navmesh::{NavMesh, NavTriangle};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque};

/// A graph the search can walk.
pub trait SearchGraph {
    /// The number of nodes; nodes are `0..node_count()`.
    fn node_count(&self) -> usize;
    /// The position of `node`.
    fn position(&self, node: usize) -> [f64; 3];
    /// Appends the neighbours of `node` to `out`.
    fn neighbors(&self, node: usize, out: &mut Vec<usize>);
    /// The cost of the edge `a → b`; at least the Euclidean distance between them (see the
    /// module contract).
    fn edge_cost(&self, a: usize, b: usize) -> f64 {
        distance(self.position(a), self.position(b))
    }
}

/// Euclidean distance: [`norm`] of `a − b`.
pub fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

/// Euclidean length `sqrt(x² + y² + z²)` with `+ − × ÷ √` only (correctly rounded by IEEE 754,
/// the same bits on every platform).
///
/// No overflow or underflow on the way: for a largest component inside `[2^-500, 2^500]` the
/// plain form is used (its bits are kept); outside it every component is first scaled by the same
/// exact power of two so the largest lies in `[1, 2)`, the plain form is taken there and the
/// result scaled back. The error is therefore that of the plain form at normal magnitude:
/// measured within 1 ulp of a double-double reference over a seeded sweep of the whole exponent
/// range (2D and 3D, both ends), and within 1 ulp of the subnormal grid for subnormal lengths.
/// The result is +∞ only when the true length exceeds `f64::MAX`. A NaN component gives NaN;
/// otherwise an infinite component gives +∞.
pub fn norm(v: [f64; 3]) -> f64 {
    let largest = alice_det_math::max64(alice_det_math::max64(v[0].abs(), v[1].abs()), v[2].abs());
    // inside `[2^-500, 2^500]` (or for 0, NaN and ∞) the squares neither overflow nor lose bits
    // that matter: the plain form, whose bits the pinned paths depend on
    if largest == 0.0 || !largest.is_finite() || (PLAIN_LOW..=PLAIN_HIGH).contains(&largest) {
        return plain_norm(v);
    }
    // otherwise move the largest component into `[1, 2)` by an exact power of two, take the plain
    // form there and move back: the same arithmetic as at normal magnitude, one rounding at the end
    let k = binary_exponent(largest);
    times_pow2(plain_norm(v.map(|c| times_pow2(c, -k))), k)
}

/// `sqrt(x² + y² + z²)` as written.
fn plain_norm([x, y, z]: [f64; 3]) -> f64 {
    (x * x + y * y + z * z).sqrt()
}

/// The plain form is used for a largest component in `[PLAIN_LOW, PLAIN_HIGH]` = `[2^-500, 2^500]`.
const PLAIN_LOW: f64 = f64::from_bits((1023 - 500) << 52);
const PLAIN_HIGH: f64 = f64::from_bits((1023 + 500) << 52);

/// `2^e` for `e` in `[-1022, 1023]`, from its bits.
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// `x · 2^e` for `e` in `[-1074, 1074]`, as two normal factors: exact whenever the result is
/// normal, one rounding when it is subnormal.
fn times_pow2(x: f64, e: i32) -> f64 {
    let half = e / 2;
    x * pow2(half) * pow2(e - half)
}

/// The binary exponent of a finite non-zero `m` (`m ∈ [2^k, 2^(k+1))`), subnormals included.
fn binary_exponent(m: f64) -> i32 {
    let raw = ((m.to_bits() >> 52) & 0x7ff) as i32;
    if raw == 0 {
        binary_exponent(m * pow2(64)) - 64
    } else {
        raw - 1023
    }
}

/// The shortest path from `start` to `goal` (both included), or `None` when either is out of
/// range or the goal is not reachable.
pub fn astar<G: SearchGraph + ?Sized>(graph: &G, start: usize, goal: usize) -> Option<Vec<usize>> {
    astar_filtered(graph, start, goal, |_| true)
}

/// [`astar`] that only enters nodes for which `allow` is true (`start` and `goal` included).
pub fn astar_filtered<G: SearchGraph + ?Sized, F: Fn(usize) -> bool>(
    graph: &G,
    start: usize,
    goal: usize,
    allow: F,
) -> Option<Vec<usize>> {
    let n = graph.node_count();
    if start >= n || goal >= n || !allow(start) || !allow(goal) {
        return None;
    }
    if start == goal {
        return Some(vec![start]);
    }
    let target = graph.position(goal);
    let mut g = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];
    let mut closed = vec![false; n];
    let mut open = BinaryHeap::new();
    g[start] = 0.0;
    open.extend([Open {
        f: distance(graph.position(start), target),
        g: 0.0,
        node: start,
    }]);
    let mut nb = Vec::new();
    while let Some(Open { node, .. }) = open.pop() {
        if closed[node] {
            continue;
        }
        if node == goal {
            let mut path = vec![goal];
            let mut cur = goal;
            while cur != start {
                cur = parent[cur];
                path.extend([cur]);
            }
            path.reverse();
            return Some(path);
        }
        closed[node] = true;
        nb.clear();
        graph.neighbors(node, &mut nb);
        for &next in &nb {
            if next >= n || closed[next] || !allow(next) {
                continue;
            }
            let cost = graph.edge_cost(node, next);
            debug_assert!(
                cost >= distance(graph.position(node), graph.position(next)),
                "edge cost {cost} of {node} → {next} is below the Euclidean distance"
            );
            let tentative = g[node] + cost;
            if tentative < g[next] {
                g[next] = tentative;
                parent[next] = node;
                open.extend([Open {
                    f: tentative + distance(graph.position(next), target),
                    g: tentative,
                    node: next,
                }]);
            } else if tentative == g[next] && node < parent[next] {
                // an exact tie: the smaller parent index wins, so neighbour order does not matter
                parent[next] = node;
            }
        }
    }
    None
}

/// An open-set entry; the heap pops the smallest `(f, g, node)`.
struct Open {
    f: f64,
    g: f64,
    node: usize,
}

impl PartialEq for Open {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Open {}
impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Open {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reversed: `BinaryHeap` is a max-heap
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| other.g.total_cmp(&self.g))
            .then_with(|| other.node.cmp(&self.node))
    }
}

/// The sum of the edge costs along `path`.
pub fn path_cost<G: SearchGraph + ?Sized>(graph: &G, path: &[usize]) -> f64 {
    path.windows(2).map(|w| graph.edge_cost(w[0], w[1])).sum()
}

/// Clusters of a `width × height` cell grid, `cluster_size` cells square.
///
/// Returns the cluster count and the cluster of each of two triangles per cell (row-major), the
/// layout of [`build_occupancy_navmesh`] with no blocked cell.
pub fn cluster_grid(width: usize, height: usize, cluster_size: usize) -> (usize, Vec<usize>) {
    let size = cluster_size.max(1);
    let cw = width.div_ceil(size);
    let ch = height.div_ceil(size);
    let mut map = Vec::with_capacity(width * height * 2);
    for y in 0..height {
        for x in 0..width {
            let c = (y / size) * cw + x / size;
            map.extend([c, c]);
        }
    }
    (cw * ch, map)
}

/// A* restricted to a chain of clusters.
///
/// The chain is a fewest-hops route from the start's cluster to the goal's (breadth first over
/// cluster adjacency, lower cluster index first), then [`astar_filtered`] runs inside it. Start
/// and goal in one cluster is plain [`astar`]. Not always a shortest path; always a valid one.
pub fn hierarchical_astar<G: SearchGraph + ?Sized>(
    graph: &G,
    start: usize,
    goal: usize,
    node_cluster: &[usize],
) -> Option<Vec<usize>> {
    let n = graph.node_count();
    if start >= n || goal >= n || node_cluster.len() < n {
        return None;
    }
    let (sc, gc) = (node_cluster[start], node_cluster[goal]);
    if sc == gc {
        return astar(graph, start, goal);
    }
    // cluster adjacency, sorted so the breadth-first order is fixed
    let mut adj = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut nb = Vec::new();
    for a in 0..n {
        nb.clear();
        graph.neighbors(a, &mut nb);
        for &b in &nb {
            if b < n && node_cluster[a] != node_cluster[b] {
                adj.entry(node_cluster[a])
                    .or_default()
                    .insert(node_cluster[b]);
            }
        }
    }
    let mut parent = BTreeMap::<usize, usize>::new();
    let mut queue = VecDeque::from([sc]);
    parent.insert(sc, sc);
    while let Some(c) = queue.pop_front() {
        if c == gc {
            break;
        }
        for &d in adj.get(&c).into_iter().flatten() {
            if let std::collections::btree_map::Entry::Vacant(e) = parent.entry(d) {
                e.insert(c);
                queue.extend([d]);
            }
        }
    }
    if !parent.contains_key(&gc) {
        return None;
    }
    let mut chain = BTreeSet::from([gc]);
    let mut c = gc;
    while c != sc {
        c = parent[&c];
        chain.insert(c);
    }
    astar_filtered(graph, start, goal, |node| {
        chain.contains(&node_cluster[node])
    })
}

/// A navmesh over a `width × height` occupancy grid (`blocked[y * width + x]`).
///
/// Cells of size `cell` from `origin`: two triangles per free cell, `(bl, br, tl)` then
/// `(br, tr, tl)`, in row-major cell order; triangles that share an edge are neighbours.
pub fn build_occupancy_navmesh(
    blocked: &[bool],
    width: usize,
    height: usize,
    cell: f64,
    origin: Vec2,
) -> NavMesh {
    assert_eq!(
        blocked.len(),
        width * height,
        "the grid has width × height cells"
    );
    let mut triangles: Vec<NavTriangle> = Vec::with_capacity(width * height * 2);
    // the first triangle of each free cell, to wire neighbours
    let mut first = vec![usize::MAX; width * height];
    for y in 0..height {
        for x in 0..width {
            if blocked[y * width + x] {
                continue;
            }
            let x0 = origin.x + x as f64 * cell;
            let y0 = origin.y + y as f64 * cell;
            let (bl, br) = (Vec2::new(x0, y0), Vec2::new(x0 + cell, y0));
            let (tl, tr) = (Vec2::new(x0, y0 + cell), Vec2::new(x0 + cell, y0 + cell));
            first[y * width + x] = triangles.len();
            triangles.extend([NavTriangle::new(bl, br, tl), NavTriangle::new(br, tr, tl)]);
        }
    }
    let link = |a: usize, b: usize, triangles: &mut Vec<NavTriangle>| {
        for (from, to) in [(a, b), (b, a)] {
            let slot = triangles[from]
                .neighbors
                .iter()
                .position(Option::is_none)
                .expect("a triangle has at most 3 neighbours");
            triangles[from].neighbors[slot] = Some(to);
        }
    };
    for y in 0..height {
        for x in 0..width {
            let t = first[y * width + x];
            if t == usize::MAX {
                continue;
            }
            // the diagonal inside the cell
            link(t, t + 1, &mut triangles);
            // (bl, br, tl) shares its left edge with the left cell's (br, tr, tl)
            if x > 0 && first[y * width + x - 1] != usize::MAX {
                link(t, first[y * width + x - 1] + 1, &mut triangles);
            }
            // and its bottom edge with the lower cell's (br, tr, tl)
            if y > 0 && first[(y - 1) * width + x] != usize::MAX {
                link(t, first[(y - 1) * width + x] + 1, &mut triangles);
            }
        }
    }
    NavMesh::new(triangles)
}

/// A navmesh is searched over its triangle centroids, in its plane (`z = 0`).
impl SearchGraph for NavMesh {
    fn node_count(&self) -> usize {
        self.triangles.len()
    }
    fn position(&self, node: usize) -> [f64; 3] {
        let c = self.triangles[node].centroid();
        [c.x, c.y, 0.0]
    }
    fn neighbors(&self, node: usize, out: &mut Vec<usize>) {
        out.extend(self.triangles[node].neighbors.iter().flatten().copied());
    }
}
