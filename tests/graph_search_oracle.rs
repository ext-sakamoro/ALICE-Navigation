//! Oracles for `graph_search`. Shortest costs come from an independent O(n²) Dijkstra written
//! here (no heap, no heuristic), never from the search under test.

use alice_navigation::core_types::{Bounds2D, Vec2};
use alice_navigation::graph_search::{
    astar, astar_filtered, build_occupancy_navmesh, cluster_grid, distance, hierarchical_astar,
    path_cost, SearchGraph,
};
use alice_navigation::navmesh::build_grid_navmesh;

/// A graph given by positions and adjacency lists, with an optional cost scale per edge.
#[derive(Clone)]
struct Graph {
    pos: Vec<[f64; 3]>,
    adj: Vec<Vec<usize>>,
    scale: f64,
}

impl SearchGraph for Graph {
    fn node_count(&self) -> usize {
        self.pos.len()
    }
    fn position(&self, n: usize) -> [f64; 3] {
        self.pos[n]
    }
    fn neighbors(&self, n: usize, out: &mut Vec<usize>) {
        out.extend(self.adj[n].iter().copied());
    }
    fn edge_cost(&self, a: usize, b: usize) -> f64 {
        distance(self.pos[a], self.pos[b]) * self.scale
    }
}

/// Independent reference: O(n²) Dijkstra, returns the shortest cost to every node.
fn reference_costs<G: SearchGraph>(g: &G, start: usize, allow: &dyn Fn(usize) -> bool) -> Vec<f64> {
    let n = g.node_count();
    let mut d = vec![f64::INFINITY; n];
    let mut done = vec![false; n];
    if !allow(start) {
        return d;
    }
    d[start] = 0.0;
    let mut nb = Vec::new();
    loop {
        let mut u = usize::MAX;
        for i in 0..n {
            if !done[i] && d[i].is_finite() && (u == usize::MAX || d[i] < d[u]) {
                u = i;
            }
        }
        if u == usize::MAX {
            return d;
        }
        done[u] = true;
        nb.clear();
        g.neighbors(u, &mut nb);
        for &v in &nb {
            if allow(v) {
                let c = d[u] + g.edge_cost(u, v);
                if c < d[v] {
                    d[v] = c;
                }
            }
        }
    }
}

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

/// Random 3D graph: points in a box with height variation, each joined to its 4 nearest
/// (symmetric).
fn random_graph(seed: u64, n: usize) -> Graph {
    let mut g = Lcg(seed);
    let pos: Vec<[f64; 3]> = (0..n)
        .map(|_| [g.next() * 20.0, g.next() * 20.0, g.next() * 4.0])
        .collect();
    let mut adj = vec![Vec::new(); n];
    for i in 0..n {
        let mut by: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        by.sort_by(|&a, &b| distance(pos[i], pos[a]).total_cmp(&distance(pos[i], pos[b])));
        for &j in by.iter().take(4) {
            if !adj[i].contains(&j) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    Graph {
        pos,
        adj,
        scale: 1.0,
    }
}

fn random_blocked(seed: u64, w: usize, h: usize) -> Vec<bool> {
    let mut g = Lcg(seed);
    (0..w * h).map(|_| g.next() < 0.25).collect()
}

fn assert_valid_path<G: SearchGraph>(g: &G, p: &[usize], start: usize, goal: usize) {
    assert_eq!(p.first(), Some(&start));
    assert_eq!(p.last(), Some(&goal));
    let mut nb = Vec::new();
    for w in p.windows(2) {
        nb.clear();
        g.neighbors(w[0], &mut nb);
        assert!(nb.contains(&w[1]), "{} → {} is not an edge", w[0], w[1]);
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

#[test]
fn astar_cost_equals_dijkstra_on_3d_graphs() {
    let mut compared = 0;
    for seed in 0..20 {
        let g = random_graph(seed, 60);
        for (s, t) in [(0, 59), (3, 41), (17, 8), (25, 52)] {
            let d = reference_costs(&g, s, &|_| true);
            match astar(&g, s, t) {
                Some(p) => {
                    assert_valid_path(&g, &p, s, t);
                    assert!(
                        close(path_cost(&g, &p), d[t]),
                        "seed {seed}: {} vs {}",
                        path_cost(&g, &p),
                        d[t]
                    );
                    compared += 1;
                }
                None => assert!(d[t].is_infinite(), "seed {seed}: missed a reachable goal"),
            }
        }
    }
    assert!(compared > 0, "0 compared");
}

#[test]
fn astar_cost_equals_dijkstra_on_occupancy_navmeshes() {
    let mut compared = 0;
    for seed in 0..20 {
        let (w, h) = (12, 9);
        let mesh =
            build_occupancy_navmesh(&random_blocked(seed, w, h), w, h, 0.5, Vec2::new(0.0, 0.0));
        let n = mesh.node_count();
        for (s, t) in [(0, n - 1), (n / 3, 2 * n / 3), (1, n / 2)] {
            let d = reference_costs(&mesh, s, &|_| true);
            match astar(&mesh, s, t) {
                Some(p) => {
                    assert_valid_path(&mesh, &p, s, t);
                    assert!(close(path_cost(&mesh, &p), d[t]), "seed {seed}");
                    compared += 1;
                }
                None => assert!(d[t].is_infinite(), "seed {seed}: missed a reachable goal"),
            }
        }
    }
    assert!(compared > 0, "0 compared");
}

#[test]
fn a_costlier_metric_still_gives_shortest_paths() {
    // costs 1.5 × distance satisfy the contract (>= distance); the path is still the cheapest
    let mut compared = 0;
    for seed in 0..10 {
        let mut g = random_graph(100 + seed, 50);
        g.scale = 1.5;
        let d = reference_costs(&g, 0, &|_| true);
        if let Some(p) = astar(&g, 0, 49) {
            assert!(close(path_cost(&g, &p), d[49]));
            compared += 1;
        }
    }
    assert!(compared > 0, "0 compared");
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "edge cost")]
fn a_cost_below_the_distance_is_caught_in_debug() {
    let mut g = random_graph(7, 30);
    g.scale = 0.5;
    let _ = astar(&g, 0, 29);
}

#[test]
fn the_result_does_not_depend_on_neighbour_order() {
    // a full grid has many equal-cost paths; the stated tie rule picks one, whatever the order
    let mesh = build_grid_navmesh(
        Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(8.0, 6.0)),
        8,
        6,
    );
    let a = astar(&mesh, 0, mesh.node_count() - 1).unwrap();
    let pos: Vec<[f64; 3]> = (0..mesh.node_count()).map(|i| mesh.position(i)).collect();
    let mut adj: Vec<Vec<usize>> = (0..mesh.node_count())
        .map(|i| {
            let mut v = Vec::new();
            mesh.neighbors(i, &mut v);
            v
        })
        .collect();
    let forward = Graph {
        pos: pos.clone(),
        adj: adj.clone(),
        scale: 1.0,
    };
    for v in &mut adj {
        v.reverse();
    }
    let reversed = Graph {
        pos,
        adj,
        scale: 1.0,
    };
    assert_eq!(astar(&forward, 0, mesh.node_count() - 1).unwrap(), a);
    assert_eq!(astar(&reversed, 0, mesh.node_count() - 1).unwrap(), a);
}

#[test]
fn filtered_search_stays_inside_and_is_shortest_there() {
    let mut compared = 0;
    for seed in 0..15 {
        let g = random_graph(200 + seed, 60);
        let allow = |n: usize| n % 5 != 3;
        let d = reference_costs(&g, 0, &allow);
        match astar_filtered(&g, 0, 59, allow) {
            Some(p) => {
                assert!(
                    p.iter().all(|&n| allow(n)),
                    "entered a node the filter forbids"
                );
                assert!(close(path_cost(&g, &p), d[59]));
                compared += 1;
            }
            None => assert!(d[59].is_infinite()),
        }
    }
    assert!(compared > 0, "0 compared");
    let g = random_graph(1, 20);
    assert_eq!(
        astar_filtered(&g, 0, 19, |n| n != 0),
        None,
        "a forbidden start"
    );
    assert_eq!(
        astar_filtered(&g, 0, 19, |n| n != 19),
        None,
        "a forbidden goal"
    );
}

#[test]
fn edge_cases() {
    let g = random_graph(3, 10);
    assert_eq!(astar(&g, 4, 4), Some(vec![4]));
    assert_eq!(astar(&g, 0, 10), None);
    assert_eq!(astar(&g, 10, 0), None);
    let two = Graph {
        pos: vec![[0.0; 3], [1.0, 0.0, 0.0]],
        adj: vec![vec![], vec![]],
        scale: 1.0,
    };
    assert_eq!(astar(&two, 0, 1), None, "disconnected");
}

#[test]
fn cluster_grid_closed_form() {
    let (count, map) = cluster_grid(5, 3, 2);
    assert_eq!(count, 3 * 2);
    assert_eq!(map.len(), 5 * 3 * 2);
    for y in 0..3 {
        for x in 0..5 {
            let c = (y / 2) * 3 + x / 2;
            assert_eq!(map[(y * 5 + x) * 2], c);
            assert_eq!(map[(y * 5 + x) * 2 + 1], c);
        }
    }
}

#[test]
fn hierarchical_paths_are_valid_and_never_shorter_than_optimal() {
    let (w, h) = (16, 12);
    let mut compared = 0;
    for seed in 0..10 {
        let mut blocked = vec![false; w * h];
        let mut g = Lcg(300 + seed);
        for b in blocked.iter_mut() {
            *b = g.next() < 0.15;
        }
        let mesh = build_occupancy_navmesh(&blocked, w, h, 1.0, Vec2::new(0.0, 0.0));
        // the cluster of each triangle, from the free cells in row-major order
        let (_, full) = cluster_grid(w, h, 4);
        let map: Vec<usize> = (0..w * h)
            .filter(|&c| !blocked[c])
            .flat_map(|c| [full[c * 2], full[c * 2 + 1]])
            .collect();
        assert_eq!(map.len(), mesh.node_count());
        let (s, t) = (0, mesh.node_count() - 1);
        let d = reference_costs(&mesh, s, &|_| true);
        if let Some(p) = hierarchical_astar(&mesh, s, t, &map) {
            assert_valid_path(&mesh, &p, s, t);
            assert!(path_cost(&mesh, &p) + 1e-9 >= d[t]);
            compared += 1;
        }
        // inside one cluster it is plain A*
        assert_eq!(hierarchical_astar(&mesh, 0, 1, &map), astar(&mesh, 0, 1));
    }
    assert!(compared > 0, "0 compared");
}

#[test]
fn occupancy_navmesh_layout() {
    let (w, h) = (5, 4);
    let mut blocked = vec![false; w * h];
    blocked[w + 2] = true;
    blocked[3 * w] = true;
    let mesh = build_occupancy_navmesh(&blocked, w, h, 0.5, Vec2::new(1.0, 2.0));
    assert_eq!(mesh.node_count(), 2 * (w * h - 2));
    // row-major free cells, (bl, br, tl) then (br, tr, tl)
    let mut k = 0;
    for y in 0..h {
        for x in 0..w {
            if blocked[y * w + x] {
                continue;
            }
            let (x0, y0) = (1.0 + x as f64 * 0.5, 2.0 + y as f64 * 0.5);
            let t0 = mesh.triangles[k].vertices;
            let t1 = mesh.triangles[k + 1].vertices;
            assert_eq!(
                [t0[0].x, t0[0].y, t0[1].x, t0[1].y, t0[2].x, t0[2].y],
                [x0, y0, x0 + 0.5, y0, x0, y0 + 0.5]
            );
            assert_eq!(
                [t1[0].x, t1[0].y, t1[1].x, t1[1].y, t1[2].x, t1[2].y],
                [x0 + 0.5, y0, x0 + 0.5, y0 + 0.5, x0, y0 + 0.5]
            );
            k += 2;
        }
    }
    // neighbours are symmetric and share exactly one edge (two vertices)
    let mut nb = Vec::new();
    for i in 0..mesh.node_count() {
        nb.clear();
        mesh.neighbors(i, &mut nb);
        for &j in &nb {
            let mut back = Vec::new();
            mesh.neighbors(j, &mut back);
            assert!(back.contains(&i), "{i} → {j} is one-way");
            let shared = mesh.triangles[i]
                .vertices
                .iter()
                .filter(|a| {
                    mesh.triangles[j]
                        .vertices
                        .iter()
                        .any(|b| a.x == b.x && a.y == b.y)
                })
                .count();
            assert_eq!(shared, 2);
        }
    }
    // an interior triangle of a full grid has 3 neighbours
    let full = build_occupancy_navmesh(&[false; 9], 3, 3, 1.0, Vec2::new(0.0, 0.0));
    let mut c = Vec::new();
    full.neighbors(8, &mut c); // cell (1, 1), first triangle
    assert_eq!(c.len(), 3);
}

#[test]
fn find_path_follows_the_shortest_triangle_chain() {
    let mesh = build_grid_navmesh(
        Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(6.0, 4.0)),
        6,
        4,
    );
    let (a, b) = (Vec2::new(0.2, 0.3), Vec2::new(5.8, 3.6));
    let path = mesh.find_path(a, b).unwrap();
    let (s, t) = (
        mesh.find_triangle(a).unwrap(),
        mesh.find_triangle(b).unwrap(),
    );
    let chain = astar(&mesh, s, t).unwrap();
    assert_eq!(path.len(), chain.len());
    for (k, &tri) in chain.iter().enumerate().skip(1).take(chain.len() - 2) {
        let c = mesh.triangles[tri].centroid();
        assert_eq!((path[k].x, path[k].y), (c.x, c.y));
    }
}

#[test]
fn arithmetic_is_restricted_to_correctly_rounded_operations() {
    // + − × ÷ and sqrt only: no transcendental and no fused operation in the search
    let src = include_str!("../src/graph_search.rs");
    let forbidden = [
        ".mul_add(",
        ".powf(",
        ".powi(",
        ".exp(",
        ".ln(",
        ".log",
        ".sin(",
        ".cos(",
        ".tan(",
        ".atan",
        ".hypot(",
        ".cbrt(",
        ".exp2(",
        ".asin(",
        ".acos(",
    ];
    let mut scanned = 0;
    for line in src.lines() {
        let code = line.split("//").next().unwrap_or("");
        if code.trim().is_empty() {
            continue;
        }
        scanned += 1;
        for f in forbidden {
            assert!(!code.contains(f), "forbidden operation `{f}` in: {line}");
        }
    }
    assert!(scanned > 50, "scanned only {scanned} code lines");
}

/// FNV-1a over the paths of a fixed corpus; the same value on every platform (CI: Linux, macOS,
/// Windows, wasm32-wasip1).
const GOLDEN_PATH_HASH: u64 = 0x88a4_dac4_0830_0556;

#[test]
fn path_hash_is_platform_independent() {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |v: u64| {
        for b in v.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    };
    let mut compared = 0;
    for seed in 500..520 {
        let g = random_graph(seed, 80);
        if let Some(p) = astar(&g, 0, 79) {
            for n in &p {
                eat(*n as u64);
            }
            eat(path_cost(&g, &p).to_bits());
            compared += p.len();
        }
    }
    eprintln!("[graph_search golden] hash {h:#018x} over {compared} path nodes");
    assert!(compared > 0, "0 path nodes compared");
    assert_eq!(h, GOLDEN_PATH_HASH);
}

/// Edge costs `distance × (1 + w)`, `w ∈ [0, 1)` per undirected edge: the contract holds, and the
/// first path that reaches a node is often not the cheapest one.
struct Weighted {
    g: Graph,
    w: std::collections::HashMap<(usize, usize), f64>,
}

impl SearchGraph for Weighted {
    fn node_count(&self) -> usize {
        self.g.node_count()
    }
    fn position(&self, n: usize) -> [f64; 3] {
        self.g.position(n)
    }
    fn neighbors(&self, n: usize, out: &mut Vec<usize>) {
        self.g.neighbors(n, out);
    }
    fn edge_cost(&self, a: usize, b: usize) -> f64 {
        distance(self.g.pos[a], self.g.pos[b]) * (1.0 + self.w[&(a.min(b), a.max(b))])
    }
}

fn weighted(seed: u64, n: usize) -> Weighted {
    let g = random_graph(seed, n);
    let mut r = Lcg(seed ^ 0xabcd);
    let mut w = std::collections::HashMap::new();
    for a in 0..n {
        for &b in &g.adj[a] {
            w.entry((a.min(b), a.max(b))).or_insert_with(|| r.next());
        }
    }
    Weighted { g, w }
}

#[test]
fn astar_cost_equals_dijkstra_with_uneven_edge_costs() {
    let mut compared = 0;
    for seed in 0..40 {
        let g = weighted(900 + seed, 70);
        for (s, t) in [(0, 69), (5, 33), (60, 2)] {
            let d = reference_costs(&g, s, &|_| true);
            match astar(&g, s, t) {
                Some(p) => {
                    assert_valid_path(&g, &p, s, t);
                    assert!(
                        close(path_cost(&g, &p), d[t]),
                        "seed {seed}: {} vs {}",
                        path_cost(&g, &p),
                        d[t]
                    );
                    compared += 1;
                }
                None => assert!(d[t].is_infinite()),
            }
        }
    }
    assert!(compared > 0, "0 compared");
}

#[test]
fn an_exact_tie_goes_to_the_smaller_parent_index() {
    // a diamond whose two routes cost exactly the same (all four edges are √2)
    let pos = vec![
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [2.0, 0.0, 0.0],
    ];
    for adj in [
        vec![vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2]],
        vec![vec![2, 1], vec![3, 0], vec![3, 0], vec![2, 1]],
    ] {
        let g = Graph {
            pos: pos.clone(),
            adj,
            scale: 1.0,
        };
        assert_eq!(astar(&g, 0, 3), Some(vec![0, 1, 3]));
        assert_eq!(astar(&g, 3, 0), Some(vec![3, 1, 0]));
    }
}

/// A graph whose node 0 lists a neighbour that does not exist.
struct Dangling(Graph);
impl SearchGraph for Dangling {
    fn node_count(&self) -> usize {
        self.0.node_count()
    }
    fn position(&self, n: usize) -> [f64; 3] {
        self.0.position(n)
    }
    fn neighbors(&self, n: usize, out: &mut Vec<usize>) {
        if n == 0 {
            out.extend([usize::MAX, self.0.node_count()]);
        }
        self.0.neighbors(n, out);
    }
}

#[test]
fn a_neighbour_out_of_range_is_ignored() {
    let g = random_graph(42, 30);
    let d = Dangling(g.clone());
    assert_eq!(astar(&d, 0, 29), astar(&g, 0, 29));
    let map: Vec<usize> = (0..30).map(|i| i / 10).collect();
    assert_eq!(
        hierarchical_astar(&d, 0, 29, &map),
        hierarchical_astar(&g, 0, 29, &map)
    );
}

#[test]
fn hierarchical_edge_cases() {
    let g = random_graph(5, 30);
    let map: Vec<usize> = (0..30).map(|i| i / 10).collect();
    assert_eq!(hierarchical_astar(&g, 0, 30, &map), None);
    assert_eq!(hierarchical_astar(&g, 30, 0, &map), None);
    assert_eq!(
        hierarchical_astar(&g, 0, 29, &map[..29]),
        None,
        "a cluster map shorter than the graph"
    );
    // start and goal in one cluster whose only connection runs through another cluster: plain A*
    // (a restriction to the one cluster would find nothing)
    let line = Graph {
        pos: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        adj: vec![vec![1], vec![0, 2], vec![1]],
        scale: 1.0,
    };
    assert_eq!(
        hierarchical_astar(&line, 0, 2, &[0, 1, 0]),
        Some(vec![0, 1, 2])
    );
}

#[test]
fn a_navmesh_is_searched_at_its_centroids() {
    let mesh = build_occupancy_navmesh(
        &[false, true, false, false],
        2,
        2,
        1.5,
        Vec2::new(-1.0, 3.0),
    );
    for i in 0..mesh.node_count() {
        let v = mesh.triangles[i].vertices;
        let c = [
            (v[0].x + v[1].x + v[2].x) / 3.0,
            (v[0].y + v[1].y + v[2].y) / 3.0,
            0.0,
        ];
        assert_eq!(mesh.position(i), c);
    }
}

#[test]
fn navmesh_triangles_closed_form() {
    // bounds (1, 2)–(7, 6), 3 × 2 cells of 2 × 2
    let mesh = build_grid_navmesh(
        Bounds2D::new(Vec2::new(1.0, 2.0), Vec2::new(7.0, 6.0)),
        3,
        2,
    );
    assert_eq!(mesh.triangles.len(), 12);
    for r in 0..2 {
        for c in 0..3 {
            let (x0, y0) = (1.0 + 2.0 * c as f64, 2.0 + 2.0 * r as f64);
            let t0 = mesh.triangles[(r * 3 + c) * 2].vertices;
            let t1 = mesh.triangles[(r * 3 + c) * 2 + 1].vertices;
            let xy = |t: [Vec2; 3]| [t[0].x, t[0].y, t[1].x, t[1].y, t[2].x, t[2].y];
            assert_eq!(xy(t0), [x0, y0, x0 + 2.0, y0, x0, y0 + 2.0]);
            assert_eq!(xy(t1), [x0 + 2.0, y0, x0 + 2.0, y0 + 2.0, x0, y0 + 2.0]);
        }
    }
    // the centroid is the mean of the vertices; containment by barycentric coordinates
    let t = mesh.triangles[0];
    let c = t.centroid();
    assert_eq!(
        (c.x, c.y),
        ((1.0 + 3.0 + 1.0) / 3.0, (2.0 + 2.0 + 4.0) / 3.0)
    );
    assert!(t.contains(c));
    assert!(t.contains(Vec2::new(1.0, 2.0)), "a vertex");
    assert!(t.contains(Vec2::new(2.0, 3.0)), "the hypotenuse");
    assert!(!t.contains(Vec2::new(2.6, 3.6)), "past the hypotenuse");
    assert!(!t.contains(Vec2::new(0.9, 2.5)), "left of the triangle");
    assert!(!t.contains(Vec2::new(1.5, 1.9)), "below the triangle");
    assert!((t.area() - 2.0).abs() < 1e-12);
}

/// Explicit edge costs at coincident positions (the heuristic is 0, the contract holds).
struct Table(Vec<Vec<(usize, f64)>>);
impl SearchGraph for Table {
    fn node_count(&self) -> usize {
        self.0.len()
    }
    fn position(&self, _: usize) -> [f64; 3] {
        [0.0; 3]
    }
    fn neighbors(&self, n: usize, out: &mut Vec<usize>) {
        out.extend(self.0[n].iter().map(|e| e.0));
    }
    fn edge_cost(&self, a: usize, b: usize) -> f64 {
        self.0[a].iter().find(|e| e.0 == b).unwrap().1
    }
}

#[test]
fn a_later_tie_from_a_smaller_parent_takes_over() {
    // 0 → 2 → 3 costs 5 + 15 and is found first (2 is popped at g = 5); 0 → 1 → 3 costs 10 + 10
    // and arrives later with the same total: the smaller parent (1) wins
    let t = Table(vec![
        vec![(1, 10.0), (2, 5.0)],
        vec![(0, 10.0), (3, 10.0)],
        vec![(0, 5.0), (3, 15.0)],
        vec![(1, 10.0), (2, 15.0)],
    ]);
    assert_eq!(astar(&t, 0, 3), Some(vec![0, 1, 3]));
}

#[test]
fn containment_in_a_scalene_triangle() {
    use alice_navigation::navmesh::NavTriangle;
    // vertices (0, 0), (4, 1), (1, 3): inside, on an edge, and outside past each edge
    let t = NavTriangle::new(
        Vec2::new(0.0, 0.0),
        Vec2::new(4.0, 1.0),
        Vec2::new(1.0, 3.0),
    );
    assert!(t.contains(Vec2::new(1.6, 1.3)), "the centroid");
    assert!(t.contains(Vec2::new(2.0, 0.5)), "the midpoint of an edge");
    assert!(
        !t.contains(Vec2::new(2.0, 0.3)),
        "below the edge (0, 0)–(4, 1)"
    );
    assert!(
        !t.contains(Vec2::new(3.0, 2.5)),
        "past the edge (4, 1)–(1, 3)"
    );
    assert!(
        !t.contains(Vec2::new(0.2, 1.5)),
        "left of the edge (0, 0)–(1, 3)"
    );
    assert!((t.area() - 5.5).abs() < 1e-12);
}

#[test]
fn grid_navmesh_equals_an_occupancy_navmesh_with_no_blocked_cell() {
    // same vertices and the same neighbour sets, on a non-square grid with an offset
    let (cols, rows) = (5, 3);
    let a = build_grid_navmesh(
        Bounds2D::new(Vec2::new(-2.0, 1.0), Vec2::new(8.0, 7.0)),
        cols,
        rows,
    );
    let b = build_occupancy_navmesh(
        &vec![false; cols * rows],
        cols,
        rows,
        2.0,
        Vec2::new(-2.0, 1.0),
    );
    assert_eq!(a.triangles.len(), b.triangles.len());
    for i in 0..a.triangles.len() {
        let (va, vb) = (a.triangles[i].vertices, b.triangles[i].vertices);
        for k in 0..3 {
            assert_eq!(
                (va[k].x, va[k].y),
                (vb[k].x, vb[k].y),
                "triangle {i} vertex {k}"
            );
        }
        let set = |m: &alice_navigation::navmesh::NavMesh| {
            let mut v = Vec::new();
            m.neighbors(i, &mut v);
            v.sort_unstable();
            v
        };
        assert_eq!(set(&a), set(&b), "neighbours of triangle {i}");
    }
}
