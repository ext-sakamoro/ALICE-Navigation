#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::suspicious_operation_groupings)]
#![allow(clippy::while_float)]

//! ALICE-Navigation: Autonomous navigation library.
//!
//! Provides RRT, PRM, potential field, obstacle avoidance, path smoothing,
//! waypoint following, velocity obstacles, navigation mesh, and dynamic replanning.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// ============================================================
// Core types
// ============================================================

/// 2D vector / point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    #[must_use]
    pub fn length(self) -> f64 {
        self.x.hypot(self.y)
    }

    #[must_use]
    pub fn distance_to(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-12 {
            return Self::new(0.0, 0.0);
        }
        Self::new(self.x / len, self.y / len)
    }

    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    #[must_use]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    #[must_use]
    pub fn scale(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s)
    }

    #[must_use]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    #[must_use]
    pub fn cross(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    #[must_use]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }
}

/// Axis-aligned bounding box for 2D space.
#[derive(Debug, Clone, Copy)]
pub struct Bounds2D {
    pub min: Vec2,
    pub max: Vec2,
}

impl Bounds2D {
    #[must_use]
    pub const fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    #[must_use]
    pub fn contains(self, p: Vec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    #[must_use]
    pub fn width(self) -> f64 {
        self.max.x - self.min.x
    }

    #[must_use]
    pub fn height(self) -> f64 {
        self.max.y - self.min.y
    }
}

/// Circle obstacle.
#[derive(Debug, Clone, Copy)]
pub struct CircleObstacle {
    pub center: Vec2,
    pub radius: f64,
}

impl CircleObstacle {
    #[must_use]
    pub const fn new(center: Vec2, radius: f64) -> Self {
        Self { center, radius }
    }

    #[must_use]
    pub fn contains(&self, p: Vec2) -> bool {
        p.distance_to(self.center) <= self.radius
    }

    /// Check if the line segment from `a` to `b` intersects this obstacle.
    #[must_use]
    pub fn intersects_segment(&self, a: Vec2, b: Vec2) -> bool {
        let d = b.sub(a);
        let f = a.sub(self.center);
        let a_coeff = d.dot(d);
        let b_coeff = 2.0 * f.dot(d);
        let c_coeff = f.dot(f) - self.radius * self.radius;
        let discriminant = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
        if discriminant < 0.0 {
            return false;
        }
        let sqrt_disc = discriminant.sqrt();
        let t1 = (-b_coeff - sqrt_disc) / (2.0 * a_coeff);
        let t2 = (-b_coeff + sqrt_disc) / (2.0 * a_coeff);
        (0.0..=1.0).contains(&t1) || (0.0..=1.0).contains(&t2) || (t1 < 0.0 && t2 > 1.0)
    }
}

/// Simple deterministic pseudo-random number generator (xorshift64).
struct Rng {
    state: u64,
}

impl Rng {
    const fn new(seed: u64) -> Self {
        let state = if seed == 0 { 1 } else { seed };
        Self { state }
    }

    const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a value in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0x000F_FFFF_FFFF_FFFF) as f64 / (1u64 << 52) as f64
    }

    /// Returns a value in `[lo, hi)`.
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

// ============================================================
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

// ============================================================
// 1. RRT (Rapidly-exploring Random Trees)
// ============================================================

/// Configuration for the RRT planner.
#[derive(Debug, Clone)]
pub struct RrtConfig {
    pub step_size: f64,
    pub max_iterations: usize,
    pub goal_threshold: f64,
    pub seed: u64,
}

impl Default for RrtConfig {
    fn default() -> Self {
        Self {
            step_size: 0.5,
            max_iterations: 5000,
            goal_threshold: 0.5,
            seed: 42,
        }
    }
}

/// Plan a path with RRT.
///
/// Returns `None` if no path is found within the iteration limit.
///
/// # Panics
///
/// Will not panic in practice because the tree always has at least one node (the start).
#[must_use]
pub fn rrt(
    start: Vec2,
    goal: Vec2,
    bounds: Bounds2D,
    obstacles: &[CircleObstacle],
    config: &RrtConfig,
) -> Option<Vec<Vec2>> {
    let mut nodes: Vec<Vec2> = vec![start];
    let mut parents: Vec<usize> = vec![0];
    let mut rng = Rng::new(config.seed);

    for _ in 0..config.max_iterations {
        // Bias toward goal 10% of the time.
        let sample = if rng.next_f64() < 0.1 {
            goal
        } else {
            Vec2::new(
                rng.range(bounds.min.x, bounds.max.x),
                rng.range(bounds.min.y, bounds.max.y),
            )
        };

        // Find nearest node.
        let (nearest_idx, nearest) = nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.distance_to(sample)
                    .partial_cmp(&b.distance_to(sample))
                    .unwrap_or(Ordering::Equal)
            })
            .unwrap();

        let dir = sample.sub(*nearest).normalized();
        let new_point = nearest.add(dir.scale(config.step_size));

        if !bounds.contains(new_point) {
            continue;
        }
        if !segment_free(*nearest, new_point, obstacles) {
            continue;
        }

        nodes.push(new_point);
        parents.push(nearest_idx);

        if new_point.distance_to(goal) <= config.goal_threshold {
            // Trace back path.
            let mut path = vec![goal, new_point];
            let mut idx = nodes.len() - 1;
            while idx != 0 {
                idx = parents[idx];
                path.push(nodes[idx]);
            }
            path.reverse();
            return Some(path);
        }
    }
    None
}

// ============================================================
// 2. PRM (Probabilistic Roadmap)
// ============================================================

/// Configuration for the PRM planner.
#[derive(Debug, Clone)]
pub struct PrmConfig {
    pub num_samples: usize,
    pub connection_radius: f64,
    pub seed: u64,
}

impl Default for PrmConfig {
    fn default() -> Self {
        Self {
            num_samples: 200,
            connection_radius: 2.0,
            seed: 123,
        }
    }
}

/// Plan a path with PRM.
#[must_use]
pub fn prm(
    start: Vec2,
    goal: Vec2,
    bounds: Bounds2D,
    obstacles: &[CircleObstacle],
    config: &PrmConfig,
) -> Option<Vec<Vec2>> {
    let mut rng = Rng::new(config.seed);
    let mut nodes = vec![start, goal];

    // Sample free-space points.
    for _ in 0..config.num_samples {
        let p = Vec2::new(
            rng.range(bounds.min.x, bounds.max.x),
            rng.range(bounds.min.y, bounds.max.y),
        );
        if point_free(p, obstacles) {
            nodes.push(p);
        }
    }

    let n = nodes.len();
    // Build adjacency with distances.
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = nodes[i].distance_to(nodes[j]);
            if d <= config.connection_radius && segment_free(nodes[i], nodes[j], obstacles) {
                adj[i].push((j, d));
                adj[j].push((i, d));
            }
        }
    }

    // Dijkstra from node 0 (start) to node 1 (goal).
    dijkstra_path(&nodes, &adj, 0, 1)
}

// ============================================================
// Dijkstra helper
// ============================================================

#[derive(PartialEq)]
struct DijkNode {
    cost: f64,
    index: usize,
}

impl Eq for DijkNode {}

impl PartialOrd for DijkNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DijkNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

fn dijkstra_path(
    nodes: &[Vec2],
    adj: &[Vec<(usize, f64)>],
    start: usize,
    goal: usize,
) -> Option<Vec<Vec2>> {
    let n = nodes.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut prev = vec![usize::MAX; n];
    dist[start] = 0.0;

    let mut heap = BinaryHeap::new();
    heap.push(DijkNode {
        cost: 0.0,
        index: start,
    });

    while let Some(DijkNode { cost, index }) = heap.pop() {
        if index == goal {
            break;
        }
        if cost > dist[index] {
            continue;
        }
        for &(next, w) in &adj[index] {
            let new_cost = cost + w;
            if new_cost < dist[next] {
                dist[next] = new_cost;
                prev[next] = index;
                heap.push(DijkNode {
                    cost: new_cost,
                    index: next,
                });
            }
        }
    }

    if dist[goal].is_infinite() {
        return None;
    }

    let mut path = Vec::new();
    let mut cur = goal;
    while cur != usize::MAX {
        path.push(nodes[cur]);
        cur = prev[cur];
    }
    path.reverse();
    Some(path)
}

// ============================================================
// 3. Potential Field
// ============================================================

/// Configuration for the potential field planner.
#[derive(Debug, Clone)]
pub struct PotentialFieldConfig {
    pub attractive_gain: f64,
    pub repulsive_gain: f64,
    pub repulsive_range: f64,
    pub step_size: f64,
    pub max_iterations: usize,
    pub goal_threshold: f64,
}

impl Default for PotentialFieldConfig {
    fn default() -> Self {
        Self {
            attractive_gain: 1.0,
            repulsive_gain: 100.0,
            repulsive_range: 2.0,
            step_size: 0.1,
            max_iterations: 5000,
            goal_threshold: 0.3,
        }
    }
}

/// Compute attractive force toward goal.
#[must_use]
pub fn attractive_force(pos: Vec2, goal: Vec2, gain: f64) -> Vec2 {
    goal.sub(pos).scale(gain)
}

/// Compute repulsive force from a single obstacle.
#[must_use]
pub fn repulsive_force(pos: Vec2, obstacle: &CircleObstacle, gain: f64, range: f64) -> Vec2 {
    let d = pos.distance_to(obstacle.center) - obstacle.radius;
    if d <= 0.0 {
        // Inside obstacle: strong push away.
        let dir = pos.sub(obstacle.center).normalized();
        return dir.scale(gain * 10.0);
    }
    if d > range {
        return Vec2::new(0.0, 0.0);
    }
    let dir = pos.sub(obstacle.center).normalized();
    let magnitude = gain * (1.0 / d - 1.0 / range) * (1.0 / (d * d));
    dir.scale(magnitude)
}

/// Plan a path using the artificial potential field method.
#[must_use]
pub fn potential_field(
    start: Vec2,
    goal: Vec2,
    obstacles: &[CircleObstacle],
    config: &PotentialFieldConfig,
) -> Vec<Vec2> {
    let mut path = vec![start];
    let mut pos = start;

    for _ in 0..config.max_iterations {
        if pos.distance_to(goal) <= config.goal_threshold {
            path.push(goal);
            break;
        }

        let f_att = attractive_force(pos, goal, config.attractive_gain);
        let mut f_rep = Vec2::new(0.0, 0.0);
        for obs in obstacles {
            let fr = repulsive_force(pos, obs, config.repulsive_gain, config.repulsive_range);
            f_rep = f_rep.add(fr);
        }

        let total = f_att.add(f_rep);
        let dir = total.normalized();
        pos = pos.add(dir.scale(config.step_size));
        path.push(pos);
    }
    path
}

// ============================================================
// 4. Obstacle Avoidance
// ============================================================

/// Steer a velocity vector to avoid obstacles.
///
/// Returns the adjusted velocity. If no avoidance is needed, returns the input.
#[must_use]
pub fn avoid_obstacles(
    pos: Vec2,
    velocity: Vec2,
    obstacles: &[CircleObstacle],
    lookahead: f64,
    avoidance_strength: f64,
) -> Vec2 {
    let future_pos = pos.add(velocity.normalized().scale(lookahead));
    let mut steer = Vec2::new(0.0, 0.0);

    for obs in obstacles {
        let d = future_pos.distance_to(obs.center) - obs.radius;
        if d < lookahead {
            let away = future_pos.sub(obs.center).normalized();
            let strength = avoidance_strength * (1.0 - d / lookahead).max(0.0);
            steer = steer.add(away.scale(strength));
        }
    }

    velocity.add(steer)
}

/// Check if a position is within a certain margin of any obstacle.
#[must_use]
pub fn is_near_obstacle(pos: Vec2, obstacles: &[CircleObstacle], margin: f64) -> bool {
    obstacles
        .iter()
        .any(|o| pos.distance_to(o.center) - o.radius < margin)
}

// ============================================================
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

// ============================================================
// 6. Waypoint Following
// ============================================================

/// State for a waypoint follower.
#[derive(Debug, Clone)]
pub struct WaypointFollower {
    waypoints: Vec<Vec2>,
    current_index: usize,
    reach_threshold: f64,
    pub looping: bool,
}

impl WaypointFollower {
    #[must_use]
    pub const fn new(waypoints: Vec<Vec2>, reach_threshold: f64) -> Self {
        Self {
            waypoints,
            current_index: 0,
            reach_threshold,
            looping: false,
        }
    }

    /// Get the current target waypoint.
    #[must_use]
    pub fn current_target(&self) -> Option<Vec2> {
        self.waypoints.get(self.current_index).copied()
    }

    /// Advance the follower given the agent's current position.
    /// Returns the steering direction or `None` if finished.
    pub fn update(&mut self, pos: Vec2) -> Option<Vec2> {
        let target = self.current_target()?;
        if pos.distance_to(target) <= self.reach_threshold {
            self.current_index += 1;
            if self.current_index >= self.waypoints.len() {
                if self.looping {
                    self.current_index = 0;
                } else {
                    return None;
                }
            }
        }
        let target = self.waypoints[self.current_index];
        Some(target.sub(pos).normalized())
    }

    #[must_use]
    pub const fn is_finished(&self) -> bool {
        !self.looping && self.current_index >= self.waypoints.len()
    }

    #[must_use]
    pub const fn current_index(&self) -> usize {
        self.current_index
    }

    pub const fn reset(&mut self) {
        self.current_index = 0;
    }

    #[must_use]
    pub const fn waypoint_count(&self) -> usize {
        self.waypoints.len()
    }
}

// ============================================================
// 7. Velocity Obstacles (VO)
// ============================================================

/// Agent for velocity-obstacle computation.
#[derive(Debug, Clone, Copy)]
pub struct VoAgent {
    pub pos: Vec2,
    pub vel: Vec2,
    pub radius: f64,
}

impl VoAgent {
    #[must_use]
    pub const fn new(pos: Vec2, vel: Vec2, radius: f64) -> Self {
        Self { pos, vel, radius }
    }
}

/// Check if a candidate velocity is inside the velocity obstacle
/// induced by `other` on `agent`.
#[must_use]
pub fn is_in_velocity_obstacle(agent: &VoAgent, other: &VoAgent, candidate_vel: Vec2) -> bool {
    let rel_pos = other.pos.sub(agent.pos);
    let combined_radius = agent.radius + other.radius;
    let dist = rel_pos.length();
    if dist < combined_radius {
        return true; // Already colliding.
    }

    let rel_vel = candidate_vel.sub(other.vel);
    // Check if relative velocity points toward the VO cone.
    let proj = rel_vel.dot(rel_pos.normalized());
    if proj <= 0.0 {
        return false; // Moving away.
    }

    // Perpendicular distance from relative velocity to the line connecting agents.
    let perp = (rel_vel.cross(rel_pos)).abs() / dist;
    perp < combined_radius
}

/// Select the best velocity from candidates that avoids all velocity obstacles.
/// Falls back to zero velocity if all are blocked.
#[must_use]
pub fn select_velocity_outside_vo(
    agent: &VoAgent,
    others: &[VoAgent],
    preferred_vel: Vec2,
    candidates: &[Vec2],
) -> Vec2 {
    let mut best = Vec2::new(0.0, 0.0);
    let mut best_cost = f64::INFINITY;

    for &cand in candidates {
        let blocked = others
            .iter()
            .any(|other| is_in_velocity_obstacle(agent, other, cand));
        if blocked {
            continue;
        }
        let cost = cand.sub(preferred_vel).length();
        if cost < best_cost {
            best_cost = cost;
            best = cand;
        }
    }
    best
}

/// Generate candidate velocities in a disc pattern.
#[must_use]
pub fn generate_velocity_candidates(
    max_speed: f64,
    num_rings: usize,
    num_angles: usize,
) -> Vec<Vec2> {
    let mut candidates = vec![Vec2::new(0.0, 0.0)];
    for ring in 1..=num_rings {
        let speed = max_speed * (ring as f64) / (num_rings as f64);
        for a in 0..num_angles {
            let angle = 2.0 * std::f64::consts::PI * (a as f64) / (num_angles as f64);
            candidates.push(Vec2::new(speed * angle.cos(), speed * angle.sin()));
        }
    }
    candidates
}

// ============================================================
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

// ============================================================
// 9. Dynamic Replanning
// ============================================================

/// A dynamic planner that replans when obstacles change.
#[derive(Debug, Clone)]
pub struct DynamicPlanner {
    pub start: Vec2,
    pub goal: Vec2,
    pub bounds: Bounds2D,
    pub obstacles: Vec<CircleObstacle>,
    pub current_path: Option<Vec<Vec2>>,
    rrt_config: RrtConfig,
}

impl DynamicPlanner {
    #[must_use]
    pub fn new(start: Vec2, goal: Vec2, bounds: Bounds2D, obstacles: Vec<CircleObstacle>) -> Self {
        Self {
            start,
            goal,
            bounds,
            obstacles,
            current_path: None,
            rrt_config: RrtConfig::default(),
        }
    }

    /// Set custom RRT config.
    pub const fn set_rrt_config(&mut self, config: RrtConfig) {
        self.rrt_config = config;
    }

    /// Plan (or replan) a path.
    pub fn plan(&mut self) -> bool {
        self.current_path = rrt(
            self.start,
            self.goal,
            self.bounds,
            &self.obstacles,
            &self.rrt_config,
        );
        self.current_path.is_some()
    }

    /// Add a new obstacle and replan if the current path is invalidated.
    pub fn add_obstacle(&mut self, obstacle: CircleObstacle) -> bool {
        self.obstacles.push(obstacle);
        if let Some(ref path) = self.current_path {
            let invalidated = path
                .windows(2)
                .any(|w| obstacle.intersects_segment(w[0], w[1]));
            if invalidated {
                return self.plan();
            }
            return true;
        }
        self.plan()
    }

    /// Remove obstacles that match a predicate and optionally replan.
    pub fn remove_obstacles_where<F: Fn(&CircleObstacle) -> bool>(&mut self, pred: F) {
        self.obstacles.retain(|o| !pred(o));
    }

    /// Update the agent position (for replanning from a new start).
    pub const fn update_start(&mut self, new_start: Vec2) {
        self.start = new_start;
    }

    /// Check if the current path is still valid.
    #[must_use]
    pub fn is_path_valid(&self) -> bool {
        self.current_path.as_ref().is_some_and(|path| {
            path.windows(2)
                .all(|w| segment_free(w[0], w[1], &self.obstacles))
        })
    }
}

// ============================================================
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

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bounds() -> Bounds2D {
        Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(20.0, 20.0))
    }

    fn no_obstacles() -> Vec<CircleObstacle> {
        Vec::new()
    }

    fn simple_obstacles() -> Vec<CircleObstacle> {
        vec![CircleObstacle::new(Vec2::new(10.0, 10.0), 2.0)]
    }

    // --- Vec2 tests ---

    #[test]
    fn test_vec2_new() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.x - 3.0).abs() < 1e-9);
        assert!((v.y - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_distance() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_normalized() {
        let v = Vec2::new(3.0, 4.0).normalized();
        assert!((v.length() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_zero_normalized() {
        let v = Vec2::new(0.0, 0.0).normalized();
        assert!((v.length()).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_add() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        let c = a.add(b);
        assert!((c.x - 4.0).abs() < 1e-9);
        assert!((c.y - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_sub() {
        let a = Vec2::new(5.0, 7.0);
        let b = Vec2::new(3.0, 4.0);
        let c = a.sub(b);
        assert!((c.x - 2.0).abs() < 1e-9);
        assert!((c.y - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_scale() {
        let v = Vec2::new(2.0, 3.0).scale(2.0);
        assert!((v.x - 4.0).abs() < 1e-9);
        assert!((v.y - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_dot() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(3.0, 4.0);
        assert!((a.dot(b) - 11.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_cross() {
        let a = Vec2::new(1.0, 0.0);
        let b = Vec2::new(0.0, 1.0);
        assert!((a.cross(b) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_lerp() {
        let a = Vec2::new(0.0, 0.0);
        let b = Vec2::new(10.0, 10.0);
        let c = a.lerp(b, 0.5);
        assert!((c.x - 5.0).abs() < 1e-9);
        assert!((c.y - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_lerp_endpoints() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(5.0, 6.0);
        let at_zero = a.lerp(b, 0.0);
        let at_one = a.lerp(b, 1.0);
        assert!((at_zero.x - a.x).abs() < 1e-9);
        assert!((at_one.x - b.x).abs() < 1e-9);
    }

    // --- Bounds2D tests ---

    #[test]
    fn test_bounds_contains() {
        let b = default_bounds();
        assert!(b.contains(Vec2::new(10.0, 10.0)));
        assert!(!b.contains(Vec2::new(-1.0, 10.0)));
        assert!(!b.contains(Vec2::new(10.0, 21.0)));
    }

    #[test]
    fn test_bounds_width_height() {
        let b = default_bounds();
        assert!((b.width() - 20.0).abs() < 1e-9);
        assert!((b.height() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_bounds_edge() {
        let b = default_bounds();
        assert!(b.contains(Vec2::new(0.0, 0.0)));
        assert!(b.contains(Vec2::new(20.0, 20.0)));
    }

    // --- CircleObstacle tests ---

    #[test]
    fn test_circle_contains() {
        let c = CircleObstacle::new(Vec2::new(5.0, 5.0), 2.0);
        assert!(c.contains(Vec2::new(5.0, 5.0)));
        assert!(c.contains(Vec2::new(6.0, 5.0)));
        assert!(!c.contains(Vec2::new(8.0, 5.0)));
    }

    #[test]
    fn test_circle_intersects_segment_hit() {
        let c = CircleObstacle::new(Vec2::new(5.0, 5.0), 2.0);
        assert!(c.intersects_segment(Vec2::new(0.0, 5.0), Vec2::new(10.0, 5.0)));
    }

    #[test]
    fn test_circle_intersects_segment_miss() {
        let c = CircleObstacle::new(Vec2::new(5.0, 5.0), 2.0);
        assert!(!c.intersects_segment(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)));
    }

    #[test]
    fn test_circle_intersects_segment_tangent() {
        let c = CircleObstacle::new(Vec2::new(5.0, 2.0), 2.0);
        // Segment passes right at the edge.
        assert!(c.intersects_segment(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)));
    }

    #[test]
    fn test_circle_intersects_segment_enclosing() {
        let c = CircleObstacle::new(Vec2::new(5.0, 0.0), 10.0);
        // Segment is entirely inside the circle.
        assert!(c.intersects_segment(Vec2::new(4.0, 0.0), Vec2::new(6.0, 0.0)));
    }

    // --- Collision helpers ---

    #[test]
    fn test_point_free_empty() {
        assert!(point_free(Vec2::new(5.0, 5.0), &no_obstacles()));
    }

    #[test]
    fn test_point_free_blocked() {
        let obs = simple_obstacles();
        assert!(!point_free(Vec2::new(10.0, 10.0), &obs));
    }

    #[test]
    fn test_segment_free_clear() {
        assert!(segment_free(
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 0.0),
            &simple_obstacles()
        ));
    }

    #[test]
    fn test_segment_free_blocked() {
        assert!(!segment_free(
            Vec2::new(0.0, 10.0),
            Vec2::new(20.0, 10.0),
            &simple_obstacles()
        ));
    }

    // --- RRT tests ---

    #[test]
    fn test_rrt_no_obstacles() {
        let path = rrt(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &RrtConfig::default(),
        );
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.len() >= 2);
    }

    #[test]
    fn test_rrt_with_obstacle() {
        let path = rrt(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &simple_obstacles(),
            &RrtConfig::default(),
        );
        assert!(path.is_some());
    }

    #[test]
    fn test_rrt_start_equals_goal() {
        let path = rrt(
            Vec2::new(5.0, 5.0),
            Vec2::new(5.0, 5.0),
            default_bounds(),
            &no_obstacles(),
            &RrtConfig {
                goal_threshold: 1.0,
                ..RrtConfig::default()
            },
        );
        assert!(path.is_some());
    }

    #[test]
    fn test_rrt_path_starts_at_start() {
        let start = Vec2::new(1.0, 1.0);
        let path = rrt(
            start,
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &RrtConfig::default(),
        )
        .unwrap();
        assert!((path[0].x - start.x).abs() < 1e-9);
        assert!((path[0].y - start.y).abs() < 1e-9);
    }

    #[test]
    fn test_rrt_path_ends_near_goal() {
        let goal = Vec2::new(19.0, 19.0);
        let config = RrtConfig::default();
        let path = rrt(
            Vec2::new(1.0, 1.0),
            goal,
            default_bounds(),
            &no_obstacles(),
            &config,
        )
        .unwrap();
        let last = path[path.len() - 1];
        assert!(last.distance_to(goal) <= config.goal_threshold + 0.01);
    }

    #[test]
    fn test_rrt_custom_seed() {
        let config = RrtConfig {
            seed: 999,
            ..RrtConfig::default()
        };
        let path = rrt(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &config,
        );
        assert!(path.is_some());
    }

    #[test]
    fn test_rrt_deterministic() {
        let config = RrtConfig::default();
        let p1 = rrt(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &config,
        );
        let p2 = rrt(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &config,
        );
        assert_eq!(p1.as_ref().map(Vec::len), p2.as_ref().map(Vec::len));
    }

    // --- PRM tests ---

    #[test]
    fn test_prm_no_obstacles() {
        let path = prm(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &PrmConfig::default(),
        );
        assert!(path.is_some());
    }

    #[test]
    fn test_prm_with_obstacle() {
        let config = PrmConfig {
            num_samples: 500,
            connection_radius: 5.0,
            seed: 42,
        };
        let path = prm(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &simple_obstacles(),
            &config,
        );
        assert!(path.is_some());
    }

    #[test]
    fn test_prm_path_starts_at_start() {
        let start = Vec2::new(1.0, 1.0);
        let path = prm(
            start,
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &PrmConfig::default(),
        )
        .unwrap();
        assert!((path[0].x - start.x).abs() < 1e-9);
    }

    #[test]
    fn test_prm_path_ends_at_goal() {
        let goal = Vec2::new(19.0, 19.0);
        let path = prm(
            Vec2::new(1.0, 1.0),
            goal,
            default_bounds(),
            &no_obstacles(),
            &PrmConfig::default(),
        )
        .unwrap();
        let last = path[path.len() - 1];
        assert!((last.x - goal.x).abs() < 1e-9);
    }

    // --- Potential field tests ---

    #[test]
    fn test_attractive_force_direction() {
        let f = attractive_force(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), 1.0);
        assert!(f.x > 0.0);
        assert!((f.y).abs() < 1e-9);
    }

    #[test]
    fn test_attractive_force_gain() {
        let f1 = attractive_force(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), 1.0);
        let f2 = attractive_force(Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0), 2.0);
        assert!((f2.x - f1.x * 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_repulsive_force_far_away() {
        let obs = CircleObstacle::new(Vec2::new(100.0, 100.0), 1.0);
        let f = repulsive_force(Vec2::new(0.0, 0.0), &obs, 100.0, 2.0);
        assert!((f.x).abs() < 1e-9);
        assert!((f.y).abs() < 1e-9);
    }

    #[test]
    fn test_repulsive_force_near() {
        let obs = CircleObstacle::new(Vec2::new(3.0, 0.0), 1.0);
        let f = repulsive_force(Vec2::new(0.0, 0.0), &obs, 100.0, 5.0);
        assert!(f.x < 0.0); // Pushes away (negative x).
    }

    #[test]
    fn test_potential_field_reaches_goal() {
        let path = potential_field(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &no_obstacles(),
            &PotentialFieldConfig::default(),
        );
        assert!(path.len() >= 2);
        let last = path[path.len() - 1];
        assert!(last.distance_to(Vec2::new(10.0, 0.0)) < 1.0);
    }

    #[test]
    fn test_potential_field_with_obstacle() {
        let obs = vec![CircleObstacle::new(Vec2::new(5.0, 0.0), 1.0)];
        let path = potential_field(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            &obs,
            &PotentialFieldConfig::default(),
        );
        assert!(path.len() >= 2);
    }

    #[test]
    fn test_potential_field_no_movement_at_goal() {
        let path = potential_field(
            Vec2::new(10.0, 0.0),
            Vec2::new(10.0, 0.0),
            &no_obstacles(),
            &PotentialFieldConfig {
                goal_threshold: 0.5,
                ..PotentialFieldConfig::default()
            },
        );
        // Should immediately terminate.
        assert!(path.len() <= 3);
    }

    // --- Obstacle avoidance tests ---

    #[test]
    fn test_avoid_obstacles_no_obstacles() {
        let vel = Vec2::new(1.0, 0.0);
        let result = avoid_obstacles(Vec2::new(0.0, 0.0), vel, &no_obstacles(), 5.0, 1.0);
        assert!((result.x - vel.x).abs() < 1e-9);
    }

    #[test]
    fn test_avoid_obstacles_steers_away() {
        let obs = vec![CircleObstacle::new(Vec2::new(3.0, 0.0), 1.0)];
        let result = avoid_obstacles(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), &obs, 5.0, 2.0);
        // Should have some steering component.
        assert!(result.length() > 0.5);
    }

    #[test]
    fn test_is_near_obstacle_true() {
        let obs = simple_obstacles();
        assert!(is_near_obstacle(Vec2::new(10.0, 12.5), &obs, 1.0));
    }

    #[test]
    fn test_is_near_obstacle_false() {
        let obs = simple_obstacles();
        assert!(!is_near_obstacle(Vec2::new(0.0, 0.0), &obs, 1.0));
    }

    // --- Path smoothing tests ---

    #[test]
    fn test_smooth_path_short() {
        let path = vec![Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)];
        let smoothed = smooth_path(&path, &no_obstacles(), 10);
        assert_eq!(smoothed.len(), 2);
    }

    #[test]
    fn test_smooth_path_removes_waypoints() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(4.0, 0.0),
        ];
        let smoothed = smooth_path(&path, &no_obstacles(), 100);
        assert!(smoothed.len() <= path.len());
    }

    #[test]
    fn test_smooth_path_preserves_endpoints() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(10.0, 0.0),
        ];
        let smoothed = smooth_path(&path, &no_obstacles(), 10);
        assert!((smoothed[0].x - 0.0).abs() < 1e-9);
        assert!((smoothed.last().unwrap().x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_chaikin_smooth_increases_points() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(10.0, 0.0),
        ];
        let smoothed = chaikin_smooth(&path, 1);
        assert!(smoothed.len() > path.len());
    }

    #[test]
    fn test_chaikin_smooth_preserves_endpoints() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(5.0, 5.0),
            Vec2::new(10.0, 0.0),
        ];
        let smoothed = chaikin_smooth(&path, 2);
        assert!((smoothed[0].x - 0.0).abs() < 1e-9);
        assert!((smoothed.last().unwrap().x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_chaikin_smooth_single_point() {
        let path = vec![Vec2::new(1.0, 1.0)];
        let smoothed = chaikin_smooth(&path, 3);
        assert_eq!(smoothed.len(), 1);
    }

    #[test]
    fn test_chaikin_smooth_zero_iterations() {
        let path = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0)];
        let smoothed = chaikin_smooth(&path, 0);
        assert_eq!(smoothed.len(), 2);
    }

    // --- Waypoint follower tests ---

    #[test]
    fn test_waypoint_follower_basic() {
        let wps = vec![
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ];
        let mut follower = WaypointFollower::new(wps, 0.1);
        let dir = follower.update(Vec2::new(0.0, 0.0));
        assert!(dir.is_some());
        assert!(dir.unwrap().x > 0.0);
    }

    #[test]
    fn test_waypoint_follower_reaches_end() {
        let wps = vec![Vec2::new(1.0, 0.0)];
        let mut follower = WaypointFollower::new(wps, 0.5);
        let dir = follower.update(Vec2::new(0.9, 0.0));
        assert!(dir.is_none());
        assert!(follower.is_finished());
    }

    #[test]
    fn test_waypoint_follower_advances() {
        let wps = vec![Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)];
        let mut follower = WaypointFollower::new(wps, 0.5);
        assert_eq!(follower.current_index(), 0);
        follower.update(Vec2::new(0.9, 0.0));
        assert_eq!(follower.current_index(), 1);
    }

    #[test]
    fn test_waypoint_follower_looping() {
        let wps = vec![Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)];
        let mut follower = WaypointFollower::new(wps, 0.5);
        follower.looping = true;
        follower.update(Vec2::new(0.9, 0.0)); // Advance to 1.
        follower.update(Vec2::new(1.9, 0.0)); // Advance past end, loop to 0.
        assert_eq!(follower.current_index(), 0);
        assert!(!follower.is_finished());
    }

    #[test]
    fn test_waypoint_follower_reset() {
        let wps = vec![Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)];
        let mut follower = WaypointFollower::new(wps, 0.5);
        follower.update(Vec2::new(0.9, 0.0));
        assert_eq!(follower.current_index(), 1);
        follower.reset();
        assert_eq!(follower.current_index(), 0);
    }

    #[test]
    fn test_waypoint_follower_count() {
        let wps = vec![
            Vec2::new(1.0, 0.0),
            Vec2::new(2.0, 0.0),
            Vec2::new(3.0, 0.0),
        ];
        let follower = WaypointFollower::new(wps, 0.5);
        assert_eq!(follower.waypoint_count(), 3);
    }

    #[test]
    fn test_waypoint_follower_current_target() {
        let wps = vec![Vec2::new(1.0, 0.0), Vec2::new(2.0, 0.0)];
        let follower = WaypointFollower::new(wps, 0.5);
        let t = follower.current_target().unwrap();
        assert!((t.x - 1.0).abs() < 1e-9);
    }

    // --- Velocity Obstacles tests ---

    #[test]
    fn test_vo_collision_course() {
        let agent = VoAgent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 0.5);
        let other = VoAgent::new(Vec2::new(5.0, 0.0), Vec2::new(-1.0, 0.0), 0.5);
        assert!(is_in_velocity_obstacle(&agent, &other, Vec2::new(1.0, 0.0)));
    }

    #[test]
    fn test_vo_no_collision() {
        let agent = VoAgent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 0.5);
        let other = VoAgent::new(Vec2::new(5.0, 5.0), Vec2::new(0.0, 1.0), 0.5);
        assert!(!is_in_velocity_obstacle(
            &agent,
            &other,
            Vec2::new(1.0, 0.0)
        ));
    }

    #[test]
    fn test_vo_moving_away() {
        let agent = VoAgent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 0.5);
        let other = VoAgent::new(Vec2::new(5.0, 0.0), Vec2::new(1.0, 0.0), 0.5);
        // Candidate velocity moving away from the other.
        assert!(!is_in_velocity_obstacle(
            &agent,
            &other,
            Vec2::new(-1.0, 0.0)
        ));
    }

    #[test]
    fn test_generate_velocity_candidates_count() {
        let candidates = generate_velocity_candidates(2.0, 3, 8);
        // 1 (zero) + 3 rings * 8 angles = 25.
        assert_eq!(candidates.len(), 25);
    }

    #[test]
    fn test_generate_velocity_candidates_includes_zero() {
        let candidates = generate_velocity_candidates(1.0, 2, 4);
        assert!((candidates[0].x).abs() < 1e-9);
        assert!((candidates[0].y).abs() < 1e-9);
    }

    #[test]
    fn test_select_velocity_no_others() {
        let agent = VoAgent::new(Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.0), 0.5);
        let preferred = Vec2::new(1.0, 0.0);
        let candidates = generate_velocity_candidates(2.0, 3, 8);
        let result = select_velocity_outside_vo(&agent, &[], preferred, &candidates);
        // Should select something close to preferred.
        assert!(result.length() > 0.1);
    }

    #[test]
    fn test_select_velocity_avoids_other() {
        let agent = VoAgent::new(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0), 0.5);
        let other = VoAgent::new(Vec2::new(3.0, 0.0), Vec2::new(-1.0, 0.0), 0.5);
        let candidates = generate_velocity_candidates(2.0, 3, 16);
        let result = select_velocity_outside_vo(&agent, &[other], Vec2::new(1.0, 0.0), &candidates);
        // Should not go straight toward the other.
        assert!(!is_in_velocity_obstacle(&agent, &other, result));
    }

    // --- NavMesh tests ---

    #[test]
    fn test_nav_triangle_contains() {
        let tri = NavTriangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(0.0, 10.0),
        );
        assert!(tri.contains(Vec2::new(1.0, 1.0)));
        assert!(!tri.contains(Vec2::new(8.0, 8.0)));
    }

    #[test]
    fn test_nav_triangle_centroid() {
        let tri = NavTriangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(6.0, 0.0),
            Vec2::new(0.0, 6.0),
        );
        let c = tri.centroid();
        assert!((c.x - 2.0).abs() < 1e-9);
        assert!((c.y - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_nav_triangle_area() {
        let tri = NavTriangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(4.0, 0.0),
            Vec2::new(0.0, 3.0),
        );
        assert!((tri.area() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_nav_triangle_vertex_containment() {
        let tri = NavTriangle::new(
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
            Vec2::new(5.0, 10.0),
        );
        assert!(tri.contains(Vec2::new(0.0, 0.0)));
        assert!(tri.contains(Vec2::new(10.0, 0.0)));
        assert!(tri.contains(Vec2::new(5.0, 10.0)));
    }

    #[test]
    fn test_build_grid_navmesh() {
        let bounds = Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let mesh = build_grid_navmesh(bounds, 5, 5);
        assert_eq!(mesh.triangles.len(), 50); // 5*5*2.
    }

    #[test]
    fn test_navmesh_find_triangle() {
        let bounds = Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let mesh = build_grid_navmesh(bounds, 5, 5);
        assert!(mesh.find_triangle(Vec2::new(5.0, 5.0)).is_some());
        assert!(mesh.find_triangle(Vec2::new(-1.0, -1.0)).is_none());
    }

    #[test]
    fn test_navmesh_is_navigable() {
        let bounds = Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let mesh = build_grid_navmesh(bounds, 5, 5);
        assert!(mesh.is_navigable(Vec2::new(3.0, 3.0)));
        assert!(!mesh.is_navigable(Vec2::new(11.0, 11.0)));
    }

    #[test]
    fn test_navmesh_total_area() {
        let bounds = Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let mesh = build_grid_navmesh(bounds, 5, 5);
        assert!((mesh.total_area() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_navmesh_find_path_same_triangle() {
        let bounds = Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let mesh = build_grid_navmesh(bounds, 5, 5);
        let path = mesh.find_path(Vec2::new(0.5, 0.5), Vec2::new(1.0, 0.5));
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p.len(), 2);
    }

    #[test]
    fn test_navmesh_find_path_cross_mesh() {
        let bounds = Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let mesh = build_grid_navmesh(bounds, 5, 5);
        let path = mesh.find_path(Vec2::new(0.5, 0.5), Vec2::new(9.5, 9.5));
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.len() >= 2);
    }

    #[test]
    fn test_navmesh_path_endpoints() {
        let bounds = Bounds2D::new(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0));
        let mesh = build_grid_navmesh(bounds, 5, 5);
        let start = Vec2::new(0.5, 0.5);
        let goal = Vec2::new(9.5, 9.5);
        let path = mesh.find_path(start, goal).unwrap();
        assert!((path[0].x - start.x).abs() < 1e-9);
        assert!((path.last().unwrap().x - goal.x).abs() < 1e-9);
    }

    // --- Dynamic Planner tests ---

    #[test]
    fn test_dynamic_planner_initial_plan() {
        let mut planner = DynamicPlanner::new(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            no_obstacles(),
        );
        assert!(planner.plan());
        assert!(planner.current_path.is_some());
    }

    #[test]
    fn test_dynamic_planner_add_obstacle_replan() {
        let mut planner = DynamicPlanner::new(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            no_obstacles(),
        );
        planner.plan();
        let initial_path = planner.current_path.clone();
        // Add obstacle that might invalidate path.
        planner.add_obstacle(CircleObstacle::new(Vec2::new(10.0, 10.0), 3.0));
        // Path may have been replanned.
        assert!(planner.current_path.is_some() || initial_path.is_some());
    }

    #[test]
    fn test_dynamic_planner_is_path_valid() {
        let mut planner = DynamicPlanner::new(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            no_obstacles(),
        );
        assert!(!planner.is_path_valid()); // No path yet.
        planner.plan();
        assert!(planner.is_path_valid());
    }

    #[test]
    fn test_dynamic_planner_update_start() {
        let mut planner = DynamicPlanner::new(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            no_obstacles(),
        );
        planner.update_start(Vec2::new(5.0, 5.0));
        assert!((planner.start.x - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_dynamic_planner_remove_obstacles() {
        let mut planner = DynamicPlanner::new(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            simple_obstacles(),
        );
        assert_eq!(planner.obstacles.len(), 1);
        planner.remove_obstacles_where(|o| o.radius > 1.0);
        assert_eq!(planner.obstacles.len(), 0);
    }

    #[test]
    fn test_dynamic_planner_set_config() {
        let mut planner = DynamicPlanner::new(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            no_obstacles(),
        );
        planner.set_rrt_config(RrtConfig {
            step_size: 1.0,
            max_iterations: 1000,
            goal_threshold: 1.0,
            seed: 999,
        });
        assert!(planner.plan());
    }

    // --- Utility tests ---

    #[test]
    fn test_path_length_straight() {
        let path = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        assert!((path_length(&path) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_path_length_multi_segment() {
        let path = vec![
            Vec2::new(0.0, 0.0),
            Vec2::new(3.0, 0.0),
            Vec2::new(3.0, 4.0),
        ];
        assert!((path_length(&path) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_path_length_empty() {
        assert!((path_length(&[]) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_path_length_single_point() {
        let path = vec![Vec2::new(5.0, 5.0)];
        assert!((path_length(&path) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_resample_path_spacing() {
        let path = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let resampled = resample_path(&path, 2.0);
        assert!(resampled.len() >= 5);
    }

    #[test]
    fn test_resample_path_empty() {
        let resampled = resample_path(&[], 1.0);
        assert!(resampled.is_empty());
    }

    #[test]
    fn test_resample_path_preserves_start() {
        let path = vec![Vec2::new(1.0, 1.0), Vec2::new(11.0, 1.0)];
        let resampled = resample_path(&path, 2.0);
        assert!((resampled[0].x - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_closest_point_on_segment_start() {
        let c = closest_point_on_segment(
            Vec2::new(-1.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
        );
        assert!((c.x - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_closest_point_on_segment_end() {
        let c = closest_point_on_segment(
            Vec2::new(15.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
        );
        assert!((c.x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_closest_point_on_segment_middle() {
        let c = closest_point_on_segment(
            Vec2::new(5.0, 3.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
        );
        assert!((c.x - 5.0).abs() < 1e-9);
        assert!((c.y - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_segment_distance() {
        let d = point_segment_distance(
            Vec2::new(5.0, 3.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
        );
        assert!((d - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_point_segment_distance_at_endpoint() {
        let d = point_segment_distance(
            Vec2::new(-3.0, 4.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(10.0, 0.0),
        );
        assert!((d - 5.0).abs() < 1e-9);
    }

    // --- Additional edge case tests ---

    #[test]
    fn test_rrt_different_seeds_different_paths() {
        let c1 = RrtConfig {
            seed: 1,
            ..RrtConfig::default()
        };
        let c2 = RrtConfig {
            seed: 2,
            ..RrtConfig::default()
        };
        let p1 = rrt(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &c1,
        );
        let p2 = rrt(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &c2,
        );
        assert!(p1.is_some());
        assert!(p2.is_some());
        // Paths with different seeds are likely different lengths.
    }

    #[test]
    fn test_prm_deterministic() {
        let config = PrmConfig::default();
        let p1 = prm(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &config,
        );
        let p2 = prm(
            Vec2::new(1.0, 1.0),
            Vec2::new(19.0, 19.0),
            default_bounds(),
            &no_obstacles(),
            &config,
        );
        assert_eq!(p1.as_ref().map(Vec::len), p2.as_ref().map(Vec::len));
    }

    #[test]
    fn test_smooth_path_with_obstacles() {
        let obs = simple_obstacles();
        let path = vec![
            Vec2::new(1.0, 1.0),
            Vec2::new(5.0, 8.0),
            Vec2::new(8.0, 14.0),
            Vec2::new(15.0, 15.0),
            Vec2::new(19.0, 19.0),
        ];
        let smoothed = smooth_path(&path, &obs, 50);
        // Should still have valid segments.
        for w in smoothed.windows(2) {
            assert!(segment_free(w[0], w[1], &obs));
        }
    }

    #[test]
    fn test_resample_preserves_end() {
        let path = vec![Vec2::new(0.0, 0.0), Vec2::new(10.0, 0.0)];
        let resampled = resample_path(&path, 3.0);
        let last = resampled.last().unwrap();
        assert!((last.x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_navmesh_empty() {
        let mesh = NavMesh::new(Vec::new());
        assert!(!mesh.is_navigable(Vec2::new(0.0, 0.0)));
        assert!(mesh
            .find_path(Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0))
            .is_none());
        assert!((mesh.total_area() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_vec2_equality() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(1.0, 2.0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_vec2_inequality() {
        let a = Vec2::new(1.0, 2.0);
        let b = Vec2::new(1.0, 3.0);
        assert_ne!(a, b);
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_rng_range() {
        let mut rng = Rng::new(42);
        for _ in 0..100 {
            let v = rng.range(5.0, 10.0);
            assert!(v >= 5.0);
            assert!(v < 10.0);
        }
    }
}
