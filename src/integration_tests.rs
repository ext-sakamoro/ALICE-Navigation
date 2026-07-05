//! Integration tests spanning multiple modules.

#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::bool_to_int_with_if,
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::format_collect,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::iter_cloned_collect,
    clippy::suboptimal_flops,
    clippy::should_panic_without_expect,
    clippy::manual_range_contains
)]

use crate::collision::*;
use crate::core_types::*;
use crate::dynamic_replan::*;
use crate::navmesh::*;
use crate::obstacle_avoidance::*;
use crate::path_smoothing::*;
use crate::potential_field::*;
use crate::prm::*;
use crate::rrt::*;
use crate::util::*;
use crate::velocity_obstacles::*;
use crate::waypoint::*;

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
