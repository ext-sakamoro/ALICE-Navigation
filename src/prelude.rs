//! Convenience re-export (= `use alice_navigation::prelude::*;`).

pub use crate::collision::{point_free, segment_free};
pub use crate::core_types::{Bounds2D, CircleObstacle, Vec2};
pub use crate::dynamic_replan::DynamicPlanner;
pub use crate::navmesh::{build_grid_navmesh, NavMesh, NavTriangle};
pub use crate::obstacle_avoidance::{avoid_obstacles, is_near_obstacle};
pub use crate::path_smoothing::{chaikin_smooth, smooth_path};
pub use crate::potential_field::{
    attractive_force, potential_field, repulsive_force, PotentialFieldConfig,
};
pub use crate::prm::{prm, PrmConfig};
pub use crate::rrt::{rrt, RrtConfig};
pub use crate::util::{
    closest_point_on_segment, path_length, point_segment_distance, resample_path,
};
pub use crate::velocity_obstacles::{
    generate_velocity_candidates, is_in_velocity_obstacle, select_velocity_outside_vo, VoAgent,
};
pub use crate::waypoint::WaypointFollower;
