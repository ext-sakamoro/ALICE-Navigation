//! ALICE-Navigation: Autonomous navigation library.
//!
//! Provides RRT, PRM, potential field, obstacle avoidance, path smoothing,
//! waypoint following, velocity obstacles, navigation mesh, and dynamic replanning.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::suboptimal_flops,
    clippy::should_implement_trait,
    clippy::suspicious_operation_groupings,
    clippy::while_float,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::wildcard_imports,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_lossless,
    clippy::float_cmp
)]

pub mod collision;
pub mod core_types;
pub mod dynamic_replan;
pub mod navmesh;
pub mod obstacle_avoidance;
pub mod path_smoothing;
pub mod potential_field;
pub mod prelude;
pub mod prm;
pub mod rrt;
pub mod util;
pub mod velocity_obstacles;
pub mod waypoint;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::collision::*;
pub use crate::core_types::*;
pub use crate::dynamic_replan::*;
pub use crate::navmesh::*;
pub use crate::obstacle_avoidance::*;
pub use crate::path_smoothing::*;
pub use crate::potential_field::*;
pub use crate::prm::*;
pub use crate::rrt::*;
pub use crate::util::*;
pub use crate::velocity_obstacles::*;
pub use crate::waypoint::*;
