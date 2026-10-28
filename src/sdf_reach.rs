//! Conservative reachability over a signed distance field.
//!
//! The field is any [`SdfField`] (alice-physics' trait; alice-sdf implements it for its compiled
//! and live shapes): positive distance in free space, negative inside obstacles. An agent is a
//! vertical capsule given by its radius and the heights of its axis samples.
//!
//! # Premise
//!
//! * The capsule axis spans `h_min..=h_max` (the first and last height), and consecutive heights
//!   are at most `s` apart, both endpoints included.
//! * The field is `L`-Lipschitz (`|d(p) - d(q)| <= L |p - q|`); `L = 1` for exact SDFs and their
//!   CSG. A field that underestimates distance (a union of void primitives, for example) keeps
//!   the guarantee below but makes the completeness bound hold only against its own values.
//!
//! # Guarantee
//!
//! The floor plane is a grid of `c × c` cells. A cell is *free* when at every height `h_i`
//! `d(centre + h_i) / L > r + sqrt(c²/2 + s²/4)`: every point of the capsule axis over the cell
//! is within `c/√2` horizontally and `s/2` vertically of a sample, so it is more than `r` from
//! every obstacle. Reachability is 4-neighbour flood fill over free cells.
//!
//! * `reachable == true` (sound): along the witness (cell centres, each 4-adjacent to the next)
//!   the whole capsule clears every obstacle.
//! * `reachable == false` is **not** a proof of impossibility (not complete). A passage of clear
//!   width `w` is always found when `w >= 2r + 2·sqrt(c²/2 + s²/4) + c` (axis-aligned) or
//!   `w >= 2r + 2·sqrt(c²/2 + s²/4) + √2·c` (any direction).
//!
//! # Determinism
//!
//! Integer grid, cell centres `origin + (i + 0.5)·c` in f32, a fixed neighbour order and the
//! first-discovered parent: given bit-identical field values, the result and the witness are
//! bit-identical on every platform.
//!
//! The grid is a floor plane (`nx × nz`) today; [`ReachGrid`] and [`Clearance`] are built through
//! constructors so a volume grid (flying agents) can be added without breaking callers.

use alice_physics::SdfField;

/// The largest grid accepted (`nx · nz`, 4096 × 4096): the grid is evaluated in full, so a
/// larger area is split into tiles.
pub const MAX_CELLS: usize = 1 << 24;

/// The grid the agent moves on.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ReachGrid {
    origin: [f32; 3],
    cell: f32,
    nx: usize,
    nz: usize,
}

impl ReachGrid {
    /// A floor plane at height `origin[1]`, `nx × nz` cells of size `cell` starting at
    /// `(origin[0], origin[2])`.
    pub const fn floor(origin: [f32; 3], cell: f32, nx: usize, nz: usize) -> Self {
        Self {
            origin,
            cell,
            nx,
            nz,
        }
    }
    pub const fn cell(&self) -> f32 {
        self.cell
    }
    pub const fn cells(&self) -> usize {
        self.nx * self.nz
    }
}

/// The agent: a vertical capsule.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Clearance {
    radius: f32,
    heights: Vec<f32>,
    lipschitz: f32,
}

impl Clearance {
    /// A capsule of `radius` whose axis is sampled at `heights` above the floor (ascending; the
    /// first and last are the ends of the axis).
    pub fn capsule(radius: f32, heights: &[f32]) -> Self {
        Self {
            radius,
            heights: heights.to_vec(),
            lipschitz: 1.0,
        }
    }
    /// The field's Lipschitz bound (default 1).
    #[must_use]
    pub const fn with_lipschitz(mut self, lipschitz: f32) -> Self {
        self.lipschitz = lipschitz;
        self
    }
    /// The distance a free cell's samples must exceed: `L · (r + sqrt(c²/2 + s²/4))`, `s` the
    /// largest gap between consecutive heights (0 for one height).
    pub fn margin(&self, cell: f32) -> f32 {
        let s = self
            .heights
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(0.0f32, f32::max);
        self.lipschitz * (self.radius + (cell * cell * 0.5 + s * s * 0.25).sqrt())
    }

    fn is_valid(&self) -> bool {
        self.radius.is_finite()
            && self.radius > 0.0
            && self.lipschitz.is_finite()
            && self.lipschitz > 0.0
            && !self.heights.is_empty()
            && self.heights.iter().all(|h| h.is_finite())
            && self.heights.windows(2).all(|w| w[1] > w[0])
    }
}

impl ReachGrid {
    fn is_valid(&self) -> bool {
        self.cell.is_finite()
            && self.cell > 0.0
            && self.origin.iter().all(|v| v.is_finite())
            && self.nx > 0
            && self.nz > 0
            && self.nx.checked_mul(self.nz).is_some_and(|n| n <= MAX_CELLS)
    }

    /// The cell holding `p` (floor-plane coordinates), or `None` outside the grid.
    fn cell_of(&self, p: [f32; 3]) -> Option<usize> {
        let fx = ((p[0] - self.origin[0]) / self.cell).floor();
        let fz = ((p[2] - self.origin[2]) / self.cell).floor();
        if !(fx >= 0.0 && fz >= 0.0 && fx < self.nx as f32 && fz < self.nz as f32) {
            return None;
        }
        Some(fz as usize * self.nx + fx as usize)
    }

    fn centre(&self, i: usize) -> [f32; 3] {
        [
            self.origin[0] + ((i % self.nx) as f32 + 0.5) * self.cell,
            self.origin[1],
            self.origin[2] + ((i / self.nx) as f32 + 0.5) * self.cell,
        ]
    }
}

/// `parent` of a cell not reached.
const UNSET: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ReachError {
    /// The grid has no cell or more than [`MAX_CELLS`], or its origin or cell size is not finite,
    /// or the cell size is not positive.
    InvalidGrid,
    /// The radius is not finite and positive, the heights are empty, not finite or not
    /// ascending, or the Lipschitz bound is not finite and positive.
    InvalidClearance,
    /// The start point lies outside the grid.
    StartOutOfGrid,
    /// The start cell is not free.
    StartNotFree,
    /// The field returned a non-finite distance (NaN never means "blocked").
    NonFiniteDistance { at: [f32; 3] },
}

/// Everything reachable from one start.
#[derive(Clone, Debug)]
pub struct ReachMap {
    grid: ReachGrid,
    parent: Vec<u32>,
    free_cells: usize,
}

impl ReachMap {
    pub fn reached(&self, p: [f32; 3]) -> bool {
        self.grid
            .cell_of(p)
            .is_some_and(|i| self.parent[i] != UNSET)
    }
    /// Cell centres (at floor height) from the start to `p`, or `None` when `p` is not reached.
    pub fn witness(&self, p: [f32; 3]) -> Option<Vec<[f32; 3]>> {
        let mut i = self.grid.cell_of(p)?;
        if self.parent[i] == UNSET {
            return None;
        }
        let mut path = vec![self.grid.centre(i)];
        while self.parent[i] as usize != i {
            i = self.parent[i] as usize;
            path.push(self.grid.centre(i));
        }
        path.reverse();
        Some(path)
    }
    pub const fn free_cells(&self) -> usize {
        self.free_cells
    }
    pub const fn checked_cells(&self) -> usize {
        self.grid.cells()
    }
}

/// Flood fill of the free cells from `from`.
pub fn reach_map<F: SdfField + ?Sized>(
    field: &F,
    grid: &ReachGrid,
    from: [f32; 3],
    agent: &Clearance,
) -> Result<ReachMap, ReachError> {
    if !grid.is_valid() {
        return Err(ReachError::InvalidGrid);
    }
    if !agent.is_valid() {
        return Err(ReachError::InvalidClearance);
    }
    let margin = agent.margin(grid.cell);
    // every cell is evaluated (not only the reached ones) so a non-finite value anywhere is an
    // error, whatever the start
    let n = grid.cells();
    let mut free = vec![false; n];
    let mut free_cells = 0;
    for (i, f) in free.iter_mut().enumerate() {
        let c = grid.centre(i);
        let mut ok = true;
        for h in &agent.heights {
            let at = [c[0], c[1] + h, c[2]];
            let d = field.distance(at[0], at[1], at[2]);
            if !d.is_finite() {
                return Err(ReachError::NonFiniteDistance { at });
            }
            ok &= d > margin;
        }
        *f = ok;
        free_cells += usize::from(ok);
    }
    let start = grid.cell_of(from).ok_or(ReachError::StartOutOfGrid)?;
    if !free[start] {
        return Err(ReachError::StartNotFree);
    }
    let mut parent = vec![UNSET; n];
    parent[start] = start as u32;
    let mut queue = std::collections::VecDeque::from([start]);
    let nx = grid.nx;
    while let Some(i) = queue.pop_front() {
        let (ix, iz) = (i % nx, i / nx);
        // fixed order (-x, +x, -z, +z): the witness is the same on every platform
        let next = [
            (ix > 0).then(|| i - 1),
            (ix + 1 < nx).then(|| i + 1),
            (iz > 0).then(|| i - nx),
            (iz + 1 < grid.nz).then(|| i + nx),
        ];
        for j in next.into_iter().flatten() {
            if free[j] && parent[j] == UNSET {
                parent[j] = i as u32;
                queue.extend([j]);
            }
        }
    }
    Ok(ReachMap {
        grid: *grid,
        parent,
        free_cells,
    })
}

/// Is `to` reachable from `from`, with the witness when it is.
pub fn reach<F: SdfField + ?Sized>(
    field: &F,
    grid: &ReachGrid,
    from: [f32; 3],
    to: [f32; 3],
    agent: &Clearance,
) -> Result<Option<Vec<[f32; 3]>>, ReachError> {
    Ok(reach_map(field, grid, from, agent)?.witness(to))
}
