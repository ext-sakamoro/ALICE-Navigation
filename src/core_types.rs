//! Core types (`Vec2` / `Bounds2D` / `CircleObstacle`).

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
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) const fn new(seed: u64) -> Self {
        let state = if seed == 0 { 1 } else { seed };
        Self { state }
    }

    pub(crate) const fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a value in `[0, 1)`.
    pub(crate) fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0x000F_FFFF_FFFF_FFFF) as f64 / (1u64 << 52) as f64
    }

    /// Returns a value in `[lo, hi)`.
    pub(crate) fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}
