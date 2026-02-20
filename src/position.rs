/// Normalized position in [0, 1] for curve domains.
///
/// Used as the key type for curve data maps (e.g., `KeyDataMap<Position, Real>`
/// for real-valued curves).

#[cfg(feature = "rkyv")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A normalized position in [0, 1] for curve stop domains.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
#[cfg_attr(
    feature = "rkyv",
    rkyv(attr(derive(Debug, Clone, Copy, PartialEq, PartialOrd)))
)]
pub struct Position(pub f32);

impl Eq for Position {}

impl Ord for Position {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl Hash for Position {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<Position> for f32 {
    fn from(p: Position) -> f32 {
        p.0
    }
}

impl From<f32> for Position {
    fn from(v: f32) -> Position {
        Position(v)
    }
}

// Manual trait impls for ArchivedPosition (f32 doesn't impl Ord/Eq/Hash).
#[cfg(feature = "rkyv")]
impl Eq for ArchivedPosition {}

#[cfg(feature = "rkyv")]
impl Ord for ArchivedPosition {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = f32::from(self.0);
        let b = f32::from(other.0);
        a.total_cmp(&b)
    }
}

#[cfg(feature = "rkyv")]
impl Hash for ArchivedPosition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        f32::from(self.0).to_bits().hash(state);
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
