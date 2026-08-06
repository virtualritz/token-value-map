//! Normalized position in [0, 1] for curve domains.
//!
//! Used as the key type for curve data maps (e.g., `KeyDataMap<Position, Real>`
//! for real-valued curves).

#[cfg(feature = "rkyv")]
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A normalized position in [0, 1] for curve stop domains.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "facet", derive(facet::Facet))]
#[cfg_attr(feature = "facet", facet(transparent))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "rkyv", derive(Archive, RkyvSerialize, RkyvDeserialize))]
#[cfg_attr(feature = "rkyv", rkyv(attr(derive(Debug, Clone, Copy))))]
pub struct Position(pub f32);

// AIDEV-NOTE: `PartialEq`/`PartialOrd` are implemented manually rather than
// derived so all four comparison traits agree on the *total* float order used
// by [`Ord`]. Deriving them would use IEEE semantics, which disagree with
// `total_cmp` on `NaN` and on `+0.0` vs `-0.0` -- and since `Position` is a
// `BTreeMap` key that inconsistency corrupts the map's ordering invariant.
// `Hash` uses `to_bits`, which partitions values identically to `total_cmp`.
impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Position {}

impl PartialOrd for Position {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

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
// AIDEV-NOTE: as for [`Position`], all four comparison traits are implemented
// by hand so they agree on the total float order -- see the note above.
#[cfg(feature = "rkyv")]
impl PartialEq for ArchivedPosition {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

#[cfg(feature = "rkyv")]
impl Eq for ArchivedPosition {}

#[cfg(feature = "rkyv")]
impl PartialOrd for ArchivedPosition {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

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
