# Session: Naming Consistency + rkyv Fix (v0.2.3)

## Completed

- Renamed `closest_sample` → `sample_closest_at` on `KeyDataMap` (+ deprecated alias).
- Renamed `add_sample` → `add_at` on `Value` and `GenericValue` (+ deprecated aliases).
- Updated 6 internal call sites in `animated_data.rs` from `closest_sample` → `sample_closest_at`.
- Added `#![cfg_attr(feature = "rkyv", feature(trivial_bounds))]` to `lib.rs` to fix rkyv compilation with `BTreeMap1`.
- Bumped version to `0.2.3`.
- All feature combos compile (default, rkyv, serde, facet).
- All tests pass.

## Left to Do

- Commit, tag `v0.2.3`, push, publish.
- `cargo yank --version 0.2.2`.
