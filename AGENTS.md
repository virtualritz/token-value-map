# Repository Guidelines

## Shared Blueprints

This repository includes `virtualritz/blueprints` as `.blueprints`.

- [Agent Behavior Rules](.blueprints/base/AGENTS.md)
- [Git Safety](.blueprints/base/git-safety.md)
- [Writing Style](.blueprints/base/writing-style.md)
- [Rust Agent Rules](.blueprints/lang/rust/AGENTS.md)
- [Rust Testing](.blueprints/lang/rust/testing.md)
- [Rust Numeric Performance](.blueprints/lang/rust/numeric-performance.md) --
  **mandatory** for any change whose purpose is speed, or any refactor
  touching a hot numeric path: consider occupancy, cutting the work, float
  reassociation and portable SIMD, and record the decision, rejections
  included.

Project-specific instructions in this file override shared blueprint rules.
