# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The Golden Rule

When unsure about implementation details, ALWAYS ask the developer.

## Project Context

This is a Rust crate for time-based data mapping with animation and interpolation support. The crate provides types for storing and manipulating data that changes over time, with automatic interpolation between keyframes.

## Critical Architecture Decisions

### Core Data Structure

- **Flat value storage**: Values stored as either uniform (constant) or animated (time-varying).
- **Generic data types**: Support for scalar types (Boolean, Integer, Real, String), vector types (Vector2, Vector3, Color, Matrix3, Matrix4, Normal3, Point3), and collections of these types.
- **Time representation**: Uses fixed-point `Tick` from `frame-tick` crate for precise timing.
- **Token-based mapping**: String tokens (using `ustr`) map to values in `TokenValueMap`.

### Important Types

- **[`Value`]** -- A value that can be either uniform or animated over time.
- **[`Data`]** -- Variant enum containing all supported data types.
- **[`AnimatedData`]** -- Time-indexed data with interpolation support.
- **[`TimeDataMap`]** -- A mapping from time to data values.
- **[`TokenValueMap`]** -- A collection of named values indexed by tokens.
- **[`Shutter`]** -- Configuration for motion blur sampling.

### Key Features

- Arbitrary vector lengths with automatic padding.
- Multiple interpolation methods (linear, quadratic, hermite).
- Motion blur sampling support.
- Optional Lua bindings.
- Feature-gated 2D/3D types.

### Performance Requirements

- Use `rayon` for parallelism in bulk operations.
- Prefer `SmallVec` for small collections to avoid heap allocations.
- Use `enum_dispatch` for performance-critical trait dispatch.

## Build and Development Commands

```bash
# Build the project
cargo build

# Build with all features
cargo build --all-features

# Build with specific features
cargo build --features "2d,3d,lua,serde"

# Run tests
cargo test

# Run a specific test
cargo test test_name

# Build with optimizations
cargo build --release

# Format code
cargo fmt

# Run clippy linter and fix issues
cargo clippy --fix --allow-dirty

# Check code without building
cargo check

# Check with specific features
cargo check --no-default-features --features "2d,3d,vec_variants"
```

## Feature Flags

- `default`: Includes `2d` and `vec_variants`.
- `lua`: Enable Lua bindings via `mlua`.
- `serde`: Enable serialization support.
- `2d`: Enable 2D types (`vector2`, `matrix3`).
- `3d`: Enable 3D types (`vector3`, `matrix4`, `normal3`, `point3`).
- `vec_variants`: Enable vector collection types.

## External Crate Integration

- Time handling: `frame-tick` crate for fixed-point time representation.
- Linear algebra: `nalgebra` for vector and matrix types.
- String interning: `ustr` for efficient token storage.
- Parallelism: `rayon` for parallel processing.
- Lua support: `mlua` (optional).

## Code Style and Patterns

### Anchor comments

Add specially formatted comments throughout the codebase, where appropriate, for yourself as inline knowledge that can be easily `rg`ped (grepped) for.

### Guidelines:

- **CRITICAL: ALWAYS run `cargo check` and ensure the code compiles BEFORE committing!** Never commit code that doesn't build. If there are compilation errors, fix them first. This is non-negotiable.

- ALWAYS run `cargo clippy --fix` before committing. If clippy brings up any issues, fix them, then repeat until there are no more issues brought up by clippy. Finally run `cargo fmt`, then commit.

- Use `AIDEV-NOTE:`, `AIDEV-TODO:`, or `AIDEV-QUESTION:` (all-caps prefix) for comments aimed at AI and developers.
- **Important:** Before scanning files, always first try to **grep for existing anchors** `AIDEV-*` in relevant subdirectories.
- **Update relevant anchors** when modifying associated code.
- **Do not remove `AIDEV-NOTE`s** without explicit human instruction.
- Make sure to add relevant anchor comments, whenever a file or piece of code is:
  - too complex, or
  - very important, or
  - confusing, or
  - could have a bug

- DO NOT change any public-facing API without presenting a change proposal to the user first, including a rationale and getting permission to do so after.

- Write idiomatic and canonical Rust code. I.e. avoid patterns common in imperative languages like C/C++/JS/TS that can be expressed more elegantly, concise and with more leeway for the compiler to optimize, in Rust.

- PREFER functional style over imperative style. I.e. use for_each or map instead of for loops, use collect instead of pre-allocating a Vec and using push.

- USE rayon to parallelize whenever larger amounts of data are being processed.

- AVOID unnecessary allocations, conversions, copies.

- AVOID using `unsafe` code unless absolutely necessary.

- AVOID return statements; structure functions with if ... if else ... else blocks instead.

- Prefer using the stack, use SmallVec whenever it makes sense.

- NAMING follows the rules laid out in this document: https://raw.githubusercontent.com/rust-lang/api-guidelines/refs/heads/master/src/naming.md

## What AI Must NEVER Do

1. **Never modify test files** -- Tests encode human intent.
2. **Never change API contracts** -- Breaks real applications.
3. **Never commit secrets** -- Use environment variables.
4. **Never assume business logic** -- Always ask.
5. **Never remove AIDEV- comments** -- They're there for a reason.

## Writing Instructions For User Interaction And Documentation

These instructions apply to any communication (e.g. feedback you print to the user) as well as any documentation you write.

- Be concise.

- AVOID weasel words.

- Use simple sentences. But feel free to use technical jargon.

- Do NOT overexplain basic concepts. Assume the user is technically proficient.

- AVOID flattering, corporate-ish or marketing language. Maintain a neutral viewpoint.

- AVOID vague and/or generic claims which may seem correct but are not substantiated by the context.

## Documentation

- All code comments MUST end with a period.

- All doc comments should also end with a period unless they're headlines. This includes list items.

- ENSURE an en-dash is expressed as two dashes like so: --. En-dashes are not used for connecting words, e.g. "compile-time".

- All references to types, keywords, symbols etc. MUST be enclosed in backticks: `struct` `Foo`.

- For each part of the docs, every first reference to a type, keyword, symbol etc. that is NOT the item itself that is being described MUST be linked to the relevant section in the docs like so: [`Foo`].

- NEVER use fully qualified paths in doc links. Use [`Foo`](foo::bar::Foo) instead of [`foo::bar::Foo`].

## Traits

- All public-facing types must implement `Debug`, `Clone`, `Hash`, `PartialEq`, and `Eq`. Also `Copy`, if it can be trivially derived.

## Save Points

- Regularly update what you did and what is left to do for the current session in CLAUDE-CONTINUE.md.

- Content not relevant to the current session found in this file should be removed. Make especially sure to update the file in time before you run out of context and have to compress.

- Make sure to commit the file to the current branch before you run out of context and have to compress.