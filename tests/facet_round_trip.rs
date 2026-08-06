//! Round trips of every persisted shape through `facet`.
//!
//! Run with the consumer's feature selection:
//!
//! ```text
//! cargo test --features glam,3d,facet,serde,curves,interpolation --test facet_round_trip
//! ```
//!
//! Equality is bit-exact throughout. `Value` is `Eq`, so "the value that went
//! in comes back" is a machine-checkable claim rather than a tolerance.
//!
//! The format here is JSON rather than TOML on purpose. What this crate owns is
//! the facet *representation*; which text format a consumer writes is the
//! consumer's business, and `facet-toml` 0.46.1 has an unrelated serializer bug
//! -- it omits the separator between array elements that are themselves arrays
//! or inline tables -- that akatela carries a vendored fix for. Proving the
//! representation against a format without that bug keeps the two failures
//! apart. akatela's own suite proves the TOML end of it.

#![cfg(all(feature = "facet", feature = "builtin-types"))]

use facet::Facet;
use frame_tick::Tick;
use std::collections::BTreeMap;
use strum::IntoEnumIterator;
use token_value_map::*;

/// A document shaped like a consumer's: values reached through a map inside a
/// struct, which is where a naive representation stops working.
#[derive(Facet, Debug, Clone, PartialEq)]
struct Document {
    parameters: BTreeMap<std::string::String, Value>,
}

fn round_trip(value: Value) -> anyhow::Result<Value> {
    let document = Document {
        parameters: BTreeMap::from([("p".to_string(), value)]),
    };
    let text = facet_json::to_string(&document)?;
    let back: Document = facet_json::from_str(&text)
        .map_err(|error| anyhow::anyhow!("{error}\n--- document ---\n{text}"))?;
    Ok(back.parameters["p"].clone())
}

#[cfg(all(feature = "interpolation", feature = "curves"))]
fn assert_round_trips(value: Value) -> anyhow::Result<()> {
    let back = round_trip(value.clone())?;
    assert_eq!(value, back);
    Ok(())
}

/// One representative value per [`DataType`].
///
/// The `match` is exhaustive with no catch-all, so a variant added to `Data`
/// breaks this build rather than silently joining a blind spot.
fn representative(data_type: DataType) -> Data {
    match data_type {
        DataType::Boolean => Data::Boolean(Boolean(true)),
        DataType::Integer => Data::Integer(Integer(-9_007_199_254_740_993)),
        DataType::Real => Data::Real(Real(0.001)),
        DataType::String => Data::String(String("a\\nb, \"q\" [x] = # ".to_string())),
        DataType::Color => Data::Color(Color([0.25, -0.5, 1.0, 0.125])),
        #[cfg(feature = "vector2")]
        DataType::Vector2 => Data::Vector2(Vector2(math::vec2_from_array([1.5, -2.5]))),
        #[cfg(feature = "vector3")]
        DataType::Vector3 => Data::Vector3(Vector3(math::vec3_from_array([1.5, -2.5, 3.5]))),
        #[cfg(feature = "matrix3")]
        DataType::Matrix3 => Data::Matrix3(Matrix3(math::mat3_from_column_slice(&ASYMMETRIC_3X3))),
        #[cfg(feature = "normal3")]
        DataType::Normal3 => Data::Normal3(Normal3(math::vec3_from_array([0.0, 1.0, 0.0]))),
        #[cfg(feature = "point3")]
        DataType::Point3 => Data::Point3(Point3(math::point3_from_array([-1.0, 2.0, -3.0]))),
        #[cfg(feature = "matrix4")]
        DataType::Matrix4 => Data::Matrix4(Matrix4(math::mat4_from_column_slice(&ASYMMETRIC_4X4))),
        DataType::BooleanVec => Data::BooleanVec(BooleanVec(vec![true, false, true])),
        DataType::IntegerVec => Data::IntegerVec(IntegerVec(vec![1, -2, 3])),
        DataType::RealVec => Data::RealVec(RealVec(vec![
            -1.0,
            0.0,
            f64::from_bits(1),
            -1.234_567_890_123_456_7e-8,
        ])),
        DataType::ColorVec => Data::ColorVec(ColorVec(vec![[1.0, 0.0, 0.0, 1.0], [0.0; 4]])),
        DataType::StringVec => Data::StringVec(StringVec(vec![
            "with, a comma".to_string(),
            std::string::String::new(),
        ])),
        #[cfg(all(feature = "vector2", feature = "vec_variants"))]
        DataType::Vector2Vec => Data::Vector2Vec(Vector2Vec(vec![
            math::vec2_from_array([1.0, 2.0]),
            math::vec2_from_array([3.0, 4.0]),
        ])),
        #[cfg(all(feature = "vector3", feature = "vec_variants"))]
        DataType::Vector3Vec => {
            Data::Vector3Vec(Vector3Vec(vec![math::vec3_from_array([1.0, 2.0, 3.0])]))
        }
        #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
        DataType::Matrix3Vec => Data::Matrix3Vec(Matrix3Vec(vec![math::mat3_from_column_slice(
            &ASYMMETRIC_3X3,
        )])),
        #[cfg(all(feature = "normal3", feature = "vec_variants"))]
        DataType::Normal3Vec => {
            Data::Normal3Vec(Normal3Vec(vec![math::vec3_from_array([0.0, 0.0, 1.0])]))
        }
        #[cfg(all(feature = "point3", feature = "vec_variants"))]
        DataType::Point3Vec => {
            Data::Point3Vec(Point3Vec(vec![math::point3_from_array([9.0, 8.0, 7.0])]))
        }
        #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
        DataType::Matrix4Vec => Data::Matrix4Vec(Matrix4Vec(vec![math::mat4_from_column_slice(
            &ASYMMETRIC_4X4,
        )])),
        #[cfg(feature = "curves")]
        DataType::RealCurve => Data::RealCurve(RealCurve::linear()),
        #[cfg(feature = "curves")]
        DataType::ColorCurve => Data::ColorCurve(ColorCurve::black_to_white()),
    }
}

/// A matrix whose transpose differs from itself in every off-diagonal element.
#[cfg(feature = "matrix3")]
const ASYMMETRIC_3X3: [f32; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

/// The 4x4 equivalent.
#[cfg(feature = "matrix4")]
const ASYMMETRIC_4X4: [f64; 16] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
];

#[test]
fn every_data_variant_round_trips() -> anyhow::Result<()> {
    for data_type in DataType::iter() {
        let data = representative(data_type);
        assert_eq!(
            data.data_type(),
            data_type,
            "the representative for {data_type:?} is of the wrong type"
        );
        let back = round_trip(Value::Uniform(data.clone()))?;
        assert_eq!(Value::Uniform(data), back, "{data_type:?} did not survive");
    }
    Ok(())
}

#[test]
fn a_uniform_value_stays_uniform() -> anyhow::Result<()> {
    let back = round_trip(Value::Uniform(Data::Real(Real(1.0))))?;
    assert!(matches!(back, Value::Uniform(_)));
    Ok(())
}

#[test]
fn an_animated_value_keeps_every_keyframe() -> anyhow::Result<()> {
    let value = Value::animated(vec![
        (Tick::new(0), 1.0),
        (Tick::new(24), 2.5),
        (Tick::new(48), -0.75),
    ])?;
    let back = round_trip(value.clone())?;
    assert!(matches!(back, Value::Animated(_)));
    assert_eq!(back.sample_count(), 3);
    assert_eq!(value, back);
    Ok(())
}

/// The trap `FR-008a` refuses: three of the four [`BezierHandle`] variants carry
/// a full `T`, so a handle on an animated `Vector3` *is* a `Vector3`. A handle
/// modelled as a scalar pair fails this test by construction.
#[cfg(all(feature = "interpolation", feature = "vector3"))]
#[test]
fn a_bezier_handle_on_a_non_scalar_round_trips() -> anyhow::Result<()> {
    let mut value = Value::animated(vec![
        (
            Tick::new(0),
            Data::Vector3(Vector3(math::vec3_from_array([0.0, 0.0, 0.0]))),
        ),
        (
            Tick::new(24),
            Data::Vector3(Vector3(math::vec3_from_array([1.0, 2.0, 3.0]))),
        ),
    ])?;

    let Value::Animated(AnimatedData::Vector3(map)) = &mut value else {
        anyhow::bail!("expected an animated Vector3");
    };
    let handle = Key {
        interpolation_in: Interpolation::Bezier(BezierHandle::SlopePerSecond(Vector3(
            math::vec3_from_array([0.5, -0.25, 0.125]),
        ))),
        interpolation_out: Interpolation::Bezier(BezierHandle::Delta {
            time: Tick::new(3),
            value: Vector3(math::vec3_from_array([-1.0, 0.0, 1.0])),
        }),
    };
    // SAFETY: the map was built with a keyframe at tick 24 immediately above.
    let entry = unsafe { map.values.as_mut_btree_map() }
        .get_mut(&Tick::new(24))
        .unwrap();
    entry.1 = Some(handle.clone());

    let back = round_trip(value.clone())?;
    assert_eq!(value, back);

    let Value::Animated(AnimatedData::Vector3(back_map)) = &back else {
        anyhow::bail!("expected an animated Vector3 back");
    };
    assert_eq!(
        back_map.values.as_btree_map()[&Tick::new(24)].1,
        Some(handle),
        "the bezier handle's payload did not survive"
    );
    assert_eq!(
        back_map.values.as_btree_map()[&Tick::new(0)].1,
        None,
        "a keyframe with no interpolation key gained one"
    );
    Ok(())
}

/// `RealCurve` is itself a keyframed map, in the `Position` domain rather than
/// the tick domain, and shares the one keyframe-map proxy.
#[cfg(all(feature = "interpolation", feature = "curves"))]
#[test]
fn a_curve_round_trips_with_its_stop_interpolation() -> anyhow::Result<()> {
    let mut curve = RealCurve::linear();
    // SAFETY: `RealCurve::linear` places a stop at position 1.0.
    let stop = unsafe { curve.0.values.as_mut_btree_map() }
        .get_mut(&Position(1.0))
        .unwrap();
    stop.1 = Some(Key {
        interpolation_in: Interpolation::Hold,
        interpolation_out: Interpolation::Bezier(BezierHandle::Angle(0.75)),
    });

    assert_round_trips(Value::Uniform(Data::RealCurve(curve)))
}

/// A double transpose is invisible to a round trip, so the layout is asserted
/// against a hand-written expected order rather than against the input.
#[cfg(feature = "matrix3")]
#[test]
fn matrix_layout_is_column_major_on_the_wire() -> anyhow::Result<()> {
    let matrix = Matrix3(math::mat3_from_column_slice(&ASYMMETRIC_3X3));
    let proxy = MathProxy::<[f32; 9]>::from(&matrix);
    assert_eq!(proxy.0, ASYMMETRIC_3X3);
    // Column-major means element (row 0, col 1) is at index 3, not index 1.
    assert_eq!(math::mat3(&matrix.0, 0, 1), 4.0);
    Ok(())
}

#[test]
fn adversarial_strings_round_trip_exactly() -> anyhow::Result<()> {
    let cases = [
        "with, a comma",
        "with \" a quote",
        "with \\ a backslash",
        "with \\n two characters",
        "with \n a real newline",
        "[bracketed]",
        "key = value",
        "# not a comment",
        " leading space",
        "trailing space ",
        "",
        "ramp/v1;linear;0:0.0,1:1.0",
    ];
    for case in cases {
        let value = Value::Uniform(Data::String(String(case.to_string())));
        let back = round_trip(value.clone())?;
        assert_eq!(value, back, "{case:?} did not survive");
    }

    // The pair the hand-written unescaper conflated must stay distinguishable.
    let escaped = round_trip(Value::Uniform(Data::String(String("a\\nb".to_string()))))?;
    let real = round_trip(Value::Uniform(Data::String(String("a\nb".to_string()))))?;
    assert_ne!(escaped, real);
    Ok(())
}

#[test]
fn floats_round_trip_bit_exactly() -> anyhow::Result<()> {
    for bits in [
        1u64,
        0x8000_0000_0000_0000,
        f64::MIN_POSITIVE.to_bits(),
        (-1.234_567_890_123_456_7e-8f64).to_bits(),
        0.001f64.to_bits(),
    ] {
        let value = Value::Uniform(Data::Real(Real(f64::from_bits(bits))));
        let back = round_trip(value.clone())?;
        assert_eq!(value, back, "bit pattern {bits:#x} did not survive");
    }
    Ok(())
}

#[test]
fn a_keyframe_map_without_a_first_keyframe_is_a_named_error() {
    let text = r#"{"parameters":{"p":{"Animated":{"Real":{"rest":[]}}}}}"#;
    let error = facet_json::from_str::<Document>(text)
        .expect_err("an empty keyframe map must not deserialize");
    let message = format!("{error}");
    assert!(
        message.contains("first"),
        "the error must name the missing keyframe: {message}"
    );
}

#[test]
fn a_payload_that_does_not_match_its_tag_is_an_error() {
    let text = r#"{"parameters":{"p":{"Uniform":{"Integer":"not an integer"}}}}"#;
    assert!(
        facet_json::from_str::<Document>(text).is_err(),
        "a mistyped payload must not deserialize"
    );
}

#[test]
fn a_malformed_array_element_is_an_error_not_a_truncation() {
    let text = r#"{"parameters":{"p":{"Uniform":{"RealVec":[1.0,"x",3.0]}}}}"#;
    assert!(
        facet_json::from_str::<Document>(text).is_err(),
        "a malformed array element must error rather than being dropped"
    );
}

#[test]
fn equal_values_write_identical_documents() -> anyhow::Result<()> {
    let ascending = Value::animated(vec![
        (Tick::new(0), 1.0),
        (Tick::new(24), 2.0),
        (Tick::new(48), 3.0),
    ])?;
    let descending = Value::animated(vec![
        (Tick::new(48), 3.0),
        (Tick::new(24), 2.0),
        (Tick::new(0), 1.0),
    ])?;
    assert_eq!(ascending, descending);

    let render = |value: Value| -> anyhow::Result<std::string::String> {
        Ok(facet_json::to_string(&Document {
            parameters: BTreeMap::from([("p".to_string(), value)]),
        })?)
    };
    assert_eq!(render(ascending)?, render(descending)?);
    Ok(())
}

#[test]
fn a_token_value_map_round_trips() -> anyhow::Result<()> {
    #[derive(Facet, Debug, Clone, PartialEq)]
    struct Wrapper {
        map: TokenValueMap,
    }

    let mut map = TokenValueMap::new();
    map.insert("alpha", Value::Uniform(Data::Real(Real(0.5))));
    map.insert(
        "beta",
        Value::animated(vec![(Tick::new(0), 1.0), (Tick::new(10), 2.0)])?,
    );

    let wrapper = Wrapper { map };
    let text = facet_json::to_string(&wrapper)?;
    let back: Wrapper = facet_json::from_str(&text)
        .map_err(|error| anyhow::anyhow!("{error}\n--- document ---\n{text}"))?;
    assert_eq!(wrapper, back);
    Ok(())
}
