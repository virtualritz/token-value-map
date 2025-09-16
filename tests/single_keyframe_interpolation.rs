use anyhow::Result;
use token_value_map::{AnimatedData, Data, Time, Value};

#[test]
fn single_keyframe_interpolation() -> Result<()> {
    // Test with a single keyframe at time 0.
    let mut animated = AnimatedData::from((Time::from_secs(0.0), Data::Real(42.0.into())));

    // Interpolating at the exact keyframe time should return the keyframe value.
    let result = animated.interpolate(Time::from_secs(0.0));
    assert_eq!(result, Data::Real(42.0.into()));

    // Interpolating before the keyframe should return the keyframe value.
    let result = animated.interpolate(Time::from_secs(-1.0));
    assert_eq!(result, Data::Real(42.0.into()));

    // Interpolating after the keyframe should return the keyframe value.
    let result = animated.interpolate(Time::from_secs(1.0));
    assert_eq!(result, Data::Real(42.0.into()));

    Ok(())
}

#[test]
fn single_keyframe_matrix_interpolation() -> Result<()> {
    // Test with a single matrix keyframe.
    let matrix_data = Data::Matrix3(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0].into());

    let mut animated = AnimatedData::from((Time::from_secs(10.0), matrix_data.clone()));

    // Interpolating at any time should return the single keyframe value.
    let result = animated.interpolate(Time::from_secs(10.0));
    assert_eq!(result, matrix_data);

    let result = animated.interpolate(Time::from_secs(0.0));
    assert_eq!(result, matrix_data);

    let result = animated.interpolate(Time::from_secs(20.0));
    assert_eq!(result, matrix_data);

    Ok(())
}

#[test]
fn value_with_single_animated_keyframe() -> Result<()> {
    // Test Value::Animated with a single keyframe.
    let value = Value::Animated(AnimatedData::from((
        Time::from_secs(5.0),
        Data::IntegerVec(vec![100, 200].into()),
    )));

    // Interpolating should work without panic.
    let result = value.interpolate(Time::from_secs(5.0));
    assert_eq!(result, Data::IntegerVec(vec![100, 200].into()));

    let result = value.interpolate(Time::from_secs(0.0));
    assert_eq!(result, Data::IntegerVec(vec![100, 200].into()));

    let result = value.interpolate(Time::from_secs(10.0));
    assert_eq!(result, Data::IntegerVec(vec![100, 200].into()));

    Ok(())
}

#[test]
fn edge_case_two_keyframes_at_same_time() -> Result<()> {
    // This shouldn't normally happen, but test it anyway.
    // When two keyframes are at the same time, the later one should be used.
    let mut animated = AnimatedData::from((Time::from_secs(5.0), Data::Real(1.0.into())));
    animated.try_insert(Time::from_secs(5.0), Data::Real(2.0.into()))?;

    // Should return the second value (BTreeMap behavior).
    let result = animated.interpolate(Time::from_secs(5.0));
    assert_eq!(result, Data::Real(2.0.into()));

    Ok(())
}
