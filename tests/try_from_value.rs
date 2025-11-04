use anyhow::Result;
use token_value_map::*;

#[test]
fn try_from_value_basic_types() -> Result<()> {
    // Test Boolean
    let value = Value::uniform(true);
    let boolean: Boolean = value.try_into()?;
    assert!(boolean.0);

    // Test Integer
    let value = Value::uniform(42i64);
    let integer: Integer = value.try_into()?;
    assert_eq!(integer.0, 42);

    // Test Real
    let value = Value::uniform(3.5f64);
    let real: Real = value.try_into()?;
    assert_eq!(real.0, 3.5);

    // Test String
    let value = Value::uniform("hello");
    let string: String = value.try_into()?;
    assert_eq!(string.0, "hello");

    Ok(())
}

#[test]
fn try_from_value_with_conversion() -> Result<()> {
    // Test converting a Real to Integer through try_convert
    let value = Value::uniform(42.7f64);
    let integer: Integer = value.try_into()?;
    assert_eq!(integer.0, 42);

    // Test converting an Integer to Real
    let value = Value::uniform(42i64);
    let real: Real = value.try_into()?;
    assert_eq!(real.0, 42.0);

    // Test converting a Boolean to String
    let value = Value::uniform(true);
    let string: String = value.try_into()?;
    assert_eq!(string.0, "true");

    Ok(())
}

#[test]
fn try_from_value_vector_types() -> Result<()> {
    // Test Color
    let value = Value::uniform([1.0, 0.5, 0.0, 1.0]);
    let color: Color = value.try_into()?;
    assert_eq!(color.0, [1.0, 0.5, 0.0, 1.0]);

    // Test Vector2
    #[cfg(feature = "vector2")]
    {
        let value = Value::uniform([1.0, 2.0]);
        let vec2: Vector2 = value.try_into()?;
        assert_eq!(vec2.0.x, 1.0);
        assert_eq!(vec2.0.y, 2.0);
    }

    // Test Vector3
    #[cfg(feature = "vector3")]
    {
        let value = Value::uniform([1.0, 2.0, 3.0]);
        let vec3: Vector3 = value.try_into()?;
        assert_eq!(vec3.0.x, 1.0);
        assert_eq!(vec3.0.y, 2.0);
        assert_eq!(vec3.0.z, 3.0);
    }

    Ok(())
}

#[test]
fn try_from_value_animated_fails() -> Result<()> {
    // Test that animated values cannot be converted to simple types
    let animated = Value::animated(vec![
        (frame_tick::Tick::new(0), 1.0),
        (frame_tick::Tick::new(10), 2.0),
    ])?;

    let result: Result<Real> = animated.try_into();
    assert!(result.is_err());
    assert!(
        result
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("Cannot convert animated value")
    );

    Ok(())
}

#[test]
fn try_from_value_conversion_errors() {
    // Test conversion that should fail (vec types don't support all
    // conversions)
    let value = Value::uniform(42i64);

    // Try to convert to a vector type that try_convert doesn't support
    let result: Result<BooleanVec> = value.try_into();
    assert!(result.is_err());
}

#[test]
fn try_from_value_ref_basic_types() -> Result<()> {
    // Test Boolean
    let value = Value::uniform(true);
    let boolean: Boolean = (&value).try_into()?;
    assert!(boolean.0);

    // Test Integer
    let value = Value::uniform(42i64);
    let integer: Integer = (&value).try_into()?;
    assert_eq!(integer.0, 42);

    // Test Real
    let value = Value::uniform(3.5f64);
    let real: Real = (&value).try_into()?;
    assert_eq!(real.0, 3.5);

    // Test String
    let value = Value::uniform("hello");
    let string: String = (&value).try_into()?;
    assert_eq!(string.0, "hello");

    Ok(())
}

#[test]
fn try_from_value_ref_with_conversion() -> Result<()> {
    // Test converting a Real to Integer through try_convert
    let value = Value::uniform(42.7f64);
    let integer: Integer = (&value).try_into()?;
    assert_eq!(integer.0, 42);

    // Test converting an Integer to Real
    let value = Value::uniform(42i64);
    let real: Real = (&value).try_into()?;
    assert_eq!(real.0, 42.0);

    // Test converting a Boolean to String
    let value = Value::uniform(true);
    let string: String = (&value).try_into()?;
    assert_eq!(string.0, "true");

    Ok(())
}

#[test]
fn try_from_value_ref_animated_fails() -> Result<()> {
    // Test that animated values cannot be converted to simple types
    let animated = Value::animated(vec![
        (frame_tick::Tick::new(0), 1.0),
        (frame_tick::Tick::new(10), 2.0),
    ])?;

    let result: Result<Real> = (&animated).try_into();
    assert!(result.is_err());
    assert!(
        result
            .as_ref()
            .unwrap_err()
            .to_string()
            .contains("Cannot convert animated value")
    );

    Ok(())
}
