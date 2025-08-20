use core::num::NonZeroU16;
use token_value_map::*;

#[test]
fn test_value_from_primitives() {
    let _: Data = 42i64.into();
    let _: Data = 3.14f64.into();
    let _: Data = true.into();
    let _: Data = "hello".into();
    #[cfg(feature = "vector2")]
    {
        let _: Data = [1.0, 2.0].into();
    }
    #[cfg(feature = "vector3")]
    {
        let _: Data = [1.0, 2.0, 3.0].into();
    }
    let _: Data = [1.0, 0.5, 0.0, 1.0].into(); // Color
}

#[test]
fn test_value_to_primitives() {
    let v: Data = 42i64.into();
    assert_eq!(i64::try_from(v).unwrap(), 42);

    let v: Data = [1.0, 0.5, 0.0, 1.0].into();
    assert_eq!(<[f32; 4]>::try_from(v).unwrap(), [1.0, 0.5, 0.0, 1.0]);
}

#[test]
fn test_data_type() {
    assert_eq!(Data::from(42i64).data_type(), DataType::Integer);
    assert_eq!(Data::from(3.14).data_type(), DataType::Real);
    assert_eq!(Data::from(true).data_type(), DataType::Boolean);
    assert_eq!(Data::from("hello").data_type(), DataType::String);
}

#[test]
fn test_try_convert() {
    // Integer to other types
    let v = Data::from(42i64);
    assert_eq!(v.try_convert(DataType::Real).unwrap(), Data::from(42.0));
    assert_eq!(v.try_convert(DataType::Boolean).unwrap(), Data::from(true));
    assert_eq!(v.try_convert(DataType::String).unwrap(), Data::from("42"));

    // Boolean conversions
    let v = Data::from(true);
    assert_eq!(v.try_convert(DataType::Integer).unwrap(), Data::from(1i64));
    assert_eq!(v.try_convert(DataType::Real).unwrap(), Data::from(1.0));

    let v = Data::from(false);
    assert_eq!(v.try_convert(DataType::Integer).unwrap(), Data::from(0i64));

    // String parsing
    let v = Data::from("123");
    assert_eq!(
        v.try_convert(DataType::Integer).unwrap(),
        Data::from(123i64)
    );
    assert_eq!(v.try_convert(DataType::Real).unwrap(), Data::from(123.0));

    let v = Data::from("true");
    assert_eq!(v.try_convert(DataType::Boolean).unwrap(), Data::from(true));

    // Fill conversions
    #[cfg(feature = "vector2")]
    {
        let v = Data::from(5.0);
        let vec2 = v.try_convert(DataType::Vector2).unwrap();
        assert_eq!(
            nalgebra::Vector2::<f32>::try_from(vec2).unwrap(),
            nalgebra::Vector2::new(5.0, 5.0)
        );
    }

    #[cfg(feature = "vector3")]
    {
        let v = Data::from(5.0);
        let vec3 = v.try_convert(DataType::Vector3).unwrap();
        assert_eq!(
            nalgebra::Vector3::<f32>::try_from(vec3).unwrap(),
            nalgebra::Vector3::new(5.0, 5.0, 5.0)
        );
    }

    // Color from grayscale
    let v = Data::from(0.5);
    let color = v.try_convert(DataType::Color).unwrap();
    assert_eq!(<[f32; 4]>::try_from(color).unwrap(), [0.5, 0.5, 0.5, 1.0]);

    // Vec3 to Color
    #[cfg(feature = "vector3")]
    {
        let v = Data::from([1.0, 0.5, 0.25]);
        let color = v.try_convert(DataType::Color).unwrap();
        assert_eq!(<[f32; 4]>::try_from(color).unwrap(), [1.0, 0.5, 0.25, 1.0]);
    }
}

#[test]
fn test_string_parsing() {
    // Vec2 parsing
    #[cfg(feature = "vector2")]
    {
        let v = Data::from("[1.0, 2.0]");
        let vec2 = v.try_convert(DataType::Vector2).unwrap();
        assert_eq!(
            nalgebra::Vector2::<f32>::try_from(vec2).unwrap(),
            nalgebra::Vector2::new(1.0, 2.0)
        );
    }

    // Color hex parsing
    let v = Data::from("#FF0000");
    let color = v.try_convert(DataType::Color).unwrap();
    assert_eq!(<[f32; 4]>::try_from(color).unwrap(), [1.0, 0.0, 0.0, 1.0]);

    // Matrix parsing (diagonal)
    #[cfg(feature = "matrix3")]
    {
        let v = Data::from("5.0");
        let mat = v.try_convert(DataType::Matrix3).unwrap();
        let expected = nalgebra::Matrix3::from_diagonal_element(5.0);
        assert_eq!(nalgebra::Matrix3::<f32>::try_from(mat).unwrap(), expected);
    }
}

#[test]
fn test_attribute_value() -> anyhow::Result<()> {
    let av = Value::animated(vec![
        (Time::from_secs(0.0), 0.0),
        (Time::from_secs(1.0), 1.0),
        (Time::from_secs(2.0), 2.0),
    ])?;

    assert_eq!(av.sample_count(), 3);
    assert!(av.is_animated());

    let (before, _after) = av.sample_bracket(Time::from_secs(1.5));
    assert!(before.is_some());

    Ok(())
}

#[test]
fn test_token_value_map() -> anyhow::Result<()> {
    let mut map = TokenValueMap::new();

    #[cfg(feature = "vector3")]
    map.insert("position", [1.0, 2.0, 3.0]);
    #[cfg(all(not(feature = "vector3"), feature = "vector2"))]
    map.insert("position", [1.0, 2.0]);
    #[cfg(all(not(feature = "vector3"), not(feature = "vector2")))]
    map.insert("position", 42.0);
    map.insert(
        "color",
        Value::animated(vec![
            (Time::from_secs(0.0), [1.0, 0.0, 0.0, 1.0]),
            (Time::from_secs(1.0), [0.0, 1.0, 0.0, 1.0]),
        ])?,
    );

    assert_eq!(map.len(), 2);
    assert!(map.contains(&"position".into()));

    Ok(())
}

#[test]
fn test_value_with_animated_data() -> anyhow::Result<()> {
    // Test creating animated real values
    let animated_real = Value::animated(vec![
        (Time::from_secs(0.0), 1.0),
        (Time::from_secs(1.0), 2.0),
        (Time::from_secs(2.0), 3.0),
    ])?;

    assert!(animated_real.is_animated());
    assert_eq!(animated_real.sample_count(), 3);

    // Test exact sampling
    let sample = animated_real.sample_at(Time::from_secs(1.0));
    assert_eq!(sample, Some(Data::Real(Real(2.0))));

    // Test interpolation
    let interpolated = animated_real.interpolate(Time::from_secs(0.5));
    assert_eq!(interpolated, Data::Real(Real(1.5)));

    Ok(())
}

#[test]
#[cfg(feature = "vector3")]
fn test_value_with_animated_vectors() -> anyhow::Result<()> {
    // Test animated Vector3
    let animated_vec3 = Value::animated(vec![
        (Time::from_secs(0.0), [0.0f32, 0.0, 0.0]),
        (Time::from_secs(1.0), [1.0f32, 2.0, 3.0]),
    ])?;

    assert!(animated_vec3.is_animated());
    assert_eq!(animated_vec3.sample_count(), 2);

    // Test interpolation
    let interpolated = animated_vec3.interpolate(Time::from_secs(0.5));
    if let Data::Vector3(Vector3(v)) = interpolated {
        assert_eq!(v, nalgebra::Vector3::new(0.5, 1.0, 1.5));
    } else {
        return Err(anyhow::anyhow!(
            "Expected Vector3 data, got: {:?}",
            interpolated
        ));
    }

    Ok(())
}

#[test]
fn test_value_add_sample_conversion() -> anyhow::Result<()> {
    // Start with uniform value and add sample to make it animated
    let mut value = Value::uniform(1.0);
    assert!(!value.is_animated());
    assert_eq!(value.sample_count(), 1);

    // Add a sample - should convert to animated and drop uniform content
    value.add_sample(Time::from_secs(1.0), 2.0)?;
    // With only one sample, it's not considered "animated" yet
    assert!(!value.is_animated());
    assert_eq!(value.sample_count(), 1); // Only the new sample should remain

    // Test that we only have the new sample
    let sample = value.sample_at(Time::from_secs(1.0));
    assert_eq!(sample, Some(Data::Real(Real(2.0))));

    // Add another sample to make it truly animated
    value.add_sample(Time::from_secs(2.0), 3.0)?;
    assert!(value.is_animated());
    assert_eq!(value.sample_count(), 2);

    Ok(())
}

#[test]
fn test_value_type_safety() -> anyhow::Result<()> {
    // Test that adding different types fails
    let mut real_value = Value::animated(vec![(Time::from_secs(0.0), 1.0)])?;

    let result = real_value.add_sample(Time::from_secs(1.0), true);
    assert!(result.is_err());

    // Test that creating animated value with mixed types fails
    let mixed_result = Value::animated(vec![
        (Time::from_secs(0.0), Data::Real(Real(1.0))),
        (Time::from_secs(1.0), Data::Boolean(Boolean(true))),
    ]);
    assert!(mixed_result.is_err());

    Ok(())
}

#[test]
fn test_value_animated_boolean_no_interpolation() -> anyhow::Result<()> {
    // Boolean values should not interpolate, just use closest sample
    let animated_bool = Value::animated(vec![
        (Time::from_secs(0.0), false),
        (Time::from_secs(1.0), true),
    ])?;

    // Test that boolean uses closest sample, not interpolation
    let sample_early = animated_bool.interpolate(Time::from_secs(0.3));
    assert_eq!(sample_early, Data::Boolean(Boolean(false)));

    let sample_late = animated_bool.interpolate(Time::from_secs(0.7));
    assert_eq!(sample_late, Data::Boolean(Boolean(true)));

    Ok(())
}

#[test]
fn test_value_empty_animated_creation() {
    // Creating animated value with no samples should fail
    let empty_result: anyhow::Result<Value> = Value::animated(Vec::<(Time, f64)>::new());
    assert!(empty_result.is_err());
}

#[test]
fn test_value_uses_generic_insert() -> anyhow::Result<()> {
    // This test verifies that the simplified implementation using generic
    // insert works
    let animated_mixed = Value::animated(vec![
        (Time::from_secs(0.0), Data::Real(Real(0.0))),
        (Time::from_secs(1.0), Data::Real(Real(1.0))),
        (Time::from_secs(2.0), Data::Real(Real(2.0))),
    ])?;

    assert!(animated_mixed.is_animated());
    assert_eq!(animated_mixed.sample_count(), 3);

    // Test interpolation
    let interpolated = animated_mixed.interpolate(Time::from_secs(1.5));
    assert_eq!(interpolated, Data::Real(Real(1.5)));

    // Test adding more samples
    let mut mutable_value = animated_mixed;
    mutable_value.add_sample(Time::from_secs(3.0), Data::Real(Real(3.0)))?;
    assert_eq!(mutable_value.sample_count(), 4);

    Ok(())
}

#[test]
fn test_data_type_dispatch() {
    // Test that data_type() works for Data
    let data = Data::Real(Real(42.0));
    assert_eq!(data.data_type(), DataType::Real);
    assert_eq!(data.type_name(), "real");

    // Test that data_type() works for AnimatedData
    let animated = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));
    assert_eq!(animated.data_type(), DataType::Real);
    assert_eq!(animated.type_name(), "real");

    // Test that data_type() works for Value (uniform)
    let uniform_value = Value::uniform(42.0);
    assert_eq!(uniform_value.data_type(), DataType::Real);
    assert_eq!(uniform_value.type_name(), "real");

    // Test that data_type() works for Value (animated)
    let animated_value = Value::animated(vec![(Time::from_secs(0.0), 1.0)]).unwrap();
    assert_eq!(animated_value.data_type(), DataType::Real);
    assert_eq!(animated_value.type_name(), "real");

    // Test different types
    let bool_data = Data::Boolean(Boolean(true));
    assert_eq!(bool_data.data_type(), DataType::Boolean);
    assert_eq!(bool_data.type_name(), "boolean");

    #[cfg(feature = "vector3")]
    {
        let vec3_data = Data::Vector3(Vector3(nalgebra::Vector3::new(1.0, 2.0, 3.0)));
        assert_eq!(vec3_data.data_type(), DataType::Vector3);
        assert_eq!(vec3_data.type_name(), "vec3");
    }
}

#[test]
fn test_sample_trait_implementations() -> anyhow::Result<()> {
    // Test uniform Value sampling - should always return 1 sample with the
    // uniform value
    let uniform_real = Value::uniform(42.0);
    let shutter = Shutter {
        range: Time::from_secs(0.0)..Time::from_secs(1.0),
        opening: Time::from_secs(0.0)..Time::from_secs(1.0),
    };
    let samples: Vec<(Real, SampleWeight)> =
        uniform_real.sample(&shutter, NonZeroU16::new(5).unwrap())?;
    assert_eq!(samples.len(), 1);
    assert_eq!(samples[0].0, Real(42.0));
    assert_eq!(samples[0].1, 1.0);

    // Test uniform Vector3 sampling
    #[cfg(feature = "vector3")]
    {
        let uniform_vector = Value::uniform([1.0f32, 2.0, 3.0]);
        let samples: Vec<(Vector3, SampleWeight)> =
            uniform_vector.sample(&shutter, NonZeroU16::new(3).unwrap())?;
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].0, Vector3(nalgebra::Vector3::new(1.0, 2.0, 3.0)));
        assert_eq!(samples[0].1, 1.0);
    }

    // Test animated Value sampling - should return the requested number of
    // samples
    let animated_real = Value::animated(vec![
        (Time::from_secs(0.0), 0.0),
        (Time::from_secs(1.0), 100.0),
    ])?;
    let samples: Vec<(Real, SampleWeight)> =
        animated_real.sample(&shutter, NonZeroU16::new(3).unwrap())?;
    assert_eq!(samples.len(), 3);
    // All samples should be valid (values between 0 and 100)
    for (value, _weight) in samples {
        assert!(value.0 >= 0.0 && value.0 <= 100.0);
    }

    Ok(())
}
