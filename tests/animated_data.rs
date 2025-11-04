use anyhow::Result;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};
use token_value_map::*;

#[test]
fn animated_data_creation() -> Result<()> {
    // Note: AnimatedData can only be created with samples via
    // Value::animated() These tests now verify data types through
    // populated instances
    let boolean_data = AnimatedData::Boolean(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Boolean(true),
    )]));
    assert_eq!(boolean_data.data_type(), DataType::Boolean);
    assert!(!boolean_data.is_empty());

    let real_data = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));
    assert_eq!(real_data.data_type(), DataType::Real);
    assert!(!real_data.is_empty());

    #[cfg(feature = "vector3")]
    {
        let vector3_data = AnimatedData::Vector3(TimeDataMap::from_iter(vec![(
            Time::from_secs(0.0),
            Vector3(nalgebra::Vector3::new(1.0, 2.0, 3.0)),
        )]));
        assert_eq!(vector3_data.data_type(), DataType::Vector3);
        assert!(!vector3_data.is_empty());
    }

    Ok(())
}

#[test]
fn animated_data_insert_and_sample() -> Result<()> {
    let mut real_data = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));

    // Insert additional samples
    real_data.insert_real(Time::from_secs(1.0), Real(2.0))?;
    real_data.insert_real(Time::from_secs(2.0), Real(3.0))?;

    assert_eq!(real_data.len(), 3);
    assert!(real_data.is_animated());

    // Test exact sampling
    let sample = real_data.sample_at(Time::from_secs(1.0));
    assert_eq!(sample, Some(Data::Real(Real(2.0))));

    // Test interpolation
    let interpolated = real_data.interpolate(Time::from_secs(0.5));
    assert_eq!(interpolated, Data::Real(Real(1.5)));

    Ok(())
}

#[test]
fn animated_data_type_safety() -> Result<()> {
    let mut real_data = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));

    // This should work
    assert!(
        real_data
            .insert_real(Time::from_secs(0.0), Real(1.0))
            .is_ok()
    );

    // This should fail
    assert!(
        real_data
            .insert_integer(Time::from_secs(0.0), Integer(1))
            .is_err()
    );

    Ok(())
}

#[test]
fn animated_data_boolean_no_interpolation() -> Result<()> {
    let mut bool_data = AnimatedData::Boolean(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Boolean(false),
    )]));

    bool_data.insert_boolean(Time::from_secs(1.0), Boolean(true))?;

    // Boolean values should use closest sample, not interpolation
    let sample = bool_data.interpolate(Time::from_secs(0.3));
    assert_eq!(sample, Data::Boolean(Boolean(false)));

    let sample = bool_data.interpolate(Time::from_secs(0.7));
    assert_eq!(sample, Data::Boolean(Boolean(true)));

    Ok(())
}

#[test]
#[cfg(feature = "vector3")]
fn animated_data_vector_interpolation() -> Result<()> {
    let mut vec3_data = AnimatedData::Vector3(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Vector3(nalgebra::Vector3::new(0.0, 0.0, 0.0)),
    )]));

    vec3_data.insert_vector3(
        Time::from_secs(1.0),
        Vector3(nalgebra::Vector3::new(1.0, 2.0, 3.0)),
    )?;

    let interpolated = vec3_data.interpolate(Time::from_secs(0.5));
    assert_eq!(
        interpolated,
        Data::Vector3(Vector3(nalgebra::Vector3::new(0.5, 1.0, 1.5)))
    );

    Ok(())
}

#[test]
fn animated_data_single_sample() -> Result<()> {
    let real_data = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(42.0),
    )]));

    assert_eq!(real_data.len(), 1);
    assert!(!real_data.is_animated()); // Single sample is not animated

    // Any time should return the single value
    let sample = real_data.interpolate(Time::from_secs(100.0));
    assert_eq!(sample, Data::Real(Real(42.0)));

    Ok(())
}

#[test]
fn animated_data_hash_consistency() -> Result<()> {
    let mut real_data1 = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));
    let mut real_data2 = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));

    real_data1.insert_real(Time::from_secs(1.0), Real(2.0))?;
    real_data2.insert_real(Time::from_secs(1.0), Real(2.0))?;

    let mut hasher1 = DefaultHasher::new();
    let mut hasher2 = DefaultHasher::new();

    real_data1.hash(&mut hasher1);
    real_data2.hash(&mut hasher2);

    assert_eq!(hasher1.finish(), hasher2.finish());

    Ok(())
}

#[test]
fn animated_data_generic_insert() -> Result<()> {
    let mut real_data = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));

    // Test successful insertion with matching types
    let result = real_data.try_insert(Time::from_secs(1.0), Data::Real(Real(2.0)));
    assert!(result.is_ok());

    assert_eq!(real_data.len(), 2);

    // Test exact sampling
    let sample = real_data.sample_at(Time::from_secs(1.0));
    assert_eq!(sample, Some(Data::Real(Real(2.0))));

    // Test interpolation
    let interpolated = real_data.interpolate(Time::from_secs(0.5));
    assert_eq!(interpolated, Data::Real(Real(1.5)));

    Ok(())
}

#[test]
fn animated_data_generic_insert_type_mismatch() -> Result<()> {
    let mut real_data = AnimatedData::Real(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Real(1.0),
    )]));

    // Test that inserting wrong type fails
    let result = real_data.try_insert(Time::from_secs(0.0), Data::Boolean(Boolean(true)));
    assert!(result.is_err());

    // Should still have the original sample
    assert!(!real_data.is_empty());

    Ok(())
}

#[test]
fn animated_data_generic_insert_all_types() -> Result<()> {
    // Test that generic insert works for all types
    let mut boolean_data = AnimatedData::Boolean(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Boolean(true),
    )]));
    assert!(
        boolean_data
            .try_insert(Time::from_secs(1.0), Data::Boolean(Boolean(false)))
            .is_ok()
    );

    let mut integer_data = AnimatedData::Integer(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Integer(42),
    )]));
    assert!(
        integer_data
            .try_insert(Time::from_secs(1.0), Data::Integer(Integer(24)))
            .is_ok()
    );

    let mut color_data = AnimatedData::Color(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Color([1.0, 0.0, 0.0, 1.0]),
    )]));
    assert!(
        color_data
            .try_insert(
                Time::from_secs(1.0),
                Data::Color(Color([0.0, 1.0, 0.0, 1.0]))
            )
            .is_ok()
    );

    #[cfg(feature = "vector3")]
    {
        let mut vector3_data = AnimatedData::Vector3(TimeDataMap::from_iter(vec![(
            Time::from_secs(0.0),
            Vector3(nalgebra::Vector3::new(1.0, 2.0, 3.0)),
        )]));
        assert!(
            vector3_data
                .try_insert(
                    Time::from_secs(1.0),
                    Data::Vector3(Vector3(nalgebra::Vector3::new(4.0, 5.0, 6.0)))
                )
                .is_ok()
        );
    }

    let mut real_vec_data = AnimatedData::RealVec(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        RealVec::new(vec![1.0, 2.0, 3.0])?,
    )]));
    assert!(
        real_vec_data
            .try_insert(
                Time::from_secs(1.0),
                Data::RealVec(RealVec::new(vec![4.0, 5.0, 6.0])?)
            )
            .is_ok()
    );

    Ok(())
}

#[test]
fn from_time_data_scalar_types() -> Result<()> {
    // Test Boolean
    let boolean_animated = AnimatedData::from((Time::from_secs(1.0), Data::Boolean(Boolean(true))));
    assert_eq!(boolean_animated.data_type(), DataType::Boolean);
    assert_eq!(boolean_animated.len(), 1);
    assert_eq!(
        boolean_animated.sample_at(Time::from_secs(1.0)),
        Some(Data::Boolean(Boolean(true)))
    );

    // Test Integer
    let integer_animated = AnimatedData::from((Time::from_secs(2.0), Data::Integer(Integer(42))));
    assert_eq!(integer_animated.data_type(), DataType::Integer);
    assert_eq!(integer_animated.len(), 1);
    assert_eq!(
        integer_animated.sample_at(Time::from_secs(2.0)),
        Some(Data::Integer(Integer(42)))
    );

    // Test Real
    let real_animated = AnimatedData::from((Time::from_secs(3.0), Data::Real(Real(3.5))));
    assert_eq!(real_animated.data_type(), DataType::Real);
    assert_eq!(real_animated.len(), 1);
    assert_eq!(
        real_animated.sample_at(Time::from_secs(3.0)),
        Some(Data::Real(Real(3.5)))
    );

    // Test String
    let string_animated = AnimatedData::from((
        Time::from_secs(4.0),
        Data::String(String("hello".to_string())),
    ));
    assert_eq!(string_animated.data_type(), DataType::String);
    assert_eq!(string_animated.len(), 1);
    assert_eq!(
        string_animated.sample_at(Time::from_secs(4.0)),
        Some(Data::String(String("hello".to_string())))
    );

    Ok(())
}

#[test]
fn from_time_data_vector_types() -> Result<()> {
    // Test Color
    let color_animated = AnimatedData::from((
        Time::from_secs(1.0),
        Data::Color(Color([1.0, 0.5, 0.2, 1.0])),
    ));
    assert_eq!(color_animated.data_type(), DataType::Color);
    assert_eq!(color_animated.len(), 1);
    assert_eq!(
        color_animated.sample_at(Time::from_secs(1.0)),
        Some(Data::Color(Color([1.0, 0.5, 0.2, 1.0])))
    );

    // Test Vector2
    #[cfg(feature = "vector2")]
    {
        let vector2_animated = AnimatedData::from((
            Time::from_secs(2.0),
            Data::Vector2(Vector2(nalgebra::Vector2::new(1.0, 2.0))),
        ));
        assert_eq!(vector2_animated.data_type(), DataType::Vector2);
        assert_eq!(vector2_animated.len(), 1);
        assert_eq!(
            vector2_animated.sample_at(Time::from_secs(2.0)),
            Some(Data::Vector2(Vector2(nalgebra::Vector2::new(1.0, 2.0))))
        );
    }

    // Test Vector3
    #[cfg(feature = "vector3")]
    {
        let vector3_animated = AnimatedData::from((
            Time::from_secs(3.0),
            Data::Vector3(Vector3(nalgebra::Vector3::new(1.0, 2.0, 3.0))),
        ));
        assert_eq!(vector3_animated.data_type(), DataType::Vector3);
        assert_eq!(vector3_animated.len(), 1);
        assert_eq!(
            vector3_animated.sample_at(Time::from_secs(3.0)),
            Some(Data::Vector3(Vector3(nalgebra::Vector3::new(
                1.0, 2.0, 3.0
            ))))
        );
    }

    // Test Matrix3
    #[cfg(feature = "matrix3")]
    {
        let matrix = nalgebra::Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let matrix3_animated =
            AnimatedData::from((Time::from_secs(4.0), Data::Matrix3(Matrix3(matrix))));
        assert_eq!(matrix3_animated.data_type(), DataType::Matrix3);
        assert_eq!(matrix3_animated.len(), 1);
        assert_eq!(
            matrix3_animated.sample_at(Time::from_secs(4.0)),
            Some(Data::Matrix3(Matrix3(matrix)))
        );
    }

    Ok(())
}

#[test]
fn from_time_data_vec_types() -> Result<()> {
    // Test BooleanVec
    let boolean_vec_animated = AnimatedData::from((
        Time::from_secs(1.0),
        Data::BooleanVec(BooleanVec::new(vec![true, false, true])?),
    ));
    assert_eq!(boolean_vec_animated.data_type(), DataType::BooleanVec);
    assert_eq!(boolean_vec_animated.len(), 1);

    // Test IntegerVec
    let integer_vec_animated = AnimatedData::from((
        Time::from_secs(2.0),
        Data::IntegerVec(IntegerVec::new(vec![1, 2, 3])?),
    ));
    assert_eq!(integer_vec_animated.data_type(), DataType::IntegerVec);
    assert_eq!(integer_vec_animated.len(), 1);

    // Test RealVec
    let real_vec_animated = AnimatedData::from((
        Time::from_secs(3.0),
        Data::RealVec(RealVec::new(vec![1.0, 2.0, 3.0])?),
    ));
    assert_eq!(real_vec_animated.data_type(), DataType::RealVec);
    assert_eq!(real_vec_animated.len(), 1);

    // Test StringVec
    let string_vec_animated = AnimatedData::from((
        Time::from_secs(4.0),
        Data::StringVec(StringVec::new(vec!["a".to_string(), "b".to_string()])?),
    ));
    assert_eq!(string_vec_animated.data_type(), DataType::StringVec);
    assert_eq!(string_vec_animated.len(), 1);

    // Test ColorVec
    let color_vec_animated = AnimatedData::from((
        Time::from_secs(5.0),
        Data::ColorVec(ColorVec::new(vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ])?),
    ));
    assert_eq!(color_vec_animated.data_type(), DataType::ColorVec);
    assert_eq!(color_vec_animated.len(), 1);

    // Test Vector2Vec
    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
    {
        let vector2_vec_animated = AnimatedData::from((
            Time::from_secs(6.0),
            Data::Vector2Vec(Vector2Vec::new(vec![nalgebra::Vector2::new(1.0, 2.0)])?),
        ));
        assert_eq!(vector2_vec_animated.data_type(), DataType::Vector2Vec);
        assert_eq!(vector2_vec_animated.len(), 1);
    }

    // Test Vector3Vec
    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
    {
        let vector3_vec_animated = AnimatedData::from((
            Time::from_secs(7.0),
            Data::Vector3Vec(Vector3Vec::new(vec![nalgebra::Vector3::new(
                1.0, 2.0, 3.0,
            )])?),
        ));
        assert_eq!(vector3_vec_animated.data_type(), DataType::Vector3Vec);
        assert_eq!(vector3_vec_animated.len(), 1);
    }

    // Test Matrix3Vec
    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
    {
        let matrix = nalgebra::Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let matrix3_vec_animated = AnimatedData::from((
            Time::from_secs(8.0),
            Data::Matrix3Vec(Matrix3Vec::new(vec![matrix])?),
        ));
        assert_eq!(matrix3_vec_animated.data_type(), DataType::Matrix3Vec);
        assert_eq!(matrix3_vec_animated.len(), 1);
    }

    Ok(())
}

#[test]
fn from_time_data_interpolation_works() -> Result<()> {
    // Create animated data using From and test that interpolation still works
    let mut real_animated = AnimatedData::from((Time::from_secs(0.0), Data::Real(Real(0.0))));

    // Add another sample using try_insert
    real_animated.try_insert(Time::from_secs(2.0), Data::Real(Real(2.0)))?;

    assert_eq!(real_animated.len(), 2);
    assert!(real_animated.is_animated());

    // Test interpolation at midpoint
    let interpolated = real_animated.interpolate(Time::from_secs(1.0));
    assert_eq!(interpolated, Data::Real(Real(1.0)));

    // Test exact samples
    assert_eq!(
        real_animated.sample_at(Time::from_secs(0.0)),
        Some(Data::Real(Real(0.0)))
    );
    assert_eq!(
        real_animated.sample_at(Time::from_secs(2.0)),
        Some(Data::Real(Real(2.0)))
    );

    Ok(())
}

#[test]
#[cfg(feature = "matrix3")]
fn matrix3_interpolation_over_time() -> Result<()> {
    // Test Matrix3 interpolation with identity and rotation matrices.
    let mut matrix3_data = AnimatedData::Matrix3(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Matrix3(nalgebra::Matrix3::identity()),
    )]));

    // Add a 90-degree rotation matrix around Z axis at t=1.0.
    let rotation_90 = nalgebra::Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    matrix3_data.insert_matrix3(Time::from_secs(1.0), Matrix3(rotation_90))?;

    // Add a 180-degree rotation matrix at t=2.0.
    let rotation_180 = nalgebra::Matrix3::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0);
    matrix3_data.insert_matrix3(Time::from_secs(2.0), Matrix3(rotation_180))?;

    assert_eq!(matrix3_data.len(), 3);
    assert!(matrix3_data.is_animated());

    // Test exact samples.
    assert_eq!(
        matrix3_data.sample_at(Time::from_secs(0.0)),
        Some(Data::Matrix3(Matrix3(nalgebra::Matrix3::identity())))
    );
    assert_eq!(
        matrix3_data.sample_at(Time::from_secs(1.0)),
        Some(Data::Matrix3(Matrix3(rotation_90)))
    );
    assert_eq!(
        matrix3_data.sample_at(Time::from_secs(2.0)),
        Some(Data::Matrix3(Matrix3(rotation_180)))
    );

    // Test interpolation at t=0.5 (halfway between identity and 90-degree rotation).
    let interpolated = matrix3_data.interpolate(Time::from_secs(0.5));
    if let Data::Matrix3(Matrix3(mat)) = interpolated {
        // With 3 samples, quadratic interpolation is used.
        // The interpolated values are different from simple linear interpolation.
        assert!((mat[(0, 0)] - 0.5).abs() < 0.01);
        assert!((mat[(1, 0)] - 0.75).abs() < 0.01); // Quadratic gives 0.75 not 0.5.
        assert!((mat[(0, 1)] - (-0.75)).abs() < 0.01); // Quadratic gives -0.75 not -0.5.
        assert!((mat[(1, 1)] - 0.5).abs() < 0.01);
        assert!((mat[(2, 2)] - 1.0).abs() < 0.01);
    } else {
        panic!("Expected Matrix3 data");
    }

    // Test interpolation at t=1.5 (halfway between 90 and 180 degrees).
    let interpolated = matrix3_data.interpolate(Time::from_secs(1.5));
    if let Data::Matrix3(Matrix3(mat)) = interpolated {
        // With 3 samples, quadratic interpolation is used.
        // Based on quadratic interpolation between rotation matrices.
        assert!((mat[(0, 0)] - (-0.5)).abs() < 0.01);
        assert!((mat[(1, 0)] - 0.75).abs() < 0.01); // Quadratic result.
        assert!((mat[(0, 1)] - (-0.75)).abs() < 0.01); // Quadratic result.
        assert!((mat[(1, 1)] - (-0.5)).abs() < 0.01);
        assert!((mat[(2, 2)] - 1.0).abs() < 0.01);
    } else {
        panic!("Expected Matrix3 data");
    }

    Ok(())
}

#[test]
#[cfg(feature = "matrix4")]
fn matrix4_interpolation_over_time() -> Result<()> {
    // Test Matrix4 interpolation with transformation matrices.
    let mut matrix4_data = AnimatedData::Matrix4(TimeDataMap::from_iter(vec![(
        Time::from_secs(0.0),
        Matrix4(nalgebra::Matrix4::identity()),
    )]));

    // Add a translation matrix at t=1.0.
    let translation = nalgebra::Matrix4::new(
        1.0, 0.0, 0.0, 10.0, 0.0, 1.0, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0, 0.0, 0.0, 0.0, 1.0,
    );
    matrix4_data.insert_matrix4(Time::from_secs(1.0), Matrix4(translation))?;

    // Add a scaled translation matrix at t=2.0.
    let scaled_translation = nalgebra::Matrix4::new(
        2.0, 0.0, 0.0, 20.0, 0.0, 2.0, 0.0, 40.0, 0.0, 0.0, 2.0, 60.0, 0.0, 0.0, 0.0, 1.0,
    );
    matrix4_data.insert_matrix4(Time::from_secs(2.0), Matrix4(scaled_translation))?;

    assert_eq!(matrix4_data.len(), 3);
    assert!(matrix4_data.is_animated());

    // Test exact samples.
    assert_eq!(
        matrix4_data.sample_at(Time::from_secs(0.0)),
        Some(Data::Matrix4(Matrix4(nalgebra::Matrix4::identity())))
    );
    assert_eq!(
        matrix4_data.sample_at(Time::from_secs(1.0)),
        Some(Data::Matrix4(Matrix4(translation)))
    );
    assert_eq!(
        matrix4_data.sample_at(Time::from_secs(2.0)),
        Some(Data::Matrix4(Matrix4(scaled_translation)))
    );

    // Test interpolation at t=0.5 (halfway between identity and translation).
    let interpolated = matrix4_data.interpolate(Time::from_secs(0.5));
    if let Data::Matrix4(Matrix4(mat)) = interpolated {
        // With 3 samples, quadratic interpolation is used.
        // The scale values are also affected by quadratic interpolation.
        assert!((mat[(0, 0)] - 0.875).abs() < 0.01); // Quadratic gives 0.875.
        assert!((mat[(1, 1)] - 0.875).abs() < 0.01);
        assert!((mat[(2, 2)] - 0.875).abs() < 0.01);
        assert!((mat[(0, 3)] - 5.0).abs() < 0.01); // Translation X.
        assert!((mat[(1, 3)] - 10.0).abs() < 0.01); // Translation Y.
        assert!((mat[(2, 3)] - 15.0).abs() < 0.01); // Translation Z.
        assert!((mat[(3, 3)] - 1.0).abs() < 0.01);
    } else {
        panic!("Expected Matrix4 data");
    }

    // Test interpolation at t=1.5 (halfway between translation and scaled translation).
    let interpolated = matrix4_data.interpolate(Time::from_secs(1.5));
    if let Data::Matrix4(Matrix4(mat)) = interpolated {
        // With 3 samples, quadratic interpolation is used.
        // The scale values are affected by quadratic interpolation.
        assert!((mat[(0, 0)] - 1.375).abs() < 0.01); // Quadratic gives 1.375.
        assert!((mat[(1, 1)] - 1.375).abs() < 0.01);
        assert!((mat[(2, 2)] - 1.375).abs() < 0.01);
        assert!((mat[(0, 3)] - 15.0).abs() < 0.01); // Translation X.
        assert!((mat[(1, 3)] - 30.0).abs() < 0.01); // Translation Y.
        assert!((mat[(2, 3)] - 45.0).abs() < 0.01); // Translation Z.
        assert!((mat[(3, 3)] - 1.0).abs() < 0.01);
    } else {
        panic!("Expected Matrix4 data");
    }

    Ok(())
}

#[test]
fn from_time_value_delegates_to_from_time_data() -> Result<()> {
    // Test that From<(Time, Value)> correctly delegates to From<(Time, Data)>
    // for uniform values
    let uniform_value = Value::uniform(42.0);
    let animated_from_value = AnimatedData::from((Time::from_secs(1.0), uniform_value));

    let data = Data::Real(Real(42.0));
    let animated_from_data = AnimatedData::from((Time::from_secs(1.0), data));

    // Both should be equivalent
    assert_eq!(
        animated_from_value.data_type(),
        animated_from_data.data_type()
    );
    assert_eq!(animated_from_value.len(), animated_from_data.len());
    assert_eq!(
        animated_from_value.sample_at(Time::from_secs(1.0)),
        animated_from_data.sample_at(Time::from_secs(1.0))
    );

    Ok(())
}
