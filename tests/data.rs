use token_value_map::*;

#[test]
fn test_empty_vec_try_from_rejected() {
    // Test that empty vectors are rejected when converting to Data
    assert!(Data::try_from(Vec::<i64>::new()).is_err());
    assert!(Data::try_from(Vec::<f64>::new()).is_err());
    assert!(Data::try_from(Vec::<bool>::new()).is_err());
    assert!(Data::try_from(Vec::<std::string::String>::new()).is_err());
    assert!(Data::try_from(Vec::<&str>::new()).is_err());
    assert!(Data::try_from(Vec::<[f32; 4]>::new()).is_err());
    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
    assert!(Data::try_from(Vec::<nalgebra::Vector2<f32>>::new()).is_err());
    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
    assert!(Data::try_from(Vec::<nalgebra::Vector3<f32>>::new()).is_err());
    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
    assert!(Data::try_from(Vec::<nalgebra::Matrix3<f32>>::new()).is_err());
}

#[test]
fn test_non_empty_vec_try_from_accepted() {
    // Test that non-empty vectors are accepted when converting to Data
    assert!(Data::try_from(vec![1i64]).is_ok());
    assert!(Data::try_from(vec![1.0f64]).is_ok());
    assert!(Data::try_from(vec![true]).is_ok());
    assert!(Data::try_from(vec!["test".to_string()]).is_ok());
    assert!(Data::try_from(vec!["test"]).is_ok());
    assert!(Data::try_from(vec![[1.0f32, 0.0, 0.0, 1.0]]).is_ok());
    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
    assert!(Data::try_from(vec![nalgebra::Vector2::new(1.0f32, 2.0)]).is_ok());
    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
    assert!(
        Data::try_from(vec![nalgebra::Vector3::new(1.0f32, 2.0, 3.0)]).is_ok()
    );
    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
    assert!(Data::try_from(vec![nalgebra::Matrix3::identity()]).is_ok());
}

#[test]
fn test_vec_try_from_error_messages() {
    // Test that error messages are descriptive
    let err = Data::try_from(Vec::<i64>::new()).unwrap_err();
    assert!(err.to_string().contains("IntegerVec cannot be empty"));

    let err = Data::try_from(Vec::<f64>::new()).unwrap_err();
    assert!(err.to_string().contains("RealVec cannot be empty"));

    let err = Data::try_from(Vec::<bool>::new()).unwrap_err();
    assert!(err.to_string().contains("BooleanVec cannot be empty"));
}

#[test]
fn test_vec_data_length_validation() {
    // Test that created Vec Data types have proper lengths
    let data = Data::try_from(vec![1i64, 2, 3]).unwrap();
    assert_eq!(data.len(), 3);
    assert!(data.is_vec());

    let data = Data::try_from(vec![1.0f64, 2.0]).unwrap();
    assert_eq!(data.len(), 2);
    assert!(data.is_vec());

    // Test that single element vectors work
    let data = Data::try_from(vec![true]).unwrap();
    assert_eq!(data.len(), 1);
    assert!(data.is_vec());
}
