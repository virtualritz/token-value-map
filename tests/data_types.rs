use token_value_map::*;

#[test]
fn test_empty_vec_validation() {
    // Test that empty vectors are rejected
    assert!(IntegerVec::new(vec![]).is_err());
    assert!(RealVec::new(vec![]).is_err());
    assert!(BooleanVec::new(vec![]).is_err());
    assert!(StringVec::new(vec![]).is_err());
    assert!(ColorVec::new(vec![]).is_err());
    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
    assert!(Vector2Vec::new(vec![]).is_err());
    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
    assert!(Vector3Vec::new(vec![]).is_err());
    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
    assert!(Matrix3Vec::new(vec![]).is_err());
}

#[test]
fn test_non_empty_vec_creation() {
    // Test that non-empty vectors are accepted
    assert!(IntegerVec::new(vec![1]).is_ok());
    assert!(RealVec::new(vec![1.0]).is_ok());
    assert!(BooleanVec::new(vec![true]).is_ok());
    assert!(StringVec::new(vec!["test".to_string()]).is_ok());
    assert!(ColorVec::new(vec![[1.0, 0.0, 0.0, 1.0]]).is_ok());
    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
    assert!(Vector2Vec::new(vec![nalgebra::Vector2::new(1.0, 2.0)]).is_ok());
    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
    assert!(Vector3Vec::new(vec![nalgebra::Vector3::new(1.0, 2.0, 3.0)]).is_ok());
    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
    assert!(Matrix3Vec::new(vec![nalgebra::Matrix3::identity()]).is_ok());
}

#[test]
fn test_vec_error_messages() {
    // Test that error messages are descriptive
    let err = IntegerVec::new(vec![]).unwrap_err();
    assert!(err.to_string().contains("IntegerVec cannot be empty"));

    let err = RealVec::new(vec![]).unwrap_err();
    assert!(err.to_string().contains("RealVec cannot be empty"));

    let err = BooleanVec::new(vec![]).unwrap_err();
    assert!(err.to_string().contains("BooleanVec cannot be empty"));
}
