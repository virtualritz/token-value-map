use token_value_map::*;

#[test]
fn vec_to_matrix3_conversions() -> Result<()> {
    // RealVec to Matrix3
    let real_vec = Data::RealVec(RealVec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    let matrix3 = real_vec.try_convert(DataType::Matrix3)?;
    if let Data::Matrix3(Matrix3(m)) = matrix3 {
        use token_value_map::math::mat3;
        assert_eq!(mat3(&m, 0, 0), 1.0);
        assert_eq!(mat3(&m, 0, 1), 2.0);
        assert_eq!(mat3(&m, 2, 2), 9.0);
    } else {
        panic!("Expected Matrix3");
    }

    // IntegerVec to Matrix3
    let int_vec = Data::IntegerVec(IntegerVec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9]));
    let matrix3 = int_vec.try_convert(DataType::Matrix3)?;
    if let Data::Matrix3(Matrix3(m)) = matrix3 {
        use token_value_map::math::mat3;
        assert_eq!(mat3(&m, 0, 0), 1.0);
        assert_eq!(mat3(&m, 2, 2), 9.0);
    } else {
        panic!("Expected Matrix3");
    }

    // Insufficient length should fail
    let short_vec = Data::RealVec(RealVec(vec![1.0, 2.0, 3.0]));
    assert!(short_vec.try_convert(DataType::Matrix3).is_err());

    Ok(())
}

#[cfg(feature = "vec_variants")]
#[test]
fn test_vec_to_vec_conversions() -> Result<()> {
    use token_value_map::BooleanVec;

    // Test RealVec to IntegerVec conversion
    let real_vec = Data::RealVec(RealVec(vec![1.2, 2.7, -3.5, 0.0, 5.9]));
    let int_vec = real_vec.try_convert(DataType::IntegerVec)?;
    if let Data::IntegerVec(IntegerVec(v)) = int_vec {
        assert_eq!(v, vec![1, 3, -4, 0, 6]); // Rounded values
    } else {
        panic!("Expected IntegerVec");
    }

    // Test IntegerVec to RealVec conversion
    let int_vec = Data::IntegerVec(IntegerVec(vec![1, 2, -3, 0, 5]));
    let real_vec = int_vec.try_convert(DataType::RealVec)?;
    if let Data::RealVec(RealVec(v)) = real_vec {
        assert_eq!(v, vec![1.0, 2.0, -3.0, 0.0, 5.0]);
    } else {
        panic!("Expected RealVec");
    }

    // Test BooleanVec to IntegerVec conversion
    let bool_vec = Data::BooleanVec(BooleanVec(vec![true, false, true, false]));
    let int_vec = bool_vec.try_convert(DataType::IntegerVec)?;
    if let Data::IntegerVec(IntegerVec(v)) = int_vec {
        assert_eq!(v, vec![1, 0, 1, 0]);
    } else {
        panic!("Expected IntegerVec");
    }

    // Test IntegerVec to BooleanVec conversion
    let int_vec = Data::IntegerVec(IntegerVec(vec![0, 1, -5, 0, 100]));
    let bool_vec = int_vec.try_convert(DataType::BooleanVec)?;
    if let Data::BooleanVec(BooleanVec(v)) = bool_vec {
        assert_eq!(v, vec![false, true, true, false, true]);
    } else {
        panic!("Expected BooleanVec");
    }

    // Test BooleanVec to RealVec conversion
    let bool_vec = Data::BooleanVec(BooleanVec(vec![true, false, true]));
    let real_vec = bool_vec.try_convert(DataType::RealVec)?;
    if let Data::RealVec(RealVec(v)) = real_vec {
        assert_eq!(v, vec![1.0, 0.0, 1.0]);
    } else {
        panic!("Expected RealVec");
    }

    // Test RealVec to BooleanVec conversion
    let real_vec = Data::RealVec(RealVec(vec![0.0, 1.5, -0.5, 0.0, 0.001]));
    let bool_vec = real_vec.try_convert(DataType::BooleanVec)?;
    if let Data::BooleanVec(BooleanVec(v)) = bool_vec {
        assert_eq!(v, vec![false, true, true, false, true]);
    } else {
        panic!("Expected BooleanVec");
    }

    Ok(())
}

#[cfg(feature = "matrix4")]
#[test]
fn vec_to_matrix4_conversions() -> Result<()> {
    use token_value_map::Matrix4;
    // RealVec to Matrix4
    let real_vec = Data::RealVec(RealVec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ]));
    let matrix4 = real_vec.try_convert(DataType::Matrix4)?;
    if let Data::Matrix4(Matrix4(m)) = matrix4 {
        use token_value_map::math::mat4;
        assert_eq!(mat4(&m, 0, 0), 1.0);
        assert_eq!(mat4(&m, 0, 1), 2.0);
        assert_eq!(mat4(&m, 3, 3), 16.0);
    } else {
        panic!("Expected Matrix4");
    }

    // IntegerVec to Matrix4
    let int_vec = Data::IntegerVec(IntegerVec((1..=16).collect()));
    let matrix4 = int_vec.try_convert(DataType::Matrix4)?;
    if let Data::Matrix4(Matrix4(m)) = matrix4 {
        use token_value_map::math::mat4;
        assert_eq!(mat4(&m, 0, 0), 1.0);
        assert_eq!(mat4(&m, 3, 3), 16.0);
    } else {
        panic!("Expected Matrix4");
    }

    Ok(())
}

#[test]
fn vec_to_vector_conversions() -> Result<()> {
    // RealVec to Vector2
    #[cfg(feature = "vector2")]
    {
        let real_vec = Data::RealVec(RealVec(vec![3.0, 4.0, 5.0])); // Extra element ignored
        let vec2 = real_vec.try_convert(DataType::Vector2)?;
        if let Data::Vector2(Vector2(v)) = vec2 {
            assert_eq!(v.x, 3.0);
            assert_eq!(v.y, 4.0);
        } else {
            panic!("Expected Vector2");
        }
    }

    // RealVec to Vector3
    #[cfg(feature = "vector3")]
    {
        let real_vec = Data::RealVec(RealVec(vec![1.0, 2.0, 3.0]));
        let vec3 = real_vec.try_convert(DataType::Vector3)?;
        if let Data::Vector3(Vector3(v)) = vec3 {
            assert_eq!(v.x, 1.0);
            assert_eq!(v.y, 2.0);
            assert_eq!(v.z, 3.0);
        } else {
            panic!("Expected Vector3");
        }

        // IntegerVec to Vector3
        let int_vec = Data::IntegerVec(IntegerVec(vec![10, 20, 30]));
        let vec3 = int_vec.try_convert(DataType::Vector3)?;
        if let Data::Vector3(Vector3(v)) = vec3 {
            assert_eq!(v.x, 10.0);
            assert_eq!(v.y, 20.0);
            assert_eq!(v.z, 30.0);
        } else {
            panic!("Expected Vector3");
        }
    }

    Ok(())
}

#[test]
fn vec_to_color_conversions() -> Result<()> {
    // RealVec to Color (RGB)
    let real_vec = Data::RealVec(RealVec(vec![0.5, 0.6, 0.7]));
    let color = real_vec.try_convert(DataType::Color)?;
    if let Data::Color(Color(c)) = color {
        assert_eq!(c[0], 0.5);
        assert_eq!(c[1], 0.6);
        assert_eq!(c[2], 0.7);
        assert_eq!(c[3], 1.0); // Default alpha
    } else {
        panic!("Expected Color");
    }

    // RealVec to Color (RGBA)
    let real_vec = Data::RealVec(RealVec(vec![0.1, 0.2, 0.3, 0.4]));
    let color = real_vec.try_convert(DataType::Color)?;
    if let Data::Color(Color(c)) = color {
        assert_eq!(c[0], 0.1);
        assert_eq!(c[1], 0.2);
        assert_eq!(c[2], 0.3);
        assert_eq!(c[3], 0.4);
    } else {
        panic!("Expected Color");
    }

    // IntegerVec to Color
    let int_vec = Data::IntegerVec(IntegerVec(vec![255, 128, 64, 32]));
    let color = int_vec.try_convert(DataType::Color)?;
    if let Data::Color(Color(c)) = color {
        assert_eq!(c[0], 255.0);
        assert_eq!(c[1], 128.0);
        assert_eq!(c[2], 64.0);
        assert_eq!(c[3], 32.0);
    } else {
        panic!("Expected Color");
    }

    Ok(())
}

#[test]
fn matrix_to_vec_conversions() -> Result<()> {
    // Matrix3 to RealVec
    #[cfg(feature = "matrix3")]
    {
        let matrix = Data::Matrix3(Matrix3(token_value_map::math::mat3_from_row_slice(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        ])));
        let real_vec = matrix.try_convert(DataType::RealVec)?;
        if let Data::RealVec(RealVec(v)) = real_vec {
            assert_eq!(v.len(), 9);
            assert_eq!(v[0], 1.0);
            assert_eq!(v[8], 9.0);
        } else {
            panic!("Expected RealVec");
        }

        // Matrix3 to IntegerVec
        let int_vec = matrix.try_convert(DataType::IntegerVec)?;
        if let Data::IntegerVec(IntegerVec(v)) = int_vec {
            assert_eq!(v.len(), 9);
            assert_eq!(v[0], 1);
            assert_eq!(v[8], 9);
        } else {
            panic!("Expected IntegerVec");
        }
    }

    // Matrix4 to RealVec
    #[cfg(feature = "matrix4")]
    {
        let matrix = Data::Matrix4(Matrix4(token_value_map::math::mat4_identity()));
        let real_vec = matrix.try_convert(DataType::RealVec)?;
        if let Data::RealVec(RealVec(v)) = real_vec {
            assert_eq!(v.len(), 16);
            assert_eq!(v[0], 1.0); // First diagonal
            assert_eq!(v[5], 1.0); // Second diagonal
        } else {
            panic!("Expected RealVec");
        }
    }

    Ok(())
}

#[test]
fn vector_to_vec_conversions() -> Result<()> {
    // Vector3 to RealVec
    #[cfg(feature = "vector3")]
    {
        let vec3 = Data::Vector3(Vector3(token_value_map::math::Vec3Impl::new(1.5, 2.5, 3.5)));
        let real_vec = vec3.try_convert(DataType::RealVec)?;
        if let Data::RealVec(RealVec(v)) = real_vec {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0], 1.5);
            assert_eq!(v[1], 2.5);
            assert_eq!(v[2], 3.5);
        } else {
            panic!("Expected RealVec");
        }

        // Vector3 to IntegerVec
        let int_vec = vec3.try_convert(DataType::IntegerVec)?;
        if let Data::IntegerVec(IntegerVec(v)) = int_vec {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0], 1);
            assert_eq!(v[1], 2);
            assert_eq!(v[2], 3);
        } else {
            panic!("Expected IntegerVec");
        }
    }

    // Vector2 to RealVec
    #[cfg(feature = "vector2")]
    {
        let vec2 = Data::Vector2(Vector2(token_value_map::math::Vec2Impl::new(10.0, 20.0)));
        let real_vec = vec2.try_convert(DataType::RealVec)?;
        if let Data::RealVec(RealVec(v)) = real_vec {
            assert_eq!(v.len(), 2);
            assert_eq!(v[0], 10.0);
            assert_eq!(v[1], 20.0);
        } else {
            panic!("Expected RealVec");
        }
    }

    Ok(())
}

#[test]
#[cfg(all(feature = "vector3", feature = "vec_variants"))]
fn vec_to_vec_variants_conversions() -> Result<()> {
    // RealVec to Vector3Vec
    let real_vec = Data::RealVec(RealVec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let vec3_vec = real_vec.try_convert(DataType::Vector3Vec)?;
    if let Data::Vector3Vec(Vector3Vec(vecs)) = vec3_vec {
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].x, 1.0);
        assert_eq!(vecs[0].y, 2.0);
        assert_eq!(vecs[0].z, 3.0);
        assert_eq!(vecs[1].x, 4.0);
        assert_eq!(vecs[1].y, 5.0);
        assert_eq!(vecs[1].z, 6.0);
    } else {
        panic!("Expected Vector3Vec");
    }

    // ColorVec to Vector3Vec
    let color_vec = Data::ColorVec(ColorVec(vec![[0.1, 0.2, 0.3, 1.0], [0.4, 0.5, 0.6, 1.0]]));
    let vec3_vec = color_vec.try_convert(DataType::Vector3Vec)?;
    if let Data::Vector3Vec(Vector3Vec(vecs)) = vec3_vec {
        assert_eq!(vecs.len(), 2);
        assert_eq!(vecs[0].x, 0.1);
        assert_eq!(vecs[0].y, 0.2);
        assert_eq!(vecs[0].z, 0.3);
        assert_eq!(vecs[1].x, 0.4);
        assert_eq!(vecs[1].y, 0.5);
        assert_eq!(vecs[1].z, 0.6);
    } else {
        panic!("Expected Vector3Vec");
    }

    Ok(())
}

#[test]
#[cfg(all(feature = "matrix3", feature = "vec_variants"))]
fn vec_to_matrix_vec_conversions() -> Result<()> {
    // RealVec to Matrix3Vec
    let real_vec = Data::RealVec(RealVec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0,
    ]));
    let mat3_vec = real_vec.try_convert(DataType::Matrix3Vec)?;
    if let Data::Matrix3Vec(Matrix3Vec(mats)) = mat3_vec {
        assert_eq!(mats.len(), 2);
        assert_eq!(token_value_map::math::mat3(&mats[0], 0, 0), 1.0);
        assert_eq!(token_value_map::math::mat3(&mats[0], 2, 2), 9.0);
        assert_eq!(token_value_map::math::mat3(&mats[1], 0, 0), 10.0);
        assert_eq!(token_value_map::math::mat3(&mats[1], 2, 2), 18.0);
    } else {
        panic!("Expected Matrix3Vec");
    }

    Ok(())
}

#[test]
#[cfg(all(feature = "vector3", feature = "matrix3"))]
fn vector3_vec_to_matrix3() -> Result<()> {
    // Vector3Vec to Matrix3 (using 3 vectors as columns)
    #[cfg(feature = "vec_variants")]
    {
        let vec3_vec = Data::Vector3Vec(Vector3Vec(vec![
            token_value_map::math::Vec3Impl::new(1.0, 2.0, 3.0),
            token_value_map::math::Vec3Impl::new(4.0, 5.0, 6.0),
            token_value_map::math::Vec3Impl::new(7.0, 8.0, 9.0),
        ]));
        let matrix3 = vec3_vec.try_convert(DataType::Matrix3)?;
        if let Data::Matrix3(Matrix3(m)) = matrix3 {
            // Vectors are used as columns
            assert_eq!(token_value_map::math::mat3(&m, 0, 0), 1.0);
            assert_eq!(token_value_map::math::mat3(&m, 1, 0), 2.0);
            assert_eq!(token_value_map::math::mat3(&m, 2, 0), 3.0);
            assert_eq!(token_value_map::math::mat3(&m, 0, 1), 4.0);
            assert_eq!(token_value_map::math::mat3(&m, 1, 1), 5.0);
            assert_eq!(token_value_map::math::mat3(&m, 2, 1), 6.0);
        } else {
            panic!("Expected Matrix3");
        }
    }

    Ok(())
}

#[test]
fn color_vec_to_matrix() -> Result<()> {
    // ColorVec to Matrix3 (using RGB components of 3 colors as rows)
    #[cfg(feature = "matrix3")]
    {
        let color_vec = Data::ColorVec(ColorVec(vec![
            [1.0, 2.0, 3.0, 1.0],
            [4.0, 5.0, 6.0, 1.0],
            [7.0, 8.0, 9.0, 1.0],
        ]));
        let matrix3 = color_vec.try_convert(DataType::Matrix3)?;
        if let Data::Matrix3(Matrix3(m)) = matrix3 {
            use token_value_map::math::mat3;
            // Colors are used as rows
            assert_eq!(mat3(&m, 0, 0), 1.0);
            assert_eq!(mat3(&m, 0, 1), 2.0);
            assert_eq!(mat3(&m, 0, 2), 3.0);
            assert_eq!(mat3(&m, 1, 0), 4.0);
            assert_eq!(mat3(&m, 2, 2), 9.0);
        } else {
            panic!("Expected Matrix3");
        }
    }

    // ColorVec to Matrix4 (using RGBA components of 4 colors as rows)
    #[cfg(feature = "matrix4")]
    {
        let color_vec = Data::ColorVec(ColorVec(vec![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]));
        let matrix4 = color_vec.try_convert(DataType::Matrix4)?;
        if let Data::Matrix4(Matrix4(m)) = matrix4 {
            use token_value_map::math::mat4;
            assert_eq!(mat4(&m, 0, 0), 1.0);
            assert_eq!(mat4(&m, 0, 3), 4.0);
            assert_eq!(mat4(&m, 3, 3), 16.0);
        } else {
            panic!("Expected Matrix4");
        }
    }

    Ok(())
}
