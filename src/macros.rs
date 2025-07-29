// Internal macros for reducing code duplication

// Macro to implement DataTypeOps for enums with the same variants as DataType
macro_rules! impl_data_type_ops {
    ($enum_name:ident) => {
        impl DataTypeOps for $enum_name {
            fn data_type(&self) -> DataType {
                match self {
                    $enum_name::Boolean(_) => DataType::Boolean,
                    $enum_name::Integer(_) => DataType::Integer,
                    $enum_name::Real(_) => DataType::Real,
                    $enum_name::String(_) => DataType::String,
                    $enum_name::Color(_) => DataType::Color,
                    #[cfg(feature = "vector2")]
                    $enum_name::Vector2(_) => DataType::Vector2,
                    #[cfg(feature = "vector3")]
                    $enum_name::Vector3(_) => DataType::Vector3,
                    #[cfg(feature = "matrix3")]
                    $enum_name::Matrix3(_) => DataType::Matrix3,
                    #[cfg(feature = "normal3")]
                    $enum_name::Normal3(_) => DataType::Normal3,
                    #[cfg(feature = "point3")]
                    $enum_name::Point3(_) => DataType::Point3,
                    #[cfg(feature = "matrix4")]
                    $enum_name::Matrix4(_) => DataType::Matrix4,
                    $enum_name::BooleanVec(_) => DataType::BooleanVec,
                    $enum_name::IntegerVec(_) => DataType::IntegerVec,
                    $enum_name::RealVec(_) => DataType::RealVec,
                    $enum_name::ColorVec(_) => DataType::ColorVec,
                    $enum_name::StringVec(_) => DataType::StringVec,
                    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
                    $enum_name::Vector2Vec(_) => DataType::Vector2Vec,
                    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
                    $enum_name::Vector3Vec(_) => DataType::Vector3Vec,
                    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
                    $enum_name::Matrix3Vec(_) => DataType::Matrix3Vec,
                    #[cfg(all(feature = "normal3", feature = "vec_variants"))]
                    $enum_name::Normal3Vec(_) => DataType::Normal3Vec,
                    #[cfg(all(feature = "point3", feature = "vec_variants"))]
                    $enum_name::Point3Vec(_) => DataType::Point3Vec,
                    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
                    $enum_name::Matrix4Vec(_) => DataType::Matrix4Vec,
                }
            }

            fn type_name(&self) -> &'static str {
                match self {
                    $enum_name::Boolean(_) => "boolean",
                    $enum_name::Integer(_) => "integer",
                    $enum_name::Real(_) => "real",
                    $enum_name::String(_) => "string",
                    $enum_name::Color(_) => "color",
                    #[cfg(feature = "vector2")]
                    $enum_name::Vector2(_) => "vec2",
                    #[cfg(feature = "vector3")]
                    $enum_name::Vector3(_) => "vec3",
                    #[cfg(feature = "matrix3")]
                    $enum_name::Matrix3(_) => "mat3",
                    #[cfg(feature = "normal3")]
                    $enum_name::Normal3(_) => "normal3",
                    #[cfg(feature = "point3")]
                    $enum_name::Point3(_) => "point3",
                    #[cfg(feature = "matrix4")]
                    $enum_name::Matrix4(_) => "matrix4",
                    $enum_name::BooleanVec(_) => "boolean_vec",
                    $enum_name::IntegerVec(_) => "integer_vec",
                    $enum_name::RealVec(_) => "real_vec",
                    $enum_name::ColorVec(_) => "color_vec",
                    $enum_name::StringVec(_) => "string_vec",
                    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
                    $enum_name::Vector2Vec(_) => "vec2_vec",
                    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
                    $enum_name::Vector3Vec(_) => "vec3_vec",
                    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
                    $enum_name::Matrix3Vec(_) => "mat3_vec",
                    #[cfg(all(feature = "normal3", feature = "vec_variants"))]
                    $enum_name::Normal3Vec(_) => "normal3_vec",
                    #[cfg(all(feature = "point3", feature = "vec_variants"))]
                    $enum_name::Point3Vec(_) => "point3_vec",
                    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
                    $enum_name::Matrix4Vec(_) => "matrix4_vec",
                }
            }
        }
    };
}

// Macro to implement arithmetic operations for Data enum
macro_rules! impl_data_arithmetic {
    (binary $op_trait:ident, $op_method:ident, $op_name:literal) => {
        impl $op_trait for Data {
            type Output = Data;

            fn $op_method(self, other: Data) -> Data {
                match (&self, &other) {
                    (Data::Real(a), Data::Real(b)) => {
                        Data::Real(a.clone().$op_method(b.clone()))
                    }
                    (Data::Integer(a), Data::Integer(b)) => {
                        Data::Integer(a.clone().$op_method(b.clone()))
                    }
                    (Data::Boolean(a), Data::Boolean(b)) => {
                        Data::Boolean(a.clone().$op_method(b.clone()))
                    }
                    (Data::String(a), Data::String(b)) => {
                        Data::String(a.clone().$op_method(b.clone()))
                    }
                    (Data::Color(a), Data::Color(b)) => {
                        Data::Color(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(feature = "vector2")]
                    (Data::Vector2(a), Data::Vector2(b)) => {
                        Data::Vector2(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(feature = "vector3")]
                    (Data::Vector3(a), Data::Vector3(b)) => {
                        Data::Vector3(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(feature = "matrix3")]
                    (Data::Matrix3(a), Data::Matrix3(b)) => {
                        Data::Matrix3(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(feature = "normal3")]
                    (Data::Normal3(a), Data::Normal3(b)) => {
                        Data::Normal3(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(feature = "point3")]
                    (Data::Point3(a), Data::Point3(b)) => {
                        Data::Point3(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(feature = "matrix4")]
                    (Data::Matrix4(a), Data::Matrix4(b)) => {
                        Data::Matrix4(a.clone().$op_method(b.clone()))
                    }
                    (Data::BooleanVec(a), Data::BooleanVec(b)) => {
                        Data::BooleanVec(a.clone().$op_method(b.clone()))
                    }
                    (Data::RealVec(a), Data::RealVec(b)) => {
                        Data::RealVec(a.clone().$op_method(b.clone()))
                    }
                    (Data::IntegerVec(a), Data::IntegerVec(b)) => {
                        Data::IntegerVec(a.clone().$op_method(b.clone()))
                    }
                    (Data::StringVec(a), Data::StringVec(b)) => {
                        Data::StringVec(a.clone().$op_method(b.clone()))
                    }
                    (Data::ColorVec(a), Data::ColorVec(b)) => {
                        Data::ColorVec(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
                    (Data::Vector2Vec(a), Data::Vector2Vec(b)) => {
                        Data::Vector2Vec(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
                    (Data::Vector3Vec(a), Data::Vector3Vec(b)) => {
                        Data::Vector3Vec(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
                    (Data::Matrix3Vec(a), Data::Matrix3Vec(b)) => {
                        Data::Matrix3Vec(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(all(feature = "normal3", feature = "vec_variants"))]
                    (Data::Normal3Vec(a), Data::Normal3Vec(b)) => {
                        Data::Normal3Vec(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(all(feature = "point3", feature = "vec_variants"))]
                    (Data::Point3Vec(a), Data::Point3Vec(b)) => {
                        Data::Point3Vec(a.clone().$op_method(b.clone()))
                    }
                    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
                    (Data::Matrix4Vec(a), Data::Matrix4Vec(b)) => {
                        Data::Matrix4Vec(a.clone().$op_method(b.clone()))
                    }
                    _ => {
                        log::warn!(
                            concat!(
                                "Cannot ",
                                $op_name,
                                " {:?} and {:?}, returning first operand"
                            ),
                            self.data_type(),
                            other.data_type()
                        );
                        self
                    }
                }
            }
        }
    };
    (scalar $scalar_type:ty) => {
        impl Mul<$scalar_type> for Data {
            type Output = Data;

            fn mul(self, scalar: $scalar_type) -> Data {
                match self {
                    Data::Real(a) => Data::Real(a * scalar),
                    Data::Integer(a) => Data::Integer(a * scalar),
                    Data::Boolean(a) => Data::Boolean(a * scalar),
                    Data::String(a) => Data::String(a * scalar),
                    Data::Color(a) => Data::Color(a * scalar),
                    #[cfg(feature = "vector2")]
                    Data::Vector2(a) => Data::Vector2(a * scalar),
                    #[cfg(feature = "vector3")]
                    Data::Vector3(a) => Data::Vector3(a * scalar),
                    #[cfg(feature = "matrix3")]
                    Data::Matrix3(a) => Data::Matrix3(a * scalar),
                    #[cfg(feature = "normal3")]
                    Data::Normal3(a) => Data::Normal3(a * scalar),
                    #[cfg(feature = "point3")]
                    Data::Point3(a) => Data::Point3(a * scalar),
                    #[cfg(feature = "matrix4")]
                    Data::Matrix4(a) => Data::Matrix4(a * scalar),
                    Data::BooleanVec(a) => Data::BooleanVec(a * scalar),
                    Data::RealVec(a) => Data::RealVec(a * scalar),
                    Data::IntegerVec(a) => Data::IntegerVec(a * scalar),
                    Data::StringVec(a) => Data::StringVec(a * scalar),
                    Data::ColorVec(a) => Data::ColorVec(a * scalar),
                    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
                    Data::Vector2Vec(a) => Data::Vector2Vec(a * scalar),
                    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
                    Data::Vector3Vec(a) => Data::Vector3Vec(a * scalar),
                    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
                    Data::Matrix3Vec(a) => Data::Matrix3Vec(a * scalar),
                    #[cfg(all(feature = "normal3", feature = "vec_variants"))]
                    Data::Normal3Vec(a) => Data::Normal3Vec(a * scalar),
                    #[cfg(all(feature = "point3", feature = "vec_variants"))]
                    Data::Point3Vec(a) => Data::Point3Vec(a * scalar),
                    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
                    Data::Matrix4Vec(a) => Data::Matrix4Vec(a * scalar),
                }
            }
        }
    };
    (div $scalar_type:ty) => {
        impl Div<$scalar_type> for Data {
            type Output = Data;

            fn div(self, scalar: $scalar_type) -> Data {
                match self {
                    Data::Real(a) => Data::Real(a / scalar),
                    Data::Integer(a) => Data::Integer(a / scalar),
                    Data::Boolean(a) => Data::Boolean(a / scalar),
                    Data::String(a) => Data::String(a / scalar),
                    Data::Color(a) => Data::Color(a / scalar),
                    #[cfg(feature = "vector2")]
                    Data::Vector2(a) => Data::Vector2(a / scalar),
                    #[cfg(feature = "vector3")]
                    Data::Vector3(a) => Data::Vector3(a / scalar),
                    #[cfg(feature = "matrix3")]
                    Data::Matrix3(a) => Data::Matrix3(a / scalar),
                    #[cfg(feature = "normal3")]
                    Data::Normal3(a) => Data::Normal3(a / scalar),
                    #[cfg(feature = "point3")]
                    Data::Point3(a) => Data::Point3(a / scalar),
                    #[cfg(feature = "matrix4")]
                    Data::Matrix4(a) => Data::Matrix4(a / scalar),
                    Data::BooleanVec(a) => Data::BooleanVec(a / scalar),
                    Data::RealVec(a) => Data::RealVec(a / scalar),
                    Data::IntegerVec(a) => Data::IntegerVec(a / scalar),
                    Data::StringVec(a) => Data::StringVec(a / scalar),
                    Data::ColorVec(a) => Data::ColorVec(a / scalar),
                    #[cfg(all(feature = "vector2", feature = "vec_variants"))]
                    Data::Vector2Vec(a) => Data::Vector2Vec(a / scalar),
                    #[cfg(all(feature = "vector3", feature = "vec_variants"))]
                    Data::Vector3Vec(a) => Data::Vector3Vec(a / scalar),
                    #[cfg(all(feature = "matrix3", feature = "vec_variants"))]
                    Data::Matrix3Vec(a) => Data::Matrix3Vec(a / scalar),
                    #[cfg(all(feature = "normal3", feature = "vec_variants"))]
                    Data::Normal3Vec(a) => Data::Normal3Vec(a / scalar),
                    #[cfg(all(feature = "point3", feature = "vec_variants"))]
                    Data::Point3Vec(a) => Data::Point3Vec(a / scalar),
                    #[cfg(all(feature = "matrix4", feature = "vec_variants"))]
                    Data::Matrix4Vec(a) => Data::Matrix4Vec(a / scalar),
                }
            }
        }
    };
}

// Macro to implement Sample trait for Value enum
macro_rules! impl_sample_for_value {
    ($type:ty, $data_variant:ident) => {
        impl Sample<$type> for Value {
            fn sample(
                &self,
                shutter: &Shutter,
                samples: NonZeroU16,
            ) -> Result<Vec<($type, SampleWeight)>> {
                match self {
                    Value::Uniform(data) => match data {
                        Data::$data_variant(value) => {
                            Ok(vec![(value.clone(), 1.0)])
                        }
                        _ => Err(anyhow!(
                            concat!(
                                "Sample<",
                                stringify!($type),
                                "> called on non-",
                                stringify!($data_variant),
                                " uniform Value: {:?}"
                            ),
                            data.data_type()
                        )),
                    },
                    Value::Animated(animated_data) => {
                        animated_data.sample(shutter, samples)
                    }
                }
            }
        }
    };
}

// Macro to implement insert methods for AnimatedData
macro_rules! impl_animated_data_insert {
    ($($method_name:ident, $type:ty, $variant:ident);+ $(;)?) => {
        impl AnimatedData {
            $(
                pub fn $method_name(
                    &mut self,
                    time: Time,
                    value: $type,
                ) -> Result<()> {
                    match self {
                        AnimatedData::$variant(map) => {
                            map.insert(time, value);
                            Ok(())
                        }
                        _ => Err(anyhow!(concat!("Type mismatch: expected ", stringify!($type)))),
                    }
                }
            )+
        }
    };
}

// Macro to implement TryFrom for Vec types
macro_rules! impl_try_from_vec {
    ($($target_type:ty, $variant:ident, $type_name:literal);+ $(;)?) => {
        $(
            impl TryFrom<Data> for Vec<$target_type> {
                type Error = anyhow::Error;

                fn try_from(value: Data) -> Result<Self, Self::Error> {
                    match value {
                        Data::$variant(v) => Ok(v.0),
                        _ => Err(anyhow!(
                            concat!("Could not convert {} to Vec<", $type_name, ">"),
                            value.type_name()
                        )),
                    }
                }
            }

            impl TryFrom<&Data> for Vec<$target_type> {
                type Error = anyhow::Error;

                fn try_from(value: &Data) -> Result<Self, Self::Error> {
                    match value {
                        Data::$variant(v) => Ok(v.0.clone()),
                        _ => Err(anyhow!(
                            concat!("Could not convert &{} to Vec<", $type_name, ">"),
                            value.type_name()
                        )),
                    }
                }
            }
        )+
    };
}

// Macro to implement Sample trait for AnimatedData
macro_rules! impl_sample_for_animated_data {
    ($($type:ty, $data_variant:ident, $feature:literal);+ $(;)?) => {
        $(
            #[cfg(feature = $feature)]
            impl Sample<$type> for AnimatedData {
                fn sample(
                    &self,
                    shutter: &Shutter,
                    samples: NonZeroU16,
                ) -> Result<Vec<($type, SampleWeight)>> {
                    match self {
                        AnimatedData::$data_variant(map) => map.sample(shutter, samples),
                        _ => Err(anyhow!(
                            concat!(
                                "Sample<",
                                stringify!($type),
                                "> called on non-",
                                stringify!($data_variant),
                                " AnimatedData variant: {:?}"
                            ),
                            self.data_type()
                        )),
                    }
                }
            }
        )+
    };
    ($($type:ty, $data_variant:ident);+ $(;)?) => {
        $(
            impl Sample<$type> for AnimatedData {
                fn sample(
                    &self,
                    shutter: &Shutter,
                    samples: NonZeroU16,
                ) -> Result<Vec<($type, SampleWeight)>> {
                    match self {
                        AnimatedData::$data_variant(map) => map.sample(shutter, samples),
                        _ => Err(anyhow!(
                            concat!(
                                "Sample<",
                                stringify!($type),
                                "> called on non-",
                                stringify!($data_variant),
                                " AnimatedData variant: {:?}"
                            ),
                            self.data_type()
                        )),
                    }
                }
            }
        )+
    };
}

pub(crate) use impl_animated_data_insert;
pub(crate) use impl_data_arithmetic;
pub(crate) use impl_data_type_ops;
pub(crate) use impl_sample_for_animated_data;
pub(crate) use impl_sample_for_value;
pub(crate) use impl_try_from_vec;
