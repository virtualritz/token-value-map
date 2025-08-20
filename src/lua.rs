use super::*;
use log::error;
use mlua::{Value as LuaData, prelude::*};
use smallvec::SmallVec;

pub fn from_lua(data: LuaData) -> Result<Data, LuaError> {
    match data {
        LuaData::Nil => Err(LuaError::FromLuaConversionError {
            from: "nil",
            to: "Data".into(),
            message: Some("cannot convert nil to Data".into()),
        }),
        LuaData::Boolean(bool) => Ok(Data::Boolean(Boolean(bool))),
        LuaData::Integer(long) => {
            // FIXME: what if this is too large?
            Ok(Data::Integer(Integer(long)))
        }
        LuaData::Number(double) => Ok(Data::Real(Real(double))),
        LuaData::String(string) => Ok(Data::String(String(string.to_string_lossy()))),
        LuaData::Table(table) => {
            let table = table
                // This will recursively call `from_lua()` on each
                // value.
                .sequence_values::<Data>()
                .collect::<Result<SmallVec<[_; 10]>, LuaError>>();

            if let Ok(table) = table {
                let data_type: DataType = table[0].data_type();
                match data_type {
                    DataType::Boolean => Ok(Data::BooleanVec(BooleanVec(
                        table
                            .into_iter()
                            .filter_map(|v| {
                                v.try_convert(DataType::Boolean)
                                    .map_err(|e| {
                                        error!("{e}");
                                        e
                                    })
                                    .ok()
                            })
                            .filter_map(|v| v.to_bool().ok())
                            .collect::<Vec<_>>(),
                    ))),
                    DataType::Integer => Ok(Data::IntegerVec(IntegerVec(
                        table
                            .into_iter()
                            .filter_map(|v| {
                                v.try_convert(DataType::Integer)
                                    .map_err(|e| {
                                        error!("{e}");
                                        e
                                    })
                                    .ok()
                            })
                            .filter_map(|v| v.to_i64().ok())
                            .collect::<Vec<_>>(),
                    ))),
                    DataType::Real => Ok(Data::RealVec(RealVec(
                        table
                            .into_iter()
                            .filter_map(|v| {
                                v.try_convert(DataType::Real)
                                    .map_err(|e| {
                                        error!("{e}");
                                        e
                                    })
                                    .ok()
                            })
                            .filter_map(|v| v.to_f64().ok())
                            .collect::<Vec<_>>(),
                    ))),
                    DataType::String => Ok(Data::StringVec(StringVec(
                        table
                            .into_iter()
                            .filter_map(|v| match v {
                                Data::String(s) => Some(s.0),
                                _ => None,
                            })
                            .collect(),
                    ))),
                    _ => Err(LuaError::FromLuaConversionError {
                        from: "table",
                        to: "Data".into(),
                        message: Some("cannot convert table to Data".into()),
                    }),
                }
            } else {
                Err(LuaError::FromLuaConversionError {
                    from: "table",
                    to: "Data".into(),
                    message: Some("cannot convert table to Data".into()),
                })
            }
        }
        t => Err(LuaError::FromLuaConversionError {
            from: "unsupported",
            to: "Data".into(),
            message: Some(format!("cannot convert unsupported type {t:?} to Data")),
        }),
    }
}

pub fn from_lua_err(data: LuaData) -> Result<Data, LuaError> {
    match data {
        LuaData::Nil => Err(LuaError::FromLuaConversionError {
            from: "nil",
            to: "Data".into(),
            message: Some("cannot convert nil to Data".into()),
        }),
        LuaData::Boolean(bool) => Ok(Data::Boolean(Boolean(bool))),
        LuaData::Integer(long) => {
            // FIXME: what if this is too large?
            Ok(Data::Integer(Integer(long)))
        }
        LuaData::Number(double) => Ok(Data::Real(Real(double))),
        LuaData::String(string) => Ok(Data::String(String(string.to_string_lossy()))),
        LuaData::Table(table) => {
            let table = table
                // This will recursively call `from_lua_err()` on each
                // value.
                .sequence_values::<Data>()
                .collect::<Result<SmallVec<[_; 10]>, LuaError>>();

            if let Ok(table) = table {
                let data_type: DataType = table[0].data_type();
                match data_type {
                    DataType::Boolean => {
                        let result: Result<Vec<bool>, LuaError> = table
                            .into_iter()
                            .map(|v| {
                                v.try_convert(DataType::Boolean)
                                    .map_err(|e| LuaError::FromLuaConversionError {
                                        from: "table element",
                                        to: "Boolean".into(),
                                        message: Some(format!("conversion failed: {e}")),
                                    })
                                    .and_then(|converted| {
                                        converted.to_bool().map_err(|e| {
                                            LuaError::FromLuaConversionError {
                                                from: "converted value",
                                                to: "bool".into(),
                                                message: Some(format!("accessor failed: {e}")),
                                            }
                                        })
                                    })
                            })
                            .collect();
                        result.map(|vec| Data::BooleanVec(BooleanVec(vec)))
                    }
                    DataType::Integer => {
                        let result: Result<Vec<i64>, LuaError> = table
                            .into_iter()
                            .map(|v| {
                                v.try_convert(DataType::Integer)
                                    .map_err(|e| LuaError::FromLuaConversionError {
                                        from: "table element",
                                        to: "Integer".into(),
                                        message: Some(format!("conversion failed: {e}")),
                                    })
                                    .and_then(|converted| {
                                        converted.to_i64().map_err(|e| {
                                            LuaError::FromLuaConversionError {
                                                from: "converted value",
                                                to: "i64".into(),
                                                message: Some(format!("accessor failed: {e}")),
                                            }
                                        })
                                    })
                            })
                            .collect();
                        result.map(|vec| Data::IntegerVec(IntegerVec(vec)))
                    }
                    DataType::Real => {
                        let result: Result<Vec<f64>, LuaError> = table
                            .into_iter()
                            .map(|v| {
                                v.try_convert(DataType::Real)
                                    .map_err(|e| LuaError::FromLuaConversionError {
                                        from: "table element",
                                        to: "Real".into(),
                                        message: Some(format!("conversion failed: {e}")),
                                    })
                                    .and_then(|converted| {
                                        converted.to_f64().map_err(|e| {
                                            LuaError::FromLuaConversionError {
                                                from: "converted value",
                                                to: "f64".into(),
                                                message: Some(format!("accessor failed: {e}")),
                                            }
                                        })
                                    })
                            })
                            .collect();
                        result.map(|vec| Data::RealVec(RealVec(vec)))
                    }
                    DataType::String => {
                        let result: Result<Vec<std::string::String>, LuaError> = table
                            .into_iter()
                            .map(|v| match v {
                                Data::String(s) => Ok(s.0),
                                _ => Err(LuaError::FromLuaConversionError {
                                    from: "table element",
                                    to: "String".into(),
                                    message: Some("expected String data type".into()),
                                }),
                            })
                            .collect();
                        result.map(|vec| Data::StringVec(StringVec(vec)))
                    }
                    _ => Err(LuaError::FromLuaConversionError {
                        from: "table",
                        to: "Data".into(),
                        message: Some("cannot convert table to Data".into()),
                    }),
                }
            } else {
                Err(LuaError::FromLuaConversionError {
                    from: "table",
                    to: "Data".into(),
                    message: Some("cannot convert table to Data".into()),
                })
            }
        }
        t => Err(LuaError::FromLuaConversionError {
            from: "unsupported",
            to: "Data".into(),
            message: Some(format!("cannot convert unsupported type {t:?} to Data")),
        }),
    }
}

impl FromLua for Data {
    fn from_lua(value: LuaData, _lua: &Lua) -> Result<Self, LuaError> {
        from_lua(value)
    }
}
