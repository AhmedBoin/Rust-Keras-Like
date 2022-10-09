pub use serde::{Serialize, Deserialize};
pub use std::fs::File;
pub use std::io::Write;
pub use std::io::Read;

pub use std::f64::consts::E as e;

pub use ndarray::*;
pub use ndarray::prelude::*;
pub use ndarray_rand::RandomExt;
pub use ndarray_rand::rand_distr::Uniform;

pub use crate::layers::*;
pub use crate::models::*;
pub use crate::optimizers::*;
pub use crate::losses::*;
pub use crate::utils::*;
pub use crate::activations::*;


pub use crate::rand_array;
pub use crate::Model;
pub use crate::Dense;