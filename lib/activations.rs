#[allow(unused)]
use crate::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Activation {
    Linear,
    Relu,
    Sigmoid,
    Tanh,
    Softmax,
}