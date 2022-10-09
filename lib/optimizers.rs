use crate::prelude::*;


#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Optimizer {
    SGD(f64),
    Adam{
        lr: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64
    },
    None
}

pub trait Optimization {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, optimizer: Optimizer);
}