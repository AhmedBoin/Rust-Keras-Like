use crate::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Loss {
    MSE,
    NLL,
    None,
}

pub fn criteria(y_hat: Array2<f64>, y: Array2<f64>, loss_ty: Loss) -> (f64, Array2<f64>) {
    
    use Loss::*;
    match loss_ty {

        MSE => {
            let da = y_hat.clone() - y.clone();
            let loss = (0.5 * (y_hat - y).mapv(|a| a.powf(2.0))).mean().unwrap();
            (loss, da)
        },

        NLL => {
            let da = -((y.clone() / y_hat.clone())-((1.0-y.clone())/(1.0-y_hat.clone())));
            let loss = -(y.clone() * y_hat.mapv(|y| y.log(e)).reversed_axes() + (1.0 - y)*(1.0 - y_hat.mapv(|y| y.log(e)).reversed_axes())).mean().unwrap();
            (loss, da)
        },

        None => {
            let da = y_hat.clone() - y.clone();
            let loss = (y_hat - y).mean().unwrap();
            (loss, da)
        },
        
    }
}