use crate::prelude::*;


pub fn sigmoid(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| 1. / (1. / e.powf(-z)))
}

pub fn sigmoid_backward(z: Array2<f64>) -> Array2<f64> {
    sigmoid(z.clone()) * (1.0 - sigmoid(z))
}

pub fn relu(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| if z >= 0.0 {z} else {0.0})
}

pub fn relu_backward(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| if z >= 0.0 {1.0} else {0.0})
}

pub fn tanh(z: Array2<f64>) -> Array2<f64> {
    z.mapv(|z| z.tanh())
}

pub fn softmax(z: Array2<f64>) -> Array2<f64> {
    let z = z.mapv(|z| e.powf(z));
    z.clone() / z.mean().unwrap()
}


pub trait LayerTrait {
    fn new(perceptron: usize, prev: usize, activation: Activation) -> Self;
    
    fn typ(&self) -> String;
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Dense {
    pub w: Array2<f64>,
    pub b: Array2<f64>,
    pub activation: Activation,
}

impl LayerTrait for Dense {
    fn new(perceptron: usize, prev: usize, activation: Activation) -> Self {
        Self {
            w: rand_array!(prev, perceptron),
            b: rand_array!(1, perceptron),
            activation,
        }
    }

    fn typ(&self) -> String {
        "Dense".into()
    }
}

impl Dense {
    pub fn forward(&self, a: Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // z = w * a + b
        let z = a.dot(&self.w) + self.b.clone();

        // a = g(z)
        use Activation::*;
        let a = match self.activation {
            Linear => z.clone(),
            Relu => relu(z.clone()),
            Sigmoid => sigmoid(z.clone()),
            Tanh => tanh(z.clone()),
            Softmax => softmax(z.clone()),
        };
        
        // returns
        (z, a)
    }

    pub fn backward(&self, z: Array2<f64>, a: Array2<f64>, da: Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        // dz = da * g(z)
        use Activation::*;
        let dz = match self.activation {
            Linear => da,
            Relu => da * relu_backward(z),
            Sigmoid => da * sigmoid_backward(z),
            Tanh => da * z.mapv(|z| 1.0 - z.tanh().powf(2.0)),
            Softmax => da * softmax(z),
        };
        
        // dw = dz.a , db = dz , da = w.dz
        let dw = (a.reversed_axes().dot(&dz))/(dz.len() as f64);
        let db = dz.clone().sum_axis(Axis(0)).insert_axis(Axis(0))/(dz.len() as f64);
        let da = dz.dot(&self.w.t());

        (dw, db, da)
    }

}

impl Optimization for Dense {
    fn optimize(&mut self, dw: Array2<f64>, db: Array2<f64>, optimizer: Optimizer) {
        use Optimizer::*;
        match optimizer {
            SGD(lr) => {
                self.w = self.w.clone() - lr * dw;
                self.b = self.b.clone() - lr * db;
            },
            Adam { lr, beta1, beta2, epsilon } => {
                unimplemented!("Adam optimizer not implemented yet lr={}, beta1={}, beta2={}, epsilon={}", lr, beta1, beta2, epsilon);
            },
            None => (),
        }
    }
}