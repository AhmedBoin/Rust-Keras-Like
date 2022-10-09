use crate::prelude::*;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Sequential<T: LayerTrait> {
    pub layers: Vec<T>,
    pub optimizer: Optimizer,
    pub loss: Loss,
}


impl Sequential<Dense> {
    pub fn new(layers: &[Dense]) -> Self {
        Self {
            layers: layers.try_into().unwrap(),
            optimizer: Optimizer::None,
            loss: Loss::None,
        }
    }

    pub fn summary(&self) {
        let mut total_param = 0;
        let mut res = "\nModel Sequential\n".to_string();
        res.push_str("-------------------------------------------------------------\n");
        res.push_str("Layer (Type)\t\t Output shape\t\t No.of params\n");
        for layer in self.layers.iter() {
            let a = layer.w.len();
            let b = layer.b.len();
            total_param += a + b;
            res.push_str(&format!("{}\t\t\t  (None, {})\t\t  {}\n", layer.typ(), b, a + b));
        }
        res.push_str("-------------------------------------------------------------\n");
        res.push_str(&format!("Total params: {}\n", total_param));
        println!("{}", res);
    }

    pub fn compile(&mut self, optimizer: Optimizer, loss: Loss) {
        self.optimizer = optimizer;
        self.loss = loss;
    }

    pub fn fit(&mut self, x: Array2<f64>, y: Array2<f64>, epochs: usize, verbose: bool) {
        for i in 0..epochs {
            // cache (required for back propagation)
            let mut z_cache = vec![];
            let mut a_cache = vec![];
            let mut z: Array2<f64>;
            let mut a = x.clone();
            a_cache.push(a.clone());

            // forward propagate and cache the results
            for layer in self.layers.iter() {
                (z, a) = layer.forward(a.clone());
                z_cache.push(z.clone());
                a_cache.push(a.clone());
            }
            
            // cost computation
            let y_hat = a_cache.pop().unwrap();
            let (loss, mut da) = criteria(y_hat, y.clone(), self.loss.clone());
            if verbose {
                println!("Epoch: {}/{} cost computation: {:?}", i, epochs, loss);
            }

            // back propagation
            let mut dw_cache = vec![];
            let mut db_cache = vec![];
            let mut dw: Array2<f64>;
            let mut db: Array2<f64>;

            // loss = da
            for ((layer, z), a) in (self.layers.iter()).rev().zip((z_cache.clone().iter()).rev()).zip((a_cache.clone().iter()).rev()) {
                (dw, db, da) = layer.backward(z.clone(), a.clone(), da);
                dw_cache.insert(0, dw);
                db_cache.insert(0, db);
            }

            for ((layer, dw), db) in (self.layers.iter_mut()).zip(dw_cache.clone().iter()).zip(db_cache.clone().iter()) {
                layer.optimize(dw.clone(), db.clone(), self.optimizer.clone());
            }
        }
    }
    
    pub fn evaluate(&self, x: Array2<f64>, y: Array2<f64>) -> f64 {
            let (loss, _) = criteria(self.predict(x), y, self.loss.clone());
            loss
    }

    pub fn predict(&self, mut x: Array2<f64>) -> Array2<f64> {
        for layer in self.layers.iter() {
            (_, x) = layer.forward(x);
        }
        x
    }

    pub fn save(&self, path: &str) {
        let encoded: Vec<u8> = bincode::serialize(&self.layers).unwrap();
        let mut file = File::create(path).unwrap();
        file.write(&encoded).unwrap();
    }

    pub fn load(&self, path: &str) -> Sequential<Dense>{
        let mut file = File::open(path).unwrap();
        let mut decoded = Vec::new();
        file.read_to_end(&mut decoded).unwrap();
        let model: Sequential<_> = bincode::deserialize(&decoded[..]).unwrap();
        println!("model: {:?}", model);
        model
    }

}

