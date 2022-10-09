# Rust Keras Like
 Pure Rust implementaion for deep learning library like keras
```
use rkl::prelude::*;

fn main() {
    let x = array![[1., 2.], [3., 4.], [5., 6.]];
    let y = array![[3.], [7.], [11.]];

    let mut model = Sequential::new(&[
        Dense::new(4, 2, Activation::Linear),
        Dense::new(2, 4, Activation::Linear),
        Dense::new(1, 2, Activation::Linear),
    ]);
    
    model.summary();
    
    model.compile(Optimizer::SGD(0.01), Loss::MSE);
    
    model.fit(x, y, 1000, true);
    
    let x_test = array![[2., 3.]];
    let y_test = array![[5.]];
    
    let eval = model.evaluate(x_test.clone(), y_test);
    println!("\ncost: {}\n", eval);
    
    let prediction = model.predict(x_test);
    println!("prediction: {}", prediction);

    model.save("./test.model");
}
```
