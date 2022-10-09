#[allow(unused)]
use crate::prelude::*;

#[macro_export]
macro_rules! rand_array {
    ($($x:expr),*) => {
        {
            Array::random(($($x,)*), Uniform::new(-0.01, 0.01))
        }
    };
}

#[macro_export]
macro_rules! Model {
    (input_shape &i:expr ,$(dense $x:expr, activation $a:expr),*) => {
        {
            let x = vec![$(x)*];
            let a = vec![$(a)*];
            let mut layers = vec![];
            layers.push(dense!($i, x[0], a[0]));
            for i in 0..x.len()-1 {
                layers.push(dense!(x[i], x[i+1], a[i+1]));
            }
            Sequential::new(layers);
        }
    };
}

#[macro_export]
macro_rules! Dense {
    ($x:expr ,$y:expr, $a:expr) => {
        {
            Dense::new(
                $x, $y, $a
            );
        }
    };
}