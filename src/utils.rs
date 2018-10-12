use ndarray::prelude::*;

pub struct WrappedFunction<F: Fn(ArrayView1<f64>) -> f64> {
    pub num: usize,
    pub func: F,
}

impl<F: Fn(ArrayView1<f64>) -> f64> WrappedFunction<F> {
    pub fn new(f: F) -> WrappedFunction<F> {
        WrappedFunction{
            num: 0,
            func: f
        }
    }

    pub fn call(&mut self, arg: ArrayView1<f64>) -> f64 {
        self.num += 1;
        (self.func)(arg)
    }
}

/// Performs finite-difference approximation of the gradient of a scalar function.
pub fn approx_fprime<F>(x: ArrayView1<f64>, f: F, epsilon: ArrayView1<f64>) -> Array1<f64> 
where F: Fn(ArrayView1<f64>) -> f64 {
    let f0 = f(x);
    let mut dir = x.to_owned();
    let mut grad = x.to_owned();
    for i in 0..x.len() {
        dir[i] += epsilon[i];
        grad[i] = ( f(dir.view()) - f0 ) / epsilon[i];
        dir[i] -= epsilon[i];
    }
    grad
}


#[cfg(test)]
mod tests {

    use super::*;
    use float_cmp::ApproxEq;

    #[test]
    fn gradient() {
        let function = |x: ArrayView1<f64>| 1.0 * x[0].powi(2) + 200. * x[1].powi(2);
        let x = Array::from_vec(vec![1.0, 1.0]);
        let eps_ar = Array::from_vec(vec![1e-7, 14.14 * 1e-7]);
        let res = approx_fprime(x.view(), function, eps_ar.view());

        println!("Res: {}", res);
        assert!(res[0].approx_eq(&2.0, 1e-4, 10));
        assert!(res[1].approx_eq(&400.0, 1e-3, 10));
    }

}
