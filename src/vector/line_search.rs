use ndarray::{ArrayView1, Array1, Zip, FoldWhile};
// use vector::{Bound, Target, Gradient, unbounded_step, approximate_gradient_neg};
use scalar::GoldenRatioBuilder;
use utils::WrappedFunction;

#[derive(Builder)]
pub struct LineSearch {
    /// Maximum number of iterations allowed before the algorithm terminates
    #[builder(default = "None",setter(into))]
    pub maxiter: Option<usize>,

    /// Absolute error in function parameters between iterations that is acceptable for convergence.
    #[builder(default = "1e-8f64")]
    pub xtol: f64,

    /// Absolute error in function values between iterations that is acceptable for convergence.
    #[builder(default = "1e-8f64")]
    pub ftol: f64,
}

// TODO minimize the number of function calls.
impl LineSearch {

    pub fn minimize<F, G>(&self, f: F, mut x: Array1<f64>, dir: G) -> Array1<f64> 
    where F: Fn(ArrayView1<f64>) -> f64,
          G: Fn(ArrayView1<f64>) -> Array1<f64> {
        let maxiter = self.initialize_parameters(x.len());
        let mut fun = WrappedFunction::new(f);
        let mut f0 = fun.call(x.view());
        for _ in 0..maxiter {

            let direction = dir(x.view());
            let step = self.iterate(&mut fun, f0, &x, &direction);
            x += &(step*&direction);
            let f1 = fun.call(x.view());

            if (f1 - f0).abs() < self.ftol || step * direction.mapv(f64::abs).scalar_sum() < self.xtol {break;}
            f0 = f1;
        }
        x
    }

    #[inline]
    fn initialize_parameters(&self, n: usize) -> usize {
        match self.maxiter {
            Some(x) => x,
            None => n * 200
        }
    }

    #[inline]
    fn iterate<F>(&self, f: &mut WrappedFunction<F>, mut fprev: f64, x0: &Array1<f64>, dir: &Array1<f64>) -> f64 
    where F: Fn(ArrayView1<f64>) -> f64 {
        let mut stepsize = self.positive_step(1.0, x0, dir);
        let mut fun = |x| f.call((x0 + &(x * dir)).view());
        let mut fstepped = fun(stepsize);

        // find a bracketing interval; double the interval until f stops improving
        while fstepped < fprev {
            stepsize = self.positive_step(2.0 * stepsize, x0, dir);
            fprev = fstepped;
            fstepped = fun(stepsize);
        } 

        let gr = GoldenRatioBuilder::default()
            .xtol(self.xtol)
            .build().unwrap();
        gr.minimize(&mut fun, 0.0, stepsize)
    }

    #[inline]
    fn positive_step(&self, step: f64, start: &Array1<f64>, direction: &Array1<f64>) -> f64 {
        let step = Zip::from(start).and(direction)
            .fold_while(step, |acc, s,d| if d<&0. && acc > s/d.abs() {FoldWhile::Continue(s/d.abs())} else {FoldWhile::Continue(acc)})
            .into_inner();
        step
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::prelude::*;
    use ::utils::approx_fprime;

    #[test]
    fn test_unbounded() {
        let n = 5;
        let ls = LineSearchBuilder::default().build().unwrap();
        let f = |x: ArrayView1<f64>| (&x - 0.5).mapv(|xi| xi*xi).scalar_sum();
        let eps = Array1::ones(n)*1e-8;
        let g = |x: ArrayView1<f64>| -approx_fprime(x, f, eps.view()); // do gradient descent
        let xout = ls.minimize(&f, Array::ones(n), g);

        println!("{:?}", xout);
        println!("{:?}", g(xout.view()));
        assert!(xout.all_close(&(Array::ones(n)/2.0 as f64), 1e-5));
    }
    // #[test]
    // fn test_bounded_on_the_simplex() {
    //     let n = 10;
    //     let dir = |f: &Target, x: ArrayView1<f64>| {
    //         let mut g = approximate_gradient_neg(f, x);
    //         // project the gradient on the simplex:
    //         let mask = x.iter().zip(&g).map(|(xi, gi)| if *xi > 0.0 || *gi >= 0.0 {1.0} else {0.0}).collect::<Array1<f64>>();
    //         g *= &mask; // set the directions at borders pointing outside the border to 0
    //         g -= &(g.scalar_sum() * &mask / mask.scalar_sum()); // projection onto the simplex
    //         g
    //     };
    //     let bound = |max_step: f64, x: ArrayView1<f64>, dir: ArrayView1<f64>| {
    //         dir.iter().zip(&x)
    //         .map(|(d,f)| if *d<0.0 {f/d.abs()} else {max_step})
    //         .fold(max_step, |acc, x| if acc < x {acc} else {x})
    //     };
    //     let f = |x: ArrayView1<f64>| {
    //         x.mapv(|xi| xi*xi).scalar_sum()
    //     };
    //     let mut x0 = Array1::zeros(n);
    //     x0[0] = 1.0;

    //     let ls = LineSearchBuilder::default().build().unwrap();
    //     ls.direction = &dir;
    //     ls.bounded_step = &bound;
    //     let (xout, status) = ls.minimize(&f, x0.view());
    //     println!("{:?}", status);
    //     println!("{:?}", xout);
    // }
}