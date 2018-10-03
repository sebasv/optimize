//! This crate provides (non-linear) numerical optimization methods. 
//!
//! The crate is actively developed and expanded to include more methods.
//!

#[macro_use(s)]
extern crate ndarray;

extern crate float_cmp;
// extern crate num_traits;

pub mod scalar;
pub mod vector;

#[derive(PartialEq, Debug)]
pub enum Status {
    MaxIterReached,
    FtolConvergence,
    XtolConvergence,
    NotFinished,
}