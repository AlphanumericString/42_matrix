//! # `matrix` — an introduction to linear algebra
//!
//! A from-scratch linear algebra library written for the 42 module *"Enter the
//! Matrix"*. It implements every exercise of the subject, from the elementary
//! vector and matrix operations up to the determinant, the inverse and the
//! row-echelon form, plus the two bonus exercises (the projection matrix and
//! complex vector spaces).
//!
//! ## Design in one paragraph
//!
//! Everything is generic over a scalar [`Field`], so the exact same vector and
//! matrix code works over the real numbers (`f32`, the required field) **and**
//! over the [`Complex`] numbers. There are no dependencies and no use of the
//! standard library's mathematical functions beyond the ones the subject
//! explicitly allows (`pow`, `max` via comparisons, the fused multiply-add and,
//! for the projection matrix only, `tan`). The two data structures the subject
//! asks for — [`Vector`] and [`Matrix`] — are the heart of the crate.
//!
//! ## Quick tour
//!
//! ```
//! use matrix::prelude::*;
//!
//! // Exercises 00–02: add, subtract, scale, combine, interpolate.
//! let mut u = Vector::from([2., 3.]);
//! u.add(&Vector::from([5., 7.]));
//! assert_eq!(u, Vector::from([7., 10.]));
//!
//! // Exercise 07: matrix multiplication.
//! let a = Matrix::from([[1., 2.], [3., 4.]]);
//! let b = Matrix::from([[5., 6.], [7., 8.]]);
//! assert_eq!(a.mul_mat(&b), Matrix::from([[19., 22.], [43., 50.]]));
//!
//! // Exercise 11: determinant. Exercise 12: inverse.
//! let m = Matrix::from([[4., 7.], [2., 6.]]);
//! assert_eq!(m.determinant(), 10.0);
//! let scaled = Matrix::from([[2., 0.], [0., 2.]]);
//! assert_eq!(scaled.inverse().unwrap(), Matrix::from([[0.5, 0.], [0., 0.5]]));
//! ```

pub mod complex;
pub mod field;
pub mod matrix;
pub mod vector;

pub use complex::Complex;
pub use field::{Field, Interpolate};
pub use matrix::{projection, Matrix, MatrixError};
pub use vector::{angle_cos, cross_product, lerp, linear_combination, Vector};

/// A convenient re-export of everything a user of the library needs.
pub mod prelude {
    pub use crate::complex::Complex;
    pub use crate::field::{Field, Interpolate};
    pub use crate::matrix::{projection, Matrix, MatrixError};
    pub use crate::vector::{angle_cos, cross_product, lerp, linear_combination, Vector};
}
