//! # Complex numbers — the bonus field (exercise 15)
//!
//! The bonus of the module asks to redo every exercise with the scalar field
//! `K = C` instead of `K = R`. Because the entire library is written against the
//! [`Field`](crate::field::Field) trait, "redoing the exercises" boils down to
//! writing this one type and its trait implementation — no vector or matrix code
//! changes at all. That is the whole point of the exercise:
//!
//! > This bonus exercise is the reason why we kept most of our declarations with
//! > a generic K [...] This is meant to showcase a powerful aspect of abstract
//! > algebra: genericity.
//!
//! A complex number `z = a + i·b` is a pair of real numbers. We represent it
//! with two `f32`, matching the `f32` precision chosen for the real exercises.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::field::{sqrt, Field};

/// A complex number `re + im·i`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    /// The real part `a`.
    pub re: f32,
    /// The imaginary part `b`.
    pub im: f32,
}

impl Complex {
    /// Build `re + im·i`.
    pub const fn new(re: f32, im: f32) -> Self {
        Complex { re, im }
    }

    /// The purely imaginary unit `i`.
    pub const I: Complex = Complex::new(0.0, 1.0);
}

impl Add for Complex {
    type Output = Complex;

    /// `(a + bi) + (c + di) = (a + c) + (b + d)i`.
    fn add(self, rhs: Complex) -> Complex {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex {
    type Output = Complex;

    /// `(a + bi) - (c + di) = (a - c) + (b - d)i`.
    fn sub(self, rhs: Complex) -> Complex {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex {
    type Output = Complex;

    /// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, the usual rule with
    /// `i² = -1`.
    fn mul(self, rhs: Complex) -> Complex {
        Complex::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex {
    type Output = Complex;

    /// Division by multiplying with the conjugate of the denominator:
    /// `z / w = z * conj(w) / |w|²`, which always clears the imaginary part of
    /// the denominator.
    fn div(self, rhs: Complex) -> Complex {
        let d = rhs.re * rhs.re + rhs.im * rhs.im;
        Complex::new(
            (self.re * rhs.re + self.im * rhs.im) / d,
            (self.im * rhs.re - self.re * rhs.im) / d,
        )
    }
}

impl Neg for Complex {
    type Output = Complex;

    /// `-(a + bi) = -a - bi`.
    fn neg(self) -> Complex {
        Complex::new(-self.re, -self.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Complex) {
        *self = *self + rhs;
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Complex) {
        *self = *self - rhs;
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Complex) {
        *self = *self * rhs;
    }
}

impl DivAssign for Complex {
    fn div_assign(&mut self, rhs: Complex) {
        *self = *self / rhs;
    }
}

impl Field for Complex {
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }

    fn from_f64(x: f64) -> Self {
        Complex::new(x as f32, 0.0)
    }

    fn to_f64(self) -> f64 {
        self.re as f64
    }

    /// The modulus `|a + bi| = sqrt(a² + b²)`, a real number.
    ///
    /// This is exactly the Euclidean norm of the pair `(a, b)`, which is why the
    /// complex inner product (with conjugation) naturally produces a real,
    /// non-negative "length".
    fn abs(self) -> f64 {
        sqrt((self.re * self.re + self.im * self.im) as f64)
    }

    /// Conjugation `conj(a + bi) = a - bi`.
    ///
    /// This is the operation that makes the complex inner product
    /// *sesquilinear*: `⟨u|v⟩ = Σ conj(uᵢ)·vᵢ`. Without it, `⟨u|u⟩` would not be
    /// guaranteed real and non-negative, and the notion of a norm would break.
    fn conj(self) -> Self {
        Complex::new(self.re, -self.im)
    }
}

/// Human-readable form, e.g. `2 + 3i`, `2 - 3i`, `2`, `-i`.
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im == 0.0 {
            write!(f, "{}", self.re)
        } else if self.re == 0.0 {
            write!(f, "{}i", self.im)
        } else if self.im < 0.0 {
            write!(f, "{} - {}i", self.re, -self.im)
        } else {
            write!(f, "{} + {}i", self.re, self.im)
        }
    }
}

/// Interpolating two complex numbers: component-wise real interpolation.
impl crate::field::Interpolate for Complex {
    fn lerp(self, other: Self, t: f32) -> Self {
        Complex::new(
            self.re + (other.re - self.re) * t,
            self.im + (other.im - self.im) * t,
        )
    }
}
