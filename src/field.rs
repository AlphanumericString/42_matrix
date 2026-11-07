//! # The scalar field abstraction
//!
//! Everything in this module is generic over a *field* `K`. The subject insists
//! on this genericity from the very first page: the exercises are written with a
//! generic `K`, the bonus asks to redo everything with `K = C`, and the "Code
//! constraints" chapter explicitly says:
//!
//! > What is really important is that the underlying mathematical function is
//! > implemented in a way that is: **generic**, **coherent**, **faithful**.
//!
//! Linear algebra is *defined* over a field. A field is a set with two
//! operations (`+` and `*`), their respective neutral elements (`0` and `1`),
//! additive inverses (subtraction / negation) and multiplicative inverses
//! (division). The ring of integers, for example, is **not** a field because you
//! cannot always divide (there is no integer `1/2`). The real numbers and the
//! complex numbers *are* fields, which is exactly why they are the two examples
//! used in this module.
//!
//! Instead of hard-coding `f32` everywhere (which is what the simplest solution
//! would do), we describe the *interface* a type must provide to play the role of
//! a scalar. This is the same idea as the `Field` type mentioned in the subject:
//!
//! > Generally, it should even be expected that the generic types provided would
//! > actually have some form of type constraint in a real implementation (for
//! > example, making a `Field` type, and having the generic `K` extend `Field`).
//!
//! Keeping the code generic gives us the complex-number bonus (exercise 15) for
//! free: we only have to write one more implementation of this trait.

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// The mathematical notion of a *field*.
///
/// Implementing this trait tells the library that your type can be used as the
/// scalar type `K` of a [`crate::Vector`] or a [`crate::Matrix`].
///
/// The trait bundles together:
///
/// * The `std::ops` traits that correspond to the field laws (`+`, `-`, `*`,
///   `/`, unary `-`). This is what allows us to write `a * b + c` in the
///   generic code exactly like we would with `f32`.
/// * A handful of helpers that are impossible to express with `std::ops`
///   alone: the neutral elements, a conversion from a real literal, a
///   conjugation (a no-op outside of `C`) and a *modulus* (absolute value)
///   which is always a real number — this is what norms need.
///
/// # Why `Copy`?
///
/// Matrices and vectors are small value types whose scalars are cheap to copy
/// (`f32`, or a two-`f32` complex). Requiring `Copy` keeps every formula
/// readable and avoids a sea of `.clone()` calls. We never store huge scalars.
pub trait Field:
    Sized
    + Copy
    + Clone
    + Debug
    + Display
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// The additive neutral element `0` of the field.
    fn zero() -> Self;

    /// The multiplicative neutral element `1` of the field.
    fn one() -> Self;

    /// Build a scalar from a real (f64) literal.
    ///
    /// This is used for the places where a real constant is intrinsically part
    /// of an algorithm, such as the interpolation parameter `t` in
    /// [`crate::lerp`], or the `1/n` used to normalize a row. It is also handy
    /// to write generic tests.
    fn from_f64(x: f64) -> Self;

    /// Project a scalar onto the real line.
    ///
    /// For `R` this is the identity. For `C` it returns the real part. It is
    /// only ever used for *display* and for tests, never to define the
    /// mathematics of the library.
    fn to_f64(self) -> f64;

    /// The *modulus* (absolute value) of a scalar, returned as a real number.
    ///
    /// This is the one operation that "leaves" the field: the absolute value of
    /// a complex number is real, and the subject reminds us that "norms always
    /// return real numbers, even for complex-valued vectors". Returning `f64`
    /// gives every field a common, lossless measurement type (`f32` fits in
    /// `f64` exactly).
    fn abs(self) -> f64;

    /// The *conjugation* `z -> z̄` of the field.
    ///
    /// On real numbers there is nothing to do, so the default is the identity.
    /// On `C` it flips the sign of the imaginary part. Conjugation is what makes
    /// the inner product of a *complex* vector space behave correctly (it must
    /// be sesquilinear rather than bilinear): see
    /// [`crate::Vector::dot`] and the "Inner product on complex vector spaces"
    /// section of the subject.
    fn conj(self) -> Self {
        self
    }

    /// A fused multiply-add: `self * b + c`.
    ///
    /// The subject dedicates a whole section to this operation and to the
    /// `vfmadd*ps` family of x86 instructions. The whole point of a *fused*
    /// multiply-add is that the product is not rounded before the addition, so
    /// the result is both **faster** (one instruction) and **more accurate**
    /// (a single rounding step) than separate `*` and `+`. In Rust the standard
    /// library exposes it as [`f32::mul_add`].
    fn fma(self, b: Self, c: Self) -> Self {
        self * b + c
    }

    /// The multiplicative inverse `1 / self`.
    fn inv(self) -> Self {
        Self::one() / self
    }

    /// Is the scalar exactly zero?
    fn is_zero(self) -> bool {
        self == Self::zero()
    }

    /// The tolerance used when deciding whether a value is "numerically zero".
    ///
    /// We use a slightly generous epsilon on purpose: the subject repeatedly
    /// warns that "the results may be a bit approximate (within reason) since
    /// making the algorithm numerically stable can sometimes be difficult", and
    /// a pivot that is a tiny rounding error rather than an exact `0.0` would
    /// otherwise poison the row-echelon / determinant / inverse algorithms.
    fn epsilon() -> f64 {
        1e-6
    }

    /// Is the scalar indistinguishable from zero at our working precision?
    fn approx_zero(self) -> bool {
        self.abs() <= Self::epsilon()
    }

    /// Normalize a value for display.
    ///
    /// Row-reduction occasionally produces `-0.0` (for example `0 - 0`), which
    /// `f32` happily prints as `-0`. That is mathematically the same number and
    /// would only confuse a reader, so the real implementation collapses every
    /// kind of zero to a clean `0`. The default is the identity.
    fn normalized(self) -> Self {
        self
    }
}

/// The real number field, as required by the subject (`f32`).
///
/// The module asks us to implement everything for real numbers,
/// "which will be defined as the type `f32`". We follow that literally here,
/// even though the trait itself would work equally well with `f64`.
impl Field for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_f64(x: f64) -> Self {
        x as f32
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    /// `|x|`, computed with a comparison and a negation only.
    ///
    /// Exercises 04 and 05 only allow a *very* small set of "mathematical
    /// functions" (fused multiply-add, `pow`, `max`). `f32::abs` is arguably
    /// part of that forbidden standard library, so we avoid it and derive the
    /// absolute value from the much more primitive `if x < 0 { -x } else { x }`.
    fn abs(self) -> f64 {
        if self < 0.0 { (-self) as f64 } else { self as f64 }
    }

    /// The real fused multiply-add, backed by the CPU's FMA when available.
    fn fma(self, b: Self, c: Self) -> Self {
        f32::mul_add(self, b, c)
    }

    /// Collapse `-0.0` onto `0.0`. Since `-0.0 == 0.0` in IEEE-754, the
    /// comparison catches both signs.
    fn normalized(self) -> Self {
        if self == 0.0 { 0.0 } else { self }
    }
}

/// Compute a square root using only `pow`, which the subject explicitly allows.
///
/// Exercise 04 lists `pow` (but not `sqrt`) among the permitted mathematical
/// functions, and `sqrt(x)` is nothing more than `pow(x, 0.5)`. Spell that out
/// here so the compliance with the constraints is obvious to a reviewer.
pub(crate) fn sqrt(x: f64) -> f64 {
    x.powf(0.5)
}

/// Types that can be linearly interpolated, *i.e.* objects lying in a vector
/// space on which `u + t * (v - u)` makes sense.
///
/// Exercise 02 asks for a single `lerp` function that works on plain scalars
/// (`f32`), on [`crate::Vector`]s **and** on [`crate::Matrix`]es:
///
/// ```text
/// lerp(0., 1., 0.5)                                   // 0.5
/// lerp(Vector::from([2., 1.]), Vector::from([4., 2.]), 0.3)
/// lerp(Matrix::from(...), Matrix::from(...), 0.5)
/// ```
///
/// A single generic function can only accept all three if they share a trait,
/// so we introduce one. Each implementor decides how to combine its own pieces,
/// and the free function [`crate::lerp`] is then a one-liner.
pub trait Interpolate: Sized {
    /// Return the point at parameter `t` on the line from `self` to `other`.
    fn lerp(self, other: Self, t: f32) -> Self;
}

/// Interpolating two real scalars is the textbook `(1 - t) * a + t * b`.
impl Interpolate for f32 {
    fn lerp(self, other: Self, t: f32) -> Self {
        // Reuse `Field::fma` so the whole library funnels its floating point
        // work through the same fused operation.
        (1.0 - t).fma(self, t * other)
    }
}
