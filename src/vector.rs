//! # Vectors
//!
//! A [`Vector`] is the fundamental object of a vector space: an ordered list of
//! scalars. In finite dimension it is *always* describable as such a list, as
//! the subject explains:
//!
//! > For finite dimensions, V is always equivalent to Kⁿ [...] this means that
//! > in finite dimensions, V can ALWAYS be understood as a list/array/tuple
//! > containing n elements of K.
//!
//! We therefore store a `Vec<K>` and its length. The length is kept explicitly
//! (rather than relying on `data.len()`) to mirror the two-field shape the
//! subject suggests and to make `size()` an O(1) lookup.

use std::fmt;

use crate::field::{sqrt, Field, Interpolate};

/// A finite-dimensional vector over the field `K`.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector<K: Field> {
    /// The coordinates, in order.
    data: Vec<K>,
    /// The dimension `n` of the vector (`data.len()`).
    len: usize,
}

impl<K: Field> Vector<K> {
    /// Create the zero vector of dimension `size`.
    pub fn new(size: usize) -> Self {
        Vector {
            data: vec![K::zero(); size],
            len: size,
        }
    }

    /// Build a vector from a slice of coordinates.
    pub fn from_slice(slice: &[K]) -> Self {
        Vector {
            data: slice.to_vec(),
            len: slice.len(),
        }
    }

    /// The dimension of the vector (its "shape").
    pub fn size(&self) -> usize {
        self.len
    }

    /// Set the coordinate at index `i`. Returns `false` (and touches nothing)
    /// when `i` is out of bounds. Using a `bool` rather than panicking keeps the
    /// low-level accessor usable in generic code.
    pub fn set(&mut self, i: usize, value: K) -> bool {
        if i >= self.len {
            return false;
        }
        self.data[i] = value;
        true
    }

    /// Get the coordinate at index `i`.
    pub fn get(&self, i: usize) -> Option<K> {
        self.data.get(i).copied()
    }

    /// In-place vector addition: `self += v`.
    ///
    /// The subject's prototype stores the result in the receiver. The operation
    /// is only defined for equal dimensions; anything else "is undefined", so we
    /// assert. Time complexity is `O(n)` (one addition per coordinate) and space
    /// complexity is `O(1)` on top of the inputs.
    ///
    /// # Panics
    /// Panics if the two vectors do not have the same dimension.
    pub fn add(&mut self, v: &Vector<K>) {
        assert_eq!(self.len, v.len, "vector addition requires equal sizes");
        for (a, b) in self.data.iter_mut().zip(v.data.iter()) {
            *a += *b;
        }
    }

    /// In-place vector subtraction: `self -= v`.
    ///
    /// Same contract as [`Vector::add`].
    pub fn sub(&mut self, v: &Vector<K>) {
        assert_eq!(self.len, v.len, "vector subtraction requires equal sizes");
        for (a, b) in self.data.iter_mut().zip(v.data.iter()) {
            *a -= *b;
        }
    }

    /// In-place scaling: `self *= a`.
    ///
    /// This is the scalar multiplication `λu` whose axioms the subject lists in
    /// the "Vector space" section. `O(n)` time, `O(1)` extra space.
    pub fn scl(&mut self, a: K) {
        for x in self.data.iter_mut() {
            *x *= a;
        }
    }

    /// The dot (inner) product `⟨u|v⟩`.
    ///
    /// For real vectors this is the familiar `Σ uᵢ·vᵢ`. For complex vectors the
    /// mathematically correct bilinear form is *sesquilinear*: we must conjugate
    /// the first argument, `⟨u|v⟩ = Σ conj(uᵢ)·vᵢ` (see the subject's "Inner
    /// product on complex vector spaces"). Because [`Field::conj`] defaults to
    /// the identity on `R`, one implementation covers both fields.
    ///
    /// We accumulate with [`Field::fma`] — the fused multiply-add the subject
    /// highlights — which is both faster and more accurate than `acc + u*v`.
    pub fn dot(&self, v: &Vector<K>) -> K {
        assert_eq!(self.len, v.len, "dot product requires equal sizes");
        let mut acc = K::zero();
        for (a, b) in self.data.iter().zip(v.data.iter()) {
            acc = a.conj().fma(*b, acc);
        }
        acc
    }

    /// The 1-norm (taxicab / Manhattan norm): `Σ |xᵢ|`.
    pub fn norm_1(&self) -> f64 {
        self.data.iter().fold(0.0, |acc, x| acc + x.abs())
    }

    /// The 2-norm (Euclidean norm): `sqrt(Σ |xᵢ|²)`.
    ///
    /// The square root is expressed through `pow` because that is the only
    /// power/root function the subject allows in exercise 04 (see
    /// [`crate::field::sqrt`]). This is the norm induced by [`Vector::dot`].
    pub fn norm(&self) -> f64 {
        let sum = self.data.iter().fold(0.0, |acc, x| {
            let m = x.abs();
            acc + m * m
        });
        sqrt(sum)
    }

    /// The ∞-norm (supremum norm): `max |xᵢ|`.
    pub fn norm_inf(&self) -> f64 {
        self.data
            .iter()
            .fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc })
    }
}

/// Build a vector directly from a Rust array: `Vector::from([1., 2., 3.])`.
impl<K: Field, const N: usize> From<[K; N]> for Vector<K> {
    fn from(arr: [K; N]) -> Self {
        Vector::from_slice(&arr)
    }
}

/// Print a vector one coordinate per line between brackets, exactly like the
/// subject's examples:
///
/// ```text
/// [7.0]
/// [10.0]
/// ```
impl<K: Field> fmt::Display for Vector<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let body: Vec<String> = self.data.iter().map(|x| format!("{}", x.normalized())).collect();
        write!(f, "[{}]", body.join("]\n["))
    }
}

/// Exercise 01 — linear combination.
///
/// Given vectors `u = (u₁, …, u_k)` and scalars `λ = (λ₁, …, λ_k)`, compute
/// `Σ λᵢ·uᵢ`. This generalizes both scaling (one term) and vector addition (all
/// coefficients `1`); it is the primitive behind every matrix product.
///
/// A single pass over the `n` coordinates of one vector does `k` fused
/// multiply-adds per coordinate, so the time is `O(n·k)`. The subject's `O(n)`
/// counts the size `n` of the input, which here is the total number of
/// coordinates read across all inputs.
///
/// # Panics
/// Panics if `u` and `coefs` have different lengths.
pub fn linear_combination<K: Field>(u: &[Vector<K>], coefs: &[K]) -> Vector<K> {
    assert_eq!(
        u.len(),
        coefs.len(),
        "linear combination needs as many coefficients as vectors"
    );
    if u.is_empty() {
        return Vector::new(0);
    }
    let dim = u[0].size();
    let mut result = Vector::new(dim);
    for (vector, &coef) in u.iter().zip(coefs.iter()) {
        assert_eq!(vector.size(), dim, "all vectors must share a dimension");
        // Every coordinate accumulates `coef * vᵢ` on top of the running sum,
        // which is precisely a fused multiply-add.
        for (acc, &x) in result.data.iter_mut().zip(vector.data.iter()) {
            *acc = coef.fma(x, *acc);
        }
    }
    result
}

/// Exercise 02 — linear interpolation.
///
/// Returns the point at parameter `t` between `u` and `v`:
/// `f(u, v, t) = (1 - t)·u + t·v`. It satisfies `f(u, v, 0) = u` and
/// `f(u, v, 1) = v`, and for `t = 0.5` gives the midpoint. Although usually used
/// with `t ∈ [0, 1]`, the formula extends the points onto the whole line — which
/// is exactly what a raytracer needs to parametrize a ray — so we do not clamp.
///
/// The genericity over [`Interpolate`] lets this single function serve scalars,
/// vectors *and* matrices, as the subject's examples require.
pub fn lerp<V: Interpolate>(u: V, v: V, t: f32) -> V {
    u.lerp(v, t)
}

impl<K: Field> Interpolate for Vector<K> {
    fn lerp(self, other: Self, t: f32) -> Self {
        assert_eq!(self.size(), other.size(), "lerp requires equal sizes");
        let tk = K::from_f64(t as f64);
        let one_minus_tk = K::from_f64((1.0 - t) as f64);
        let mut result = Vector::new(self.size());
        for (i, acc) in result.data.iter_mut().enumerate() {
            *acc = one_minus_tk.fma(
                self.data[i],
                tk * other.data[i],
            );
        }
        result
    }
}

/// Exercise 05 — cosine of the angle between two vectors.
///
/// `cos(u, v) = ⟨u|v⟩ / (‖u‖·‖v‖)`. This is the function whose statistical
/// cousin is the Pearson correlation coefficient (the subject's aside): replace
/// the dot product with covariance and the norm with standard deviation and you
/// get the same formula. For complex vectors we take the real part of the inner
/// product, as required for it to be an actual angle.
///
/// The behaviour is undefined for zero vectors or mismatched sizes, so we assert.
///
/// # Panics
/// Panics if the vectors have different sizes or the denominator is zero.
pub fn angle_cos<K: Field>(u: &Vector<K>, v: &Vector<K>) -> f64 {
    assert_eq!(u.size(), v.size(), "angle_cos requires equal sizes");
    let denom = u.norm() * v.norm();
    assert!(denom != 0.0, "cosine is undefined for the zero vector");
    u.dot(v).to_f64() / denom
}

/// Exercise 06 — cross product of two 3-dimensional vectors.
///
/// Defined only in 3D, where it is the Hodge dual of the wedge product (the
/// subject's aside on the exterior algebra explains why it does not generalize
/// nicely). The result is the vector perpendicular to the parallelogram spanned
/// by `u` and `v`, with magnitude equal to its area.
///
/// # Panics
/// Panics unless both vectors are exactly 3-dimensional.
pub fn cross_product<K: Field>(u: &Vector<K>, v: &Vector<K>) -> Vector<K> {
    assert!(
        u.size() == 3 && v.size() == 3,
        "cross product is only defined in 3 dimensions"
    );
    let a = |i: usize| u.data[i];
    let b = |i: usize| v.data[i];
    Vector::from_slice(&[
        a(1) * b(2) - a(2) * b(1),
        a(2) * b(0) - a(0) * b(2),
        a(0) * b(1) - a(1) * b(0),
    ])
}
