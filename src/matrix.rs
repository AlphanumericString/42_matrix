//! # Matrices
//!
//! A [`Matrix`] is a rectangular array of scalars and, more importantly, the
//! concrete representation of a *linear map* between finite-dimensional vector
//! spaces. As the subject puts it:
//!
//! > Usually in linear algebra, a linear map `f` corresponds to a matrix. [...]
//! > the space of 2-by-3 real matrices [...] corresponds to real linear maps
//! > from `R³` to `R²`.
//!
//! ## Storage: column-major
//!
//! The subject is explicit: "Unless specified otherwise, all matrices shown in
//! the examples are in column-major order." We therefore store the coefficients
//! **column by column**. The element at row `i`, column `j` lives at
//! `data[j * rows + i]`. Storing column-major (rather than the row-major layout
//! that `[[...]]` literals suggest) matters for two reasons:
//!
//! 1. It is the layout the provided projection-matrix display software expects,
//!    and the subject warns that a matrix in the wrong order "may appear to work
//!    at first glance, but will fail under careful scrutiny".
//! 2. It is the layout of the de-facto GPU standard (OpenGL/Vulkan), for which
//!    this whole module is a warm-up.
//!
//! The [`From`] implementation still accepts the natural, human-written
//! `[[row0], [row1], ...]` literals and transposes into the internal layout, so
//! the rest of the library is layout-agnostic.

use std::fmt;

use crate::field::{Field, Interpolate};
use crate::vector::Vector;

/// An error returned by fallible matrix operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixError {
    /// The operation requires a square matrix but got a rectangular one.
    NotSquare,
    /// The matrix has no inverse (its determinant is zero).
    Singular,
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::NotSquare => write!(f, "matrix is not square"),
            MatrixError::Singular => write!(f, "matrix is singular (not invertible)"),
        }
    }
}

impl std::error::Error for MatrixError {}

/// A dense matrix over the field `K`, stored in column-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<K: Field> {
    /// Coefficients in column-major order: `(i, j) -> data[j * rows + i]`.
    data: Vec<K>,
    /// Number of rows `m`.
    rows: usize,
    /// Number of columns `n`.
    cols: usize,
}

impl<K: Field> Matrix<K> {
    /// Create the zero matrix of shape `rows × cols`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![K::zero(); rows * cols],
            rows,
            cols,
        }
    }

    /// Build a matrix from coefficients already given in **column-major** order.
    ///
    /// Returns `None` if the slice does not contain exactly `rows * cols`
    /// coefficients.
    pub fn from_col_major(values: &[K], rows: usize, cols: usize) -> Option<Self> {
        if values.len() != rows * cols {
            return None;
        }
        Some(Matrix {
            data: values.to_vec(),
            rows,
            cols,
        })
    }

    /// The shape `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Is the matrix square (`rows == cols`)?
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The linear index of `(row, col)` in the column-major buffer.
    #[inline]
    fn index(&self, row: usize, col: usize) -> usize {
        col * self.rows + row
    }

    /// Set the coefficient at `(row, col)`. Returns `false` if out of bounds.
    pub fn set(&mut self, row: usize, col: usize, value: K) -> bool {
        if row >= self.rows || col >= self.cols {
            return false;
        }
        let idx = self.index(row, col);
        self.data[idx] = value;
        true
    }

    /// Get the coefficient at `(row, col)`.
    pub fn get(&self, row: usize, col: usize) -> Option<K> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        Some(self.data[self.index(row, col)])
    }

    /// Swap two rows in place (the elementary operation used by every
    /// pivoting-based algorithm below).
    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        for c in 0..self.cols {
            let va = self.get(a, c).unwrap();
            let vb = self.get(b, c).unwrap();
            self.set(a, c, vb);
            self.set(b, c, va);
        }
    }

    /// Print the matrix for debugging (rows aligned with `{:?}` formatting).
    pub fn dump(&self) {
        print!("{}", self);
    }

    /// In-place matrix addition: `self += other`.
    ///
    /// The subject groups vectors and matrices in exercise 00; the matrix case
    /// is identical coordinate-wise. `O(m·n)` time, `O(1)` extra space.
    ///
    /// # Panics
    /// Panics if the two matrices do not have the same shape.
    pub fn add(&mut self, other: &Matrix<K>) {
        assert_eq!(self.shape(), other.shape(), "matrix addition requires equal shapes");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += *b;
        }
    }

    /// In-place matrix subtraction: `self -= other`.
    pub fn sub(&mut self, other: &Matrix<K>) {
        assert_eq!(self.shape(), other.shape(), "matrix subtraction requires equal shapes");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= *b;
        }
    }

    /// In-place scaling: `self *= a`.
    pub fn scl(&mut self, a: K) {
        for x in self.data.iter_mut() {
            *x *= a;
        }
    }

    /// Exercise 07 — matrix × vector: `A·u`, giving a vector of dimension `m`.
    ///
    /// Every output coordinate is a dot product of a row of `A` with `u`, so the
    /// work is `O(m·n)`, matching the subject's `O(nm)` budget. We accumulate
    /// with the fused multiply-add.
    ///
    /// # Panics
    /// Panics if `u`'s dimension differs from `A`'s number of columns.
    pub fn mul_vec(&self, u: &Vector<K>) -> Vector<K> {
        assert_eq!(
            self.cols,
            u.size(),
            "matrix-vector product needs |u| == columns"
        );
        let mut out = Vector::new(self.rows);
        for i in 0..self.rows {
            let mut acc = K::zero();
            for j in 0..self.cols {
                acc = self.get(i, j).unwrap().fma(u.get(j).unwrap(), acc);
            }
            out.set(i, acc);
        }
        out
    }

    /// Exercise 07 — matrix × matrix: `A·B`.
    ///
    /// The composition `f ∘ g` of two linear maps corresponds to the product of
    /// their matrices, and is only defined when the "middle space" matches:
    /// `(m×n)·(n×p) = (m×p)`. This is `O(m·n·p)` time.
    ///
    /// # Panics
    /// Panics if `self.cols != other.rows`.
    pub fn mul_mat(&self, other: &Matrix<K>) -> Matrix<K> {
        assert_eq!(
            self.cols, other.rows,
            "matrix product needs |columns| == rows of the right factor"
        );
        let mut out = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut acc = K::zero();
                for k in 0..self.cols {
                    acc = self.get(i, k).unwrap().fma(other.get(k, j).unwrap(), acc);
                }
                out.set(i, j, acc);
            }
        }
        out
    }

    /// Exercise 08 — trace: `Tr(A) = Σ Aᵢᵢ`, the sum of the diagonal.
    ///
    /// The trace is basis-independent and equals the sum of the eigenvalues
    /// (counted with multiplicity). `O(n)`.
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    pub fn trace(&self) -> K {
        assert!(self.is_square(), "trace requires a square matrix");
        let mut sum = K::zero();
        for i in 0..self.rows {
            sum += self.get(i, i).unwrap();
        }
        sum
    }

    /// Exercise 09 — transpose: `Bᵢⱼ = Aⱼᵢ`.
    ///
    /// Geometrically the transpose mirrors the matrix across its diagonal. It is
    /// the adjoint of the map for the real inner product. `O(m·n)` assignments,
    /// which is what the subject counts here.
    pub fn transpose(&self) -> Matrix<K> {
        let mut out = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j).unwrap());
            }
        }
        out
    }

    /// Exercise 10 — reduced row-echelon form.
    ///
    /// We run Gauss–Jordan elimination with **partial pivoting**: at each step
    /// we pick, among the candidate rows, the one whose coefficient in the
    /// current column has the largest modulus. Pivoting on the largest available
    /// value is the standard trick for numerical stability — dividing by a tiny
    /// number amplifies rounding error, so we avoid it whenever possible. The
    /// result is the *reduced* form (leading `1`s with zeros above and below),
    /// which is what the subject's example shows.
    ///
    /// The method "is not allowed to use any mathematical library" and lists no
    /// permitted functions for this exercise, so we use only the basic field
    /// operations. Complexity is `O(m·n·min(m,n))`, within `O(n³)`.
    pub fn row_echelon(&self) -> Matrix<K> {
        let mut m = self.clone();
        let (rows, cols) = (m.rows, m.cols);
        let mut pivot_row = 0;
        for col in 0..cols {
            if pivot_row >= rows {
                break;
            }
            // Partial pivoting: find the row with the largest modulus.
            let mut best = pivot_row;
            for r in (pivot_row + 1)..rows {
                if m.get(r, col).unwrap().abs() > m.get(best, col).unwrap().abs() {
                    best = r;
                }
            }
            // No usable pivot in this column: move on to the next column.
            if m.get(best, col).unwrap().approx_zero() {
                continue;
            }
            if best != pivot_row {
                m.swap_rows(pivot_row, best);
            }
            // Normalize the pivot row so the pivot becomes 1.
            let inv = m.get(pivot_row, col).unwrap().inv();
            for c in 0..cols {
                let v = m.get(pivot_row, c).unwrap() * inv;
                m.set(pivot_row, c, v);
            }
            // Eliminate the current column in every other row.
            for r in 0..rows {
                if r == pivot_row {
                    continue;
                }
                let factor = m.get(r, col).unwrap();
                if factor.approx_zero() {
                    continue;
                }
                for c in 0..cols {
                    let v = m.get(r, c).unwrap() - factor * m.get(pivot_row, c).unwrap();
                    m.set(r, c, v);
                }
            }
            pivot_row += 1;
        }
        m
    }

    /// Exercise 11 — determinant.
    ///
    /// Computed by Gaussian elimination: the determinant of a triangular matrix
    /// is the product of its diagonal, and a single row swap flips the sign. We
    /// reuse the partial-pivoting idea from [`Matrix::row_echelon`]. The
    /// determinant is zero exactly when the matrix is singular (its columns are
    /// linearly dependent) — the subject asks you to be ready to explain this.
    ///
    /// Geometrically `|det(A)|` is the factor by which `A` scales `n`-volumes:
    /// areas in 2D, volumes in 3D. A zero determinant means the transformation
    /// collapses space onto a lower-dimensional subspace, destroying all
    /// information — which is precisely why no inverse can exist. This is also
    /// the magnitude of the pseudoscalar produced by wedging all the columns
    /// together (the subject's exterior-algebra aside).
    ///
    /// # Panics
    /// Panics if the matrix is not square.
    pub fn determinant(&self) -> K {
        assert!(self.is_square(), "determinant requires a square matrix");
        let n = self.rows;
        let mut m = self.clone();
        let mut det = K::one();
        for col in 0..n {
            let mut best = col;
            for r in (col + 1)..n {
                if m.get(r, col).unwrap().abs() > m.get(best, col).unwrap().abs() {
                    best = r;
                }
            }
            if m.get(best, col).unwrap().approx_zero() {
                return K::zero();
            }
            if best != col {
                m.swap_rows(col, best);
                det = -det;
            }
            let pivot = m.get(col, col).unwrap();
            det *= pivot;
            let inv = pivot.inv();
            for r in (col + 1)..n {
                let factor = m.get(r, col).unwrap() * inv;
                for c in col..n {
                    // m[r][c] - factor * m[col][c], fused.
                    let v = (-factor).fma(m.get(col, c).unwrap(), m.get(r, c).unwrap());
                    m.set(r, c, v);
                }
            }
        }
        det
    }

    /// Exercise 12 — inverse, or an error if the matrix is singular.
    ///
    /// We augment `A` with the identity and reduce `[A | I]` to `[I | A⁻¹]` by
    /// Gauss–Jordan elimination. This is the standard, numerically reasonable
    /// `O(n³)` method; a cofactor/adjugate formula would be `O(n!)` and much
    /// worse. If a zero pivot is ever found, `A` is singular and we return
    /// [`MatrixError::Singular`].
    ///
    /// # Errors
    /// Returns [`MatrixError::NotSquare`] for rectangular input and
    /// [`MatrixError::Singular`] when no inverse exists.
    pub fn inverse(&self) -> Result<Matrix<K>, MatrixError> {
        if !self.is_square() {
            return Err(MatrixError::NotSquare);
        }
        let n = self.rows;
        let mut aug = Matrix::new(n, 2 * n);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j).unwrap());
            }
            aug.set(i, n + i, K::one());
        }
        for col in 0..n {
            let mut best = col;
            for r in (col + 1)..n {
                if aug.get(r, col).unwrap().abs() > aug.get(best, col).unwrap().abs() {
                    best = r;
                }
            }
            if aug.get(best, col).unwrap().approx_zero() {
                return Err(MatrixError::Singular);
            }
            aug.swap_rows(col, best);
            let inv = aug.get(col, col).unwrap().inv();
            for c in 0..2 * n {
                let v = aug.get(col, c).unwrap() * inv;
                aug.set(col, c, v);
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let factor = aug.get(r, col).unwrap();
                if factor.approx_zero() {
                    continue;
                }
                for c in 0..2 * n {
                    let v = (-factor).fma(aug.get(col, c).unwrap(), aug.get(r, c).unwrap());
                    aug.set(r, c, v);
                }
            }
        }
        let mut result = Matrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, aug.get(i, n + j).unwrap());
            }
        }
        Ok(result)
    }

    /// Exercise 13 — rank: the dimension of the image of the linear map.
    ///
    /// The rank is the number of linearly independent rows (= columns) of the
    /// matrix. It is unchanged by row operations, so we count the non-zero rows
    /// of the row-echelon form. By the rank–nullity theorem the subject presents,
    /// `rank + nullity = dim(E)`: the rank is "how many dimensions survive" the
    /// transformation, the nullity "how many are squished to zero".
    ///
    /// Runs in `O(m·n·min(m,n))` because it is just [`Matrix::row_echelon`].
    pub fn rank(&self) -> usize {
        let rref = self.row_echelon();
        let mut rank = 0;
        for i in 0..rref.rows {
            let mut nonzero = false;
            for j in 0..rref.cols {
                if !rref.get(i, j).unwrap().approx_zero() {
                    nonzero = true;
                    break;
                }
            }
            if nonzero {
                rank += 1;
            }
        }
        rank
    }
}

/// Build a matrix from a nested array written row by row — the natural way
/// humans write matrices and the way every subject example does:
/// `Matrix::from([[1., 2.], [3., 4.]])`.
///
/// The outer array indexes rows and the inner arrays index columns, while the
/// internal storage is column-major, so we transpose during construction.
impl<K: Field, const M: usize, const N: usize> From<[[K; N]; M]> for Matrix<K> {
    fn from(rows: [[K; N]; M]) -> Self {
        let mut data = Vec::with_capacity(M * N);
        for j in 0..N {
            for row in rows.iter() {
                data.push(row[j]);
            }
        }
        Matrix {
            data,
            rows: M,
            cols: N,
        }
    }
}

/// Print one row per line, each row bracketed, matching the subject:
///
/// ```text
/// [1.0, 2.0]
/// [3.0, 4.0]
/// ```
impl<K: Field> fmt::Display for Matrix<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut lines = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let cells: Vec<String> = (0..self.cols)
                .map(|j| format!("{}", self.get(i, j).unwrap().normalized()))
                .collect();
            lines.push(format!("[{}]", cells.join(", ")));
        }
        write!(f, "{}", lines.join("\n"))
    }
}

/// Element-wise linear interpolation of two matrices.
impl<K: Field> Interpolate for Matrix<K> {
    fn lerp(self, other: Self, t: f32) -> Self {
        assert_eq!(self.shape(), other.shape(), "lerp requires equal shapes");
        let tk = K::from_f64(t as f64);
        let one_minus_tk = K::from_f64((1.0 - t) as f64);
        let mut result = Matrix::new(self.rows, self.cols);
        for idx in 0..self.data.len() {
            result.data[idx] = one_minus_tk.fma(self.data[idx], tk * other.data[idx]);
        }
        result
    }
}

/// Exercise 14 (bonus) — perspective projection matrix for a 3D renderer.
///
/// Maps the *view frustum* (the truncated pyramid of everything the camera can
/// see, between the `near` and `far` planes) onto the normalized device
/// coordinates the display software expects:
///
/// * X and Y in `[-1, 1]`,
/// * Z in `[0, 1]`.
///
/// # Homogeneous coordinates
///
/// A perspective divide is not a linear map — it depends on `z` non-linearly —
/// so it cannot be a plain 4×4 matrix. The trick, which the subject explains, is
/// *homogeneous coordinates*: we work in 4D and only divide by the `w`
/// component after the multiplication. That is exactly what our `f32` scalar
/// field allows, and why the `w` row below is `[0, 0, -1, 0]`: after the
/// product, `w` equals `-z`, and dividing by it performs the perspective
/// foreshortening.
///
/// # Derivation
///
/// With a camera looking down `-z`, we want
/// `ndc_z = (a·z + b) / (-z)` to be `0` at `z = -near` and `1` at `z = -far`.
/// Solving the two equations gives `a = far/(near - far)` and
/// `b = far·near/(near - far)`.
///
/// The X and Y scales use `f = 1 / tan(fov/2)`: the tangent of half the
/// field-of-view tells us how wide the frustum is at a unit distance, and the
/// aspect `ratio = w/h` corrects the horizontal axis.
///
/// `tan` is one of the two mathematical functions this exercise explicitly
/// permits.
pub fn projection(fov: f32, ratio: f32, near: f32, far: f32) -> Matrix<f32> {
    let f = 1.0 / (fov / 2.0).tan();
    let mut m = Matrix::new(4, 4);
    // Column 0, column 1: the horizontal / vertical scaling.
    m.set(0, 0, f / ratio);
    m.set(1, 1, f);
    // Column 2: the z mapping (row 2 scales z, row 3 sets w = -z).
    m.set(2, 2, far / (near - far));
    m.set(3, 2, -1.0);
    // Column 3: the constant term of the z mapping.
    m.set(2, 3, far * near / (near - far));
    m
}
