//! Integration tests covering every exercise of the subject.
//!
//! The numeric results follow the examples given in the subject. Because the
//! library is generic over [`f32`] (and because row-reduction/determinant
//! algorithms are, as the subject warns, only "approximate within reason"), the
//! floating-point assertions use a tolerance rather than exact equality.

use matrix::prelude::*;

/// Assert two `f32` are within a small tolerance.
///
/// The tolerance is generous (`1e-3`) on purpose: the deterministic algorithms
/// run in `f32`, and the subject's own example for the 4×4 determinant is given
/// as `1032` while the real computation lands on `1031.9999`.
fn approx(a: f32, b: f32) {
    assert!((a - b).abs() < 1e-3, "expected {} ≈ {}", a, b);
}

/// Assert two vectors are element-wise approximately equal.
fn assert_vec_approx(a: &Vector<f32>, b: &Vector<f32>) {
    assert_eq!(a.size(), b.size());
    for i in 0..a.size() {
        approx(a.get(i).unwrap(), b.get(i).unwrap());
    }
}

/// Assert two matrices are element-wise approximately equal.
fn assert_mat_approx(a: &Matrix<f32>, b: &Matrix<f32>) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
    let (rows, cols) = a.shape();
    for i in 0..rows {
        for j in 0..cols {
            approx(a.get(i, j).unwrap(), b.get(i, j).unwrap());
        }
    }
}

// ---------------------------------------------------------------------------
// Exercise 00 — Add, Subtract and Scale
// ---------------------------------------------------------------------------

#[test]
fn ex00_vector_add() {
    let mut u = Vector::from([2., 3.]);
    u.add(&Vector::from([5., 7.]));
    assert_eq!(u, Vector::from([7., 10.]));
}

#[test]
fn ex00_vector_sub() {
    let mut u = Vector::from([2., 3.]);
    u.sub(&Vector::from([5., 7.]));
    assert_eq!(u, Vector::from([-3., -4.]));
}

#[test]
fn ex00_vector_scl() {
    let mut u = Vector::from([2., 3.]);
    u.scl(2.);
    assert_eq!(u, Vector::from([4., 6.]));
}

#[test]
fn ex00_matrix_add_sub_scl() {
    let mut u = Matrix::from([[1., 2.], [3., 4.]]);
    u.add(&Matrix::from([[7., 4.], [-2., 2.]]));
    assert_eq!(u, Matrix::from([[8., 6.], [1., 6.]]));

    let mut u = Matrix::from([[1., 2.], [3., 4.]]);
    u.sub(&Matrix::from([[7., 4.], [-2., 2.]]));
    assert_eq!(u, Matrix::from([[-6., -2.], [5., 2.]]));

    let mut u = Matrix::from([[1., 2.], [3., 4.]]);
    u.scl(2.);
    assert_eq!(u, Matrix::from([[2., 4.], [6., 8.]]));
}

#[test]
#[should_panic]
fn ex00_vector_add_dimension_mismatch() {
    Vector::from([1., 2.]).add(&Vector::from([1., 2., 3.]));
}

// ---------------------------------------------------------------------------
// Exercise 01 — Linear combination
// ---------------------------------------------------------------------------

#[test]
fn ex01_linear_combination_basis() {
    let e1 = Vector::from([1., 0., 0.]);
    let e2 = Vector::from([0., 1., 0.]);
    let e3 = Vector::from([0., 0., 1.]);
    assert_vec_approx(
        &linear_combination(&[e1, e2, e3], &[10., -2., 0.5]),
        &Vector::from([10., -2., 0.5]),
    );
}

#[test]
fn ex01_linear_combination_general() {
    let v1 = Vector::from([1., 2., 3.]);
    let v2 = Vector::from([0., 10., -100.]);
    assert_vec_approx(
        &linear_combination(&[v1, v2], &[10., -2.]),
        &Vector::from([10., 0., 230.]),
    );
}

// ---------------------------------------------------------------------------
// Exercise 02 — Linear interpolation
// ---------------------------------------------------------------------------

#[test]
fn ex02_lerp_scalars() {
    approx(lerp(0., 1., 0.), 0.);
    approx(lerp(0., 1., 1.), 1.);
    approx(lerp(0., 1., 0.5), 0.5);
    approx(lerp(21., 42., 0.3), 27.3);
}

#[test]
fn ex02_lerp_vectors() {
    assert_vec_approx(
        &lerp(Vector::from([2., 1.]), Vector::from([4., 2.]), 0.3),
        &Vector::from([2.6, 1.3]),
    );
}

#[test]
fn ex02_lerp_matrices() {
    assert_mat_approx(
        &lerp(
            Matrix::from([[2., 1.], [3., 4.]]),
            Matrix::from([[20., 10.], [30., 40.]]),
            0.5,
        ),
        &Matrix::from([[11., 5.5], [16.5, 22.]]),
    );
}

// ---------------------------------------------------------------------------
// Exercise 03 — Dot product
// ---------------------------------------------------------------------------

#[test]
fn ex03_dot() {
    approx(Vector::from([0., 0.]).dot(&Vector::from([1., 1.])), 0.);
    approx(Vector::from([1., 1.]).dot(&Vector::from([1., 1.])), 2.);
    approx(Vector::from([-1., 6.]).dot(&Vector::from([3., 2.])), 9.);
}

// ---------------------------------------------------------------------------
// Exercise 04 — Norms
// ---------------------------------------------------------------------------

#[test]
fn ex04_norms() {
    let u = Vector::from([0., 0., 0.]);
    approx(u.norm_1() as f32, 0.);
    approx(u.norm() as f32, 0.);
    approx(u.norm_inf() as f32, 0.);

    let u = Vector::from([1., 2., 3.]);
    approx(u.norm_1() as f32, 6.);
    approx(u.norm() as f32, 3.74165738);
    approx(u.norm_inf() as f32, 3.);

    let u = Vector::from([-1., -2.]);
    approx(u.norm_1() as f32, 3.);
    approx(u.norm() as f32, 2.236067977);
    approx(u.norm_inf() as f32, 2.);
}

// ---------------------------------------------------------------------------
// Exercise 05 — Cosine
// ---------------------------------------------------------------------------

#[test]
fn ex05_angle_cos() {
    approx(
        angle_cos(&Vector::from([1., 0.]), &Vector::from([1., 0.])) as f32,
        1.,
    );
    approx(
        angle_cos(&Vector::from([1., 0.]), &Vector::from([0., 1.])) as f32,
        0.,
    );
    approx(
        angle_cos(&Vector::from([-1., 1.]), &Vector::from([1., -1.])) as f32,
        -1.,
    );
    approx(
        angle_cos(&Vector::from([2., 1.]), &Vector::from([4., 2.])) as f32,
        1.,
    );
    approx(
        angle_cos(&Vector::from([1., 2., 3.]), &Vector::from([4., 5., 6.])) as f32,
        0.974631846,
    );
}

// ---------------------------------------------------------------------------
// Exercise 06 — Cross product
// ---------------------------------------------------------------------------

#[test]
fn ex06_cross_product() {
    assert_vec_approx(
        &cross_product(&Vector::from([0., 0., 1.]), &Vector::from([1., 0., 0.])),
        &Vector::from([0., 1., 0.]),
    );
    assert_vec_approx(
        &cross_product(&Vector::from([1., 2., 3.]), &Vector::from([4., 5., 6.])),
        &Vector::from([-3., 6., -3.]),
    );
    assert_vec_approx(
        &cross_product(
            &Vector::from([4., 2., -3.]),
            &Vector::from([-2., -5., 16.]),
        ),
        &Vector::from([17., -58., -16.]),
    );
}

// ---------------------------------------------------------------------------
// Exercise 07 — Linear map, matrix multiplication
// ---------------------------------------------------------------------------

#[test]
fn ex07_mul_vec() {
    assert_vec_approx(
        &Matrix::from([[1., 0.], [0., 1.]]).mul_vec(&Vector::from([4., 2.])),
        &Vector::from([4., 2.]),
    );
    assert_vec_approx(
        &Matrix::from([[2., 0.], [0., 2.]]).mul_vec(&Vector::from([4., 2.])),
        &Vector::from([8., 4.]),
    );
    assert_vec_approx(
        &Matrix::from([[2., -2.], [-2., 2.]]).mul_vec(&Vector::from([4., 2.])),
        &Vector::from([4., -4.]),
    );
}

#[test]
fn ex07_mul_mat() {
    assert_mat_approx(
        &Matrix::from([[1., 0.], [0., 1.]]).mul_mat(&Matrix::from([[1., 0.], [0., 1.]])),
        &Matrix::from([[1., 0.], [0., 1.]]),
    );
    assert_mat_approx(
        &Matrix::from([[1., 0.], [0., 1.]]).mul_mat(&Matrix::from([[2., 1.], [4., 2.]])),
        &Matrix::from([[2., 1.], [4., 2.]]),
    );
    assert_mat_approx(
        &Matrix::from([[3., -5.], [6., 8.]]).mul_mat(&Matrix::from([[2., 1.], [4., 2.]])),
        &Matrix::from([[-14., -7.], [44., 22.]]),
    );
}

#[test]
fn ex07_mul_non_square() {
    // (2x3) * (3x2) = (2x2)
    let a = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
    let b = Matrix::from([[7., 8.], [9., 10.], [11., 12.]]);
    assert_mat_approx(&a.mul_mat(&b), &Matrix::from([[58., 64.], [139., 154.]]));
}

// ---------------------------------------------------------------------------
// Exercise 08 — Trace
// ---------------------------------------------------------------------------

#[test]
fn ex08_trace() {
    approx(Matrix::from([[1., 0.], [0., 1.]]).trace(), 2.);
    approx(
        Matrix::from([[2., -5., 0.], [4., 3., 7.], [-2., 3., 4.]]).trace(),
        9.,
    );
    approx(
        Matrix::from([[-2., -8., 4.], [1., -23., 4.], [0., 6., 4.]]).trace(),
        -21.,
    );
}

// ---------------------------------------------------------------------------
// Exercise 09 — Transpose
// ---------------------------------------------------------------------------

#[test]
fn ex09_transpose() {
    let m = Matrix::from([[1., 2.], [3., 4.]]);
    assert_mat_approx(&m.transpose(), &Matrix::from([[1., 3.], [2., 4.]]));

    let m = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
    assert_mat_approx(
        &m.transpose(),
        &Matrix::from([[1., 4.], [2., 5.], [3., 6.]]),
    );

    // The transpose of the transpose is the original.
    assert_mat_approx(&m.transpose().transpose(), &m);
}

// ---------------------------------------------------------------------------
// Exercise 10 — Row-echelon form
// ---------------------------------------------------------------------------

#[test]
fn ex10_row_echelon() {
    assert_mat_approx(
        &Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).row_echelon(),
        &Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
    );
    assert_mat_approx(
        &Matrix::from([[1., 2.], [3., 4.]]).row_echelon(),
        &Matrix::from([[1., 0.], [0., 1.]]),
    );
    assert_mat_approx(
        &Matrix::from([[1., 2.], [2., 4.]]).row_echelon(),
        &Matrix::from([[1., 2.], [0., 0.]]),
    );
    assert_mat_approx(
        &Matrix::from([
            [8., 5., -2., 4., 28.],
            [4., 2.5, 20., 4., -4.],
            [8., 5., 1., 4., 17.],
        ])
        .row_echelon(),
        &Matrix::from([
            [1., 0.625, 0., 0., -12.1666667],
            [0., 0., 1., 0., -3.6666667],
            [0., 0., 0., 1., 29.5],
        ]),
    );
}

// ---------------------------------------------------------------------------
// Exercise 11 — Determinant
// ---------------------------------------------------------------------------

#[test]
fn ex11_determinant() {
    approx(Matrix::from([[1., -1.], [-1., 1.]]).determinant(), 0.);
    approx(
        Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]).determinant(),
        8.,
    );
    approx(
        Matrix::from([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]).determinant(),
        -174.,
    );
    approx(
        Matrix::from([
            [8., 5., -2., 4.],
            [4., 2.5, 20., 4.],
            [8., 5., 1., 4.],
            [28., -4., 17., 1.],
        ])
        .determinant(),
        1032.,
    );
}

#[test]
fn ex11_determinant_1x1_and_2x2() {
    approx(Matrix::from([[5.]]).determinant(), 5.);
    approx(Matrix::from([[1., 2.], [3., 4.]]).determinant(), -2.);
}

// ---------------------------------------------------------------------------
// Exercise 12 — Inverse
// ---------------------------------------------------------------------------

#[test]
fn ex12_inverse() {
    let id = Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
    assert_mat_approx(&id.inverse().unwrap(), &id);

    assert_mat_approx(
        &Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
            .inverse()
            .unwrap(),
        &Matrix::from([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 0.5]]),
    );

    let a = Matrix::from([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]);
    let inv = a.inverse().unwrap();
    // A · A⁻¹ = I
    assert_mat_approx(&a.mul_mat(&inv), &id);
    assert_mat_approx(
        &inv,
        &Matrix::from([
            [0.649425287, 0.097701149, -0.655172414],
            [-0.781609195, -0.126436782, 0.965517241],
            [0.143678161, 0.074712644, -0.206896552],
        ]),
    );
}

#[test]
fn ex12_inverse_singular_errors() {
    assert_eq!(
        Matrix::from([[1., 2.], [2., 4.]]).inverse(),
        Err(MatrixError::Singular)
    );
}

#[test]
fn ex12_inverse_not_square_errors() {
    assert_eq!(
        Matrix::from([[1., 2., 3.], [4., 5., 6.]]).inverse(),
        Err(MatrixError::NotSquare)
    );
}

// ---------------------------------------------------------------------------
// Exercise 13 — Rank
// ---------------------------------------------------------------------------

#[test]
fn ex13_rank() {
    assert_eq!(
        Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).rank(),
        3
    );
    assert_eq!(
        Matrix::from([[1., 2., 0., 0.], [2., 4., 0., 0.], [-1., 2., 1., 1.]]).rank(),
        2
    );
    assert_eq!(
        Matrix::from([
            [8., 5., -2.],
            [4., 7., 20.],
            [7., 6., 1.],
            [21., 18., 7.],
        ])
        .rank(),
        3
    );
}

// ---------------------------------------------------------------------------
// Exercise 14 (bonus) — Projection matrix
// ---------------------------------------------------------------------------

#[test]
fn ex14_projection_maps_near_and_far_to_ndc_z() {
    let (near, far) = (0.1_f32, 100.0_f32);
    let p = projection(std::f32::consts::FRAC_PI_2, 1.0, near, far);

    // A point on the near plane has w = -z = near and ndc z = 0.
    let s = p.mul_vec(&Vector::from([0., 0., -near, 1.]));
    approx(s.get(2).unwrap() / s.get(3).unwrap(), 0.0);

    // A point on the far plane has ndc z = 1.
    let s = p.mul_vec(&Vector::from([0., 0., -far, 1.]));
    approx(s.get(2).unwrap() / s.get(3).unwrap(), 1.0);

    // The homogeneous w component is -z.
    approx(s.get(3).unwrap(), far);
}

// ---------------------------------------------------------------------------
// Exercise 15 (bonus) — Complex vector spaces
// ---------------------------------------------------------------------------

#[test]
fn ex15_complex_field_is_generic() {
    use matrix::Complex;

    // Same `Vector` type, a different scalar field — the library code did not
    // change at all.
    let u = Vector::from([Complex::new(1., 1.), Complex::new(2., 0.)]);
    let v = Vector::from([Complex::new(1., -1.), Complex::new(0., 3.)]);

    // ⟨u|v⟩ = conj(1+i)(1-i) + conj(2)(3i) = -2i + 6i... computed correctly:
    assert_eq!(u.dot(&v), Complex::new(0., 4.));

    // |u| = sqrt(1² + 1² + 2²) = sqrt(6)
    approx(u.norm() as f32, 2.4494897);

    // cos = Re(⟨u|v⟩) / (|u||v|) = 0 / ... = 0
    approx(angle_cos(&u, &v) as f32, 0.0);

    // Complex matrix arithmetic: determinant of [[1, i], [-i, 1]] = 1 - (i·-i)
    // = 1 - 1 = 0.
    let a = Matrix::from([
        [Complex::new(1., 0.), Complex::new(0., 1.)],
        [Complex::new(0., -1.), Complex::new(1., 0.)],
    ]);
    assert_eq!(a.determinant(), Complex::new(0., 0.));
}
