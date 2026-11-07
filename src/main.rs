//! Demo binary for the `matrix` crate.
//!
//! Running `cargo run` walks through every exercise of the subject using the
//! exact examples from the subject, printing the results. `cargo test` runs the
//! assertion-based test-suite (see `tests/` and the unit tests in each module).

use matrix::prelude::*;

fn main() {
    ex00_add_sub_scale();
    ex01_linear_combination();
    ex02_lerp();
    ex03_dot();
    ex04_norms();
    ex05_angle_cos();
    ex06_cross_product();
    ex07_mul();
    ex08_trace();
    ex09_transpose();
    ex10_row_echelon();
    ex11_determinant();
    ex12_inverse();
    ex13_rank();
    ex14_projection();
    ex15_complex();
}

/// Exercise 00 — add, subtract and scale vectors and matrices.
fn ex00_add_sub_scale() {
    section("Exercise 00 — Add, Subtract and Scale");

    let mut u = Vector::from([2., 3.]);
    u.add(&Vector::from([5., 7.]));
    println!("add  ->\n{}", u);

    let mut u = Vector::from([2., 3.]);
    u.sub(&Vector::from([5., 7.]));
    println!("sub  ->\n{}", u);

    let mut u = Vector::from([2., 3.]);
    u.scl(2.);
    println!("scl  ->\n{}", u);

    let mut m = Matrix::from([[1., 2.], [3., 4.]]);
    m.add(&Matrix::from([[7., 4.], [-2., 2.]]));
    println!("matrix add ->\n{}", m);

    let mut m = Matrix::from([[1., 2.], [3., 4.]]);
    m.sub(&Matrix::from([[7., 4.], [-2., 2.]]));
    println!("matrix sub ->\n{}", m);

    let mut m = Matrix::from([[1., 2.], [3., 4.]]);
    m.scl(2.);
    println!("matrix scl ->\n{}", m);
}

/// Exercise 01 — linear combination.
fn ex01_linear_combination() {
    section("Exercise 01 — Linear combination");

    let e1 = Vector::from([1., 0., 0.]);
    let e2 = Vector::from([0., 1., 0.]);
    let e3 = Vector::from([0., 0., 1.]);
    println!(
        "{}",
        linear_combination(&[e1, e2, e3], &[10., -2., 0.5])
    );

    let v1 = Vector::from([1., 2., 3.]);
    let v2 = Vector::from([0., 10., -100.]);
    println!("{}", linear_combination(&[v1, v2], &[10., -2.]));
}

/// Exercise 02 — linear interpolation of scalars, vectors and matrices.
fn ex02_lerp() {
    section("Exercise 02 — Linear interpolation");

    println!("{}", lerp(0., 1., 0.));
    println!("{}", lerp(0., 1., 1.));
    println!("{}", lerp(0., 1., 0.5));
    println!("{}", lerp(21., 42., 0.3));

    let r = lerp(Vector::from([2., 1.]), Vector::from([4., 2.]), 0.3);
    println!("vector ->\n{}", r);

    let r = lerp(
        Matrix::from([[2., 1.], [3., 4.]]),
        Matrix::from([[20., 10.], [30., 40.]]),
        0.5,
    );
    println!("matrix ->\n{}", r);
}

/// Exercise 03 — dot product.
fn ex03_dot() {
    section("Exercise 03 — Dot product");

    println!("{}", Vector::from([0., 0.]).dot(&Vector::from([1., 1.])));
    println!("{}", Vector::from([1., 1.]).dot(&Vector::from([1., 1.])));
    println!("{}", Vector::from([-1., 6.]).dot(&Vector::from([3., 2.])));
}

/// Exercise 04 — the 1-, 2- and ∞-norms.
fn ex04_norms() {
    section("Exercise 04 — Norms");

    for u in [
        Vector::from([0., 0., 0.]),
        Vector::from([1., 2., 3.]),
        Vector::from([-1., -2.]),
    ] {
        println!("{}, {}, {}", u.norm_1(), u.norm(), u.norm_inf());
    }
}

/// Exercise 05 — cosine of the angle between two vectors.
fn ex05_angle_cos() {
    section("Exercise 05 — Cosine");

    let cases = [
        (Vector::from([1., 0.]), Vector::from([1., 0.])),
        (Vector::from([1., 0.]), Vector::from([0., 1.])),
        (Vector::from([-1., 1.]), Vector::from([1., -1.])),
        (Vector::from([2., 1.]), Vector::from([4., 2.])),
        (Vector::from([1., 2., 3.]), Vector::from([4., 5., 6.])),
    ];
    for (u, v) in cases {
        println!("{}", angle_cos(&u, &v));
    }
}

/// Exercise 06 — cross product.
fn ex06_cross_product() {
    section("Exercise 06 — Cross product");

    let cases = [
        (Vector::from([0., 0., 1.]), Vector::from([1., 0., 0.])),
        (Vector::from([1., 2., 3.]), Vector::from([4., 5., 6.])),
        (Vector::from([4., 2., -3.]), Vector::from([-2., -5., 16.])),
    ];
    for (u, v) in cases {
        println!("{}\n---", cross_product(&u, &v));
    }
}

/// Exercise 07 — matrix × vector and matrix × matrix.
fn ex07_mul() {
    section("Exercise 07 — Linear map, matrix multiplication");

    let cases = [
        (Matrix::from([[1., 0.], [0., 1.]]), Vector::from([4., 2.])),
        (Matrix::from([[2., 0.], [0., 2.]]), Vector::from([4., 2.])),
        (Matrix::from([[2., -2.], [-2., 2.]]), Vector::from([4., 2.])),
    ];
    for (m, v) in cases {
        println!("{}\n---", m.mul_vec(&v));
    }

    let cases = [
        (Matrix::from([[1., 0.], [0., 1.]]), Matrix::from([[1., 0.], [0., 1.]])),
        (Matrix::from([[1., 0.], [0., 1.]]), Matrix::from([[2., 1.], [4., 2.]])),
        (Matrix::from([[3., -5.], [6., 8.]]), Matrix::from([[2., 1.], [4., 2.]])),
    ];
    for (a, b) in cases {
        println!("{}\n---", a.mul_mat(&b));
    }
}

/// Exercise 08 — trace.
fn ex08_trace() {
    section("Exercise 08 — Trace");

    println!("{}", Matrix::from([[1., 0.], [0., 1.]]).trace());
    println!(
        "{}",
        Matrix::from([[2., -5., 0.], [4., 3., 7.], [-2., 3., 4.]]).trace()
    );
    println!(
        "{}",
        Matrix::from([[-2., -8., 4.], [1., -23., 4.], [0., 6., 4.]]).trace()
    );
}

/// Exercise 09 — transpose.
fn ex09_transpose() {
    section("Exercise 09 — Transpose");

    let m = Matrix::from([[1., 2., 3.], [4., 5., 6.]]);
    println!("original:\n{}\ntransposed:\n{}", m, m.transpose());
}

/// Exercise 10 — reduced row-echelon form.
fn ex10_row_echelon() {
    section("Exercise 10 — Row-echelon form");

    let cases = [
        Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        Matrix::from([[1., 2.], [3., 4.]]),
        Matrix::from([[1., 2.], [2., 4.]]),
        Matrix::from([
            [8., 5., -2., 4., 28.],
            [4., 2.5, 20., 4., -4.],
            [8., 5., 1., 4., 17.],
        ]),
    ];
    for m in cases {
        println!("{}\n---", m.row_echelon());
    }
}

/// Exercise 11 — determinant.
fn ex11_determinant() {
    section("Exercise 11 — Determinant");

    println!("{}", Matrix::from([[1., -1.], [-1., 1.]]).determinant());
    println!("{}", Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]).determinant());
    println!(
        "{}",
        Matrix::from([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]).determinant()
    );
    println!(
        "{}",
        Matrix::from([
            [8., 5., -2., 4.],
            [4., 2.5, 20., 4.],
            [8., 5., 1., 4.],
            [28., -4., 17., 1.],
        ])
        .determinant()
    );
}

/// Exercise 12 — inverse.
fn ex12_inverse() {
    section("Exercise 12 — Inverse");

    let cases = [
        Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        Matrix::from([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]]),
        Matrix::from([[8., 5., -2.], [4., 7., 20.], [7., 6., 1.]]),
    ];
    for m in cases {
        match m.inverse() {
            Ok(inv) => println!("{}\n---", inv),
            Err(e) => println!("error: {}", e),
        }
    }

    // A singular matrix must report an error rather than produce garbage.
    let singular = Matrix::from([[1., 2.], [2., 4.]]);
    println!("singular -> {:?}", singular.inverse().err());
}

/// Exercise 13 — rank.
fn ex13_rank() {
    section("Exercise 13 — Rank");

    println!("{}", Matrix::from([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]).rank());
    println!(
        "{}",
        Matrix::from([[1., 2., 0., 0.], [2., 4., 0., 0.], [-1., 2., 1., 1.]]).rank()
    );
    println!(
        "{}",
        Matrix::from([
            [8., 5., -2.],
            [4., 7., 20.],
            [7., 6., 1.],
            [21., 18., 7.],
        ])
        .rank()
    );
}

/// Exercise 14 (bonus) — the projection matrix.
fn ex14_projection() {
    section("Exercise 14 (bonus) — Projection matrix");

    let p = projection(std::f32::consts::FRAC_PI_2, 16.0 / 9.0, 0.1, 100.0);
    println!("{}", p);
}

/// Exercise 15 (bonus) — the same library over the complex field.
fn ex15_complex() {
    section("Exercise 15 (bonus) — Complex vector spaces");

    let u = Vector::from([Complex::new(1., 1.), Complex::new(2., 0.)]);
    let v = Vector::from([Complex::new(1., -1.), Complex::new(0., 3.)]);
    println!("dot  -> {}", u.dot(&v));
    println!("norm -> {}", u.norm());
    println!("cos  -> {}", angle_cos(&u, &v));

    let a = Matrix::from([
        [Complex::new(1., 0.), Complex::new(0., 1.)],
        [Complex::new(0., -1.), Complex::new(1., 0.)],
    ]);
    println!("det  -> {}", a.determinant());
    println!("rank -> {}", a.rank());
}

/// Small printing helper so the demo output stays readable.
fn section(title: &str) {
    println!("\n===== {} =====", title);
}
