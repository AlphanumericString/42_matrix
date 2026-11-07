# matrix — *Enter the Matrix*

A dependency-free, from-scratch linear algebra library in Rust for the 42 module
**"Enter the Matrix — An introduction to Linear Algebra"**. It implements every
exercise of the subject (00 → 13) plus both bonuses (14: projection matrix,
15: complex vector spaces).

```sh
cargo run      # walks through every exercise using the subject's own examples
cargo test     # 28 integration tests + 1 doctest, all green
```

---

## Table of contents

- [What is implemented](#what-is-implemented)
- [Big technical choices](#big-technical-choices)
  - [1. Generic over a `Field`, not hard-coded to `f32`](#1-generic-over-a-field-not-hard-coded-to-f32)
  - [2. Column-major storage](#2-column-major-storage)
  - [3. Fused multiply-add everywhere it is allowed](#3-fused-multiply-add-everywhere-it-is-allowed)
  - [4. No dependencies, no forbidden math functions](#4-no-dependencies-no-forbidden-math-functions)
  - [5. Numerical stability: partial pivoting and an epsilon](#5-numerical-stability-partial-pivoting-and-an-epsilon)
  - [6. Errors for the fallible operations](#6-errors-for-the-fallible-operations)
  - [7. Pure functions instead of in-place mutation where it helps](#7-pure-functions-instead-of-in-place-mutation-where-it-helps)
- [Complexity analysis](#complexity-analysis)
- [Project layout](#project-layout)
- [Testing strategy](#testing-strategy)

---

## What is implemented

| Exercise | Concept | API |
|---|---|---|
| 00 | Add, subtract, scale | `Vector::{add,sub,scl}`, `Matrix::{add,sub,scl}` |
| 01 | Linear combination | `linear_combination` |
| 02 | Linear interpolation | `lerp` (over the `Interpolate` trait) |
| 03 | Dot product | `Vector::dot` |
| 04 | Norms (1, 2, ∞) | `Vector::{norm_1,norm,norm_inf}` |
| 05 | Cosine of the angle | `angle_cos` |
| 06 | Cross product | `cross_product` |
| 07 | Linear map / matrix product | `Matrix::{mul_vec,mul_mat}` |
| 08 | Trace | `Matrix::trace` |
| 09 | Transpose | `Matrix::transpose` |
| 10 | Row-echelon form | `Matrix::row_echelon` |
| 11 | Determinant | `Matrix::determinant` |
| 12 | Inverse | `Matrix::inverse` |
| 13 | Rank | `Matrix::rank` |
| 14 ★ | Projection matrix | `projection` |
| 15 ★ | Complex vector spaces | `Complex` + the `Field` impl |

★ = bonus.

---

## Big technical choices

### 1. Generic over a `Field`, not hard-coded to `f32`

The subject insists, in three separate places, that the code be **generic** over
the scalar field `K`, and the final bonus asks to redo *everything* with
`K = C`. The naive approach is to write `f32` everywhere; that passes every
exercise except the bonus, which then requires rewriting all of the code.

Instead, the whole library is written against one trait, [`Field`](src/field.rs):

```rust
pub trait Field: Copy + /* + - * / , Neg, Assign forms */ {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(x: f64) -> Self;
    fn abs(self) -> f64;        // modulus, always real
    fn conj(self) -> Self { self }              // no-op outside C
    fn fma(self, b: Self, c: Self) -> Self { self * b + c }
    fn inv(self) -> Self { Self::one() / self }
    // ... neutral elements, epsilon, helpers
}
```

There are exactly two implementations:

- `impl Field for f32` — the required real field;
- `impl Field for Complex` — the bonus, added *without touching a single line of
  vector or matrix code*. This is the payoff: the bonus exercise becomes ~150
  lines of complex arithmetic instead of a second library.

This mirrors what the subject hints at:

> Generally, it should even be expected that the generic types provided would
> actually have some form of type constraint in a real implementation (for
> example, making a `Field` type, and having the generic `K` extend `Field`).

Two design details worth calling out:

- **`abs` returns `f64`, not `K`.** The absolute value of a complex number is
  real, and "norms always return real numbers, even for complex-valued vectors"
  (subject, exercise 04). `f64` is the common, lossless measurement type: every
  `f32` fits into an `f64` exactly.
- **`conj` exists for the inner product.** A complex inner product must be
  *sesquilinear* (`⟨u|v⟩ = Σ conj(uᵢ)·vᵢ`), otherwise `⟨u|u⟩` would not be a
  non-negative real number and the norm would break. On `R` conjugation is the
  identity, so one `dot` implementation is correct for both fields.

### 2. Column-major storage

> Unless specified otherwise, all matrices shown in the examples are in
> column-major order.

Matrices store their coefficients **column by column**: element `(i, j)` lives
at `data[j * rows + i]`. The reasons:

1. It is the layout the supplied projection-matrix renderer expects, and the
   subject explicitly warns that a matrix "in the wrong major order may appear to
   work at first glance, but will fail under careful scrutiny."
2. It is the layout of the GPU graphics standard (OpenGL/Vulkan), which this
   module is a warm-up for.

Human-written literals are still natural: `Matrix::from([[1., 2.], [3., 4.]])`
takes an array **of rows** and transposes into the internal buffer during
construction, so the rest of the code never worries about the layout.

### 3. Fused multiply-add everywhere it is allowed

The subject dedicates a whole section to the `vfmadd*ps` instructions and the
fused multiply-add. `Field::fma(a, b, c)` computes `a*b + c` with a **single
rounding step** (and, on x86 with FMA, a single instruction), which is both
faster and more accurate than separate `*` and `+`. It is used in:

- `linear_combination`, `Vector::dot`, `mul_vec`, `mul_mat` (all exercises that
  explicitly permit it),
- the elimination steps of `determinant` and `inverse`,
- the `lerp` implementations.

For `f32` it is backed by the standard library's `f32::mul_add`.

### 4. No dependencies, no forbidden math functions

`Cargo.toml` has **zero dependencies**, and the standard library's mathematical
functions are used only where the subject explicitly allows them:

| Where | Allowed | How we comply |
|---|---|---|
| Ex. 04 (norms) | `pow`, `max` | square root is `pow(x, 0.5)`; `max`/`|·|` via comparisons |
| Ex. 05 (cosine) | only FMA | `cos` is *never called*: the cosine is `dot/(‖u‖‖v‖)`, and the norms call the exercise-04 code |
| Ex. 14 (projection) | `tan`, FMA | `f32::tan` for the field-of-view |
| Others | none/FMA | only `+ - * /` and comparisons |

`f32::abs` is deliberately avoided; the absolute value is derived from a
comparison and a negation so that the "allowed functions" lists are respected to
the letter.

### 5. Numerical stability: partial pivoting and an epsilon

Row-echelon form, determinant and inverse all use **partial pivoting**: at each
step they choose the pivot with the largest modulus among the candidates.
Dividing by a tiny number amplifies floating-point error, so avoiding it
whenever possible is the standard stability technique. The subject anticipates
this:

> The results may be a bit approximate (within reason) since making the
> algorithm numerically stable can sometimes be difficult.

To decide whether a pivot is "zero" we use `Field::approx_zero`, which compares
the modulus against `epsilon = 1e-6`. An exact `== 0.0` test would miss a pivot
that is a rounding error rather than a true zero. Values that are merely *small*
are not clamped, however: only exact zeros are normalized so that a `-0.0`
never reaches the output.

### 6. Errors for the fallible operations

Operations with genuinely undefined inputs (`add` on mismatched shapes, `cross`
on non-3D vectors, …) follow the subject's contract and **panic** — the subject
calls these cases "undefined". But `inverse` is *defined* to report failure for a
singular matrix, so it returns a proper `Result` instead:

```rust
pub enum MatrixError { NotSquare, Singular }
```

`ex12_inverse_singular_errors` verifies that `[[1,2],[2,4]].inverse()` yields
`Err(Singular)` rather than garbage.

### 7. Pure functions instead of in-place mutation where it helps

The subject allows a choice between accumulator-style methods and pure functions:

> You may also implement both versions [...] We do ask some amount of coherence
> though.

We are coherent about it: the elementary, cheap, allocation-free operations
(`add`, `sub`, `scl`) mutate in place as the prototypes suggest, while the
operations that *produce a new object anyway* (`transpose`, `row_echelon`,
`determinant`, `inverse`, `rank`, `mul_*`) borrow `&self` and return the result.
This keeps the code readable and side-effect-free without violating any
prototype.

---

## Complexity analysis

The subject checks time/space complexity during peer review. Each algorithm meets
its budget:

| Operation | Time | Space (extra) |
|---|---|---|
| `add` / `sub` / `scl` (vector or matrix) | `O(n)` / `O(mn)` | `O(1)` |
| `linear_combination` | `O(Σ coords)` | `O(n)` |
| `lerp` | `O(n)` | `O(n)` |
| `dot` | `O(n)` | `O(1)` |
| `norm*` | `O(n)` | `O(1)` |
| `angle_cos` | `O(n)` | `O(1)` |
| `cross_product` | `O(1)` | `O(1)` |
| `mul_vec` (`m×n` by `n`) | `O(mn)` | `O(m)` |
| `mul_mat` (`m×n` by `n×p`) | `O(mnp)` | `O(mp)` |
| `transpose` | `O(mn)` | `O(mn)` |
| `row_echelon` | `O(mn·min(m,n))` ≤ `O(n³)` | `O(mn)` |
| `determinant` | `O(n³)` | `O(n²)` |
| `inverse` | `O(n³)` | `O(n²)` |
| `rank` | `O(mn·min(m,n))` ≤ `O(n³)` | `O(mn)` |

The determinant is implemented by Gaussian elimination rather than the
cofactor/Laplace expansion. Both are correct for the required `n ≤ 4`, but the
elimination method stays `O(n³)` for *any* `n`, whereas the cofactor approach is
`O(n!)` and would be a landmine.

---

## Project layout

```
src/
├── lib.rs       crate root, re-exports, `prelude`
├── field.rs     the `Field` trait + `impl for f32`, the `Interpolate` trait
├── complex.rs   the `Complex` bonus field
├── vector.rs    `Vector`, exercises 01–06
├── matrix.rs    `Matrix`, exercises 07–14
└── main.rs      a runnable demo of every exercise
tests/
└── exercises.rs 28 assertion-based tests, one or more per exercise
```

Everything is documented with `//!` and `///` doc comments; most non-obvious
lines explain *why* they are written that way (numeric stability, layout,
genericity, the allowed-functions constraints, the geometry behind the
determinant, …).

## Testing strategy

- **28 integration tests** in `tests/exercises.rs`, mirroring the subject's
  expected outputs, plus edge cases: non-square products, dimension mismatches
  (expected to panic), singular and non-square inverses, `transpose(transpose)=id`,
  and a projection matrix checked against the actual NDC mapping of the near and
  far planes.
- **1 doctest** in `lib.rs` showing the end-to-end usage.
- Floating-point results are compared with a tolerance, because the subject
  itself only guarantees results "approximate within reason" — e.g. its own 4×4
  determinant example is given as `1032`, while the computation yields
  `1031.9999`.
