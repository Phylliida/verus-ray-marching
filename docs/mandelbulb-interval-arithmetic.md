# Infinite Zoom on the Mandelbulb via Interval Arithmetic

## The Problem

The 2D Mandelbrot set supports infinite zoom with exact arithmetic because the
iteration `z_{n+1} = z_n^2 + c` is purely polynomial — every step is just
addition and multiplication of complex numbers. This works perfectly with exact
rationals (verus-mandelbrot) or multi-precision integers on GPU.

The 3D Mandelbulb uses the "triplex power" formula, which converts to spherical
coordinates, applies the power, and converts back:

```
r     = |z| = sqrt(x^2 + y^2 + z^2)
theta = acos(y / r)
phi   = atan2(z, x)

z^n = r^n * ( sin(n*theta)*cos(n*phi),
              cos(n*theta),
              sin(n*theta)*sin(n*phi) )
```

This involves transcendental functions (acos, atan2, sin, cos) that cannot be
computed exactly in rational arithmetic. The verus-interval-arithmetic library
has no interval versions of these functions.

## The Solution: Polynomial Triplex Power via Chebyshev Identities

For a **fixed integer power n**, the triplex power can be expressed as a
polynomial in Cartesian coordinates (x, y, z) without any trigonometric
functions. The key identities:

### Chebyshev Polynomials

The Chebyshev polynomial of the first kind T_n satisfies:
```
cos(n*theta) = T_n(cos(theta))
```

The Chebyshev polynomial of the second kind U_n satisfies:
```
sin(n*theta) = U_{n-1}(cos(theta)) * sin(theta)
```

Both T_n and U_n are polynomials with integer coefficients.

### Substitution

In the triplex power formula:
- `cos(theta) = y / r`  (rational in coordinates)
- `sin(theta) = sqrt(x^2 + z^2) / r`  (involves one sqrt)
- `cos(phi) = x / sqrt(x^2 + z^2)`
- `sin(phi) = z / sqrt(x^2 + z^2)`

Let `s = sqrt(x^2 + z^2)` (the "horizontal radius"). Then:

```
cos(n*theta) = T_n(y/r)
sin(n*theta) = U_{n-1}(y/r) * s/r

cos(n*phi) and sin(n*phi) use the same Chebyshev trick on phi:
  cos(phi) = x/s,  sin(phi) = z/s
  cos(n*phi) = T_n(x/s)    -- but this still has sqrt
  sin(n*phi) = U_{n-1}(x/s) * z/s
```

### Eliminating the Square Root

The full triplex power expression, after multiplying through by r^n:

```
triplex_power(x,y,z, n) = (
    r^n * U_{n-1}(y/r) * (s/r) * T_n(x/s),
    r^n * T_n(y/r),
    r^n * U_{n-1}(y/r) * (s/r) * U_{n-1}(x/s) * (z/s)
)
```

When you expand T_n(y/r) and U_{n-1}(y/r), the powers of r in the denominators
cancel with the r^n factor. Similarly, powers of s cancel. The result is a
**homogeneous polynomial** in (x, y, z) of degree n.

For the x-component (using de Moivre on both theta and phi):
```
x_out = sum over k,j of [coefficients] * x^a * y^b * z^c * (x^2+z^2)^d
```

All terms are polynomial in (x, y, z). No sqrt, no trig.

### Example: n = 2

```
T_2(t) = 2t^2 - 1
U_1(t) = 2t

triplex_power(x,y,z, 2) = (
    2*x*y,
    2*y^2 - x^2 - z^2,
    2*y*z
)
```

This is the "Mandelbulb power-2" (which produces a boring shape — it's
topologically equivalent to the 2D Mandelbrot rotated around the y-axis).

### Example: n = 8 (Classic Mandelbulb)

The Chebyshev polynomials for n=8:
```
T_8(t) = 128t^8 - 256t^6 + 160t^4 - 32t^2 + 1
U_7(t) = 256t^7 - 448t^5 + 240t^3 - 48t
```

The full expansion yields a degree-8 homogeneous polynomial in (x, y, z) with
~50 terms per component. Large but entirely algebraic.

## Architecture for Verified Infinite Zoom

### What's Already Available

| Component | Status | Location |
|-----------|--------|----------|
| 1D interval arithmetic | Done | verus-interval-arithmetic |
| 2D complex intervals | Done | verus-mandelbrot/complex_interval.rs |
| Interval add/sub/mul/square/pow | Done | verus-interval-arithmetic |
| Dyadic reduction (precision control) | Done | verus-interval-arithmetic |
| Escape verification (certainly_lt) | Done | verus-interval-arithmetic |
| Mandelbrot runtime pipeline | Done | verus-mandelbrot/runtime_mandelbrot.rs |

### What's Needed

1. **3D Interval Boxes** — `Box3DInterval { x, y, z: Interval }`
   - Component-wise add, sub, scale
   - Cross-term mul for polynomial evaluation
   - Magnitude squared: `x.square() + y.square() + z.square()`
   - Pattern: copy ComplexInterval to 3 components

2. **Polynomial Triplex Power** — for a chosen n (e.g., 8)
   - Derive Cartesian polynomial form using Chebyshev expansion
   - Implement as interval polynomial evaluation
   - Verify: `triplex_poly_n(box) contains triplex_power(p)` for all p in box

3. **Escape Verification** — same as 2D
   - `certainly_gt(magnitude_squared(z), bailout_squared)`
   - Already supported by interval comparison operations

4. **Precision Management** — dyadic reduction after each iteration
   - Same strategy as 2D: reduce denominators to powers of 2
   - Keeps rational denominators bounded during deep zoom

### Verification Chain

```
  Point p in pixel box B
  → p in interval box [x_lo,x_hi] x [y_lo,y_hi] x [z_lo,z_hi]
  → triplex_poly_8(p) in triplex_poly_8(B)    [interval containment]
  → iterate(p, k) in iterate(B, k)             [induction on k]
  → |iterate(p,k)|^2 in magnitude_sq(iterate(B,k))
  → certainly_gt(magnitude_sq, bailout)         [proven escape]
  → p is NOT in the Mandelbulb                  [soundness]
```

For pixels where escape cannot be proven within max_iters, we color them as
"possibly in set" — same as 2D.

## Practical Considerations

### Performance

The n=8 polynomial has ~50 terms per component × 3 components = ~150
multiplications per iteration step (vs. 2 for 2D Mandelbrot). Each
multiplication on intervals requires comparing 4 products. So roughly
**300x more arithmetic per iteration** than 2D Mandelbrot.

For GPU implementation, this is still tractable — modern GPUs can handle
hundreds of multiply-adds per pixel per frame. But deep zooms (high iteration
counts) will be slower than 2D.

### Precision

Each iteration step multiplies the number of significant digits by n (=8).
After k iterations, you need ~8^k digits of precision to resolve fine detail.
This is the same exponential growth as 2D (where it's 2^k), but faster.

For GPU multi-precision (N ints per value), each zoom doubling requires
roughly one more int of precision. Practical limit is N=8-16 ints (256-512
bits), giving ~77-154 decimal digits — enough for zoom depths of 10^77 to
10^154.

### Alternative: Perturbation Theory

For extremely deep zooms (beyond 10^100), perturbation theory computes
`delta_n = z_n - Z_n` where Z_n is a high-precision reference orbit computed
on CPU. The GPU then only needs to track the small perturbation delta_n in
standard precision. This technique is well-established for 2D Mandelbrot
and extends naturally to 3D if the iteration is polynomial.
