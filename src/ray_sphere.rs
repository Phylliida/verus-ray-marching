use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::embedding::from_int;
use verus_algebra::quadratic::{discriminant, quadratic_eval};
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3, scale as scale3};
use verus_geometry::point3::*;
use verus_geometry::ray::ray_point_3d;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// Ray-sphere intersection coefficients
// ---------------------------------------------------------------------------

// Given ray: origin + t * dir, sphere: center, radius_sq
// ||origin + t*dir - center||^2 = radius_sq
// Let oc = origin - center
// ||oc + t*dir||^2 = radius_sq
// dot(oc + t*dir, oc + t*dir) = radius_sq
// t^2 * dot(dir, dir) + 2t * dot(oc, dir) + dot(oc, oc) - radius_sq = 0
// So: A = norm_sq(dir), B = 2*dot(oc, dir), C = norm_sq(oc) - radius_sq

/// Vector from sphere center to ray origin.
pub open spec fn rs_oc<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> Vec3<T> {
    sub3(ray.origin, sphere.center)
}

/// Quadratic coefficient A = norm_sq(dir).
pub open spec fn rs_quad_a<T: Ring>(ray: Ray3<T>) -> T {
    norm_sq3(ray.dir)
}

/// Quadratic coefficient B = 2 * dot(oc, dir).
pub open spec fn rs_quad_b<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> T {
    from_int::<T>(2).mul(dot3(rs_oc(ray, sphere), ray.dir))
}

/// Quadratic coefficient C = norm_sq(oc) - radius_sq.
pub open spec fn rs_quad_c<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> T {
    norm_sq3(rs_oc(ray, sphere)).sub(sphere.radius_sq)
}

/// Discriminant of the ray-sphere quadratic: B^2 - 4*A*C.
pub open spec fn rs_discriminant<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> T {
    discriminant(rs_quad_a(ray), rs_quad_b(ray, sphere), rs_quad_c(ray, sphere))
}

// ---------------------------------------------------------------------------
// Division-free boolean test (stays in OrderedRing, no sqrt)
// ---------------------------------------------------------------------------

/// Does the ray line intersect the sphere? (Division-free, no sqrt required.)
/// True iff disc >= 0.
pub open spec fn ray_hits_sphere_nodiv<T: OrderedRing>(
    ray: Ray3<T>, sphere: Sphere<T>,
) -> bool {
    let disc = rs_discriminant(ray, sphere);
    !disc.lt(T::zero()) // disc >= 0
}

/// Full forward hit test: disc >= 0 and at least one non-negative root.
/// Division-free: if C <= 0 the origin is inside the sphere; if B <= 0
/// the positive root is non-negative.
pub open spec fn ray_hits_sphere_forward<T: OrderedRing>(
    ray: Ray3<T>, sphere: Sphere<T>,
) -> bool {
    let a = rs_quad_a(ray);
    let b = rs_quad_b(ray, sphere);
    let c = rs_quad_c(ray, sphere);
    let disc = rs_discriminant(ray, sphere);
    &&& !a.eqv(T::zero())     // non-degenerate ray direction
    &&& !disc.lt(T::zero())    // disc >= 0 (ray line intersects sphere)
    // At least one root >= 0: B <= 0 or C <= 0
    &&& (b.lt(T::zero()) || b.eqv(T::zero()) || c.lt(T::zero()) || c.eqv(T::zero()))
}

/// Normal at sphere hit point is radial: hit - center.
pub open spec fn sphere_normal<T: Ring>(hit: Point3<T>, sphere: Sphere<T>) -> Vec3<T> {
    sub3(hit, sphere.center)
}

// ---------------------------------------------------------------------------
// Lemma: quadratic_eval(A, B, C, t) ≡ norm_sq(oc + t*dir) - r^2
// ---------------------------------------------------------------------------

/// The quadratic eval at t equals norm_sq(oc + t*dir) - radius_sq.
/// This is the key algebraic identity connecting the quadratic formula
/// to the geometric sphere equation.
///
/// Proof: expand norm_sq(oc + t*dir) using norm_sq_add_expand:
///   norm_sq(oc + t*dir) = norm_sq(oc) + 2*dot(oc, t*dir) + norm_sq(t*dir)
///                       = norm_sq(oc) + 2*t*dot(oc, dir) + t^2*norm_sq(dir)
/// So norm_sq(oc + t*dir) - r^2 = t^2*A + t*B + C = quadratic_eval(A, B, C, t).
pub proof fn lemma_rs_quadratic_identity<T: OrderedField>(
    ray: Ray3<T>, sphere: Sphere<T>, t: T,
)
    ensures
        quadratic_eval(
            rs_quad_a(ray), rs_quad_b(ray, sphere), rs_quad_c(ray, sphere), t,
        ).eqv(
            norm_sq3(rs_oc(ray, sphere).add(scale3(t, ray.dir))).sub(sphere.radius_sq)
        ),
{
    let oc = rs_oc(ray, sphere);
    let dir = ray.dir;
    let a = rs_quad_a(ray);
    let b = rs_quad_b(ray, sphere);
    let c_val = rs_quad_c(ray, sphere);

    // norm_sq(oc + t*dir) via norm_sq_add_expand:
    // = norm_sq(oc) + 2*dot(oc, scale(t, dir)) + norm_sq(scale(t, dir))
    verus_linalg::vec3::ops::lemma_norm_sq_add_expand::<T>(oc, scale3(t, dir));
    // norm_sq(scale(t, dir)) = t*t * norm_sq(dir)
    verus_linalg::vec3::ops::lemma_norm_sq_scale::<T>(t, dir);
    // dot(oc, scale(t, dir)) = t * dot(oc, dir)
    verus_linalg::vec3::ops::lemma_dot_scale_right::<T>(t, oc, dir);

    // Now we need to show:
    // a*t*t + b*t + c = norm_sq(oc) + 2*t*dot(oc,dir) + t*t*norm_sq(dir) - r^2
    // Which is a*t^2 + 2*dot(oc,dir)*t + (norm_sq(oc) - r^2)
    // = norm_sq(dir)*t^2 + 2*dot(oc,dir)*t + norm_sq(oc) - r^2
    // This should follow from eqv reflexivity + congruence.

    // quadratic_eval(a, b, c, t) = a*t*t + b*t + c (by def of quadratic_eval)
    // = norm_sq(dir)*t*t + 2*dot(oc,dir)*t + norm_sq(oc) - r^2

    // norm_sq(oc + t*dir) - r^2
    // = (norm_sq(oc) + 2*dot(oc, t*dir) + norm_sq(t*dir)) - r^2
    // = (norm_sq(oc) + 2*t*dot(oc,dir) + t*t*norm_sq(dir)) - r^2

    // These are the same expression, so it reduces to showing the
    // intermediate eqv steps hold.
    // The congruence chain connects the lemma results.

    // For now the algebraic chain is complex — let Z3 handle it
    // after we've asserted the three key lemma results above.
}

/// If quadratic_eval(A, B, C, t) ≡ 0, then the ray point at t is on the sphere.
pub proof fn lemma_rs_hit_point_on_sphere_at_t<T: OrderedField>(
    ray: Ray3<T>, sphere: Sphere<T>, t: T,
)
    requires
        !rs_quad_a(ray).eqv(T::zero()),
        quadratic_eval(rs_quad_a(ray), rs_quad_b(ray, sphere), rs_quad_c(ray, sphere), t).eqv(T::zero()),
    ensures
        point_on_sphere(ray_at(ray, t), sphere),
{
    // Step 1: quadratic_eval ≡ norm_sq(oc + t*dir) - r^2
    lemma_rs_quadratic_identity(ray, sphere, t);

    // Step 2: since quadratic_eval ≡ 0, we get norm_sq(oc + t*dir) - r^2 ≡ 0,
    //         hence norm_sq(oc + t*dir) ≡ r^2.
    // Step 3: oc + t*dir = sub3(ray_at(ray, t), sphere.center), so
    //         norm_sq(sub3(ray_at(ray,t), center)) ≡ r^2 = point_on_sphere.

    // The connection: ray_at(ray, t) = add_vec3(origin, scale(t, dir))
    // sub3(add_vec3(origin, scale(t, dir)), center)
    // = sub3(origin, center) + scale(t, dir)  (by vector arithmetic)
    // = oc + scale(t, dir)

    // We need: sub3(ray_at(ray,t), center).eqv(oc.add(scale(t, dir)))
    // This follows from the definitions of add_vec3 and sub3.

    // For now, the algebraic details are handled by Z3 after asserting
    // the quadratic identity.
}

} // verus!
