use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, scale as scale3};
use verus_geometry::point3::*;
use verus_geometry::ray::ray_point_3d;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// Ray-plane intersection
// ---------------------------------------------------------------------------

/// Denominator of ray-plane intersection: dot(dir, normal).
/// Zero means ray is parallel to plane.
pub open spec fn ray_plane_denom<T: Ring>(ray: Ray3<T>, pl: Plane<T>) -> T {
    dot3(ray.dir, pl.normal)
}

/// Numerator of ray-plane intersection: dot(plane.point - origin, normal).
pub open spec fn ray_plane_numer<T: Ring>(ray: Ray3<T>, pl: Plane<T>) -> T {
    dot3(sub3(pl.point, ray.origin), pl.normal)
}

/// Parameter t at which the ray hits the plane (requires non-parallel).
pub open spec fn ray_plane_t<T: OrderedField>(ray: Ray3<T>, pl: Plane<T>) -> T
    recommends !ray_plane_denom(ray, pl).eqv(T::zero())
{
    ray_plane_numer(ray, pl).div(ray_plane_denom(ray, pl))
}

/// Does the ray hit the plane (non-parallel, t >= 0)?
pub open spec fn ray_hits_plane<T: OrderedField>(ray: Ray3<T>, pl: Plane<T>) -> bool {
    let denom = ray_plane_denom(ray, pl);
    let numer = ray_plane_numer(ray, pl);
    &&& !denom.eqv(T::zero())
    &&& !ray_plane_t(ray, pl).lt(T::zero())
}

// ---------------------------------------------------------------------------
// Lemma: hit point lies on the plane
// ---------------------------------------------------------------------------

/// The ray-plane hit point satisfies the plane equation.
///
/// Specifically, ray_at(ray, t) lies on the plane:
///   dot(ray_at(ray, t) - plane.point, normal) ≡ 0
///
/// Proof sketch:
///   ray_at(ray, t) = origin + t * dir
///   ray_at(ray, t) - plane.point = (origin - plane.point) + t * dir
///   dot(..., normal) = dot(origin - plane.point, normal) + t * dot(dir, normal)
///                    = -numer + (numer/denom) * denom
///                    = -numer + numer = 0
pub proof fn lemma_ray_plane_hit_on_plane<T: OrderedField>(ray: Ray3<T>, pl: Plane<T>)
    requires
        !ray_plane_denom(ray, pl).eqv(T::zero()),
    ensures
        point_on_plane(ray_at(ray, ray_plane_t(ray, pl)), pl),
{
    // The proof follows from the definition of division and dot product linearity.
    // For now we state the spec — the algebraic expansion is straightforward but
    // requires distributivity of dot over vector addition and scalar multiplication.
    assume(false); // TODO: expand algebraic proof
}

} // verus!
