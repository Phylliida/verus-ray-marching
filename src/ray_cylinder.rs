use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::embedding::from_int;
use verus_algebra::quadratic::discriminant;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3, scale as scale3};
use verus_geometry::point3::*;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// Ray-cylinder intersection
// ---------------------------------------------------------------------------

// Infinite cylinder: points p such that ||p - base - dot(p - base, axis) * axis||^2 = r^2
// (distance from point to axis line equals radius)
//
// Substituting ray: p = origin + t * dir
// Let d = origin - base_center
// Let proj_d = d - dot(d, axis) * axis     (d projected perpendicular to axis)
// Let proj_dir = dir - dot(dir, axis) * axis (dir projected perpendicular to axis)
// Then: ||proj_d + t * proj_dir||^2 = r^2
// Which gives quadratic: A*t^2 + B*t + C = 0
// A = norm_sq(proj_dir)
// B = 2 * dot(proj_d, proj_dir)
// C = norm_sq(proj_d) - r^2

/// Vector from cylinder base to ray origin.
pub open spec fn rc_offset<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>) -> Vec3<T> {
    sub3(ray.origin, cyl.base_center)
}

/// Ray direction projected perpendicular to cylinder axis.
pub open spec fn rc_perp_dir<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>) -> Vec3<T> {
    let d_along = dot3(ray.dir, cyl.axis_dir);
    ray.dir.sub(scale3(d_along, cyl.axis_dir))
}

/// Offset vector projected perpendicular to cylinder axis.
pub open spec fn rc_perp_offset<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>) -> Vec3<T> {
    let oc = rc_offset(ray, cyl);
    let oc_along = dot3(oc, cyl.axis_dir);
    oc.sub(scale3(oc_along, cyl.axis_dir))
}

/// Quadratic coefficient A for cylinder intersection.
pub open spec fn rc_quad_a<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>) -> T {
    norm_sq3(rc_perp_dir(ray, cyl))
}

/// Quadratic coefficient B for cylinder intersection.
pub open spec fn rc_quad_b<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>) -> T {
    from_int::<T>(2).mul(dot3(rc_perp_offset(ray, cyl), rc_perp_dir(ray, cyl)))
}

/// Quadratic coefficient C for cylinder intersection.
pub open spec fn rc_quad_c<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>) -> T {
    norm_sq3(rc_perp_offset(ray, cyl)).sub(cyl.radius_sq)
}

/// Discriminant of the ray-cylinder quadratic.
pub open spec fn rc_discriminant<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>) -> T {
    discriminant(rc_quad_a(ray, cyl), rc_quad_b(ray, cyl), rc_quad_c(ray, cyl))
}

/// Does the ray hit the infinite cylinder? (Division-free test.)
pub open spec fn ray_hits_inf_cylinder<T: OrderedRing>(
    ray: Ray3<T>, cyl: Cylinder<T>,
) -> bool {
    !rc_discriminant(ray, cyl).lt(T::zero())
}

/// Height along axis at parameter t: dot(offset + t*dir, axis).
pub open spec fn rc_height_at_t<T: Ring>(ray: Ray3<T>, cyl: Cylinder<T>, t: T) -> T {
    let oc = rc_offset(ray, cyl);
    dot3(oc, cyl.axis_dir).add(t.mul(dot3(ray.dir, cyl.axis_dir)))
}

/// Does the height at t satisfy the finite cylinder bounds?
/// We use symmetric bounds: |height| <= half_height.
pub open spec fn rc_height_in_bounds<T: OrderedRing>(
    ray: Ray3<T>, cyl: Cylinder<T>, t: T,
) -> bool {
    let h = rc_height_at_t(ray, cyl, t);
    !cyl.half_height.lt(h) && !h.lt(cyl.half_height.neg()) // -half_height <= h <= half_height
}

} // verus!
