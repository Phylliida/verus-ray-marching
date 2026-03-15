use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3, scale as scale3};
use verus_geometry::point3::*;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// Unified normal computation for all primitives
// ---------------------------------------------------------------------------

/// Normal at a sphere hit point: hit - center (unnormalized, outward).
pub open spec fn normal_sphere<T: Ring>(hit: Point3<T>, sphere: Sphere<T>) -> Vec3<T> {
    sub3(hit, sphere.center)
}

/// Normal at a plane hit point: the plane's stored normal.
pub open spec fn normal_plane<T: Ring>(pl: Plane<T>) -> Vec3<T> {
    pl.normal
}

/// Normal at a box hit point: axis-aligned face normal.
/// face: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
pub open spec fn normal_box<T: Ring>(face: nat) -> Vec3<T> {
    if face == 0 {
        Vec3 { x: T::one(), y: T::zero(), z: T::zero() }
    } else if face == 1 {
        Vec3 { x: T::one().neg(), y: T::zero(), z: T::zero() }
    } else if face == 2 {
        Vec3 { x: T::zero(), y: T::one(), z: T::zero() }
    } else if face == 3 {
        Vec3 { x: T::zero(), y: T::one().neg(), z: T::zero() }
    } else if face == 4 {
        Vec3 { x: T::zero(), y: T::zero(), z: T::one() }
    } else {
        Vec3 { x: T::zero(), y: T::zero(), z: T::one().neg() }
    }
}

/// Normal at a cylinder hit point: radial component perpendicular to axis.
/// normal = (hit - base) - dot(hit - base, axis) * axis
pub open spec fn normal_cylinder<T: Ring>(hit: Point3<T>, cyl: Cylinder<T>) -> Vec3<T> {
    let d = sub3(hit, cyl.base_center);
    let along = dot3(d, cyl.axis_dir);
    d.sub(scale3(along, cyl.axis_dir))
}

// ---------------------------------------------------------------------------
// Lemmas
// ---------------------------------------------------------------------------

/// Sphere normal is radial: normal_sphere(hit, sphere) = hit - center.
/// This is definitionally true (no proof needed).

/// Cylinder normal is perpendicular to axis:
/// dot(normal_cylinder(hit, cyl), cyl.axis_dir) ≡ 0
/// when axis is a unit vector (norm_sq(axis) ≡ 1).
pub proof fn lemma_cylinder_normal_perp_to_axis<T: OrderedField>(
    hit: Point3<T>, cyl: Cylinder<T>,
)
    requires
        norm_sq3(cyl.axis_dir).eqv(T::one()),
    ensures
        dot3(normal_cylinder(hit, cyl), cyl.axis_dir).eqv(T::zero()),
{
    // dot(d - dot(d,a)*a, a) = dot(d,a) - dot(d,a)*dot(a,a) = dot(d,a) - dot(d,a)*1 = 0
    assume(false); // TODO: bilinearity expansion
}

} // verus!
