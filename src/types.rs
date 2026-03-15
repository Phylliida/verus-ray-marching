use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{scale as scale3, dot as dot3, norm_sq as norm_sq3};
use verus_geometry::point3::*;
use verus_geometry::ray::ray_point_3d;

verus! {

// ---------------------------------------------------------------------------
// Ray
// ---------------------------------------------------------------------------

/// A 3D ray: origin + t * dir for t >= 0.
pub struct Ray3<T: Ring> {
    pub origin: Point3<T>,
    pub dir: Vec3<T>,
}

/// The point on a ray at parameter t.
pub open spec fn ray_at<T: Ring>(ray: Ray3<T>, t: T) -> Point3<T> {
    ray_point_3d(ray.origin, ray.dir, t)
}

// ---------------------------------------------------------------------------
// Hit record
// ---------------------------------------------------------------------------

/// Record of a ray-surface intersection.
pub struct HitRecord<T: Ring> {
    pub t: T,
    pub point: Point3<T>,
    pub normal: Vec3<T>,
    pub object_id: nat,
}

/// The hit point is actually on the ray at the recorded t value.
pub open spec fn hit_record_on_ray<T: Ring>(hit: HitRecord<T>, ray: Ray3<T>) -> bool {
    hit.point.eqv(ray_at(ray, hit.t))
}

/// The hit is in front of the ray origin.
pub open spec fn hit_record_forward<T: OrderedRing>(hit: HitRecord<T>) -> bool {
    hit.t.le(hit.t) && !hit.t.lt(T::zero()) // t >= 0
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

/// A sphere defined by center and squared radius (avoids sqrt in definition).
pub struct Sphere<T: Ring> {
    pub center: Point3<T>,
    pub radius_sq: T,
}

/// Sphere is valid: radius_sq >= 0 (nonnegative).
pub open spec fn sphere_valid<T: OrderedRing>(s: Sphere<T>) -> bool {
    !s.radius_sq.lt(T::zero()) // radius_sq >= 0
}

/// A point is on the sphere surface.
pub open spec fn point_on_sphere<T: Ring>(p: Point3<T>, s: Sphere<T>) -> bool {
    norm_sq3(sub3(p, s.center)).eqv(s.radius_sq)
}

// ---------------------------------------------------------------------------
// Plane
// ---------------------------------------------------------------------------

/// An infinite plane defined by a point on the plane and its normal.
pub struct Plane<T: Ring> {
    pub point: Point3<T>,
    pub normal: Vec3<T>,
}

/// A point lies on the plane: dot(p - plane.point, normal) ≡ 0.
pub open spec fn point_on_plane<T: Ring>(p: Point3<T>, pl: Plane<T>) -> bool {
    dot3(sub3(p, pl.point), pl.normal).eqv(T::zero())
}

// ---------------------------------------------------------------------------
// Axis-aligned bounding box
// ---------------------------------------------------------------------------

/// An axis-aligned bounding box.
pub struct Box3<T: OrderedRing> {
    pub min: Point3<T>,
    pub max: Point3<T>,
}

/// AABB is valid: min <= max on all axes.
pub open spec fn aabb_valid<T: OrderedRing>(b: Box3<T>) -> bool {
    &&& b.min.x.le(b.max.x)
    &&& b.min.y.le(b.max.y)
    &&& b.min.z.le(b.max.z)
}

/// A point is inside the AABB (inclusive).
pub open spec fn point_in_aabb<T: OrderedRing>(p: Point3<T>, b: Box3<T>) -> bool {
    &&& b.min.x.le(p.x) && p.x.le(b.max.x)
    &&& b.min.y.le(p.y) && p.y.le(b.max.y)
    &&& b.min.z.le(p.z) && p.z.le(b.max.z)
}

// ---------------------------------------------------------------------------
// Cylinder
// ---------------------------------------------------------------------------

/// An infinite cylinder defined by a base center, axis direction, and squared radius.
/// For a finite cylinder, additionally store height bounds.
pub struct Cylinder<T: Ring> {
    pub base_center: Point3<T>,
    pub axis_dir: Vec3<T>,
    pub radius_sq: T,
    pub half_height: T,
}

} // verus!
