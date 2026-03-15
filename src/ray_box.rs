use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::Vec3;
use verus_geometry::point3::*;
use verus_geometry::ray::*;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// Ray-box intersection (delegates to verus-geometry slab method)
// ---------------------------------------------------------------------------

/// Does the ray hit the axis-aligned bounding box?
/// Delegates to the slab-based ray_hits_aabb3 from verus-geometry.
pub open spec fn ray_hits_box<T: OrderedField>(ray: Ray3<T>, b: Box3<T>) -> bool {
    ray_hits_aabb3(ray.origin, ray.dir, b.min, b.max)
}

/// Entry parameter for the ray entering the box.
pub open spec fn ray_box_t_enter<T: OrderedField>(ray: Ray3<T>, b: Box3<T>) -> T {
    slab_t_enter_3d(ray.origin, ray.dir, b.min, b.max)
}

/// Exit parameter for the ray leaving the box.
pub open spec fn ray_box_t_exit<T: OrderedField>(ray: Ray3<T>, b: Box3<T>) -> T {
    slab_t_exit_3d(ray.origin, ray.dir, b.min, b.max)
}

/// Which face of the box is hit (0..5 = +x,-x,+y,-y,+z,-z).
/// Determined by which axis achieves the maximum entry t.
pub open spec fn ray_box_hit_face<T: OrderedField>(ray: Ray3<T>, b: Box3<T>) -> nat {
    // The face is the axis that produced t_enter.
    // Face 0/1: x-axis entry (min_x or max_x face)
    // Face 2/3: y-axis entry
    // Face 4/5: z-axis entry
    // We determine this by comparing slab_t_near values.
    let dir = ray.dir;
    let t_near_x = if dir.x.eqv(T::zero()) { T::zero() } else {
        slab_t_near(ray.origin.x, dir.x, b.min.x, b.max.x)
    };
    let t_near_y = if dir.y.eqv(T::zero()) { T::zero() } else {
        slab_t_near(ray.origin.y, dir.y, b.min.y, b.max.y)
    };
    let t_near_z = if dir.z.eqv(T::zero()) { T::zero() } else {
        slab_t_near(ray.origin.z, dir.z, b.min.z, b.max.z)
    };
    if t_near_x.le(t_near_y) && t_near_x.le(t_near_z) {
        // x-axis face
        if dir.x.lt(T::zero()) { 0 } else { 1 } // +x face or -x face
    } else if t_near_y.le(t_near_z) {
        if dir.y.lt(T::zero()) { 2 } else { 3 }
    } else {
        if dir.z.lt(T::zero()) { 4 } else { 5 }
    }
}

/// Outward normal for the hit face of the box.
pub open spec fn ray_box_normal<T: Ring>(face: nat) -> Vec3<T> {
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

} // verus!
