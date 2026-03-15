use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::embedding::from_int;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{scale as scale3};
use verus_geometry::point3::*;
use crate::types::*;
use crate::ray_box::*;

verus! {

// ---------------------------------------------------------------------------
// Affine transform (uniform scale + translate)
// ---------------------------------------------------------------------------

/// A similarity transform: scale then translate.
/// Maps point p to scale * p + translate.
pub struct AffineTransform<T: Ring> {
    pub scale: T,
    pub translate: Vec3<T>,
}

/// Apply the transform to a point.
pub open spec fn transform_point<T: Ring>(tf: AffineTransform<T>, p: Point3<T>) -> Point3<T> {
    let scaled = Point3 {
        x: tf.scale.mul(p.x),
        y: tf.scale.mul(p.y),
        z: tf.scale.mul(p.z),
    };
    add_vec3(scaled, tf.translate)
}

/// Apply the transform to an AABB.
pub open spec fn transform_aabb<T: OrderedRing>(tf: AffineTransform<T>, b: Box3<T>) -> Box3<T> {
    Box3 {
        min: transform_point(tf, b.min),
        max: transform_point(tf, b.max),
    }
}

/// Transform a ray into a child's local coordinate system.
/// If the child transform is: child_point = scale * local + translate,
/// then local = (child_point - translate) / scale.
/// So the ray in local coords: origin' = (origin - translate) / scale, dir' = dir / scale.
/// But since we only care about intersection (t is invariant under uniform scaling of both
/// origin and dir), we can use: origin' = (origin - translate) / scale, dir' = dir.
/// Actually, for exact intersection we need:
///   origin_local = (origin - translate) * recip(scale)
///   dir_local = dir * recip(scale)
/// But t is the same either way if both are scaled.
/// Simpler: origin_local = (origin - translate), dir_local = dir, then compare against
/// the scaled AABB. This avoids division entirely.
pub open spec fn inverse_transform_ray<T: OrderedField>(
    tf: AffineTransform<T>, ray: Ray3<T>,
) -> Ray3<T>
    recommends !tf.scale.eqv(T::zero())
{
    let inv_s = tf.scale.recip();
    let shifted = sub3(ray.origin, Point3 { x: tf.translate.x, y: tf.translate.y, z: tf.translate.z });
    Ray3 {
        origin: Point3 {
            x: inv_s.mul(shifted.x),
            y: inv_s.mul(shifted.y),
            z: inv_s.mul(shifted.z),
        },
        dir: Vec3 {
            x: inv_s.mul(ray.dir.x),
            y: inv_s.mul(ray.dir.y),
            z: inv_s.mul(ray.dir.z),
        },
    }
}

// ---------------------------------------------------------------------------
// Fractal description
// ---------------------------------------------------------------------------

/// A self-similar fractal described by an IFS (iterated function system).
pub struct FractalDesc<T: OrderedField> {
    pub transforms: Seq<AffineTransform<T>>,
    pub base_aabb: Box3<T>,
}

/// The geometry of a fractal at depth N is the union of all leaf AABBs
/// obtained by recursively applying transforms.
pub open spec fn fractal_leaf_aabbs<T: OrderedField>(
    desc: FractalDesc<T>, depth: nat,
) -> Seq<Box3<T>>
    decreases depth,
{
    if depth == 0 {
        seq![desc.base_aabb]
    } else {
        let children = fractal_leaf_aabbs(desc, (depth - 1) as nat);
        // Each child AABB gets all transforms applied
        Seq::new(
            (desc.transforms.len() * children.len()) as nat,
            |i: int| {
                let tf_idx = i / children.len() as int;
                let child_idx = i % children.len() as int;
                transform_aabb(desc.transforms[tf_idx], children[child_idx])
            },
        )
    }
}

// ---------------------------------------------------------------------------
// Recursive ray-fractal intersection
// ---------------------------------------------------------------------------

/// Does the ray hit any leaf of the fractal at the given depth?
/// Uses AABB pruning at each level of recursion.
pub open spec fn ray_hits_fractal<T: OrderedField>(
    ray: Ray3<T>, desc: FractalDesc<T>, depth: nat,
) -> bool
    decreases depth, 0nat,
{
    if depth == 0 {
        // At leaf: test against base AABB
        ray_hits_box(ray, desc.base_aabb)
    } else {
        // Test each child: transform ray into child's local frame, recurse
        exists|i: int| 0 <= i < desc.transforms.len() &&
            #[trigger] ray_hits_fractal(
                inverse_transform_ray(desc.transforms[i], ray),
                desc,
                (depth - 1) as nat,
            )
    }
}

// ---------------------------------------------------------------------------
// Unfolding helpers (needed by exec layer; open spec + decreases doesn't
// always auto-unfold with variable depth arguments)
// ---------------------------------------------------------------------------

pub proof fn lemma_ray_hits_fractal_base<T: OrderedField>(
    ray: Ray3<T>, desc: FractalDesc<T>,
)
    ensures
        ray_hits_fractal(ray, desc, 0) == ray_hits_box(ray, desc.base_aabb),
{}

} // verus!
