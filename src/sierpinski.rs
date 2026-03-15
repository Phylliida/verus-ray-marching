use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::embedding::from_int;
use verus_linalg::vec3::Vec3;
use verus_geometry::point3::*;
use crate::types::*;
use crate::fractal::*;

verus! {

// ---------------------------------------------------------------------------
// Sierpinski tetrahedron: 4 children at vertices, scale 1/2
// ---------------------------------------------------------------------------

/// Build the 4 Sierpinski tetrahedron transforms.
/// Each child is scaled by 1/2 and placed at a vertex of the parent tetrahedron.
/// Vertices of a regular tetrahedron inscribed in the unit cube:
///   v0 = (0, 0, 0), v1 = (1, 0, 0), v2 = (1/2, sqrt(3)/2, 0), v3 = (1/2, sqrt(3)/6, sqrt(6)/3)
///
/// For simplicity with rational arithmetic, we use a right-angled tetrahedron:
///   v0 = (0, 0, 0), v1 = (1, 0, 0), v2 = (0, 1, 0), v3 = (0, 0, 1)
/// Each transform: scale = 1/2, translate = vertex / 2.
pub open spec fn sierpinski_transforms<T: OrderedField>() -> Seq<AffineTransform<T>> {
    let half = from_int::<T>(2).recip();
    seq![
        // Child at vertex (0, 0, 0): scale 1/2, translate (0, 0, 0)
        AffineTransform {
            scale: half,
            translate: Vec3 { x: T::zero(), y: T::zero(), z: T::zero() },
        },
        // Child at vertex (1, 0, 0): scale 1/2, translate (1/2, 0, 0)
        AffineTransform {
            scale: half,
            translate: Vec3 { x: half, y: T::zero(), z: T::zero() },
        },
        // Child at vertex (0, 1, 0): scale 1/2, translate (0, 1/2, 0)
        AffineTransform {
            scale: half,
            translate: Vec3 { x: T::zero(), y: half, z: T::zero() },
        },
        // Child at vertex (0, 0, 1): scale 1/2, translate (0, 0, 1/2)
        AffineTransform {
            scale: half,
            translate: Vec3 { x: T::zero(), y: T::zero(), z: half },
        },
    ]
}

/// There are exactly 4 Sierpinski transforms.
pub proof fn lemma_sierpinski_transform_count<T: OrderedField>()
    ensures sierpinski_transforms::<T>().len() == 4,
{
    // Follows from the seq! macro expansion.
}

/// Build a Sierpinski fractal descriptor with the unit cube as base AABB.
pub open spec fn sierpinski_fractal<T: OrderedField>() -> FractalDesc<T> {
    FractalDesc {
        transforms: sierpinski_transforms(),
        base_aabb: Box3 {
            min: Point3 { x: T::zero(), y: T::zero(), z: T::zero() },
            max: Point3 { x: T::one(), y: T::one(), z: T::one() },
        },
    }
}

} // verus!
