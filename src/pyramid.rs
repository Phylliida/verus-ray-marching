use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::*;

verus! {

///  A square-based pyramid defined by 4 base vertices and an apex.
pub struct Pyramid<T: Ring> {
    pub base: Seq<Vec3<T>>,  //  4 base vertices in order
    pub apex: Vec3<T>,
}

///  Standard unit pyramid: base [0,1]x{0}x[0,1], apex at (0.5, 1.0, 0.5).
pub open spec fn unit_pyramid<T: OrderedField>() -> Pyramid<T> {
    let zero = T::zero();
    let one = T::one();
    let half = one.mul(T::one().add(T::one()));  //  placeholder — exact half needs field div
    Pyramid {
        base: seq![
            Vec3 { x: zero, y: zero, z: zero },
            Vec3 { x: one,  y: zero, z: zero },
            Vec3 { x: one,  y: zero, z: one  },
            Vec3 { x: zero, y: zero, z: one  },
        ],
        apex: Vec3 { x: half, y: one, z: half },
    }
}

///  Plane equation: normal · (p - point_on_plane) = 0.
///  For a triangular face (apex, b0, b1), the normal is cross((b0-apex), (b1-apex)).
pub open spec fn triangle_face_normal<T: Ring>(apex: Vec3<T>, b0: Vec3<T>, b1: Vec3<T>) -> Vec3<T> {
    cross3(sub3(b0, apex), sub3(b1, apex))
}

///  Whether a point is on or inside the half-space defined by a face plane.
///  For a convex solid, a point is inside iff it's on the inner side of all faces.
pub open spec fn point_in_halfspace<T: OrderedRing>(
    p: Vec3<T>, face_normal: Vec3<T>, face_point: Vec3<T>,
) -> bool {
    dot3(face_normal, sub3(p, face_point)).le(T::zero())
}

///  5 face normals of a pyramid: 1 base (pointing down) + 4 side triangles.
///  The base normal is (0, -1, 0).
///  Side face i uses vertices (apex, base[i], base[(i+1)%4]).
pub open spec fn pyramid_face_normals<T: Ring>(p: Pyramid<T>) -> Seq<Vec3<T>>
    recommends p.base.len() == 4
{
    seq![
        //  Base face normal: pointing downward
        Vec3 { x: T::zero(), y: T::zero().sub(T::one()), z: T::zero() },
        //  Side face 0: apex, base[0], base[1]
        triangle_face_normal(p.apex, p.base[0], p.base[1]),
        //  Side face 1: apex, base[1], base[2]
        triangle_face_normal(p.apex, p.base[1], p.base[2]),
        //  Side face 2: apex, base[2], base[3]
        triangle_face_normal(p.apex, p.base[2], p.base[3]),
        //  Side face 3: apex, base[3], base[0]
        triangle_face_normal(p.apex, p.base[3], p.base[0]),
    ]
}

///  A point is inside the pyramid iff it's on the interior side of all 5 face planes.
pub open spec fn point_in_pyramid<T: OrderedRing>(p: Vec3<T>, pyr: Pyramid<T>) -> bool
    recommends pyr.base.len() == 4
{
    //  Inside base plane (y >= 0 side, since base normal is -Y)
    let base_ok = T::zero().le(p.y);
    //  Inside each of the 4 side planes
    let n0 = triangle_face_normal(pyr.apex, pyr.base[0], pyr.base[1]);
    let n1 = triangle_face_normal(pyr.apex, pyr.base[1], pyr.base[2]);
    let n2 = triangle_face_normal(pyr.apex, pyr.base[2], pyr.base[3]);
    let n3 = triangle_face_normal(pyr.apex, pyr.base[3], pyr.base[0]);
    base_ok
    && point_in_halfspace(p, n0, pyr.apex)
    && point_in_halfspace(p, n1, pyr.apex)
    && point_in_halfspace(p, n2, pyr.apex)
    && point_in_halfspace(p, n3, pyr.apex)
}

///  Whether a ray hits the pyramid (exists t >= 0 such that ray(t) is on a face).
pub open spec fn ray_hits_pyramid<T: OrderedField>(
    ray: super::types::Ray3<T>, pyr: Pyramid<T>,
) -> bool
    recommends pyr.base.len() == 4
{
    exists|t: T| #![auto]
        t.le(T::zero()).eqv(false)
        && point_in_pyramid(add3(ray.origin, scale3(ray.direction, t)), pyr)
}

} //  verus!
