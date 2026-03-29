use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{scale as scale3, dot as dot3};
use verus_geometry::point3::*;
use crate::types::*;

verus! {

//  ---------------------------------------------------------------------------
//  Primitive enum
//  ---------------------------------------------------------------------------

///  A geometric primitive for ray intersection.
pub enum Primitive<T: OrderedField> {
    PrimSphere(Sphere<T>),
    PrimPlane(Plane<T>),
    PrimBox(Box3<T>),
    PrimCylinder(Cylinder<T>),
}

//  ---------------------------------------------------------------------------
//  Scene object
//  ---------------------------------------------------------------------------

///  A scene object: a primitive with an AABB for culling and a material ID.
pub struct SceneObject<T: OrderedField> {
    pub primitive: Primitive<T>,
    pub aabb: Box3<T>,
    pub material_id: nat,
}

//  ---------------------------------------------------------------------------
//  Camera
//  ---------------------------------------------------------------------------

///  A pinhole camera.
pub struct Camera<T: Ring> {
    pub origin: Point3<T>,
    pub forward: Vec3<T>,
    pub right: Vec3<T>,
    pub up: Vec3<T>,
}

///  Generate a camera ray for normalized image coordinates (u, v) in [-1, 1].
pub open spec fn camera_ray<T: Ring>(cam: Camera<T>, u: T, v: T) -> Ray3<T> {
    //  dir = forward + u * right + v * up
    let dir = cam.forward.add(scale3(u, cam.right)).add(scale3(v, cam.up));
    Ray3 { origin: cam.origin, dir }
}

//  ---------------------------------------------------------------------------
//  Scene traversal (spec)
//  ---------------------------------------------------------------------------

///  Does the ray hit a given primitive? (Dispatches to per-type tests.)
pub open spec fn ray_hits_primitive<T: OrderedField>(
    ray: Ray3<T>, prim: Primitive<T>,
) -> bool {
    match prim {
        Primitive::PrimSphere(s) => crate::ray_sphere::ray_hits_sphere_nodiv(ray, s),
        Primitive::PrimPlane(pl) => crate::ray_plane::ray_hits_plane(ray, pl),
        Primitive::PrimBox(b) => crate::ray_box::ray_hits_box(ray, b),
        Primitive::PrimCylinder(c) => crate::ray_cylinder::ray_hits_inf_cylinder(ray, c),
    }
}

///  Closest hit among all scene objects (linear scan).
///  Returns the index of the closest hit object, or None.
pub open spec fn scene_closest_hit_index<T: OrderedField>(
    scene: Seq<SceneObject<T>>, ray: Ray3<T>,
) -> Option<nat>
    decreases scene.len(),
{
    if scene.len() == 0 {
        None
    } else {
        let last_idx = (scene.len() - 1) as nat;
        let rest_hit = scene_closest_hit_index(scene.drop_last(), ray);
        if ray_hits_primitive(ray, scene[last_idx as int].primitive) {
            match rest_hit {
                None => Some(last_idx),
                Some(prev_idx) => {
                    //  Would compare t values; for now just pick the earlier index
                    Some(prev_idx)
                },
            }
        } else {
            rest_hit
        }
    }
}

} //  verus!
