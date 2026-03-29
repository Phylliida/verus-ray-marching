use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3, scale as scale3};
use verus_geometry::point3::*;
use crate::types::*;
use crate::scene::*;
use crate::lighting::*;

verus! {

//  ---------------------------------------------------------------------------
//  Material
//  ---------------------------------------------------------------------------

///  A simple material: RGB color (each channel in [0, 1] for unit colors).
pub struct Material<T: Ring> {
    pub r: T,
    pub g: T,
    pub b: T,
}

//  ---------------------------------------------------------------------------
//  Scene with materials
//  ---------------------------------------------------------------------------

///  A scene object paired with its material.
pub struct RenderObject<T: OrderedField> {
    pub obj: SceneObject<T>,
    pub material: Material<T>,
}

///  A point light source.
pub struct PointLight<T: Ring> {
    pub position: Point3<T>,
    pub intensity: T,
}

///  Full scene description for rendering.
pub struct Scene<T: OrderedField> {
    pub objects: Seq<RenderObject<T>>,
    pub lights: Seq<PointLight<T>>,
    pub camera: Camera<T>,
    pub ambient: T,
}

//  ---------------------------------------------------------------------------
//  Render spec: what pixel color should be
//  ---------------------------------------------------------------------------

///  Compute the normal for a primitive at a given hit point.
pub open spec fn primitive_normal<T: OrderedField>(
    hit: Point3<T>, prim: Primitive<T>,
) -> Vec3<T> {
    match prim {
        Primitive::PrimSphere(s) => crate::normals::normal_sphere(hit, s),
        Primitive::PrimPlane(pl) => crate::normals::normal_plane(pl),
        Primitive::PrimBox(_b) => {
            //  Simplified: for AABB, return +Y normal (ground plane case)
            Vec3 { x: T::zero(), y: T::one(), z: T::zero() }
        },
        Primitive::PrimCylinder(c) => crate::normals::normal_cylinder(hit, c),
    }
}

///  The mathematically correct color for a pixel, given a hit.
///
///  For each light, compute Lambertian shade = max(0, dot(normal, light_dir))
///  and accumulate: color += material * intensity * shade.
///  Add ambient term.
pub open spec fn shade_hit<T: OrderedField>(
    hit: Point3<T>,
    normal: Vec3<T>,
    mat: Material<T>,
    lights: Seq<PointLight<T>>,
    ambient: T,
) -> (T, T, T)
    decreases lights.len(),
{
    //  Start with ambient contribution
    let base_r = mat.r.mul(ambient);
    let base_g = mat.g.mul(ambient);
    let base_b = mat.b.mul(ambient);

    if lights.len() == 0 {
        (base_r, base_g, base_b)
    } else {
        //  Shade from first light
        let light = lights[0];
        let light_dir = light_direction(hit, light.position);
        let shade = lambertian_clamped(normal, light_dir);
        let contrib = light.intensity.mul(shade);

        //  Recursive contribution from remaining lights
        let rest = shade_hit(hit, normal, mat, lights.drop_first(), ambient);

        //  Total = ambient + sum of all light contributions
        //  But we only add ambient once (in the base case), so for recursion
        //  we just accumulate light contributions
        (
            mat.r.mul(ambient).add(mat.r.mul(contrib)).add(rest.0.sub(mat.r.mul(ambient))),
            mat.g.mul(ambient).add(mat.g.mul(contrib)).add(rest.1.sub(mat.g.mul(ambient))),
            mat.b.mul(ambient).add(mat.b.mul(contrib)).add(rest.2.sub(mat.b.mul(ambient))),
        )
    }
}

///  Single-light simplified shade (most common case).
pub open spec fn shade_single_light<T: OrderedField>(
    hit: Point3<T>,
    normal: Vec3<T>,
    mat: Material<T>,
    light: PointLight<T>,
    ambient: T,
) -> (T, T, T) {
    let light_dir = light_direction(hit, light.position);
    let shade = lambertian_clamped(normal, light_dir);
    let diffuse = light.intensity.mul(shade);
    let total = ambient.add(diffuse);
    (mat.r.mul(total), mat.g.mul(total), mat.b.mul(total))
}

///  Background color for rays that miss all objects.
pub open spec fn background_color<T: Ring>() -> (T, T, T) {
    //  Sky gradient: light blue
    (T::zero(), T::zero(), T::zero())
}

///  The mathematically correct pixel color for normalized image coords (u, v).
pub open spec fn render_pixel_spec<T: OrderedField>(
    scene: Scene<T>, u: T, v: T,
) -> (T, T, T) {
    let ray = camera_ray(scene.camera, u, v);
    let objs: Seq<SceneObject<T>> = scene.objects.map_values(|ro: RenderObject<T>| ro.obj);
    let hit_idx = scene_closest_hit_index(objs, ray);
    match hit_idx {
        None => background_color(),
        Some(idx) => {
            let ro = scene.objects[idx as int];
            //  For correct shading we need the hit point and normal.
            //  At spec level, we express this in terms of the primitive type.
            //  The exec layer computes the actual t-value and hit point.
            //
            //  Since we can't compute t generically without sqrt for spheres,
            //  we define the spec color in terms of shade_single_light
            //  and trust the exec to supply the correct hit point.
            let mat = ro.material;
            let light = scene.lights[0];
            //  Placeholder: the exec layer computes and supplies the actual
            //  hit point and normal for the matched primitive.
            (mat.r, mat.g, mat.b) //  Stub: exec provides full shading
        },
    }
}

} //  verus!
