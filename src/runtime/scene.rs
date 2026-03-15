use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::runtime::vec3::RuntimeVec3;
use verus_linalg::runtime::copy_rational;
use verus_geometry::runtime::point3::*;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
use crate::types::*;
use crate::scene::*;

verus! {

// ---------------------------------------------------------------------------
// RuntimePrimitive
// ---------------------------------------------------------------------------

/// Runtime primitive enum mirroring spec Primitive.
pub enum RuntimePrimitive {
    PrimSphere(RuntimeSphere),
    PrimPlane(RuntimePlane),
    PrimBox(RuntimeBox3),
    PrimCylinder(RuntimeCylinder),
}

impl View for RuntimePrimitive {
    type V = Primitive<RationalModel>;
    open spec fn view(&self) -> Primitive<RationalModel> {
        match self {
            RuntimePrimitive::PrimSphere(s) => Primitive::PrimSphere(s@),
            RuntimePrimitive::PrimPlane(p) => Primitive::PrimPlane(p@),
            RuntimePrimitive::PrimBox(b) => Primitive::PrimBox(b@),
            RuntimePrimitive::PrimCylinder(c) => Primitive::PrimCylinder(c@),
        }
    }
}

impl RuntimePrimitive {
    pub open spec fn wf_spec(&self) -> bool {
        match self {
            RuntimePrimitive::PrimSphere(s) => s.wf_spec(),
            RuntimePrimitive::PrimPlane(p) => p.wf_spec(),
            RuntimePrimitive::PrimBox(b) => b.wf_spec(),
            RuntimePrimitive::PrimCylinder(c) => c.wf_spec(),
        }
    }
}

// ---------------------------------------------------------------------------
// RuntimeSceneObject
// ---------------------------------------------------------------------------

/// Runtime scene object: primitive + AABB + material ID.
pub struct RuntimeSceneObject {
    pub primitive: RuntimePrimitive,
    pub aabb: RuntimeBox3,
    pub material_id: u64,
    pub model: Ghost<SceneObject<RationalModel>>,
}

impl View for RuntimeSceneObject {
    type V = SceneObject<RationalModel>;
    open spec fn view(&self) -> SceneObject<RationalModel> {
        self.model@
    }
}

impl RuntimeSceneObject {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.primitive.wf_spec()
        &&& self.aabb.wf_spec()
        &&& self.primitive@ == self@.primitive
        &&& self.aabb@ == self@.aabb
        &&& self.material_id as nat == self@.material_id
    }
}

// ---------------------------------------------------------------------------
// RuntimeCamera
// ---------------------------------------------------------------------------

/// Runtime pinhole camera.
pub struct RuntimeCamera {
    pub origin: RuntimePoint3,
    pub forward: RuntimeVec3,
    pub right: RuntimeVec3,
    pub up: RuntimeVec3,
    pub model: Ghost<Camera<RationalModel>>,
}

impl View for RuntimeCamera {
    type V = Camera<RationalModel>;
    open spec fn view(&self) -> Camera<RationalModel> {
        self.model@
    }
}

impl RuntimeCamera {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.origin.wf_spec()
        &&& self.forward.wf_spec()
        &&& self.right.wf_spec()
        &&& self.up.wf_spec()
        &&& self.origin@ == self@.origin
        &&& self.forward@ == self@.forward
        &&& self.right@ == self@.right
        &&& self.up@ == self@.up
    }
}

// ---------------------------------------------------------------------------
// ray_hits_primitive_exec
// ---------------------------------------------------------------------------

/// Does the ray hit the given primitive?
pub fn ray_hits_primitive_exec(ray: &RuntimeRay3, prim: &RuntimePrimitive) -> (out: bool)
    requires
        ray.wf_spec(),
        prim.wf_spec(),
    ensures
        out == ray_hits_primitive::<RationalModel>(ray@, prim@),
{
    match prim {
        RuntimePrimitive::PrimSphere(s) => {
            crate::runtime::ray_sphere::ray_hits_sphere_nodiv_exec(ray, s)
        },
        RuntimePrimitive::PrimPlane(pl) => {
            crate::runtime::ray_plane::ray_hits_plane_exec(ray, pl)
        },
        RuntimePrimitive::PrimBox(b) => {
            crate::runtime::ray_box::ray_hits_box_exec(ray, b)
        },
        RuntimePrimitive::PrimCylinder(c) => {
            crate::runtime::ray_cylinder::ray_hits_inf_cylinder_exec(ray, c)
        },
    }
}

// ---------------------------------------------------------------------------
// camera_ray_exec
// ---------------------------------------------------------------------------

/// Generate a camera ray for normalized image coordinates (u, v).
pub fn camera_ray_exec(
    cam: &RuntimeCamera,
    u: &RuntimeRational,
    v: &RuntimeRational,
) -> (out: RuntimeRay3)
    requires
        cam.wf_spec(),
        u.wf_spec(),
        v.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == camera_ray::<RationalModel>(cam@, u@, v@),
{
    // dir = forward + u * right + v * up
    let u_right = RuntimeVec3::scale_exec(u, &cam.right);
    let v_up = RuntimeVec3::scale_exec(v, &cam.up);
    let dir1 = cam.forward.add_exec(&u_right);
    let dir = dir1.add_exec(&v_up);

    // Copy origin
    let origin = RuntimePoint3::new(
        copy_rational(&cam.origin.x),
        copy_rational(&cam.origin.y),
        copy_rational(&cam.origin.z),
    );

    RuntimeRay3::new(origin, dir)
}

// ---------------------------------------------------------------------------
// scene_closest_hit_index_exec — iterative version of spec
// ---------------------------------------------------------------------------

/// Find the index of the first hit object in the scene (linear scan).
/// Mirrors `scene_closest_hit_index` spec but uses an iterative loop.
pub fn scene_closest_hit_index_exec(
    scene: &Vec<RuntimeSceneObject>,
    ray: &RuntimeRay3,
) -> (out: Option<usize>)
    requires
        ray.wf_spec(),
        forall|i: int| 0 <= i < scene@.len() ==>
            (#[trigger] scene@[i]).wf_spec(),
    ensures
        match out {
            None => scene_closest_hit_index::<RationalModel>(
                scene@.map_values(|obj: RuntimeSceneObject| obj@), ray@).is_none(),
            Some(idx) => {
                &&& (idx as int) < scene@.len()
                &&& scene_closest_hit_index::<RationalModel>(
                    scene@.map_values(|obj: RuntimeSceneObject| obj@), ray@)
                    == Some(idx as nat)
            },
        },
{
    let mut result: Option<usize> = None;
    let mut i: usize = 0;

    let ghost scene_spec: Seq<SceneObject<RationalModel>> =
        scene@.map_values(|obj: RuntimeSceneObject| obj@);

    proof {
        // Base case: empty prefix has no hit
        assert(scene_closest_hit_index::<RationalModel>(
            scene_spec.take(0), ray@).is_none());
    }

    while i < scene.len()
        invariant_ensures
            0 <= i <= scene@.len(),
            scene_spec == scene@.map_values(|obj: RuntimeSceneObject| obj@),
            ray.wf_spec(),
            forall|j: int| 0 <= j < scene@.len() ==>
                (#[trigger] scene@[j]).wf_spec(),
            // result mirrors spec on the prefix scene_spec.take(i)
            match result {
                None => scene_closest_hit_index::<RationalModel>(
                    scene_spec.take(i as int), ray@).is_none(),
                Some(idx) => {
                    &&& (idx as int) < i
                    &&& scene_closest_hit_index::<RationalModel>(
                        scene_spec.take(i as int), ray@) == Some(idx as nat)
                },
            },
        decreases scene@.len() - i,
    {
        let hit = ray_hits_primitive_exec(ray, &scene[i].primitive);

        proof {
            // Relate take(i+1) to take(i) via the spec's recursive structure.
            // scene_spec.take(i+1).drop_last() == scene_spec.take(i)
            assert(scene_spec.take((i + 1) as int).drop_last() =~= scene_spec.take(i as int));
            assert(scene_spec.take((i + 1) as int).len() == (i + 1) as nat);
            assert(scene_spec.take((i + 1) as int)[(i as int)] == scene_spec[i as int]);
        }

        if hit {
            match result {
                None => {
                    // First hit — spec would return Some(i) for this prefix
                    result = Some(i);
                },
                Some(_prev) => {
                    // Already have a hit; keep the earlier one (spec picks prev_idx)
                },
            }
        }

        i = i + 1;
    }

    proof {
        // Final: take(scene.len()) == full sequence
        assert(scene_spec.take(scene@.len() as int) =~= scene_spec);
    }

    result
}

} // verus!
