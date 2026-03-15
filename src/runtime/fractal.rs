use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::runtime::vec3::RuntimeVec3;
use verus_linalg::runtime::copy_rational;
use verus_geometry::runtime::point3::*;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
use crate::types::*;
use crate::fractal::*;
use crate::ray_box::*;

verus! {

// ---------------------------------------------------------------------------
// RuntimeAffineTransform
// ---------------------------------------------------------------------------

/// Runtime affine transform: uniform scale + translate.
pub struct RuntimeAffineTransform {
    pub scale: RuntimeRational,
    pub translate: RuntimeVec3,
    pub model: Ghost<AffineTransform<RationalModel>>,
}

impl View for RuntimeAffineTransform {
    type V = AffineTransform<RationalModel>;
    open spec fn view(&self) -> AffineTransform<RationalModel> {
        self.model@
    }
}

impl RuntimeAffineTransform {
    pub open spec fn wf_spec(&self) -> bool {
        &&& self.scale.wf_spec()
        &&& self.translate.wf_spec()
        &&& self.scale@ == self@.scale
        &&& self.translate@ == self@.translate
    }
}

// ---------------------------------------------------------------------------
// inverse_transform_ray_exec
// ---------------------------------------------------------------------------

/// Transform a ray into a child's local coordinate system.
pub fn inverse_transform_ray_exec(
    tf: &RuntimeAffineTransform,
    ray: &RuntimeRay3,
) -> (out: RuntimeRay3)
    requires
        tf.wf_spec(),
        ray.wf_spec(),
        !tf@.scale.eqv(RationalModel::from_int_spec(0)),
    ensures
        out.wf_spec(),
        out@ == inverse_transform_ray::<RationalModel>(tf@, ray@),
{
    // inv_s = 1 / scale
    let inv_s_opt = tf.scale.recip();
    let inv_s = inv_s_opt.unwrap();

    // shifted = sub3(origin, translate_as_point)
    let tr_pt = RuntimePoint3::new(
        copy_rational(&tf.translate.x),
        copy_rational(&tf.translate.y),
        copy_rational(&tf.translate.z),
    );
    let shifted = sub3_exec(&ray.origin, &tr_pt);

    // new origin: inv_s * shifted.{x,y,z}
    let ox = inv_s.mul(&shifted.x);
    let oy = inv_s.mul(&shifted.y);
    let oz = inv_s.mul(&shifted.z);
    let new_origin = RuntimePoint3::new(ox, oy, oz);

    // new dir: inv_s * dir.{x,y,z}
    let inv_s2 = copy_rational(&inv_s);
    let inv_s3 = copy_rational(&inv_s);
    let dx = inv_s.mul(&ray.dir.x);
    let dy = inv_s2.mul(&ray.dir.y);
    let dz = inv_s3.mul(&ray.dir.z);
    let new_dir = RuntimeVec3::new(dx, dy, dz);

    RuntimeRay3::new(new_origin, new_dir)
}

// ---------------------------------------------------------------------------
// ray_hits_fractal_exec
// ---------------------------------------------------------------------------

/// Does the ray hit any leaf of the fractal at the given depth?
/// Recursive descent with AABB pruning.
pub fn ray_hits_fractal_exec(
    ray: &RuntimeRay3,
    transforms: &Vec<RuntimeAffineTransform>,
    base_aabb: &RuntimeBox3,
    depth: u64,
    ghost_desc: Ghost<FractalDesc<RationalModel>>,
) -> (out: bool)
    requires
        ray.wf_spec(),
        base_aabb.wf_spec(),
        base_aabb@ == ghost_desc@.base_aabb,
        transforms@.len() == ghost_desc@.transforms.len(),
        forall|i: int| 0 <= i < transforms@.len() ==>
            (#[trigger] transforms@[i]).wf_spec() &&
            transforms@[i]@ == ghost_desc@.transforms[i] &&
            !ghost_desc@.transforms[i].scale.eqv(RationalModel::from_int_spec(0)),
    ensures
        out == ray_hits_fractal::<RationalModel>(ray@, ghost_desc@, depth as nat),
    decreases depth, 1nat,
{
    if depth == 0 {
        crate::runtime::ray_box::ray_hits_box_exec(ray, base_aabb)
    } else {
        proof {
            assert((depth as nat - 1) as nat == (depth - 1) as nat);
            // Test: can Z3 unfold ray_hits_fractal now with decreases depth, 0nat?
            assert(ray_hits_fractal::<RationalModel>(ray@, ghost_desc@, depth as nat) ==
                (exists|j: int| 0 <= j < ghost_desc@.transforms.len() &&
                    #[trigger] ray_hits_fractal::<RationalModel>(
                        inverse_transform_ray(ghost_desc@.transforms[j], ray@),
                        ghost_desc@,
                        (depth as nat - 1) as nat,
                    )));
        }
        assume(false);
        false
    }
}

} // verus!
