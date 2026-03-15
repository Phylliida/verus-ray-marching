use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::runtime::vec3::RuntimeVec3;
use verus_linalg::runtime::copy_rational;
use verus_geometry::runtime::point3::*;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
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
    ghost_depth: Ghost<nat>,
) -> (out: bool)
    requires
        ray.wf_spec(),
        base_aabb.wf_spec(),
        base_aabb@ == ghost_desc@.base_aabb,
        ghost_depth@ == depth as nat,
        transforms@.len() == ghost_desc@.transforms.len(),
        forall|i: int| 0 <= i < transforms@.len() ==>
            (#[trigger] transforms@[i]).wf_spec() &&
            transforms@[i]@ == ghost_desc@.transforms[i] &&
            !ghost_desc@.transforms[i].scale.eqv(RationalModel::from_int_spec(0)),
    ensures
        out == ray_hits_fractal::<RationalModel>(ray@, ghost_desc@, ghost_depth@),
    decreases depth,
{
    if depth == 0 {
        let out = crate::runtime::ray_box::ray_hits_box_exec(ray, base_aabb);
        proof {
            assert(ghost_depth@ == 0nat);
            assert(ray_hits_fractal::<RationalModel>(ray@, ghost_desc@, ghost_depth@) ==
                ray_hits_box::<RationalModel>(ray@, ghost_desc@.base_aabb));
        }
        out
    } else {
        let mut i: usize = 0;
        let ghost sub_depth: nat = (ghost_depth@ - 1) as nat;
        while i < transforms.len()
            invariant
                depth > 0,
                ghost_depth@ == depth as nat,
                ghost_depth@ > 0,
                sub_depth == (ghost_depth@ - 1) as nat,
                0 <= i <= transforms@.len(),
                ray.wf_spec(),
                base_aabb.wf_spec(),
                base_aabb@ == ghost_desc@.base_aabb,
                transforms@.len() == ghost_desc@.transforms.len(),
                forall|j: int| 0 <= j < transforms@.len() ==>
                    (#[trigger] transforms@[j]).wf_spec() &&
                    transforms@[j]@ == ghost_desc@.transforms[j] &&
                    !ghost_desc@.transforms[j].scale.eqv(RationalModel::from_int_spec(0)),
                forall|j: int| 0 <= j < i ==>
                    !#[trigger] ray_hits_fractal::<RationalModel>(
                        inverse_transform_ray(ghost_desc@.transforms[j], ray@),
                        ghost_desc@,
                        sub_depth,
                    ),
            decreases transforms@.len() - i,
        {
            let child_ray = inverse_transform_ray_exec(&transforms[i], ray);
            proof {
                assert(sub_depth == (depth - 1) as nat);
            }
            let hit = ray_hits_fractal_exec(
                &child_ray, transforms, base_aabb, depth - 1, ghost_desc,
                Ghost(sub_depth),
            );
            if hit {
                return true;
            }
            i += 1;
        }
        false
    }
}

} // verus!
