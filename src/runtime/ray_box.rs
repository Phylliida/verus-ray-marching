use vstd::prelude::*;
use verus_linalg::vec3::Vec3;
use verus_geometry::runtime::point3::RuntimePoint3;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
use crate::ray_box::*;

verus! {

/// Does the ray hit the AABB? Delegates to verus-geometry's ray_hits_aabb3_exec.
pub fn ray_hits_box_exec(ray: &RuntimeRay3, b: &RuntimeBox3) -> (out: bool)
    requires
        ray.wf_spec(),
        b.wf_spec(),
    ensures
        out == ray_hits_box::<RationalModel>(ray@, b@),
{
    // verus-geometry's ray_hits_aabb3_exec takes direction as RuntimePoint3,
    // so we convert the RuntimeVec3 dir to RuntimePoint3.
    let dir_as_pt = RuntimePoint3::new(
        verus_linalg::runtime::copy_rational(&ray.dir.x),
        verus_linalg::runtime::copy_rational(&ray.dir.y),
        verus_linalg::runtime::copy_rational(&ray.dir.z),
    );
    proof {
        // Bridge: dir_as_pt@ fields == ray@.dir fields
        assert(Vec3 { x: dir_as_pt@.x, y: dir_as_pt@.y, z: dir_as_pt@.z } == ray@.dir);
    }
    verus_geometry::runtime::ray::ray_hits_aabb3_exec(
        &ray.origin, &dir_as_pt, &b.min, &b.max,
    )
}

} // verus!
