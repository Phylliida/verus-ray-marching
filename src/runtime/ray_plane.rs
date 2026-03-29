use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::vec3::ops::dot as dot3;
use verus_geometry::runtime::point3::*;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
use crate::ray_plane::*;

verus! {

///  Compute ray-plane denominator: dot(dir, normal).
pub fn ray_plane_denom_exec(ray: &RuntimeRay3, pl: &RuntimePlane) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        pl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == ray_plane_denom::<RationalModel>(ray@, pl@),
{
    ray.dir.dot_exec(&pl.normal)
}

///  Compute ray-plane numerator: dot(pl.point - origin, normal).
pub fn ray_plane_numer_exec(ray: &RuntimeRay3, pl: &RuntimePlane) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        pl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == ray_plane_numer::<RationalModel>(ray@, pl@),
{
    let diff = sub3_exec(&pl.point, &ray.origin);
    dot3_exec(&diff, &pl.normal)
}

///  Compute ray-plane parameter t = numer / denom.
pub fn ray_plane_t_exec(ray: &RuntimeRay3, pl: &RuntimePlane) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        pl.wf_spec(),
        !ray_plane_denom::<RationalModel>(ray@, pl@).eqv(RationalModel::from_int_spec(0)),
    ensures
        out.wf_spec(),
        out@ == ray_plane_t::<RationalModel>(ray@, pl@),
{
    let numer = ray_plane_numer_exec(ray, pl);
    let denom = ray_plane_denom_exec(ray, pl);
    numer.div(&denom)
}

///  Does the ray hit the plane?
pub fn ray_hits_plane_exec(ray: &RuntimeRay3, pl: &RuntimePlane) -> (out: bool)
    requires
        ray.wf_spec(),
        pl.wf_spec(),
    ensures
        out == ray_hits_plane::<RationalModel>(ray@, pl@),
{
    let denom = ray_plane_denom_exec(ray, pl);
    let zero = RuntimeRational::from_int(0);
    if denom.eq(&zero) {
        return false;
    }
    let t = ray_plane_t_exec(ray, pl);
    !t.lt(&zero)
}

} //  verus!
