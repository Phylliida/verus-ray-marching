use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::embedding::from_int;
use verus_algebra::archimedean::nat_mul;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::runtime::vec3::RuntimeVec3;
use verus_linalg::runtime::copy_rational;
use verus_geometry::runtime::point3::*;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
use crate::ray_cylinder::*;

verus! {

///  Compute the offset vector: ray.origin - cyl.base_center.
fn rc_offset_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: RuntimeVec3)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rc_offset::<RationalModel>(ray@, cyl@),
{
    sub3_exec(&ray.origin, &cyl.base_center)
}

///  Compute the perpendicular direction: dir - dot(dir, axis) * axis.
fn rc_perp_dir_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: RuntimeVec3)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rc_perp_dir::<RationalModel>(ray@, cyl@),
{
    let d_along = ray.dir.dot_exec(&cyl.axis_dir);
    let proj = RuntimeVec3::scale_exec(&d_along, &cyl.axis_dir);
    ray.dir.sub_exec(&proj)
}

///  Compute the perpendicular offset: oc - dot(oc, axis) * axis.
fn rc_perp_offset_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: RuntimeVec3)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rc_perp_offset::<RationalModel>(ray@, cyl@),
{
    let oc = rc_offset_exec(ray, cyl);
    let oc_along = oc.dot_exec(&cyl.axis_dir);
    let proj = RuntimeVec3::scale_exec(&oc_along, &cyl.axis_dir);
    oc.sub_exec(&proj)
}

///  Quadratic coefficient A = norm_sq(perp_dir).
fn rc_quad_a_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rc_quad_a::<RationalModel>(ray@, cyl@),
{
    let pd = rc_perp_dir_exec(ray, cyl);
    pd.norm_sq_exec()
}

///  Quadratic coefficient B = 2 * dot(perp_offset, perp_dir).
fn rc_quad_b_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rc_quad_b::<RationalModel>(ray@, cyl@),
{
    let po = rc_perp_offset_exec(ray, cyl);
    let pd = rc_perp_dir_exec(ray, cyl);
    let d = po.dot_exec(&pd);
    let two = RuntimeRational::from_int(2);
    proof {
        assert(from_int::<RationalModel>(2) == RationalModel::from_int_spec(2)) by {
            let one = RationalModel::from_int_spec(1);
            assert(nat_mul::<RationalModel>(0, one) == RationalModel::from_int_spec(0));
            assert(nat_mul::<RationalModel>(1, one) == RationalModel::from_int_spec(1));
            assert(nat_mul::<RationalModel>(2, one) == RationalModel::from_int_spec(2));
        };
    }
    two.mul(&d)
}

///  Quadratic coefficient C = norm_sq(perp_offset) - radius_sq.
fn rc_quad_c_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rc_quad_c::<RationalModel>(ray@, cyl@),
{
    let po = rc_perp_offset_exec(ray, cyl);
    let nsq = po.norm_sq_exec();
    nsq.sub(&cyl.radius_sq)
}

///  Discriminant = B^2 - 4*A*C.
fn rc_discriminant_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rc_discriminant::<RationalModel>(ray@, cyl@),
{
    let a = rc_quad_a_exec(ray, cyl);
    let b = rc_quad_b_exec(ray, cyl);
    let c = rc_quad_c_exec(ray, cyl);
    let four = RuntimeRational::from_int(4);
    let b_sq = b.mul(&b);
    let ac = a.mul(&c);
    let four_ac = four.mul(&ac);
    proof {
        assert(from_int::<RationalModel>(4) == RationalModel::from_int_spec(4)) by {
            let one = RationalModel::from_int_spec(1);
            assert(nat_mul::<RationalModel>(0, one) == RationalModel::from_int_spec(0));
            assert(nat_mul::<RationalModel>(1, one) == RationalModel::from_int_spec(1));
            assert(nat_mul::<RationalModel>(2, one) == RationalModel::from_int_spec(2));
            assert(nat_mul::<RationalModel>(3, one) == RationalModel::from_int_spec(3));
            assert(nat_mul::<RationalModel>(4, one) == RationalModel::from_int_spec(4));
        };
    }
    b_sq.sub(&four_ac)
}

///  Does the ray hit the infinite cylinder? disc >= 0.
pub fn ray_hits_inf_cylinder_exec(ray: &RuntimeRay3, cyl: &RuntimeCylinder) -> (out: bool)
    requires
        ray.wf_spec(),
        cyl.wf_spec(),
    ensures
        out == ray_hits_inf_cylinder::<RationalModel>(ray@, cyl@),
{
    let disc = rc_discriminant_exec(ray, cyl);
    let zero = RuntimeRational::from_int(0);
    !disc.lt(&zero)
}

} //  verus!
