use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::embedding::from_int;
use verus_algebra::archimedean::nat_mul;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3};
use verus_geometry::runtime::point3::*;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
use crate::ray_sphere::*;

verus! {

/// Compute oc = origin - center.
fn rs_oc_exec(ray: &RuntimeRay3, sphere: &RuntimeSphere) -> (out: verus_linalg::runtime::vec3::RuntimeVec3)
    requires
        ray.wf_spec(),
        sphere.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rs_oc::<RationalModel>(ray@, sphere@),
{
    sub3_exec(&ray.origin, &sphere.center)
}

/// Compute quadratic coefficient A = norm_sq(dir).
fn rs_quad_a_exec(ray: &RuntimeRay3) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rs_quad_a::<RationalModel>(ray@),
{
    ray.dir.norm_sq_exec()
}

/// Compute quadratic coefficient B = 2 * dot(oc, dir).
fn rs_quad_b_exec(ray: &RuntimeRay3, sphere: &RuntimeSphere) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        sphere.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rs_quad_b::<RationalModel>(ray@, sphere@),
{
    let oc = rs_oc_exec(ray, sphere);
    let d = oc.dot_exec(&ray.dir);
    let two = RuntimeRational::from_int(2);
    two.mul(&d)
}

/// Compute quadratic coefficient C = norm_sq(oc) - radius_sq.
fn rs_quad_c_exec(ray: &RuntimeRay3, sphere: &RuntimeSphere) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        sphere.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rs_quad_c::<RationalModel>(ray@, sphere@),
{
    let oc = rs_oc_exec(ray, sphere);
    let nsq = oc.norm_sq_exec();
    nsq.sub(&sphere.radius_sq)
}

/// Compute discriminant B^2 - 4*A*C.
fn rs_discriminant_exec(ray: &RuntimeRay3, sphere: &RuntimeSphere) -> (out: RuntimeRational)
    requires
        ray.wf_spec(),
        sphere.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == rs_discriminant::<RationalModel>(ray@, sphere@),
{
    let a = rs_quad_a_exec(ray);
    let b = rs_quad_b_exec(ray, sphere);
    let c = rs_quad_c_exec(ray, sphere);
    let four = RuntimeRational::from_int(4);
    let b_sq = b.mul(&b);
    let ac = a.mul(&c);
    let four_ac = four.mul(&ac);
    proof {
        // Bridge: from_int::<RationalModel>(4) == RationalModel::from_int_spec(4)
        // Unfold nat_mul(4, one) step by step:
        //   nat_mul(0, one) = zero() = from_int_spec(0)
        //   nat_mul(1, one) = one.add(nat_mul(0, one)) = from_int_spec(1).add_spec(from_int_spec(0)) = from_int_spec(1)
        //   nat_mul(2, one) = one.add(nat_mul(1, one)) = from_int_spec(1).add_spec(from_int_spec(1)) = from_int_spec(2)
        //   nat_mul(3, one) = one.add(nat_mul(2, one)) = from_int_spec(1).add_spec(from_int_spec(2)) = from_int_spec(3)
        //   nat_mul(4, one) = one.add(nat_mul(3, one)) = from_int_spec(1).add_spec(from_int_spec(3)) = from_int_spec(4)
        let one = RationalModel::from_int_spec(1);
        assert(nat_mul::<RationalModel>(0, one) == RationalModel::from_int_spec(0));
        assert(nat_mul::<RationalModel>(1, one) == RationalModel::from_int_spec(1));
        assert(nat_mul::<RationalModel>(2, one) == RationalModel::from_int_spec(2));
        assert(nat_mul::<RationalModel>(3, one) == RationalModel::from_int_spec(3));
        assert(nat_mul::<RationalModel>(4, one) == RationalModel::from_int_spec(4));
        assert(from_int::<RationalModel>(4) == RationalModel::from_int_spec(4));
    }
    b_sq.sub(&four_ac)
}

/// Division-free sphere intersection test: disc >= 0.
pub fn ray_hits_sphere_nodiv_exec(ray: &RuntimeRay3, sphere: &RuntimeSphere) -> (out: bool)
    requires
        ray.wf_spec(),
        sphere.wf_spec(),
    ensures
        out == ray_hits_sphere_nodiv::<RationalModel>(ray@, sphere@),
{
    let disc = rs_discriminant_exec(ray, sphere);
    let zero = RuntimeRational::from_int(0);
    !disc.lt(&zero)
}

/// Full forward hit test: disc >= 0 and at least one non-negative root.
pub fn ray_hits_sphere_forward_exec(ray: &RuntimeRay3, sphere: &RuntimeSphere) -> (out: bool)
    requires
        ray.wf_spec(),
        sphere.wf_spec(),
    ensures
        out == ray_hits_sphere_forward::<RationalModel>(ray@, sphere@),
{
    let a = rs_quad_a_exec(ray);
    let b = rs_quad_b_exec(ray, sphere);
    let c = rs_quad_c_exec(ray, sphere);
    let disc = rs_discriminant_exec(ray, sphere);
    let zero = RuntimeRational::from_int(0);

    if a.eq(&zero) {
        return false;
    }
    if disc.lt(&zero) {
        return false;
    }
    // At least one root >= 0: B <= 0 or C <= 0
    b.lt(&zero) || b.eq(&zero) || c.lt(&zero) || c.eq(&zero)
}

} // verus!
