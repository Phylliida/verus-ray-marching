use vstd::prelude::*;
use verus_rational::runtime_rational::RuntimeRational;
use verus_linalg::runtime::vec3::RuntimeVec3;
use verus_geometry::runtime::point3::*;
use crate::runtime::RationalModel;
use crate::runtime::types::*;
use crate::normals::*;

verus! {

///  Normal at a sphere hit point: hit - center (unnormalized, outward).
pub fn normal_sphere_exec(hit: &RuntimePoint3, sphere: &RuntimeSphere) -> (out: RuntimeVec3)
    requires
        hit.wf_spec(),
        sphere.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == normal_sphere::<RationalModel>(hit@, sphere@),
{
    sub3_exec(hit, &sphere.center)
}

///  Normal at a plane hit point: the plane's stored normal.
pub fn normal_plane_exec(pl: &RuntimePlane) -> (out: RuntimeVec3)
    requires
        pl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == normal_plane::<RationalModel>(pl@),
{
    RuntimeVec3::new(
        verus_linalg::runtime::copy_rational(&pl.normal.x),
        verus_linalg::runtime::copy_rational(&pl.normal.y),
        verus_linalg::runtime::copy_rational(&pl.normal.z),
    )
}

///  Normal at a box hit point: axis-aligned face normal.
///  face: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
pub fn normal_box_exec(face: u64) -> (out: RuntimeVec3)
    requires
        face <= 5,
    ensures
        out.wf_spec(),
        out@ == normal_box::<RationalModel>(face as nat),
{
    let zero = RuntimeRational::from_int(0);
    let one = RuntimeRational::from_int(1);
    let neg_one = RuntimeRational::from_int(-1);
    if face == 0 {
        RuntimeVec3::new(one, zero, RuntimeRational::from_int(0))
    } else if face == 1 {
        RuntimeVec3::new(neg_one, zero, RuntimeRational::from_int(0))
    } else if face == 2 {
        RuntimeVec3::new(zero, one, RuntimeRational::from_int(0))
    } else if face == 3 {
        RuntimeVec3::new(zero, neg_one, RuntimeRational::from_int(0))
    } else if face == 4 {
        RuntimeVec3::new(RuntimeRational::from_int(0), RuntimeRational::from_int(0), one)
    } else {
        RuntimeVec3::new(RuntimeRational::from_int(0), RuntimeRational::from_int(0), neg_one)
    }
}

///  Normal at a cylinder hit point: radial component perpendicular to axis.
///  normal = (hit - base) - dot(hit - base, axis) * axis
pub fn normal_cylinder_exec(hit: &RuntimePoint3, cyl: &RuntimeCylinder) -> (out: RuntimeVec3)
    requires
        hit.wf_spec(),
        cyl.wf_spec(),
    ensures
        out.wf_spec(),
        out@ == normal_cylinder::<RationalModel>(hit@, cyl@),
{
    let d = sub3_exec(hit, &cyl.base_center);
    let along = d.dot_exec(&cyl.axis_dir);
    let proj = RuntimeVec3::scale_exec(&along, &cyl.axis_dir);
    d.sub_exec(&proj)
}

} //  verus!
