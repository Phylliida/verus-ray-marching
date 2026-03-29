use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::*;

verus! {

///  Torus centered at `center` with the tube axis along Y.
///  Major radius R (center of tube to axis), minor radius r (tube radius).
///  Stored as squared radii to avoid sqrt in spec.
pub struct Torus<T: OrderedField> {
    pub center: Vec3<T>,
    pub major_radius_sq: T,
    pub minor_radius_sq: T,
}

///  Distance squared from point to the Y-axis through the torus center,
///  projected onto the XZ-plane: (px - cx)^2 + (pz - cz)^2
pub open spec fn xz_dist_sq<T: Ring>(p: Vec3<T>, center: Vec3<T>) -> T {
    let dx = p.x.sub(center.x);
    let dz = p.z.sub(center.z);
    dx.mul(dx).add(dz.mul(dz))
}

///  A point lies on the torus surface iff:
///    (sqrt(x^2 + z^2) - R)^2 + y^2 = r^2
///  Equivalently (squaring to eliminate sqrt):
///    (x^2 + y^2 + z^2 + R^2 - r^2)^2 = 4 R^2 (x^2 + z^2)
///  where x,y,z are relative to the torus center.
pub open spec fn point_on_torus<T: OrderedField>(p: Vec3<T>, torus: Torus<T>) -> bool {
    let rel = sub3(p, torus.center);
    let sum_sq = norm_sq3(rel);
    let lhs_inner = sum_sq.add(torus.major_radius_sq).sub(torus.minor_radius_sq);
    let lhs = lhs_inner.mul(lhs_inner);
    let xz_sq = rel.x.mul(rel.x).add(rel.z.mul(rel.z));
    let four = T::one().add(T::one()).add(T::one()).add(T::one());
    let rhs = four.mul(torus.major_radius_sq).mul(xz_sq);
    lhs.eqv(rhs)
}

///  Quartic coefficients for the ray-torus intersection.
///  Substituting ray(t) = o + t*d into the implicit torus equation yields
///  c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0 = 0.
pub open spec fn ray_torus_quartic_c4<T: Ring>(d: Vec3<T>) -> T {
    let dd = dot3(d, d);
    dd.mul(dd)
}

pub open spec fn ray_torus_quartic_c3<T: Ring>(o: Vec3<T>, d: Vec3<T>) -> T {
    let four = T::one().add(T::one()).add(T::one()).add(T::one());
    let dd = dot3(d, d);
    let od = dot3(o, d);
    four.mul(dd).mul(od)
}

pub open spec fn ray_torus_quartic_c2<T: OrderedField>(
    o: Vec3<T>, d: Vec3<T>, torus: Torus<T>,
) -> T {
    let two = T::one().add(T::one());
    let four = two.add(two);
    let dd = dot3(d, d);
    let od = dot3(o, d);
    let oo = dot3(o, o);
    let alpha = oo.add(torus.major_radius_sq).sub(torus.minor_radius_sq);
    let dd_xz = d.x.mul(d.x).add(d.z.mul(d.z));
    four.mul(od).mul(od)
        .add(two.mul(dd).mul(alpha))
        .sub(four.mul(torus.major_radius_sq).mul(dd_xz))
}

pub open spec fn ray_torus_quartic_c1<T: OrderedField>(
    o: Vec3<T>, d: Vec3<T>, torus: Torus<T>,
) -> T {
    let two = T::one().add(T::one());
    let four = two.add(two);
    let eight = four.add(four);
    let od = dot3(o, d);
    let oo = dot3(o, o);
    let alpha = oo.add(torus.major_radius_sq).sub(torus.minor_radius_sq);
    let od_xz = o.x.mul(d.x).add(o.z.mul(d.z));
    four.mul(od).mul(alpha)
        .sub(eight.mul(torus.major_radius_sq).mul(od_xz))
}

pub open spec fn ray_torus_quartic_c0<T: OrderedField>(
    o: Vec3<T>, torus: Torus<T>,
) -> T {
    let four = T::one().add(T::one()).add(T::one()).add(T::one());
    let oo = dot3(o, o);
    let alpha = oo.add(torus.major_radius_sq).sub(torus.minor_radius_sq);
    let oo_xz = o.x.mul(o.x).add(o.z.mul(o.z));
    alpha.mul(alpha)
        .sub(four.mul(torus.major_radius_sq).mul(oo_xz))
}

///  Whether a ray intersects the torus (exists non-negative t satisfying the quartic).
pub open spec fn ray_hits_torus<T: OrderedField>(
    ray: super::types::Ray3<T>, torus: Torus<T>,
) -> bool {
    let o = sub3(ray.origin, torus.center);
    let d = ray.direction;
    exists|t: T| #![auto]
        t.le(T::zero()).eqv(false)
        && point_on_torus(add3(ray.origin, scale3(d, t)), torus)
}

} //  verus!
