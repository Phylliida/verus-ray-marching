use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::lemmas::additive_group_lemmas;
use verus_algebra::lemmas::ring_lemmas;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3, scale as scale3};
use verus_geometry::point3::*;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// Unified normal computation for all primitives
// ---------------------------------------------------------------------------

/// Normal at a sphere hit point: hit - center (unnormalized, outward).
pub open spec fn normal_sphere<T: Ring>(hit: Point3<T>, sphere: Sphere<T>) -> Vec3<T> {
    sub3(hit, sphere.center)
}

/// Normal at a plane hit point: the plane's stored normal.
pub open spec fn normal_plane<T: Ring>(pl: Plane<T>) -> Vec3<T> {
    pl.normal
}

/// Normal at a box hit point: axis-aligned face normal.
/// face: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z.
pub open spec fn normal_box<T: Ring>(face: nat) -> Vec3<T> {
    if face == 0 {
        Vec3 { x: T::one(), y: T::zero(), z: T::zero() }
    } else if face == 1 {
        Vec3 { x: T::one().neg(), y: T::zero(), z: T::zero() }
    } else if face == 2 {
        Vec3 { x: T::zero(), y: T::one(), z: T::zero() }
    } else if face == 3 {
        Vec3 { x: T::zero(), y: T::one().neg(), z: T::zero() }
    } else if face == 4 {
        Vec3 { x: T::zero(), y: T::zero(), z: T::one() }
    } else {
        Vec3 { x: T::zero(), y: T::zero(), z: T::one().neg() }
    }
}

/// Normal at a cylinder hit point: radial component perpendicular to axis.
/// normal = (hit - base) - dot(hit - base, axis) * axis
pub open spec fn normal_cylinder<T: Ring>(hit: Point3<T>, cyl: Cylinder<T>) -> Vec3<T> {
    let d = sub3(hit, cyl.base_center);
    let along = dot3(d, cyl.axis_dir);
    d.sub(scale3(along, cyl.axis_dir))
}

// ---------------------------------------------------------------------------
// Lemma: cylinder normal is perpendicular to axis
// ---------------------------------------------------------------------------

/// dot(normal_cylinder(hit, cyl), axis) ≡ 0 when norm_sq(axis) ≡ 1.
///
/// Proof:
///   Let d = sub3(hit, base), h = dot(d, axis), normal = d - scale(h, axis).
///   dot(normal, axis) = dot(d - h*axis, axis)
///     ≡ dot(d, axis) + dot(-(h*axis), axis)          [sub → add neg, distributes]
///     ≡ h + (-(h * dot(axis, axis)))                  [dot_neg_left, dot_scale_left]
///     ≡ h + (-(h * 1))                                [norm_sq ≡ 1]
///     ≡ h + (-h)                                      [mul_one]
///     ≡ 0                                             [additive inverse]
pub proof fn lemma_cylinder_normal_perp_to_axis<T: OrderedField>(
    hit: Point3<T>, cyl: Cylinder<T>,
)
    requires
        norm_sq3(cyl.axis_dir).eqv(T::one()),
    ensures
        dot3(normal_cylinder(hit, cyl), cyl.axis_dir).eqv(T::zero()),
{
    let d = sub3(hit, cyl.base_center);
    let axis = cyl.axis_dir;
    let h = dot3(d, axis);
    let ha = scale3(h, axis);
    let normal = d.sub(ha);

    // Step 1: normal = d.sub(ha) ≡ d.add(ha.neg())
    Vec3::<T>::axiom_sub_is_add_neg(d, ha);
    let neg_ha = ha.neg();

    // Step 2: dot(d.add(neg_ha), axis) ≡ dot(d, axis) + dot(neg_ha, axis)
    verus_linalg::vec3::ops::lemma_dot_distributes_left::<T>(d, neg_ha, axis);
    // dot(normal, axis) ≡ dot(d.add(neg_ha), axis) by congruence
    Vec3::<T>::axiom_eqv_reflexive(axis);
    verus_linalg::vec3::ops::lemma_dot_congruence::<T>(normal, d.add(neg_ha), axis, axis);
    // Chain: dot(normal, axis) ≡ dot(d, axis) + dot(neg_ha, axis)
    T::axiom_eqv_transitive(
        dot3(normal, axis),
        dot3(d.add(neg_ha), axis),
        dot3(d, axis).add(dot3(neg_ha, axis)),
    );

    // Step 3: dot(neg_ha, axis) ≡ -dot(ha, axis) by dot_neg_left
    verus_linalg::vec3::ops::lemma_dot_neg_left::<T>(ha, axis);

    // Step 4: dot(ha, axis) = dot(scale(h, axis), axis) ≡ h * dot(axis, axis) by dot_scale_left
    verus_linalg::vec3::ops::lemma_dot_scale_left::<T>(h, axis, axis);

    // Step 5: dot(axis, axis) = norm_sq(axis) ≡ 1, so h * dot(axis, axis) ≡ h * 1
    ring_lemmas::lemma_mul_congruence_right::<T>(h, norm_sq3(axis), T::one());
    // h * dot(axis, axis) ≡ h * 1
    T::axiom_eqv_transitive(
        dot3(ha, axis), h.mul(norm_sq3(axis)), h.mul(T::one()),
    );

    // Step 6: h * 1 ≡ h
    T::axiom_mul_one_right(h);
    T::axiom_eqv_transitive(dot3(ha, axis), h.mul(T::one()), h);

    // Step 7: -dot(ha, axis) ≡ -h by neg congruence
    additive_group_lemmas::lemma_neg_congruence::<T>(dot3(ha, axis), h);
    // dot(neg_ha, axis) ≡ -h
    T::axiom_eqv_transitive(dot3(neg_ha, axis), dot3(ha, axis).neg(), h.neg());

    // Step 8: h + (-h) ≡ 0
    T::axiom_add_inverse_right(h);

    // Step 9: dot(d, axis) + dot(neg_ha, axis) ≡ h + (-h) by congruence
    T::axiom_eqv_reflexive(h);
    additive_group_lemmas::lemma_add_congruence::<T>(
        dot3(d, axis), h,
        dot3(neg_ha, axis), h.neg(),
    );

    // Step 10: h + (-h) ≡ 0 (from step 8)
    // Chain: dot(normal, axis) ≡ h + (-h) ≡ 0
    T::axiom_eqv_transitive(
        dot3(normal, axis),
        dot3(d, axis).add(dot3(neg_ha, axis)),
        h.add(h.neg()),
    );
    T::axiom_eqv_transitive(
        dot3(normal, axis),
        h.add(h.neg()),
        T::zero(),
    );
}

} // verus!
