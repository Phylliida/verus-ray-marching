use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::lemmas::additive_group_lemmas;
use verus_algebra::lemmas::ring_lemmas;
use verus_algebra::lemmas::field_lemmas;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, scale as scale3};
use verus_geometry::point3::*;
use verus_geometry::ray::ray_point_3d;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// Ray-plane intersection
// ---------------------------------------------------------------------------

/// Denominator of ray-plane intersection: dot(dir, normal).
pub open spec fn ray_plane_denom<T: Ring>(ray: Ray3<T>, pl: Plane<T>) -> T {
    dot3(ray.dir, pl.normal)
}

/// Numerator of ray-plane intersection: dot(plane.point - origin, normal).
pub open spec fn ray_plane_numer<T: Ring>(ray: Ray3<T>, pl: Plane<T>) -> T {
    dot3(sub3(pl.point, ray.origin), pl.normal)
}

/// Parameter t at which the ray hits the plane.
pub open spec fn ray_plane_t<T: OrderedField>(ray: Ray3<T>, pl: Plane<T>) -> T
    recommends !ray_plane_denom(ray, pl).eqv(T::zero())
{
    ray_plane_numer(ray, pl).div(ray_plane_denom(ray, pl))
}

/// Does the ray hit the plane (non-parallel, t >= 0)?
pub open spec fn ray_hits_plane<T: OrderedField>(ray: Ray3<T>, pl: Plane<T>) -> bool {
    let denom = ray_plane_denom(ray, pl);
    &&& !denom.eqv(T::zero())
    &&& !ray_plane_t(ray, pl).lt(T::zero())
}

// ---------------------------------------------------------------------------
// Lemma: hit point lies on the plane
// ---------------------------------------------------------------------------

/// The ray-plane hit point satisfies the plane equation:
///   dot(ray_at(ray, t) - plane.point, normal) ≡ 0
///
/// Proof:
///   ray_at(ray, t) - pl.point ≡ (origin - pl.point) + t * dir   (shift)
///   dot((origin - pl.point) + t*dir, n)
///     ≡ dot(origin - pl.point, n) + dot(t*dir, n)               (linearity)
///     ≡ -numer + t * denom                                       (antisymmetry + scale)
///     ≡ -numer + (numer/denom) * denom                           (definition of t)
///     ≡ -numer + numer                                           (div_mul_cancel)
///     ≡ 0                                                        (additive inverse)
pub proof fn lemma_ray_plane_hit_on_plane<T: OrderedField>(ray: Ray3<T>, pl: Plane<T>)
    requires
        !ray_plane_denom(ray, pl).eqv(T::zero()),
    ensures
        point_on_plane(ray_at(ray, ray_plane_t(ray, pl)), pl),
{
    let t = ray_plane_t(ray, pl);
    let n = pl.normal;
    let dir = ray.dir;
    let numer = ray_plane_numer(ray, pl);
    let denom = ray_plane_denom(ray, pl);

    // Step 1: sub3(ray_at(ray, t), pl.point) ≡ sub3(origin, pl.point) + scale(t, dir)
    lemma_sub3_shift_add(ray.origin, scale3(t, dir), pl.point);
    let diff = sub3(ray_at(ray, t), pl.point);
    let shifted = sub3(ray.origin, pl.point).add(scale3(t, dir));

    // Step 2: dot(diff, n) ≡ dot(shifted, n) by dot congruence
    Vec3::<T>::axiom_eqv_reflexive(n);
    verus_linalg::vec3::ops::lemma_dot_congruence::<T>(diff, shifted, n, n);

    // Step 3: dot(shifted, n) ≡ dot(sub3(origin, pl.point), n) + dot(scale(t, dir), n)
    // by dot distributes left
    verus_linalg::vec3::ops::lemma_dot_distributes_left::<T>(
        sub3(ray.origin, pl.point), scale3(t, dir), n,
    );

    // Step 4: dot(sub3(origin, pl.point), n) ≡ -numer
    // sub3(origin, pl.point) ≡ sub3(pl.point, origin).neg() by antisymmetry
    lemma_sub3_antisymmetric(ray.origin, pl.point);
    // dot(sub3(origin, pl.point), n) ≡ dot(sub3(pl.point, origin).neg(), n)
    Vec3::<T>::axiom_eqv_reflexive(n);
    verus_linalg::vec3::ops::lemma_dot_congruence::<T>(
        sub3(ray.origin, pl.point), sub3(pl.point, ray.origin).neg(), n, n,
    );
    // dot(neg(v), n) ≡ -dot(v, n)
    verus_linalg::vec3::ops::lemma_dot_neg_left::<T>(sub3(pl.point, ray.origin), n);
    // Chain: dot(sub3(origin, pl.point), n) ≡ -numer
    T::axiom_eqv_transitive(
        dot3(sub3(ray.origin, pl.point), n),
        dot3(sub3(pl.point, ray.origin).neg(), n),
        numer.neg(),
    );

    // Step 5: dot(scale(t, dir), n) ≡ t * dot(dir, n) = t * denom
    verus_linalg::vec3::ops::lemma_dot_scale_left::<T>(t, dir, n);

    // Step 6: dot(shifted, n) ≡ -numer + t * denom
    // From step 3: dot(shifted, n) ≡ dot(sub3(origin, pl.point), n) + dot(scale(t, dir), n)
    // From steps 4,5: ≡ -numer + t * denom (by add congruence)
    let term1 = dot3(sub3(ray.origin, pl.point), n);
    let term2 = dot3(scale3(t, dir), n);
    additive_group_lemmas::lemma_add_congruence::<T>(
        term1, numer.neg(),
        term2, t.mul(denom),
    );
    // dot(shifted, n) ≡ -numer + t * denom
    T::axiom_eqv_transitive(
        dot3(shifted, n),
        term1.add(term2),
        numer.neg().add(t.mul(denom)),
    );

    // Step 7: t * denom = (numer/denom) * denom ≡ numer by div_mul_cancel
    field_lemmas::lemma_div_mul_cancel::<T>(numer, denom);

    // Step 8: -numer + t * denom ≡ -numer + numer
    additive_group_lemmas::lemma_add_congruence_right::<T>(
        numer.neg(), t.mul(denom), numer,
    );

    // Step 9: -numer + numer ≡ 0
    additive_group_lemmas::lemma_add_inverse_left::<T>(numer);

    // Final chain: dot(diff, n) ≡ 0
    T::axiom_eqv_transitive(
        dot3(diff, n),
        dot3(shifted, n),
        numer.neg().add(t.mul(denom)),
    );
    T::axiom_eqv_transitive(
        dot3(diff, n),
        numer.neg().add(t.mul(denom)),
        numer.neg().add(numer),
    );
    T::axiom_eqv_transitive(
        dot3(diff, n),
        numer.neg().add(numer),
        T::zero(),
    );
}

} // verus!
