use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::embedding::from_int;
use verus_algebra::convex::two;
use verus_algebra::quadratic::{discriminant, quadratic_eval};
use verus_algebra::lemmas::ring_lemmas;
use verus_algebra::lemmas::additive_group_lemmas;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3, scale as scale3};
use verus_geometry::point3::*;
use verus_geometry::ray::ray_point_3d;
use crate::types::*;

verus! {

//  ---------------------------------------------------------------------------
//  Ray-sphere intersection coefficients
//  ---------------------------------------------------------------------------

///  Vector from sphere center to ray origin.
pub open spec fn rs_oc<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> Vec3<T> {
    sub3(ray.origin, sphere.center)
}

///  Quadratic coefficient A = norm_sq(dir).
pub open spec fn rs_quad_a<T: Ring>(ray: Ray3<T>) -> T {
    norm_sq3(ray.dir)
}

///  Quadratic coefficient B = 2 * dot(oc, dir).
pub open spec fn rs_quad_b<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> T {
    two::<T>().mul(dot3(rs_oc(ray, sphere), ray.dir))
}

///  Quadratic coefficient C = norm_sq(oc) - radius_sq.
pub open spec fn rs_quad_c<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> T {
    norm_sq3(rs_oc(ray, sphere)).sub(sphere.radius_sq)
}

///  Discriminant of the ray-sphere quadratic: B^2 - 4*A*C.
pub open spec fn rs_discriminant<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>) -> T {
    discriminant(rs_quad_a(ray), rs_quad_b(ray, sphere), rs_quad_c(ray, sphere))
}

//  ---------------------------------------------------------------------------
//  Division-free boolean test (stays in OrderedRing, no sqrt)
//  ---------------------------------------------------------------------------

///  Does the ray line intersect the sphere? (Division-free, no sqrt required.)
///  True iff disc >= 0.
pub open spec fn ray_hits_sphere_nodiv<T: OrderedRing>(
    ray: Ray3<T>, sphere: Sphere<T>,
) -> bool {
    let disc = rs_discriminant(ray, sphere);
    !disc.lt(T::zero()) //  disc >= 0
}

///  Full forward hit test: disc >= 0 and at least one non-negative root.
pub open spec fn ray_hits_sphere_forward<T: OrderedRing>(
    ray: Ray3<T>, sphere: Sphere<T>,
) -> bool {
    let a = rs_quad_a(ray);
    let b = rs_quad_b(ray, sphere);
    let c = rs_quad_c(ray, sphere);
    let disc = rs_discriminant(ray, sphere);
    &&& !a.eqv(T::zero())     //  non-degenerate ray direction
    &&& !disc.lt(T::zero())    //  disc >= 0
    //  At least one root >= 0: B <= 0 or C <= 0
    &&& (b.lt(T::zero()) || b.eqv(T::zero()) || c.lt(T::zero()) || c.eqv(T::zero()))
}

///  Normal at sphere hit point is radial: hit - center.
pub open spec fn sphere_normal<T: Ring>(hit: Point3<T>, sphere: Sphere<T>) -> Vec3<T> {
    sub3(hit, sphere.center)
}

//  ---------------------------------------------------------------------------
//  Lemma: sub3(ray_at(ray, t), center) ≡ oc + scale(t, dir)
//  ---------------------------------------------------------------------------

///  The displacement from sphere center to the ray point at t equals
///  oc + t*dir, where oc = origin - center.
proof fn lemma_rs_displacement<T: Ring>(ray: Ray3<T>, sphere: Sphere<T>, t: T)
    ensures
        sub3(ray_at(ray, t), sphere.center).eqv(
            rs_oc(ray, sphere).add(scale3(t, ray.dir))
        ),
{
    //  ray_at(ray, t) = add_vec3(origin, scale(t, dir))
    //  sub3(add_vec3(origin, scale(t, dir)), center) ≡ sub3(origin, center) + scale(t, dir)
    //  This is lemma_sub3_shift_add(origin, scale(t, dir), center)
    lemma_sub3_shift_add(ray.origin, scale3(t, ray.dir), sphere.center);
}

//  ---------------------------------------------------------------------------
//  Lemma: norm_sq(oc + t*dir) ≡ A*t² + B*t + norm_sq(oc)
//  ---------------------------------------------------------------------------

///  norm_sq(oc + t*dir) expands to norm_sq(dir)*t² + 2*dot(oc,dir)*t + norm_sq(oc).
///
///  That is: norm_sq(oc + t*dir) ≡ A*t*t + B*t + norm_sq(oc).
proof fn lemma_rs_norm_sq_expand<T: OrderedField>(ray: Ray3<T>, sphere: Sphere<T>, t: T)
    ensures
        norm_sq3(rs_oc(ray, sphere).add(scale3(t, ray.dir))).eqv(
            rs_quad_a(ray).mul(t).mul(t)
                .add(rs_quad_b(ray, sphere).mul(t))
                .add(norm_sq3(rs_oc(ray, sphere)))
        ),
{
    let oc = rs_oc(ray, sphere);
    let dir = ray.dir;
    let td = scale3(t, dir);

    //  Step 1: norm_sq(oc + td) ≡ norm_sq(oc) + 2*dot(oc, td) + norm_sq(td)
    verus_linalg::vec3::ops::lemma_norm_sq_add_expand::<T>(oc, td);
    //  Step 2: norm_sq(td) ≡ t*t * norm_sq(dir)
    verus_linalg::vec3::ops::lemma_norm_sq_scale::<T>(t, dir);
    //  Step 3: dot(oc, td) ≡ t * dot(oc, dir)
    verus_linalg::vec3::ops::lemma_dot_scale_right::<T>(t, oc, dir);

    //  After Step 1, Z3 knows:
    //    norm_sq(oc+td) ≡ norm_sq(oc) + two()*dot(oc, td) + norm_sq(td)
    //
    //  After Steps 2-3, Z3 knows:
    //    norm_sq(td) ≡ t*t*norm_sq(dir)
    //    dot(oc, td) ≡ t*dot(oc, dir)
    //
    //  We need to chain substitutions to reach the target:
    //    A*t*t + B*t + norm_sq(oc)
    //  where A = norm_sq(dir), B = two() * dot(oc, dir)

    let nsq_oc = norm_sq3(oc);
    let dot_oc_td = dot3(oc, td);
    let dot_oc_dir = dot3(oc, dir);
    let nsq_td = norm_sq3(td);
    let nsq_dir = norm_sq3(dir);

    //  Substitute dot: two()*dot(oc, td) ≡ two()*(t*dot(oc, dir)) = two()*t*dot(oc,dir)
    ring_lemmas::lemma_mul_congruence_right::<T>(two::<T>(), dot_oc_td, t.mul(dot_oc_dir));
    //  two()*dot(oc, td) ≡ two() * (t * dot(oc, dir))
    //  = (two() * t) * dot(oc, dir) by associativity... actually it's (two()*dot(oc,dir))*t by commutativity
    //  B*t = (two()*dot(oc,dir))*t
    //  two()*(t*dot(oc,dir)) vs (two()*dot(oc,dir))*t
    //  = two()*t*dot(oc,dir) vs two()*dot(oc,dir)*t — equal by commutativity of mul

    //  two()*(t*dot(oc,dir)) ≡ (two()*t)*dot(oc,dir) by associativity
    T::axiom_mul_associative(two::<T>(), t, dot_oc_dir);
    T::axiom_eqv_symmetric(two::<T>().mul(t).mul(dot_oc_dir), two::<T>().mul(t.mul(dot_oc_dir)));
    //  two()*dot(oc,td) ≡ (two()*t)*dot(oc,dir) by transitivity
    T::axiom_eqv_transitive(
        two::<T>().mul(dot_oc_td),
        two::<T>().mul(t.mul(dot_oc_dir)),
        two::<T>().mul(t).mul(dot_oc_dir),
    );

    //  (two()*t)*dot(oc,dir) ≡ (two()*dot(oc,dir))*t by commutativity of mul
    T::axiom_mul_commutative(two::<T>().mul(t), dot_oc_dir);
    //  dot(oc,dir)*(two()*t) ≡ (dot(oc,dir)*two())*t
    T::axiom_mul_associative(dot_oc_dir, two::<T>(), t);
    T::axiom_eqv_symmetric(dot_oc_dir.mul(two::<T>()).mul(t), dot_oc_dir.mul(two::<T>().mul(t)));
    //  So: dot(oc,dir)*(two()*t) ≡ (dot(oc,dir)*two())*t
    //  And: (two()*t)*dot(oc,dir) ≡ dot(oc,dir)*(two()*t) (from commutative above)
    T::axiom_eqv_transitive(
        two::<T>().mul(t).mul(dot_oc_dir),
        dot_oc_dir.mul(two::<T>().mul(t)),
        dot_oc_dir.mul(two::<T>()).mul(t),
    );
    //  dot(oc,dir)*two() ≡ two()*dot(oc,dir) by commutativity
    T::axiom_mul_commutative(dot_oc_dir, two::<T>());
    //  (dot(oc,dir)*two())*t ≡ (two()*dot(oc,dir))*t by congruence
    T::axiom_eqv_reflexive(t);
    ring_lemmas::lemma_mul_congruence::<T>(
        dot_oc_dir.mul(two::<T>()), two::<T>().mul(dot_oc_dir), t, t,
    );
    //  Full chain: two()*dot(oc,td) ≡ B*t
    let b = rs_quad_b(ray, sphere); //  = two()*dot(oc,dir)
    T::axiom_eqv_transitive(
        two::<T>().mul(dot_oc_td),
        two::<T>().mul(t).mul(dot_oc_dir),
        dot_oc_dir.mul(two::<T>()).mul(t),
    );
    T::axiom_eqv_transitive(
        two::<T>().mul(dot_oc_td),
        dot_oc_dir.mul(two::<T>()).mul(t),
        b.mul(t),
    );

    //  Substitute nsq: nsq_td ≡ t*t*nsq_dir = A*t*t... wait, A*t*t = nsq_dir*t*t
    //  But nsq_scale gives nsq_td ≡ t.mul(t).mul(nsq_dir) which is (t*t)*A
    //  We need A*t*t = nsq_dir.mul(t).mul(t)
    //  (t*t)*nsq_dir vs nsq_dir*t*t — commute the outer mul
    T::axiom_mul_commutative(t.mul(t), nsq_dir);
    //  nsq_td ≡ nsq_dir*(t*t) = A*(t*t)
    T::axiom_eqv_transitive(nsq_td, t.mul(t).mul(nsq_dir), nsq_dir.mul(t.mul(t)));
    //  A*(t*t) ≡ (A*t)*t by associativity (reverse)
    T::axiom_mul_associative(nsq_dir, t, t);
    T::axiom_eqv_symmetric(nsq_dir.mul(t).mul(t), nsq_dir.mul(t.mul(t)));
    //  nsq_td ≡ A*t*t
    T::axiom_eqv_transitive(nsq_td, nsq_dir.mul(t.mul(t)), nsq_dir.mul(t).mul(t));

    let a = rs_quad_a(ray); //  = nsq_dir

    //  Now substitute into the expanded norm_sq:
    //  norm_sq(oc+td) ≡ nsq_oc + two()*dot(oc,td) + nsq_td
    //                 ≡ nsq_oc + B*t + A*t*t   (after substitutions)

    //  Apply congruences to the sum:
    //  nsq_oc + two()*dot(oc,td) + nsq_td
    //  The add_expand gives: nsq_oc.add(two()*dot(oc,td)).add(nsq_td)
    //  We need:              a.mul(t).mul(t).add(b.mul(t)).add(nsq_oc)

    //  First, substitute the last summand: nsq_td → A*t*t
    T::axiom_eqv_reflexive(nsq_oc.add(two::<T>().mul(dot_oc_td)));
    additive_group_lemmas::lemma_add_congruence_right::<T>(
        nsq_oc.add(two::<T>().mul(dot_oc_td)),
        nsq_td,
        a.mul(t).mul(t),
    );

    //  Next, substitute the middle: two()*dot(oc,td) → B*t
    T::axiom_eqv_reflexive(nsq_oc);
    additive_group_lemmas::lemma_add_congruence::<T>(
        nsq_oc, nsq_oc,
        two::<T>().mul(dot_oc_td), b.mul(t),
    );
    //  nsq_oc + two()*dot(oc,td) ≡ nsq_oc + B*t

    //  Congruence on the outer add:
    additive_group_lemmas::lemma_add_congruence::<T>(
        nsq_oc.add(two::<T>().mul(dot_oc_td)),
        nsq_oc.add(b.mul(t)),
        nsq_td,
        a.mul(t).mul(t),
    );

    //  norm_sq(oc+td) ≡ nsq_oc + B*t + A*t*t
    let expanded = nsq_oc.add(two::<T>().mul(dot_oc_td)).add(nsq_td);
    let substituted = nsq_oc.add(b.mul(t)).add(a.mul(t).mul(t));
    T::axiom_eqv_transitive(
        norm_sq3(oc.add(td)),
        expanded,
        substituted,
    );

    //  Now rearrange: nsq_oc + B*t + A*t*t ≡ A*t*t + B*t + nsq_oc
    //  This is (x + y) + z ≡ (z + y) + x where x=nsq_oc, y=B*t, z=A*t*t
    //  First: (nsq_oc + B*t) + A*t*t ≡ A*t*t + (nsq_oc + B*t) by commutativity
    T::axiom_add_commutative(nsq_oc.add(b.mul(t)), a.mul(t).mul(t));
    //  A*t*t + (nsq_oc + B*t) ≡ (A*t*t + nsq_oc) + B*t by ... hmm

    //  Actually, simpler: use add_commutative + associative to move A*t*t to front
    //  (nsq_oc + B*t) + A*t*t
    //  ≡ A*t*t + (nsq_oc + B*t) by commutativity
    //  ≡ (A*t*t + nsq_oc) + B*t by associativity (reverse)
    //  Hmm, we need (A*t*t + B*t) + nsq_oc

    //  Let me use: x + y + z ≡ z + x + y first...
    //  Actually: (x + y) + z ≡ (z + x) + y by swap
    //  Or use add_swap_last style

    //  (nsq_oc + B*t) + A*t*t ≡ (nsq_oc + A*t*t) + B*t by add_swap_last
    T::axiom_add_associative(nsq_oc, b.mul(t), a.mul(t).mul(t));
    T::axiom_add_commutative(b.mul(t), a.mul(t).mul(t));
    additive_group_lemmas::lemma_add_congruence_right::<T>(nsq_oc, b.mul(t).add(a.mul(t).mul(t)), a.mul(t).mul(t).add(b.mul(t)));
    T::axiom_eqv_transitive(
        nsq_oc.add(b.mul(t)).add(a.mul(t).mul(t)),
        nsq_oc.add(b.mul(t).add(a.mul(t).mul(t))),
        nsq_oc.add(a.mul(t).mul(t).add(b.mul(t))),
    );
    T::axiom_add_associative(nsq_oc, a.mul(t).mul(t), b.mul(t));
    T::axiom_eqv_symmetric(
        nsq_oc.add(a.mul(t).mul(t)).add(b.mul(t)),
        nsq_oc.add(a.mul(t).mul(t).add(b.mul(t))),
    );
    T::axiom_eqv_transitive(
        nsq_oc.add(b.mul(t)).add(a.mul(t).mul(t)),
        nsq_oc.add(a.mul(t).mul(t).add(b.mul(t))),
        nsq_oc.add(a.mul(t).mul(t)).add(b.mul(t)),
    );

    //  (nsq_oc + A*t*t) + B*t ≡ (A*t*t + nsq_oc) + B*t by commuting inner
    T::axiom_add_commutative(nsq_oc, a.mul(t).mul(t));
    T::axiom_add_congruence_left(nsq_oc.add(a.mul(t).mul(t)), a.mul(t).mul(t).add(nsq_oc), b.mul(t));
    T::axiom_eqv_transitive(
        nsq_oc.add(b.mul(t)).add(a.mul(t).mul(t)),
        nsq_oc.add(a.mul(t).mul(t)).add(b.mul(t)),
        a.mul(t).mul(t).add(nsq_oc).add(b.mul(t)),
    );

    //  (A*t*t + nsq_oc) + B*t ≡ (A*t*t + B*t) + nsq_oc by add_swap_last
    T::axiom_add_associative(a.mul(t).mul(t), nsq_oc, b.mul(t));
    T::axiom_add_commutative(nsq_oc, b.mul(t));
    additive_group_lemmas::lemma_add_congruence_right::<T>(a.mul(t).mul(t), nsq_oc.add(b.mul(t)), b.mul(t).add(nsq_oc));
    T::axiom_eqv_transitive(
        a.mul(t).mul(t).add(nsq_oc).add(b.mul(t)),
        a.mul(t).mul(t).add(nsq_oc.add(b.mul(t))),
        a.mul(t).mul(t).add(b.mul(t).add(nsq_oc)),
    );
    T::axiom_add_associative(a.mul(t).mul(t), b.mul(t), nsq_oc);
    T::axiom_eqv_symmetric(
        a.mul(t).mul(t).add(b.mul(t)).add(nsq_oc),
        a.mul(t).mul(t).add(b.mul(t).add(nsq_oc)),
    );
    T::axiom_eqv_transitive(
        a.mul(t).mul(t).add(nsq_oc).add(b.mul(t)),
        a.mul(t).mul(t).add(b.mul(t).add(nsq_oc)),
        a.mul(t).mul(t).add(b.mul(t)).add(nsq_oc),
    );

    //  Full chain:
    //  norm_sq(oc+td) ≡ nsq_oc + B*t + A*t*t ≡ A*t*t + B*t + nsq_oc
    T::axiom_eqv_transitive(
        norm_sq3(oc.add(td)),
        substituted,
        a.mul(t).mul(t).add(nsq_oc).add(b.mul(t)),
    );
    T::axiom_eqv_transitive(
        norm_sq3(oc.add(td)),
        a.mul(t).mul(t).add(nsq_oc).add(b.mul(t)),
        a.mul(t).mul(t).add(b.mul(t)).add(nsq_oc),
    );
}

//  ---------------------------------------------------------------------------
//  Main theorem: hit point lies on sphere
//  ---------------------------------------------------------------------------

///  If quadratic_eval(A, B, C, t) ≡ 0, then the ray point at t is on the sphere.
pub proof fn lemma_rs_hit_point_on_sphere_at_t<T: OrderedField>(
    ray: Ray3<T>, sphere: Sphere<T>, t: T,
)
    requires
        quadratic_eval(rs_quad_a(ray), rs_quad_b(ray, sphere), rs_quad_c(ray, sphere), t).eqv(T::zero()),
    ensures
        point_on_sphere(ray_at(ray, t), sphere),
{
    let oc = rs_oc(ray, sphere);
    let dir = ray.dir;
    let a = rs_quad_a(ray);
    let b = rs_quad_b(ray, sphere);
    let c_val = rs_quad_c(ray, sphere);
    let nsq_oc = norm_sq3(oc);
    let td = scale3(t, dir);

    //  Capture the requires clause fact with the local names
    let qeval_early = quadratic_eval(a, b, c_val, t);
    assert(qeval_early.eqv(T::zero()));

    //  Step 1: sub3(ray_at(ray, t), center) ≡ oc + td
    lemma_rs_displacement(ray, sphere, t);
    let diff = sub3(ray_at(ray, t), sphere.center);
    let oc_td = oc.add(td);

    //  Step 2: norm_sq(diff) ≡ norm_sq(oc + td) by congruence
    verus_linalg::vec3::ops::lemma_norm_sq_congruence::<T>(diff, oc_td);

    //  Step 3: norm_sq(oc + td) ≡ A*t*t + B*t + nsq_oc
    lemma_rs_norm_sq_expand(ray, sphere, t);

    //  Step 4: norm_sq(diff) ≡ A*t*t + B*t + nsq_oc by transitivity
    T::axiom_eqv_transitive(
        norm_sq3(diff),
        norm_sq3(oc_td),
        a.mul(t).mul(t).add(b.mul(t)).add(nsq_oc),
    );

    //  Step 5: A*t*t + B*t + nsq_oc = quadratic_eval(A, B, C, t) + r^2
    //  Because C = nsq_oc - r^2, so nsq_oc = C + r^2.
    //  quadratic_eval(A, B, C, t) = A*t*t + B*t + C
    //  quadratic_eval + r^2 = A*t*t + B*t + C + r^2 = A*t*t + B*t + nsq_oc
    //
    //  A*t*t + B*t + nsq_oc ≡ (A*t*t + B*t + C) + r^2
    //  Because nsq_oc ≡ C + r^2 (i.e., nsq_oc ≡ nsq_oc - r^2 + r^2)
    let r2 = sphere.radius_sq;
    additive_group_lemmas::lemma_sub_then_add_cancel::<T>(nsq_oc, r2);
    //  nsq_oc.sub(r2).add(r2) ≡ nsq_oc
    //  i.e., C.add(r2) ≡ nsq_oc
    T::axiom_eqv_symmetric(c_val.add(r2), nsq_oc);
    //  nsq_oc ≡ C + r2

    //  Substitute in the sum: A*t*t + B*t + nsq_oc ≡ A*t*t + B*t + (C + r2)
    T::axiom_eqv_reflexive(a.mul(t).mul(t).add(b.mul(t)));
    additive_group_lemmas::lemma_add_congruence::<T>(
        a.mul(t).mul(t).add(b.mul(t)), a.mul(t).mul(t).add(b.mul(t)),
        nsq_oc, c_val.add(r2),
    );

    //  (A*t*t + B*t) + (C + r2) ≡ ((A*t*t + B*t) + C) + r2 by associativity (reverse)
    T::axiom_add_associative(a.mul(t).mul(t).add(b.mul(t)), c_val, r2);
    T::axiom_eqv_symmetric(
        a.mul(t).mul(t).add(b.mul(t)).add(c_val).add(r2),
        a.mul(t).mul(t).add(b.mul(t)).add(c_val.add(r2)),
    );

    //  Chain: A*t*t + B*t + nsq_oc ≡ quadratic_eval(A,B,C,t) + r2
    let qeval = a.mul(t).mul(t).add(b.mul(t)).add(c_val);
    T::axiom_eqv_transitive(
        a.mul(t).mul(t).add(b.mul(t)).add(nsq_oc),
        a.mul(t).mul(t).add(b.mul(t)).add(c_val.add(r2)),
        qeval.add(r2),
    );

    //  Chain: norm_sq(diff) ≡ qeval + r2
    T::axiom_eqv_transitive(
        norm_sq3(diff),
        a.mul(t).mul(t).add(b.mul(t)).add(nsq_oc),
        qeval.add(r2),
    );

    //  Step 6: qeval ≡ 0 (given), so qeval + r2 ≡ 0 + r2 ≡ r2
    //  qeval uses a.mul(t).mul(t) but quadratic_eval uses a.mul(t.mul(t)).
    //  Bridge via associativity: (a*t)*t ≡ a*(t*t)
    T::axiom_mul_associative(a, t, t);
    //  a.mul(t).mul(t) ≡ a.mul(t.mul(t)), so qeval ≡ qeval_early
    T::axiom_eqv_reflexive(b.mul(t).add(c_val));
    additive_group_lemmas::lemma_add_congruence::<T>(
        a.mul(t).mul(t), a.mul(t.mul(t)),
        b.mul(t).add(c_val), b.mul(t).add(c_val),
    );
    //  a.mul(t).mul(t).add(b.mul(t).add(c_val)) ≡ a.mul(t.mul(t)).add(b.mul(t).add(c_val))
    //  But qeval = a.mul(t).mul(t).add(b.mul(t)).add(c_val) = (a*t*t + b*t) + c
    //  qeval_early = a.mul(t.mul(t)).add(b.mul(t)).add(c_val) = (a*(t*t) + b*t) + c
    //  Need: (a*t*t + b*t) + c ≡ (a*(t*t) + b*t) + c
    T::axiom_eqv_reflexive(b.mul(t));
    additive_group_lemmas::lemma_add_congruence::<T>(
        a.mul(t).mul(t), a.mul(t.mul(t)),
        b.mul(t), b.mul(t),
    );
    //  a.mul(t).mul(t).add(b.mul(t)) ≡ a.mul(t.mul(t)).add(b.mul(t))
    T::axiom_eqv_reflexive(c_val);
    additive_group_lemmas::lemma_add_congruence::<T>(
        a.mul(t).mul(t).add(b.mul(t)), a.mul(t.mul(t)).add(b.mul(t)),
        c_val, c_val,
    );
    //  qeval ≡ qeval_early
    T::axiom_eqv_transitive(qeval, qeval_early, T::zero());
    T::axiom_eqv_reflexive(r2);
    additive_group_lemmas::lemma_add_congruence::<T>(qeval, T::zero(), r2, r2);
    additive_group_lemmas::lemma_add_zero_left::<T>(r2);
    T::axiom_eqv_transitive(qeval.add(r2), T::zero().add(r2), r2);

    //  Final: norm_sq(diff) ≡ r2
    T::axiom_eqv_transitive(norm_sq3(diff), qeval.add(r2), r2);
}

} //  verus!
