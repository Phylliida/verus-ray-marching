use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::Vec3;
use verus_geometry::point3::*;
use crate::types::*;

verus! {

// ---------------------------------------------------------------------------
// CSG ray interval types
// ---------------------------------------------------------------------------

/// A ray interval [t_enter, t_exit] with surface normals at entry and exit.
pub struct RayInterval<T: Ring> {
    pub t_enter: T,
    pub t_exit: T,
    pub normal_enter: Vec3<T>,
    pub normal_exit: Vec3<T>,
}

/// A ray interval is valid: t_enter <= t_exit.
pub open spec fn interval_valid<T: OrderedRing>(iv: RayInterval<T>) -> bool {
    iv.t_enter.le(iv.t_exit)
}

/// An interval list is sorted and non-overlapping.
pub open spec fn intervals_sorted<T: OrderedRing>(ivs: Seq<RayInterval<T>>) -> bool {
    &&& forall|i: int| 0 <= i < ivs.len() ==> interval_valid(ivs[i])
    &&& forall|i: int| 0 <= i < ivs.len() - 1 ==>
        ivs[i].t_exit.le(ivs[i + 1].t_enter) || ivs[i].t_exit.lt(ivs[i + 1].t_enter)
}

// ---------------------------------------------------------------------------
// CSG node
// ---------------------------------------------------------------------------

/// CSG tree node.
pub enum CsgOp {
    Union,
    Intersection,
    Difference,
}

// ---------------------------------------------------------------------------
// CSG interval operations
// ---------------------------------------------------------------------------

/// Union of two interval lists (concatenation).
/// The point-set semantics is: a point is in the union iff it's in A or B.
pub open spec fn csg_union_intervals<T: OrderedRing>(
    a: Seq<RayInterval<T>>, b: Seq<RayInterval<T>>,
) -> Seq<RayInterval<T>> {
    a.add(b)
}

/// Intersection of two sorted interval lists.
pub open spec fn csg_intersect_intervals<T: OrderedRing>(
    a: Seq<RayInterval<T>>, b: Seq<RayInterval<T>>,
) -> Seq<RayInterval<T>> {
    Seq::empty() // placeholder for proper sweep-line spec
}

/// Difference of two sorted interval lists: A minus B.
pub open spec fn csg_difference_intervals<T: OrderedRing>(
    a: Seq<RayInterval<T>>, b: Seq<RayInterval<T>>,
) -> Seq<RayInterval<T>> {
    Seq::empty() // placeholder
}

/// A parameter t is covered by some interval in the list.
pub open spec fn t_in_intervals<T: OrderedRing>(t: T, ivs: Seq<RayInterval<T>>) -> bool {
    exists|i: int| 0 <= i < ivs.len() && ivs[i].t_enter.le(t) && t.le(ivs[i].t_exit)
}

/// Point-set correctness of union: t in union(A,B) iff t in A or t in B.
///
/// Since union is concatenation, an index into a.add(b) either falls in the
/// a-range [0, a.len()) or the b-range [a.len(), a.len()+b.len()).
pub proof fn lemma_csg_union_correct<T: OrderedRing>(
    t: T, a: Seq<RayInterval<T>>, b: Seq<RayInterval<T>>,
)
    ensures
        t_in_intervals(t, csg_union_intervals(a, b)) <==>
            (t_in_intervals(t, a) || t_in_intervals(t, b)),
{
    let ab = a.add(b);

    // Forward: t in ab → t in a or t in b
    if t_in_intervals(t, ab) {
        let i = choose|i: int| 0 <= i < ab.len() && ab[i].t_enter.le(t) && t.le(ab[i].t_exit);
        if i < a.len() as int {
            // Index in a's range: ab[i] == a[i]
            assert(a[i].t_enter.le(t) && t.le(a[i].t_exit));
            assert(t_in_intervals(t, a));
        } else {
            // Index in b's range: ab[i] == b[i - a.len()]
            let j = i - a.len() as int;
            assert(0 <= j < b.len());
            assert(b[j].t_enter.le(t) && t.le(b[j].t_exit));
            assert(t_in_intervals(t, b));
        }
    }

    // Backward: t in a or t in b → t in ab
    if t_in_intervals(t, a) {
        let i = choose|i: int| 0 <= i < a.len() && a[i].t_enter.le(t) && t.le(a[i].t_exit);
        assert(0 <= i < ab.len());
        assert(ab[i].t_enter.le(t) && t.le(ab[i].t_exit));
        assert(t_in_intervals(t, ab));
    }
    if t_in_intervals(t, b) {
        let j = choose|j: int| 0 <= j < b.len() && b[j].t_enter.le(t) && t.le(b[j].t_exit);
        let i = j + a.len() as int;
        assert(0 <= i < ab.len());
        assert(ab[i] == b[j]);
        assert(ab[i].t_enter.le(t) && t.le(ab[i].t_exit));
        assert(t_in_intervals(t, ab));
    }
}

} // verus!
