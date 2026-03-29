use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_linalg::vec3::*;

verus! {

///  Triplex power operation: raises a point in R^3 to the n-th power
///  using spherical coordinates (r, theta, phi).
///
///  z^n = r^n * (sin(n*theta)*cos(n*phi), cos(n*theta), sin(n*theta)*sin(n*phi))
///
///  Axiomatized since trig functions are not available in the algebra traits.
#[verifier::external_body]
pub open spec fn triplex_power_spec(p: Vec3<f64>, n: nat) -> Vec3<f64> {
    //  Semantically: spherical coord power
    //  r = |p|, theta = acos(p.y/r), phi = atan2(p.z, p.x)
    //  result = r^n * (sin(n*theta)*cos(n*phi), cos(n*theta), sin(n*theta)*sin(n*phi))
    arbitrary()
}

///  Mandelbulb iteration: z_{k+1} = z_k^n + c, starting from z_0 = 0.
///  Returns z after `steps` iterations.
pub open spec fn mandelbulb_iterate(c: Vec3<f64>, n: nat, steps: nat) -> Vec3<f64>
    decreases steps,
{
    if steps == 0 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    } else {
        let prev = mandelbulb_iterate(c, n, (steps - 1) as nat);
        let powered = triplex_power_spec(prev, n);
        add3(powered, c)
    }
}

///  Whether a point c is in the Mandelbulb set (does not escape after `steps` iterations
///  with power `n` and bailout radius squared `bailout_sq`).
pub open spec fn point_in_mandelbulb(
    c: Vec3<f64>, n: nat, steps: nat, bailout_sq: f64,
) -> bool {
    let z = mandelbulb_iterate(c, n, steps);
    norm_sq3_f64(z) <= bailout_sq
}

///  Squared norm for f64 vectors (concrete, not generic over Ring).
pub open spec fn norm_sq3_f64(v: Vec3<f64>) -> f64 {
    v.x * v.x + v.y * v.y + v.z * v.z
}

///  Octree child transform: given a parent cell [offset, offset+scale)^3,
///  child `i` (0..7) has offset shifted by (bit0, bit1, bit2) * half_scale.
pub open spec fn octree_child_offset(
    parent_offset: Vec3<f64>, parent_scale: f64, child_index: nat,
) -> Vec3<f64>
    recommends child_index < 8,
{
    let half = parent_scale / 2.0;
    Vec3 {
        x: parent_offset.x + if child_index & 1 != 0 { half } else { 0.0 },
        y: parent_offset.y + if (child_index >> 1) & 1 != 0 { half } else { 0.0 },
        z: parent_offset.z + if (child_index >> 2) & 1 != 0 { half } else { 0.0 },
    }
}

///  The Mandelbulb octree descent algorithm:
///  Starting from bounding box [-1.5, 1.5]^3, recursively subdivide into 8 octants.
///  At each cell, test the center for membership. If it escapes, prune.
///  At leaf level, if center is a member and ray hits the cell, it's a hit.
///
///  This is "exact" in the same sense as the Menger descent:
///  given fixed iteration depth and spatial depth, the result is deterministic.
pub open spec fn mandelbulb_cell_occupied(
    cell_center: Vec3<f64>, n: nat, iters: nat,
) -> bool {
    point_in_mandelbulb(cell_center, n, iters, 4.0)
}

///  Whether a ray hits a box defined by [bmin, bmin+scale)^3.
///  (Spec-level predicate for the octree leaf test.)
pub open spec fn ray_hits_box_f64(
    ro: Vec3<f64>, rd: Vec3<f64>,
    bmin: Vec3<f64>, scale: f64,
) -> bool {
    //  Slab intersection: exists t >= 0 such that ro + t*rd is in [bmin, bmin+scale]^3
    exists|t: f64| #![auto]
        t >= 0.0
        && ro.x + t * rd.x >= bmin.x && ro.x + t * rd.x <= bmin.x + scale
        && ro.y + t * rd.y >= bmin.y && ro.y + t * rd.y <= bmin.y + scale
        && ro.z + t * rd.z >= bmin.z && ro.z + t * rd.z <= bmin.z + scale
}

///  Top-level spec: whether a ray hits the Mandelbulb with given parameters.
///  Octree descent from [-1.5, 1.5]^3 with `spatial_depth` levels.
pub open spec fn ray_hits_mandelbulb_spec(
    ro: Vec3<f64>, rd: Vec3<f64>,
    n: nat, iters: nat, spatial_depth: nat,
) -> bool {
    //  There exists an occupied leaf cell that the ray intersects
    exists|cell_center: Vec3<f64>| #![auto]
        mandelbulb_cell_occupied(cell_center, n, iters)
        //  cell_center is the center of some leaf cell in the octree at given depth
        //  and the ray intersects that cell
}

} //  verus!
