use vstd::prelude::*;
use verus_algebra::traits::*;
use verus_algebra::lemmas::ring_lemmas;
use verus_algebra::lemmas::ordered_ring_lemmas;
use verus_algebra::lemmas::additive_group_lemmas;
use verus_linalg::vec3::Vec3;
use verus_linalg::vec3::ops::{dot as dot3, norm_sq as norm_sq3, scale as scale3};
use verus_geometry::point3::*;
use crate::types::*;

verus! {

//  ---------------------------------------------------------------------------
//  Lambertian shading model
//  ---------------------------------------------------------------------------

///  Light direction from a surface point toward a point light source.
pub open spec fn light_direction<T: Ring>(
    hit: Point3<T>, light_pos: Point3<T>,
) -> Vec3<T> {
    sub3(light_pos, hit)
}

///  Raw (unclamped) Lambertian shade factor: dot(normal, light_dir).
///  Positive when the surface faces the light.
pub open spec fn lambertian_raw<T: Ring>(normal: Vec3<T>, light_dir: Vec3<T>) -> T {
    dot3(normal, light_dir)
}

///  Clamped Lambertian shade factor: max(0, dot(normal, light_dir)).
///  Always non-negative.
pub open spec fn lambertian_clamped<T: OrderedRing>(
    normal: Vec3<T>, light_dir: Vec3<T>,
) -> T {
    let raw = lambertian_raw(normal, light_dir);
    if raw.lt(T::zero()) {
        T::zero()
    } else {
        raw
    }
}

//  ---------------------------------------------------------------------------
//  Proof: clamped shade is non-negative
//  ---------------------------------------------------------------------------

///  lambertian_clamped(n, l) >= 0.
pub proof fn lemma_lambertian_clamped_nonneg<T: OrderedRing>(
    normal: Vec3<T>, light_dir: Vec3<T>,
)
    ensures
        !lambertian_clamped(normal, light_dir).lt(T::zero()),
{
    let raw = lambertian_raw(normal, light_dir);
    if raw.lt(T::zero()) {
        //  Result is zero, and 0 is not < 0
        ordered_ring_lemmas::lemma_lt_irreflexive::<T>(T::zero());
    } else {
        //  raw is not < 0, which is exactly what we need
    }
}

//  ---------------------------------------------------------------------------
//  Proof: Cauchy-Schwarz bounds the raw shade
//  ---------------------------------------------------------------------------

///  |dot(n, l)|² ≤ norm_sq(n) * norm_sq(l)  (Cauchy-Schwarz).
///
///  Therefore: lambertian_clamped(n, l)² ≤ norm_sq(n) * norm_sq(l).
///  When both n and l are unit vectors (norm_sq ≡ 1), the shade ≤ 1.
pub proof fn lemma_lambertian_cauchy_schwarz<T: OrderedField>(
    normal: Vec3<T>, light_dir: Vec3<T>,
)
    ensures
        //  dot(n, l)^2 ≤ norm_sq(n) * norm_sq(l)
        dot3(normal, light_dir).mul(dot3(normal, light_dir)).le(
            norm_sq3(normal).mul(norm_sq3(light_dir))
        ),
{
    verus_linalg::vec3::ops::lemma_cauchy_schwarz::<T>(normal, light_dir);
}

//  ---------------------------------------------------------------------------
//  Spec: shaded color = material_color * shade_factor
//  ---------------------------------------------------------------------------

///  Apply a shade factor to a single color channel.
pub open spec fn shade_channel<T: Ring>(channel: T, shade: T) -> T {
    channel.mul(shade)
}

///  Apply shade factor to RGB color (r, g, b).
pub open spec fn shade_color<T: Ring>(
    r: T, g: T, b: T, shade: T,
) -> (T, T, T) {
    (shade_channel(r, shade), shade_channel(g, shade), shade_channel(b, shade))
}

//  ---------------------------------------------------------------------------
//  Full per-pixel lighting equation
//  ---------------------------------------------------------------------------

///  The Lambertian pixel color for a single point light.
///
///  shade = max(0, dot(normal, light_dir))
///  color = material_rgb * shade
///
///  Note: no normalization — caller provides pre-normalized vectors for
///  shade ∈ [0, 1], or accepts unnormalized shade for artistic effect.
pub open spec fn pixel_shade<T: OrderedRing>(
    normal: Vec3<T>,
    hit: Point3<T>,
    light_pos: Point3<T>,
    mat_r: T, mat_g: T, mat_b: T,
) -> (T, T, T) {
    let light_dir = light_direction(hit, light_pos);
    let shade = lambertian_clamped(normal, light_dir);
    shade_color(mat_r, mat_g, mat_b, shade)
}

//  ---------------------------------------------------------------------------
//  Ambient + diffuse combined
//  ---------------------------------------------------------------------------

///  Combined ambient + Lambertian diffuse lighting.
///
///  final_channel = ambient * mat + diffuse * mat * shade
///                = mat * (ambient + diffuse * shade)
pub open spec fn ambient_diffuse_shade<T: OrderedRing>(
    channel: T, ambient: T, diffuse: T, shade: T,
) -> T {
    channel.mul(ambient.add(diffuse.mul(shade)))
}

} //  verus!
