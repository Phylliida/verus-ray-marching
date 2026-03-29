#[cfg(verus_keep_ghost)]
use verus_rational::rational::Rational;

///  The scalar model type for all runtime ray marching: verified rational numbers.
#[cfg(verus_keep_ghost)]
pub type RationalModel = Rational;

pub mod types;
pub mod ray_plane;
pub mod ray_sphere;
pub mod ray_box;
pub mod ray_cylinder;
pub mod normals;
pub mod fractal;
pub mod scene;
