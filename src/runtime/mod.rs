#[cfg(verus_keep_ghost)]
use verus_rational::rational::Rational;

/// The scalar model type for all runtime ray marching: verified rational numbers.
#[cfg(verus_keep_ghost)]
pub type RationalModel = Rational;
