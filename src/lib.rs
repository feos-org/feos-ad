pub mod eos;

mod core;
pub use core::{
    EquationOfStateAD, HelmholtzEnergyWrapper, IdealGasAD, NamedParameters, ParametersAD,
    PhaseEquilibriumAD, ResidualHelmholtzEnergy, StateAD, TotalHelmholtzEnergy,
};

#[cfg(feature = "parameter_fit")]
mod parameter_fit;
#[cfg(feature = "parameter_fit")]
pub use parameter_fit::{equilibrium_liquid_density, liquid_density, vapor_pressure};
