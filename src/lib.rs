mod core;
pub mod eos;
mod parameter_fit;
pub use core::{
    EquationOfStateAD, HelmholtzEnergyWrapper, IdealGasAD, NamedParameters, ParametersAD,
    PhaseEquilibriumAD, ResidualHelmholtzEnergy, StateAD, TotalHelmholtzEnergy,
};
pub use parameter_fit::vapor_pressure;
