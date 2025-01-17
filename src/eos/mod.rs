//! A collection of equation of state models.
use std::collections::HashMap;

mod gc_pcsaft;
pub(crate) mod ideal_gas;
pub(crate) mod pcsaft;
pub use gc_pcsaft::{GcPcSaft, GcPcSaftParameters};
pub use ideal_gas::Joback;
pub use pcsaft::{PcSaftBinary, PcSaftPure};

/// Input for group-contribution models that allows for derivatives.
pub struct ChemicalRecord<D> {
    pub groups: HashMap<&'static str, D>,
    pub bonds: HashMap<[&'static str; 2], D>,
}

impl<D> ChemicalRecord<D> {
    pub fn new(groups: HashMap<&'static str, D>, bonds: HashMap<[&'static str; 2], D>) -> Self {
        Self { groups, bonds }
    }
}
