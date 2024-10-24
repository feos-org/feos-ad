use crate::HelmholtzEnergy;
use feos_core::{Components, IdealGas};
use ndarray::{arr1, Array1};
use num_dual::DualNum;

const RGAS: f64 = 6.022140857 * 1.38064852;
const T0: f64 = 298.15;
const T0_2: f64 = 298.15 * 298.15;
const T0_3: f64 = T0 * T0_2;
const T0_4: f64 = T0_2 * T0_2;
const T0_5: f64 = T0 * T0_4;
const P0: f64 = 1.0e5;
const A3: f64 = 1e-30;
const KB: f64 = 1.38064852e-23;

#[derive(Clone, Copy)]
pub struct Joback {
    parameters: [f64; 5],
}

impl Joback {
    pub fn new<D: DualNum<f64>>(parameters: [D; 5]) -> Self {
        Self {
            parameters: parameters.map(|p| p.re()),
        }
    }

    fn ln_lambda3<D: DualNum<f64> + Copy>(parameters: [D; 5], temperature: D) -> D {
        let [a, b, c, d, e] = parameters;
        let t = temperature;
        let t2 = t * t;
        let t3 = t2 * t;
        let t4 = t2 * t2;
        let f = (temperature * KB / (P0 * A3)).ln();
        let h = (t2 - T0_2) * 0.5 * b
            + (t3 - T0_3) * c / 3.0
            + (t4 - T0_4) * 0.25 * d
            + (t4 * t - T0_5) * 0.2 * e
            + (t - T0) * a;
        let s = (t - T0) * b
            + (t2 - T0_2) * 0.5 * c
            + (t3 - T0_3) * d / 3.0
            + (t4 - T0_4) * 0.25 * e
            + (t / T0).ln() * a;
        (h - t * s) / (t * RGAS) + f
    }
}

impl HelmholtzEnergy<5> for Joback {
    fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: [D; 5],
        temperature: D,
        density: D,
    ) -> D {
        let ln_lambda_3 = Self::ln_lambda3(parameters, temperature);
        density * (density.ln() + ln_lambda_3 - 1.0) * temperature
    }
}

impl Components for Joback {
    fn components(&self) -> usize {
        1
    }

    fn subset(&self, _: &[usize]) -> Self {
        *self
    }
}

impl IdealGas for Joback {
    fn ln_lambda3<D: DualNum<f64> + Copy>(&self, temperature: D) -> Array1<D> {
        let parameters = self.parameters.map(D::from);
        arr1(&[Self::ln_lambda3(parameters, temperature)])
    }

    fn ideal_gas_model(&self) -> String {
        "Joback".into()
    }
}

#[cfg(test)]
pub mod test {
    use super::{HelmholtzEnergy as _, Joback as JobackAD};
    use approx::assert_relative_eq;
    use feos::ideal_gas::{Joback, JobackRecord};
    use feos_core::parameter::Parameter;
    use feos_core::{Contributions::IdealGas, EosResult, EquationOfState, ReferenceSystem, State};
    use ndarray::arr1;
    use quantity::{KELVIN, KILO, METER, MOL};
    use std::sync::Arc;

    pub fn joback() -> EosResult<([f64; 5], Arc<Joback>)> {
        let a = 1.5;
        let b = 3.4e-2;
        let c = 180.0e-4;
        let d = 2.2e-6;
        let e = 0.03e-8;
        let eos = Arc::new(Joback::from_model_records(vec![JobackRecord::new(
            a, b, c, d, e,
        )])?);
        let params = [a, b, c, d, e];
        Ok((params, eos))
    }

    #[test]
    fn test_joback() -> EosResult<()> {
        let (params, joback) = joback()?;
        let eos = Arc::new(EquationOfState::ideal_gas(joback));

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = arr1(&[1.3]) * KILO * MOL;

        let state = State::new_nvt(&eos, temperature, volume, &moles)?;
        let a_feos = state.helmholtz_energy(IdealGas);
        let mu_feos = state.chemical_potential(IdealGas);
        let p_feos = state.pressure(IdealGas);
        let s_feos = state.molar_entropy(IdealGas);
        let h_feos = state.molar_enthalpy(IdealGas);

        let t = temperature.to_reduced();
        let rho = (moles.get(0) / volume).to_reduced();
        let a_ad = JobackAD::helmholtz_energy_density(params, t, rho);
        let mu_ad = JobackAD::chemical_potential(params, t, rho);
        let p_ad = JobackAD::pressure(params, t, rho);
        let s_ad = JobackAD::molar_entropy(params, t, rho);
        let h_ad = JobackAD::molar_enthalpy(params, t, rho);

        println!(
            "\nHelmholtz energy density:\n{}",
            (a_feos / volume).to_reduced(),
        );
        println!("{a_ad}");
        assert_relative_eq!((a_feos / volume).to_reduced(), a_ad, max_relative = 1e-14);

        println!("\nChemical potential:\n{}", mu_feos.get(0).to_reduced());
        println!("{mu_ad}");
        assert_relative_eq!(mu_feos.get(0).to_reduced(), mu_ad, max_relative = 1e-14);

        println!("\nPressure:\n{}", p_feos.to_reduced());
        println!("{p_ad}");
        assert_relative_eq!(p_feos.to_reduced(), p_ad, max_relative = 1e-14);

        println!("\nMolar entropy:\n{}", s_feos.to_reduced());
        println!("{s_ad}");
        assert_relative_eq!(s_feos.to_reduced(), s_ad, max_relative = 1e-14);

        println!("\nMolar enthalpy:\n{}", h_feos.to_reduced());
        println!("{h_ad}");
        assert_relative_eq!(h_feos.to_reduced(), h_ad, max_relative = 1e-14);

        Ok(())
    }
}
