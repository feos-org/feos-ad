use crate::{HelmholtzEnergy, Joback, PcSaft};
use feos_core::{
    DensityInitialization, EosResult, EquationOfState, PhaseEquilibrium, ReferenceSystem, State,
};
use nalgebra::{Const, SVector};
use ndarray::arr1;
use num_dual::{jacobian, DualNum, DualVec};
use quantity::{Density, MolarEnergy, MolarEntropy, Pressure, Temperature, MOL};
use std::sync::Arc;

#[derive(Clone, Copy)]
pub struct StateAD<D> {
    pub pcsaft: [D; 8],
    pub joback: [D; 5],
    pub temperature: Temperature<D>,
    pub density: Density<D>,
    reduced_temperature: D,
    reduced_density: D,
}

impl<D: DualNum<f64> + Copy> StateAD<D> {
    pub fn new(pcsaft: [D; 8], joback: [D; 5], temperature: D, density: D) -> Self {
        Self {
            pcsaft,
            joback,
            temperature: Temperature::from_reduced(temperature),
            density: Density::from_reduced(density),
            reduced_temperature: temperature,
            reduced_density: density,
        }
    }

    fn from_state<
        E,
        F: Fn(
            [DualVec<D, f64, Const<2>>; 8],
            [DualVec<D, f64, Const<2>>; 5],
            DualVec<D, f64, Const<2>>,
            DualVec<D, f64, Const<2>>,
        ) -> SVector<DualVec<D, f64, Const<2>>, 2>,
    >(
        pcsaft: [D; 8],
        joback: [D; 5],
        state: State<E>,
        f: F,
        rhs: SVector<D, 2>,
    ) -> Self {
        let x = SVector::from([
            D::from(state.temperature.to_reduced()),
            D::from(state.density.to_reduced()),
        ]);
        let pc = pcsaft.map(DualVec::from_re);
        let jo = joback.map(DualVec::from_re);
        let (mut f, jac) = jacobian(|x| f(pc, jo, x[0], x[1]), x);
        f -= rhs;
        let det = (jac[(0, 0)] * jac[(1, 1)] - jac[(0, 1)] * jac[(1, 0)]).recip();
        let temperature = x[0] - det * (jac[(1, 1)] * f[0] - jac[(0, 1)] * f[1]);
        let density = x[1] - det * (jac[(0, 0)] * f[1] - jac[(1, 0)] * f[0]);
        Self::new(pcsaft, joback, temperature, density)
    }

    pub fn critical_point(pcsaft: [D; 8], joback: [D; 5]) -> EosResult<Self> {
        let eos = Arc::new(PcSaft::new(pcsaft));
        let state = State::critical_point(&eos, None, None, Default::default())?;
        Ok(Self::from_state(
            pcsaft,
            joback,
            state,
            Self::criticality_conditions,
            SVector::from([D::from(0.0); 2]),
        ))
    }

    pub fn new_tp(
        pcsaft: [D; 8],
        joback: [D; 5],
        temperature: Temperature<D>,
        pressure: Pressure<D>,
        density_initialization: DensityInitialization,
    ) -> EosResult<Self> {
        let t = temperature.re();
        let p = pressure.re();
        let eos = Arc::new(PcSaft::new(pcsaft));
        let moles = arr1(&[1.0]) * MOL;
        let state = State::new_npt(&eos, t, p, &moles, density_initialization)?;

        let rho = D::from(state.density.to_reduced());
        let t = temperature.into_reduced();
        let (_, p, dp_drho) = PcSaft::dp_drho(pcsaft, t, rho);

        let density = rho - (p - pressure.into_reduced()) / dp_drho;
        Ok(Self::new(pcsaft, joback, t, density))
    }

    pub fn new_ps(
        pcsaft: [D; 8],
        joback: [D; 5],
        pressure: Pressure<D>,
        molar_entropy: MolarEntropy<D>,
        density_initialization: DensityInitialization,
        initial_temperature: Option<Temperature>,
    ) -> EosResult<Self> {
        let moles = arr1(&[1.0]) * MOL;
        let residual = Arc::new(PcSaft::new(pcsaft));
        let ideal_gas = Arc::new(Joback::new(joback));
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let state = State::new_nps(
            &eos,
            pressure.re(),
            molar_entropy.re(),
            &moles,
            density_initialization,
            initial_temperature,
        )?;
        Ok(Self::from_state(
            pcsaft,
            joback,
            state,
            Self::pressure_entropy,
            SVector::from([pressure.into_reduced(), molar_entropy.into_reduced()]),
        ))
    }

    pub fn new_ph(
        pcsaft: [D; 8],
        joback: [D; 5],
        pressure: Pressure<D>,
        molar_enthalpy: MolarEnergy<D>,
        density_initialization: DensityInitialization,
        initial_temperature: Option<Temperature>,
    ) -> EosResult<Self> {
        let moles = arr1(&[1.0]) * MOL;
        let residual = Arc::new(PcSaft::new(pcsaft));
        let ideal_gas = Arc::new(Joback::new(joback));
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let state = State::new_nph(
            &eos,
            pressure.re(),
            molar_enthalpy.re(),
            &moles,
            density_initialization,
            initial_temperature,
        )?;
        Ok(Self::from_state(
            pcsaft,
            joback,
            state,
            Self::pressure_enthalpy,
            SVector::from([pressure.into_reduced(), molar_enthalpy.into_reduced()]),
        ))
    }

    fn criticality_conditions<D2: DualNum<f64> + Copy>(
        pcsaft: [D2; 8],
        joback: [D2; 5],
        temperature: D2,
        density: D2,
    ) -> SVector<D2, 2> {
        PcSaft::criticality_conditions(pcsaft, temperature, density)
            + Joback::criticality_conditions(joback, temperature, density)
    }

    fn pressure_entropy<D2: DualNum<f64> + Copy>(
        pcsaft: [D2; 8],
        joback: [D2; 5],
        temperature: D2,
        density: D2,
    ) -> SVector<D2, 2> {
        PcSaft::pressure_entropy(pcsaft, temperature, density)
            + Joback::pressure_entropy(joback, temperature, density)
    }

    fn pressure_enthalpy<D2: DualNum<f64> + Copy>(
        pcsaft: [D2; 8],
        joback: [D2; 5],
        temperature: D2,
        density: D2,
    ) -> SVector<D2, 2> {
        PcSaft::pressure_enthalpy(pcsaft, temperature, density)
            + Joback::pressure_enthalpy(joback, temperature, density)
    }

    pub fn pressure(&self) -> Pressure<D> {
        Pressure::from_reduced(
            PcSaft::pressure(self.pcsaft, self.reduced_temperature, self.reduced_density)
                + self.reduced_temperature * self.reduced_density,
        )
    }

    pub fn molar_entropy(&self) -> MolarEntropy<D> {
        MolarEntropy::from_reduced(
            PcSaft::molar_entropy(self.pcsaft, self.reduced_temperature, self.reduced_density)
                + Joback::molar_entropy(
                    self.joback,
                    self.reduced_temperature,
                    self.reduced_density,
                ),
        )
    }

    pub fn molar_enthalpy(&self) -> MolarEnergy<D> {
        MolarEnergy::from_reduced(
            PcSaft::molar_enthalpy(self.pcsaft, self.reduced_temperature, self.reduced_density)
                + Joback::molar_enthalpy(
                    self.joback,
                    self.reduced_temperature,
                    self.reduced_density,
                ),
        )
    }
}

pub struct PhaseEquilibriumAD<D> {
    pub liquid: StateAD<D>,
    pub vapor: StateAD<D>,
}

impl<D: DualNum<f64> + Copy> PhaseEquilibriumAD<D> {
    pub fn new_t(
        pcsaft: [D; 8],
        joback: [D; 5],
        temperature: Temperature<D>,
    ) -> EosResult<(Self, Pressure<D>)> {
        let eos = Arc::new(PcSaft::new(pcsaft));
        let vle = PhaseEquilibrium::pure(&eos, temperature.re(), None, Default::default())?;
        let rho1 = D::from(vle.liquid().density.to_reduced());
        let rho2 = D::from(vle.vapor().density.to_reduced());
        let t = temperature.into_reduced();
        let (f1, p1, dp_drho1) = PcSaft::dp_drho(pcsaft, t, rho1);
        let (f2, p2, dp_drho2) = PcSaft::dp_drho(pcsaft, t, rho2);
        let p = -(rho2 * f1 - rho1 * f2 + rho1 * rho2 * t * (rho1 / rho2).ln()) / (rho2 - rho1);
        let density1 = rho1 - (p1 - p) / dp_drho1;
        let density2 = rho2 - (p2 - p) / dp_drho2;
        Ok((
            Self {
                liquid: StateAD::new(pcsaft, joback, t, density1),
                vapor: StateAD::new(pcsaft, joback, t, density2),
            },
            Pressure::from_reduced(p),
        ))
    }

    pub fn vapor_pressure(parameters: [D; 8], temperature: D) -> EosResult<D> {
        let t = Temperature::from_reduced(temperature.re());
        let eos = Arc::new(PcSaft::new(parameters));
        let vle = PhaseEquilibrium::pure(&eos, t, None, Default::default())?;
        let rho1 = D::from(vle.liquid().density.to_reduced());
        let rho2 = D::from(vle.vapor().density.to_reduced());

        let f1 = PcSaft::helmholtz_energy_density(parameters, temperature, rho1);
        let f2 = PcSaft::helmholtz_energy_density(parameters, temperature, rho2);
        Ok(
            -(rho2 * f1 - rho1 * f2 + rho1 * rho2 * temperature * (rho1 / rho2).ln())
                / (rho2 - rho1),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::joback::test::joback;
    use crate::pcsaft::test::pcsaft;
    use approx::assert_relative_eq;
    use feos_core::{Contributions, EosResult, PhaseEquilibrium};
    use num_dual::{Dual, Dual64};
    use quantity::{JOULE, KELVIN};

    #[test]
    fn test_critical_point() -> EosResult<()> {
        let (mut pcsaft, eos) = pcsaft()?;
        let mut pcsaft_dual = pcsaft.map(Dual64::from);
        pcsaft_dual[0] = pcsaft_dual[0].derivative();
        let joback = [Dual::from(0.0); 5];
        let cp = State::critical_point(&eos, None, None, Default::default())?;
        let state = StateAD::critical_point(pcsaft_dual, joback)?;
        let t = state.temperature.re();
        let rho = state.density.re();
        println!("{:.5} {:.5}", t, rho);
        println!("{:.5} {:.5}", cp.temperature, cp.density);
        assert_relative_eq!(t, cp.temperature, max_relative = 1e-10);
        assert_relative_eq!(rho, cp.density, max_relative = 1e-10);

        let h = 1e-8;
        pcsaft[0] += h;
        let eos = Arc::new(PcSaft::new(pcsaft));
        let cp_h = State::critical_point(&eos, None, None, Default::default())?;
        let dt = (cp_h.temperature - cp.temperature).to_reduced() / h;
        let drho = (cp_h.density - cp.density).to_reduced() / h;

        println!(
            "{:.5e} {:.5e}",
            state.reduced_temperature.eps, state.reduced_density.eps
        );
        println!("{:.5e} {:.5e}", dt, drho);
        assert_relative_eq!(state.reduced_temperature.eps, dt, max_relative = 1e-6);
        assert_relative_eq!(state.reduced_density.eps, drho, max_relative = 1e-6);
        Ok(())
    }

    #[test]
    fn test_state_ps() -> EosResult<()> {
        let (pcsaft, residual) = pcsaft()?;
        let (joback, ideal_gas) = joback()?;
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let p = vle.liquid().pressure(Contributions::Total);
        let s = vle.liquid().molar_entropy(Contributions::Total);
        let t = vle.liquid().temperature;
        let state = StateAD::new_ps(pcsaft, joback, p, s, DensityInitialization::Liquid, Some(t))?;
        println!("{:.5} {:.5}", state.temperature, state.density);
        println!(
            "{:.5} {:.5}",
            vle.liquid().temperature,
            vle.liquid().density,
        );
        assert_relative_eq!(
            state.temperature,
            vle.liquid().temperature,
            max_relative = 1e-10
        );
        assert_relative_eq!(state.density, vle.liquid().density, max_relative = 1e-10);
        Ok(())
    }

    #[test]
    fn test_state_ps_derivative() -> EosResult<()> {
        let (pcsaft, residual) = pcsaft()?;
        let (joback, ideal_gas) = joback()?;
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let h = 1e-3 * JOULE / KELVIN / MOL;
        let state_h = State::new_nps(
            &eos,
            vle.liquid().pressure(Contributions::Total),
            vle.liquid().molar_entropy(Contributions::Total) + h,
            &vle.liquid().moles,
            DensityInitialization::Liquid,
            Some(vle.liquid().temperature),
        )?;
        let p = vle.liquid().pressure(Contributions::Total).to_reduced();
        let s = vle
            .liquid()
            .molar_entropy(Contributions::Total)
            .to_reduced();
        let t = vle.liquid().temperature;
        let pcsaft: [Dual64; 8] = pcsaft.map(Dual::from);
        let joback: [Dual64; 5] = joback.map(Dual::from);
        let p = Pressure::from_reduced(Dual::from(p));
        let s = MolarEntropy::from_reduced(Dual::from(s).derivative());
        let state = StateAD::new_ps(pcsaft, joback, p, s, DensityInitialization::Liquid, Some(t))?;
        println!(
            "{:.5e} {:.5e}",
            state.reduced_temperature.eps, state.reduced_density.eps
        );
        println!(
            "{:.5e} {:.5e}",
            ((state_h.temperature - vle.liquid().temperature) / h).to_reduced(),
            ((state_h.density - vle.liquid().density) / h).to_reduced(),
        );
        assert_relative_eq!(
            state.reduced_temperature.eps,
            ((state_h.temperature - vle.liquid().temperature) / h).to_reduced(),
            max_relative = 1e-6
        );
        assert_relative_eq!(
            state.reduced_density.eps,
            ((state_h.density - vle.liquid().density) / h).to_reduced(),
            max_relative = 1e-6
        );
        Ok(())
    }

    #[test]
    fn test_state_ph() -> EosResult<()> {
        let (pcsaft, residual) = pcsaft()?;
        let (joback, ideal_gas) = joback()?;
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let p = vle.liquid().pressure(Contributions::Total);
        let h = vle.liquid().molar_enthalpy(Contributions::Total);
        let t = vle.liquid().temperature;
        let state = StateAD::new_ph(pcsaft, joback, p, h, DensityInitialization::Liquid, Some(t))?;
        println!("{:.5} {:.5}", state.temperature, state.density);
        println!(
            "{:.5} {:.5}",
            vle.liquid().temperature,
            vle.liquid().density,
        );
        assert_relative_eq!(
            state.temperature,
            vle.liquid().temperature,
            max_relative = 1e-10
        );
        assert_relative_eq!(state.density, vle.liquid().density, max_relative = 1e-10);
        Ok(())
    }

    #[test]
    fn test_state_ph_derivative() -> EosResult<()> {
        let (pcsaft, residual) = pcsaft()?;
        let (joback, ideal_gas) = joback()?;
        let eos = Arc::new(EquationOfState::new(ideal_gas, residual));
        let vle = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let delta = 1e-1 * JOULE / MOL;
        let state_h = State::new_nph(
            &eos,
            vle.liquid().pressure(Contributions::Total),
            vle.liquid().molar_enthalpy(Contributions::Total) + delta,
            &vle.liquid().moles,
            DensityInitialization::Liquid,
            Some(vle.liquid().temperature),
        )?;
        let p = vle.liquid().pressure(Contributions::Total).to_reduced();
        let h = vle
            .liquid()
            .molar_enthalpy(Contributions::Total)
            .to_reduced();
        let t = vle.liquid().temperature;
        let pcsaft: [Dual64; 8] = pcsaft.map(Dual::from);
        let joback: [Dual64; 5] = joback.map(Dual::from);
        let p = Pressure::from_reduced(Dual::from(p));
        let h = MolarEnergy::from_reduced(Dual::from(h).derivative());
        let state = StateAD::new_ph(pcsaft, joback, p, h, DensityInitialization::Liquid, Some(t))?;
        println!(
            "{:.5e} {:.5e}",
            state.reduced_temperature.eps, state.reduced_density.eps
        );
        println!(
            "{:.5e} {:.5e}",
            ((state_h.temperature - vle.liquid().temperature) / delta).to_reduced(),
            ((state_h.density - vle.liquid().density) / delta).to_reduced(),
        );
        assert_relative_eq!(
            state.reduced_temperature.eps,
            ((state_h.temperature - vle.liquid().temperature) / delta).to_reduced(),
            max_relative = 1e-6
        );
        assert_relative_eq!(
            state.reduced_density.eps,
            ((state_h.density - vle.liquid().density) / delta).to_reduced(),
            max_relative = 1e-6
        );
        Ok(())
    }

    #[test]
    fn test_phase_equilibrium() -> EosResult<()> {
        let (parameters, eos) = pcsaft()?;
        let temperature = 250.0 * KELVIN;
        let (vle, p) = PhaseEquilibriumAD::new_t(parameters, [0.0; 5], temperature)?;
        let rho_l = vle.liquid.density;
        let rho_v = vle.vapor.density;
        let vle_feos = PhaseEquilibrium::pure(&eos, temperature, None, Default::default())?;
        let p_feos = vle_feos.vapor().pressure(Contributions::Total);
        println!("{:.5} {:.5} {:.5}", rho_l, rho_v, p);
        println!(
            "{:.5} {:.5} {:.5}",
            vle_feos.liquid().density,
            vle_feos.vapor().density,
            p_feos
        );
        println!("{} {}", vle.liquid.pressure(), vle.vapor.pressure());
        assert_relative_eq!(rho_l, vle_feos.liquid().density, max_relative = 1e-10);
        assert_relative_eq!(rho_v, vle_feos.vapor().density, max_relative = 1e-10);
        assert_relative_eq!(p, p_feos, max_relative = 1e-10);
        Ok(())
    }

    #[test]
    fn test_phase_equilibrium_derivative() -> EosResult<()> {
        let (parameters, eos) = pcsaft()?;
        let parameters: [Dual64; 8] = parameters.map(Dual::from);
        let (vle, p) = PhaseEquilibriumAD::new_t(
            parameters,
            [Dual::from(0.0); 5],
            Temperature::from_reduced(Dual::from(250.0).derivative()),
        )?;
        let rho_l = vle.liquid.reduced_density;
        let rho_v = vle.vapor.reduced_density;
        let p = p.into_reduced();
        let vle_feos = PhaseEquilibrium::pure(&eos, 250.0 * KELVIN, None, Default::default())?;
        let h = 1e-5 * KELVIN;
        let vle_feos_h =
            PhaseEquilibrium::pure(&eos, 250.0 * KELVIN + h, None, Default::default())?;
        let drho_l = ((vle_feos_h.liquid().density - vle_feos.liquid().density) / h).to_reduced();
        let drho_v = ((vle_feos_h.vapor().density - vle_feos.vapor().density) / h).to_reduced();
        let dp = ((vle_feos_h.vapor().pressure(Contributions::Total)
            - vle_feos.vapor().pressure(Contributions::Total))
            / h)
            .to_reduced();
        println!("{:11.5e} {:11.5e} {:11.5e}", rho_l.eps, rho_v.eps, p.eps);
        println!("{:11.5e} {:11.5e} {:11.5e}", drho_l, drho_v, dp,);
        println!(
            "{} {}",
            vle.liquid.pressure().into_reduced(),
            vle.vapor.pressure().into_reduced()
        );
        assert_relative_eq!(rho_l.eps, drho_l, max_relative = 1e-6);
        assert_relative_eq!(rho_v.eps, drho_v, max_relative = 1e-6);
        assert_relative_eq!(p.eps, dp, max_relative = 1e-6);
        Ok(())
    }
}
