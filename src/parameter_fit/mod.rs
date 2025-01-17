use crate::{HelmholtzEnergyWrapper, ResidualHelmholtzEnergy};
use feos_core::{
    DensityInitialization::Liquid, EosResult, PhaseEquilibrium, ReferenceSystem, State,
};
use nalgebra::{Const, SVector};
use ndarray::arr1;
use num_dual::DualVec;
use quantity::{Density, Moles, Pressure, Temperature};

type Gradient<const P: usize> = DualVec<f64, f64, Const<P>>;

pub fn vapor_pressure<R: ResidualHelmholtzEnergy<1>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 1>,
    temperature: Temperature,
) -> EosResult<Pressure<Gradient<P>>> {
    let vle = PhaseEquilibrium::pure(&eos.eos, temperature, None, Default::default())?;

    let v1 = 1.0 / vle.liquid().density.to_reduced();
    let v2 = 1.0 / vle.vapor().density.to_reduced();
    let t = temperature.into_reduced();
    let (a1, a2) = {
        let t = DualVec::from(t);
        let v1 = DualVec::from(v1);
        let v2 = DualVec::from(v2);
        let x = SVector::from([DualVec::from(1.0)]);

        let a1 = R::residual_molar_helmholtz_energy(&eos.parameters, t, v1, &x);
        let a2 = R::residual_molar_helmholtz_energy(&eos.parameters, t, v2, &x);
        (a1, a2)
    };

    let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
    Ok(Pressure::from_reduced(p))
}

pub fn equilibrium_liquid_density<R: ResidualHelmholtzEnergy<1>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 1>,
    temperature: Temperature,
) -> EosResult<(Pressure<Gradient<P>>, Density<Gradient<P>>)> {
    let vle = PhaseEquilibrium::pure(&eos.eos, temperature, None, Default::default())?;

    let v_l = 1.0 / vle.liquid().density.to_reduced();
    let v_v = 1.0 / vle.vapor().density.to_reduced();
    let t = temperature.into_reduced();
    let (f_l, p_l, dp_l, a_v) = {
        let t = DualVec::from(temperature.into_reduced());
        let v_l = DualVec::from(v_l);
        let v_v = DualVec::from(v_v);
        let x = SVector::from([DualVec::from(1.0)]);

        let (f_l, p_l, dp_l) = R::dp_drho(&eos.parameters, t, v_l, &x);
        let a_v = R::residual_molar_helmholtz_energy(&eos.parameters, t, v_v, &x);
        (f_l, p_l, dp_l, a_v)
    };

    let p = -(f_l * v_l - a_v + t * (v_v / v_l).ln()) / (v_l - v_v);
    let rho = (p - p_l) / dp_l + 1.0 / v_l;
    Ok((Pressure::from_reduced(p), Density::from_reduced(rho)))
}

pub fn liquid_density<R: ResidualHelmholtzEnergy<1>, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, Gradient<P>, 1>,
    temperature: Temperature,
    pressure: Pressure,
) -> EosResult<Density<Gradient<P>>> {
    let moles = Moles::from_reduced(arr1(&[1.0]));
    let state = State::new_npt(&eos.eos, temperature, pressure, &moles, Liquid)?;

    let t = temperature.into_reduced();
    let v = 1.0 / state.density.to_reduced();
    let p0 = pressure.into_reduced();
    let (p, dp) = {
        let t = DualVec::from(t);
        let v = DualVec::from(v);
        let x = SVector::from([DualVec::from(1.0)]);
        let (_, p, dp) = R::dp_drho(&eos.parameters, t, v, &x);

        (p, dp)
    };

    let rho = -(p - p0) / dp + 1.0 / v;
    Ok(Density::from_reduced(rho))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::eos::pcsaft::test::{pcsaft, pcsaft_non_assoc};
    use crate::eos::PcSaftPure;
    use crate::{ParametersAD, PhaseEquilibriumAD, StateAD};
    use approx::assert_relative_eq;
    use nalgebra::U1;
    use quantity::{BAR, KELVIN, LITER, MOL, PASCAL};

    #[test]
    fn test_vapor_pressure_derivatives() -> EosResult<()> {
        let pcsaft_params = [
            "m",
            "sigma",
            "epsilon_k",
            "mu",
            "kappa_ab",
            "epsilon_k_ab",
            "na",
            "nb",
        ];
        let (pcsaft, _) = pcsaft()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(pcsaft_params);
        let temperature = 250.0 * KELVIN;
        let p = vapor_pressure(&pcsaft_ad, temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(Const::<8>, U1));

        println!("{:.5}", p);
        println!("{:.5?}", grad);

        for (i, par) in pcsaft_params.into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let (_, p_h) = PhaseEquilibriumAD::new_t(&pcsaft_h, temperature)?;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let dp = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_vapor_pressure_derivatives_fit() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let p = vapor_pressure(&pcsaft_ad, temperature)?;
        let p = p.convert_into(PASCAL);
        let (p, grad) = (p.re, p.eps.unwrap_generic(Const::<3>, U1));

        println!("{:.5}", p);
        println!("{:.5?}", grad);

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let (_, p_h) = PhaseEquilibriumAD::new_t(&pcsaft_h, temperature)?;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let dp = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_equilibrium_liquid_density_derivatives_fit() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let (p, rho) = equilibrium_liquid_density(&pcsaft_ad, temperature)?;
        let p = p.convert_into(PASCAL);
        let rho = rho.convert_into(MOL / LITER);
        let (p, p_grad) = (p.re, p.eps.unwrap_generic(Const::<3>, U1));
        let (rho, rho_grad) = (rho.re, rho.eps.unwrap_generic(Const::<3>, U1));

        println!("{:.5} {:.5}", p, rho);
        println!("{:.5?}", p_grad);
        println!("{:.5?}", rho_grad);

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let (vle, p_h) = PhaseEquilibriumAD::new_t(&pcsaft_h, temperature)?;
            let v_h = vle.liquid.molar_volume;
            let dp_h = (p_h.convert_into(PASCAL) - p) / h;
            let drho_h = (v_h.convert_into(LITER / MOL).recip() - rho) / h;
            let dp = p_grad[i];
            let drho = rho_grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e} {:11.5} {:11.5} {:.3e}",
                dp_h,
                dp,
                ((dp_h - dp) / dp).abs(),
                drho_h,
                drho,
                ((drho_h - drho) / drho).abs()
            );
            assert_relative_eq!(dp, dp_h, max_relative = 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_liquid_density_derivatives_fit() -> EosResult<()> {
        let (pcsaft, _) = pcsaft_non_assoc()?;
        let pcsaft = pcsaft.wrap();
        let pcsaft_ad = pcsaft.named_derivatives(["m", "sigma", "epsilon_k"]);
        let temperature = 150.0 * KELVIN;
        let pressure = BAR;
        let rho = liquid_density(&pcsaft_ad, temperature, pressure)?;
        let rho = rho.convert_into(MOL / LITER);
        let (rho, grad) = (rho.re, rho.eps.unwrap_generic(Const::<3>, U1));

        println!("{:.5}", rho);
        println!("{:.5?}", grad);

        for (i, par) in ["m", "sigma", "epsilon_k"].into_iter().enumerate() {
            let mut params = pcsaft.parameters;
            let h = params[i] * 1e-7;
            params[i] += h;
            let pcsaft_h = PcSaftPure(params).wrap();
            let v_h = StateAD::new_tp(
                &pcsaft_h,
                temperature,
                pressure,
                SVector::from([1.0]),
                Liquid,
            )?
            .molar_volume;
            let drho_h = (v_h.convert_into(LITER / MOL).recip() - rho) / h;
            let drho = grad[i];
            println!(
                "{par:12}: {:11.5} {:11.5} {:.3e}",
                drho_h,
                drho,
                ((drho_h - drho) / drho).abs()
            );
            assert_relative_eq!(drho, drho_h, max_relative = 1e-6);
        }
        Ok(())
    }
}
