use crate::{core::NamedParameters, HelmholtzEnergyWrapper, ResidualHelmholtzEnergy};
use feos_core::{EosResult, PhaseEquilibrium, ReferenceSystem};
use nalgebra::{Const, SVector};
use num_dual::DualVec;
use quantity::{Pressure, Temperature};

pub fn vapor_pressure<R: ResidualHelmholtzEnergy<1> + NamedParameters, const P: usize>(
    eos: &HelmholtzEnergyWrapper<R, f64, 1>,
    parameters: [&str; P],
    temperature: Temperature,
) -> EosResult<Pressure<DualVec<f64, f64, Const<P>>>> {
    let vle = PhaseEquilibrium::pure(&eos.eos, temperature.re(), None, Default::default())?;

    let v1 = 1.0 / vle.liquid().density.to_reduced();
    let v2 = 1.0 / vle.vapor().density.to_reduced();
    let t = temperature.into_reduced();
    let (a1, a2) = {
        let t = DualVec::from(temperature.into_reduced());
        let v1 = DualVec::from(v1);
        let v2 = DualVec::from(v2);
        let x = SVector::from([DualVec::from(1.0)]);
        let params = eos.named_derivatives(parameters);

        let a1 = R::residual_molar_helmholtz_energy(&params, t, v1, &x);
        let a2 = R::residual_molar_helmholtz_energy(&params, t, v2, &x);
        (a1, a2)
    };

    let p = -(a1 - a2 + t * (v2 / v1).ln()) / (v1 - v2);
    Ok(Pressure::from_reduced(p))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::eos::pcsaft::test::{pcsaft, pcsaft_non_assoc};
    use crate::eos::PcSaftPure;
    use crate::{ParametersAD, PhaseEquilibriumAD};
    use approx::assert_relative_eq;
    use nalgebra::U1;
    use quantity::{KELVIN, PASCAL};

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
        let temperature = 250.0 * KELVIN;
        let p = vapor_pressure(&pcsaft, pcsaft_params, temperature)?;
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
        let temperature = 150.0 * KELVIN;
        let p = vapor_pressure(&pcsaft, ["m", "sigma", "epsilon_k"], temperature)?;
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
}
