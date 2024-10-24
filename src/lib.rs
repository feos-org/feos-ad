use nalgebra::SVector;
use num_dual::{
    first_derivative, gradient, second_derivative, third_derivative, Dual, Dual2, Dual3, DualNum,
    DualVec,
};

mod pcsaft;
use pcsaft::PcSaft;
mod gc_pcsaft;
mod joback;
use joback::Joback;
mod orc;
pub use orc::ORC;
mod state;
pub use state::{PhaseEquilibriumAD, StateAD};

#[cfg(feature = "python")]
mod python;

pub trait HelmholtzEnergy<const N: usize> {
    fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: [D; N],
        temperature: D,
        density: D,
    ) -> D;

    fn dp_drho<D: DualNum<f64> + Copy>(
        parameters: [D; N],
        temperature: D,
        density: D,
    ) -> (D, D, D) {
        let parameters = parameters.map(Dual2::from_re);
        let t = Dual2::from_re(temperature);
        let (f, df, d2f) = second_derivative(
            |density| Self::helmholtz_energy_density(parameters, t, density),
            density,
        );
        (
            f,
            -f + density * df + density * temperature,
            density * d2f + temperature,
        )
    }

    fn criticality_conditions<D: DualNum<f64> + Copy>(
        parameters: [D; N],
        temperature: D,
        density: D,
    ) -> SVector<D, 2> {
        let parameters = parameters.map(Dual3::from_re);
        let temperature = Dual3::from_re(temperature);
        let (_, _, dmu, d2mu) = third_derivative(
            |density| Self::helmholtz_energy_density(parameters, temperature, density),
            density,
        );
        SVector::from([dmu, d2mu])
    }

    fn pressure_entropy<D: DualNum<f64> + Copy>(
        parameters: [D; N],
        temperature: D,
        density: D,
    ) -> SVector<D, 2> {
        let parameters = parameters.map(DualVec::from_re);
        let x = SVector::from([temperature, density]);
        let (f, df) = gradient(
            |x| Self::helmholtz_energy_density(parameters, x[0], x[1]),
            x,
        );
        let p = -f + density * df[1];
        let s = -df[0] / density;
        SVector::from([p, s])
    }

    fn pressure_enthalpy<D: DualNum<f64> + Copy>(
        parameters: [D; N],
        temperature: D,
        density: D,
    ) -> SVector<D, 2> {
        let parameters = parameters.map(DualVec::from_re);
        let x = SVector::from([temperature, density]);
        let (f, df) = gradient(
            |x| Self::helmholtz_energy_density(parameters, x[0], x[1]),
            x,
        );
        let p = -f + density * df[1];
        let h = -temperature * df[0] / density + df[1];
        SVector::from([p, h])
    }

    fn chemical_potential<D: DualNum<f64> + Copy>(
        parameters: [D; N],
        temperature: D,
        density: D,
    ) -> D {
        let parameters = parameters.map(Dual::from_re);
        let temperature = Dual::from_re(temperature);
        let (_, mu) = first_derivative(
            |density| Self::helmholtz_energy_density(parameters, temperature, density),
            density,
        );
        mu
    }

    fn pressure<D: DualNum<f64> + Copy>(parameters: [D; N], temperature: D, density: D) -> D {
        let parameters = parameters.map(Dual::from_re);
        let temperature = Dual::from_re(temperature);
        let (f, mu) = first_derivative(
            |density| Self::helmholtz_energy_density(parameters, temperature, density),
            density,
        );
        mu * density - f
    }

    fn molar_entropy<D: DualNum<f64> + Copy>(parameters: [D; N], temperature: D, density: D) -> D {
        let parameters = parameters.map(Dual::from_re);
        let density = Dual::from_re(density);
        let (_, df_dt) = first_derivative(
            |temperature| Self::helmholtz_energy_density(parameters, temperature, density),
            temperature,
        );
        -df_dt / density.re
    }

    fn molar_enthalpy<D: DualNum<f64> + Copy>(parameters: [D; N], temperature: D, density: D) -> D {
        let mu = Self::chemical_potential(parameters, temperature, density);
        let s = Self::molar_entropy(parameters, temperature, density);
        mu + s * temperature
    }
}
