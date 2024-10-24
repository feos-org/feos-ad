use super::HelmholtzEnergy;
use feos_core::{Components, Residual, StateHD};
use ndarray::{arr1, Array1, ScalarOperand};
use num_dual::DualNum;
use quantity::{MolarWeight, GRAM, MOL};
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

const MAX_ETA: f64 = 0.5;

pub const A0: [f64; 7] = [
    0.91056314451539,
    0.63612814494991,
    2.68613478913903,
    -26.5473624914884,
    97.7592087835073,
    -159.591540865600,
    91.2977740839123,
];
pub const A1: [f64; 7] = [
    -0.30840169182720,
    0.18605311591713,
    -2.50300472586548,
    21.4197936296668,
    -65.2558853303492,
    83.3186804808856,
    -33.7469229297323,
];
pub const A2: [f64; 7] = [
    -0.09061483509767,
    0.45278428063920,
    0.59627007280101,
    -1.72418291311787,
    -4.13021125311661,
    13.7766318697211,
    -8.67284703679646,
];
pub const B0: [f64; 7] = [
    0.72409469413165,
    2.23827918609380,
    -4.00258494846342,
    -21.00357681484648,
    26.8556413626615,
    206.5513384066188,
    -355.60235612207947,
];
pub const B1: [f64; 7] = [
    -0.57554980753450,
    0.69950955214436,
    3.89256733895307,
    -17.21547164777212,
    192.6722644652495,
    -161.8264616487648,
    -165.2076934555607,
];
pub const B2: [f64; 7] = [
    0.09768831158356,
    -0.25575749816100,
    -9.15585615297321,
    20.64207597439724,
    -38.80443005206285,
    93.6267740770146,
    -29.66690558514725,
];

// Dipole parameters
pub const AD: [[f64; 3]; 5] = [
    [0.30435038064, 0.95346405973, -1.16100802773],
    [-0.13585877707, -1.83963831920, 4.52586067320],
    [1.44933285154, 2.01311801180, 0.97512223853],
    [0.35569769252, -7.37249576667, -12.2810377713],
    [-2.06533084541, 8.23741345333, 5.93975747420],
];

pub const BD: [[f64; 3]; 5] = [
    [0.21879385627, -0.58731641193, 3.48695755800],
    [-1.18964307357, 1.24891317047, -14.9159739347],
    [1.16268885692, -0.50852797392, 15.3720218600],
    [0.0; 3],
    [0.0; 3],
];

pub const CD: [[f64; 3]; 4] = [
    [-0.06467735252, -0.95208758351, -0.62609792333],
    [0.19758818347, 2.99242575222, 1.29246858189],
    [-0.80875619458, -2.38026356489, 1.65427830900],
    [0.69028490492, -0.27012609786, -3.43967436378],
];

#[derive(Clone, Copy)]
pub struct PcSaft<const N: usize> {
    molarweight: f64,
    parameters: [f64; N],
}

impl<const N: usize> PcSaft<N> {
    pub fn new<D: DualNum<f64>>(parameters: [D; N]) -> Self {
        Self {
            molarweight: 0.0,
            parameters: parameters.map(|p| p.re()),
        }
    }
}

impl<const N: usize> PcSaft<N> {
    fn helmholtz_energy_non_assoc<D: DualNum<f64> + Copy>(
        m: D,
        sigma: D,
        epsilon_k: D,
        mu: D,
        temperature: D,
        density: D,
    ) -> (D, [D; 2]) {
        // temperature dependent segment diameter
        let diameter = sigma * (-(epsilon_k * (-3.) / temperature).exp() * 0.12 + 1.0);

        let eta = m * density * diameter.powi(3) * FRAC_PI_6;
        let eta2 = eta * eta;
        let eta3 = eta2 * eta;
        let eta_m1 = (-eta + 1.0).recip();
        let eta_m2 = eta_m1 * eta_m1;
        let etas = [
            D::one(),
            eta,
            eta2,
            eta3,
            eta2 * eta2,
            eta2 * eta3,
            eta3 * eta3,
        ];

        // hard sphere
        let hs = m * density * (eta * 4.0 - eta2 * 3.0) * eta_m2;

        // hard chain
        let g = (-eta * 0.5 + 1.0) * eta_m1 * eta_m2;
        let hc = -density * (m - 1.0) * g.ln();

        // dispersion
        let e = epsilon_k / temperature;
        let s3 = sigma.powi(3);
        let mut i1 = D::zero();
        let mut i2 = D::zero();
        let m1 = (m - 1.0) / m;
        let m2 = (m - 2.0) / m;
        for i in 0..7 {
            i1 += (m1 * (m2 * A2[i] + A1[i]) + A0[i]) * etas[i];
            i2 += (m1 * (m2 * B2[i] + B1[i]) + B0[i]) * etas[i];
        }
        let c1 = (m * (eta * 8.0 - eta2 * 2.0) * eta_m2 * eta_m2 + 1.0
            - (m - 1.0) * (eta * 20.0 - eta2 * 27.0 + eta2 * eta * 12.0 - eta2 * eta2 * 2.0)
                / ((eta - 1.0) * (eta - 2.0)).powi(2))
        .recip();
        let i = i1 * 2.0 + c1 * i2 * m * e;
        let disp = -density * density * m.powi(2) * e * s3 * i * PI;

        // dipoles
        let mu2 = mu.powi(2) / (m * temperature) / 1.380649e-4;
        let m_dipole = if m.re() > 2.0 { D::from(2.0) } else { m };
        let m1 = (m_dipole - 1.0) / m_dipole;
        let m2 = m1 * (m_dipole - 2.0) / m_dipole;
        let mut j1 = D::zero();
        let mut j2 = D::zero();
        for i in 0..5 {
            let a = m2 * AD[i][2] + m1 * AD[i][1] + AD[i][0];
            let b = m2 * BD[i][2] + m1 * BD[i][1] + BD[i][0];
            j1 += (a + b * e) * etas[i];
            if i < 4 {
                j2 += (m2 * CD[i][2] + m1 * CD[i][1] + CD[i][0]) * etas[i];
            }
        }

        // mu is factored out of these expressions to deal with the case where mu=0
        let phi2 = -density * density * j1 / s3 * PI;
        let phi3 = -density * density * density * j2 / s3 * PI_SQ_43;
        let dipole = phi2 * phi2 * mu2 * mu2 / (phi2 - phi3 * mu2);

        ((hs + hc + disp + dipole) * temperature, [eta, eta_m1])
    }
}

impl HelmholtzEnergy<4> for PcSaft<4> {
    fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: [D; 4],
        temperature: D,
        density: D,
    ) -> D {
        let [m, sigma, epsilon_k, mu] = parameters;
        let (f, _) =
            Self::helmholtz_energy_non_assoc(m, sigma, epsilon_k, mu, temperature, density);
        f
    }
}

impl HelmholtzEnergy<8> for PcSaft<8> {
    fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: [D; 8],
        temperature: D,
        density: D,
    ) -> D {
        let [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb] = parameters;
        let (non_assoc, [eta, eta_m1]) =
            Self::helmholtz_energy_non_assoc(m, sigma, epsilon_k, mu, temperature, density);

        // association
        let delta_assoc = ((epsilon_k_ab / temperature).exp() - 1.0) * sigma.powi(3) * kappa_ab;
        let k = eta * eta_m1;
        let delta = (k * (k * 0.5 + 1.5) + 1.0) * eta_m1 * delta_assoc;
        let rhoa = na * density;
        let rhob = nb * density;
        let aux = (rhoa - rhob) * delta + 1.0;
        let sqrt = (aux * aux + rhob * delta * 4.0).sqrt();
        let xa = (sqrt + 1.0 + (rhob - rhoa) * delta).recip() * 2.0;
        let xb = (sqrt + 1.0 - (rhob - rhoa) * delta).recip() * 2.0;
        let assoc = rhoa * (xa.ln() - xa * 0.5 + 0.5) + rhob * (xb.ln() - xb * 0.5 + 0.5);

        non_assoc + assoc * temperature
    }
}

impl<const N: usize> Components for PcSaft<N> {
    fn components(&self) -> usize {
        1
    }

    fn subset(&self, _: &[usize]) -> Self {
        *self
    }
}

impl<const N: usize> Residual for PcSaft<N>
where
    Self: HelmholtzEnergy<N>,
{
    fn compute_max_density(&self, _: &Array1<f64>) -> f64 {
        let m = self.parameters[0];
        let sigma = self.parameters[1];
        MAX_ETA / (FRAC_PI_6 * m * sigma.powi(3))
    }

    fn molar_weight(&self) -> MolarWeight<Array1<f64>> {
        arr1(&[self.molarweight]) * GRAM / MOL
    }

    fn residual_helmholtz_energy_contributions<D2: DualNum<f64> + Copy + ScalarOperand>(
        &self,
        state: &StateHD<D2>,
    ) -> Vec<(String, D2)> {
        let temperature = state.temperature;
        let density = state.partial_density[0];
        let parameters = self.parameters.map(D2::from);
        let a = Self::helmholtz_energy_density(parameters, temperature, density) * state.volume
            / temperature;
        vec![("PC-SAFT".into(), a)]
    }
}

#[cfg(test)]
pub mod test {
    use super::{HelmholtzEnergy as _, PcSaft as PcSaftAD};
    use approx::assert_relative_eq;
    use feos::pcsaft::{PcSaft, PcSaftParameters, PcSaftRecord};
    use feos_core::parameter::Parameter;
    use feos_core::{Contributions::Residual, EosResult, ReferenceSystem, State};
    use ndarray::arr1;
    use quantity::{KELVIN, KILO, METER, MOL};
    use std::sync::Arc;

    pub fn pcsaft() -> EosResult<([f64; 8], Arc<PcSaft>)> {
        let m = 1.5;
        let sigma = 3.4;
        let epsilon_k = 180.0;
        let mu = 2.2;
        let kappa_ab = 0.03;
        let epsilon_k_ab = 2500.;
        let na = 2.0;
        let nb = 1.0;
        let params = PcSaftParameters::from_model_records(vec![PcSaftRecord::new(
            m,
            sigma,
            epsilon_k,
            Some(mu),
            None,
            Some(kappa_ab),
            Some(epsilon_k_ab),
            Some(na),
            Some(nb),
            None,
            None,
            None,
            None,
        )])?;
        let eos = Arc::new(PcSaft::new(Arc::new(params)));
        let params = [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb];
        Ok((params, eos))
    }

    #[test]
    fn test_pcsaft() -> EosResult<()> {
        let (params, eos) = pcsaft()?;

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = arr1(&[1.3]) * KILO * MOL;

        let state = State::new_nvt(&eos, temperature, volume, &moles)?;
        let a_feos = state.residual_helmholtz_energy();
        let mu_feos = state.residual_chemical_potential();
        let p_feos = state.pressure(Residual);
        let s_feos = state.residual_molar_entropy();
        let h_feos = state.residual_molar_enthalpy();

        let t = temperature.to_reduced();
        let rho = (moles.get(0) / volume).to_reduced();
        let a_ad = PcSaftAD::helmholtz_energy_density(params, t, rho);
        let mu_ad = PcSaftAD::chemical_potential(params, t, rho);
        let p_ad = PcSaftAD::pressure(params, t, rho);
        let s_ad = PcSaftAD::molar_entropy(params, t, rho);
        let h_ad = PcSaftAD::molar_enthalpy(params, t, rho);

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
