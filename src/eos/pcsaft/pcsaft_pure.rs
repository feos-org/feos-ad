use super::{A0, A1, A2, AD, B0, B1, B2, BD, CD, MAX_ETA};
use crate::eos::ChemicalRecord;
use crate::{NamedParameters, ParametersAD, ResidualHelmholtzEnergy};
use nalgebra::SVector;
use num_dual::DualNum;
use std::f64::consts::{FRAC_PI_6, PI};

const PI_SQ_43: f64 = 4.0 / 3.0 * PI * PI;

const GROUPS: [&str; 22] = [
    "CH3", "CH2", ">CH", ">C<", "=CH2", "=CH", "=C<", "C≡CH", "CH2_hex", "CH_hex", "CH2_pent",
    "CH_pent", "CH_arom", "C_arom", "CH=O", ">C=O", "OCH3", "OCH2", "HCOO", "COO", "OH", "NH2",
];
const M: [f64; 22] = [
    0.61198, 0.45606, 0.14304, -0.66997, 0.36939, 0.56361, 0.86367, 1.3279, 0.39496, 0.0288,
    0.46742, 0.03314, 0.42335, 0.15371, 1.5774, 1.223, 1.6539, 1.1349, 1.7525, 1.5063, 0.402,
    0.40558,
];
const SIGMA: [f64; 22] = [
    3.7202, 3.89, 4.8597, -1.7878, 4.0264, 3.5519, 3.1815, 2.9421, 3.9126, 8.9779, 3.7272, 7.719,
    3.727, 3.9622, 2.8035, 2.8124, 3.0697, 3.2037, 2.9043, 2.8166, 3.2859, 3.6456,
];

const EPSILON_K: [f64; 22] = [
    229.9, 239.01, 347.64, 107.68, 289.49, 216.69, 156.31, 223.05, 289.03, 1306.7, 267.16, 1297.7,
    274.41, 527.2, 242.99, 249.04, 196.05, 187.13, 229.63, 222.52, 488.66, 467.59,
];
const MU: [f64; 22] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.4556, 3.2432, 1.3866,
    2.744, 2.7916, 3.1652, 0.0, 0.0,
];
const KAPPA_AB: [f64; 22] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.006825, 0.026662,
];
const EPSILON_K_AB: [f64; 22] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 2517.0, 1064.6,
];
const NA: [f64; 22] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0,
];
const NB: [f64; 22] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 1.0,
];

/// Optimized implementation of PC-SAFT for a single component.
#[derive(Clone, Copy)]
pub struct PcSaftPure<const N: usize>(pub [f64; N]);

fn helmholtz_energy_density_non_assoc<D: DualNum<f64> + Copy>(
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

fn helmholtz_energy_density<D: DualNum<f64> + Copy>(
    parameters: &[D; 8],
    temperature: D,
    density: D,
) -> D {
    let [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb] = *parameters;
    let (non_assoc, [eta, eta_m1]) =
        helmholtz_energy_density_non_assoc(m, sigma, epsilon_k, mu, temperature, density);

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

impl PcSaftPure<8> {
    pub fn from_groups<D: DualNum<f64> + Copy>(group_counts: [D; 22]) -> [D; 8] {
        let m: D = M.into_iter().zip(group_counts).map(|(m, g)| g * m).sum();
        let m_sigma3 = M.into_iter().zip(SIGMA).map(|(m, s)| m * s.powi(3));
        let m_sigma3: D = m_sigma3.zip(group_counts).map(|(ms3, g)| g * ms3).sum();
        let sigma = (m_sigma3 / m).cbrt();
        let m_epsilon_k = M.into_iter().zip(EPSILON_K).map(|(m, e)| m * e);
        let m_epsilon_k: D = m_epsilon_k.zip(group_counts).map(|(me, g)| g * me).sum();
        let epsilon_k = m_epsilon_k / m;
        let mu = MU.into_iter().zip(group_counts).map(|(m, g)| g * m).sum();
        let kappa_ab = KAPPA_AB
            .into_iter()
            .zip(group_counts)
            .map(|(m, g)| g * m)
            .sum();
        let epsilon_k_ab = EPSILON_K_AB
            .into_iter()
            .zip(group_counts)
            .map(|(m, g)| g * m)
            .sum();
        let na = NA.into_iter().zip(group_counts).map(|(m, g)| g * m).sum();
        let nb = NB.into_iter().zip(group_counts).map(|(m, g)| g * m).sum();

        [m, sigma, epsilon_k, mu, kappa_ab, epsilon_k_ab, na, nb]
    }

    pub fn from_chemical_record<D: DualNum<f64> + Copy>(
        ChemicalRecord { groups, bonds: _ }: &ChemicalRecord<D>,
    ) -> [D; 8] {
        let group_counts = GROUPS.map(|g| groups[g]);
        Self::from_groups(group_counts)
    }
}

impl<const N: usize> ParametersAD for PcSaftPure<N> {
    type Parameters<D: DualNum<f64> + Copy> = [D; N];

    fn params<D: DualNum<f64> + Copy>(&self) -> Self::Parameters<D> {
        self.0.map(D::from)
    }

    fn params_from_inner<D: DualNum<f64> + Copy, D2: DualNum<f64, Inner = D> + Copy>(
        parameters: &Self::Parameters<D>,
    ) -> Self::Parameters<D2> {
        parameters.map(D2::from_inner)
    }
}

impl ResidualHelmholtzEnergy<1> for PcSaftPure<8> {
    const RESIDUAL: &str = "PC-SAFT (pure)";

    fn compute_max_density(&self, _: &SVector<f64, 1>) -> f64 {
        let m = self.0[0];
        let sigma = self.0[1];
        MAX_ETA / (FRAC_PI_6 * m * sigma.powi(3))
    }

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, 1>,
    ) -> D {
        let density = partial_density.data.0[0][0];
        helmholtz_energy_density(parameters, temperature, density)
    }
}

impl ResidualHelmholtzEnergy<1> for PcSaftPure<4> {
    const RESIDUAL: &str = "PC-SAFT (pure)";

    fn compute_max_density(&self, _: &SVector<f64, 1>) -> f64 {
        let m = self.0[0];
        let sigma = self.0[1];
        MAX_ETA / (FRAC_PI_6 * m * sigma.powi(3))
    }

    fn residual_helmholtz_energy_density<D: DualNum<f64> + Copy>(
        parameters: &Self::Parameters<D>,
        temperature: D,
        partial_density: &SVector<D, 1>,
    ) -> D {
        let density = partial_density.data.0[0][0];
        let [m, sigma, epsilon_k, mu] = *parameters;
        helmholtz_energy_density_non_assoc(m, sigma, epsilon_k, mu, temperature, density).0
    }
}

impl<const N: usize> NamedParameters for PcSaftPure<N> {
    fn index_parameters_mut<'a, D: DualNum<f64> + Copy>(
        parameters: &'a mut [D; N],
        index: &str,
    ) -> &'a mut D {
        match index {
            "m" => &mut parameters[0],
            "sigma" => &mut parameters[1],
            "epsilon_k" => &mut parameters[2],
            "mu" => &mut parameters[3],
            "kappa_ab" => &mut parameters[4],
            "epsilon_k_ab" => &mut parameters[5],
            "na" => &mut parameters[6],
            "nb" => &mut parameters[7],
            _ => panic!("{index} is not a valid PC-SAFT parameter!"),
        }
    }
}

#[cfg(test)]
pub mod test {
    use super::{PcSaftPure, ResidualHelmholtzEnergy};
    use crate::eos::pcsaft::test::pcsaft;
    use approx::assert_relative_eq;
    use feos_core::{Contributions::Total, EosResult, ReferenceSystem, State};
    use nalgebra::SVector;
    use ndarray::arr1;
    use quantity::{KELVIN, KILO, METER, MOL};

    #[test]
    fn test_pcsaft_pure() -> EosResult<()> {
        let (pcsaft, eos) = pcsaft()?;
        let pcsaft = pcsaft.0;

        let temperature = 300.0 * KELVIN;
        let volume = 2.3 * METER * METER * METER;
        let moles = arr1(&[1.3]) * KILO * MOL;

        let state = State::new_nvt(&eos, temperature, volume, &moles)?;
        let a_feos = state.residual_molar_helmholtz_energy();
        let mu_feos = state.residual_chemical_potential();
        let p_feos = state.pressure(Total);
        let s_feos = state.residual_molar_entropy();
        let h_feos = state.residual_molar_enthalpy();

        let total_moles = moles.sum();
        let t = temperature.to_reduced();
        let v = (volume / total_moles).to_reduced();
        let x = SVector::from_fn(|i, _| moles.get(i).convert_into(total_moles));
        let a_ad = PcSaftPure::residual_molar_helmholtz_energy(&pcsaft, t, v, &x);
        let mu_ad = PcSaftPure::residual_chemical_potential(&pcsaft, t, v, &x);
        let p_ad = PcSaftPure::pressure(&pcsaft, t, v, &x);
        let s_ad = PcSaftPure::residual_molar_entropy(&pcsaft, t, v, &x);
        let h_ad = PcSaftPure::residual_molar_enthalpy(&pcsaft, t, v, &x);

        println!("\nMolar Helmholtz energy:\n{}", a_feos.to_reduced(),);
        println!("{a_ad}");
        assert_relative_eq!(a_feos.to_reduced(), a_ad, max_relative = 1e-14);

        println!("\nChemical potential:\n{}", mu_feos.get(0).to_reduced());
        println!("{}", mu_ad[0]);
        assert_relative_eq!(mu_feos.get(0).to_reduced(), mu_ad[0], max_relative = 1e-14);

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
