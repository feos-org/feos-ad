use crate::{PhaseEquilibriumAD, StateAD};
use feos_core::{DensityInitialization, EosResult, ReferenceSystem};
use nalgebra::{SVector, U1};
use num_dual::{try_jacobian, Derivative, DualNum, DualVec};
use num_traits::Zero;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use quantity::{
    HeatCapacityRate, MoleFlowRate, Power, Pressure, Temperature, BAR, CELSIUS, KELVIN, KILO, MEGA,
    WATT,
};

struct PressureChanger<'a, D> {
    inlet: &'a StateAD<D>,
    outlet: StateAD<D>,
}

impl<'a, D: DualNum<f64> + Copy> PressureChanger<'a, D> {
    fn pump(inlet: &'a StateAD<D>, pressure: Pressure<D>, efficiency: f64) -> EosResult<Self> {
        // calculate isentropic state
        let state2s = StateAD::new_ps(
            inlet.pcsaft,
            inlet.joback,
            pressure,
            inlet.molar_entropy(),
            DensityInitialization::Liquid,
            Some(inlet.temperature.re()),
        )?;

        // calculate real state
        let h1 = inlet.molar_enthalpy();
        let h2s = state2s.molar_enthalpy();
        let h2 = h1 + (h2s - h1) / efficiency;
        let outlet = StateAD::new_ph(
            inlet.pcsaft,
            inlet.joback,
            pressure,
            h2,
            DensityInitialization::Liquid,
            Some(state2s.temperature.re()),
        )?;

        Ok(Self { inlet, outlet })
    }

    fn turbine(
        inlet: &'a StateAD<D>,
        pressure: Pressure<D>,
        vle_condenser: &PhaseEquilibriumAD<D>,
        efficiency: f64,
    ) -> EosResult<Self> {
        // calculate isentropic state
        let s1 = inlet.molar_entropy();
        let s_v = vle_condenser.vapor.molar_entropy();
        let h2s = if s1.re() < s_v.re() {
            let s_l = vle_condenser.liquid.molar_entropy();
            let h_l = vle_condenser.liquid.molar_enthalpy();
            let h_v = vle_condenser.vapor.molar_enthalpy();
            let x = (s1 - s_l) / (s_v - s_l);
            h_l + x * (h_v - h_l)
        } else {
            StateAD::new_ps(
                inlet.pcsaft,
                inlet.joback,
                pressure,
                s1,
                DensityInitialization::Vapor,
                Some(inlet.temperature.re()),
            )?
            .molar_enthalpy()
        };

        // calculate real state
        let h1 = inlet.molar_enthalpy();
        let h2 = h1 + (h2s - h1) * efficiency;
        let outlet = StateAD::new_ph(
            inlet.pcsaft,
            inlet.joback,
            pressure,
            h2,
            DensityInitialization::Vapor,
            Some(inlet.temperature.re()),
        )?;

        Ok(Self { inlet, outlet })
    }

    fn power(&self, flow_rate: MoleFlowRate<D>) -> Power<D> {
        flow_rate * (self.outlet.molar_enthalpy() - self.inlet.molar_enthalpy())
    }
}

#[cfg_attr(feature = "python", pyclass)]
pub struct ORC {
    c_p_hs: HeatCapacityRate,
    t_hs: Temperature,
    dt_hs: Temperature,
    eta_st: f64,
    eta_sp: f64,
    p_min: Pressure,
    p_max: Pressure,
    p_min_r: f64,
    p_max_r: f64,
    t_cool_in: Temperature,
    t_cool_out: Temperature,
    dt_cool: Temperature,
}

impl ORC {
    fn solve_dual<D: DualNum<f64> + Copy>(
        &self,
        pcsaft: [D; 8],
        joback: [D; 5],
        x: [D; 3],
    ) -> EosResult<SVector<D, 9>> {
        // unpack process variables
        let [t_cond, t_evap, dt_sh] = x;
        let t_cond = Temperature::from_reduced(t_cond * 300.0);
        let t_evap = Temperature::from_reduced(t_evap * 300.0);
        let dt_sh = Temperature::from_reduced(dt_sh * 50.0);

        // calculate isobars
        let (vle_cond, p_cond) = PhaseEquilibriumAD::new_t(pcsaft, joback, t_cond)?;
        let (vle_evap, p_evap) = PhaseEquilibriumAD::new_t(pcsaft, joback, t_evap)?;

        // calculate pump
        let pump = PressureChanger::pump(&vle_cond.liquid, p_evap, self.eta_sp)?;

        // calculate superheating
        let turbine_in = StateAD::new_tp(
            pcsaft,
            joback,
            t_evap + dt_sh,
            p_evap,
            DensityInitialization::Vapor,
        )?;

        // calculate turbine
        let turbine = PressureChanger::turbine(&turbine_in, p_cond, &vle_cond, self.eta_st)?;

        // calculate mass flow rate
        let t_hs_pinch = vle_evap.liquid.temperature + self.dt_hs;
        let m_wf = (t_hs_pinch - self.t_hs) * self.c_p_hs
            / (vle_evap.liquid.molar_enthalpy() - turbine_in.molar_enthalpy());

        // target
        let target = (pump.power(m_wf) + turbine.power(m_wf)).convert_into(MEGA * WATT);

        // pinch constraint heat source
        let pinch_hs = ((-turbine_in.temperature + self.t_hs) / self.dt_hs).into_value() - 1.0;

        // pinch constraint condenser
        let h1 = vle_cond.liquid.molar_enthalpy();
        let h2 = vle_cond.vapor.molar_enthalpy();
        let h3 = turbine.outlet.molar_enthalpy();
        let t_cool_pinch =
            (h2 - h1) / (h3 - h1) * (self.t_cool_out - self.t_cool_in) + self.t_cool_in;
        let pinch_cond = ((t_cool_pinch - t_cond) / self.dt_cool).into_value() - 1.0;

        // critical pressure
        let p_crit = StateAD::critical_point(pcsaft, joback)?.pressure();

        // reduced pressure constraints
        let pr_cond = p_cond.convert_into(p_crit) - self.p_min_r;
        let pr_evap = -p_evap.convert_into(p_crit) + self.p_max_r;

        // absolute pressure constraints
        let pa_cond = (p_cond - self.p_min).convert_into(BAR);
        let pa_evap = (-p_evap + self.p_max).convert_into(BAR);

        // pressure constraint
        let p_const = p_evap.convert_into(p_cond) - 1.0;

        // turbine outlet constraint
        let turb_out = turbine.outlet.temperature.convert_into(t_cond) - 1.0;

        Ok(SVector::from([
            target, pinch_hs, pinch_cond, pr_cond, pr_evap, pa_cond, pa_evap, p_const, turb_out,
        ]))
    }
}

impl Default for ORC {
    fn default() -> Self {
        Self {
            c_p_hs: 65.0 * KILO * WATT / KELVIN,
            t_hs: 175.0 * CELSIUS,
            dt_hs: 20.0 * KELVIN,
            eta_st: 0.65,
            eta_sp: 0.8,
            p_min: BAR,
            p_max: 50.0 * BAR,
            p_min_r: 1e-3,
            p_max_r: 0.8,
            t_cool_in: 25. * CELSIUS,
            t_cool_out: 40.0 * CELSIUS,
            dt_cool: -10.0 * KELVIN,
        }
    }
}

#[cfg_attr(feature = "python", pymethods)]
impl ORC {
    #[cfg(feature = "python")]
    #[new]
    pub fn py_new() -> Self {
        Default::default()
    }

    pub fn solve(&self, pcsaft: [f64; 8], joback: [f64; 5], x: [f64; 3]) -> [f64; 9] {
        self.solve_dual(pcsaft, joback, x).unwrap().data.0[0]
    }

    pub fn jacobian_pattern(
        &self,
        pcsaft: [f64; 8],
        joback: [f64; 5],
        x: [f64; 3],
    ) -> EosResult<[[bool; 9]; 16]> {
        let mut vars = [0.0; 16];
        vars[..8].copy_from_slice(&pcsaft);
        vars[8..13].copy_from_slice(&joback);
        vars[13..].copy_from_slice(&x);
        let mut pattern = [[false; 9]; 16];
        for i in 0..vars.len() {
            let mut vars_dual: [DualVec<f64, f64, U1>; 16] = vars.map(DualVec::from_re);
            vars_dual[i].eps = Derivative::derivative_generic(U1, U1, 0);
            let mut pcsaft = [DualVec::zero(); 8];
            pcsaft.copy_from_slice(&vars_dual[..8]);
            let mut joback = [DualVec::zero(); 5];
            joback.copy_from_slice(&vars_dual[8..13]);
            let mut x = [DualVec::zero(); 3];
            x.copy_from_slice(&vars_dual[13..]);
            let out = self.solve_dual(pcsaft, joback, x)?.data.0[0];
            pattern[i] = out.map(|x| x.eps != Derivative::none());
        }
        Ok(pattern)
    }

    pub fn jacobian(
        &self,
        pcsaft: [f64; 8],
        joback: [f64; 5],
        x: [f64; 3],
    ) -> ([f64; 9], [[f64; 9]; 16]) {
        let mut vars = [0.0; 16];
        vars[..8].copy_from_slice(&pcsaft);
        vars[8..13].copy_from_slice(&joback);
        vars[13..].copy_from_slice(&x);
        let vars = SVector::from(vars);
        let (f, jac) = try_jacobian(
            |vars| {
                let vars = vars.data.0[0];
                let mut pcsaft = [DualVec::zero(); 8];
                pcsaft.copy_from_slice(&vars[..8]);
                let mut joback = [DualVec::zero(); 5];
                joback.copy_from_slice(&vars[8..13]);
                let mut x = [DualVec::zero(); 3];
                x.copy_from_slice(&vars[13..]);
                self.solve_dual(pcsaft, joback, x)
            },
            vars,
        )
        .unwrap();
        (f.data.0[0], jac.data.0)
    }

    #[cfg(feature = "python")]
    fn __getstate__(&self) -> [f64; 12] {
        [
            self.c_p_hs.convert_into(KILO * WATT / KELVIN),
            self.t_hs.convert_into(KELVIN),
            self.dt_hs.convert_into(KELVIN),
            self.eta_st,
            self.eta_sp,
            self.p_min.convert_into(BAR),
            self.p_max.convert_into(BAR),
            self.p_min_r,
            self.p_max_r,
            self.t_cool_in.convert_into(KELVIN),
            self.t_cool_out.convert_into(KELVIN),
            self.dt_cool.convert_into(KELVIN),
        ]
    }

    #[cfg(feature = "python")]
    fn __setstate__(&mut self, state: [f64; 12]) {
        self.c_p_hs = state[0] * KILO * WATT / KELVIN;
        self.t_hs = state[1] * KELVIN;
        self.dt_hs = state[2] * KELVIN;
        self.eta_st = state[3];
        self.eta_sp = state[4];
        self.p_min = state[5] * BAR;
        self.p_max = state[6] * BAR;
        self.p_min_r = state[7];
        self.p_max_r = state[8];
        self.t_cool_in = state[9] * KELVIN;
        self.t_cool_out = state[10] * KELVIN;
        self.dt_cool = state[11] * KELVIN;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::joback::test::joback;
    use crate::pcsaft::test::pcsaft;
    use feos_core::EosResult;

    #[test]
    fn test_orc() -> EosResult<()> {
        let (pcsaft, _) = pcsaft()?;
        let (joback, _) = joback()?;
        let x0 = [1.0, 1.2, 0.1];

        let orc = ORC::default();
        let out = orc.solve(pcsaft, joback, x0);
        println!("{out:?}");
        let pattern = orc.jacobian_pattern(pcsaft, joback, x0)?;
        println!("{pattern:?}");
        let (out, jac) = orc.jacobian(pcsaft, joback, x0);
        println!("{out:?}");
        println!("{jac:?}");

        for (p, j) in pattern.iter().zip(&jac) {
            for (&p, j) in p.iter().zip(j) {
                println!("{p:5} {j:9.3e} {:5} {:5}", j.is_zero(), p || j.is_zero());
                assert!(p || j.is_zero())
            }
        }

        Ok(())
    }
}
