use super::ORC;
use pyo3::prelude::*;

// fn extract_args(args: [f64; 15]) -> ([f64; 8], [f64; 5], f64, f64) {
//     let mut pcsaft = [0.0; 8];
//     pcsaft.copy_from_slice(&args[..8]);
//     let mut joback = [0.0; 5];
//     joback.copy_from_slice(&args[8..13]);
//     let temperature = args[13];
//     let density = args[14];
//     (pcsaft, joback, temperature, density)
// }

// #[pyfunction]
// #[pyo3(signature=(args, _fixed=None, fgh=2))]
// fn chemical_potential(
//     py: Python,
//     args: [f64; 15],
//     _fixed: Option<[bool; 15]>,
//     fgh: usize,
// ) -> Bound<'_, PyAny> {
//     let (pcsaft, joback, temperature, density) = extract_args(args);

//     if fgh == 0 {
//         PyFloat::new_bound(
//             py,
//             Model::chemical_potential(pcsaft, joback, temperature, density),
//         )
//         .into_any()
//     } else if fgh == 1 {
//         let (f, g) = Model::chemical_potential_gradients(pcsaft, joback, temperature, density);
//         let tuple: Py<PyAny> = (f, g).into_py(py);
//         tuple.into_bound(py)
//     } else if fgh == 2 {
//         unimplemented!()
//     } else {
//         unreachable!()
//     }
// }

// #[pyfunction]
// #[pyo3(signature=(args, _fixed=None, fgh=2))]
// fn pressure(
//     py: Python,
//     args: [f64; 15],
//     _fixed: Option<[bool; 15]>,
//     fgh: usize,
// ) -> Bound<'_, PyAny> {
//     let (pcsaft, joback, temperature, density) = extract_args(args);

//     if fgh == 0 {
//         PyFloat::new_bound(py, Model::pressure(pcsaft, joback, temperature, density)).into_any()
//     } else if fgh == 1 {
//         let (f, g) = Model::pressure_gradients(pcsaft, joback, temperature, density);
//         let tuple: Py<PyAny> = (f, g).into_py(py);
//         tuple.into_bound(py)
//     } else if fgh == 2 {
//         unimplemented!()
//     } else {
//         unreachable!()
//     }
// }

// #[pyfunction]
// #[pyo3(signature=(args, _fixed=None, fgh=2))]
// fn entropy(
//     py: Python,
//     args: [f64; 15],
//     _fixed: Option<[bool; 15]>,
//     fgh: usize,
// ) -> Bound<'_, PyAny> {
//     let (pcsaft, joback, temperature, density) = extract_args(args);

//     if fgh == 0 {
//         PyFloat::new_bound(py, Model::entropy(pcsaft, joback, temperature, density)).into_any()
//     } else if fgh == 1 {
//         let (f, g) = Model::entropy_gradients(pcsaft, joback, temperature, density);
//         let tuple: Py<PyAny> = (f, g).into_py(py);
//         tuple.into_bound(py)
//     } else if fgh == 2 {
//         unimplemented!()
//     } else {
//         unreachable!()
//     }
// }

// #[pyfunction]
// #[pyo3(signature=(args, _fixed=None, fgh=2))]
// fn enthalpy(
//     py: Python,
//     args: [f64; 15],
//     _fixed: Option<[bool; 15]>,
//     fgh: usize,
// ) -> Bound<'_, PyAny> {
//     let (pcsaft, joback, temperature, density) = extract_args(args);

//     if fgh == 0 {
//         PyFloat::new_bound(py, Model::enthalpy(pcsaft, joback, temperature, density)).into_any()
//     } else if fgh == 1 {
//         let (f, g) = Model::enthalpy_gradients(pcsaft, joback, temperature, density);
//         let tuple: Py<PyAny> = (f, g).into_py(py);
//         tuple.into_bound(py)
//     } else if fgh == 2 {
//         unimplemented!()
//     } else {
//         unreachable!()
//     }
// }

// #[pyfunction]
// fn properties(pcsaft: [f64; 8], joback: [f64; 5], temperature: f64, density: f64) -> [f64; 4] {
//     Model::properties(pcsaft, joback, temperature, density)
//         .data
//         .0[0]
// }

// #[pyfunction]
// fn property_gradients(
//     pcsaft: [f64; 8],
//     joback: [f64; 5],
//     temperature: f64,
//     density: f64,
// ) -> ([f64; 4], [[f64; 4]; 15]) {
//     let (f, g) = Model::property_gradients(pcsaft, joback, temperature, density);
//     (f.data.0[0], g.data.0)
// }

#[pymodule]
pub fn feos_ad(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ORC>()
}
