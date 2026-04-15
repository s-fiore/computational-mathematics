mod python;

pub mod linalg;
pub mod linear_regression;
pub mod stats;

pub use linear_regression::LinearRegression;

use pyo3::prelude::*;

#[pymodule]
fn rust_numeric(module: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register_python_bindings(module)
}
