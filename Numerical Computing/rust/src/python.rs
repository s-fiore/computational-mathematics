#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::LinearRegression;
use crate::linalg;
use crate::stats;

fn invalid_input(message: &str) -> PyErr {
    PyValueError::new_err(message.to_string())
}

fn expect_option<T>(value: Option<T>, message: &str) -> PyResult<T> {
    value.ok_or_else(|| invalid_input(message))
}

fn ensure_non_empty_matrix(matrix: &[Vec<f64>], name: &str) -> PyResult<()> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(invalid_input(&format!(
            "{name} must be a non-empty 2D array"
        )));
    }
    if matrix.iter().any(|row| row.len() != matrix[0].len()) {
        return Err(invalid_input(&format!(
            "{name} must be a rectangular 2D array"
        )));
    }
    Ok(())
}

#[pyfunction]
fn sum(data: Vec<f64>) -> f64 {
    stats::sum(&data)
}

#[pyfunction]
fn mean(data: Vec<f64>) -> PyResult<f64> {
    expect_option(stats::mean(&data), "mean requires at least one value")
}

#[pyfunction(signature = (data, ddof=None))]
fn variance(data: Vec<f64>, ddof: Option<usize>) -> PyResult<f64> {
    let ddof = ddof.unwrap_or(0);
    expect_option(
        stats::var(&data, ddof),
        "variance requires len(data) > ddof",
    )
}

#[pyfunction(name = "std", signature = (data, ddof=None))]
fn standard_deviation(data: Vec<f64>, ddof: Option<usize>) -> PyResult<f64> {
    let ddof = ddof.unwrap_or(0);
    expect_option(
        stats::std(&data, ddof),
        "standard deviation requires len(data) > ddof",
    )
}

#[pyfunction]
fn median(data: Vec<f64>) -> PyResult<f64> {
    expect_option(stats::median(&data), "median requires at least one value")
}

#[pyfunction]
fn percentile(data: Vec<f64>, q: f64) -> PyResult<f64> {
    expect_option(
        stats::percentile(&data, q),
        "percentile requires non-empty data and q in [0, 100]",
    )
}

#[pyfunction(signature = (x, y, ddof=None))]
fn covariance(x: Vec<f64>, y: Vec<f64>, ddof: Option<usize>) -> PyResult<f64> {
    let ddof = ddof.unwrap_or(0);
    expect_option(
        stats::cov(&x, &y, ddof),
        "covariance requires matching lengths and len(x) > ddof",
    )
}

#[pyfunction]
fn correlation(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    expect_option(
        stats::corrcoef(&x, &y),
        "correlation requires matching non-constant inputs",
    )
}

#[pyfunction]
fn dot(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    expect_option(linalg::dot(&x, &y), "dot requires vectors of equal length")
}

#[pyfunction(signature = (x, ord=None))]
fn norm(x: Vec<f64>, ord: Option<f64>) -> PyResult<f64> {
    let ord = ord.unwrap_or(2.0);
    expect_option(
        linalg::norm(&x, ord),
        "norm requires a non-empty vector and a positive order",
    )
}

#[pyfunction]
fn normalize(x: Vec<f64>) -> Vec<f64> {
    linalg::normalize(&x)
}

#[pyfunction]
fn cross(x: Vec<f64>, y: Vec<f64>) -> PyResult<Vec<f64>> {
    if x.len() != 3 || y.len() != 3 {
        return Err(invalid_input("cross requires two 3D vectors"));
    }

    let x_arr = [x[0], x[1], x[2]];
    let y_arr = [y[0], y[1], y[2]];
    Ok(linalg::cross(&x_arr, &y_arr).to_vec())
}

#[pyfunction]
fn add(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    ensure_non_empty_matrix(&a, "a")?;
    ensure_non_empty_matrix(&b, "b")?;
    expect_option(
        linalg::add(&a, &b),
        "add requires matrices with identical shapes",
    )
}

#[pyfunction]
fn scale(a: Vec<Vec<f64>>, alpha: f64) -> PyResult<Vec<Vec<f64>>> {
    ensure_non_empty_matrix(&a, "a")?;
    Ok(linalg::scale(&a, alpha))
}

#[pyfunction]
fn matmul(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    ensure_non_empty_matrix(&a, "a")?;
    ensure_non_empty_matrix(&b, "b")?;
    expect_option(
        linalg::matmul(&a, &b),
        "matmul requires inner dimensions to match",
    )
}


#[pyfunction]
fn transpose(a: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    ensure_non_empty_matrix(&a, "a")?;
    Ok(linalg::transpose(&a))
}

#[pyfunction]
fn trace(a: Vec<Vec<f64>>) -> PyResult<f64> {
    ensure_non_empty_matrix(&a, "a")?;
    expect_option(linalg::trace(&a), "trace requires a square matrix")
}

#[pyfunction]
fn eye(n: usize) -> Vec<Vec<f64>> {
    linalg::eye(n)
}

#[pyfunction]
fn determinant(a: Vec<Vec<f64>>) -> PyResult<f64> {
    ensure_non_empty_matrix(&a, "a")?;
    expect_option(
        linalg::det(&a),
        "determinant requires a non-empty square matrix",
    )
}

#[pyfunction]
fn inverse(a: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    ensure_non_empty_matrix(&a, "a")?;
    expect_option(
        linalg::inv(&a),
        "inverse requires a non-singular square matrix",
    )
}

#[pyclass(name = "LinearRegression")]
struct PyLinearRegression {
    model: LinearRegression,
}

#[pymethods]
impl PyLinearRegression {
    #[new]
    fn new() -> Self {
        Self {
            model: LinearRegression::new(),
        }
    }

    fn fit<'py>(
        mut slf: PyRefMut<'py, Self>,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        ensure_non_empty_matrix(&x, "x")?;
        expect_option(
            slf.model.fit(&x, &y),
            "fit requires a non-empty feature matrix and matching target length",
        )?;
        Ok(slf)
    }

    fn predict(&self, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
        ensure_non_empty_matrix(&x, "x")?;
        if !self.model.coef_.is_empty() && x[0].len() != self.model.coef_.len() {
            return Err(invalid_input(
                "predict requires the same number of features used during fit",
            ));
        }
        Ok(self.model.predict(&x))
    }

    fn score(&self, x: Vec<Vec<f64>>, y: Vec<f64>) -> PyResult<f64> {
        ensure_non_empty_matrix(&x, "x")?;
        expect_option(
            self.model.score(&x, &y),
            "score requires matching dimensions and non-constant targets",
        )
    }

    #[getter]
    fn coef_(&self) -> Vec<f64> {
        self.model.coef_.clone()
    }

    #[getter]
    fn intercept_(&self) -> f64 {
        self.model.intercept_
    }
}

pub fn register_python_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sum, module)?)?;
    module.add_function(wrap_pyfunction!(mean, module)?)?;
    module.add_function(wrap_pyfunction!(variance, module)?)?;
    module.add_function(wrap_pyfunction!(standard_deviation, module)?)?;
    module.add_function(wrap_pyfunction!(median, module)?)?;
    module.add_function(wrap_pyfunction!(percentile, module)?)?;
    module.add_function(wrap_pyfunction!(covariance, module)?)?;
    module.add_function(wrap_pyfunction!(correlation, module)?)?;
    module.add_function(wrap_pyfunction!(dot, module)?)?;
    module.add_function(wrap_pyfunction!(norm, module)?)?;
    module.add_function(wrap_pyfunction!(normalize, module)?)?;
    module.add_function(wrap_pyfunction!(cross, module)?)?;
    module.add_function(wrap_pyfunction!(add, module)?)?;
    module.add_function(wrap_pyfunction!(scale, module)?)?;
    module.add_function(wrap_pyfunction!(matmul, module)?)?;
    module.add_function(wrap_pyfunction!(transpose, module)?)?;
    module.add_function(wrap_pyfunction!(trace, module)?)?;
    module.add_function(wrap_pyfunction!(eye, module)?)?;
    module.add_function(wrap_pyfunction!(determinant, module)?)?;
    module.add_function(wrap_pyfunction!(inverse, module)?)?;
    module.add_class::<PyLinearRegression>()?;
    Ok(())
}
