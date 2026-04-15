# Rust vs. Python Benchmark Notebook

This folder contains a reproducible notebook that benchmarks the Rust implementations in this crate against equivalent NumPy and scikit-learn functionality.

## What the notebook covers

- `stats::mean` vs. `numpy.mean`
- `stats::percentile` vs. `numpy.percentile`
- `linalg::matmul` vs. `numpy.matmul`
- `linalg::inv` vs. `numpy.linalg.inv`
- `LinearRegression::{fit, predict, score}` vs. `sklearn.linear_model.LinearRegression`


## Recommended setup

From the `Numerical Computing/rust` directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r benchmarks/requirements.txt
maturin develop --release
python -m ipykernel install --user --name rust-numeric-bench --display-name "Rust Numeric Bench"
jupyter lab benchmarks/Rust v. Python.ipynb
```

## Notes on interpretation

- The Rust code is imported directly into Python through `pyo3` bindings built with `maturin`.
- The benchmark times reflect notebook-friendly, end-to-end calls from Python into Rust.
- NumPy and scikit-learn benefit from highly optimized native backends and array-native APIs, so they are a strong baseline rather than a strawman.
