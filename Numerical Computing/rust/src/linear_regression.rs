use crate::linalg::{inv, matmul, transpose};
use crate::stats::mean;

/// Ordinary Least Squares linear regression

pub struct LinearRegression {
    /// Coefficient vector beta1,..., betap, one entry per feature.
    pub coef_: Vec<f64>,
    /// Intercept (bias) term beta0.
    pub intercept_: f64,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    pub fn new() -> Self {
        LinearRegression {
            coef_: Vec::new(),
            intercept_: 0.0,
        }
    }

    /// Add a column of 1s to X to form the design matrix
    fn design_matrix(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let (n, p) = (x.len(), x[0].len());
        let mut dm = vec![vec![0.0; p + 1]; n];
        for i in 0..n {
            dm[i][0] = 1.0; // intercept column
            for j in 0..p {
                dm[i][j + 1] = x[i][j];
            }
        }
        dm
    }

    /// Fit by Ordinary Least Squares:  β = (X'X)^{-1} X'y
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Option<&mut Self> {
        if x.len() != y.len() || x.is_empty() {
            return None;
        }

        let dm = Self::design_matrix(x); // [1 | X],  shape (n, p+1)
        let xt = transpose(&dm); // X',       shape (p+1, n)
        let xtx = matmul(&xt, &dm)?; // X'X,      shape (p+1, p+1)
        let inv = inv(&xtx)?; // (X'X)^{-1},  shape (p+1, p+1)

        // X'y,  shape (p+1,)
        let (p1, n) = (xt.len(), y.len());
        let mut xty = vec![0.0; p1];
        for i in 0..p1 {
            for j in 0..n {
                xty[i] += xt[i][j] * y[j];
            }
        }

        // β = (X'X)^{-1} X'y,  shape (p+1,)
        let mut beta = vec![0.0; p1];
        for i in 0..p1 {
            for j in 0..p1 {
                beta[i] += inv[i][j] * xty[j];
            }
        }

        self.intercept_ = beta[0];
        self.coef_ = beta[1..].to_vec();
        Some(self)
    }

    /// y = X * beta = intercept_ + coef_ * X

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        let n = x.len();
        let mut y_hat = vec![0.0; n];
        for i in 0..n {
            y_hat[i] = self.intercept_;
            for j in 0..self.coef_.len() {
                y_hat[i] += self.coef_[j] * x[i][j];
            }
        }
        y_hat
    }

    /// coefficient of determination:  R2 = 1 − SS_res / SS_tot
    ///
    ///   SS_res = residual sum of squares
    ///   SS_tot = total sum of squares
    ///
    /// Returns `None` if dimensions mismatch or all targets are constant
    ///
    pub fn score(&self, x: &[Vec<f64>], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || y.is_empty() {
            return None;
        }

        let y_bar = mean(y)?;

        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..y.len() {
            let y_hat_i = {
                let mut yh = self.intercept_;
                for j in 0..self.coef_.len() {
                    yh += self.coef_[j] * x[i][j];
                }
                yh
            };
            ss_res += (y[i] - y_hat_i) * (y[i] - y_hat_i);
            ss_tot += (y[i] - y_bar) * (y[i] - y_bar);
        }

        if ss_tot == 0.0 {
            return None;
        }
        Some(1.0 - ss_res / ss_tot)
    }
}
