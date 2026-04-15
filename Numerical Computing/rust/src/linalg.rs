// Vector operations

/// Dot product
pub fn dot(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() {
        return None;
    }
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i] * y[i];
    }
    Some(sum)
}

/// p-norm of a vector.
///   ord = 1.0 (Manhattan)
///   ord = 2.0 (Euclidean)
///   ord = f64::INFINITY (Chebyshev / max-norm)
pub fn norm(x: &[f64], ord: f64) -> Option<f64> {
    if x.is_empty() || (ord <= 0.0 && !ord.is_infinite()) {
        return None;
    }

    if ord == 1.0 {
        let mut s = 0.0;
        for xi in x {
            s += xi.abs();
        }
        Some(s)
    } else if ord == 2.0 {
        let mut s = 0.0;
        for xi in x {
            s += xi * xi;
        }
        Some(s.sqrt())
    } else if ord.is_infinite() {
        let mut max = x[0].abs();
        for xi in x {
            if xi.abs() > max {
                max = xi.abs();
            }
        }
        Some(max)
    } else {
        let mut s = 0.0;
        for xi in x {
            s += xi.abs().powf(ord);
        }
        Some(s.powf(1.0 / ord))
    }
}

/// Scale a vector to unit l2 length
pub fn normalize(x: &[f64]) -> Vec<f64> {
    let n = norm(x, 2.0).unwrap_or(1.0);
    let mut result = vec![0.0; x.len()];
    for i in 0..x.len() {
        result[i] = x[i] / n;
    }
    result
}

/// 3-D cross product
pub fn cross(x: &[f64; 3], y: &[f64; 3]) -> [f64; 3] {
    [
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0],
    ]
}

// Matrix operations
/// Element-wise addition:  C = A + B
pub fn add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let (rows, cols) = (a.len(), a[0].len());
    if rows != b.len() || cols != b[0].len() {
        return None;
    }
    let mut c = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    Some(c)
}

/// Scalar multiplication:  B = alpha*(A)
pub fn scale(a: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    let (rows, cols) = (a.len(), a[0].len());
    let mut b = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            b[i][j] = alpha * a[i][j];
        }
    }
    b
}

/// Matrix multiplication:  C = A*B (m,n),  where A is (m,k) and B is (k,n)

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let (m, k_a) = (a.len(), a[0].len());
    let (k_b, n) = (b.len(), b[0].len());
    if k_a != k_b {
        return None;
    }
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for k in 0..k_a {
                c[i][j] += a[i][k] * b[k][j]; // Cᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
            }
        }
    }
    Some(c)
}

/// Transpose:  A', where (A')ij = Aji

pub fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (rows, cols) = (a.len(), a[0].len());
    let mut at = vec![vec![0.0; rows]; cols];
    for i in 0..rows {
        for j in 0..cols {
            at[j][i] = a[i][j];
        }
    }
    at
}

/// Trace: sum of diagonal entries; A must be square

pub fn trace(a: &[Vec<f64>]) -> Option<f64> {
    if a.len() != a[0].len() {
        return None;
    }
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += a[i][i];
    }
    Some(sum)
}

/// identity matrix (n,n)

pub fn eye(n: usize) -> Vec<Vec<f64>> {
    let mut i = vec![vec![0.0; n]; n];
    for k in 0..n {
        i[k][k] = 1.0;
    }
    i
}

/// Determinant via cofactor expansion along the first row

pub fn det(a: &[Vec<f64>]) -> Option<f64> {
    let n = a.len();
    if a.iter().any(|row| row.len() != n) {
        return None;
    }

    if n == 1 {
        return Some(a[0][0]);
    }
    if n == 2 {
        return Some(a[0][0] * a[1][1] - a[0][1] * a[1][0]);
    }

    let mut d = 0.0;
    for col in 0..n {
        let minor: Vec<Vec<f64>> = a[1..]
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .filter(|(j, _)| *j != col)
                    .map(|(_, v)| *v)
                    .collect()
            })
            .collect();
        let sign = if col % 2 == 0 { 1.0 } else { -1.0 };
        d += sign * a[0][col] * det(&minor)?;
    }
    Some(d)
}

/// Matrix inverse via Gauss-Jordan elimination
/// Returns `None` if A is singular or not square

pub fn inv(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if a.iter().any(|row| row.len() != n) {
        return None;
    }

    // Build augmented matrix [A | I]
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            for j in 0..n {
                r.push(if i == j { 1.0 } else { 0.0 });
            }
            r
        })
        .collect();

    // Forward and backward elimination (reduced row echelon form)
    for col in 0..n {
        // Partial pivoting: swap in the row with the largest pivot
        let pivot = (col..n).find(|&row| aug[row][col].abs() > 1e-10)?;
        aug.swap(col, pivot);

        // Normalise pivot row so the diagonal entry becomes 1
        let diag = aug[col][col];
        for val in aug[col].iter_mut() {
            *val /= diag;
        }

        // Eliminate this column from every other row
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for k in 0..2 * n {
                    aug[row][k] -= factor * aug[col][k];
                }
            }
        }
    }

    // Extract the right-hand half
    Some(aug.iter().map(|row| row[n..].to_vec()).collect())
}
