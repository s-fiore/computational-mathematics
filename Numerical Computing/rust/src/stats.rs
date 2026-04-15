// Statistical functions

/// Sum of all elements
pub fn sum(data: &[f64]) -> f64 {
    let mut s = 0.0;
    for x in data {
        s += x;
    }
    s
}

/// Arithmetic mean

pub fn mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    Some(sum(data) / data.len() as f64)
}

/// Variance
pub fn var(data: &[f64], ddof: usize) -> Option<f64> {
    let n = data.len();
    if n <= ddof {
        return None;
    }

    let mu = mean(data)?;
    let mut sq = 0.0;
    for x in data {
        sq += (x - mu) * (x - mu);
    }
    Some(sq / (n - ddof) as f64)
}

/// Standard deviation:
pub fn std(data: &[f64], ddof: usize) -> Option<f64> {
    var(data, ddof).map(|v| v.sqrt())
}

// Median and percentile
pub fn median(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    } else {
        Some(sorted[n / 2])
    }
}

/// q-th percentile using linear interpolation
pub fn percentile(data: &[f64], q: f64) -> Option<f64> {
    if data.is_empty() || !(0.0..=100.0).contains(&q) {
        return None;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let idx = q / 100.0 * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let t = idx - lo as f64; // interpolation weight
    Some(sorted[lo] * (1.0 - t) + sorted[hi] * t)
}

// Covariance and correlation
/// Covariance of two equal-length arrays:
pub fn cov(x: &[f64], y: &[f64], ddof: usize) -> Option<f64> {
    let n = x.len();
    if n != y.len() || n <= ddof {
        return None;
    }

    let mu_x = mean(x)?;
    let mu_y = mean(y)?;
    let mut s = 0.0;
    for i in 0..n {
        s += (x[i] - mu_x) * (y[i] - mu_y);
    }
    Some(s / (n - ddof) as f64)
}

/// Pearson correlation coefficient
pub fn corrcoef(x: &[f64], y: &[f64]) -> Option<f64> {
    let cov_xy = cov(x, y, 1)?;
    let std_x = std(x, 1)?;
    let std_y = std(y, 1)?;
    if std_x == 0.0 || std_y == 0.0 {
        return None;
    }
    Some(cov_xy / (std_x * std_y))
}
