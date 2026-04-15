use rust_numeric::LinearRegression;
use rust_numeric::linalg;
use rust_numeric::stats;

fn main() {
    // Linear Regression:
    // True relationship:  y = 2x + 1
    let x: Vec<Vec<f64>> = (1..=5).map(|i| vec![i as f64]).collect();
    let y = vec![3.1, 5.0, 6.9, 9.1, 11.0];

    let mut model = LinearRegression::new();
    model.fit(&x, &y).expect("fit failed");

    println!("=== LinearRegression (1 feature) ===");
    println!("intercept_ : {:.4}", model.intercept_);
    println!("coef_      : {:.4}", model.coef_[0]);
    println!("predict([[6.0]]): {:.4}", model.predict(&[vec![6.0]])[0]);
    println!("R2         : {:.4}", model.score(&x, &y).unwrap());

    // Linear Regression: multiple features
    // True relationship:  y = 2x1 + 3x2 + 1
    // Features must NOT be collinear — using independent (x1, x2) pairs.
    let x2 = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0],
        vec![3.0, 4.0],
        vec![4.0, 3.0],
        vec![5.0, 5.0],
    ];
    let y2 = vec![9.1, 7.9, 18.8, 18.2, 26.0];

    let mut model2 = LinearRegression::new();
    model2.fit(&x2, &y2).expect("fit failed");

    println!("\n=== LinearRegression (2 features) ===");
    println!("intercept_ : {:.4}", model2.intercept_);
    println!(
        "coef_      : {:?}",
        model2
            .coef_
            .iter()
            .map(|c| format!("{:.4}", c))
            .collect::<Vec<_>>()
    );
    println!(
        "predict([[6, 6]]): {:.4}",
        model2.predict(&[vec![6.0, 6.0]])[0]
    );
    println!("R2         : {:.4}", model2.score(&x2, &y2).unwrap());

    // linalg
    let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
    let a_inv = linalg::inv(&a).unwrap();
    let check = linalg::matmul(&a, &a_inv).unwrap();

    println!("\n=== linalg ===");
    println!("det(A)           : {:.4}", linalg::det(&a).unwrap());
    println!("trace(A)         : {:.4}", linalg::trace(&a).unwrap());
    println!(
        "inv(A)[0]        : [{:.4}, {:.4}]",
        a_inv[0][0], a_inv[0][1]
    );
    println!(
        "A · inv(A) ≈ I   : [[{:.1},{:.1}],[{:.1},{:.1}]]",
        check[0][0], check[0][1], check[1][0], check[1][1]
    );
    println!("eye(2)           : {:?}", linalg::eye(2));

    let v = vec![3.0, 4.0];
    println!("norm([3,4], 2)   : {:.4}", linalg::norm(&v, 2.0).unwrap());
    println!("dot([3,4],[3,4]) : {:.4}", linalg::dot(&v, &v).unwrap());

    let b = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    println!("transpose(B)     : {:?}", linalg::transpose(&b));

    // stats
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    let data2 = vec![1.0, 3.0, 4.0, 4.0, 6.0, 5.0, 8.0, 10.0];

    println!("\n=== stats ===");
    println!("sum             : {:.4}", stats::sum(&data));
    println!("mean            : {:.4}", stats::mean(&data).unwrap());
    println!("var  (ddof=0)   : {:.4}", stats::var(&data, 0).unwrap());
    println!("var  (ddof=1)   : {:.4}", stats::var(&data, 1).unwrap());
    println!("std  (ddof=0)   : {:.4}", stats::std(&data, 0).unwrap());
    println!("std  (ddof=1)   : {:.4}", stats::std(&data, 1).unwrap());
    println!("median          : {:.4}", stats::median(&data).unwrap());
    println!(
        "percentile(75)  : {:.4}",
        stats::percentile(&data, 75.0).unwrap()
    );
    println!(
        "cov(x,y, ddof=1): {:.4}",
        stats::cov(&data, &data2, 1).unwrap()
    );
    println!(
        "corrcoef        : {:.4}",
        stats::corrcoef(&data, &data2).unwrap()
    );

    // Edge cases
    println!("\n=== edge cases ===");
    let bad = model.predict(&[vec![1.0, 2.0]]); // wrong feature count: will be wrong
    println!("wrong feature count (silent): {:.4}", bad[0]); // coef_ only has 1 entry; extra cols ignored
    let mut empty_model = LinearRegression::new();
    let empty_fit = empty_model.fit(&[], &[]);
    println!(
        "fit with empty X: {}",
        if empty_fit.is_none() { "None" } else { "Some" }
    );
}
