//! The placement valuations are regression output, so the regression itself
//! needs checking against data whose answer is known in advance.

use catan_sim::analysis::ols;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[test]
fn ols_recovers_known_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let names: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    // y = 5 + 2a - 3b + 0c + noise, with correlated regressors so the fit has
    // to actually invert X'X rather than get away with treating them as
    // independent.
    let truth = [2.0, -3.0, 0.0];
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..8000 {
        let a: f64 = rng.random_range(0.0..10.0);
        let b: f64 = 0.6 * a + rng.random_range(0.0..4.0);
        let c: f64 = rng.random_range(0.0..5.0);
        let noise: f64 = rng.random_range(-2.0..2.0);
        x.push(vec![a, b, c]);
        y.push(5.0 + truth[0] * a + truth[1] * b + truth[2] * c + noise);
    }
    let f = ols(&names, &x, &y).expect("fit");
    println!("intercept {:.3} (want 5.0), r2 {:.3}", f.intercept, f.r2);
    for i in 0..3 {
        println!(
            "  {}: {:.4} +/- {:.4} (want {:.1}), t={:.1}",
            f.names[i],
            f.coef[i],
            f.se[i],
            truth[i],
            f.t(i)
        );
    }
    for i in 0..3 {
        assert!(
            (f.coef[i] - truth[i]).abs() < 0.05,
            "coefficient {} was {:.4}, wanted {:.1}",
            f.names[i],
            f.coef[i],
            truth[i]
        );
        // The known value must sit inside the reported interval.
        let (lo, hi) = f.ci(i);
        assert!(lo <= truth[i] && truth[i] <= hi, "CI missed the true value");
    }
    assert!((f.intercept - 5.0).abs() < 0.15);
    // The irrelevant regressor must not look significant.
    assert!(f.t(2).abs() < 3.0, "found a spurious effect: t={:.2}", f.t(2));
}

#[test]
fn ols_standard_errors_have_the_right_scale() {
    // Refit the same model on many independent samples; the spread of the
    // estimates must match the standard error the fit reports.
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let names = vec!["x".to_string()];
    let mut est = Vec::new();
    let mut reported = 0.0;
    for _ in 0..200 {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for _ in 0..500 {
            let v: f64 = rng.random_range(0.0..10.0);
            x.push(vec![v]);
            y.push(1.5 * v + rng.random_range(-5.0..5.0));
        }
        let f = ols(&names, &x, &y).unwrap();
        est.push(f.coef[0]);
        reported += f.se[0];
    }
    reported /= 200.0;
    let m = est.iter().sum::<f64>() / est.len() as f64;
    let sd = (est.iter().map(|e| (e - m).powi(2)).sum::<f64>() / (est.len() - 1) as f64).sqrt();
    println!("estimates: mean {m:.4} sd {sd:.4} | reported se {reported:.4}");
    assert!((m - 1.5).abs() < 0.02, "biased estimator: {m}");
    assert!(
        (sd / reported - 1.0).abs() < 0.15,
        "standard errors are off by {:.0}%",
        100.0 * (sd / reported - 1.0)
    );
}
