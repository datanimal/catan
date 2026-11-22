//! The win-probability model would drive a spectator overlay, so the fit, the
//! ranking measure and the calibration measure each need checking against
//! data whose answer is known.

use catan_sim::winprob::{auc, calibration, fit_logit};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[test]
fn logit_recovers_known_coefficients() {
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let names: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
    // log-odds = -1 + 1.5a - 0.8b, with a and b correlated.
    let (ia, ta, tb) = (-1.0, 1.5, -0.8);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..40000 {
        let a: f64 = rng.random_range(-2.0..2.0);
        let b: f64 = 0.5 * a + rng.random_range(-1.5..1.5);
        let p = 1.0 / (1.0 + (-(ia + ta * a + tb * b)).exp());
        x.push(vec![a, b]);
        y.push(rng.random_bool(p) as u8 as f64);
    }
    let f = fit_logit(&names, &x, &y).expect("fit");
    println!(
        "intercept {:.3} (want {ia}), a {:.3} (want {ta}), b {:.3} (want {tb}), AUC {:.3}, {} iters",
        f.intercept, f.coef[0], f.coef[1], f.auc, f.iterations
    );
    assert!((f.intercept - ia).abs() < 0.08, "intercept {}", f.intercept);
    assert!((f.coef[0] - ta).abs() < 0.08, "coef a {}", f.coef[0]);
    assert!((f.coef[1] - tb).abs() < 0.08, "coef b {}", f.coef[1]);
    assert!(f.iterations < 40, "did not converge");
    // The fit must beat predicting the base rate.
    assert!(f.brier < f.brier_baseline * 0.95);
}

#[test]
fn constant_features_are_dropped_not_fatal() {
    // Right after setup several features are identically zero. A singular
    // Hessian must not take the whole fit down with it.
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let names: Vec<String> = ["real", "always_zero"].iter().map(|s| s.to_string()).collect();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for _ in 0..5000 {
        let a: f64 = rng.random_range(-2.0..2.0);
        let p = 1.0 / (1.0 + (-a).exp());
        x.push(vec![a, 0.0]);
        y.push(rng.random_bool(p) as u8 as f64);
    }
    let f = fit_logit(&names, &x, &y).expect("fit should survive a constant column");
    assert_eq!(f.dropped, vec!["always_zero".to_string()]);
    assert_eq!(f.coef[1], 0.0, "a dropped feature must carry no weight");
    assert!((f.coef[0] - 1.0).abs() < 0.12, "coef {}", f.coef[0]);
}

#[test]
fn auc_matches_known_cases() {
    // Perfect separation.
    let preds = vec![0.1, 0.2, 0.8, 0.9];
    let y = vec![0.0, 0.0, 1.0, 1.0];
    assert!((auc(&preds, &y) - 1.0).abs() < 1e-9);
    // Exactly reversed.
    let y_rev = vec![1.0, 1.0, 0.0, 0.0];
    assert!((auc(&preds, &y_rev) - 0.0).abs() < 1e-9);
    // All tied is pure chance.
    let flat = vec![0.5; 4];
    assert!((auc(&flat, &y) - 0.5).abs() < 1e-9);
    // Random scores land near a half.
    let mut rng = ChaCha8Rng::seed_from_u64(8);
    let n = 20000;
    let p: Vec<f64> = (0..n).map(|_| rng.random_range(0.0..1.0)).collect();
    let t: Vec<f64> = (0..n).map(|_| rng.random_bool(0.3) as u8 as f64).collect();
    let a = auc(&p, &t);
    println!("random AUC {a:.4}");
    assert!((a - 0.5).abs() < 0.02);
}

#[test]
fn calibration_detects_honest_and_dishonest_models() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    // An honest model: outcomes really do happen at the stated rate.
    let mut preds = Vec::new();
    let mut y = Vec::new();
    for _ in 0..60000 {
        let p: f64 = rng.random_range(0.0..1.0);
        preds.push(p);
        y.push(rng.random_bool(p) as u8 as f64);
    }
    let bins = calibration(&preds, &y, 10);
    let worst = bins
        .iter()
        .map(|b| (b.mean_predicted - b.actual).abs())
        .fold(0.0f64, f64::max);
    println!("honest model: worst bin gap {:.4}", worst);
    assert!(worst < 0.03, "an honest model looked miscalibrated: {worst}");

    // An overconfident model: it stretches its probabilities to the extremes.
    let skewed: Vec<f64> = preds.iter().map(|p| (p * 1.6 - 0.3).clamp(0.0, 1.0)).collect();
    let bins2 = calibration(&skewed, &y, 10);
    let worst2 = bins2
        .iter()
        .filter(|b| b.n >= 200)
        .map(|b| (b.mean_predicted - b.actual).abs())
        .fold(0.0f64, f64::max);
    println!("overconfident model: worst bin gap {:.4}", worst2);
    assert!(
        worst2 > 0.1,
        "calibration failed to catch an overconfident model: {worst2}"
    );
}
