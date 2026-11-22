//! Live win probability: the groundwork for a spectator "eval bar".
//!
//! A chess evaluation bar works because chess is perfect information and the
//! engine is far stronger than the players. Catan breaks both assumptions, so
//! this module is built to measure how much that costs rather than to paper
//! over it:
//!
//! - **Hidden information.** Opponents' resource cards and unplayed
//!   development cards are secret, and victory-point cards are secret
//!   *victory points*. So two models are fitted: one on what a spectator could
//!   actually read off a live game, and one on the true state. The gap between
//!   them is the price of the hidden information.
//! - **Calibration, not accuracy.** A meter that reads 70% must win 70% of the
//!   time or it is lying to viewers. Discrimination (AUC) and calibration
//!   (reliability table, Brier score) are reported separately, because a model
//!   can rank positions well while being badly miscalibrated.
//!
//! The probabilities are conditional on *these bots* finishing the game. For
//! strong human play the ranking should survive but the numbers would need
//! recalibrating against human games.

use crate::analysis::invert;
use crate::metrics::GameRecord;
use serde::{Deserialize, Serialize};

// ----------------------------------------------------------- logistic fit ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Logit {
    pub names: Vec<String>,
    /// Features left out of the fit because they never varied in this sample.
    /// Right after setup, for instance, nobody has played a knight or built a
    /// city, and a constant column makes the Hessian singular.
    pub dropped: Vec<String>,
    pub intercept: f64,
    pub coef: Vec<f64>,
    pub se: Vec<f64>,
    pub n: usize,
    /// Area under the ROC curve: how well positions are ranked. 0.5 is chance.
    pub auc: f64,
    /// Mean squared error of the probabilities. Lower is better; predicting
    /// the base rate every time scores p(1-p).
    pub brier: f64,
    /// Brier score of always predicting the base rate, for comparison.
    pub brier_baseline: f64,
    pub iterations: usize,
}

#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// Fit `P(y=1) = sigmoid(a + x.b)` by Newton-Raphson on the log-likelihood.
pub fn fit_logit(names: &[String], x: &[Vec<f64>], y: &[f64]) -> Option<Logit> {
    let n = y.len();
    let k = names.len();
    if n <= k + 10 {
        return None;
    }
    // Standardise the regressors so the Newton steps are well scaled, then
    // convert back at the end. Constant columns are dropped first: they carry
    // no information and would leave the Hessian singular.
    let full_mean: Vec<f64> = (0..k)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let full_sd: Vec<f64> = (0..k)
        .map(|j| {
            let m = full_mean[j];
            (x.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / n as f64).sqrt()
        })
        .collect();
    let keep: Vec<usize> = (0..k).filter(|&j| full_sd[j] > 1e-9).collect();
    let dropped: Vec<String> = (0..k)
        .filter(|j| full_sd[*j] <= 1e-9)
        .map(|j| names[j].clone())
        .collect();
    if keep.is_empty() || n <= keep.len() + 10 {
        return None;
    }
    let kk = keep.len();
    let mean: Vec<f64> = keep.iter().map(|&j| full_mean[j]).collect();
    let sd: Vec<f64> = keep.iter().map(|&j| full_sd[j]).collect();
    let z: Vec<Vec<f64>> = x
        .iter()
        .map(|r| (0..kk).map(|i| (r[keep[i]] - mean[i]) / sd[i]).collect())
        .collect();
    let k = kk;

    // Coefficients, with the intercept in slot k.
    let mut b = vec![0.0f64; k + 1];
    let base = y.iter().sum::<f64>() / n as f64;
    b[k] = (base / (1.0 - base)).ln();
    let mut iterations = 0;
    let mut cov = vec![vec![0.0; k + 1]; k + 1];

    for _ in 0..40 {
        iterations += 1;
        let mut grad = vec![0.0f64; k + 1];
        let mut hess = vec![vec![0.0f64; k + 1]; k + 1];
        for (row, &yi) in z.iter().zip(y.iter()) {
            let lin = b[k] + (0..k).map(|j| b[j] * row[j]).sum::<f64>();
            let p = sigmoid(lin);
            let w = (p * (1.0 - p)).max(1e-10);
            let r = yi - p;
            for i in 0..k {
                grad[i] += r * row[i];
            }
            grad[k] += r;
            for i in 0..=k {
                let xi = if i < k { row[i] } else { 1.0 };
                for j in 0..=k {
                    let xj = if j < k { row[j] } else { 1.0 };
                    hess[i][j] += w * xi * xj;
                }
            }
        }
        // A touch of ridge keeps the solve stable when two features are
        // nearly collinear, which costs nothing at this sample size.
        for i in 0..=k {
            hess[i][i] += 1e-6 * n as f64;
        }
        let inv = invert(&hess)?;
        let mut step = vec![0.0f64; k + 1];
        let mut max_step = 0.0f64;
        for i in 0..=k {
            step[i] = (0..=k).map(|j| inv[i][j] * grad[j]).sum();
            max_step = max_step.max(step[i].abs());
        }
        // Damp the first steps so a separable feature cannot blow up.
        let damp = if max_step > 2.0 { 2.0 / max_step } else { 1.0 };
        for i in 0..=k {
            b[i] += damp * step[i];
        }
        cov = inv;
        if max_step < 1e-8 {
            break;
        }
    }

    // Predictions, then AUC and Brier on the original scale.
    let preds: Vec<f64> = z
        .iter()
        .map(|row| sigmoid(b[k] + (0..k).map(|j| b[j] * row[j]).sum::<f64>()))
        .collect();
    let brier = preds
        .iter()
        .zip(y.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / n as f64;
    let brier_baseline = y.iter().map(|t| (base - t).powi(2)).sum::<f64>() / n as f64;

    // Undo the standardisation, scattering back into the full feature list so
    // that callers can index coefficients by their original position.
    let mut coef = vec![0.0f64; names.len()];
    let mut se = vec![0.0f64; names.len()];
    for i in 0..k {
        coef[keep[i]] = b[i] / sd[i];
        se[keep[i]] = cov[i][i].max(0.0).sqrt() / sd[i];
    }
    let intercept = b[k] - (0..k).map(|i| b[i] * mean[i] / sd[i]).sum::<f64>();

    Some(Logit {
        names: names.to_vec(),
        dropped,
        intercept,
        coef,
        se,
        n,
        auc: auc(&preds, y),
        brier,
        brier_baseline,
        iterations,
    })
}

/// Area under the ROC curve, by rank sum.
pub fn auc(preds: &[f64], y: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..preds.len()).collect();
    idx.sort_by(|&a, &b| preds[a].partial_cmp(&preds[b]).unwrap());
    // Average ranks over ties.
    let mut rank = vec![0.0f64; preds.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && preds[idx[j + 1]] == preds[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            rank[idx[k]] = avg;
        }
        i = j + 1;
    }
    let pos: f64 = y.iter().sum();
    let neg = y.len() as f64 - pos;
    if pos == 0.0 || neg == 0.0 {
        return f64::NAN;
    }
    let sum_pos: f64 = rank.iter().zip(y.iter()).filter(|(_, t)| **t > 0.5).map(|(r, _)| r).sum();
    (sum_pos - pos * (pos + 1.0) / 2.0) / (pos * neg)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalibBin {
    pub lo: f64,
    pub hi: f64,
    pub n: usize,
    pub mean_predicted: f64,
    pub actual: f64,
}

/// Reliability table: within each band of predicted probability, how often the
/// outcome actually happened.
pub fn calibration(preds: &[f64], y: &[f64], bins: usize) -> Vec<CalibBin> {
    (0..bins)
        .filter_map(|b| {
            let lo = b as f64 / bins as f64;
            let hi = (b + 1) as f64 / bins as f64;
            let sel: Vec<usize> = (0..preds.len())
                .filter(|&i| preds[i] >= lo && (preds[i] < hi || (b == bins - 1 && preds[i] <= hi)))
                .collect();
            if sel.is_empty() {
                return None;
            }
            Some(CalibBin {
                lo,
                hi,
                n: sel.len(),
                mean_predicted: sel.iter().map(|&i| preds[i]).sum::<f64>() / sel.len() as f64,
                actual: sel.iter().map(|&i| y[i]).sum::<f64>() / sel.len() as f64,
            })
        })
        .collect()
}

// ------------------------------------------------------------- feature set --

/// Which information the model is allowed to use.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Info {
    /// Only what a spectator could read off a live game: public score, piece
    /// counts, hand *sizes*, cards played face up.
    Spectator,
    /// Everything, hidden victory-point cards included.
    Full,
}

pub fn feature_names(info: Info) -> Vec<String> {
    let mut v: Vec<&str> = vec![
        "vp",
        "vp_lead",
        "pips",
        "pips_lead",
        "breadth",
        "hand_size",
        "dev_in_hand",
        "longest_road",
        "largest_army",
        "knights",
        "settlements",
        "cities",
        "own_turns",
    ];
    if info == Info::Full {
        v.push("hidden_vp_cards");
    }
    v.into_iter().map(|s| s.to_string()).collect()
}

/// Extract one row per player from the snapshot at `marker`, with features
/// expressed relative to the rest of the table where that matters.
pub fn rows(records: &[GameRecord], marker: usize, info: Info) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for g in records {
        if g.players.iter().any(|p| p.metrics.snapshots.len() <= marker) {
            continue;
        }
        let snaps: Vec<&crate::metrics::PhaseSnapshot> = g
            .players
            .iter()
            .map(|p| &p.metrics.snapshots[marker])
            .collect();
        for (i, p) in g.players.iter().enumerate() {
            let s = snaps[i];
            let own_vp = match info {
                Info::Spectator => s.vp_public as f64,
                Info::Full => s.vp as f64,
            };
            // Standing relative to the strongest rival is what decides a race.
            let best_other_vp = snaps
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, o)| match info {
                    Info::Spectator => o.vp_public as f64,
                    Info::Full => o.vp as f64,
                })
                .fold(f64::MIN, f64::max);
            let best_other_pips = snaps
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, o)| o.pips as f64)
                .fold(f64::MIN, f64::max);
            let mut row = vec![
                own_vp,
                own_vp - best_other_vp,
                s.pips as f64,
                s.pips as f64 - best_other_pips,
                s.breadth as f64,
                s.hand_size as f64,
                s.dev_in_hand as f64,
                s.has_longest_road as u8 as f64,
                s.has_largest_army as u8 as f64,
                s.knights as f64,
                s.settlements as f64,
                s.cities as f64,
                s.own_turns as f64,
            ];
            if info == Info::Full {
                row.push((s.vp as i32 - s.vp_public as i32) as f64);
            }
            x.push(row);
            y.push(p.won as u8 as f64);
        }
    }
    (x, y)
}

/// Fit and score a model at one phase boundary.
///
/// The model is fitted on one part of the games and scored on another. An
/// in-sample calibration table flatters itself by construction, and a meter
/// that is only honest about positions it has already seen is no use on a
/// live game.
pub struct Report {
    pub info: Info,
    pub marker: usize,
    pub fit: Logit,
    /// Rows the model was scored on, which it never saw while fitting.
    pub test_n: usize,
    pub test_auc: f64,
    pub test_brier: f64,
    pub test_brier_baseline: f64,
    pub bins: Vec<CalibBin>,
    /// Largest gap between predicted and actual in any well-populated bin.
    pub max_miscalibration: f64,
}

impl Report {
    /// Fraction of the Brier score removed relative to always predicting the
    /// base rate.
    pub fn skill(&self) -> f64 {
        1.0 - self.test_brier / self.test_brier_baseline
    }
}

fn predict(fit: &Logit, r: &[f64]) -> f64 {
    sigmoid(fit.intercept + (0..fit.coef.len()).map(|j| fit.coef[j] * r[j]).sum::<f64>())
}

pub fn evaluate(records: &[GameRecord], marker: usize, info: Info) -> Option<Report> {
    let names = feature_names(info);
    // Split by game, not by row, so the four players of one game never land
    // on opposite sides of the split -- their outcomes are dependent.
    let cut = records.len() * 7 / 10;
    if cut < 50 || records.len() - cut < 50 {
        return None;
    }
    let (train_recs, test_recs) = records.split_at(cut);
    let (xtr, ytr) = rows(train_recs, marker, info);
    let (xte, yte) = rows(test_recs, marker, info);
    if xtr.is_empty() || xte.is_empty() {
        return None;
    }
    let fit = fit_logit(&names, &xtr, &ytr)?;

    let preds: Vec<f64> = xte.iter().map(|r| predict(&fit, r)).collect();
    let base = ytr.iter().sum::<f64>() / ytr.len() as f64;
    let test_brier = preds
        .iter()
        .zip(yte.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / yte.len() as f64;
    let test_brier_baseline =
        yte.iter().map(|t| (base - t).powi(2)).sum::<f64>() / yte.len() as f64;
    let bins = calibration(&preds, &yte, 10);
    let max_miscalibration = bins
        .iter()
        .filter(|b| b.n >= 200)
        .map(|b| (b.mean_predicted - b.actual).abs())
        .fold(0.0f64, f64::max);
    Some(Report {
        info,
        marker,
        test_n: yte.len(),
        test_auc: auc(&preds, &yte),
        test_brier,
        test_brier_baseline,
        fit,
        bins,
        max_miscalibration,
    })
}
