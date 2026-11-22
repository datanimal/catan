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

// ------------------------------------------------------- per-turn model ----

/// Features for a per-turn estimate, read from a [`TurnSample`].
///
/// Two things matter here that the phase-boundary model did not need. First,
/// **game progress**: the same position is worth more at turn 70 than at turn
/// 20, because there is less time for anyone to catch up. Second, the features
/// are relative to the table, since a race is decided by standing rather than
/// by absolute totals.
pub fn turn_feature_names(info: Info) -> Vec<String> {
    let mut v: Vec<&str> = vec![
        "vp",
        "vp_lead",
        "vp_to_target",
        "table_max_vp",
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
        "turn",
        "is_my_turn",
        // How far the race has run, as a fraction of the target.
        "progress",
        // The same features again, scaled by progress. Without these one set
        // of coefficients has to serve the whole game, and a lead that means
        // little on turn 10 gets priced as if it meant what it does on turn
        // 70. That showed up as calibration being out by 25 points in the
        // opening fifth.
        "vp_lead_x_progress",
        "pips_lead_x_progress",
        "dev_x_progress",
        "my_gap_to_target",
    ];
    if info == Info::Full {
        v.push("hidden_vp_cards");
    }
    v.into_iter().map(|s| s.to_string()).collect()
}

/// Build the feature row for one player at one moment.
pub fn turn_features(
    sample: &crate::metrics::TurnSample,
    me: usize,
    vp_target: u8,
    info: Info,
) -> Vec<f64> {
    let s = &sample.players[me];
    let own_vp = match info {
        Info::Spectator => s.vp_public as f64,
        Info::Full => s.vp_true as f64,
    };
    let vp_of = |p: &crate::metrics::PlayerSample| match info {
        Info::Spectator => p.vp_public as f64,
        Info::Full => p.vp_true as f64,
    };
    let best_other_vp = sample
        .players
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != me)
        .map(|(_, o)| vp_of(o))
        .fold(f64::MIN, f64::max);
    let best_other_pips = sample
        .players
        .iter()
        .enumerate()
        .filter(|(j, _)| *j != me)
        .map(|(_, o)| o.pips as f64)
        .fold(f64::MIN, f64::max);
    let table_max = sample.players.iter().map(vp_of).fold(f64::MIN, f64::max);

    let mut row = vec![
        own_vp,
        own_vp - best_other_vp,
        // How much further the leader has to go: small means the game is
        // nearly over and positions are close to settled.
        (vp_target as f64 - table_max).max(0.0),
        table_max,
        s.pips as f64,
        s.pips as f64 - best_other_pips,
        s.breadth as f64,
        s.hand_size as f64,
        s.dev_in_hand as f64,
        s.longest_road as u8 as f64,
        s.largest_army as u8 as f64,
        s.knights as f64,
        s.settlements as f64,
        s.cities as f64,
        sample.turn as f64,
        (sample.current as usize == me) as u8 as f64,
    ];
    let progress = (table_max / vp_target as f64).clamp(0.0, 1.0);
    row.push(progress);
    row.push((own_vp - best_other_vp) * progress);
    row.push((s.pips as f64 - best_other_pips) * progress);
    row.push(s.dev_in_hand as f64 * progress);
    row.push((vp_target as f64 - own_vp).max(0.0));
    if info == Info::Full {
        row.push((s.vp_true as i32 - s.vp_public as i32) as f64);
    }
    row
}

/// Progress bands, as a fraction of the victory-point target reached by the
/// leader. A separate model is fitted for each.
///
/// One set of coefficients cannot serve a whole game. Early on almost nothing
/// separates the players and the honest answer is close to even; late, a
/// single point of lead is nearly decisive. Fitting one line through both
/// regimes left calibration out by 25 points in the opening fifth even after
/// progress interactions were added, so each band gets its own fit.
pub const STAGE_BANDS: [(f64, f64, &str); 5] = [
    (0.0, 0.30, "opening"),
    (0.30, 0.50, "early"),
    (0.50, 0.70, "middle"),
    (0.70, 0.85, "late"),
    (0.85, 1.01, "endgame"),
];

/// One band's coefficients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageModel {
    pub label: String,
    pub lo: f64,
    pub hi: f64,
    pub intercept: f64,
    pub coef: Vec<f64>,
    /// Platt recalibration of the normalised probability, as
    /// `sigmoid(a + b * logit(p))`.
    ///
    /// Each seat's probability is fitted on its own and then divided through
    /// to make the table sum to one. That division is not the right transform
    /// -- it shrinks everything proportionally -- and it left a systematic
    /// tilt: positions quoted at 55% actually won 59%. These two numbers are
    /// fitted on the training predictions to take the tilt out.
    #[serde(default = "platt_default_a")]
    pub platt_a: f64,
    #[serde(default = "platt_default_b")]
    pub platt_b: f64,
    pub rows: usize,
    pub auc: f64,
    pub brier: f64,
    pub worst_gap: f64,
    pub dropped: Vec<String>,
}

/// A fitted model, ready to be saved and reloaded for live use.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WinModel {
    pub info_full: bool,
    pub vp_target: u8,
    pub names: Vec<String>,
    pub stages: Vec<StageModel>,
    /// Games the model was trained on, for provenance.
    pub trained_on_games: usize,
    pub lineup: Vec<String>,
    pub test_auc: f64,
    pub test_brier: f64,
    pub test_brier_baseline: f64,
    pub worst_calibration_gap: f64,
}

fn platt_default_a() -> f64 {
    0.0
}
fn platt_default_b() -> f64 {
    1.0
}

#[inline]
fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// How far the race has run, from the leader's score. Observable live.
pub fn progress_of(sample: &crate::metrics::TurnSample, vp_target: u8, info: Info) -> f64 {
    let m = sample
        .players
        .iter()
        .map(|p| match info {
            Info::Spectator => p.vp_public as f64,
            Info::Full => p.vp_true as f64,
        })
        .fold(f64::MIN, f64::max);
    (m / vp_target as f64).clamp(0.0, 1.0)
}

impl WinModel {
    pub fn info(&self) -> Info {
        if self.info_full {
            Info::Full
        } else {
            Info::Spectator
        }
    }

    /// The band this position falls in.
    pub fn stage_for(&self, sample: &crate::metrics::TurnSample) -> &StageModel {
        let p = progress_of(sample, self.vp_target, self.info());
        self.stages
            .iter()
            .find(|s| p >= s.lo && p < s.hi)
            .unwrap_or_else(|| self.stages.last().expect("at least one stage"))
    }

    /// Raw per-player probability, before the table is made to sum to one.
    pub fn raw(&self, sample: &crate::metrics::TurnSample, me: usize) -> f64 {
        let st = self.stage_for(sample);
        let f = turn_features(sample, me, self.vp_target, self.info());
        sigmoid(st.intercept + (0..st.coef.len()).map(|j| st.coef[j] * f[j]).sum::<f64>())
    }

    /// Win probability for every seat: fitted, normalised so the table sums
    /// to one, recalibrated, then normalised again.
    ///
    /// Exactly one player wins, so the probabilities must add up; each seat is
    /// fitted on its own, so nothing makes them. Dividing through enforces it
    /// but tilts the result, which the Platt step then corrects.
    pub fn predict_table(&self, sample: &crate::metrics::TurnSample) -> Vec<f64> {
        let n = sample.players.len();
        let raw: Vec<f64> = (0..n).map(|i| self.raw(sample, i)).collect();
        let total: f64 = raw.iter().sum();
        if total <= 1e-12 || !total.is_finite() {
            return vec![1.0 / n as f64; n];
        }
        let st = self.stage_for(sample);
        let adjusted: Vec<f64> = raw
            .iter()
            .map(|p| sigmoid(st.platt_a + st.platt_b * logit(p / total)))
            .collect();
        let t2: f64 = adjusted.iter().sum();
        if t2 <= 1e-12 || !t2.is_finite() {
            return vec![1.0 / n as f64; n];
        }
        adjusted.iter().map(|p| p / t2).collect()
    }

    /// Normalised but un-recalibrated probabilities, used while fitting the
    /// Platt step.
    fn predict_table_uncalibrated(&self, sample: &crate::metrics::TurnSample) -> Vec<f64> {
        let n = sample.players.len();
        let raw: Vec<f64> = (0..n).map(|i| self.raw(sample, i)).collect();
        let total: f64 = raw.iter().sum();
        if total <= 1e-12 || !total.is_finite() {
            return vec![1.0 / n as f64; n];
        }
        raw.iter().map(|p| p / total).collect()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(path, json)
    }

    pub fn load(path: &str) -> std::io::Result<WinModel> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

/// Train a per-turn model, holding out the last 30% of games to score it.
pub fn train_turn_model(
    records: &[GameRecord],
    vp_target: u8,
    info: Info,
    lineup: Vec<String>,
) -> Option<(WinModel, Vec<CalibBin>)> {
    if records.iter().all(|g| g.timeline.is_empty()) {
        return None;
    }
    let names = turn_feature_names(info);
    let cut = records.len() * 7 / 10;
    if cut < 20 || records.len() - cut < 20 {
        return None;
    }
    let (tr, te) = records.split_at(cut);

    // Fit each progress band on its own rows.
    let mut stages = Vec::new();
    for &(lo, hi, label) in STAGE_BANDS.iter() {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for g in tr {
            for sample in &g.timeline {
                let p = progress_of(sample, vp_target, info);
                if p < lo || p >= hi {
                    continue;
                }
                for me in 0..sample.players.len() {
                    x.push(turn_features(sample, me, vp_target, info));
                    y.push(g.players[me].won as u8 as f64);
                }
            }
        }
        // A band with nothing in it, or with every row the same outcome,
        // falls back to the base rate rather than a degenerate fit.
        let fit = if x.len() > names.len() + 50 {
            fit_logit(&names, &x, &y)
        } else {
            None
        };
        let (intercept, coef, dropped) = match fit {
            Some(f) => (f.intercept, f.coef, f.dropped),
            None => {
                let base = if y.is_empty() {
                    0.25
                } else {
                    (y.iter().sum::<f64>() / y.len() as f64).clamp(1e-3, 1.0 - 1e-3)
                };
                (
                    (base / (1.0 - base)).ln(),
                    vec![0.0; names.len()],
                    names.clone(),
                )
            }
        };
        stages.push(StageModel {
            label: label.to_string(),
            lo,
            hi,
            intercept,
            coef,
            platt_a: 0.0,
            platt_b: 1.0,
            rows: x.len(),
            auc: f64::NAN,
            brier: f64::NAN,
            worst_gap: f64::NAN,
            dropped,
        });
    }

    let mut model = WinModel {
        info_full: info == Info::Full,
        vp_target,
        names,
        stages,
        trained_on_games: tr.len(),
        lineup,
        test_auc: 0.0,
        test_brier: 0.0,
        test_brier_baseline: 0.0,
        worst_calibration_gap: 0.0,
    };

    // Fit the recalibration on the training games, before any held-out data
    // is touched.
    for si in 0..model.stages.len() {
        let (lo, hi, _) = STAGE_BANDS[si];
        let mut z = Vec::new();
        let mut y = Vec::new();
        for g in tr {
            for sample in &g.timeline {
                let p = progress_of(sample, vp_target, info);
                if p < lo || p >= hi {
                    continue;
                }
                let table = model.predict_table_uncalibrated(sample);
                for (i, prob) in table.iter().enumerate() {
                    z.push(vec![logit(*prob)]);
                    y.push(g.players[i].won as u8 as f64);
                }
            }
        }
        if z.len() > 200 {
            if let Some(f) = fit_logit(&["logit_p".to_string()], &z, &y) {
                model.stages[si].platt_a = f.intercept;
                model.stages[si].platt_b = f.coef[0];
            }
        }
    }

    // Score on held-out games, with the whole pipeline applied, since that is
    // how the model is used.
    let mut preds = Vec::new();
    let mut ys = Vec::new();
    let mut per_stage: Vec<(Vec<f64>, Vec<f64>)> =
        (0..model.stages.len()).map(|_| (Vec::new(), Vec::new())).collect();
    for g in te {
        for sample in &g.timeline {
            let table = model.predict_table(sample);
            let p = progress_of(sample, vp_target, info);
            let si = STAGE_BANDS
                .iter()
                .position(|&(lo, hi, _)| p >= lo && p < hi)
                .unwrap_or(STAGE_BANDS.len() - 1);
            for (i, prob) in table.iter().enumerate() {
                let won = g.players[i].won as u8 as f64;
                preds.push(*prob);
                ys.push(won);
                per_stage[si].0.push(*prob);
                per_stage[si].1.push(won);
            }
        }
    }
    if ys.is_empty() {
        return None;
    }
    let base = 1.0 / records[0].players.len() as f64;
    model.test_auc = auc(&preds, &ys);
    model.test_brier =
        preds.iter().zip(ys.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / ys.len() as f64;
    model.test_brier_baseline =
        ys.iter().map(|t| (base - t).powi(2)).sum::<f64>() / ys.len() as f64;
    let bins = calibration(&preds, &ys, 10);
    model.worst_calibration_gap = bins
        .iter()
        .filter(|b| b.n >= 200)
        .map(|b| (b.mean_predicted - b.actual).abs())
        .fold(0.0f64, f64::max);
    for (i, (p, t)) in per_stage.iter().enumerate() {
        if t.is_empty() {
            continue;
        }
        model.stages[i].auc = auc(p, t);
        model.stages[i].brier =
            p.iter().zip(t.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / t.len() as f64;
        model.stages[i].worst_gap = calibration(p, t, 10)
            .iter()
            .filter(|b| b.n >= 100)
            .map(|b| (b.mean_predicted - b.actual).abs())
            .fold(0.0f64, f64::max);
    }
    Some((model, bins))
}

/// How closely the instant model tracks a Monte Carlo rollout of the same
/// position. The rollout is the accurate reference; the model is a
/// compressed approximation of it that runs in microseconds.
pub struct RolloutAgreement {
    pub band: String,
    pub positions: usize,
    /// Mean absolute difference in win probability, in points.
    pub mean_abs_diff: f64,
    pub worst_abs_diff: f64,
    /// Correlation between the two estimates.
    pub correlation: f64,
}

/// Compare the model against rollouts on positions sampled from `records`.
///
/// Both are estimates of the same thing, but of different quality: rollouts
/// play the position out thousands of times, the model compresses it into a
/// handful of coefficients. Reporting the gap says which one a given use needs
/// -- an instant overlay can afford the model's error, a post-game analysis
/// probably should not.
pub fn compare_to_rollouts(
    records: &[GameRecord],
    model: &WinModel,
    states: &[(f64, crate::state::GameState, crate::metrics::TurnSample)],
    lineup: &[crate::bots::Strategy],
    rollouts: usize,
    seed: u64,
) -> Vec<RolloutAgreement> {
    let _ = records;
    let mut out = Vec::new();
    for &(lo, hi, label) in STAGE_BANDS.iter() {
        let mut diffs: Vec<f64> = Vec::new();
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        let mut n = 0usize;
        for (progress, st, sample) in states {
            if *progress < lo || *progress >= hi {
                continue;
            }
            let m = model.predict_table(sample);
            let r = crate::rollout::evaluate(st, lineup, rollouts, seed + n as u64, true);
            for i in 0..m.len() {
                diffs.push((m[i] - r.win_prob[i]).abs());
                xs.push(m[i]);
                ys.push(r.win_prob[i]);
            }
            n += 1;
        }
        if diffs.is_empty() {
            continue;
        }
        let mean_abs = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let worst = diffs.iter().cloned().fold(0.0f64, f64::max);
        let mx = xs.iter().sum::<f64>() / xs.len() as f64;
        let my = ys.iter().sum::<f64>() / ys.len() as f64;
        let cov: f64 = xs.iter().zip(ys.iter()).map(|(a, b)| (a - mx) * (b - my)).sum();
        let vx: f64 = xs.iter().map(|a| (a - mx).powi(2)).sum();
        let vy: f64 = ys.iter().map(|b| (b - my).powi(2)).sum();
        out.push(RolloutAgreement {
            band: label.to_string(),
            positions: n,
            mean_abs_diff: mean_abs,
            worst_abs_diff: worst,
            correlation: if vx > 0.0 && vy > 0.0 {
                cov / (vx.sqrt() * vy.sqrt())
            } else {
                f64::NAN
            },
        });
    }
    out
}

/// Accuracy of a trained model broken down by how far the game has run, which
/// is what tells you whether a meter is worth showing early on.
pub struct StageAccuracy {
    pub label: String,
    pub n: usize,
    pub auc: f64,
    pub brier: f64,
    pub worst_gap: f64,
}

pub fn accuracy_by_stage(model: &WinModel, records: &[GameRecord]) -> Vec<StageAccuracy> {
    let bands: [(f64, f64, &str); 5] = [
        (0.0, 0.2, "first fifth"),
        (0.2, 0.4, "second fifth"),
        (0.4, 0.6, "middle fifth"),
        (0.6, 0.8, "fourth fifth"),
        (0.8, 1.01, "last fifth"),
    ];
    bands
        .iter()
        .map(|&(lo, hi, label)| {
            let mut preds = Vec::new();
            let mut ys = Vec::new();
            for g in records {
                let total = g.timeline.len() as f64;
                if total < 5.0 {
                    continue;
                }
                for (si, sample) in g.timeline.iter().enumerate() {
                    let frac = si as f64 / total;
                    if frac < lo || frac >= hi {
                        continue;
                    }
                    let table = model.predict_table(sample);
                    for (i, p) in table.iter().enumerate() {
                        preds.push(*p);
                        ys.push(g.players[i].won as u8 as f64);
                    }
                }
            }
            let n = ys.len();
            let brier = if n == 0 {
                f64::NAN
            } else {
                preds
                    .iter()
                    .zip(ys.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<f64>()
                    / n as f64
            };
            let worst = calibration(&preds, &ys, 10)
                .iter()
                .filter(|b| b.n >= 100)
                .map(|b| (b.mean_predicted - b.actual).abs())
                .fold(0.0f64, f64::max);
            StageAccuracy {
                label: label.to_string(),
                n,
                auc: if n == 0 { f64::NAN } else { auc(&preds, &ys) },
                brier,
                worst_gap: worst,
            }
        })
        .collect()
}
