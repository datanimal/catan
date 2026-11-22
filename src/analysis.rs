//! Aggregation, and the luck/skill decomposition.
//!
//! The question "did this strategy win because it played well or because it
//! rolled well?" is answered by regressing victory points on dice luck across
//! every player-game in the run:
//!
//! ```text
//! VP = a + b * dice_luck + e
//! ```
//!
//! `dice_luck` is cards above what average dice would have paid that exact
//! board position over that exact number of rolls. It is genuinely exogenous:
//! a player chooses where to build, but not which numbers come up afterwards.
//! The **skill index** is victory points with that effect taken out:
//!
//! ```text
//! skill_index = VP - b * (dice_luck - mean dice_luck)
//! ```
//!
//! It is in victory points and centred on the field average, so 8.4 means
//! "performed like an 8.4-point player once the dice are accounted for".
//! `skill_index_z` expresses the same thing in standard deviations.
//!
//! # What is deliberately *not* adjusted for
//!
//! An earlier version of this model also regressed on a combined
//! "robber luck" term that included cards discarded on a 7. That was wrong.
//! Discarding is mostly a consequence of choosing to sit on a big hand, so it
//! is a style choice, and the fitted coefficient came out *negative* -- the
//! regression had found that players who discard more tend to score more,
//! because holding many cards goes with having a strong engine. Adjusting for
//! it would have subtracted a skill rather than a luck.
//!
//! So the headline index adjusts for the dice alone. Theft, robber blocking
//! and monopolies are reported separately as `theft_luck`, and a secondary
//! `skill_index_theft_adj` additionally removes them -- flagged because
//! opponents aim the robber at whoever is winning, which makes that term
//! partly a consequence of doing well.

use crate::metrics::{GameRecord, PlayerResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ------------------------------------------------------------ regression ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LuckModel {
    pub n: usize,
    pub intercept: f64,
    /// Victory points gained per extra card of dice luck. The headline
    /// adjustment uses this and nothing else.
    pub beta_dice: f64,
    /// Victory points per card of theft luck, fitted in a separate two-term
    /// regression. Reported for the secondary index only; see the module note
    /// on why it is not in the headline figure.
    pub beta_theft: f64,
    pub mean_dice: f64,
    pub mean_theft: f64,
    /// Fraction of the variance in VP explained by dice luck alone.
    pub r2: f64,
    /// Spread of VP once dice luck is removed; the unit of `skill_index_z`.
    pub resid_sd: f64,
}

/// Solve a small symmetric linear system by Gaussian elimination with
/// partial pivoting. Returns None if the system is singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n)
            .max_by(|&i, &j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap())?;
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in (col + 1)..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Fit `VP ~ 1 + dice_luck` over every player-game, plus a secondary
/// two-term fit that also includes theft luck.
pub fn fit_luck_model(records: &[GameRecord]) -> LuckModel {
    let rows: Vec<(f64, f64, f64)> = records
        .iter()
        .flat_map(|g| g.players.iter())
        .map(|p| (p.vp as f64, p.metrics.dice_luck(), p.metrics.theft_luck()))
        .collect();
    let n = rows.len();
    let fallback = LuckModel {
        n,
        intercept: 0.0,
        beta_dice: 0.0,
        beta_theft: 0.0,
        mean_dice: 0.0,
        mean_theft: 0.0,
        r2: 0.0,
        resid_sd: 1.0,
    };
    if n < 20 {
        return fallback;
    }
    let mean_dice = rows.iter().map(|r| r.1).sum::<f64>() / n as f64;
    let mean_theft = rows.iter().map(|r| r.2).sum::<f64>() / n as f64;
    let mean_vp = rows.iter().map(|r| r.0).sum::<f64>() / n as f64;

    // Headline: simple regression on dice luck.
    let mut sdd = 0.0;
    let mut sdy = 0.0;
    for &(y, d, _) in &rows {
        sdd += (d - mean_dice).powi(2);
        sdy += (d - mean_dice) * (y - mean_vp);
    }
    if sdd <= 0.0 {
        return fallback;
    }
    let beta_dice = sdy / sdd;

    let mut sse = 0.0;
    let mut sst = 0.0;
    for &(y, d, _) in &rows {
        let pred = mean_vp + beta_dice * (d - mean_dice);
        sse += (y - pred).powi(2);
        sst += (y - mean_vp).powi(2);
    }
    let dof = (n as f64 - 2.0).max(1.0);

    // Secondary: add theft luck, for the alternative index.
    let mut sxx = [[0.0f64; 2]; 2];
    let mut sxy = [0.0f64; 2];
    for &(y, d, t) in &rows {
        let x = [d - mean_dice, t - mean_theft];
        let yc = y - mean_vp;
        for i in 0..2 {
            sxy[i] += x[i] * yc;
            for j in 0..2 {
                sxx[i][j] += x[i] * x[j];
            }
        }
    }
    let beta_theft = solve(
        vec![vec![sxx[0][0], sxx[0][1]], vec![sxx[1][0], sxx[1][1]]],
        vec![sxy[0], sxy[1]],
    )
    .map(|b| b[1])
    .unwrap_or(0.0);

    LuckModel {
        n,
        intercept: mean_vp - beta_dice * mean_dice,
        beta_dice,
        beta_theft,
        mean_dice,
        mean_theft,
        r2: if sst > 0.0 { 1.0 - sse / sst } else { 0.0 },
        resid_sd: (sse / dof).sqrt(),
    }
}

impl LuckModel {
    /// A player's victory points with the dice effect removed. The headline
    /// skill measure.
    pub fn skill_of(&self, p: &PlayerResult) -> f64 {
        p.vp as f64 - self.beta_dice * (p.metrics.dice_luck() - self.mean_dice)
    }

    /// Also removes theft and robber blocking. Partly endogenous -- the robber
    /// follows the leader -- so treat it as a cross-check, not the headline.
    pub fn skill_theft_adj_of(&self, p: &PlayerResult) -> f64 {
        self.skill_of(p) - self.beta_theft * (p.metrics.theft_luck() - self.mean_theft)
    }
}

// --------------------------------------------------------------- helpers ----

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// 95% confidence interval for a mean.
fn ci95(v: &[f64]) -> (f64, f64) {
    if v.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let m = mean(v);
    let e = 1.96 * sd(v) / (v.len() as f64).sqrt();
    (m - e, m + e)
}

/// Wilson score interval -- behaves sensibly for rates near 0 and 1.
pub fn wilson(successes: usize, n: usize) -> (f64, f64) {
    if n == 0 {
        return (f64::NAN, f64::NAN);
    }
    let z = 1.96f64;
    let p = successes as f64 / n as f64;
    let nf = n as f64;
    let denom = 1.0 + z * z / nf;
    let centre = p + z * z / (2.0 * nf);
    let margin = z * ((p * (1.0 - p) / nf) + z * z / (4.0 * nf * nf)).sqrt();
    (
        ((centre - margin) / denom).max(0.0),
        ((centre + margin) / denom).min(1.0),
    )
}

// ------------------------------------------------------------ strategies ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyStats {
    pub name: String,
    pub games: usize,
    pub wins: usize,
    pub win_rate: f64,
    pub win_rate_lo: f64,
    pub win_rate_hi: f64,
    pub mean_vp: f64,

    // --- luck: things that happened to the player ---
    pub mean_dice_luck: f64,
    /// Cards taken from / denied this player by opponents.
    pub mean_theft_luck: f64,
    /// Cards thrown away on a 7 -- a measure of hoarding, not of luck.
    pub mean_discarded: f64,
    pub mean_bank_shortage: f64,

    // --- placement: the board position they earned ---
    /// Expected cards per roll from their buildings, averaged over the game.
    pub mean_rate_per_roll: f64,
    pub mean_init_pips: f64,
    pub mean_final_pips: f64,
    pub mean_breadth: f64,

    // --- conversion: what they did with what they got ---
    pub mean_inflow: f64,
    pub vp_per_100_cards: f64,

    // --- the headline ---
    /// Victory points with the dice effect regressed out.
    pub skill_index: f64,
    pub skill_index_lo: f64,
    pub skill_index_hi: f64,
    /// The same, in standard deviations above the field.
    pub skill_index_z: f64,
    /// Secondary index that also removes theft; see the module note.
    pub skill_index_theft_adj: f64,

    // --- behaviour ---
    pub mean_settlements: f64,
    pub mean_cities: f64,
    pub mean_roads: f64,
    pub mean_dev_bought: f64,
    pub mean_knights: f64,
    pub mean_bank_trades: f64,
    pub mean_trades_accepted: f64,
    pub mean_trades_given: f64,
    pub longest_road_rate: f64,
    pub largest_army_rate: f64,
    pub mean_vp_from_cards: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SeatStats {
    pub seat: u8,
    pub games: usize,
    pub wins: usize,
    pub win_rate: f64,
    pub win_rate_lo: f64,
    pub win_rate_hi: f64,
    pub mean_vp: f64,
    pub mean_init_pips: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bucket {
    pub label: String,
    pub games: usize,
    pub wins: usize,
    pub win_rate: f64,
    pub win_rate_lo: f64,
    pub win_rate_hi: f64,
    pub mean_vp: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Summary {
    pub games: usize,
    pub mean_turns: f64,
    pub timeouts: usize,
    pub model: LuckModel,
    pub strategies: Vec<StrategyStats>,
    pub seats: Vec<SeatStats>,
    /// Win rate by total pips of the two opening settlements.
    pub opening_pips: Vec<Bucket>,
    /// Win rate by how many distinct resources the opening covers.
    pub opening_breadth: Vec<Bucket>,
    pub roll_histogram: [u32; 13],
}

fn bucketize(
    rows: &[(f64, bool, f64)],
    edges: &[(f64, f64, &str)],
) -> Vec<Bucket> {
    edges
        .iter()
        .map(|&(lo, hi, label)| {
            let sel: Vec<&(f64, bool, f64)> =
                rows.iter().filter(|r| r.0 >= lo && r.0 < hi).collect();
            let games = sel.len();
            let wins = sel.iter().filter(|r| r.1).count();
            let (l, h) = wilson(wins, games);
            Bucket {
                label: label.to_string(),
                games,
                wins,
                win_rate: if games == 0 {
                    f64::NAN
                } else {
                    wins as f64 / games as f64
                },
                win_rate_lo: l,
                win_rate_hi: h,
                mean_vp: mean(&sel.iter().map(|r| r.2).collect::<Vec<_>>()),
            }
        })
        .collect()
}

pub fn summarize(records: &[GameRecord]) -> Summary {
    let model = fit_luck_model(records);
    // `skill_of` centres the luck terms, so the mean skill index across every
    // player-game equals the mean victory points. That is the right centre for
    // the z-score -- not the regression intercept, which is the value at zero
    // luck rather than at average luck.
    let field_mean_vp = mean(
        &records
            .iter()
            .flat_map(|g| g.players.iter())
            .map(|p| p.vp as f64)
            .collect::<Vec<_>>(),
    );

    // Group player-games by strategy name.
    let mut by_name: BTreeMap<&str, Vec<&PlayerResult>> = BTreeMap::new();
    for g in records {
        for p in &g.players {
            by_name.entry(p.policy.as_str()).or_default().push(p);
        }
    }

    let strategies = by_name
        .iter()
        .map(|(name, ps)| {
            let n = ps.len();
            let wins = ps.iter().filter(|p| p.won).count();
            let (wlo, whi) = wilson(wins, n);
            let skills: Vec<f64> = ps.iter().map(|p| model.skill_of(p)).collect();
            let (slo, shi) = ci95(&skills);
            let sk = mean(&skills);
            let f = |g: fn(&PlayerResult) -> f64| mean(&ps.iter().map(|p| g(p)).collect::<Vec<_>>());
            let inflow = f(|p| p.metrics.total_inflow() as f64);
            let vp = f(|p| p.vp as f64);
            StrategyStats {
                name: name.to_string(),
                games: n,
                wins,
                win_rate: wins as f64 / n as f64,
                win_rate_lo: wlo,
                win_rate_hi: whi,
                mean_vp: vp,
                mean_dice_luck: f(|p| p.metrics.dice_luck()),
                mean_theft_luck: f(|p| p.metrics.theft_luck()),
                mean_discarded: f(|p| p.metrics.discarded()),
                mean_bank_shortage: f(|p| p.metrics.bank_shortage_loss as f64),
                mean_rate_per_roll: f(|p| {
                    let r = p.metrics.rolls_seen;
                    if r == 0 {
                        0.0
                    } else {
                        p.metrics.expected_total() / r as f64
                    }
                }),
                mean_init_pips: f(|p| p.metrics.init_pips as f64),
                mean_final_pips: f(|p| p.metrics.final_pips as f64),
                mean_breadth: f(|p| p.metrics.final_res_breadth as f64),
                mean_inflow: inflow,
                vp_per_100_cards: if inflow > 0.0 { 100.0 * vp / inflow } else { 0.0 },
                skill_index: sk,
                skill_index_lo: slo,
                skill_index_hi: shi,
                skill_index_z: if model.resid_sd > 0.0 {
                    (sk - field_mean_vp) / model.resid_sd
                } else {
                    0.0
                },
                skill_index_theft_adj: mean(
                    &ps.iter().map(|p| model.skill_theft_adj_of(p)).collect::<Vec<_>>(),
                ),
                mean_settlements: f(|p| p.metrics.settlements_built as f64),
                mean_cities: f(|p| p.metrics.cities_built as f64),
                mean_roads: f(|p| p.metrics.roads_built as f64),
                mean_dev_bought: f(|p| p.metrics.dev_bought as f64),
                mean_knights: f(|p| p.metrics.knights_played as f64),
                mean_bank_trades: f(|p| p.metrics.bank_trades as f64),
                mean_trades_accepted: f(|p| p.metrics.trades_accepted_by_others as f64),
                mean_trades_given: f(|p| p.metrics.trades_accepted_for_others as f64),
                longest_road_rate: f(|p| p.metrics.had_longest_road as u8 as f64),
                largest_army_rate: f(|p| p.metrics.had_largest_army as u8 as f64),
                mean_vp_from_cards: f(|p| p.metrics.vp_from_cards as f64),
            }
        })
        .collect::<Vec<_>>();

    // Seat effects.
    let n_seats = records.first().map(|g| g.players.len()).unwrap_or(0);
    let seats = (0..n_seats as u8)
        .map(|s| {
            let ps: Vec<&PlayerResult> = records
                .iter()
                .filter_map(|g| g.players.iter().find(|p| p.seat == s))
                .collect();
            let n = ps.len();
            let wins = ps.iter().filter(|p| p.won).count();
            let (lo, hi) = wilson(wins, n);
            SeatStats {
                seat: s,
                games: n,
                wins,
                win_rate: if n == 0 { f64::NAN } else { wins as f64 / n as f64 },
                win_rate_lo: lo,
                win_rate_hi: hi,
                mean_vp: mean(&ps.iter().map(|p| p.vp as f64).collect::<Vec<_>>()),
                mean_init_pips: mean(
                    &ps.iter().map(|p| p.metrics.init_pips as f64).collect::<Vec<_>>(),
                ),
            }
        })
        .collect();

    // Opening-placement effects.
    let pip_rows: Vec<(f64, bool, f64)> = records
        .iter()
        .flat_map(|g| g.players.iter())
        .map(|p| (p.metrics.init_pips as f64, p.won, p.vp as f64))
        .collect();
    let opening_pips = bucketize(
        &pip_rows,
        &[
            (0.0, 16.0, "<16 pips"),
            (16.0, 18.0, "16-17 pips"),
            (18.0, 20.0, "18-19 pips"),
            (20.0, 22.0, "20-21 pips"),
            (22.0, 24.0, "22-23 pips"),
            (24.0, 1e9, "24+ pips"),
        ],
    );
    let breadth_rows: Vec<(f64, bool, f64)> = records
        .iter()
        .flat_map(|g| g.players.iter())
        .map(|p| (p.metrics.final_res_breadth as f64, p.won, p.vp as f64))
        .collect();
    let opening_breadth = bucketize(
        &breadth_rows,
        &[
            (0.0, 3.0, "<=2 resources"),
            (3.0, 4.0, "3 resources"),
            (4.0, 5.0, "4 resources"),
            (5.0, 1e9, "all 5 resources"),
        ],
    );

    let mut hist = [0u32; 13];
    for g in records {
        for (i, c) in g.roll_histogram.iter().enumerate() {
            hist[i] += c;
        }
    }

    Summary {
        games: records.len(),
        mean_turns: mean(&records.iter().map(|g| g.turns as f64).collect::<Vec<_>>()),
        timeouts: records.iter().filter(|g| g.timed_out).count(),
        model,
        strategies,
        seats,
        opening_pips,
        opening_breadth,
        roll_histogram: hist,
    }
}

// ------------------------------------------------------------------ OLS ----

/// Ordinary least squares with standard errors, for valuing board features.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ols {
    pub names: Vec<String>,
    pub intercept: f64,
    pub coef: Vec<f64>,
    pub se: Vec<f64>,
    pub n: usize,
    pub r2: f64,
    /// Residual standard deviation of the outcome.
    pub resid_sd: f64,
    /// Mean of each regressor, for evaluating effects at typical values.
    pub means: Vec<f64>,
    /// Covariance matrix of the coefficients. Needed to put an interval on a
    /// combination of them, such as "the value of a 2:1 port at 6 pips of
    /// that resource", which mixes a main effect with an interaction.
    pub cov: Vec<Vec<f64>>,
}

impl Ols {
    pub fn t(&self, i: usize) -> f64 {
        if self.se[i] > 0.0 {
            self.coef[i] / self.se[i]
        } else {
            0.0
        }
    }
    /// 95% interval for coefficient `i`.
    pub fn ci(&self, i: usize) -> (f64, f64) {
        (
            self.coef[i] - 1.96 * self.se[i],
            self.coef[i] + 1.96 * self.se[i],
        )
    }
    pub fn by_name(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }
    pub fn coef_of(&self, name: &str) -> f64 {
        self.by_name(name).map(|i| self.coef[i]).unwrap_or(0.0)
    }

    /// Value of a linear combination of coefficients, with its standard
    /// error: `sum(w_i * beta_i)` and `sqrt(w' Cov w)`.
    pub fn combo(&self, terms: &[(&str, f64)]) -> (f64, f64) {
        let idx: Vec<(usize, f64)> = terms
            .iter()
            .filter_map(|(n, w)| self.by_name(n).map(|i| (i, *w)))
            .collect();
        let val: f64 = idx.iter().map(|(i, w)| w * self.coef[*i]).sum();
        let mut var = 0.0;
        for &(i, wi) in &idx {
            for &(j, wj) in &idx {
                var += wi * wj * self.cov[i][j];
            }
        }
        (val, var.max(0.0).sqrt())
    }
    pub fn se_of(&self, name: &str) -> f64 {
        self.by_name(name).map(|i| self.se[i]).unwrap_or(0.0)
    }
}

/// Invert a square matrix by Gauss-Jordan elimination.
fn invert(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a: Vec<Vec<f64>> = m
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.extend((0..n).map(|j| if i == j { 1.0 } else { 0.0 }));
            r
        })
        .collect();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| {
            a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()
        })?;
        if a[piv][col].abs() < 1e-10 {
            return None;
        }
        a.swap(col, piv);
        let d = a[col][col];
        for k in 0..2 * n {
            a[col][k] /= d;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let f = a[row][col];
            if f == 0.0 {
                continue;
            }
            for k in 0..2 * n {
                a[row][k] -= f * a[col][k];
            }
        }
    }
    Some(a.into_iter().map(|r| r[n..].to_vec()).collect())
}

/// Fit `y ~ 1 + x` by least squares. `x` is row-major: one Vec per
/// observation, each of length `names.len()`.
pub fn ols(names: &[String], x: &[Vec<f64>], y: &[f64]) -> Option<Ols> {
    let n = y.len();
    let k = names.len();
    if n <= k + 2 {
        return None;
    }
    // Centre the regressors, so the normal equations stay well conditioned
    // and the intercept drops out of the system.
    let means: Vec<f64> = (0..k)
        .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let ybar = y.iter().sum::<f64>() / n as f64;

    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    for (row, &yi) in x.iter().zip(y.iter()) {
        let c: Vec<f64> = (0..k).map(|j| row[j] - means[j]).collect();
        let yc = yi - ybar;
        for i in 0..k {
            xty[i] += c[i] * yc;
            for j in 0..k {
                xtx[i][j] += c[i] * c[j];
            }
        }
    }
    let inv = invert(&xtx)?;
    let coef: Vec<f64> = (0..k)
        .map(|i| (0..k).map(|j| inv[i][j] * xty[j]).sum())
        .collect();

    let mut sse = 0.0;
    let mut sst = 0.0;
    for (row, &yi) in x.iter().zip(y.iter()) {
        let pred: f64 = ybar
            + (0..k).map(|j| coef[j] * (row[j] - means[j])).sum::<f64>();
        sse += (yi - pred).powi(2);
        sst += (yi - ybar).powi(2);
    }
    let dof = (n - k - 1) as f64;
    let s2 = sse / dof;
    let se: Vec<f64> = (0..k).map(|i| (s2 * inv[i][i]).max(0.0).sqrt()).collect();

    Some(Ols {
        names: names.to_vec(),
        intercept: ybar - (0..k).map(|j| coef[j] * means[j]).sum::<f64>(),
        coef,
        se,
        n,
        r2: if sst > 0.0 { 1.0 - sse / sst } else { 0.0 },
        resid_sd: s2.sqrt(),
        means,
        cov: (0..k)
            .map(|i| (0..k).map(|j| s2 * inv[i][j]).collect())
            .collect(),
    })
}
