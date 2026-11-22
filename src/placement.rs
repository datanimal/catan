//! Valuing an opening position, in pips.
//!
//! # Why the openings are randomised
//!
//! In a normal game a bot *chooses* where to start, so its opening position is
//! entangled with its strategy, its preferences and the board it was dealt.
//! Regressing outcomes on chosen placements would measure the bots' taste, not
//! the value of the ground. So in this experiment every player places their
//! opening settlements **uniformly at random** among the legal intersections
//! and then plays the same strategy. Opening position becomes exogenous, and
//! the regression coefficients can be read as the value of the position.
//!
//! The price is extrapolation: random openings include spots no one would ever
//! take, so the fit covers a wider range than real play. The report prints the
//! distribution of total pips so that range is visible, and the per-bucket
//! means let the linear fit be checked against the raw data.
//!
//! # The pip as a unit of account
//!
//! One pip is one of the 36 dice combinations, i.e. a 1/36 chance per roll of
//! one card. `lambda` is the fitted victory-point value of one pip of the
//! average resource. Every other quantity -- a specific resource, a port, a
//! shared number -- is then reported as **adjusted pips**: its value divided
//! by `lambda`. So "a 2:1 ore port is worth 3.4 adjusted pips" means it is
//! worth as much as adding 3.4 pips of average production.

use crate::analysis::{self, Ols};
use crate::bots::Strategy;
use crate::game::GameConfig;
use crate::metrics::GameRecord;
use crate::sim::{self, SimConfig};
use crate::types::*;

#[derive(Clone, Debug)]
pub struct PlacementConfig {
    pub games: usize,
    /// The strategy every player uses after the opening.
    pub strategy: Strategy,
    pub n_players: usize,
    pub seed: u64,
    pub game: GameConfig,
}

/// Feature names for the structural model, in order.
fn feature_names() -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for r in Resource::ALL {
        v.push(format!("pips_{}", r.name()));
    }
    v.push("paired_road".into());
    v.push("paired_city".into());
    v.push("port3".into());
    for r in Resource::ALL {
        v.push(format!("port2_{}", r.name()));
    }
    for r in Resource::ALL {
        v.push(format!("port2x_{}", r.name()));
    }
    v.push("port3_x_pips".into());
    v.push("port_at_1".into());
    v.push("port_at_2".into());
    v
}

fn features(f: &crate::metrics::OpeningFeatures) -> Vec<f64> {
    let mut v: Vec<f64> = Vec::new();
    for r in Resource::ALL {
        v.push(f.pips[r.idx()] as f64);
    }
    v.push(f.paired_road as f64);
    v.push(f.paired_city as f64);
    v.push(f.port3 as u8 as f64);
    for r in Resource::ALL {
        v.push(f.port2[r.idx()] as u8 as f64);
    }
    // Interactions: a 2:1 port is only worth something if you produce the
    // resource it trades.
    for r in Resource::ALL {
        v.push(f.port2[r.idx()] as u8 as f64 * f.pips[r.idx()] as f64);
    }
    v.push(f.port3 as u8 as f64 * f.total_pips as f64);
    v.push((f.port_dist == 1) as u8 as f64);
    v.push((f.port_dist == 2) as u8 as f64);
    v
}

#[derive(Clone, Debug)]
pub struct PipBucket {
    pub lo: u32,
    pub hi: u32,
    pub n: usize,
    pub mean_adj_vp: f64,
    pub win_rate: f64,
}

pub struct PlacementReport {
    pub n: usize,
    pub games: usize,
    /// Openings below this pip count were excluded from the headline fit.
    pub min_pips: u32,
    /// Full-range fit, for comparison with the restricted headline fit.
    pub wide: Option<Box<PlacementReport>>,
    /// adj_vp ~ total_pips. Its slope is `lambda`.
    pub pip_model: Ols,
    /// Marginal victory points per pip of average production.
    pub lambda: f64,
    /// adj_vp ~ total_pips + total_pips^2, to check for curvature.
    pub curve_model: Ols,
    /// The structural model: resources, pairing, ports, proximity.
    pub full: Ols,
    /// The same with a diversity term added.
    pub with_breadth: Ols,
    /// Win probability rather than victory points, as a cross-check.
    pub win_model: Ols,
    pub buckets: Vec<PipBucket>,
    /// Chance of ending the game with a port, by opening distance to one.
    pub port_reach: Vec<(u8, usize, f64)>,
    pub mean_total_pips: f64,
    pub mean_pips_by_res: [f64; N_RES],
}

/// Openings weaker than this are excluded from the headline fit. Uniformly
/// random placement produces many positions no player would ever take (the
/// mean is around 13 pips against roughly 19-20 for chosen openings), and the
/// value of a pip falls as pips accumulate. Fitting over the whole range would
/// calibrate the pip unit on positions nobody plays, which would then misprice
/// everything quoted in those units.
pub const REALISTIC_MIN_PIPS: u32 = 14;

/// Write one row per opening position for independent analysis.
pub fn write_rows(path: &str, recs: &[GameRecord]) -> std::io::Result<usize> {
    use std::io::Write;
    let model = analysis::fit_luck_model(recs);
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    let mut n = 0;
    for g in recs {
        for p in &g.players {
            let o = &p.metrics.opening;
            if o.numbers.is_empty() {
                continue;
            }
            let row = serde_json::json!({
                "won": p.won, "vp": p.vp, "adj_vp": model.skill_of(p),
                "dice_luck": p.metrics.dice_luck(),
                "total_pips": o.total_pips, "breadth": o.breadth,
                "pips": o.pips, "paired_road": o.paired_road,
                "paired_city": o.paired_city, "port2": o.port2, "port3": o.port3,
                "port_dist": o.port_dist, "numbers": o.numbers,
                "final_port_mask": p.metrics.final_port_mask,
            });
            writeln!(w, "{row}")?;
            n += 1;
        }
    }
    Ok(n)
}

pub fn run_records(cfg: &PlacementConfig) -> Vec<GameRecord> {
    let lineup: Vec<Strategy> = (0..cfg.n_players)
        .map(|_| cfg.strategy.clone().with_random_opening().relabel("subject"))
        .collect();
    let mut sc = SimConfig::with_strategies(lineup, cfg.games);
    sc.base_seed = cfg.seed;
    sc.rotate_seats = false; // every seat plays the same strategy anyway
    sc.game = cfg.game.clone();
    sc.game.n_players = cfg.n_players;
    sim::run(&sc)
}

pub fn run(cfg: &PlacementConfig) -> PlacementReport {
    let recs = run_records(cfg);
    let mut rep = analyse(&recs, REALISTIC_MIN_PIPS);
    rep.wide = Some(Box::new(analyse(&recs, 0)));
    rep
}

pub fn analyse(recs: &[GameRecord], min_pips: u32) -> PlacementReport {
    // Luck-adjusted victory points: the dice effect is removed so that what
    // is left reflects the position and the play, not the rolls.
    let model = analysis::fit_luck_model(recs);
    let names = feature_names();

    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<f64> = Vec::new();
    let mut ywin: Vec<f64> = Vec::new();
    let mut xpip: Vec<Vec<f64>> = Vec::new();
    let mut xcurve: Vec<Vec<f64>> = Vec::new();
    let mut xbreadth: Vec<Vec<f64>> = Vec::new();
    let mut pip_tot = 0.0;
    let mut pip_res = [0.0f64; N_RES];

    for g in recs {
        for p in &g.players {
            let f = &p.metrics.opening;
            // Skip anything degenerate (a player who never placed).
            if f.numbers.is_empty() || f.total_pips < min_pips {
                continue;
            }
            let adj = model.skill_of(p);
            let feats = features(f);
            let tp = f.total_pips as f64;
            pip_tot += tp;
            for i in 0..N_RES {
                pip_res[i] += f.pips[i] as f64;
            }
            let mut wb = feats.clone();
            wb.push(f.breadth as f64);
            x.push(feats);
            xbreadth.push(wb);
            xpip.push(vec![tp]);
            xcurve.push(vec![tp, tp * tp]);
            y.push(adj);
            ywin.push(p.won as u8 as f64);
        }
    }
    let n = y.len();
    let nf = n as f64;

    let pip_model = analysis::ols(&["total_pips".to_string()], &xpip, &y).expect("pip fit");
    let curve_model = analysis::ols(
        &["total_pips".to_string(), "total_pips_sq".to_string()],
        &xcurve,
        &y,
    )
    .expect("curve fit");
    let full = analysis::ols(&names, &x, &y).expect("structural fit");
    let mut bn = names.clone();
    bn.push("breadth".into());
    let with_breadth = analysis::ols(&bn, &xbreadth, &y).expect("breadth fit");
    let win_model = analysis::ols(&names, &x, &ywin).expect("win fit");

    // Raw means per pip bucket, so the linear fit can be eyeballed.
    let edges: Vec<(u32, u32)> = vec![
        (0, 8),
        (8, 11),
        (11, 14),
        (14, 17),
        (17, 20),
        (20, 23),
        (23, 100),
    ];
    let mut buckets = Vec::new();
    for (lo, hi) in edges {
        let sel: Vec<usize> = recs
            .iter()
            .flat_map(|g| g.players.iter())
            .enumerate()
            .filter(|(_, p)| {
                let t = p.metrics.opening.total_pips;
                t >= lo && t < hi && !p.metrics.opening.numbers.is_empty()
            })
            .map(|(i, _)| i)
            .collect();
        let rows: Vec<&crate::metrics::PlayerResult> = recs
            .iter()
            .flat_map(|g| g.players.iter())
            .filter(|p| {
                let t = p.metrics.opening.total_pips;
                t >= lo && t < hi && !p.metrics.opening.numbers.is_empty()
            })
            .collect();
        let _ = sel;
        if rows.is_empty() {
            continue;
        }
        buckets.push(PipBucket {
            lo,
            hi,
            n: rows.len(),
            mean_adj_vp: rows.iter().map(|p| model.skill_of(p)).sum::<f64>() / rows.len() as f64,
            win_rate: rows.iter().filter(|p| p.won).count() as f64 / rows.len() as f64,
        });
    }

    // How often does an opening that is d roads from a port end up owning one?
    let mut port_reach = Vec::new();
    for d in 0..=4u8 {
        let rows: Vec<&crate::metrics::PlayerResult> = recs
            .iter()
            .flat_map(|g| g.players.iter())
            .filter(|p| {
                p.metrics.opening.port_dist == d
                    && p.metrics.opening.total_pips >= min_pips
                    && !p.metrics.opening.numbers.is_empty()
            })
            .collect();
        if rows.len() < 30 {
            continue;
        }
        let got = rows
            .iter()
            .filter(|p| p.metrics.final_port_mask != 0)
            .count();
        port_reach.push((d, rows.len(), got as f64 / rows.len() as f64));
    }

    PlacementReport {
        n,
        games: recs.len(),
        min_pips,
        wide: None,
        lambda: pip_model.coef[0],
        pip_model,
        curve_model,
        full,
        with_breadth,
        win_model,
        buckets,
        port_reach,
        mean_total_pips: pip_tot / nf,
        mean_pips_by_res: {
            let mut m = [0.0; N_RES];
            for i in 0..N_RES {
                m[i] = pip_res[i] / nf;
            }
            m
        },
    }
}

impl PlacementReport {
    /// A coefficient expressed in adjusted pips, with its 95% interval.
    pub fn in_pips(&self, name: &str) -> (f64, f64, f64) {
        let c = self.full.coef_of(name);
        let s = self.full.se_of(name);
        (
            c / self.lambda,
            (c - 1.96 * s) / self.lambda,
            (c + 1.96 * s) / self.lambda,
        )
    }
}

// ---------------------------------------------------------------- report ----

impl PlacementReport {
    pub fn print(&self) {
        let lam = self.lambda;
        println!("=== opening placement valuation ===");
        println!(
            "{} games, {} opening positions used, all placed uniformly at random",
            self.games, self.n
        );
        if self.min_pips > 0 {
            println!(
                "headline fit restricted to openings of {}+ pips, the range real play \
                 occupies",
                self.min_pips
            );
        }
        println!(
            "mean opening: {:.1} total pips  (brick {:.1}, lumber {:.1}, wool {:.1}, grain {:.1}, ore {:.1})",
            self.mean_total_pips,
            self.mean_pips_by_res[0],
            self.mean_pips_by_res[1],
            self.mean_pips_by_res[2],
            self.mean_pips_by_res[3],
            self.mean_pips_by_res[4],
        );

        println!("\n--- the unit of account ---");
        println!(
            "one pip of average production = {:.4} VP  (+/- {:.4})",
            lam,
            1.96 * self.pip_model.se[0]
        );
        let c1 = self.curve_model.coef[0];
        let c2 = self.curve_model.coef[1];
        println!(
            "curvature check: VP = {:.4}*pips {:+.5}*pips^2   (t on the square term {:+.1})",
            c1,
            c2,
            self.curve_model.t(1)
        );
        if self.curve_model.t(1).abs() > 3.0 {
            let at = self.mean_total_pips;
            println!(
                "  the square term matters: marginal value is {:.4} VP/pip at {:.0} pips,",
                c1 + 2.0 * c2 * at, at
            );
            println!(
                "  falling to {:.4} at {:.0} pips -- so adjusted pips below are averages",
                c1 + 2.0 * c2 * (at + 6.0),
                at + 6.0
            );
        }
        println!("\n  raw means by opening strength (checks the straight line):");
        println!("    {:<14}{:>9}{:>12}{:>9}", "total pips", "n", "adj VP", "win%");
        for b in &self.buckets {
            let label = if b.hi >= 100 {
                format!("{}+", b.lo)
            } else {
                format!("{}-{}", b.lo, b.hi - 1)
            };
            println!(
                "    {:<14}{:>9}{:>12.3}{:>8.1}%",
                label,
                b.n,
                b.mean_adj_vp,
                100.0 * b.win_rate
            );
        }

        println!("\n--- value of a pip by resource (adjusted pips, average = 1.00) ---");
        println!(
            "{:<10}{:>10}{:>12}{:>16}{:>8}",
            "resource", "VP/pip", "adj pips", "95% CI", "t"
        );
        for r in Resource::ALL {
            let name = format!("pips_{}", r.name());
            let i = self.full.by_name(&name).unwrap();
            let (v, lo, hi) = self.in_pips(&name);
            println!(
                "{:<10}{:>10.4}{:>12.2}{:>8.2}-{:<7.2}{:>8.1}",
                r.name(),
                self.full.coef[i],
                v,
                lo,
                hi,
                self.full.t(i)
            );
        }

        println!("\n--- shared numbers (income that arrives together) ---");
        println!(
            "{:<28}{:>12}{:>16}{:>8}",
            "paired pips", "adj pips", "95% CI", "t"
        );
        for (key, label) in [
            ("paired_road", "brick+lumber same number"),
            ("paired_city", "grain+ore same number"),
        ] {
            let i = self.full.by_name(key).unwrap();
            let (v, lo, hi) = self.in_pips(key);
            println!(
                "{:<28}{:>+12.2}{:>+8.2}-{:<+7.2}{:>8.1}",
                label, v, lo, hi, self.full.t(i)
            );
        }
        println!("  read as: value *on top of* the pips themselves, per paired pip");

        println!("\n--- ports on an opening settlement (adjusted pips) ---");
        println!(
            "{:<16}{:>15}{:>15}{:>15}{:>15}",
            "port", "at 0 pips", "at 3 pips", "at 6 pips", "at 9 pips"
        );
        for r in Resource::ALL {
            let main = format!("port2_{}", r.name());
            let inter = format!("port2x_{}", r.name());
            let mut cells = Vec::new();
            for p in [0.0, 3.0, 6.0, 9.0] {
                let (v, se) = self.full.combo(&[(&main, 1.0), (&inter, p)]);
                cells.push(format!("{:+.2} +/-{:.2}", v / lam, 1.96 * se / lam));
            }
            println!(
                "{:<16}{:>15}{:>15}{:>15}{:>15}",
                format!("2:1 {}", r.name()),
                cells[0],
                cells[1],
                cells[2],
                cells[3]
            );
        }
        println!("  columns: pips of *that* resource the opening produces");
        let mut cells = Vec::new();
        for tp in [14.0, 18.0, 22.0, 26.0] {
            let (v, se) = self.full.combo(&[("port3", 1.0), ("port3_x_pips", tp)]);
            cells.push(format!("{:+.2} +/-{:.2}", v / lam, 1.96 * se / lam));
        }
        println!(
            "\n{:<16}{:>15}{:>15}{:>15}{:>15}",
            "3:1 any", cells[0], cells[1], cells[2], cells[3]
        );
        println!("  columns: TOTAL pips of the opening (14, 18, 22, 26)");

        println!("\n--- proximity to a port, if not on one (baseline: 3+ roads away) ---");
        for (key, label) in [("port_at_1", "1 road away"), ("port_at_2", "2 roads away")] {
            let i = self.full.by_name(key).unwrap();
            let (v, lo, hi) = self.in_pips(key);
            println!(
                "  {:<14}{:>+8.2} adj pips  ({:+.2} to {:+.2}, t={:+.1})",
                label, v, lo, hi, self.full.t(i)
            );
        }
        println!("\n  chance of ending the game owning a port:");
        for (d, n, rate) in &self.port_reach {
            println!(
                "    opening {} road(s) from a port: {:>5.1}%   (n={})",
                d,
                100.0 * rate,
                n
            );
        }

        println!("\n--- resource diversity ---");
        let i = self.with_breadth.by_name("breadth").unwrap();
        let c = self.with_breadth.coef[i];
        let se = self.with_breadth.se[i];
        println!(
            "  each extra distinct resource: {:+.3} VP = {:+.2} adj pips ({:+.2} to {:+.2}, t={:+.1})",
            c,
            c / lam,
            (c - 1.96 * se) / lam,
            (c + 1.96 * se) / lam,
            self.with_breadth.t(i)
        );
        println!("  (holding the pip counts of every resource fixed)");

        println!("\n--- cross-check: same model against win probability ---");
        let wl = self.win_model.coef_of("pips_grain").abs().max(1e-9);
        let _ = wl;
        println!(
            "{:<16}{:>14}{:>14}",
            "feature", "VP model", "win model"
        );
        let win_per_pip: f64 = Resource::ALL
            .iter()
            .map(|r| self.win_model.coef_of(&format!("pips_{}", r.name())))
            .sum::<f64>()
            / N_RES as f64;
        for r in Resource::ALL {
            let n = format!("pips_{}", r.name());
            println!(
                "{:<16}{:>13.2}{:>14.2}",
                format!("pips_{}", r.name()),
                self.full.coef_of(&n) / lam,
                self.win_model.coef_of(&n) / win_per_pip
            );
        }
        println!(
            "  both columns are relative to an average pip; close agreement means the"
        );
        println!("  ranking is not an artefact of scoring victory points rather than wins");
        println!(
            "\nmodel fit: R^2 {:.3} on luck-adjusted VP, residual SD {:.2} VP, n={}",
            self.full.r2, self.full.resid_sd, self.full.n
        );
        if let Some(w) = &self.wide {
            println!(
                "\n--- robustness: the same model over the full random range ---"
            );
            println!(
                "  pip unit there is {:.4} VP (vs {:.4} here), n={}",
                w.lambda, self.lambda, w.n
            );
            println!("  {:<10}{:>14}{:>14}", "resource", "restricted", "full range");
            for r in Resource::ALL {
                let n = format!("pips_{}", r.name());
                println!(
                    "  {:<10}{:>14.2}{:>14.2}",
                    r.name(),
                    self.full.coef_of(&n) / self.lambda,
                    w.full.coef_of(&n) / w.lambda
                );
            }
            println!(
                "  agreement between the two means the ranking does not depend on which"
            );
            println!("  openings are included");
        }
    }
}
