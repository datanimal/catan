//! Tests of the measurement apparatus. A rules engine can be correct while the
//! metrics built on it are biased, which would silently distort every
//! experiment, so the yardsticks get their own tests.

use catan_sim::analysis;
use catan_sim::sim::{self, SimConfig};
use catan_sim::types::prob;

fn run(lineup: &[&str], games: usize, seed: u64) -> Vec<catan_sim::metrics::GameRecord> {
    let mut cfg = SimConfig::new(lineup.iter().map(|s| s.to_string()).collect(), games);
    cfg.base_seed = seed;
    sim::run(&cfg)
}

#[test]
fn dice_luck_is_unbiased() {
    // dice_luck is actual production minus what average dice would have paid
    // the same board position over the same rolls. Across a large field it
    // must average to zero, or every luck-adjusted number is skewed.
    let recs = run(&["balanced", "city_builder", "dev_rush", "expander"], 1500, 11);
    let rows: Vec<f64> = recs
        .iter()
        .flat_map(|g| g.players.iter())
        .map(|p| p.metrics.dice_luck())
        .collect();
    let n = rows.len() as f64;
    let mean = rows.iter().sum::<f64>() / n;
    let sd = (rows.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    let se = sd / n.sqrt();
    println!("dice_luck: mean {mean:+.4} cards, sd {sd:.2}, se {se:.4}, n {n}");
    // Within four standard errors of zero.
    assert!(
        mean.abs() < 4.0 * se,
        "dice_luck is biased: mean {mean:+.4} against se {se:.4}"
    );
    // And it must actually vary -- a metric stuck at zero would also pass the
    // test above while measuring nothing.
    assert!(sd > 5.0, "dice_luck has implausibly little spread: {sd}");
}

#[test]
fn production_and_expectation_agree_in_aggregate() {
    let recs = run(&["balanced", "trader", "dev_rush", "expander"], 800, 23);
    let mut produced = 0i64;
    let mut expected = 0.0f64;
    let mut shortage = 0i64;
    for g in &recs {
        for p in &g.players {
            produced += p.metrics.produced_total() as i64;
            expected += p.metrics.expected_total();
            shortage += p.metrics.bank_shortage_loss as i64;
        }
    }
    let ratio = (produced + shortage) as f64 / expected;
    println!(
        "produced {produced} (+{shortage} unpayable) vs expected {expected:.0} -> ratio {ratio:.4}"
    );
    assert!(
        (ratio - 1.0).abs() < 0.01,
        "production deviates from expectation by {:.2}%",
        100.0 * (ratio - 1.0)
    );
}

#[test]
fn dice_are_fair() {
    let recs = run(&["balanced", "balanced", "balanced", "balanced"], 600, 5);
    let mut hist = [0u32; 13];
    for g in &recs {
        for (i, c) in g.roll_histogram.iter().enumerate() {
            hist[i] += c;
        }
    }
    let total: u32 = hist.iter().sum();
    // Chi-squared against the 2d6 distribution, 10 degrees of freedom.
    // The 99.9% critical value is 29.6.
    let chi2: f64 = (2..=12u8)
        .map(|r| {
            let exp = prob(r) * total as f64;
            let obs = hist[r as usize] as f64;
            (obs - exp).powi(2) / exp
        })
        .sum();
    println!("chi-squared over {total} rolls = {chi2:.2} (10 dof, reject above 29.6)");
    assert!(chi2 < 29.6, "dice do not look like 2d6: chi2 = {chi2:.2}");
}

#[test]
fn deck_dice_have_the_same_mean_but_less_variance() {
    use catan_sim::game::DiceMode;
    let mut a = SimConfig::new(vec!["balanced".into(); 4], 400);
    a.base_seed = 99;
    let mut b = a.clone();
    b.game.dice = DiceMode::Deck36;

    let spread = |recs: &[catan_sim::metrics::GameRecord]| {
        let v: Vec<f64> = recs
            .iter()
            .flat_map(|g| g.players.iter())
            .map(|p| p.metrics.dice_luck())
            .collect();
        let n = v.len() as f64;
        let m = v.iter().sum::<f64>() / n;
        let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        (m, sd)
    };
    let (m1, sd1) = spread(&sim::run(&a));
    let (m2, sd2) = spread(&sim::run(&b));
    println!("2d6: mean {m1:+.3} sd {sd1:.2} | deck36: mean {m2:+.3} sd {sd2:.2}");
    // The deck removes streaks, so luck must spread less.
    assert!(
        sd2 < sd1,
        "the dice deck should reduce luck variance: {sd2:.2} vs {sd1:.2}"
    );
}

#[test]
fn seat_rotation_balances_the_lineup() {
    // With rotation on and a game count that is a multiple of the player
    // count, every strategy must play every seat equally often -- otherwise a
    // seat effect would be read as a strategy effect.
    let lineup = ["balanced", "city_builder", "dev_rush", "expander"];
    let recs = run(&lineup, 400, 7);
    for name in lineup {
        let mut per_seat = [0usize; 4];
        for g in &recs {
            for p in &g.players {
                if p.policy == name {
                    per_seat[p.seat as usize] += 1;
                }
            }
        }
        println!("{name}: {per_seat:?}");
        assert!(
            per_seat.iter().all(|&c| c == 100),
            "{name} did not play every seat equally: {per_seat:?}"
        );
    }
}

#[test]
fn win_rates_sum_to_one_and_the_model_fits() {
    let recs = run(&["balanced", "city_builder", "dev_rush", "expander"], 1200, 31);
    let s = analysis::summarize(&recs);
    let total: f64 = s.strategies.iter().map(|t| t.win_rate).sum();
    println!("win rates sum to {total:.4}");
    assert!((total - 1.0).abs() < 1e-9, "win rates sum to {total}");
    assert_eq!(s.timeouts, 0);
    // An extra resource card has to be worth something, but not much.
    println!(
        "beta_dice = {:.4} VP/card, R^2 = {:.3}",
        s.model.beta_dice, s.model.r2
    );
    assert!(
        s.model.beta_dice > 0.01 && s.model.beta_dice < 0.3,
        "implausible value for a resource card: {}",
        s.model.beta_dice
    );
    // The skill index is centred on the field, so its average across all
    // strategies must come back to the mean victory point total.
    let field_vp: f64 = recs
        .iter()
        .flat_map(|g| g.players.iter())
        .map(|p| p.vp as f64)
        .sum::<f64>()
        / (recs.len() * 4) as f64;
    let field_skill: f64 =
        s.strategies.iter().map(|t| t.skill_index * t.games as f64).sum::<f64>()
            / (recs.len() * 4) as f64;
    println!("mean VP {field_vp:.4} vs mean skill index {field_skill:.4}");
    assert!((field_vp - field_skill).abs() < 1e-6);
}

#[test]
fn a_better_strategy_beats_a_random_one() {
    // A sanity floor: the heuristic bots must be clearly better than noise.
    // If this ever fails, the scoring engine has stopped working.
    let recs = run(&["balanced", "random", "random", "random"], 500, 3);
    let s = analysis::summarize(&recs);
    let bal = s.strategies.iter().find(|t| t.name == "balanced").unwrap();
    let rnd = s.strategies.iter().find(|t| t.name == "random").unwrap();
    println!(
        "balanced {:.1}% vs random {:.1}%",
        100.0 * bal.win_rate,
        100.0 * rnd.win_rate
    );
    assert!(
        bal.win_rate > 0.6,
        "balanced only won {:.1}% against three random bots",
        100.0 * bal.win_rate
    );
}
