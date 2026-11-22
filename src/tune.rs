//! Strategy search by self-play.
//!
//! A (1+lambda) hill climb. Each round, several mutations of the reigning
//! champion are generated and each is tested against the same field.
//!
//! # The champion is re-measured every round
//!
//! It is tempting to compare candidates against a fixed null of `1/n_players`,
//! which is correct in pure self-play by symmetry. Against a *fixed* field of
//! other strategies it is badly wrong: if the champion already beats that
//! field at 30%, a 26% bar promotes every candidate regardless of merit, and
//! the search degenerates into a random walk that selects on noise. So the
//! champion is re-evaluated each round and a candidate must beat the champion
//! itself.
//!
//! # Common random numbers
//!
//! The champion and every candidate in a round are run on the *same seeds*,
//! so they meet the same boards and the same dice. Both sides then share the
//! same luck, and the variance of the difference between them is far smaller
//! than the variance of either one alone -- many more effective comparisons
//! per game played.
//!
//! Testing against copies of the champion is the standard way to make
//! self-play progress, but it can reward a strategy that merely counters the
//! champion rather than one that is strong in general. The final champion is
//! therefore re-tested against a different field from the one it was tuned on.

use crate::analysis;
use crate::bots::{Strategy, Weights, PARAM_KEYS};
use crate::game::GameConfig;
use crate::sim::{self, SimConfig};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct TuneConfig {
    pub rounds: usize,
    /// Mutations tried per round.
    pub candidates: usize,
    pub games_per_eval: usize,
    pub n_players: usize,
    pub seed: u64,
    pub game: GameConfig,
    /// How many standard errors above chance a candidate must score.
    pub accept_sigma: f64,
    /// Opponents to test candidates against. Empty means copies of the
    /// champion (pure self-play). A fixed, varied field instead asks for
    /// general strength rather than for a counter to one opponent, which is
    /// what self-play alone can drift towards.
    pub opponents: Vec<Strategy>,
}

impl Default for TuneConfig {
    fn default() -> Self {
        TuneConfig {
            rounds: 10,
            candidates: 4,
            games_per_eval: 2500,
            n_players: 4,
            seed: 0xC0FFEE,
            game: GameConfig::default(),
            accept_sigma: 1.5,
            opponents: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Trial {
    pub win_rate: f64,
    pub skill_index: f64,
    pub dice_luck: f64,
    pub changes: Vec<(String, f64, f64)>,
}

#[derive(Clone, Debug)]
pub struct RoundLog {
    pub round: usize,
    pub best: Trial,
    /// The champion's win rate on the same seeds, the bar to beat.
    pub champion_win_rate: f64,
    pub margin: f64,
    pub accepted: bool,
    pub games_used: usize,
}

/// Play `cand` against `players-1` copies of `champ` and report how the
/// candidate did. Seats are rotated, so position cannot favour either side.
pub fn evaluate(
    cand: &Weights,
    champ: &Weights,
    games: usize,
    n_players: usize,
    seed: u64,
    game: &GameConfig,
    opponents: &[Strategy],
) -> Trial {
    let mut lineup = vec![Strategy::Heuristic(cand.clone()).relabel("candidate")];
    if opponents.is_empty() {
        for _ in 1..n_players {
            lineup.push(Strategy::Heuristic(champ.clone()).relabel("champion"));
        }
    } else {
        // Cycle the given field to fill the remaining seats.
        for i in 1..n_players {
            lineup.push(opponents[(i - 1) % opponents.len()].clone());
        }
    }
    let mut cfg = SimConfig::with_strategies(lineup, games);
    cfg.base_seed = seed;
    cfg.rotate_seats = true;
    cfg.game = game.clone();
    cfg.game.n_players = n_players;
    let recs = sim::run(&cfg);
    let s = analysis::summarize(&recs);
    let c = s
        .strategies
        .iter()
        .find(|t| t.name == "candidate")
        .expect("candidate must appear");
    Trial {
        win_rate: c.win_rate,
        skill_index: c.skill_index,
        dice_luck: c.mean_dice_luck,
        changes: Vec::new(),
    }
}

/// Perturb one to three parameters multiplicatively.
fn mutate(w: &Weights, rng: &mut impl Rng) -> (Weights, Vec<(String, f64, f64)>) {
    let mut out = w.clone();
    let mut changes = Vec::new();
    let k = rng.random_range(1..=3usize);
    let mut keys: Vec<&str> = PARAM_KEYS.to_vec();
    keys.shuffle(rng);
    for key in keys.into_iter().take(k) {
        let before = w.get(key);
        // Mostly modest nudges, occasionally a big jump to escape a local
        // optimum. `accept_floor` can be negative, so it moves additively.
        let big = rng.random_bool(0.15);
        let sigma = if big { 0.9 } else { 0.3 };
        let after = if key == "accept_floor" {
            before + rng.random_range(-1.5..1.5) * (if big { 2.0 } else { 1.0 })
        } else if before.abs() < 1e-9 {
            // A parameter pinned at zero needs an additive kick to move.
            rng.random_range(0.5..5.0)
        } else {
            before * (rng.random_range(-sigma..sigma) as f64).exp()
        };
        out.set(key, after);
        let actual = out.get(key);
        if (actual - before).abs() > 1e-9 {
            changes.push((key.to_string(), before, actual));
        }
    }
    (out, changes)
}

pub struct TuneResult {
    pub champion: Weights,
    pub log: Vec<RoundLog>,
    pub games_used: usize,
}

pub fn run(
    cfg: &TuneConfig,
    base: Weights,
    mut on_round: impl FnMut(&RoundLog, &Weights),
) -> TuneResult {
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut champion = base;
    champion.label = "champion".into();
    let mut log = Vec::new();
    let mut games_used = 0usize;

    for round in 1..=cfg.rounds {
        // One seed for the whole round: champion and candidates face the same
        // boards and dice, so the comparison between them is far less noisy.
        let seed = cfg.seed.wrapping_add((round as u64) << 20);

        let champ_trial = evaluate(
            &champion,
            &champion,
            cfg.games_per_eval,
            cfg.n_players,
            seed,
            &cfg.game,
            &cfg.opponents,
        );
        games_used += cfg.games_per_eval;

        let mut best: Option<(Trial, Weights)> = None;
        for _ in 0..cfg.candidates {
            let (cand, changes) = mutate(&champion, &mut rng);
            let mut t = evaluate(
                &cand,
                &champion,
                cfg.games_per_eval,
                cfg.n_players,
                seed,
                &cfg.game,
                &cfg.opponents,
            );
            t.changes = changes;
            games_used += cfg.games_per_eval;
            if best
                .as_ref()
                .map(|(b, _)| t.win_rate > b.win_rate)
                .unwrap_or(true)
            {
                best = Some((t, cand));
            }
        }
        let (trial, cand) = best.expect("at least one candidate");

        // Conservative standard error for the difference of two win rates,
        // ignoring the positive correlation the shared seeds induce.
        let p = ((champ_trial.win_rate + trial.win_rate) / 2.0).clamp(0.01, 0.99);
        let se_diff = (2.0 * p * (1.0 - p) / cfg.games_per_eval as f64).sqrt();
        let margin = trial.win_rate - champ_trial.win_rate;
        let accepted = margin > cfg.accept_sigma * se_diff;
        if accepted {
            champion = cand;
            champion.label = "champion".into();
        }
        let entry = RoundLog {
            round,
            best: trial,
            champion_win_rate: champ_trial.win_rate,
            margin,
            accepted,
            games_used,
        };
        on_round(&entry, &champion);
        log.push(entry);
    }

    TuneResult {
        champion,
        log,
        games_used,
    }
}

/// Standard error of the champion-versus-candidate difference, for reporting.
pub fn diff_se(cfg: &TuneConfig) -> f64 {
    let p0 = 1.0 / cfg.n_players as f64;
    (2.0 * p0 * (1.0 - p0) / cfg.games_per_eval as f64).sqrt()
}
