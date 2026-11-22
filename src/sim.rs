//! Parallel batch runner.
//!
//! Seat position matters in Catan -- placing first and last in the snake order
//! are not equivalent -- so by default the lineup is rotated between games.
//! Over a run that is a multiple of the player count, every strategy plays
//! every seat an equal number of times, and a seat effect can no longer
//! masquerade as a strategy effect.

use crate::bots::{self, Strategy};
use crate::game::{Game, GameConfig};
use crate::metrics::GameRecord;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub games: usize,
    /// One strategy per seat.
    pub lineup: Vec<Strategy>,
    /// Rotate the lineup by one seat each game to cancel seat advantage.
    pub rotate_seats: bool,
    pub base_seed: u64,
    pub game: GameConfig,
}

impl SimConfig {
    /// Build from strategy specs, panicking on an unknown name. Call
    /// [`check_lineup`] first to report typos gracefully.
    pub fn new(lineup: Vec<String>, games: usize) -> SimConfig {
        let parsed = lineup
            .iter()
            .map(|s| Strategy::parse(s).unwrap_or_else(|| panic!("unknown strategy: {s}")))
            .collect();
        SimConfig::with_strategies(parsed, games)
    }

    pub fn with_strategies(lineup: Vec<Strategy>, games: usize) -> SimConfig {
        let mut game = GameConfig::default();
        game.n_players = lineup.len();
        SimConfig {
            games,
            lineup,
            rotate_seats: true,
            base_seed: 0x5EED,
            game,
        }
    }

    pub fn labels(&self) -> Vec<String> {
        self.lineup.iter().map(|s| s.label()).collect()
    }
}

/// Run one game. `idx` selects the seed and the seat rotation.
pub fn run_one(cfg: &SimConfig, idx: usize) -> GameRecord {
    let n = cfg.lineup.len();
    let shift = if cfg.rotate_seats { idx % n } else { 0 };
    let seed = cfg.base_seed.wrapping_add(idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let policies = (0..n)
        .map(|seat| cfg.lineup[(seat + shift) % n].make(seed ^ ((seat as u64 + 1) << 32)))
        .collect();
    let mut g = Game::new(cfg.game.clone(), policies, seed);
    g.run()
}

/// Run the whole batch across all available cores.
pub fn run(cfg: &SimConfig) -> Vec<GameRecord> {
    (0..cfg.games)
        .into_par_iter()
        .map(|i| run_one(cfg, i))
        .collect()
}

/// Validate a lineup before running, so typos fail fast.
pub fn check_lineup(lineup: &[String]) -> Result<(), String> {
    for name in lineup {
        if Strategy::parse(name).is_none() {
            return Err(format!(
                "unknown strategy '{name}'. Available: {}. You can also \
                 override weights, e.g. 'balanced+w_dev=40,port=9'.",
                bots::available().join(", ")
            ));
        }
    }
    if lineup.len() < 2 || lineup.len() > 6 {
        return Err(format!("need 2..=6 players, got {}", lineup.len()));
    }
    Ok(())
}
