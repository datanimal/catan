//! Parallel batch runner.
//!
//! Seat position matters in Catan -- placing first and last in the snake order
//! are not equivalent -- so by default the lineup is rotated between games.
//! Over a run that is a multiple of the player count, every strategy plays
//! every seat an equal number of times, and a seat effect can no longer
//! masquerade as a strategy effect.

use crate::bots;
use crate::game::{Game, GameConfig};
use crate::metrics::GameRecord;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub games: usize,
    /// Strategy names, one per seat.
    pub lineup: Vec<String>,
    /// Rotate the lineup by one seat each game to cancel seat advantage.
    pub rotate_seats: bool,
    pub base_seed: u64,
    pub game: GameConfig,
}

impl SimConfig {
    pub fn new(lineup: Vec<String>, games: usize) -> SimConfig {
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
}

/// Run one game. `idx` selects the seed and the seat rotation.
pub fn run_one(cfg: &SimConfig, idx: usize) -> GameRecord {
    let n = cfg.lineup.len();
    let shift = if cfg.rotate_seats { idx % n } else { 0 };
    let seed = cfg.base_seed.wrapping_add(idx as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let policies = (0..n)
        .map(|seat| {
            let name = &cfg.lineup[(seat + shift) % n];
            bots::make(name, seed ^ ((seat as u64 + 1) << 32))
                .unwrap_or_else(|| panic!("unknown strategy: {name}"))
        })
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
    let known = bots::available();
    for name in lineup {
        if !known.contains(&name.as_str()) {
            return Err(format!(
                "unknown strategy '{name}'. Available: {}",
                known.join(", ")
            ));
        }
    }
    if lineup.len() < 2 || lineup.len() > 6 {
        return Err(format!("need 2..=6 players, got {}", lineup.len()));
    }
    Ok(())
}
