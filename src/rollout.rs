//! Monte Carlo evaluation of a position: play it out many times and count.
//!
//! This is the model-free way to value a position, and it needs no training
//! data. The cost is time, and one subtlety worth being explicit about.
//!
//! # Unknown hands
//!
//! A spectator does not know what anyone holds. Rolling out from a single
//! guessed hand would give a confident answer to the wrong question. With
//! `resample_hands` the hidden cards are redrawn from the bank before each
//! rollout, keeping each player's hand *size* — which is public — so the
//! result averages over the hands the position is consistent with instead of
//! betting on one of them.
//!
//! The probabilities are conditional on the given strategies finishing the
//! game, exactly as a chess evaluation is conditional on the engine's play.

use crate::bots::Strategy;
use crate::game::{Game, GameConfig};
use crate::state::GameState;
use crate::types::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct RolloutResult {
    /// Win probability per seat; sums to one plus the draw share.
    pub win_prob: Vec<f64>,
    /// Games that hit the turn limit without a winner.
    pub unresolved: usize,
    pub rollouts: usize,
    /// Standard error on each probability, from the binomial.
    pub se: Vec<f64>,
}

/// Redraw every player's hidden cards from the bank, preserving hand sizes.
fn resample(st: &mut GameState, rng: &mut impl Rng) {
    let n = st.n_players();
    let sizes: Vec<i16> = (0..n).map(|p| st.players[p].hand_size()).collect();
    // Return everything to the bank, then deal it back out at random.
    for p in 0..n {
        for i in 0..N_RES {
            st.bank[i] += st.players[p].hand[i];
            st.players[p].hand[i] = 0;
        }
    }
    for p in 0..n {
        for _ in 0..sizes[p] {
            let total: i16 = st.bank.iter().sum();
            if total <= 0 {
                break;
            }
            let mut k = rng.random_range(0..total);
            for i in 0..N_RES {
                if k < st.bank[i] {
                    st.bank[i] -= 1;
                    st.players[p].hand[i] += 1;
                    break;
                }
                k -= st.bank[i];
            }
        }
    }
}

/// Play `rollouts` games from `st` and report how often each seat wins.
pub fn evaluate(
    st: &GameState,
    lineup: &[Strategy],
    rollouts: usize,
    seed: u64,
    resample_hands: bool,
) -> RolloutResult {
    let n = st.n_players();
    assert_eq!(lineup.len(), n, "one strategy per seat");
    let mut cfg = GameConfig::default();
    cfg.n_players = n;
    cfg.vp_target = st.vp_target;
    cfg.hand_limit = st.hand_limit;
    // A position can be resumed late, so allow generous headroom past the
    // turn it starts on rather than a fixed count.
    cfg.turn_limit = st.turn + 400;

    let tally: Vec<(Vec<u32>, usize)> = (0..rollouts)
        .into_par_iter()
        .map(|i| {
            let s = seed
                .wrapping_add(i as u64)
                .wrapping_mul(0x9E3779B97F4A7C15);
            let mut rng = ChaCha8Rng::seed_from_u64(s);
            let mut local = st.clone();
            local.turn_limit = cfg.turn_limit;
            if resample_hands {
                resample(&mut local, &mut rng);
            }
            let policies = (0..n)
                .map(|seat| lineup[seat].make(s ^ ((seat as u64 + 1) << 32)))
                .collect();
            let rec = Game::from_state(local, cfg.clone(), policies, s).run();
            let mut wins = vec![0u32; n];
            let mut unresolved = 0;
            match rec.winner {
                Some(w) => wins[w as usize] += 1,
                None => unresolved = 1,
            }
            (wins, unresolved)
        })
        .collect();

    let mut wins = vec![0u32; n];
    let mut unresolved = 0usize;
    for (w, u) in tally {
        for i in 0..n {
            wins[i] += w[i];
        }
        unresolved += u;
    }
    let denom = rollouts as f64;
    let win_prob: Vec<f64> = wins.iter().map(|&w| w as f64 / denom).collect();
    let se = win_prob
        .iter()
        .map(|p| (p * (1.0 - p) / denom).sqrt())
        .collect();
    RolloutResult {
        win_prob,
        unresolved,
        rollouts,
        se,
    }
}
