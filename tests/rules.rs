//! Rules conformance: step through many games checking every invariant after
//! every single action, across all strategies and board layouts.

use catan_sim::board::BoardLayout;
use catan_sim::bots;
use catan_sim::game::{DiceMode, Game, GameConfig};
use catan_sim::invariants;
use catan_sim::state::Phase;

fn lineups() -> Vec<Vec<&'static str>> {
    vec![
        vec!["balanced", "city_builder", "dev_rush", "expander"],
        vec!["trader", "hermit", "random", "balanced"],
        vec!["random", "random", "random"],
        vec!["dev_rush", "expander"],
        vec!["balanced", "trader", "city_builder", "random", "hermit", "expander"],
    ]
}

#[test]
fn invariants_hold_after_every_action() {
    let mut checked = 0u64;
    for (li, names) in lineups().iter().enumerate() {
        for seed in 0..18u64 {
            let cfg = GameConfig {
                n_players: names.len(),
                layout: match seed % 3 {
                    0 => BoardLayout::Standard,
                    1 => BoardLayout::Random,
                    _ => BoardLayout::RandomBalanced,
                },
                random_ports: seed % 2 == 0,
                dice: if seed % 4 == 0 {
                    DiceMode::Deck36
                } else {
                    DiceMode::TwoD6
                },
                ..Default::default()
            };
            let pols = names
                .iter()
                .enumerate()
                .map(|(i, n)| bots::make(n, seed * 31 + i as u64).unwrap())
                .collect();
            let mut g = Game::new(cfg, pols, seed + li as u64 * 1000);
            let mut steps = 0;
            while g.st.phase != Phase::Over && steps < 200_000 {
                g.step();
                steps += 1;
                checked += 1;
                if let Err(e) = invariants::check(&g.st) {
                    panic!(
                        "lineup {names:?} seed {seed} step {steps} ({:?}): {e}",
                        g.st.phase
                    );
                }
            }
            assert!(steps < 200_000, "game did not terminate");
            invariants::check_final(&g.st)
                .unwrap_or_else(|e| panic!("lineup {names:?} seed {seed}: {e}"));
        }
    }
    assert!(
        checked > 60_000,
        "expected a meaningful number of checks, got {checked}"
    );
    println!("validated {checked} actions");
}

#[test]
fn games_finish_and_the_winner_has_the_target() {
    let cfg = GameConfig::default();
    let mut timeouts = 0;
    for seed in 0..200u64 {
        let pols = (0..4)
            .map(|i| {
                bots::make(
                    ["balanced", "city_builder", "dev_rush", "expander"][i],
                    seed * 11 + i as u64,
                )
                .unwrap()
            })
            .collect();
        let rec = Game::new(cfg.clone(), pols, seed).run();
        if rec.timed_out {
            timeouts += 1;
            continue;
        }
        let w = rec.winner.expect("a finished game has a winner");
        let wp = &rec.players[w as usize];
        assert!(wp.won);
        assert!(wp.vp >= cfg.vp_target, "winner had {} VP", wp.vp);
        // Exactly one player may be *declared* the winner -- but more than
        // one may sit at or above the target. A player can be pushed to 10 on
        // somebody else's turn, when a road or settlement transfers the
        // longest-road card to them, and the rules only let you win on your
        // own turn. If another player reaches the target first, the game ends
        // with two players at 10 and one winner. This happens in roughly 1
        // game in 1300.
        assert_eq!(rec.players.iter().filter(|p| p.won).count(), 1);
        for p in &rec.players {
            if !p.won && p.vp >= cfg.vp_target {
                assert!(
                    p.metrics.had_longest_road || p.metrics.had_largest_army,
                    "seat {} reached {} VP without winning and without a \
                     special card, which should be impossible",
                    p.seat,
                    p.vp
                );
            }
        }
        // The victory points must add up from their parts.
        let m = &wp.metrics;
        let from_buildings = 2 + m.settlements_built as u8 + m.cities_built as u8;
        let expected = from_buildings
            + m.vp_from_cards
            + 2 * m.had_longest_road as u8
            + 2 * m.had_largest_army as u8;
        assert_eq!(
            wp.vp, expected,
            "VP breakdown mismatch: {} vs {} (seed {seed})",
            wp.vp, expected
        );
    }
    assert_eq!(timeouts, 0, "{timeouts} of 200 games failed to finish");
}

#[test]
fn same_seed_gives_the_same_game() {
    let cfg = GameConfig::default();
    let build = || {
        (0..4)
            .map(|i| bots::make(["balanced", "trader", "dev_rush", "expander"][i], 77 + i as u64).unwrap())
            .collect::<Vec<_>>()
    };
    for seed in [1u64, 42, 12345] {
        let a = Game::new(cfg.clone(), build(), seed).run();
        let b = Game::new(cfg.clone(), build(), seed).run();
        assert_eq!(a.winner, b.winner);
        assert_eq!(a.turns, b.turns);
        assert_eq!(a.roll_histogram, b.roll_histogram);
        for (x, y) in a.players.iter().zip(b.players.iter()) {
            assert_eq!(x.vp, y.vp);
            assert_eq!(x.metrics.produced_total(), y.metrics.produced_total());
        }
    }
}

#[test]
fn a_bot_that_refuses_trades_never_trades() {
    // The 'hermit' preset is the control case for trading experiments, so it
    // must genuinely never trade with another player.
    let cfg = GameConfig::default();
    for seed in 0..60u64 {
        let pols = (0..4)
            .map(|i| bots::make(["hermit", "trader", "trader", "trader"][i], seed * 5 + i as u64).unwrap())
            .collect();
        let rec = Game::new(cfg.clone(), pols, seed).run();
        let h = &rec.players[0];
        assert_eq!(h.policy, "hermit");
        assert_eq!(h.metrics.trades_proposed, 0, "hermit proposed a trade");
        assert_eq!(
            h.metrics.trades_accepted_for_others, 0,
            "hermit accepted a trade"
        );
        assert_eq!(
            h.metrics.player_trade_in, 0,
            "hermit received cards in a trade"
        );
    }
}

#[test]
fn disabling_trades_stops_all_trading() {
    let cfg = GameConfig {
        allow_player_trades: false,
        ..Default::default()
    };
    for seed in 0..40u64 {
        let pols = (0..4)
            .map(|i| bots::make("trader", seed * 3 + i as u64).unwrap())
            .collect();
        let rec = Game::new(cfg.clone(), pols, seed).run();
        for p in &rec.players {
            assert_eq!(p.metrics.trades_proposed, 0);
            assert_eq!(p.metrics.player_trade_in, 0);
            // Bank and port trades must still work.
        }
        assert!(rec.players.iter().any(|p| p.metrics.bank_trades > 0));
    }
}
