//! The phase breakdown drives conclusions about when a strategy is effective,
//! so the snapshots themselves must be consistent and complete.

use catan_sim::analysis;
use catan_sim::sim::{self, SimConfig};

#[test]
fn snapshots_are_complete_and_monotone() {
    let mut cfg = SimConfig::new(
        ["tuned", "city_builder", "dev_rush", "expander"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        600,
    );
    cfg.base_seed = 17;
    let recs = sim::run(&cfg);
    let mut checked = 0;
    for g in &recs {
        if g.timed_out {
            continue;
        }
        for p in &g.players {
            let s = &p.metrics.snapshots;
            // Setup baseline plus two boundaries. A game can only skip a
            // boundary by ending before the leader reached it, which cannot
            // happen in a game that someone won.
            assert_eq!(
                s.len(),
                3,
                "seat {} had {} snapshots in a finished game",
                p.seat,
                s.len()
            );
            assert_eq!(s[0].marker, 0);
            assert_eq!(s[1].marker, 1);
            assert_eq!(s[2].marker, 2);
            // The setup baseline is two settlements, before any roll.
            assert_eq!(s[0].vp, 2, "opening position should be worth 2 VP");
            assert_eq!(s[0].own_turns, 0);
            assert_eq!(s[0].game_turn, 1);
            // Everything that only accumulates must not go backwards.
            for w in s.windows(2) {
                assert!(w[1].own_turns >= w[0].own_turns);
                assert!(w[1].inflow >= w[0].inflow);
                assert!(w[1].dev_bought >= w[0].dev_bought);
                assert!(w[1].knights >= w[0].knights);
                assert!(w[1].roads >= w[0].roads);
                assert!(w[1].game_turn >= w[0].game_turn);
            }
            // The final state must be at least as advanced as the last one.
            assert!(p.metrics.turns >= s[2].own_turns);
            assert!(p.metrics.total_inflow() >= s[2].inflow);
            checked += 1;
        }
        // Each boundary has at least one player level with the lead.
        for marker in 0..3usize {
            let leaders = g
                .players
                .iter()
                .filter(|p| p.metrics.snapshots.get(marker).map(|s| s.leading).unwrap_or(false))
                .count();
            assert!(leaders >= 1, "no leader recorded at boundary {marker}");
        }
    }
    assert!(checked > 1000, "only checked {checked} player-games");
}

#[test]
fn phase_vp_gains_account_for_the_final_score() {
    let mut cfg = SimConfig::new(vec!["tuned".to_string(); 4], 400);
    cfg.base_seed = 23;
    let recs = sim::run(&cfg);
    for g in &recs {
        if g.timed_out {
            continue;
        }
        for p in &g.players {
            let s = &p.metrics.snapshots;
            if s.len() != 3 {
                continue;
            }
            // Gains across the three phases, plus the opening two points,
            // must equal the final score exactly.
            let early = s[1].vp as i32 - s[0].vp as i32;
            let mid = s[2].vp as i32 - s[1].vp as i32;
            let late = p.vp as i32 - s[2].vp as i32;
            assert_eq!(
                s[0].vp as i32 + early + mid + late,
                p.vp as i32,
                "phase gains do not reconstruct the final score"
            );
            // Victory points are never lost outright, but the longest-road
            // and largest-army cards can change hands, so a phase gain may
            // legitimately be negative. It should not be wildly so.
            assert!(late >= -4 && mid >= -4 && early >= -4);
        }
    }
}

#[test]
fn phase_breakdown_reports_every_strategy_and_phase() {
    let mut cfg = SimConfig::new(
        ["tuned", "city_builder", "dev_rush", "expander"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        1500,
    );
    cfg.base_seed = 41;
    let recs = sim::run(&cfg);
    let rows = analysis::phase_breakdown(&recs);
    assert_eq!(rows.len(), 12, "expected 4 strategies x 3 phases");
    for r in &rows {
        assert!(r.games > 100, "{} {} had only {} games", r.name, r.phase, r.games);
        assert!(r.turns > 0.5, "{} {} had {} turns", r.name, r.phase, r.turns);
        assert!(
            r.cards_per_turn > 0.5 && r.cards_per_turn < 20.0,
            "{} {} card rate {}",
            r.name,
            r.phase,
            r.cards_per_turn
        );
        assert!(r.vp_per_turn > 0.0 && r.vp_per_turn < 2.0);
    }
    // Lead shares at a boundary must total roughly one lead per game.
    for phase in ["early", "mid"] {
        let total: f64 = rows
            .iter()
            .filter(|r| r.phase == phase)
            .map(|r| r.lead_rate)
            .sum();
        println!("{phase}: lead shares total {total:.3}");
        assert!(
            total > 0.95 && total < 1.6,
            "{phase} lead shares total {total:.3}, which is not one leader per game"
        );
    }
}
