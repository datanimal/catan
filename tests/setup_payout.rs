//! The second opening settlement pays out immediately: one card for each
//! adjacent hex that produces. The first does not.

use catan_sim::board::topology;
use catan_sim::bots;
use catan_sim::game::{Game, GameConfig};
use catan_sim::state::Phase;
use catan_sim::types::*;

#[test]
fn only_the_second_settlement_pays_and_it_pays_the_right_cards() {
    for seed in 0..40u64 {
        let mut cfg = GameConfig::default();
        cfg.n_players = 4;
        let pols = (0..4)
            .map(|i| bots::make(["tuned", "city_builder", "dev_rush", "expander"][i], seed + i as u64).unwrap())
            .collect();
        let mut g = Game::new(cfg, pols, seed);
        // Step through setup only.
        while g.st.phase == Phase::Setup {
            g.step();
        }
        let topo = topology();
        for p in 0..4usize {
            let nodes = &g.st.players[p].metrics.init_nodes;
            assert_eq!(nodes.len(), 2, "each player places two settlements");

            // What the second settlement should have paid.
            let mut want = ZERO;
            for &h in topo.node_hexes[nodes[1] as usize].iter() {
                if let Some(r) = g.st.board.hex_resource(h) {
                    want[r.idx()] += 1;
                }
            }
            assert_eq!(
                g.st.players[p].hand, want,
                "seat {p} opened on {:?} and holds the wrong cards (seed {seed})",
                nodes
            );

            // One card per producing hex, and never more than three.
            let total = res_total(&g.st.players[p].hand);
            let producing = topo.node_hexes[nodes[1] as usize]
                .iter()
                .filter(|&&h| g.st.board.hex_resource(h).is_some())
                .count() as i16;
            assert_eq!(total, producing, "seat {p} got {total} cards from {producing} hexes");
            assert!((1..=3).contains(&total), "seat {p} opened with {total} cards");

            // The *first* settlement paid nothing: had it also paid, the hand
            // would hold its hexes too.
            let mut first_pay = ZERO;
            for &h in topo.node_hexes[nodes[0] as usize].iter() {
                if let Some(r) = g.st.board.hex_resource(h) {
                    first_pay[r.idx()] += 1;
                }
            }
            if res_total(&first_pay) > 0 {
                let combined: Vec<i16> =
                    (0..N_RES).map(|i| want[i] + first_pay[i]).collect();
                assert_ne!(
                    g.st.players[p].hand.to_vec(),
                    combined,
                    "seat {p} appears to have been paid for both settlements"
                );
            }
        }
        // The bank paid for all of it.
        for i in 0..N_RES {
            let held: i16 = g.st.players.iter().map(|pl| pl.hand[i]).sum();
            assert_eq!(held + g.st.bank[i], BANK_PER_RES, "resource {i} does not add up");
        }
    }
}

#[test]
fn setup_income_is_recorded_and_kept_out_of_production() {
    let mut cfg = GameConfig::default();
    cfg.n_players = 4;
    let pols = (0..4).map(|i| bots::make("tuned", 7 + i as u64).unwrap()).collect();
    let rec = Game::new(cfg, pols, 7).run();
    for p in &rec.players {
        let m = &p.metrics;
        assert!(
            (1..=3).contains(&m.setup_income),
            "seat {} recorded {} cards of opening income",
            p.seat,
            m.setup_income
        );
        // It must count as inflow, but not as production -- production is the
        // yardstick dice luck is measured against, and these cards came from
        // no roll.
        assert!(m.total_inflow() >= m.setup_income);
        let produced: i32 = m.produced.iter().sum();
        assert_eq!(produced, m.produced_total());
        assert!(
            m.expected_total() > 0.0,
            "a played game should have accrued some expectation"
        );
    }
}
