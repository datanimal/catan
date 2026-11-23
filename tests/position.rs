//! The position format is the integration contract: anything that can emit it
//! can be evaluated. It therefore has to round-trip faithfully and must reject
//! impossible positions rather than quietly scoring them.

use catan_sim::bots;
use catan_sim::game::{Game, GameConfig};
use catan_sim::invariants;
use catan_sim::position::{Position, Ref};
use catan_sim::state::Phase;

fn mid_game(seed: u64, until_vp: u8) -> catan_sim::state::GameState {
    let mut cfg = GameConfig::default();
    cfg.n_players = 4;
    let pols = (0..4)
        .map(|i| {
            bots::make(
                ["tuned", "city_builder", "dev_rush", "expander"][i],
                seed * 13 + i as u64,
            )
            .unwrap()
        })
        .collect();
    let mut g = Game::new(cfg, pols, seed);
    while g.st.phase != Phase::Over {
        let lead = (0..g.st.n_players() as u8).map(|p| g.st.vp(p)).max().unwrap_or(0);
        if lead >= until_vp && g.st.phase == Phase::PreRoll {
            break;
        }
        g.step();
    }
    g.st
}

#[test]
fn positions_round_trip_through_json() {
    for seed in 0..25u64 {
        for until in [4u8, 7, 9] {
            let st = mid_game(seed, until);
            let pos = Position::from_state(&st);
            // Through JSON and back, as an external tool would send it.
            let text = serde_json::to_string(&pos).expect("serialise");
            let back: Position = serde_json::from_str(&text).expect("parse");
            let st2 = back
                .to_state()
                .unwrap_or_else(|e| panic!("seed {seed} until {until}: {e}"));

            assert_eq!(st2.n_players(), st.n_players());
            assert_eq!(st2.current, st.current);
            assert_eq!(st2.vp_target, st.vp_target);
            assert_eq!(st2.board.robber, st.board.robber);
            assert_eq!(st2.board.numbers, st.board.numbers);
            // Every piece back where it was.
            for n in 0..catan_sim::types::N_NODES {
                assert_eq!(st2.node[n], st.node[n], "intersection {n} differs");
            }
            for e in 0..catan_sim::types::N_EDGES {
                assert_eq!(st2.edge[e], st.edge[e], "path {e} differs");
            }
            for p in 0..st.n_players() {
                assert_eq!(st2.players[p].hand, st.players[p].hand);
                assert_eq!(st2.players[p].knights_played, st.players[p].knights_played);
                // Scores must survive, hidden cards included.
                assert_eq!(st2.vp(p as u8), st.vp(p as u8), "score of seat {p} differs");
                assert_eq!(st2.vp_public(p as u8), st.vp_public(p as u8));
                assert_eq!(st2.players[p].port_mask, st.players[p].port_mask);
            }
            // Special cards must be re-derived, not carried over by luck.
            assert_eq!(st2.longest_road, st.longest_road);
            assert_eq!(st2.largest_army, st.largest_army);
            invariants::check(&st2).expect("round-tripped position must be legal");
        }
    }
}

#[test]
fn impossible_positions_are_rejected() {
    let st = mid_game(3, 6);
    let good = Position::from_state(&st);

    // Two buildings on the same intersection.
    let mut p = good.clone();
    let n = p.players[1].settlements[0].node().unwrap();
    p.players[0].settlements.push(Ref::Index(n));
    assert!(p.to_state().is_err(), "a double-occupied intersection was accepted");

    // Buildings on adjacent intersections, which the distance rule forbids.
    let mut p = good.clone();
    let topo = catan_sim::board::topology();
    let occupied = p.players[0].settlements[0].node().unwrap();
    let neighbour = topo.node_neighbors[occupied as usize][0];
    let taken = |pl: &catan_sim::position::PlayerPos| {
        pl.settlements
            .iter()
            .chain(pl.cities.iter())
            .any(|r| r.node().ok() == Some(neighbour))
    };
    if p.players.iter().all(|pl| !taken(pl)) {
        p.players[0].settlements.push(Ref::Index(neighbour));
        assert!(
            p.to_state().is_err(),
            "a distance-rule violation was accepted"
        );
    }

    // More resources than exist in the game.
    let mut p = good.clone();
    p.players[0].hand.insert("ore".into(), 40);
    assert!(p.to_state().is_err(), "40 ore was accepted");

    // Nonsense names and out-of-range indices.
    let mut p = good.clone();
    p.board.terrain[0] = "swamp".into();
    assert!(p.to_state().is_err(), "an unknown terrain was accepted");

    let mut p = good.clone();
    p.players[0].roads.push(Ref::Index(200));
    assert!(p.to_state().is_err(), "a path that does not exist was accepted");

    let mut p = good.clone();
    p.current = 9;
    assert!(p.to_state().is_err(), "a player not at the table was accepted");

    // Too many pieces.
    let mut p = good.clone();
    p.players[0].cities = (0..6).map(Ref::Index).collect();
    assert!(p.to_state().is_err(), "six cities were accepted");

    // And the untouched original must still be fine.
    assert!(good.to_state().is_ok());
}

#[test]
fn rollouts_are_sane_and_sum_to_one() {
    use catan_sim::bots::Strategy;
    use catan_sim::rollout;
    let lineup: Vec<Strategy> = ["tuned", "city_builder", "dev_rush", "expander"]
        .iter()
        .map(|s| Strategy::parse(s).unwrap())
        .collect();

    // From a position where one player is on the brink, they should be the
    // clear favourite.
    let st = mid_game(5, 9);
    let leader = (0..st.n_players() as u8)
        .max_by_key(|&p| st.vp(p))
        .unwrap();
    let r = rollout::evaluate(&st, &lineup, 1200, 99, false);
    let total: f64 = r.win_prob.iter().sum();
    println!(
        "leader is seat {leader} at {} VP; probabilities {:?}",
        st.vp(leader),
        r.win_prob.iter().map(|p| (100.0 * p).round()).collect::<Vec<_>>()
    );
    assert!(
        (total + r.unresolved as f64 / r.rollouts as f64 - 1.0).abs() < 1e-9,
        "probabilities and unresolved games must account for every rollout"
    );
    assert!(
        r.win_prob[leader as usize] > 0.4,
        "a player one point from victory won only {:.0}% of rollouts",
        100.0 * r.win_prob[leader as usize]
    );
    // Every seat's estimate must be a probability.
    assert!(r.win_prob.iter().all(|p| *p >= 0.0 && *p <= 1.0));

    // Resampling hidden hands must not change the answer wildly, since it
    // preserves hand sizes and the bank.
    let r2 = rollout::evaluate(&st, &lineup, 1200, 99, true);
    let shift: f64 = r
        .win_prob
        .iter()
        .zip(r2.win_prob.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    println!("redrawing hands moved the estimate by at most {:.1} points", 100.0 * shift);
    assert!(shift < 0.30, "redrawing hands moved an estimate by {shift}");
}

#[test]
fn a_resumed_game_finishes_legally() {
    use catan_sim::game::Game;
    let st = mid_game(8, 6);
    let mut cfg = GameConfig::default();
    cfg.n_players = 4;
    cfg.turn_limit = st.turn + 400;
    let start_turn = st.turn;
    let pols = (0..4).map(|i| bots::make("tuned", 500 + i as u64).unwrap()).collect();
    let mut g = Game::from_state(st, cfg, pols, 1234);
    let mut steps = 0;
    while g.st.phase != Phase::Over && steps < 200_000 {
        g.step();
        steps += 1;
        invariants::check(&g.st).expect("a resumed game must stay legal");
    }
    assert!(g.st.phase == Phase::Over);
    assert!(g.st.turn > start_turn, "the resumed game made no progress");
    invariants::check_final(&g.st).expect("final position must be legal");
}
