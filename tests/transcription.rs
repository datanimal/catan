//! Transcribing a board from a picture is error-prone, and a misreading is
//! usually a perfectly *legal* position, so the legality check cannot catch
//! it. These tests cover the machinery that can.

use catan_sim::bots;
use catan_sim::game::{Game, GameConfig};
use catan_sim::position::{Position, Ref};
use catan_sim::state::Phase;

fn sample_position(seed: u64) -> Position {
    let mut cfg = GameConfig::default();
    cfg.n_players = 4;
    let pols = (0..4)
        .map(|i| bots::make(["tuned", "city_builder", "dev_rush", "expander"][i], seed + i as u64).unwrap())
        .collect();
    let mut g = Game::new(cfg, pols, seed);
    while g.st.phase != Phase::Over {
        let lead = (0..4u8).map(|p| g.st.vp(p)).max().unwrap_or(0);
        if lead >= 6 && g.st.phase == Phase::PreRoll {
            break;
        }
        g.step();
    }
    Position::from_state(&g.st)
}

#[test]
fn hex_relative_names_and_indices_agree() {
    let pos = sample_position(21);
    let st_named = pos.to_state().expect("named form resolves");
    // Rewrite every reference as a raw index and check nothing moved.
    let mut by_index = pos.clone();
    for p in by_index.players.iter_mut() {
        p.settlements = p
            .settlements
            .iter()
            .map(|r| Ref::Index(r.node().unwrap()))
            .collect();
        p.cities = p.cities.iter().map(|r| Ref::Index(r.node().unwrap())).collect();
        p.roads = p.roads.iter().map(|r| Ref::Index(r.edge().unwrap())).collect();
    }
    let st_idx = by_index.to_state().expect("index form resolves");
    for n in 0..catan_sim::types::N_NODES {
        assert_eq!(st_named.node[n], st_idx.node[n]);
    }
    for e in 0..catan_sim::types::N_EDGES {
        assert_eq!(st_named.edge[e], st_idx.edge[e]);
    }

    // Every alternative name for a spot must resolve to the same place.
    for n in 0..catan_sim::types::N_NODES as u8 {
        let names = catan_sim::board::node_names(n);
        assert!(!names.is_empty());
        for nm in names {
            assert_eq!(Ref::Named(nm.clone()).node().unwrap(), n, "{nm} moved");
        }
    }
    for e in 0..catan_sim::types::N_EDGES as u8 {
        for nm in catan_sim::board::edge_names(e) {
            assert_eq!(Ref::Named(nm.clone()).edge().unwrap(), e, "{nm} moved");
        }
    }
}

#[test]
fn a_clean_board_raises_no_warnings() {
    for seed in 0..12u64 {
        let pos = sample_position(seed * 7 + 1);
        let w = pos.check_composition();
        assert!(w.is_empty(), "a simulated board was flagged: {w:?}");
    }
}

#[test]
fn misreadings_are_caught_even_though_they_are_legal() {
    let pos = sample_position(5);

    // Reading a 6 as an 8 keeps the board perfectly legal, and breaks the
    // token recipe.
    let mut p = pos.clone();
    let six = p.board.numbers.iter().position(|&n| n == 6).unwrap();
    p.board.numbers[six] = 8;
    assert!(
        p.to_state().is_ok(),
        "the misread board should still be a legal board"
    );
    let w = p.check_composition();
    println!("misread token: {w:?}");
    assert!(!w.is_empty(), "a duplicated 8 went unnoticed");
    assert!(w.iter().any(|m| m.contains("number tokens")));

    // Reading a pasture as a field: legal, wrong, and caught.
    let mut p = pos.clone();
    let past = p
        .board
        .terrain
        .iter()
        .position(|t| t == "pasture")
        .unwrap();
    p.board.terrain[past] = "fields".into();
    assert!(p.to_state().is_ok());
    let w = p.check_composition();
    println!("misread terrain: {w:?}");
    assert!(w.iter().any(|m| m.contains("pasture")));
    assert!(w.iter().any(|m| m.contains("fields")));

    // Losing a port.
    let mut p = pos.clone();
    p.board.ports.as_mut().unwrap().pop();
    let w = p.check_composition();
    assert!(w.iter().any(|m| m.contains("ports")), "a missing port went unnoticed");
}

#[test]
fn bad_names_are_reported_clearly() {
    let pos = sample_position(9);
    for (bad, expect) in [
        ("7:Q", "corner"),
        ("99:N", "hexes 0-18"),
        ("sevenN", "<hex>:<corner>"),
        ("x:N", "hex number"),
    ] {
        let mut p = pos.clone();
        p.players[0].settlements = vec![Ref::Named(bad.into())];
        let err = p.to_state().unwrap_err();
        println!("{bad:<10} -> {err}");
        assert!(
            err.contains(expect),
            "error for '{bad}' did not mention {expect}: {err}"
        );
    }
}
