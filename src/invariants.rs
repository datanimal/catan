//! Runtime consistency checks.
//!
//! These are the properties that must hold after *every* action, and the test
//! suite asserts them step by step through thousands of games. They are the
//! reason the numbers coming out of a run can be trusted: a rules bug that
//! quietly created or destroyed resources would change every conclusion.

use crate::board::topology;
use crate::state::GameState;
use crate::types::*;

/// Total resource cards in the game: the bank starts with 19 of each and
/// cards only ever move, never appear or vanish.
pub const TOTAL_CARDS: i16 = BANK_PER_RES * N_RES as i16;

pub fn check(st: &GameState) -> Result<(), String> {
    let topo = topology();

    // 1. Resource conservation, per resource and in total.
    for i in 0..N_RES {
        let held: i16 = st.players.iter().map(|p| p.hand[i]).sum();
        let total = held + st.bank[i];
        if total != BANK_PER_RES {
            return Err(format!(
                "resource {} does not add up: {} in hands + {} in bank = {} (want {})",
                Resource::from_idx(i).name(),
                held,
                st.bank[i],
                total,
                BANK_PER_RES
            ));
        }
        if st.bank[i] < 0 {
            return Err(format!("bank went negative for {}", Resource::from_idx(i).name()));
        }
    }

    // 2. No negative hands.
    for (p, pl) in st.players.iter().enumerate() {
        for i in 0..N_RES {
            if pl.hand[i] < 0 {
                return Err(format!("player {p} has {} of resource {i}", pl.hand[i]));
            }
        }
    }

    // 3. Development cards: 25 in the deck, in hands, or played.
    let in_hands: u32 = st
        .players
        .iter()
        .map(|p| p.dev.iter().map(|&c| c as u32).sum::<u32>()
            + p.dev_new.iter().map(|&c| c as u32).sum::<u32>())
        .sum();
    let knights: u32 = st.players.iter().map(|p| p.knights_played as u32).sum();
    // Non-knight cards that have been played are discarded and untracked, so
    // the accounting bound is one-sided: nothing may exceed the deck size.
    let accounted = st.dev_deck.len() as u32 + in_hands + knights;
    if accounted > 25 {
        return Err(format!("development cards exceed the deck: {accounted} > 25"));
    }

    // 4. Piece limits.
    for p in 0..st.n_players() as u8 {
        let roads = st.edge.iter().filter(|&&e| e == Some(p)).count() as u8;
        let settlements = st
            .node
            .iter()
            .filter(|s| matches!(s, Some(pc) if pc.owner == p && pc.building == Building::Settlement))
            .count() as u8;
        let cities = st
            .node
            .iter()
            .filter(|s| matches!(s, Some(pc) if pc.owner == p && pc.building == Building::City))
            .count() as u8;
        if roads > MAX_ROADS {
            return Err(format!("player {p} has {roads} roads"));
        }
        if settlements > MAX_SETTLEMENTS {
            return Err(format!("player {p} has {settlements} settlements"));
        }
        if cities > MAX_CITIES {
            return Err(format!("player {p} has {cities} cities"));
        }
        // Supply counters must agree with what is on the board.
        if roads + st.players[p as usize].roads_left != MAX_ROADS {
            return Err(format!(
                "player {p} road supply mismatch: {roads} placed + {} left",
                st.players[p as usize].roads_left
            ));
        }
        if settlements + st.players[p as usize].settlements_left != MAX_SETTLEMENTS {
            return Err(format!(
                "player {p} settlement supply mismatch: {settlements} placed + {} left",
                st.players[p as usize].settlements_left
            ));
        }
        if cities + st.players[p as usize].cities_left != MAX_CITIES {
            return Err(format!(
                "player {p} city supply mismatch: {cities} placed + {} left",
                st.players[p as usize].cities_left
            ));
        }
    }

    // 5. The distance rule: no two buildings on adjacent intersections.
    for n in 0..N_NODES {
        if st.node[n].is_none() {
            continue;
        }
        for &nb in topo.node_neighbors[n].iter() {
            if st.node[nb as usize].is_some() {
                return Err(format!("buildings on adjacent intersections {n} and {nb}"));
            }
        }
    }

    // 6. Special cards are only held when earned.
    if let Some(h) = st.longest_road {
        if st.players[h as usize].road_len < 5 {
            return Err(format!(
                "player {h} holds longest road with only {} roads",
                st.players[h as usize].road_len
            ));
        }
    }
    if let Some(h) = st.largest_army {
        if st.players[h as usize].knights_played < 3 {
            return Err(format!(
                "player {h} holds largest army with only {} knights",
                st.players[h as usize].knights_played
            ));
        }
    }

    // 7. The robber is on a real hex.
    if st.board.robber as usize >= N_HEXES {
        return Err("robber is off the board".into());
    }

    // 8. Every road connects to something that player owns.
    for e in 0..N_EDGES {
        let Some(owner) = st.edge[e] else { continue };
        let [a, b] = topo.edge_nodes[e];
        let linked = [a, b].iter().any(|&n| {
            matches!(st.node[n as usize], Some(pc) if pc.owner == owner)
                || topo.node_edges[n as usize]
                    .iter()
                    .any(|&o| o as usize != e && st.edge[o as usize] == Some(owner))
        });
        if !linked {
            return Err(format!("player {owner}'s road on edge {e} is orphaned"));
        }
    }

    Ok(())
}

/// Checks that only make sense once the game is over.
pub fn check_final(st: &GameState) -> Result<(), String> {
    check(st)?;
    if let Some(w) = st.winner {
        if st.vp(w) < st.vp_target {
            return Err(format!(
                "player {w} won with {} VP, target was {}",
                st.vp(w),
                st.vp_target
            ));
        }
        // Exactly one winner.
        let winners = (0..st.n_players() as u8)
            .filter(|&p| st.vp(p) >= st.vp_target)
            .count();
        if winners != 1 {
            return Err(format!("{winners} players are at or above the target"));
        }
    }
    Ok(())
}
