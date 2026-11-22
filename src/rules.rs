//! Legality rules and action enumeration. Pure functions over `GameState`.

use crate::action::Action;
use crate::board::topology;
use crate::state::{GameState, Phase};
use crate::types::*;

/// Is `node` free to build a settlement on, ignoring road connection?
/// (The distance rule: no building on an adjacent node.)
pub fn node_open(st: &GameState, node: u8) -> bool {
    if st.node[node as usize].is_some() {
        return false;
    }
    topology().node_neighbors[node as usize]
        .iter()
        .all(|&nb| st.node[nb as usize].is_none())
}

/// Does `p` have a road or building touching `node`?
pub fn touches(st: &GameState, p: u8, node: u8) -> bool {
    if let Some(pc) = st.node[node as usize] {
        if pc.owner == p {
            return true;
        }
    }
    topology().node_edges[node as usize]
        .iter()
        .any(|&e| st.edge[e as usize] == Some(p))
}

/// Nodes where `p` could place a settlement (distance rule + road network).
pub fn settlement_spots(st: &GameState, p: u8) -> Vec<u8> {
    (0..N_NODES as u8)
        .filter(|&n| node_open(st, n) && road_reaches(st, p, n))
        .collect()
}

/// Does one of `p`'s roads end at `node`?
pub fn road_reaches(st: &GameState, p: u8, node: u8) -> bool {
    topology().node_edges[node as usize]
        .iter()
        .any(|&e| st.edge[e as usize] == Some(p))
}

/// Edges where `p` could build a road: empty, and reachable from one of their
/// own roads or buildings. A road may not be built out of a node occupied by
/// an opponent's settlement or city.
pub fn road_spots(st: &GameState, p: u8) -> Vec<u8> {
    let topo = topology();
    (0..N_EDGES as u8)
        .filter(|&e| {
            if st.edge[e as usize].is_some() {
                return false;
            }
            topo.edge_nodes[e as usize].iter().any(|&n| {
                match st.node[n as usize] {
                    // Own building: always a valid springboard.
                    Some(pc) if pc.owner == p => true,
                    // Opponent building blocks passage through this node.
                    Some(_) => false,
                    None => road_reaches(st, p, n),
                }
            })
        })
        .collect()
}

/// Nodes where `p` can upgrade a settlement to a city.
pub fn city_spots(st: &GameState, p: u8) -> Vec<u8> {
    (0..N_NODES as u8)
        .filter(|&n| {
            matches!(st.node[n as usize], Some(pc)
                if pc.owner == p && pc.building == Building::Settlement)
        })
        .collect()
}

/// Players who can be robbed if the robber goes to `hex`.
pub fn robbery_victims(st: &GameState, hex: u8, thief: u8) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::new();
    for &n in topology().hex_nodes[hex as usize].iter() {
        if let Some(pc) = st.node[n as usize] {
            if pc.owner != thief
                && !v.contains(&pc.owner)
                && st.players[pc.owner as usize].hand_size() > 0
            {
                v.push(pc.owner);
            }
        }
    }
    v
}

fn playable(st: &GameState, p: u8, c: DevCard) -> bool {
    !st.dev_played_this_turn && st.players[p as usize].dev[c.idx()] > 0
}

/// Every legal action for the player who must act. Discards are handled by a
/// policy hook rather than enumerated, since the combinations are vast.
pub fn legal_actions(st: &GameState) -> Vec<Action> {
    let p = st.actor();
    let me = &st.players[p as usize];
    let mut out: Vec<Action> = Vec::new();

    match st.phase {
        Phase::Over | Phase::Discarding => {}

        Phase::Setup => {
            if st.setup_need_road {
                let topo = topology();
                for &e in topo.node_edges[st.last_settlement as usize].iter() {
                    if st.edge[e as usize].is_none() {
                        out.push(Action::PlaceInitialRoad(e));
                    }
                }
            } else {
                for n in 0..N_NODES as u8 {
                    if node_open(st, n) {
                        out.push(Action::PlaceInitialSettlement(n));
                    }
                }
            }
        }

        Phase::MovingRobber => {
            for hex in 0..N_HEXES as u8 {
                if hex == st.board.robber {
                    continue;
                }
                let victims = robbery_victims(st, hex, p);
                if victims.is_empty() {
                    out.push(Action::MoveRobber { hex, victim: None });
                } else {
                    for v in victims {
                        out.push(Action::MoveRobber {
                            hex,
                            victim: Some(v),
                        });
                    }
                }
            }
        }

        Phase::PreRoll => {
            out.push(Action::Roll);
            // Only a knight is worth playing before the roll (to shift the
            // robber off your own hex, or to claim largest army first); the
            // other cards are restricted to the build phase.
            if playable(st, p, DevCard::Knight) {
                out.push(Action::PlayKnight);
            }
        }

        Phase::Main => {
            // Road Building grants two roads immediately; while they are
            // pending, placing them is the only legal move.
            if st.free_roads > 0 {
                let roads = road_spots(st, p);
                if !roads.is_empty() && me.roads_left > 0 {
                    for e in roads {
                        out.push(Action::BuildRoad(e));
                    }
                    return out;
                }
            }

            out.push(Action::EndTurn);

            if me.roads_left > 0 && can_afford(&me.hand, &COST_ROAD) {
                for e in road_spots(st, p) {
                    out.push(Action::BuildRoad(e));
                }
            }
            if me.settlements_left > 0 && can_afford(&me.hand, &COST_SETTLEMENT) {
                for n in settlement_spots(st, p) {
                    out.push(Action::BuildSettlement(n));
                }
            }
            if me.cities_left > 0 && can_afford(&me.hand, &COST_CITY) {
                for n in city_spots(st, p) {
                    out.push(Action::BuildCity(n));
                }
            }
            if !st.dev_deck.is_empty() && can_afford(&me.hand, &COST_DEV) {
                out.push(Action::BuyDev);
            }

            dev_card_actions(st, p, &mut out);

            // Bank and port trades.
            for give in Resource::ALL {
                let rate = me.trade_rate(give) as i16;
                if me.hand[give.idx()] < rate {
                    continue;
                }
                for want in Resource::ALL {
                    if want == give || st.bank[want.idx()] <= 0 {
                        continue;
                    }
                    out.push(Action::BankTrade {
                        give,
                        want,
                        rate: rate as u8,
                    });
                }
            }

            // Player-to-player offers: 1-for-1 and 2-for-1, which covers the
            // overwhelming majority of real table trades.
            if st.proposals_made < st.max_proposals && st.n_players() > 1 {
                for give in Resource::ALL {
                    let have = me.hand[give.idx()];
                    if have < 1 {
                        continue;
                    }
                    for want in Resource::ALL {
                        if want == give {
                            continue;
                        }
                        let mut g = ZERO;
                        g[give.idx()] = 1;
                        let mut w = ZERO;
                        w[want.idx()] = 1;
                        out.push(Action::ProposeTrade { give: g, want: w });
                        if have >= 2 {
                            let mut g2 = ZERO;
                            g2[give.idx()] = 2;
                            out.push(Action::ProposeTrade {
                                give: g2,
                                want: w,
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

fn dev_card_actions(st: &GameState, p: u8, out: &mut Vec<Action>) {
    if playable(st, p, DevCard::Knight) {
        out.push(Action::PlayKnight);
    }
    if playable(st, p, DevCard::RoadBuilding)
        && st.players[p as usize].roads_left > 0
        && !road_spots(st, p).is_empty()
    {
        out.push(Action::PlayRoadBuilding);
    }
    if playable(st, p, DevCard::YearOfPlenty) {
        for a in Resource::ALL {
            for b in Resource::ALL {
                if (b as u8) < (a as u8) {
                    continue;
                }
                if st.bank[a.idx()] > 0 && st.bank[b.idx()] > 0 {
                    out.push(Action::PlayYearOfPlenty(a, b));
                }
            }
        }
    }
    if playable(st, p, DevCard::Monopoly) {
        for r in Resource::ALL {
            out.push(Action::PlayMonopoly(r));
        }
    }
}
