//! A serialisable game position, so a state can come from outside the
//! simulator: a live game, a video, a replay log.
//!
//! This is the integration contract. Anything that can emit this JSON can be
//! evaluated, which keeps platform-specific log parsing at the edges instead
//! of in the engine.
//!
//! Positions are **validated** on the way in, against the same invariants the
//! engine holds itself to. A transcription slip — two settlements on adjacent
//! intersections, more roads than a player owns, resource counts that do not
//! add up — is reported rather than silently evaluated, because a win
//! probability computed from an impossible position is worse than no number
//! at all.

use crate::board::{topology, Board, BoardLayout, PortKind};
use crate::state::{GameState, Phase, Piece};
use crate::types::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// A board, described the way a human reading one off a screen would.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoardSpec {
    /// Terrain of each of the 19 hexes, in rows of 3/4/5/4/3 (see
    /// `board::HEX_AXIAL`).
    pub terrain: Vec<String>,
    /// Number token per hex; 0 on the desert.
    pub numbers: Vec<u8>,
    /// Which hex the robber sits on. Defaults to the desert.
    #[serde(default)]
    pub robber: Option<u8>,
    /// Ports as `(kind, edge)`, kind being a resource name or "any".
    /// Omit to use the standard arrangement.
    #[serde(default)]
    pub ports: Option<Vec<(String, u8)>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlayerPos {
    #[serde(default)]
    pub name: String,
    /// Intersections holding a settlement.
    #[serde(default)]
    pub settlements: Vec<u8>,
    /// Intersections holding a city.
    #[serde(default)]
    pub cities: Vec<u8>,
    /// Paths holding a road.
    #[serde(default)]
    pub roads: Vec<u8>,
    /// Resource cards, as `{"brick": 2, "ore": 1}`. Unknown hands may be left
    /// out, in which case `hand_size` is used and the mix is guessed.
    #[serde(default)]
    pub hand: std::collections::BTreeMap<String, i16>,
    /// Number of cards held, when the mix is not known.
    #[serde(default)]
    pub hand_size: Option<i16>,
    /// Knights already played, face up.
    #[serde(default)]
    pub knights_played: u8,
    /// Unplayed development cards held, when the faces are not known.
    #[serde(default)]
    pub dev_in_hand: u8,
    /// Victory-point cards held, if known.
    #[serde(default)]
    pub victory_point_cards: u8,
    /// Holds the longest-road card. This cannot be worked out from the board:
    /// when players tie for the longest road the card stays with whoever got
    /// there first, which is a fact about history. Both cards are face up, so
    /// a spectator can simply read them off.
    #[serde(default)]
    pub longest_road: bool,
    /// Holds the largest-army card, for the same reason.
    #[serde(default)]
    pub largest_army: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Position {
    pub board: BoardSpec,
    pub players: Vec<PlayerPos>,
    /// Whose turn it is.
    #[serde(default)]
    pub current: u8,
    #[serde(default = "default_target")]
    pub vp_target: u8,
    /// Turns played so far, used by the progress features.
    #[serde(default)]
    pub turn: u32,
}

fn default_target() -> u8 {
    10
}

fn parse_terrain(s: &str) -> Result<Terrain, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "hills" | "brick" => Terrain::Hills,
        "forest" | "lumber" | "wood" => Terrain::Forest,
        "pasture" | "wool" | "sheep" => Terrain::Pasture,
        "fields" | "grain" | "wheat" => Terrain::Fields,
        "mountains" | "ore" => Terrain::Mountains,
        "desert" => Terrain::Desert,
        other => return Err(format!("unknown terrain '{other}'")),
    })
}

fn parse_resource(s: &str) -> Result<Resource, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "brick" => Resource::Brick,
        "lumber" | "wood" => Resource::Lumber,
        "wool" | "sheep" => Resource::Wool,
        "grain" | "wheat" => Resource::Grain,
        "ore" => Resource::Ore,
        other => return Err(format!("unknown resource '{other}'")),
    })
}

impl Position {
    /// Build a validated `GameState`. Errors describe what is wrong with the
    /// position rather than panicking part-way through.
    pub fn to_state(&self) -> Result<GameState, String> {
        if self.board.terrain.len() != N_HEXES || self.board.numbers.len() != N_HEXES {
            return Err(format!(
                "a board needs {N_HEXES} terrains and {N_HEXES} numbers, got {} and {}",
                self.board.terrain.len(),
                self.board.numbers.len()
            ));
        }
        if !(2..=6).contains(&self.players.len()) {
            return Err(format!("need 2..=6 players, got {}", self.players.len()));
        }
        if self.current as usize >= self.players.len() {
            return Err(format!("current player {} is not at the table", self.current));
        }

        // Start from a standard board and overwrite it, so ports and the
        // derived tables are built by the same code the simulator uses.
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut board = Board::new(BoardLayout::Standard, false, &mut rng);
        for h in 0..N_HEXES {
            board.terrain[h] = parse_terrain(&self.board.terrain[h])?;
            board.numbers[h] = self.board.numbers[h];
            if board.terrain[h] == Terrain::Desert {
                board.numbers[h] = 0;
            } else if !(2..=12).contains(&board.numbers[h]) || board.numbers[h] == 7 {
                return Err(format!(
                    "hex {h} has number {}, which is not a token",
                    board.numbers[h]
                ));
            }
        }
        if let Some(ports) = &self.board.ports {
            let topo = topology();
            board.ports.clear();
            board.node_port_mask = vec![0u8; N_NODES];
            for (kind, edge) in ports {
                if *edge as usize >= N_EDGES {
                    return Err(format!("port on path {edge}, which does not exist"));
                }
                let k = if kind.eq_ignore_ascii_case("any") || kind == "3:1" {
                    PortKind::Generic
                } else {
                    PortKind::Specific(parse_resource(kind)?)
                };
                let bit = match k {
                    PortKind::Generic => 1u8 << 5,
                    PortKind::Specific(r) => 1u8 << r.idx(),
                };
                for &n in topo.edge_nodes[*edge as usize].iter() {
                    board.node_port_mask[n as usize] |= bit;
                }
                board.ports.push(crate::board::Port { kind: k, edge: *edge });
            }
        }
        // Rebuild the roll table and place the robber.
        board.roll_hexes = Default::default();
        for h in 0..N_HEXES {
            if board.terrain[h] != Terrain::Desert && board.numbers[h] >= 2 {
                let n = board.numbers[h] as usize;
                board.roll_hexes[n].push(h as u8);
            }
        }
        board.robber = match self.board.robber {
            Some(h) if (h as usize) < N_HEXES => h,
            Some(h) => return Err(format!("robber on hex {h}, which does not exist")),
            None => (0..N_HEXES)
                .find(|&h| board.terrain[h] == Terrain::Desert)
                .unwrap_or(0) as u8,
        };

        let mut st = GameState::new(board, self.players.len(), self.vp_target, 400);
        st.phase = Phase::PreRoll;
        st.current = self.current;
        st.turn = self.turn.max(1);
        // The deck is whatever has not been drawn. Cards that were played and
        // discarded cannot be seen from outside, so counting only what is
        // held and what has been played face up makes this an upper bound on
        // the deck -- which is the right direction to err, since it never
        // claims a card exists that does not.
        let mut deck = DevCard::standard_deck();
        let mut drawn = 0usize;

        for (pi, p) in self.players.iter().enumerate() {
            let owner = pi as u8;
            for &n in p.settlements.iter().chain(p.cities.iter()) {
                if n as usize >= N_NODES {
                    return Err(format!("intersection {n} does not exist"));
                }
            }
            for &n in &p.settlements {
                if st.node[n as usize].is_some() {
                    return Err(format!("two buildings on intersection {n}"));
                }
                st.node[n as usize] = Some(Piece {
                    owner,
                    building: Building::Settlement,
                });
            }
            for &n in &p.cities {
                if st.node[n as usize].is_some() {
                    return Err(format!("two buildings on intersection {n}"));
                }
                st.node[n as usize] = Some(Piece {
                    owner,
                    building: Building::City,
                });
            }
            for &e in &p.roads {
                if e as usize >= N_EDGES {
                    return Err(format!("path {e} does not exist"));
                }
                if st.edge[e as usize].is_some() {
                    return Err(format!("two roads on path {e}"));
                }
                st.edge[e as usize] = Some(owner);
            }
            let pl = &mut st.players[pi];
            pl.settlements_left = MAX_SETTLEMENTS
                .checked_sub(p.settlements.len() as u8)
                .ok_or_else(|| format!("player {pi} has more than {MAX_SETTLEMENTS} settlements"))?;
            pl.cities_left = MAX_CITIES
                .checked_sub(p.cities.len() as u8)
                .ok_or_else(|| format!("player {pi} has more than {MAX_CITIES} cities"))?;
            pl.roads_left = MAX_ROADS
                .checked_sub(p.roads.len() as u8)
                .ok_or_else(|| format!("player {pi} has more than {MAX_ROADS} roads"))?;
            pl.knights_played = p.knights_played;

            // Resources: an explicit mix if given, otherwise spread a stated
            // hand size evenly, which is the best a spectator could do.
            if !p.hand.is_empty() {
                for (k, v) in &p.hand {
                    if *v < 0 {
                        return Err(format!("player {pi} has {v} {k}"));
                    }
                    pl.hand[parse_resource(k)?.idx()] = *v;
                }
            } else if let Some(size) = p.hand_size {
                if size < 0 {
                    return Err(format!("player {pi} has a hand of {size}"));
                }
                for i in 0..size {
                    pl.hand[(i as usize) % N_RES] += 1;
                }
            }
            // Development cards. Victory-point cards are taken as given,
            // since they are *hidden victory points* and a score is wrong
            // without them. The rest are unknown faces, so they are split in
            // the proportions of the deck they came from -- mostly knights,
            // which is what 14 of the 20 non-victory cards are. Calling them
            // all knights would overflow the 14 that exist once several
            // players hold cards.
            let vp_cards = p.victory_point_cards.min(p.dev_in_hand.max(p.victory_point_cards));
            pl.dev[DevCard::VictoryPoint.idx()] = vp_cards;
            let others = p.dev_in_hand.saturating_sub(vp_cards);
            let knights = ((others as f64 * 14.0 / 20.0).round() as u8).min(others);
            pl.dev[DevCard::Knight.idx()] += knights;
            let mut rest = others - knights;
            for c in [DevCard::RoadBuilding, DevCard::YearOfPlenty, DevCard::Monopoly] {
                if rest == 0 {
                    break;
                }
                pl.dev[c.idx()] += 1;
                rest -= 1;
            }
            pl.dev[DevCard::Knight.idx()] += rest;
            drawn += p.knights_played as usize + p.dev_in_hand as usize;
        }

        // Port access is derived from where the buildings stand, not carried
        // in the file: a settlement on a port intersection trades at that
        // port's rate, and forgetting to grant it would silently make every
        // rollout trade at 4:1.
        for pi in 0..self.players.len() {
            let owner = pi as u8;
            for n in 0..N_NODES as u8 {
                if st.node[n as usize].map(|pc| pc.owner == owner).unwrap_or(false) {
                    st.grant_ports(owner, n);
                }
            }
        }

        // The bank holds whatever the players do not.
        for i in 0..N_RES {
            let held: i16 = st.players.iter().map(|p| p.hand[i]).sum();
            if held > BANK_PER_RES {
                return Err(format!(
                    "players hold {held} {}, but only {BANK_PER_RES} exist",
                    Resource::from_idx(i).name()
                ));
            }
            st.bank[i] = BANK_PER_RES - held;
        }
        deck.shuffle(&mut rng);
        deck.truncate(25usize.saturating_sub(drawn));
        st.dev_deck = deck;
        st.update_longest_road();
        st.update_largest_army();
        // Honour explicitly recorded card holders, since a tie cannot be
        // resolved from the board. Each claim is still checked against the
        // threshold, so a mis-transcribed file is reported rather than
        // silently awarding two points.
        if let Some(pi) = self.players.iter().position(|p| p.longest_road) {
            if st.players[pi].road_len < 5 {
                return Err(format!(
                    "player {pi} is marked as holding the longest road but has only {} roads \
                     in a row",
                    st.players[pi].road_len
                ));
            }
            st.longest_road = Some(pi as u8);
        }
        if self.players.iter().filter(|p| p.longest_road).count() > 1 {
            return Err("more than one player holds the longest-road card".into());
        }
        if let Some(pi) = self.players.iter().position(|p| p.largest_army) {
            if st.players[pi].knights_played < 3 {
                return Err(format!(
                    "player {pi} is marked as holding the largest army but has played only {} \
                     knights",
                    st.players[pi].knights_played
                ));
            }
            st.largest_army = Some(pi as u8);
        }
        if self.players.iter().filter(|p| p.largest_army).count() > 1 {
            return Err("more than one player holds the largest-army card".into());
        }

        // Finally hold the position to the engine's own standards.
        crate::invariants::check(&st).map_err(|e| format!("position is not legal: {e}"))?;
        Ok(st)
    }

    /// Read a position from a JSON file.
    pub fn load(path: &str) -> Result<Position, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&text).map_err(|e| e.to_string())
    }

    /// Describe a `GameState` as a position, for round-tripping and for
    /// generating example files.
    pub fn from_state(st: &GameState) -> Position {
        let terrain = st
            .board
            .terrain
            .iter()
            .map(|t| {
                match t {
                    Terrain::Hills => "hills",
                    Terrain::Forest => "forest",
                    Terrain::Pasture => "pasture",
                    Terrain::Fields => "fields",
                    Terrain::Mountains => "mountains",
                    Terrain::Desert => "desert",
                }
                .to_string()
            })
            .collect();
        let ports = st
            .board
            .ports
            .iter()
            .map(|p| {
                let k = match p.kind {
                    PortKind::Generic => "any".to_string(),
                    PortKind::Specific(r) => r.name().to_string(),
                };
                (k, p.edge)
            })
            .collect();
        Position {
            board: BoardSpec {
                terrain,
                numbers: st.board.numbers.to_vec(),
                robber: Some(st.board.robber),
                ports: Some(ports),
            },
            players: (0..st.n_players())
                .map(|pi| {
                    let owner = pi as u8;
                    PlayerPos {
                        name: format!("p{pi}"),
                        settlements: (0..N_NODES as u8)
                            .filter(|&n| {
                                matches!(st.node[n as usize], Some(pc)
                                    if pc.owner == owner && pc.building == Building::Settlement)
                            })
                            .collect(),
                        cities: (0..N_NODES as u8)
                            .filter(|&n| {
                                matches!(st.node[n as usize], Some(pc)
                                    if pc.owner == owner && pc.building == Building::City)
                            })
                            .collect(),
                        roads: (0..N_EDGES as u8)
                            .filter(|&e| st.edge[e as usize] == Some(owner))
                            .collect(),
                        hand: Resource::ALL
                            .iter()
                            .filter(|r| st.players[pi].hand[r.idx()] > 0)
                            .map(|r| (r.name().to_string(), st.players[pi].hand[r.idx()]))
                            .collect(),
                        hand_size: None,
                        knights_played: st.players[pi].knights_played,
                        dev_in_hand: st.players[pi].dev_in_hand(),
                        victory_point_cards: st.players[pi].vp_cards(),
                        longest_road: st.longest_road == Some(owner),
                        largest_army: st.largest_army == Some(owner),
                    }
                })
                .collect(),
            current: st.current,
            vp_target: st.vp_target,
            turn: st.turn,
        }
    }
}
