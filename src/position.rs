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
    pub ports: Option<Vec<(String, Ref)>>,
}

/// An intersection, written either as a raw index or, far more usefully for
/// transcribing from a picture, as `"<hex>:<corner>"` such as `"7:N"`.
///
/// Corners are N, NE, SE, S, SW, NW and sides are NE, E, SE, SW, W, NW, with
/// hexes numbered 0-18 in rows of 3/4/5/4/3 from the top. Because neighbouring
/// hexes share corners, the same intersection has up to three names; any of
/// them is accepted, and `catan_sim key` prints the board with its hex numbers
/// so a screenshot can be read off directly.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Ref {
    Index(u8),
    Named(String),
}

impl Ref {
    pub fn node(&self) -> Result<u8, String> {
        match self {
            Ref::Index(i) if (*i as usize) < N_NODES => Ok(*i),
            Ref::Index(i) => Err(format!("intersection {i} does not exist")),
            Ref::Named(s) => {
                let (h, c) = s
                    .split_once(':')
                    .ok_or_else(|| format!("'{s}' is not <hex>:<corner>, e.g. 7:N"))?;
                let hex: u8 = h
                    .trim()
                    .parse()
                    .map_err(|_| format!("'{h}' is not a hex number"))?;
                if hex as usize >= N_HEXES {
                    return Err(format!(
                        "'{s}' names hex {hex}, but the board has hexes 0-{}",
                        N_HEXES - 1
                    ));
                }
                crate::board::corner_node(hex, c.trim()).ok_or_else(|| {
                    format!("'{s}' is not a corner; use one of {:?}", crate::board::CORNER_NAMES)
                })
            }
        }
    }
    pub fn edge(&self) -> Result<u8, String> {
        match self {
            Ref::Index(i) if (*i as usize) < N_EDGES => Ok(*i),
            Ref::Index(i) => Err(format!("path {i} does not exist")),
            Ref::Named(s) => {
                let (h, c) = s
                    .split_once(':')
                    .ok_or_else(|| format!("'{s}' is not <hex>:<side>, e.g. 7:NW"))?;
                let hex: u8 = h
                    .trim()
                    .parse()
                    .map_err(|_| format!("'{h}' is not a hex number"))?;
                if hex as usize >= N_HEXES {
                    return Err(format!(
                        "'{s}' names hex {hex}, but the board has hexes 0-{}",
                        N_HEXES - 1
                    ));
                }
                crate::board::side_edge(hex, c.trim()).ok_or_else(|| {
                    format!("'{s}' is not a side; use one of {:?}", crate::board::SIDE_NAMES)
                })
            }
        }
    }
}

fn resolve_nodes(v: &[Ref]) -> Result<Vec<u8>, String> {
    v.iter().map(|r| r.node()).collect()
}
fn resolve_edges(v: &[Ref]) -> Result<Vec<u8>, String> {
    v.iter().map(|r| r.edge()).collect()
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlayerPos {
    #[serde(default)]
    pub name: String,
    /// Intersections holding a settlement.
    #[serde(default)]
    pub settlements: Vec<Ref>,
    /// Intersections holding a city.
    #[serde(default)]
    pub cities: Vec<Ref>,
    /// Paths holding a road.
    #[serde(default)]
    pub roads: Vec<Ref>,
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

/// The name a node is written out as: the lowest-numbered hex that touches it.
fn canonical_node(n: u8) -> String {
    crate::board::node_names(n)
        .into_iter()
        .next()
        .unwrap_or_else(|| n.to_string())
}

fn canonical_edge(e: u8) -> String {
    crate::board::edge_names(e)
        .into_iter()
        .next()
        .unwrap_or_else(|| e.to_string())
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
            for (kind, edge_ref) in ports {
                let edge = edge_ref.edge()?;
                let k = if kind.eq_ignore_ascii_case("any") || kind == "3:1" {
                    PortKind::Generic
                } else {
                    PortKind::Specific(parse_resource(kind)?)
                };
                let bit = match k {
                    PortKind::Generic => 1u8 << 5,
                    PortKind::Specific(r) => 1u8 << r.idx(),
                };
                for &n in topo.edge_nodes[edge as usize].iter() {
                    board.node_port_mask[n as usize] |= bit;
                }
                board.ports.push(crate::board::Port { kind: k, edge });
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
            let settlements = resolve_nodes(&p.settlements)
                .map_err(|e| format!("player {pi}: {e}"))?;
            let cities = resolve_nodes(&p.cities).map_err(|e| format!("player {pi}: {e}"))?;
            let roads = resolve_edges(&p.roads).map_err(|e| format!("player {pi}: {e}"))?;
            for &n in &settlements {
                if st.node[n as usize].is_some() {
                    return Err(format!("two buildings on intersection {n}"));
                }
                st.node[n as usize] = Some(Piece {
                    owner,
                    building: Building::Settlement,
                });
            }
            for &n in &cities {
                if st.node[n as usize].is_some() {
                    return Err(format!("two buildings on intersection {n}"));
                }
                st.node[n as usize] = Some(Piece {
                    owner,
                    building: Building::City,
                });
            }
            for &e in &roads {
                if st.edge[e as usize].is_some() {
                    return Err(format!("two roads on path {e}"));
                }
                st.edge[e as usize] = Some(owner);
            }
            let pl = &mut st.players[pi];
            pl.settlements_left = MAX_SETTLEMENTS
                .checked_sub(settlements.len() as u8)
                .ok_or_else(|| format!("player {pi} has more than {MAX_SETTLEMENTS} settlements"))?;
            pl.cities_left = MAX_CITIES
                .checked_sub(cities.len() as u8)
                .ok_or_else(|| format!("player {pi} has more than {MAX_CITIES} cities"))?;
            pl.roads_left = MAX_ROADS
                .checked_sub(roads.len() as u8)
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

    /// Check the position against the standard board's fixed recipe.
    ///
    /// This is the safety net for a transcribed board. A standard game has
    /// exactly 4 forest, 4 pasture, 4 fields, 3 hills, 3 mountains and 1
    /// desert, and one specific multiset of number tokens. Misreading a 6 as
    /// an 8, or a pasture as a field, breaks the recipe and is caught here --
    /// which matters because `to_state` only rejects positions that are
    /// *illegal*, and a misreading is usually perfectly legal and simply
    /// wrong. A confident number computed from the wrong board is the worst
    /// outcome, so these are reported loudly rather than silently tolerated.
    ///
    /// They are warnings, not errors: variants and expansions legitimately
    /// use other mixes.
    pub fn check_composition(&self) -> Vec<String> {
        let mut out = Vec::new();
        let parsed: Vec<Terrain> = self
            .board
            .terrain
            .iter()
            .filter_map(|t| parse_terrain(t).ok())
            .collect();
        if parsed.len() == N_HEXES {
            let count = |t: Terrain| parsed.iter().filter(|&&x| x == t).count();
            for (t, want, name) in [
                (Terrain::Forest, 4, "forest"),
                (Terrain::Pasture, 4, "pasture"),
                (Terrain::Fields, 4, "fields"),
                (Terrain::Hills, 3, "hills"),
                (Terrain::Mountains, 3, "mountains"),
                (Terrain::Desert, 1, "desert"),
            ] {
                let got = count(t);
                if got != want {
                    out.push(format!(
                        "{got} {name} hexes, but a standard board has {want} -- check the board \
                         was read correctly"
                    ));
                }
            }
        }
        let mut nums: Vec<u8> = self.board.numbers.iter().copied().filter(|&n| n > 0).collect();
        nums.sort_unstable();
        let want = standard_numbers();
        if nums != want {
            // Say precisely which tokens are wrong, since that points at the
            // hex that was misread.
            let mut missing = want.clone();
            for n in &nums {
                if let Some(i) = missing.iter().position(|m| m == n) {
                    missing.remove(i);
                }
            }
            let mut extra = nums.clone();
            for n in &want {
                if let Some(i) = extra.iter().position(|m| m == n) {
                    extra.remove(i);
                }
            }
            out.push(format!(
                "number tokens do not match a standard board: {} too many, {} missing",
                if extra.is_empty() { "none".into() } else { format!("{extra:?}") },
                if missing.is_empty() { "none".into() } else { format!("{missing:?}") },
            ));
        }
        if let Some(ports) = &self.board.ports {
            if ports.len() != 9 {
                out.push(format!("{} ports, but a standard board has 9", ports.len()));
            }
        }
        // A settlement named by two different hexes that disagree would have
        // been caught on resolution; a duplicated intersection would not.
        let mut all = Vec::new();
        for p in &self.players {
            for r in p.settlements.iter().chain(p.cities.iter()) {
                if let Ok(n) = r.node() {
                    all.push(n);
                }
            }
        }
        let before = all.len();
        all.sort_unstable();
        all.dedup();
        if all.len() != before {
            out.push("the same intersection is listed more than once".into());
        }
        out
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
                (k, Ref::Named(canonical_edge(p.edge)))
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
                            .map(|n| Ref::Named(canonical_node(n)))
                            .collect(),
                        cities: (0..N_NODES as u8)
                            .filter(|&n| {
                                matches!(st.node[n as usize], Some(pc)
                                    if pc.owner == owner && pc.building == Building::City)
                            })
                            .map(|n| Ref::Named(canonical_node(n)))
                            .collect(),
                        roads: (0..N_EDGES as u8)
                            .filter(|&e| st.edge[e as usize] == Some(owner))
                            .map(|e| Ref::Named(canonical_edge(e)))
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
