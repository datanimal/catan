//! Mutable game state: players, pieces on the board, and turn bookkeeping.

use crate::board::{topology, Board, PortKind};
use crate::metrics::PlayerMetrics;
use crate::types::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Initial placement, snake order.
    Setup,
    /// Start of turn: may play a dev card before rolling.
    PreRoll,
    /// A 7 was rolled; players over the hand limit are discarding.
    Discarding,
    /// Robber must be moved (from a 7 or a knight).
    MovingRobber,
    /// Build, trade, play cards.
    Main,
    Over,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerState {
    pub hand: Res,
    /// Playable development cards, by `DevCard::idx`.
    pub dev: [u8; 5],
    /// Bought this turn, not yet playable.
    pub dev_new: [u8; 5],
    pub knights_played: u8,
    pub roads_left: u8,
    pub settlements_left: u8,
    pub cities_left: u8,
    /// Bit i (<5): 2:1 port for resource i. Bit 5: a 3:1 port.
    pub port_mask: u8,
    /// Cached longest-road length, recomputed when roads change.
    pub road_len: u8,
    pub metrics: PlayerMetrics,
}

impl PlayerState {
    pub fn new() -> Self {
        PlayerState {
            hand: ZERO,
            dev: [0; 5],
            dev_new: [0; 5],
            knights_played: 0,
            roads_left: MAX_ROADS,
            settlements_left: MAX_SETTLEMENTS,
            cities_left: MAX_CITIES,
            port_mask: 0,
            road_len: 0,
            metrics: PlayerMetrics::default(),
        }
    }

    #[inline]
    pub fn hand_size(&self) -> i16 {
        res_total(&self.hand)
    }

    #[inline]
    pub fn dev_in_hand(&self) -> u8 {
        self.dev.iter().sum::<u8>() + self.dev_new.iter().sum::<u8>()
    }

    /// Victory-point cards held (these count immediately and stay hidden).
    #[inline]
    pub fn vp_cards(&self) -> u8 {
        self.dev[DevCard::VictoryPoint.idx()] + self.dev_new[DevCard::VictoryPoint.idx()]
    }

    /// Bank/port exchange rate for giving away `r`.
    #[inline]
    pub fn trade_rate(&self, r: Resource) -> u8 {
        if self.port_mask & (1 << r.idx()) != 0 {
            2
        } else if self.port_mask & (1 << 5) != 0 {
            3
        } else {
            4
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Piece {
    pub owner: u8,
    pub building: Building,
}

#[derive(Clone, Debug)]
pub struct GameState {
    pub board: Board,
    pub players: Vec<PlayerState>,
    pub node: Vec<Option<Piece>>,
    pub edge: Vec<Option<u8>>,
    pub bank: Res,
    pub dev_deck: Vec<DevCard>,

    pub current: u8,
    pub phase: Phase,
    /// Completed turns (a turn = one player's roll-to-end-turn cycle).
    pub turn: u32,
    pub last_roll: u8,

    pub largest_army: Option<u8>,
    pub longest_road: Option<u8>,
    pub vp_target: u8,
    pub hand_limit: i16,

    // --- setup ---
    pub setup_seq: Vec<u8>,
    pub setup_idx: usize,
    pub setup_need_road: bool,
    pub last_settlement: u8,

    // --- within-turn ---
    pub rolled: bool,
    pub dev_played_this_turn: bool,
    pub free_roads: u8,
    pub proposals_made: u8,
    pub max_proposals: u8,
    pub discard_queue: Vec<u8>,
    /// Set when the robber move came from a knight (no discard step).
    pub robber_from_knight: bool,

    pub winner: Option<u8>,
    /// Safety valve against pathological non-terminating games.
    pub turn_limit: u32,
}

impl GameState {
    pub fn new(board: Board, n_players: usize, vp_target: u8, turn_limit: u32) -> GameState {
        let mut setup_seq: Vec<u8> = (0..n_players as u8).collect();
        let mut rev: Vec<u8> = setup_seq.clone();
        rev.reverse();
        setup_seq.extend(rev);
        GameState {
            board,
            players: (0..n_players).map(|_| PlayerState::new()).collect(),
            node: vec![None; N_NODES],
            edge: vec![None; N_EDGES],
            bank: [BANK_PER_RES; N_RES],
            dev_deck: Vec::new(),
            current: 0,
            phase: Phase::Setup,
            turn: 0,
            last_roll: 0,
            largest_army: None,
            longest_road: None,
            vp_target,
            hand_limit: 7,
            setup_seq,
            setup_idx: 0,
            setup_need_road: false,
            last_settlement: 0,
            rolled: false,
            dev_played_this_turn: false,
            free_roads: 0,
            proposals_made: 0,
            max_proposals: 4,
            discard_queue: Vec::new(),
            robber_from_knight: false,
            winner: None,
            turn_limit,
        }
    }

    #[inline]
    pub fn n_players(&self) -> usize {
        self.players.len()
    }

    /// Whoever must act next.
    #[inline]
    pub fn actor(&self) -> u8 {
        match self.phase {
            Phase::Setup => self.setup_seq[self.setup_idx],
            Phase::Discarding => *self.discard_queue.first().unwrap_or(&self.current),
            _ => self.current,
        }
    }

    // ------------------------------------------------------------ scoring --

    /// Victory points including hidden VP cards (true score).
    pub fn vp(&self, p: u8) -> u8 {
        self.vp_public(p) + self.players[p as usize].vp_cards()
    }

    /// Victory points visible to opponents (excludes hidden VP cards).
    pub fn vp_public(&self, p: u8) -> u8 {
        let mut v = 0u8;
        for slot in self.node.iter() {
            if let Some(pc) = slot {
                if pc.owner == p {
                    v += pc.building.vp();
                }
            }
        }
        if self.largest_army == Some(p) {
            v += 2;
        }
        if self.longest_road == Some(p) {
            v += 2;
        }
        v
    }

    pub fn settlements_built(&self, p: u8) -> u8 {
        MAX_SETTLEMENTS - self.players[p as usize].settlements_left
            + (MAX_CITIES - self.players[p as usize].cities_left)
    }

    // ------------------------------------------------------- longest road --

    /// Longest continuous road for `p`. Paths may not pass *through* a node
    /// occupied by an opponent's settlement or city.
    pub fn compute_road_len(&self, p: u8) -> u8 {
        let topo = topology();
        let mine: Vec<u8> = (0..N_EDGES as u8)
            .filter(|&e| self.edge[e as usize] == Some(p))
            .collect();
        if mine.is_empty() {
            return 0;
        }
        let mut best = 0u8;
        let mut used = [false; N_EDGES];
        // Try every endpoint of every owned road as a start.
        for &e0 in &mine {
            for &start in topo.edge_nodes[e0 as usize].iter() {
                let d = self.dfs_road(p, start, &mut used, &mine);
                best = best.max(d);
            }
        }
        best
    }

    fn dfs_road(&self, p: u8, at: u8, used: &mut [bool; N_EDGES], mine: &[u8]) -> u8 {
        let topo = topology();
        // Blocked: an opponent's building on this node stops the path here.
        if let Some(pc) = self.node[at as usize] {
            if pc.owner != p {
                return 0;
            }
        }
        let mut best = 0u8;
        for &e in topo.node_edges[at as usize].iter() {
            if used[e as usize] || !mine.contains(&e) {
                continue;
            }
            let [a, b] = topo.edge_nodes[e as usize];
            let next = if a == at { b } else { a };
            used[e as usize] = true;
            let d = 1 + self.dfs_road(p, next, used, mine);
            used[e as usize] = false;
            best = best.max(d);
        }
        best
    }

    /// Recompute cached road lengths and reassign the longest-road card.
    ///
    /// A claim needs 5+ roads. The current holder keeps the card while tied
    /// for the lead; otherwise the sole leader takes it, and a tie between
    /// non-holders leaves the card set aside (the official rule).
    pub fn update_longest_road(&mut self) {
        for p in 0..self.n_players() {
            self.players[p].road_len = self.compute_road_len(p as u8);
        }
        self.longest_road = award_card(
            self.longest_road,
            &(0..self.n_players())
                .map(|p| self.players[p].road_len)
                .collect::<Vec<_>>(),
            5,
        );
    }

    /// Largest army: 3+ knights, same tie convention as longest road.
    pub fn update_largest_army(&mut self) {
        self.largest_army = award_card(
            self.largest_army,
            &(0..self.n_players())
                .map(|p| self.players[p].knights_played)
                .collect::<Vec<_>>(),
            3,
        );
    }

    // --------------------------------------------------------- production --

    /// Grant resources for a dice roll, honouring the robber and bank limits.
    /// Returns per-player gains and the cards the bank could not pay.
    pub fn produce(&mut self, roll: u8) -> (Vec<Res>, Vec<i32>) {
        let topo = topology();
        let n = self.n_players();
        let mut want = vec![ZERO; n];
        for &h in self.board.roll_hexes[roll as usize].iter() {
            if h == self.board.robber {
                continue;
            }
            let Some(r) = self.board.hex_resource(h) else {
                continue;
            };
            for &node in topo.hex_nodes[h as usize].iter() {
                if let Some(pc) = self.node[node as usize] {
                    want[pc.owner as usize][r.idx()] += pc.building.yield_count();
                }
            }
        }
        // Official bank-shortage rule: if the bank cannot satisfy everyone who
        // is owed a resource, nobody receives it -- unless exactly one player
        // is owed, who then takes what is left.
        let mut gain = vec![ZERO; n];
        for ri in 0..N_RES {
            let demand: i16 = (0..n).map(|p| want[p][ri]).sum();
            if demand == 0 {
                continue;
            }
            let claimants = (0..n).filter(|&p| want[p][ri] > 0).count();
            if demand <= self.bank[ri] {
                for p in 0..n {
                    gain[p][ri] = want[p][ri];
                }
            } else if claimants == 1 {
                let p = (0..n).find(|&p| want[p][ri] > 0).unwrap();
                gain[p][ri] = self.bank[ri];
            }
        }
        let mut short = vec![0i32; n];
        for p in 0..n {
            for ri in 0..N_RES {
                let g = gain[p][ri];
                if g > 0 {
                    self.players[p].hand[ri] += g;
                    self.bank[ri] -= g;
                }
                short[p] += (want[p][ri] - g) as i32;
            }
        }
        (gain, short)
    }

    /// Expected cards per roll for `p` from current buildings, by resource,
    /// with the robber taken into account. This is the yardstick the luck and
    /// skill measures are built on.
    pub fn expected_rate(&self, p: u8) -> [f64; N_RES] {
        let topo = topology();
        let mut out = [0.0f64; N_RES];
        for (ni, slot) in self.node.iter().enumerate() {
            let Some(pc) = slot else { continue };
            if pc.owner != p {
                continue;
            }
            let mult = pc.building.yield_count() as f64;
            for &h in topo.node_hexes[ni].iter() {
                if h == self.board.robber {
                    continue;
                }
                if let Some(r) = self.board.hex_resource(h) {
                    out[r.idx()] += mult * prob(self.board.numbers[h as usize]);
                }
            }
        }
        out
    }

    /// Total pips of a player's buildings, ignoring the robber. A measure of
    /// raw engine size: settlements count once, cities twice.
    pub fn raw_pips(&self, p: u8) -> u32 {
        self.node
            .iter()
            .enumerate()
            .filter_map(|(ni, slot)| {
                slot.filter(|pc| pc.owner == p)
                    .map(|pc| pc.building.yield_count() as u32 * self.board.node_pips(ni as u8))
            })
            .sum()
    }

    /// How many of the five resources a player's buildings can produce.
    pub fn res_breadth(&self, p: u8) -> u8 {
        let mut seen = [false; N_RES];
        for (ni, slot) in self.node.iter().enumerate() {
            if slot.map(|pc| pc.owner == p).unwrap_or(false) {
                for (i, v) in self.board.node_pips_by_res(ni as u8).iter().enumerate() {
                    if *v > 0 {
                        seen[i] = true;
                    }
                }
            }
        }
        seen.iter().filter(|&&b| b).count() as u8
    }

    /// Refresh a player's port access after building on `node`.
    pub fn grant_ports(&mut self, p: u8, node: u8) {
        let mask = self.board.node_port_mask[node as usize];
        self.players[p as usize].port_mask |= mask;
    }

    /// Ports a player can reach, for reporting.
    pub fn port_kinds(&self, p: u8) -> Vec<PortKind> {
        let m = self.players[p as usize].port_mask;
        let mut v = Vec::new();
        for r in Resource::ALL {
            if m & (1 << r.idx()) != 0 {
                v.push(PortKind::Specific(r));
            }
        }
        if m & (1 << 5) != 0 {
            v.push(PortKind::Generic);
        }
        v
    }
}

/// Shared award logic for longest road and largest army.
fn award_card(holder: Option<u8>, values: &[u8], threshold: u8) -> Option<u8> {
    let max = values.iter().copied().max().unwrap_or(0);
    if max < threshold {
        return None;
    }
    let leaders: Vec<u8> = (0..values.len() as u8)
        .filter(|&p| values[p as usize] == max)
        .collect();
    match holder {
        // The holder keeps the card while still tied for the lead.
        Some(h) if leaders.contains(&h) => Some(h),
        _ if leaders.len() == 1 => Some(leaders[0]),
        // Tie between challengers: the card is set aside.
        _ => None,
    }
}
