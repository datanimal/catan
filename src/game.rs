//! The game driver: turn order, dice, the robber, trading, and the
//! bookkeeping that feeds the luck/skill analysis.

use crate::action::Action;
use crate::board::{topology, Board, BoardLayout};
use crate::metrics::{GameRecord, PlayerResult};
use crate::policy::{default_discard, Policy, View};
use crate::rules::{legal_actions, road_spots};
use crate::state::{GameState, Phase, Piece};
use crate::types::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum DiceMode {
    /// Two independent six-sided dice.
    TwoD6,
    /// A shuffled deck of all 36 two-dice outcomes, reshuffled when empty.
    /// Gives the same long-run distribution with far less variance, which is
    /// the standard way to ask "how much of this result was the dice?".
    Deck36,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameConfig {
    pub n_players: usize,
    pub vp_target: u8,
    pub layout: BoardLayout,
    pub random_ports: bool,
    pub dice: DiceMode,
    pub turn_limit: u32,
    pub hand_limit: i16,
    /// Cap on trade offers a player may make per turn.
    pub max_proposals: u8,
    /// Switch off player-to-player trading entirely.
    pub allow_player_trades: bool,
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig {
            n_players: 4,
            vp_target: 10,
            layout: BoardLayout::RandomBalanced,
            random_ports: false,
            dice: DiceMode::TwoD6,
            turn_limit: 400,
            hand_limit: 7,
            max_proposals: 4,
            allow_player_trades: true,
        }
    }
}

pub struct Game {
    pub st: GameState,
    policies: Vec<Box<dyn Policy>>,
    names: Vec<String>,
    rng: ChaCha8Rng,
    cfg: GameConfig,
    seed: u64,
    roll_hist: [u32; 13],
    dice_deck: Vec<u8>,
}

impl Game {
    pub fn new(cfg: GameConfig, policies: Vec<Box<dyn Policy>>, seed: u64) -> Game {
        assert_eq!(policies.len(), cfg.n_players, "one policy per seat");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let board = Board::new(cfg.layout, cfg.random_ports, &mut rng);
        let mut st = GameState::new(board, cfg.n_players, cfg.vp_target, cfg.turn_limit);
        st.hand_limit = cfg.hand_limit;
        st.max_proposals = if cfg.allow_player_trades {
            cfg.max_proposals
        } else {
            0
        };
        let mut deck = DevCard::standard_deck();
        deck.shuffle(&mut rng);
        st.dev_deck = deck;
        let names = policies.iter().map(|p| p.name()).collect();
        Game {
            st,
            policies,
            names,
            rng,
            cfg,
            seed,
            roll_hist: [0; 13],
            dice_deck: Vec::new(),
        }
    }

    /// Play to completion and return the record.
    pub fn run(&mut self) -> GameRecord {
        while self.st.phase != Phase::Over {
            if self.st.turn > self.cfg.turn_limit {
                self.st.phase = Phase::Over;
                break;
            }
            self.step();
        }
        self.finish()
    }

    /// Advance the game by one decision. Public so tests can check the
    /// invariants after every single action.
    pub fn step(&mut self) {
        if self.st.phase == Phase::Discarding {
            self.do_discard();
            return;
        }
        // Road Building with nowhere left to build: drop the pending roads.
        if self.st.free_roads > 0 && road_spots(&self.st, self.st.current).is_empty() {
            self.st.free_roads = 0;
        }
        let actor = self.st.actor();
        let actions = legal_actions(&self.st);
        if actions.is_empty() {
            // Only reachable if setup has no legal road; skip ahead.
            if self.st.phase == Phase::Setup {
                self.advance_setup();
            } else {
                self.end_turn();
            }
            return;
        }
        let pick = {
            let view = View::new(&self.st, actor);
            self.policies[actor as usize]
                .choose(&view, &actions)
                .min(actions.len() - 1)
        };
        let a = actions[pick];
        self.apply(actor, a);
    }

    // ------------------------------------------------------------- dice ----

    fn roll_dice(&mut self) -> u8 {
        match self.cfg.dice {
            DiceMode::TwoD6 => {
                let a: u8 = self.rng.random_range(1..=6);
                let b: u8 = self.rng.random_range(1..=6);
                a + b
            }
            DiceMode::Deck36 => {
                if self.dice_deck.is_empty() {
                    let mut d = Vec::with_capacity(36);
                    for a in 1..=6u8 {
                        for b in 1..=6u8 {
                            d.push(a + b);
                        }
                    }
                    d.shuffle(&mut self.rng);
                    self.dice_deck = d;
                }
                self.dice_deck.pop().unwrap()
            }
        }
    }

    // ------------------------------------------------------------ actions --

    fn apply(&mut self, p: u8, a: Action) {
        match a {
            Action::PlaceInitialSettlement(n) => {
                self.place_building(p, n, Building::Settlement);
                let pips = self.st.board.node_pips(n);
                let m = &mut self.st.players[p as usize].metrics;
                m.init_nodes.push(n);
                m.init_pips += pips;
                // The second settlement pays out immediately.
                if self.st.setup_idx >= self.st.n_players() {
                    let mut gain = ZERO;
                    for &h in topology().node_hexes[n as usize].iter() {
                        if let Some(r) = self.st.board.hex_resource(h) {
                            gain[r.idx()] += 1;
                        }
                    }
                    for i in 0..N_RES {
                        let g = gain[i].min(self.st.bank[i]);
                        self.st.players[p as usize].hand[i] += g;
                        self.st.bank[i] -= g;
                        self.st.players[p as usize].metrics.setup_income += g as i32;
                    }
                }
                self.st.setup_need_road = true;
                self.st.last_settlement = n;
            }

            Action::PlaceInitialRoad(e) => {
                self.st.edge[e as usize] = Some(p);
                self.st.players[p as usize].roads_left -= 1;
                self.advance_setup();
            }

            Action::Roll => {
                let roll = self.roll_dice();
                self.st.last_roll = roll;
                self.st.rolled = true;
                self.roll_hist[roll as usize] += 1;

                // Every player's expectation accrues on every roll -- this is
                // the baseline the dice are measured against.
                for q in 0..self.st.n_players() {
                    let rate = self.st.expected_rate(q as u8);
                    let m = &mut self.st.players[q].metrics;
                    for i in 0..N_RES {
                        m.expected_cards[i] += rate[i];
                    }
                    m.rolls_seen += 1;
                }
                {
                    let m = &mut self.st.players[p as usize].metrics;
                    m.dice_sum += roll as u32;
                    if roll == 7 {
                        m.sevens_rolled += 1;
                    }
                }

                if roll == 7 {
                    self.st.robber_from_knight = false;
                    self.begin_discards();
                } else {
                    // Cards the robber denied, before the bank is consulted.
                    let rh = self.st.board.robber;
                    if self.st.board.numbers[rh as usize] == roll {
                        let mut blocked = vec![0i32; self.st.n_players()];
                        for &n in topology().hex_nodes[rh as usize].iter() {
                            if let Some(pc) = self.st.node[n as usize] {
                                blocked[pc.owner as usize] += pc.building.yield_count() as i32;
                            }
                        }
                        for q in 0..self.st.n_players() {
                            self.st.players[q].metrics.robber_blocked += blocked[q];
                        }
                    }
                    let (gains, shortfall) = self.st.produce(roll);
                    for q in 0..self.st.n_players() {
                        let m = &mut self.st.players[q].metrics;
                        for i in 0..N_RES {
                            m.produced[i] += gains[q][i] as i32;
                        }
                        m.bank_shortage_loss += shortfall[q];
                    }
                    self.st.phase = Phase::Main;
                }
            }

            Action::MoveRobber { hex, victim } => {
                self.st.board.robber = hex;
                if let Some(v) = victim {
                    if let Some(r) = self.steal_random(v) {
                        self.st.players[p as usize].hand[r.idx()] += 1;
                        self.st.players[p as usize].metrics.stolen_from_others += 1;
                        self.st.players[v as usize].metrics.lost_to_steal += 1;
                        self.st.players[v as usize].metrics.times_robbed += 1;
                    }
                }
                // A knight played before rolling returns to the pre-roll step.
                self.st.phase = if self.st.rolled {
                    Phase::Main
                } else {
                    Phase::PreRoll
                };
            }

            Action::BuildRoad(e) => {
                if self.st.free_roads > 0 {
                    self.st.free_roads -= 1;
                } else {
                    self.pay(p, &COST_ROAD);
                }
                self.st.edge[e as usize] = Some(p);
                self.st.players[p as usize].roads_left -= 1;
                self.st.players[p as usize].metrics.roads_built += 1;
                self.st.update_longest_road();
            }

            Action::BuildSettlement(n) => {
                self.pay(p, &COST_SETTLEMENT);
                self.place_building(p, n, Building::Settlement);
                self.st.players[p as usize].metrics.settlements_built += 1;
                // A new settlement can cut an opponent's road in two.
                self.st.update_longest_road();
            }

            Action::BuildCity(n) => {
                self.pay(p, &COST_CITY);
                self.st.node[n as usize] = Some(Piece {
                    owner: p,
                    building: Building::City,
                });
                self.st.players[p as usize].cities_left -= 1;
                // The settlement piece goes back to the player's supply.
                self.st.players[p as usize].settlements_left += 1;
                self.st.players[p as usize].metrics.cities_built += 1;
            }

            Action::BuyDev => {
                self.pay(p, &COST_DEV);
                let card = self.st.dev_deck.pop().expect("deck checked non-empty");
                self.st.players[p as usize].metrics.dev_bought += 1;
                if card == DevCard::VictoryPoint {
                    // Victory point cards count the moment they are drawn.
                    self.st.players[p as usize].dev[card.idx()] += 1;
                } else {
                    self.st.players[p as usize].dev_new[card.idx()] += 1;
                }
            }

            Action::PlayKnight => {
                self.spend_dev(p, DevCard::Knight);
                self.st.players[p as usize].knights_played += 1;
                self.st.players[p as usize].metrics.knights_played += 1;
                self.st.update_largest_army();
                self.st.robber_from_knight = true;
                self.st.phase = Phase::MovingRobber;
            }

            Action::PlayRoadBuilding => {
                self.spend_dev(p, DevCard::RoadBuilding);
                self.st.players[p as usize].metrics.road_buildings_played += 1;
                self.st.free_roads = 2.min(self.st.players[p as usize].roads_left);
            }

            Action::PlayYearOfPlenty(a1, b1) => {
                self.spend_dev(p, DevCard::YearOfPlenty);
                self.st.players[p as usize].metrics.yops_played += 1;
                for r in [a1, b1] {
                    if self.st.bank[r.idx()] > 0 {
                        self.st.bank[r.idx()] -= 1;
                        self.st.players[p as usize].hand[r.idx()] += 1;
                        self.st.players[p as usize].metrics.year_of_plenty_gain += 1;
                    }
                }
            }

            Action::PlayMonopoly(r) => {
                self.spend_dev(p, DevCard::Monopoly);
                self.st.players[p as usize].metrics.monopolies_played += 1;
                let mut take = 0i16;
                for q in 0..self.st.n_players() {
                    if q as u8 == p {
                        continue;
                    }
                    let n = self.st.players[q].hand[r.idx()];
                    if n > 0 {
                        self.st.players[q].hand[r.idx()] = 0;
                        self.st.players[q].metrics.monopoly_loss += n as i32;
                        take += n;
                    }
                }
                self.st.players[p as usize].hand[r.idx()] += take;
                self.st.players[p as usize].metrics.monopoly_gain += take as i32;
            }

            Action::BankTrade { give, want, rate } => {
                let r = rate as i16;
                self.st.players[p as usize].hand[give.idx()] -= r;
                self.st.bank[give.idx()] += r;
                self.st.bank[want.idx()] -= 1;
                self.st.players[p as usize].hand[want.idx()] += 1;
                let m = &mut self.st.players[p as usize].metrics;
                m.bank_trade_out += r as i32;
                m.bank_trade_in += 1;
                m.bank_trades += 1;
            }

            Action::ProposeTrade { give, want } => {
                self.st.proposals_made += 1;
                self.st.players[p as usize].metrics.trades_proposed += 1;
                self.resolve_trade(p, give, want);
            }

            Action::EndTurn => self.end_turn(),
        }

        self.check_win(p);
    }

    // ------------------------------------------------------------ helpers --

    fn place_building(&mut self, p: u8, n: u8, b: Building) {
        self.st.node[n as usize] = Some(Piece {
            owner: p,
            building: b,
        });
        self.st.players[p as usize].settlements_left -= 1;
        self.st.grant_ports(p, n);
    }

    fn pay(&mut self, p: u8, cost: &Res) {
        for i in 0..N_RES {
            self.st.players[p as usize].hand[i] -= cost[i];
            self.st.bank[i] += cost[i];
            self.st.players[p as usize].metrics.spent[i] += cost[i] as i32;
        }
    }

    fn spend_dev(&mut self, p: u8, c: DevCard) {
        self.st.players[p as usize].dev[c.idx()] -= 1;
        self.st.dev_played_this_turn = true;
    }

    fn steal_random(&mut self, victim: u8) -> Option<Resource> {
        let hand = self.st.players[victim as usize].hand;
        let total = res_total(&hand);
        if total <= 0 {
            return None;
        }
        let mut k = self.rng.random_range(0..total);
        for i in 0..N_RES {
            if k < hand[i] {
                self.st.players[victim as usize].hand[i] -= 1;
                return Some(Resource::from_idx(i));
            }
            k -= hand[i];
        }
        None
    }

    /// Record each player's opening position once setup is complete and
    /// before any die is rolled, so these regressors are untouched by play.
    fn record_openings(&mut self) {
        use crate::board::PortKind;
        for p in 0..self.st.n_players() {
            let nodes = self.st.players[p].metrics.init_nodes.clone();
            let b = &self.st.board;
            let mut f = crate::metrics::OpeningFeatures::default();
            for &n in &nodes {
                let by = b.node_pips_by_res(n);
                for i in 0..N_RES {
                    f.pips[i] += by[i];
                }
                let m = b.node_port_mask[n as usize];
                for r in Resource::ALL {
                    if m & (1 << r.idx()) != 0 {
                        f.port2[r.idx()] = true;
                    }
                }
                if m & (1 << 5) != 0 {
                    f.port3 = true;
                }
                for &h in topology().node_hexes[n as usize].iter() {
                    if b.numbers[h as usize] > 0 {
                        f.numbers.push(b.numbers[h as usize]);
                    }
                }
            }
            f.total_pips = f.pips.iter().sum();
            f.breadth = f.pips.iter().filter(|&&v| v > 0).count() as u8;
            f.paired_road = b.paired_pips(&nodes, Resource::Brick, Resource::Lumber);
            f.paired_city = b.paired_pips(&nodes, Resource::Grain, Resource::Ore);
            f.paired_any = {
                let t = b.pips_by_res_number(&nodes);
                (2..=12)
                    .map(|n| {
                        let present: u32 = (0..N_RES).filter(|&i| t[i][n] > 0).count() as u32;
                        // Pips that share a number with a different resource.
                        if present >= 2 {
                            (0..N_RES).map(|i| t[i][n]).sum()
                        } else {
                            0
                        }
                    })
                    .sum()
            };
            f.port_dist = b.port_distance(&nodes, |m| m != 0).unwrap_or(99);
            f.port3_dist = b.port_distance(&nodes, |m| m & (1 << 5) != 0).unwrap_or(99);
            for r in Resource::ALL {
                f.port2_dist[r.idx()] = b
                    .port_distance(&nodes, |m| m & (1 << r.idx()) != 0)
                    .unwrap_or(99);
            }
            let _ = PortKind::Generic;
            self.st.players[p].metrics.opening = f;
        }
    }

    fn advance_setup(&mut self) {
        self.st.setup_need_road = false;
        self.st.setup_idx += 1;
        if self.st.setup_idx >= self.st.setup_seq.len() {
            self.record_openings();
            self.st.phase = Phase::PreRoll;
            self.st.current = 0;
            self.st.turn = 1;
            self.st.update_longest_road();
            self.note_turn_start();
        }
    }

    fn begin_discards(&mut self) {
        let limit = self.st.hand_limit;
        let mut q: Vec<u8> = Vec::new();
        // Clockwise from the current player.
        let n = self.st.n_players();
        for k in 0..n {
            let q_i = ((self.st.current as usize + k) % n) as u8;
            if self.st.players[q_i as usize].hand_size() > limit {
                q.push(q_i);
            }
        }
        if q.is_empty() {
            self.st.phase = Phase::MovingRobber;
        } else {
            self.st.discard_queue = q;
            self.st.phase = Phase::Discarding;
        }
    }

    fn do_discard(&mut self) {
        let p = self.st.discard_queue[0];
        let size = self.st.players[p as usize].hand_size();
        let n = size / 2;
        let mut chosen = {
            let view = View::new(&self.st, p);
            self.policies[p as usize].discard(&view, n)
        };
        // Validate: exactly n cards, all actually held.
        let hand = self.st.players[p as usize].hand;
        let ok = res_total(&chosen) == n && (0..N_RES).all(|i| chosen[i] >= 0 && chosen[i] <= hand[i]);
        if !ok {
            chosen = default_discard(&hand, n);
        }
        for i in 0..N_RES {
            self.st.players[p as usize].hand[i] -= chosen[i];
            self.st.bank[i] += chosen[i];
        }
        let m = &mut self.st.players[p as usize].metrics;
        m.lost_to_discard += n as i32;
        m.discards += 1;

        self.st.discard_queue.remove(0);
        if self.st.discard_queue.is_empty() {
            self.st.phase = Phase::MovingRobber;
        }
    }

    /// Offer `give` for `want` to each other player in clockwise order; the
    /// first who accepts and can pay completes the trade.
    fn resolve_trade(&mut self, from: u8, give: Res, want: Res) {
        let n = self.st.n_players();
        for k in 1..n {
            let q = ((from as usize + k) % n) as u8;
            // The responder must actually hold what is being asked of them.
            let their_hand = self.st.players[q as usize].hand;
            if !(0..N_RES).all(|i| their_hand[i] >= want[i]) {
                continue;
            }
            let accept = {
                let view = View::new(&self.st, q);
                self.policies[q as usize].accept_trade(&view, &give, &want, from)
            };
            if !accept {
                continue;
            }
            for i in 0..N_RES {
                self.st.players[from as usize].hand[i] -= give[i];
                self.st.players[q as usize].hand[i] += give[i];
                self.st.players[q as usize].hand[i] -= want[i];
                self.st.players[from as usize].hand[i] += want[i];
            }
            let g = res_total(&give) as i32;
            let w = res_total(&want) as i32;
            {
                let m = &mut self.st.players[from as usize].metrics;
                m.player_trade_out += g;
                m.player_trade_in += w;
                m.trades_accepted_by_others += 1;
            }
            {
                let m = &mut self.st.players[q as usize].metrics;
                m.player_trade_in += g;
                m.player_trade_out += w;
                m.trades_accepted_for_others += 1;
            }
            return;
        }
    }

    fn note_turn_start(&mut self) {
        let p = self.st.current;
        let pips = self.st.raw_pips(p) as f64;
        let turn = self.st.turn;
        let vp = self.st.vp(p);
        let m = &mut self.st.players[p as usize].metrics;
        m.turns += 1;
        m.pip_turns += pips;
        m.note_vp(vp, turn);
    }

    fn end_turn(&mut self) {
        let p = self.st.current;
        // Cards bought this turn become playable.
        for i in 0..5 {
            self.st.players[p as usize].dev[i] += self.st.players[p as usize].dev_new[i];
            self.st.players[p as usize].dev_new[i] = 0;
        }
        self.st.rolled = false;
        self.st.dev_played_this_turn = false;
        self.st.free_roads = 0;
        self.st.proposals_made = 0;
        self.st.current = ((p as usize + 1) % self.st.n_players()) as u8;
        self.st.turn += 1;
        self.st.phase = Phase::PreRoll;
        if self.st.turn <= self.cfg.turn_limit {
            self.note_turn_start();
        }
    }

    fn check_win(&mut self, p: u8) {
        if self.st.phase == Phase::Over {
            return;
        }
        if self.st.vp(p) >= self.st.vp_target {
            self.st.winner = Some(p);
            self.st.phase = Phase::Over;
        }
    }

    // ------------------------------------------------------------- record --

    fn finish(&mut self) -> GameRecord {
        self.st.update_longest_road();
        self.st.update_largest_army();
        let winner = self.st.winner;
        let mut players = Vec::with_capacity(self.st.n_players());
        for p in 0..self.st.n_players() as u8 {
            let vp = self.st.vp(p);
            let pips = self.st.raw_pips(p);
            let breadth = self.st.res_breadth(p);
            let lr = self.st.longest_road == Some(p);
            let la = self.st.largest_army == Some(p);
            let vpc = self.st.players[p as usize].vp_cards();
            let pm = self.st.players[p as usize].port_mask;
            let m = &mut self.st.players[p as usize].metrics;
            m.final_port_mask = pm;
            m.vp_final = vp;
            m.vp_from_cards = vpc;
            m.final_pips = pips;
            m.final_res_breadth = breadth;
            m.had_longest_road = lr;
            m.had_largest_army = la;
            players.push(PlayerResult {
                seat: p,
                policy: self.names[p as usize].clone(),
                won: winner == Some(p),
                vp,
                metrics: self.st.players[p as usize].metrics.clone(),
            });
        }
        GameRecord {
            seed: self.seed,
            turns: self.st.turn,
            winner,
            timed_out: winner.is_none(),
            players,
            roll_histogram: self.roll_hist,
        }
    }
}
