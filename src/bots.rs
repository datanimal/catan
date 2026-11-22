//! Strategy archetypes.
//!
//! All non-random bots share one scoring engine: every legal action is given
//! a score and the best is taken. What makes an archetype is its [`Weights`] —
//! how much it values cities versus roads versus development cards, which
//! resources it prizes when placing, and how it trades. Adding a strategy
//! therefore means adding a weight preset, not new engine code, which keeps
//! archetype comparisons honest: they differ only in preferences.
//!
//! Scores are on a loose scale where `EndTurn` is 0 and a good city is ~70.

use crate::action::Action;
use crate::board::PortKind;
use crate::policy::{Policy, View};
use crate::types::*;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Debug)]
pub struct Weights {
    pub label: String,
    /// Base appetite for each build.
    pub w_city: f64,
    pub w_settlement: f64,
    pub w_road: f64,
    pub w_dev: f64,
    /// Placement: value of one pip of each resource.
    pub res_pref: [f64; N_RES],
    /// Bonus per distinct resource a node touches.
    pub diversity: f64,
    /// Bonus for a node with port access.
    pub port: f64,
    /// Appetite for playing knights.
    pub knight_eagerness: f64,
    /// How hard to chase the longest-road card.
    pub longest_road_drive: f64,
    /// Willingness to convert at the bank or a port.
    pub bank_trade_drive: f64,
    /// Willingness to offer trades to other players.
    pub propose_drive: f64,
    /// Net benefit an incoming offer must clear to be accepted.
    pub accept_floor: f64,
    /// If false, every incoming offer is refused.
    pub accepts_trades: bool,
    /// Reluctance to hand resources to the player in the lead.
    pub leader_wariness: f64,
    /// Random jitter added to every score, to break ties and decorrelate.
    pub noise: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Weights {
            label: "balanced".into(),
            w_city: 55.0,
            w_settlement: 50.0,
            w_road: 8.0,
            w_dev: 18.0,
            res_pref: [1.0; N_RES],
            diversity: 2.5,
            port: 3.0,
            knight_eagerness: 8.0,
            longest_road_drive: 1.0,
            bank_trade_drive: 1.0,
            propose_drive: 1.0,
            accept_floor: 0.5,
            accepts_trades: true,
            leader_wariness: 3.0,
            noise: 0.75,
        }
    }
}

impl Weights {
    pub fn preset(name: &str) -> Option<Weights> {
        use Resource::*;
        let mut w = Weights::default();
        match name {
            "balanced" => {}
            // Ore and grain, straight to cities. Few roads.
            "city_builder" => {
                w.label = "city_builder".into();
                w.res_pref[Ore.idx()] = 1.7;
                w.res_pref[Grain.idx()] = 1.6;
                w.res_pref[Brick.idx()] = 0.7;
                w.res_pref[Lumber.idx()] = 0.8;
                w.w_city = 80.0;
                w.w_settlement = 45.0;
                w.w_road = 4.0;
                w.w_dev = 14.0;
                w.longest_road_drive = 0.2;
            }
            // Buy development cards relentlessly; largest army plus VP cards.
            "dev_rush" => {
                w.label = "dev_rush".into();
                w.res_pref[Ore.idx()] = 1.6;
                w.res_pref[Grain.idx()] = 1.4;
                w.res_pref[Wool.idx()] = 1.5;
                w.res_pref[Brick.idx()] = 0.6;
                w.res_pref[Lumber.idx()] = 0.6;
                w.w_dev = 52.0;
                w.w_city = 50.0;
                w.w_settlement = 44.0;
                w.w_road = 3.0;
                w.knight_eagerness = 22.0;
                w.longest_road_drive = 0.1;
            }
            // Wood and brick, sprawl outward, take the longest road.
            "expander" => {
                w.label = "expander".into();
                w.res_pref[Lumber.idx()] = 1.6;
                w.res_pref[Brick.idx()] = 1.6;
                w.res_pref[Ore.idx()] = 0.6;
                w.w_settlement = 62.0;
                w.w_city = 42.0;
                w.w_road = 18.0;
                w.w_dev = 10.0;
                w.longest_road_drive = 6.0;
            }
            // Leans on ports and on trading with everyone.
            "trader" => {
                w.label = "trader".into();
                w.port = 9.0;
                w.bank_trade_drive = 2.0;
                w.propose_drive = 2.5;
                w.accept_floor = -0.5;
                w.leader_wariness = 1.0;
            }
            // Identical to balanced, but never trades with a human opponent.
            // The control case for "does trading actually matter?".
            "hermit" => {
                w.label = "hermit".into();
                w.accepts_trades = false;
                w.propose_drive = 0.0;
            }
            _ => return None,
        }
        Some(w)
    }

    pub fn all_presets() -> Vec<&'static str> {
        vec![
            "balanced",
            "city_builder",
            "dev_rush",
            "expander",
            "trader",
            "hermit",
        ]
    }
}

// ---------------------------------------------------------------- random ----

/// Uniform over legal actions. The null hypothesis for every experiment.
pub struct RandomBot {
    rng: ChaCha8Rng,
}

impl RandomBot {
    pub fn new(seed: u64) -> Self {
        RandomBot {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Policy for RandomBot {
    fn name(&self) -> String {
        "random".into()
    }
    fn choose(&mut self, _v: &View, actions: &[Action]) -> usize {
        self.rng.random_range(0..actions.len())
    }
    fn accept_trade(&mut self, _v: &View, _to_me: &Res, _of_me: &Res, _from: u8) -> bool {
        self.rng.random_bool(0.5)
    }
}

// ------------------------------------------------------------- heuristic ----

pub struct HeuristicBot {
    w: Weights,
    rng: ChaCha8Rng,
}

impl HeuristicBot {
    pub fn new(w: Weights, seed: u64) -> Self {
        HeuristicBot {
            w,
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Positional value of a node for a settlement.
    fn node_value(&self, v: &View, n: u8) -> f64 {
        let b = v.board();
        let by_res = b.node_pips_by_res(n);
        let mut s = 0.0;
        let mut distinct = 0.0;
        for i in 0..N_RES {
            if by_res[i] > 0 {
                distinct += 1.0;
            }
            s += by_res[i] as f64 * self.w.res_pref[i];
        }
        s += self.w.diversity * distinct;
        if b.node_port_mask[n as usize] != 0 {
            s += self.w.port;
        }
        s
    }

    /// Extra value for a node that covers resources we barely produce.
    fn complement_bonus(&self, v: &View, n: u8) -> f64 {
        let rate = v.my_rate();
        let by_res = v.board().node_pips_by_res(n);
        let mut bonus = 0.0;
        for i in 0..N_RES {
            if by_res[i] > 0 && rate[i] < 0.08 {
                bonus += 4.0;
            }
        }
        bonus
    }

    /// Which build are we saving for? Drives trade and conversion decisions.
    fn target_cost(&self, v: &View) -> Res {
        let can_city = v.cities_left() > 0 && !v.city_spots().is_empty();
        let can_settle = v.settlements_left() > 0 && !v.settlement_spots().is_empty();
        let mut best = (f64::MIN, COST_DEV);
        if can_city {
            best = best.max_by(self.w.w_city, COST_CITY);
        }
        if can_settle {
            best = best.max_by(self.w.w_settlement, COST_SETTLEMENT);
        }
        if v.settlements_left() > 0 && !can_settle && v.roads_left() > 0 {
            // Boxed in: roads are the way out.
            best = best.max_by(self.w.w_road + 40.0, COST_ROAD);
        }
        if v.dev_deck_size() > 0 {
            best = best.max_by(self.w.w_dev, COST_DEV);
        }
        best.1
    }

    /// How many cards short of `cost` we are.
    fn deficit(hand: &Res, cost: &Res) -> i16 {
        (0..N_RES)
            .map(|i| (cost[i] - hand[i]).max(0))
            .sum()
    }

    /// Value of a road: it opens settlement spots and extends our longest run.
    fn road_value(&self, v: &View, e: u8) -> f64 {
        let topo = crate::board::topology();
        let mut best = 0.0f64;
        for &n in topo.edge_nodes[e as usize].iter() {
            if v.node(n).is_some() {
                continue;
            }
            if v.node_open(n) {
                // Settleable as soon as the road is down.
                best = best.max(0.55 * self.node_value(v, n));
            } else {
                // Look one step further out.
                for &nb in topo.node_neighbors[n as usize].iter() {
                    if v.node_open(nb) {
                        best = best.max(0.22 * self.node_value(v, nb));
                    }
                }
            }
        }
        // Longest road: worth 2 VP, so chasing it can be worth a lot.
        let mine = v.road_len(v.me());
        let rival = (0..v.n_players() as u8)
            .filter(|&q| q != v.me())
            .map(|q| v.road_len(q))
            .max()
            .unwrap_or(0);
        let holder = v.longest_road();
        if holder != Some(v.me()) && mine + 1 >= 5 && mine + 1 > rival.max(4) {
            best += 25.0 * self.w.longest_road_drive.min(2.0);
        } else if holder != Some(v.me()) {
            best += self.w.longest_road_drive * (mine as f64 * 0.5);
        }
        best
    }

    /// Who is winning, by public score.
    fn leader(&self, v: &View) -> u8 {
        (0..v.n_players() as u8)
            .max_by_key(|&q| {
                (
                    if q == v.me() {
                        v.my_vp()
                    } else {
                        v.vp_public(q)
                    },
                    v.hand_size(q),
                )
            })
            .unwrap_or(0)
    }

    fn robber_score(&self, v: &View, hex: u8, victim: Option<u8>) -> f64 {
        let topo = crate::board::topology();
        let mut s = 0.0;
        let mut hits_me = false;
        for &n in topo.hex_nodes[hex as usize].iter() {
            if let Some(pc) = v.node(n) {
                if pc.owner == v.me() {
                    hits_me = true;
                } else {
                    let vp = v.vp_public(pc.owner) as f64;
                    s += pc.building.yield_count() as f64
                        * v.board().node_pips(n) as f64
                        * 0.4
                        + vp * 2.0;
                }
            }
        }
        if hits_me {
            s -= 150.0;
        }
        // Prefer a victim who is both rich and winning.
        if let Some(vic) = victim {
            s += v.hand_size(vic) as f64 * 1.2 + v.vp_public(vic) as f64 * 2.0;
            if vic == self.leader(v) {
                s += 6.0;
            }
        }
        s
    }

    /// Rough guess at how many of `r` the opponents hold, from public info:
    /// their hand sizes, split by their share of production of each resource.
    fn estimate_opponent_holdings(&self, v: &View, r: Resource) -> f64 {
        let mut total = 0.0;
        for q in 0..v.n_players() as u8 {
            if q == v.me() {
                continue;
            }
            let rate = v.rate_of(q);
            let sum: f64 = rate.iter().sum();
            if sum <= 0.0 {
                continue;
            }
            total += v.hand_size(q) as f64 * (rate[r.idx()] / sum);
        }
        total
    }

    fn score(&mut self, v: &View, a: &Action) -> f64 {
        let hand = *v.hand();
        let target = self.target_cost(v);
        let def_now = Self::deficit(&hand, &target);

        let base = match *a {
            Action::EndTurn => 0.0,
            Action::Roll => 1000.0,

            Action::PlaceInitialSettlement(n) => {
                100.0 + self.node_value(v, n) + self.complement_bonus(v, n)
            }
            Action::PlaceInitialRoad(e) => 100.0 + self.road_value(v, e),

            Action::BuildCity(n) => {
                // A city doubles that node's yield.
                self.w.w_city + 2.2 * v.board().node_pips(n) as f64
            }
            Action::BuildSettlement(n) => self.w.w_settlement + 1.3 * self.node_value(v, n),
            Action::BuildRoad(e) => {
                let r = self.w.w_road + self.road_value(v, e);
                // Free roads from Road Building should always be used.
                if v.free_roads() > 0 { r + 500.0 } else { r }
            }
            Action::BuyDev => {
                let mut s = self.w.w_dev;
                // Nearly at largest army? Cards are worth more.
                let k = v.knights_played(v.me());
                if k >= 2 && v.largest_army() != Some(v.me()) {
                    s += 6.0;
                }
                s
            }

            Action::PlayKnight => {
                let mut s = self.w.knight_eagerness;
                let k = v.knights_played(v.me()) + 1;
                let best_other = (0..v.n_players() as u8)
                    .filter(|&q| q != v.me())
                    .map(|q| v.knights_played(q))
                    .max()
                    .unwrap_or(0);
                if k >= 3 && v.largest_army() != Some(v.me()) && k > best_other {
                    s += 45.0;
                }
                // Robber parked on one of our own hexes: shift it.
                let topo = crate::board::topology();
                if topo.hex_nodes[v.robber() as usize]
                    .iter()
                    .any(|&n| v.node(n).map(|pc| pc.owner == v.me()).unwrap_or(false))
                {
                    s += 20.0;
                }
                s
            }
            Action::PlayRoadBuilding => 12.0 + self.w.w_road,
            Action::PlayYearOfPlenty(a1, b1) => {
                let mut h2 = hand;
                h2[a1.idx()] += 1;
                h2[b1.idx()] += 1;
                let gain = (def_now - Self::deficit(&h2, &target)) as f64;
                10.0 + 8.0 * gain
            }
            Action::PlayMonopoly(r) => {
                let est = self.estimate_opponent_holdings(v, r);
                4.0 + 6.0 * est
            }

            Action::BankTrade { give, want, rate } => {
                let mut h2 = hand;
                h2[give.idx()] -= rate as i16;
                h2[want.idx()] += 1;
                let gain = (def_now - Self::deficit(&h2, &target)) as f64;
                if gain <= 0.0 {
                    -1.0
                } else {
                    // Only worth it if we are not throwing away cards we need.
                    self.w.bank_trade_drive * 9.0 * gain
                }
            }

            Action::ProposeTrade { give, want } => {
                if self.w.propose_drive <= 0.0 {
                    return -1000.0;
                }
                let mut h2 = hand;
                for i in 0..N_RES {
                    h2[i] -= give[i];
                    h2[i] += want[i];
                }
                if h2.iter().any(|&c| c < 0) {
                    return -1000.0;
                }
                let gain = (def_now - Self::deficit(&h2, &target)) as f64;
                if gain <= 0.0 {
                    -1.0
                } else {
                    // Discounted: it also hands an opponent a card, and may
                    // simply be refused.
                    self.w.propose_drive * 4.0 * gain - 2.0
                }
            }

            Action::MoveRobber { hex, victim } => 50.0 + self.robber_score(v, hex, victim),
        };

        base + self.rng.random_range(0.0..1.0) * self.w.noise
    }
}

/// Tiny helper so `target_cost` reads as a fold.
trait MaxBy {
    fn max_by(self, score: f64, cost: Res) -> (f64, Res);
}
impl MaxBy for (f64, Res) {
    fn max_by(self, score: f64, cost: Res) -> (f64, Res) {
        if score > self.0 {
            (score, cost)
        } else {
            self
        }
    }
}

impl Policy for HeuristicBot {
    fn name(&self) -> String {
        self.w.label.clone()
    }

    fn choose(&mut self, v: &View, actions: &[Action]) -> usize {
        let mut best = 0usize;
        let mut best_s = f64::MIN;
        for (i, a) in actions.iter().enumerate() {
            let s = self.score(v, a);
            if s > best_s {
                best_s = s;
                best = i;
            }
        }
        best
    }

    fn discard(&mut self, v: &View, n: i16) -> Res {
        // Keep what the next build needs; shed the rest, biggest pile first.
        let hand = *v.hand();
        let target = self.target_cost(v);
        let mut keep = ZERO;
        for i in 0..N_RES {
            keep[i] = target[i].min(hand[i]);
        }
        let mut spare = hand;
        for i in 0..N_RES {
            spare[i] -= keep[i];
        }
        let mut out = ZERO;
        let mut left = n;
        // First from the spare pool.
        while left > 0 && res_total(&spare) > 0 {
            let mut b = 0;
            for i in 1..N_RES {
                if spare[i] > spare[b] {
                    b = i;
                }
            }
            spare[b] -= 1;
            out[b] += 1;
            left -= 1;
        }
        // Then, reluctantly, from the reserved cards.
        while left > 0 {
            let mut b = usize::MAX;
            for i in 0..N_RES {
                if keep[i] - out[i].min(keep[i]) > 0
                    && (b == usize::MAX || keep[i] > keep[b])
                    && out[i] < hand[i]
                {
                    b = i;
                }
            }
            if b == usize::MAX {
                break;
            }
            out[b] += 1;
            keep[b] -= 1;
            left -= 1;
        }
        out
    }

    fn accept_trade(&mut self, v: &View, to_me: &Res, of_me: &Res, from: u8) -> bool {
        if !self.w.accepts_trades {
            return false;
        }
        let hand = *v.hand();
        if (0..N_RES).any(|i| hand[i] < of_me[i]) {
            return false;
        }
        let target = self.target_cost(v);
        let before = Self::deficit(&hand, &target);
        let mut after_hand = hand;
        for i in 0..N_RES {
            after_hand[i] += to_me[i];
            after_hand[i] -= of_me[i];
        }
        let after = Self::deficit(&after_hand, &target);
        let mut benefit = (before - after) as f64;
        // Giving away more cards than we get is a real cost.
        benefit -= 0.4 * (res_total(of_me) - res_total(to_me)) as f64;
        // Do not feed the player who is winning.
        if from == self.leader(v) {
            benefit -= self.w.leader_wariness;
        }
        // A trade that pushes us over the discard limit is less attractive.
        if res_total(&after_hand) > v.hand_limit() {
            benefit -= 0.5;
        }
        benefit > self.w.accept_floor
    }
}

// ---------------------------------------------------------------- factory ----

/// Build a policy by name. `seed` decorrelates each seat's tie-breaking.
pub fn make(name: &str, seed: u64) -> Option<Box<dyn Policy>> {
    if name == "random" {
        return Some(Box::new(RandomBot::new(seed)));
    }
    Weights::preset(name).map(|w| Box::new(HeuristicBot::new(w, seed)) as Box<dyn Policy>)
}

/// Every strategy name the simulator knows.
pub fn available() -> Vec<&'static str> {
    let mut v = Weights::all_presets();
    v.push("random");
    v
}

/// Port kinds a node grants, for reporting.
pub fn node_ports(v: &View, n: u8) -> Vec<PortKind> {
    let m = v.board().node_port_mask[n as usize];
    let mut out = Vec::new();
    for r in Resource::ALL {
        if m & (1 << r.idx()) != 0 {
            out.push(PortKind::Specific(r));
        }
    }
    if m & (1 << 5) != 0 {
        out.push(PortKind::Generic);
    }
    out
}
