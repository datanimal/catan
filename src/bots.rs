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
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

/// Diagnostic counters for the planner. Cheap relaxed atomics, kept in the
/// build so that "the feature made no difference" can always be told apart
/// from "the feature never ran".
pub static PLAN_CALLS: AtomicU64 = AtomicU64::new(0);
pub static PLAN_FOUND: AtomicU64 = AtomicU64::new(0);
pub static PLAN_CHOSEN: AtomicU64 = AtomicU64::new(0);
pub static PLAN_FLIPPED: AtomicU64 = AtomicU64::new(0);
/// Decisions where a settlement was affordable at all -- the precondition for
/// any plan. Separates "my feasibility test is too strict" from "the
/// situation genuinely does not arise".
pub static PLAN_AFFORDABLE: AtomicU64 = AtomicU64::new(0);
/// Of those, decisions where some reachable spot was strictly better than
/// anything already in reach.
pub static PLAN_BETTER_EXISTS: AtomicU64 = AtomicU64::new(0);
/// Plans that are only affordable because a Road Building card pays for part
/// of the route -- the card dodges the hand limit by supplying roads free.
pub static PLAN_VIA_RB: AtomicU64 = AtomicU64::new(0);

pub fn plan_counters() -> (u64, u64, u64, u64) {
    (
        PLAN_CALLS.load(Ordering::Relaxed),
        PLAN_FOUND.load(Ordering::Relaxed),
        PLAN_CHOSEN.load(Ordering::Relaxed),
        PLAN_FLIPPED.load(Ordering::Relaxed),
    )
}
pub fn plan_preconditions() -> (u64, u64, u64) {
    (
        PLAN_AFFORDABLE.load(Ordering::Relaxed),
        PLAN_BETTER_EXISTS.load(Ordering::Relaxed),
        PLAN_VIA_RB.load(Ordering::Relaxed),
    )
}
pub fn reset_plan_counters() {
    for c in [
        &PLAN_CALLS,
        &PLAN_FOUND,
        &PLAN_CHOSEN,
        &PLAN_FLIPPED,
        &PLAN_AFFORDABLE,
        &PLAN_BETTER_EXISTS,
        &PLAN_VIA_RB,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    /// Weight on the timing premium for playing a knight *before* rolling:
    /// the expected cards it adds to this turn's roll by freeing one of my
    /// hexes, plus the cards it denies an opponent on the same roll.
    pub knight_timing: f64,
    /// Value discount per extra road needed to reach a spot. Together with
    /// the road budget this sets the planning horizon, so holding Road
    /// Building genuinely extends how far the bot can see.
    pub reach_decay: f64,
    /// Baseline planning horizon in roads, before any bonus for free roads,
    /// a Road Building card or spare brick and lumber. Roads persist, so this
    /// is how far ahead the bot looks as a matter of course.
    pub road_horizon_base: f64,
    /// Urgency multiplier for a spot an opponent could reach as soon as I
    /// could -- racing them to it.
    pub contest_drive: f64,
    /// Value of denying opponents the intersections a new settlement shuts
    /// out under the distance rule.
    pub block_drive: f64,
    /// Weight on the sequential plan: how strongly to pursue a settlement
    /// several roads away that is better than anything already in reach.
    pub plan_drive: f64,
    /// Penalty applied to a city *only* at the exact choice point: one city
    /// is already up, and the hand can pay for either a second city or two
    /// development cards right now. This isolates the substitution, where
    /// `later_city_discount` also changes what the bot saves for and so ends
    /// up building less of everything.
    pub dev_over_second_city: f64,
    /// Bounds on own ore production (expected ore cards per roll) within
    /// which `dev_over_second_city` applies. Used to test whether the choice
    /// depends on how hard ore is to come by: a player swimming in ore can
    /// rebuild toward a city easily, while for an ore-poor player three ore
    /// in hand may be a chance that does not come again.
    pub dev_pref_min_ore: f64,
    pub dev_pref_max_ore: f64,
    /// Subtracted from the score of every city after the first. Cities and
    /// development cards compete for the same ore and grain, so this is the
    /// knob that decides which wins at the margin once one city is up.
    pub later_city_discount: f64,
    /// Weight on the cards the *second* opening settlement pays out at once:
    /// one per adjacent producing hex. Those cards arrive before the first
    /// roll, so a spot touching three producing hexes starts a turn ahead of
    /// one touching two, and a payout that happens to complete a road or a
    /// development card is worth more than three unrelated cards.
    pub setup_payout: f64,
    /// Random jitter added to every score, to break ties and decorrelate.
    pub noise: f64,
    /// Intersections to take for the opening settlements, in order, when they
    /// are still free. Used to measure what a specific opening is worth by
    /// playing it out, rather than by extrapolating fitted coefficients.
    #[serde(default)]
    pub forced_opening: Vec<u8>,
    /// Place the opening settlements uniformly at random among the legal
    /// intersections, while playing normally afterwards. Used to get
    /// *exogenous* variation in opening position: when a bot chooses where to
    /// start, its placement is entangled with its strategy and the two cannot
    /// be told apart. Randomising it breaks that link.
    pub random_opening: bool,
}

/// The tunable parameters, as a flat list for search.
pub const PARAM_KEYS: [&str; 28] = [
    "w_city",
    "w_settlement",
    "w_road",
    "w_dev",
    "pref_brick",
    "pref_lumber",
    "pref_wool",
    "pref_grain",
    "pref_ore",
    "diversity",
    "port",
    "knight_eagerness",
    "longest_road_drive",
    "bank_trade_drive",
    "propose_drive",
    "accept_floor",
    "leader_wariness",
    "knight_timing",
    "reach_decay",
    "road_horizon_base",
    "contest_drive",
    "block_drive",
    "plan_drive",
    "setup_payout",
    "later_city_discount",
    "dev_over_second_city",
    "dev_pref_min_ore",
    "dev_pref_max_ore",
];

/// Plausible bounds for each parameter, so a search cannot wander into
/// nonsense.
pub fn param_bounds(key: &str) -> (f64, f64) {
    match key {
        "w_city" | "w_settlement" => (5.0, 120.0),
        "w_road" => (0.0, 60.0),
        "w_dev" => (0.0, 90.0),
        "pref_brick" | "pref_lumber" | "pref_wool" | "pref_grain" | "pref_ore" => (0.2, 3.0),
        "diversity" => (0.0, 15.0),
        "port" => (0.0, 20.0),
        "knight_eagerness" => (0.0, 40.0),
        "longest_road_drive" => (0.0, 10.0),
        "bank_trade_drive" | "propose_drive" => (0.0, 6.0),
        "accept_floor" => (-3.0, 4.0),
        "leader_wariness" => (0.0, 10.0),
        "knight_timing" => (0.0, 60.0),
        // Below about 0.2 a road is valued only for the spot it reaches
        // directly; near 0.9 distant spots count almost as much as near ones.
        "reach_decay" => (0.15, 0.95),
        "road_horizon_base" => (1.0, 6.0),
        "contest_drive" => (0.0, 4.0),
        "block_drive" => (0.0, 4.0),
        "plan_drive" => (0.0, 4.0),
        "setup_payout" => (0.0, 8.0),
        "later_city_discount" => (-20.0, 60.0),
        "dev_over_second_city" => (0.0, 120.0),
        "dev_pref_min_ore" => (0.0, 1.0),
        "dev_pref_max_ore" => (0.0, 99.0),
        _ => (0.0, 1.0),
    }
}

impl Weights {
    pub fn get(&self, key: &str) -> f64 {
        use Resource::*;
        match key {
            "w_city" => self.w_city,
            "w_settlement" => self.w_settlement,
            "w_road" => self.w_road,
            "w_dev" => self.w_dev,
            "pref_brick" => self.res_pref[Brick.idx()],
            "pref_lumber" => self.res_pref[Lumber.idx()],
            "pref_wool" => self.res_pref[Wool.idx()],
            "pref_grain" => self.res_pref[Grain.idx()],
            "pref_ore" => self.res_pref[Ore.idx()],
            "diversity" => self.diversity,
            "port" => self.port,
            "knight_eagerness" => self.knight_eagerness,
            "longest_road_drive" => self.longest_road_drive,
            "bank_trade_drive" => self.bank_trade_drive,
            "propose_drive" => self.propose_drive,
            "accept_floor" => self.accept_floor,
            "leader_wariness" => self.leader_wariness,
            "knight_timing" => self.knight_timing,
            "reach_decay" => self.reach_decay,
            "road_horizon_base" => self.road_horizon_base,
            "contest_drive" => self.contest_drive,
            "block_drive" => self.block_drive,
            "plan_drive" => self.plan_drive,
            "setup_payout" => self.setup_payout,
            "later_city_discount" => self.later_city_discount,
            "dev_over_second_city" => self.dev_over_second_city,
            "dev_pref_min_ore" => self.dev_pref_min_ore,
            "dev_pref_max_ore" => self.dev_pref_max_ore,
            other => panic!("unknown weight '{other}'"),
        }
    }

    pub fn set(&mut self, key: &str, v: f64) {
        use Resource::*;
        let (lo, hi) = param_bounds(key);
        let v = v.clamp(lo, hi);
        match key {
            "w_city" => self.w_city = v,
            "w_settlement" => self.w_settlement = v,
            "w_road" => self.w_road = v,
            "w_dev" => self.w_dev = v,
            "pref_brick" => self.res_pref[Brick.idx()] = v,
            "pref_lumber" => self.res_pref[Lumber.idx()] = v,
            "pref_wool" => self.res_pref[Wool.idx()] = v,
            "pref_grain" => self.res_pref[Grain.idx()] = v,
            "pref_ore" => self.res_pref[Ore.idx()] = v,
            "diversity" => self.diversity = v,
            "port" => self.port = v,
            "knight_eagerness" => self.knight_eagerness = v,
            "longest_road_drive" => self.longest_road_drive = v,
            "bank_trade_drive" => self.bank_trade_drive = v,
            "propose_drive" => self.propose_drive = v,
            "accept_floor" => self.accept_floor = v,
            "leader_wariness" => self.leader_wariness = v,
            "knight_timing" => self.knight_timing = v,
            "reach_decay" => self.reach_decay = v,
            "road_horizon_base" => self.road_horizon_base = v,
            "contest_drive" => self.contest_drive = v,
            "block_drive" => self.block_drive = v,
            "plan_drive" => self.plan_drive = v,
            "setup_payout" => self.setup_payout = v,
            "later_city_discount" => self.later_city_discount = v,
            "dev_over_second_city" => self.dev_over_second_city = v,
            "dev_pref_min_ore" => self.dev_pref_min_ore = v,
            "dev_pref_max_ore" => self.dev_pref_max_ore = v,
            other => panic!("unknown weight '{other}'"),
        }
    }

    /// Compact `key=value` form, for logging a tuned strategy.
    ///
    /// Overrides are separated by `;`, not `,`, so that a whole spec survives
    /// being passed inside a comma-separated list of players.
    pub fn to_spec(&self) -> String {
        PARAM_KEYS
            .iter()
            .map(|k| format!("{k}={:.3}", self.get(k)))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// Parse `base+key=value;key=value`, e.g. `balanced+w_dev=40;port=9`.
    /// Commas are also accepted for a spec given on its own.
    pub fn from_spec(spec: &str) -> Option<Weights> {
        let (base, rest) = match spec.split_once('+') {
            Some((b, r)) => (b, Some(r)),
            None => (spec, None),
        };
        let mut w = Weights::preset(base)?;
        if let Some(rest) = rest {
            for kv in rest
                .split([';', ','])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
            {
                let (k, v) = kv.split_once('=')?;
                let v: f64 = v.trim().parse().ok()?;
                if !PARAM_KEYS.contains(&k.trim()) {
                    return None;
                }
                w.set(k.trim(), v);
            }
        }
        Some(w)
    }
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
            // All four set from the parameter sweeps in the README.
            knight_timing: 15.0,
            reach_decay: 0.8,
            // Looking further than two roads ahead measured flat from 2 to 6.
            road_horizon_base: 2.0,
            contest_drive: 1.5,
            // Denying opponents spots showed no measurable value at any
            // setting, so it is off by default; the hook is kept because the
            // null may be an artefact of having no lookahead.
            block_drive: 0.0,
            plan_drive: 1.0,
            // Off by default: measured, and weighting it only ever hurt. See
            // the README. The bots already collect 2.80 of the 3 available
            // cards as a by-product of preferring pips.
            dev_over_second_city: 0.0,
            dev_pref_min_ore: 0.0,
            dev_pref_max_ore: 99.0,
            later_city_discount: 0.0,
            setup_payout: 0.0,
            forced_opening: Vec::new(),
            noise: 0.75,
            random_opening: false,
        }
    }
}

impl Weights {
    pub fn preset(name: &str) -> Option<Weights> {
        use Resource::*;
        let mut w = Weights::default();
        match name {
            "balanced" => {}
            // Found by self-play search (`catan_sim tune`), starting from
            // `balanced`, over about 440k games. It beats every preset here:
            // 28.1% against three city_builders and 30.6% against three
            // balanced, where chance is 25%. Its character: trade far more
            // readily but guard the leader, favour grain and ore over wood and
            // brick, buy fewer development cards, and weight raw numbers over
            // spreading across resources at a single intersection.
            "tuned" => {
                w.label = "tuned".into();
                w.w_city = 54.88;
                w.w_settlement = 55.41;
                w.w_road = 6.33;
                w.w_dev = 12.70;
                w.res_pref[Brick.idx()] = 0.861;
                w.res_pref[Lumber.idx()] = 0.750;
                w.res_pref[Wool.idx()] = 1.005;
                w.res_pref[Grain.idx()] = 1.243;
                w.res_pref[Ore.idx()] = 1.105;
                w.diversity = 1.032;
                w.port = 3.000;
                w.knight_eagerness = 6.024;
                w.longest_road_drive = 1.207;
                w.bank_trade_drive = 1.194;
                w.propose_drive = 2.248;
                w.accept_floor = 0.279;
                w.leader_wariness = 3.835;
                // Once one city is up, a second competes with development
                // cards for the same ore and grain. Preferring the cards is
                // worth about +0.4 points of win rate, but only when ore is
                // plentiful enough to rebuild toward a city easily; gated on
                // ore production for that reason. See the README.
                w.dev_over_second_city = 100.0;
                w.dev_pref_min_ore = 0.18;
                w.knight_timing = 15.0;
                w.reach_decay = 0.8;
                w.contest_drive = 1.5;
                w.block_drive = 0.0;
            }
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
            "tuned",
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

/// A settlement that can be acquired within this turn's means, together with
/// the sequence of actions that gets there.
///
/// This is the bounded sequential lookahead. Scoring single actions greedily
/// cannot represent "play Road Building, lay two roads, settle the good spot",
/// because each step on its own looks worse than simply settling whatever is
/// already in reach. The planner evaluates the whole package against that
/// alternative, which is also what finally gives denial something to do: the
/// bot can now travel to a spot *in order to* shut an opponent out, instead of
/// only preferring denial among spots it was already going to take.
#[derive(Clone, Debug)]
struct Plan {
    target: u8,
    /// Edges to build, nearest first.
    route: Vec<u8>,
    /// Whether the plan spends a Road Building card to afford the route.
    needs_road_building: bool,
    /// Score of the finished position on the *settlement* scale, net of the
    /// resources the route costs. Comparable with `BuildSettlement`, which is
    /// the alternative it has to beat.
    score: f64,
    #[allow(dead_code)]
    target_kept_for_debug: (),
    /// How much it beats settling the best spot already in reach.
    excess: f64,
}

/// Per-decision road geography, computed once per `choose` rather than once
/// per action, since it is the same for every candidate move.
struct RoadCtx {
    /// Roads I would need to lay to reach each intersection.
    my_dist: [u8; N_NODES],
    /// Roads the nearest opponent would need. Used to spot contested spots.
    opp_dist: [u8; N_NODES],
    /// Best discounted value obtainable starting from each intersection,
    /// propagated back along buildable paths. This is what gives the bot a
    /// planning horizon longer than one road.
    reach: [f64; N_NODES],
    /// The same, with two extra roads available -- what Road Building unlocks.
    reach_rb: [f64; N_NODES],
    /// The edge each intersection was first reached by, for route building.
    /// Consumed when the plan is built, then kept for debugging.
    #[allow(dead_code)]
    my_via: [u8; N_NODES],
    /// The best settlement plan executable this turn, if any.
    plan: Option<Plan>,
    /// How many roads ahead the reach values look. Kept for debugging and
    /// for the `horizon` assertions in tests.
    #[allow(dead_code)]
    horizon: u8,
}

// Arrays this long have no derived Default.
impl Default for RoadCtx {
    fn default() -> Self {
        RoadCtx {
            my_dist: [u8::MAX; N_NODES],
            opp_dist: [u8::MAX; N_NODES],
            my_via: [u8::MAX; N_NODES],
            plan: None,
            reach: [0.0; N_NODES],
            reach_rb: [0.0; N_NODES],
            horizon: 2,
        }
    }
}

pub struct HeuristicBot {
    w: Weights,
    rng: ChaCha8Rng,
    ctx: RoadCtx,
}

/// Roads needed for `p` to reach each intersection, and the edge each
/// intersection was first reached by, so a route can be reconstructed.
///
/// A path may not pass through an intersection occupied by an opponent, and
/// `p`'s existing roads are free to travel along.
fn road_distances_with_paths(v: &View, p: u8) -> ([u8; N_NODES], [u8; N_NODES]) {
    let topo = crate::board::topology();
    let mut dist = [u8::MAX; N_NODES];
    let mut via = [u8::MAX; N_NODES];
    let mut dq: std::collections::VecDeque<u8> = Default::default();
    // Sources: anywhere p already has a presence.
    for n in 0..N_NODES as u8 {
        let mine_here = v.node(n).map(|pc| pc.owner == p).unwrap_or(false)
            || topo.node_edges[n as usize]
                .iter()
                .any(|&e| v.edge(e) == Some(p));
        if mine_here {
            dist[n as usize] = 0;
            dq.push_back(n);
        }
    }
    // 0-1 BFS: travelling an existing road of mine costs nothing, laying a
    // new one costs one.
    while let Some(n) = dq.pop_front() {
        // Cannot build out through an opponent's settlement or city.
        if let Some(pc) = v.node(n) {
            if pc.owner != p {
                continue;
            }
        }
        let d = dist[n as usize];
        for &e in topo.node_edges[n as usize].iter() {
            let owner = v.edge(e);
            let step = match owner {
                Some(o) if o == p => 0u8,
                Some(_) => continue, // an opponent's road blocks this path
                None => 1,
            };
            let [a, b] = topo.edge_nodes[e as usize];
            let m = if a == n { b } else { a };
            let nd = d.saturating_add(step);
            if nd < dist[m as usize] {
                dist[m as usize] = nd;
                via[m as usize] = e;
                if step == 0 {
                    dq.push_front(m);
                } else {
                    dq.push_back(m);
                }
            }
        }
    }
    (dist, via)
}

/// Distances only, for callers that do not need routes.
fn road_distances(v: &View, p: u8) -> [u8; N_NODES] {
    road_distances_with_paths(v, p).0
}

/// The edges that must be built, nearest first, to reach `target`.
fn route_to(
    v: &View,
    target: u8,
    dist: &[u8; N_NODES],
    via: &[u8; N_NODES],
) -> Vec<u8> {
    let topo = crate::board::topology();
    let mut out = Vec::new();
    let mut n = target;
    // Walk back to the player's existing network.
    while dist[n as usize] > 0 {
        let e = via[n as usize];
        if e == u8::MAX {
            return Vec::new();
        }
        if v.edge(e).is_none() {
            out.push(e);
        }
        let [a, b] = topo.edge_nodes[e as usize];
        let prev = if a == n { b } else { a };
        if dist[prev as usize] >= dist[n as usize] && v.edge(e).is_none() {
            return Vec::new(); // not making progress; bail out
        }
        n = prev;
        if out.len() > 10 {
            return Vec::new();
        }
    }
    out.reverse();
    out
}

impl HeuristicBot {
    pub fn new(w: Weights, seed: u64) -> Self {
        HeuristicBot {
            w,
            rng: ChaCha8Rng::seed_from_u64(seed),
            ctx: RoadCtx::default(),
        }
    }

    /// How far ahead to look along buildable paths.
    ///
    /// Roads persist, so there is always a reason to look a little beyond the
    /// next settlement -- hence a baseline of two. What extends the horizon is
    /// being able to lay several roads soon: free roads already granted, a
    /// Road Building card still in hand, and brick and lumber to spare. That
    /// is what lets the bot spot a spot three or four roads out that it could
    /// reach in one turn, to race for or to shut off.
    /// This does **not** include a bonus for holding an unplayed Road
    /// Building card. That bonus belongs only to the counterfactual horizon
    /// used to price the card itself: if it were folded in here, the "what
    /// would two more roads buy me" comparison would measure two roads beyond
    /// an already-extended horizon, which is close to nothing, and the card's
    /// vision value would silently evaluate to zero.
    fn road_horizon(&self, v: &View) -> u8 {
        let mut h = self.w.road_horizon_base.round().max(1.0) as u8;
        // Roads already granted by a played Road Building card count here.
        h += v.free_roads();
        let hand = v.hand();
        let affordable = hand[Resource::Brick.idx()].min(hand[Resource::Lumber.idx()]);
        h += (affordable.max(0) as u8).min(2);
        h.min(v.roads_left()).min(7).max(1)
    }

    /// Enumerate settlement targets reachable this turn and keep the best.
    ///
    /// A target `d` roads away needs `d` roads and a settlement. Free roads
    /// already granted pay for some; a Road Building card still in hand pays
    /// for two more. The rest must be affordable in cash.
    fn plan_settlement(
        &self,
        v: &View,
        my_dist: &[u8; N_NODES],
        my_via: &[u8; N_NODES],
    ) -> Option<Plan> {
        if v.settlements_left() == 0 {
            return None;
        }
        let hand = *v.hand();
        let free_now = v.free_roads();
        let rb_in_hand = !v.dev_played_this_turn()
            && v.my_dev()[DevCard::RoadBuilding.idx()] > 0
            && free_now == 0;

        // Everything is scored the way `BuildSettlement` is scored, because
        // that is the action a plan competes against. Scoring the plan's first
        // step on the *road* scale instead was the original mistake: a road
        // scores around 16 and settling a mediocre spot now around 75, so no
        // plan bonus of a plausible size could ever win.
        let settle_score = |n: u8| {
            self.w.w_settlement + 1.3 * self.node_value(v, n) + self.block_value(v, n)
        };
        // Each road the route cannot get for free costs a brick and a lumber.
        // Charging it here is what makes Road Building pivotal: it supplies
        // two roads at no cost, so it removes two of these penalties.
        const CARD_VALUE: f64 = 8.0;
        let road_penalty = 2.0 * CARD_VALUE;

        let mut best_immediate = 0.0f64;
        if can_afford(&hand, &COST_SETTLEMENT) {
            for n in 0..N_NODES as u8 {
                if my_dist[n as usize] == 0 && v.node_open(n) {
                    best_immediate = best_immediate.max(settle_score(n));
                }
            }
            PLAN_AFFORDABLE.fetch_add(1, Ordering::Relaxed);
        }

        let mut any_better = false;
        let mut best: Option<Plan> = None;
        for t in 0..N_NODES as u8 {
            let d = my_dist[t as usize];
            if d == u8::MAX || d == 0 || !v.node_open(t) {
                continue;
            }
            if d > v.roads_left() || d > 4 {
                continue;
            }
            // Roads the free ones and a Road Building card do not cover.
            let covered_free = free_now.min(d);
            let mut remaining = d - covered_free;
            let mut needs_rb = false;
            if remaining > 0 && rb_in_hand {
                remaining -= remaining.min(2);
                needs_rb = true;
            }
            let score = settle_score(t) - road_penalty * remaining as f64;
            if score > best_immediate {
                any_better = true;
            }

            let mut cost = COST_SETTLEMENT;
            for _ in 0..remaining {
                res_add(&mut cost, &COST_ROAD);
            }
            if !can_afford(&hand, &cost) {
                continue;
            }
            let route = route_to(v, t, my_dist, my_via);
            if route.is_empty() || route.len() != d as usize {
                continue;
            }
            let excess = score - best_immediate;
            if excess <= 0.0 {
                continue;
            }
            if best.as_ref().map(|b| excess > b.excess).unwrap_or(true) {
                if needs_rb {
                    PLAN_VIA_RB.fetch_add(1, Ordering::Relaxed);
                }
                best = Some(Plan {
                    target: t,
                    route,
                    needs_road_building: needs_rb,
                    score,
                    target_kept_for_debug: (),
                    excess,
                });
            }
        }
        if any_better {
            PLAN_BETTER_EXISTS.fetch_add(1, Ordering::Relaxed);
        }
        best
    }

    /// Lift the first step of the best plan onto the plan's own score.
    ///
    /// At `plan_drive` 1 the step scores exactly what the finished plan is
    /// worth, so it competes with settling now on equal terms; at 0 it keeps
    /// its ordinary greedy score.
    fn plan_bonus(&self, v: &View, a: &Action, greedy: f64) -> f64 {
        let Some(p) = &self.ctx.plan else {
            return 0.0;
        };
        let first_matches = match *a {
            Action::PlayRoadBuilding => p.needs_road_building,
            Action::BuildRoad(e) => !p.needs_road_building && p.route.first() == Some(&e),
            Action::BuildSettlement(n) => n == p.target && p.route.is_empty(),
            _ => false,
        };
        let _ = v;
        if first_matches {
            self.w.plan_drive * (p.score - greedy).max(0.0)
        } else {
            0.0
        }
    }

    /// Build the per-decision road geography.
    fn build_ctx(&mut self, v: &View) {
        let topo = crate::board::topology();
        let me = v.me();
        let (my_dist, my_via) = road_distances_with_paths(v, me);
        let mut opp_dist = [u8::MAX; N_NODES];
        for q in 0..v.n_players() as u8 {
            if q == me {
                continue;
            }
            let d = road_distances(v, q);
            for i in 0..N_NODES {
                opp_dist[i] = opp_dist[i].min(d[i]);
            }
        }

        // Seed every open intersection with its value, raised when an
        // opponent could get there as soon as I could.
        let mut seed = [0.0f64; N_NODES];
        for n in 0..N_NODES as u8 {
            if !v.node_open(n) {
                continue;
            }
            let base = self.node_value(v, n);
            let (mine, theirs) = (my_dist[n as usize], opp_dist[n as usize]);
            let urgency = if theirs <= mine {
                1.0
            } else if theirs == mine.saturating_add(1) {
                0.5
            } else {
                0.0
            };
            seed[n as usize] = base * (1.0 + self.w.contest_drive * urgency);
        }

        // Propagate value backwards along buildable paths, one road per pass,
        // discounting each step. `budget` passes means the bot sees exactly as
        // far as it could actually build this turn.
        let horizon = self.road_horizon(v);
        let relax = |passes: u8, seed: &[f64; N_NODES]| -> [f64; N_NODES] {
            let mut val = *seed;
            for _ in 0..passes {
                let mut next = val;
                for e in 0..N_EDGES as u8 {
                    if v.edge(e).is_some() {
                        continue;
                    }
                    let [a, b] = topo.edge_nodes[e as usize];
                    // Value cannot flow through an opponent's building.
                    let passable = |n: u8| {
                        v.node(n).map(|pc| pc.owner == v.me()).unwrap_or(true)
                    };
                    if passable(a) {
                        next[a as usize] =
                            next[a as usize].max(self.w.reach_decay * val[b as usize]);
                    }
                    if passable(b) {
                        next[b as usize] =
                            next[b as usize].max(self.w.reach_decay * val[a as usize]);
                    }
                }
                val = next;
            }
            val
        };
        let plan = self.plan_settlement(v, &my_dist, &my_via);
        PLAN_CALLS.fetch_add(1, Ordering::Relaxed);
        if plan.is_some() {
            PLAN_FOUND.fetch_add(1, Ordering::Relaxed);
        }
        self.ctx = RoadCtx {
            my_dist,
            my_via,
            plan,
            opp_dist,
            reach: relax(horizon.saturating_sub(1), &seed),
            // Exactly two extra roads: the counterfactual horizon if the Road
            // Building card in hand were played.
            reach_rb: relax(horizon.saturating_add(1).min(8), &seed),
            horizon,
        };
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

    /// Value of the cards the second opening settlement pays immediately.
    ///
    /// Only the second settlement pays, one card per adjacent producing hex,
    /// before anyone rolls. Counting those cards is the obvious reading and
    /// it is the wrong one: what matters is whether the payout plus *one
    /// roll* completes something, because a city on the first turn is two
    /// points and a doubled hex while the board is still empty.
    ///
    /// So each roll is checked against the payout combined with what both
    /// settlements would produce on it, and the best build that becomes
    /// affordable is weighted by that roll's probability. An opening paying
    /// two ore and a grain, next to an ore and a grain that share a number,
    /// is one roll from a city; the same three cards spread over unrelated
    /// numbers are just three cards.
    fn setup_payout_value(&self, v: &View, n: u8) -> f64 {
        // The first settlement pays nothing, so this only applies to the
        // second -- by which point one is already placed.
        if v.settlements_left() != MAX_SETTLEMENTS - 1 {
            return 0.0;
        }
        let topo = crate::board::topology();
        let b = v.board();
        let mut pay = ZERO;
        for &h in topo.node_hexes[n as usize].iter() {
            if let Some(r) = b.hex_resource(h) {
                pay[r.idx()] += 1;
            }
        }

        // Both settlements produce once the dice start.
        let mut mine: Vec<u8> = vec![n];
        for q in 0..N_NODES as u8 {
            if v.node(q).map(|pc| pc.owner == v.me()).unwrap_or(false) {
                mine.push(q);
            }
        }

        // Cards are worth something on their own.
        let mut value = 0.5 * res_total(&pay) as f64;
        for roll in 2..=12u8 {
            let mut hand = pay;
            for &q in &mine {
                for &h in topo.node_hexes[q as usize].iter() {
                    if b.numbers[h as usize] == roll {
                        if let Some(r) = b.hex_resource(h) {
                            hand[r.idx()] += 1;
                        }
                    }
                }
            }
            // Only a city counts. A road or a development card is one roll
            // away from almost every opening, so crediting those just adds
            // noise to every choice; a city is available in about one opening
            // in seventeen, and is worth two points and a doubled hex while
            // the board is still empty.
            if can_afford(&hand, &COST_CITY) {
                value += 10.0 * crate::types::prob(roll);
            }
        }
        self.w.setup_payout * value
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
            // The same discount the build decision uses, or the bot would go
            // on *saving* for a city it has decided not to prefer and would
            // build it anyway once the ore piled up.
            let w = if MAX_CITIES - v.cities_left() >= 1 {
                self.w.w_city - self.w.later_city_discount
            } else {
                self.w.w_city
            };
            best = best.max_by(w, COST_CITY);
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

    /// Value of a road: what it opens up, plus progress toward longest road.
    ///
    /// The reach values are propagated over as many roads as could actually
    /// be laid this turn, so with Road Building in hand the bot sees spots
    /// three or four roads out instead of only its next settlement.
    fn road_value(&self, v: &View, e: u8) -> f64 {
        let topo = crate::board::topology();
        let mut best = 0.0f64;
        for &n in topo.edge_nodes[e as usize].iter() {
            if let Some(pc) = v.node(n) {
                if pc.owner != v.me() {
                    continue;
                }
            }
            best = best.max(0.55 * self.ctx.reach[n as usize]);
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

    /// The best road available right now, under the current horizon and under
    /// the horizon two extra roads would buy. The gap is what Road Building
    /// is worth beyond the four resources it saves.
    fn reach_gain_from_two_roads(&self, v: &View) -> f64 {
        let topo = crate::board::topology();
        let mut now = 0.0f64;
        let mut with_rb = 0.0f64;
        for e in 0..N_EDGES as u8 {
            if v.edge(e).is_some() {
                continue;
            }
            for &n in topo.edge_nodes[e as usize].iter() {
                if let Some(pc) = v.node(n) {
                    if pc.owner != v.me() {
                        continue;
                    }
                }
                // Only edges I can actually start from count.
                if self.ctx.my_dist[n as usize] > 0 {
                    continue;
                }
                now = now.max(self.ctx.reach[n as usize]);
                with_rb = with_rb.max(self.ctx.reach_rb[n as usize]);
            }
        }
        (with_rb - now).max(0.0)
    }

    /// Intersections a settlement here would deny everyone under the distance
    /// rule. Only counts spots an opponent was actually close to.
    fn block_value(&self, v: &View, n: u8) -> f64 {
        if self.w.block_drive <= 0.0 {
            return 0.0;
        }
        let topo = crate::board::topology();
        let mut denied = 0.0;
        for &nb in topo.node_neighbors[n as usize].iter() {
            if !v.node_open(nb) {
                continue;
            }
            let theirs = self.ctx.opp_dist[nb as usize];
            let mine = self.ctx.my_dist[nb as usize];
            if theirs <= 2 && theirs <= mine {
                denied += v.board().node_pips(nb) as f64;
            }
        }
        self.w.block_drive * denied
    }

    /// Extra value of playing a knight *before* the roll rather than after.
    ///
    /// Moving the robber changes production on the roll that follows it. A
    /// knight played now therefore affects this turn's roll: it can free one
    /// of my own hexes so it pays out immediately, and it can block an
    /// opponent's hex so they miss this roll too. Played after the roll,
    /// neither applies until next time round.
    ///
    /// Discounted by the one-in-six chance this roll is a 7, which moves the
    /// robber again and throws the placement away.
    fn knight_preroll_premium(&self, v: &View) -> f64 {
        let topo = crate::board::topology();
        let b = v.board();
        let rh = v.robber();
        let rh_prob = crate::types::prob(b.numbers[rh as usize]);

        // Cards I am currently being denied on the blocked hex.
        let mut my_gain = 0.0;
        for &n in topo.hex_nodes[rh as usize].iter() {
            if let Some(pc) = v.node(n) {
                if pc.owner == v.me() {
                    my_gain += pc.building.yield_count() as f64 * rh_prob;
                }
            }
        }

        // Cards I could deny this roll by moving the robber onto someone.
        // Weighted toward the leader, and more so as the game nears its end.
        let leader = self.leader(v);
        let urgency = {
            let lead = v.leader_vp() as f64;
            let target = v.vp_target() as f64;
            // 0 early, rising to 1 as the leader closes on the target.
            ((lead - target + 4.0) / 4.0).clamp(0.0, 1.0)
        };
        let mut best_deny = 0.0f64;
        for h in 0..N_HEXES as u8 {
            if h == rh || b.hex_resource(h).is_none() {
                continue;
            }
            let p = crate::types::prob(b.numbers[h as usize]);
            let mut deny = 0.0;
            let mut hits_me = false;
            for &n in topo.hex_nodes[h as usize].iter() {
                if let Some(pc) = v.node(n) {
                    if pc.owner == v.me() {
                        hits_me = true;
                    } else {
                        let weight = if pc.owner == leader { 1.0 } else { 0.35 };
                        deny += weight * pc.building.yield_count() as f64 * p;
                    }
                }
            }
            if !hits_me {
                best_deny = best_deny.max(deny);
            }
        }

        (my_gain + urgency * best_deny) * (5.0 / 6.0)
    }

    /// Value of the best non-knight development card in hand, which playing a
    /// knight before the roll would forfeit -- only one card may be played
    /// per turn.
    fn alternative_dev_value(&self, v: &View) -> f64 {
        if v.dev_played_this_turn() {
            return 0.0;
        }
        let d = v.my_dev();
        let mut best = 0.0f64;
        if d[DevCard::RoadBuilding.idx()] > 0 {
            best = best.max(12.0 + self.w.w_road + 2.0 * self.reach_gain_from_two_roads(v));
        }
        if d[DevCard::Monopoly.idx()] > 0 {
            let est = Resource::ALL
                .iter()
                .map(|&r| self.estimate_opponent_holdings(v, r))
                .fold(0.0f64, f64::max);
            best = best.max(4.0 + 6.0 * est);
        }
        if d[DevCard::YearOfPlenty.idx()] > 0 {
            best = best.max(10.0);
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

    /// Greedy single-action score, without the plan bonus or jitter.
    fn score_base(&mut self, v: &View, a: &Action) -> f64 {
        let hand = *v.hand();
        let target = self.target_cost(v);
        let def_now = Self::deficit(&hand, &target);

        let base = match *a {
            Action::EndTurn => 0.0,
            // Rolling is the default before the roll; a knight has to earn
            // being played first, rather than being locked out by a constant.
            Action::Roll => 0.0,

            Action::PlaceInitialSettlement(n) => {
                100.0
                    + self.node_value(v, n)
                    + self.complement_bonus(v, n)
                    + self.setup_payout_value(v, n)
            }
            Action::PlaceInitialRoad(e) => 100.0 + self.road_value(v, e),

            Action::BuildCity(n) => {
                // A city doubles that node's yield.
                let base = self.w.w_city + 2.2 * v.board().node_pips(n) as f64;
                // Cities after the first compete with development cards for
                // the same ore and grain. Discounting them shifts that margin
                // without forbidding anything.
                if MAX_CITIES - v.cities_left() >= 1 {
                    // Two development cards cost two each of wool, grain and
                    // ore. When the hand covers both that and a city, this is
                    // the choice in question.
                    let two_devs: Res = [0, 0, 2, 2, 2];
                    let ore_rate = v.my_rate()[Resource::Ore.idx()];
                    let both_affordable = self.w.dev_over_second_city > 0.0
                        && can_afford(&hand, &two_devs)
                        && v.dev_deck_size() >= 2
                        && ore_rate >= self.w.dev_pref_min_ore
                        && ore_rate <= self.w.dev_pref_max_ore;
                    if both_affordable {
                        base - self.w.dev_over_second_city
                    } else {
                        base - self.w.later_city_discount
                    }
                } else {
                    base
                }
            }
            Action::BuildSettlement(n) => {
                self.w.w_settlement + 1.3 * self.node_value(v, n) + self.block_value(v, n)
            }
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
                if v.is_preroll() {
                    // Before the roll the question is not whether to play the
                    // knight -- it can still be played afterwards -- but
                    // whether playing it *now* is worth more. So score the
                    // timing difference alone, not the card.
                    //
                    // Against it: a 7 on this roll moves the robber again and
                    // throws the placement away, and playing now uses up the
                    // turn's one development card, which may be better spent.
                    let premium = self.w.knight_timing * self.knight_preroll_premium(v);
                    let forgone = (self.alternative_dev_value(v) - s).max(0.0);
                    premium - forgone - 1.0
                } else {
                    s
                }
            }
            Action::PlayRoadBuilding => {
                // Worth the four resources it saves, plus whatever the extra
                // reach opens up that was out of range before.
                //
                // Plowing to a new spot only pays off if it can be settled
                // before somebody else takes it, so the reach is worth far
                // more when a settlement is already affordable.
                // Conditioning this on holding settlement resources was
                // tried and is unreachable: whenever a settlement is
                // affordable the bot simply builds it, which outscores this
                // card by a wide margin, so the two states co-occur in only
                // 2% of plays. Capturing "plow two roads then settle the
                // better spot" needs the sequence scored as one package,
                // which this greedy single-action scorer cannot express.
                12.0 + self.w.w_road + 2.0 * self.reach_gain_from_two_roads(v)
            }
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

        base
    }

    /// Full score: greedy value, lifted onto the plan's scale if this action
    /// starts the plan, plus tie-breaking jitter.
    fn score(&mut self, v: &View, a: &Action) -> (f64, f64) {
        let base = self.score_base(v, a);
        let bonus = self.plan_bonus(v, a, base);
        let jitter = self.rng.random_range(0.0..1.0) * self.w.noise;
        (base + bonus + jitter, bonus)
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
        if matches!(actions[0], Action::PlaceInitialSettlement(_)) {
            // A forced opening takes priority: this is the first or second
            // settlement depending on how many are still in hand.
            let which = (MAX_SETTLEMENTS - v.settlements_left()) as usize;
            if let Some(&want) = self.w.forced_opening.get(which) {
                if let Some(i) = actions.iter().position(
                    |a| matches!(a, Action::PlaceInitialSettlement(n) if *n == want),
                ) {
                    return i;
                }
            }
            // Exogenous opening: pick uniformly among the legal intersections
            // so that opening position is uncorrelated with everything else.
            if self.w.random_opening {
                return self.rng.random_range(0..actions.len());
            }
        }
        // The road geography is only consulted by placement, road, Road
        // Building and pre-roll knight decisions. Most turns end with a list
        // of trades and EndTurn, where computing it would be wasted work.
        if actions.iter().any(|a| {
            matches!(
                a,
                Action::BuildRoad(_)
                    | Action::BuildSettlement(_)
                    | Action::PlaceInitialRoad(_)
                    | Action::PlaceInitialSettlement(_)
                    | Action::PlayRoadBuilding
                    | Action::PlayKnight
            )
        }) {
            self.build_ctx(v);
        }
        // Score once with the plan bonus and once without, so it is visible
        // whether the plan actually changed the decision.
        let mut best = 0usize;
        let mut best_s = f64::MIN;
        let mut best_noplan = 0usize;
        let mut best_s_noplan = f64::MIN;
        let mut best_bonus = 0.0f64;
        for (i, a) in actions.iter().enumerate() {
            let (s, bonus) = self.score(v, a);
            if s > best_s {
                best_s = s;
                best = i;
                best_bonus = bonus;
            }
            let sn = s - bonus;
            if sn > best_s_noplan {
                best_s_noplan = sn;
                best_noplan = i;
            }
        }
        if self.ctx.plan.is_some() && best_bonus > 0.0 {
            PLAN_CHOSEN.fetch_add(1, Ordering::Relaxed);
            if best != best_noplan {
                PLAN_FLIPPED.fetch_add(1, Ordering::Relaxed);
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

/// A seat's strategy: either the random baseline or a weight vector.
#[derive(Clone, Debug)]
pub enum Strategy {
    Random,
    Heuristic(Weights),
}

impl Strategy {
    /// Parse a preset name, `preset+key=value,...` to override weights, or
    /// `label:preset+...` to also give it a distinct name in reports.
    ///
    /// The label matters: two tuned variants of the same preset would
    /// otherwise both report as "balanced" and be pooled together, hiding the
    /// very comparison being run.
    pub fn parse(spec: &str) -> Option<Strategy> {
        // A label is everything before the first ':' -- but only when what
        // follows still parses, so a bare name is never mistaken for one.
        if let Some((label, rest)) = spec.split_once(':') {
            if let Some(st) = Strategy::parse_inner(rest) {
                return Some(st.relabel(label));
            }
        }
        Strategy::parse_inner(spec)
    }

    fn parse_inner(spec: &str) -> Option<Strategy> {
        if spec == "random" {
            return Some(Strategy::Random);
        }
        Weights::from_spec(spec).map(Strategy::Heuristic)
    }

    /// The name this strategy is reported under.
    pub fn label(&self) -> String {
        match self {
            Strategy::Random => "random".into(),
            Strategy::Heuristic(w) => w.label.clone(),
        }
    }

    pub fn relabel(mut self, label: &str) -> Strategy {
        if let Strategy::Heuristic(w) = &mut self {
            w.label = label.to_string();
        }
        self
    }

    /// True when this is the plain random baseline, which cannot be relabelled.
    pub fn is_random(&self) -> bool {
        matches!(self, Strategy::Random)
    }

    /// Force the opening settlements to specific intersections.
    pub fn with_forced_opening(mut self, nodes: Vec<u8>) -> Strategy {
        if let Strategy::Heuristic(w) = &mut self {
            w.forced_opening = nodes;
        }
        self
    }

    /// Place the opening settlements at random, for placement experiments.
    pub fn with_random_opening(mut self) -> Strategy {
        if let Strategy::Heuristic(w) = &mut self {
            w.random_opening = true;
        }
        self
    }

    pub fn weights(&self) -> Option<&Weights> {
        match self {
            Strategy::Heuristic(w) => Some(w),
            Strategy::Random => None,
        }
    }

    pub fn make(&self, seed: u64) -> Box<dyn Policy> {
        match self {
            Strategy::Random => Box::new(RandomBot::new(seed)),
            Strategy::Heuristic(w) => Box::new(HeuristicBot::new(w.clone(), seed)),
        }
    }
}

/// Build a policy from a preset name or weight spec.
pub fn make(spec: &str, seed: u64) -> Option<Box<dyn Policy>> {
    Strategy::parse(spec).map(|s| s.make(seed))
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
