//! Per-player accounting, built so that luck and skill can be separated.
//!
//! The chain from board position to victory points is:
//!
//! ```text
//! placement quality --> expected inflow --> dice deviation --> actual inflow
//!     --> conversion efficiency --> victory points
//! ```
//!
//! Placement quality and conversion efficiency are things a player controls;
//! the dice deviation is not. We record each link separately so an experiment
//! can ask "did this strategy win because it played well, or because it
//! rolled well?"
//!
//! The key accumulator is `expected_cards`: on **every** die roll in the game
//! (not just the player's own turns) we add the player's expected yield given
//! the buildings they have at that moment and where the robber is. Summed over
//! a game it answers "what should this board position have produced, with
//! average dice, over exactly this many rolls?" Comparing it to `produced`
//! isolates dice luck.

use crate::types::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlayerMetrics {
    // ---- production and the dice ----
    /// Cards actually received from hex production.
    pub produced: [i32; N_RES],
    /// Expected cards from production under average dice, accumulated over
    /// every roll of the game. See the module comment.
    pub expected_cards: [f64; N_RES],
    /// Cards forgone because the robber was sitting on one of their hexes.
    pub robber_blocked: i32,
    /// Cards the player was owed but the bank could not pay. A real effect of
    /// the official shortage rule, and not the dice's fault -- so it is held
    /// apart from `dice_luck`.
    pub bank_shortage_loss: i32,

    // ---- other inflows and outflows ----
    pub stolen_from_others: i32,
    pub lost_to_steal: i32,
    pub lost_to_discard: i32,
    pub monopoly_gain: i32,
    pub monopoly_loss: i32,
    pub year_of_plenty_gain: i32,
    /// Cards handed out for the second opening settlement.
    pub setup_income: i32,
    /// Cards received / given away in bank or port trades.
    pub bank_trade_in: i32,
    pub bank_trade_out: i32,
    /// Cards received / given away in player-to-player trades.
    pub player_trade_in: i32,
    pub player_trade_out: i32,
    /// Cards spent on roads, settlements, cities and dev cards.
    pub spent: [i32; N_RES],

    // ---- activity counts ----
    pub turns: u32,
    pub rolls_seen: u32,
    pub sevens_rolled: u32,
    pub dice_sum: u32,
    pub dev_bought: u32,
    pub knights_played: u32,
    /// Knights played before the roll, to capture this turn's production.
    pub knights_preroll: u32,
    pub monopolies_played: u32,
    pub yops_played: u32,
    pub road_buildings_played: u32,
    /// Road Buildings played while a settlement was already affordable --
    /// i.e. where plowing to a new spot could be completed at once.
    pub road_buildings_settle_ready: u32,
    pub roads_built: u32,
    pub settlements_built: u32,
    pub cities_built: u32,
    pub trades_proposed: u32,
    pub trades_accepted_by_others: u32,
    pub trades_accepted_for_others: u32,
    pub bank_trades: u32,
    pub discards: u32,
    pub times_robbed: u32,

    // ---- position quality ----
    /// Sum over the player's own turns of the total pips of their buildings
    /// (settlements once, cities twice). Divided by turns it gives the average
    /// "engine size" they played with.
    pub pip_turns: f64,
    /// Pips of the player's buildings at the end of the game.
    pub final_pips: u32,
    /// Number of distinct resources the player's buildings touch, at the end.
    pub final_res_breadth: u8,
    /// Nodes chosen during initial placement, in order.
    pub init_nodes: Vec<u8>,
    /// Total pips of the two opening settlements.
    pub init_pips: u32,
    /// Everything about the opening position, measured once setup ends.
    pub opening: OpeningFeatures,
    /// Port access at the end of the game, as a bitmask (see `PlayerState`).
    pub final_port_mask: u8,
    /// Position at each phase boundary, in order.
    pub snapshots: Vec<PhaseSnapshot>,

    // ---- outcome ----
    pub vp_final: u8,
    pub vp_from_cards: u8,
    pub had_longest_road: bool,
    pub had_largest_army: bool,
    /// Turn on which the player first reached each VP level, for pacing plots.
    pub turn_reached_vp: Vec<(u8, u32)>,
    pub last_vp_recorded: u8,
}

impl PlayerMetrics {
    #[inline]
    pub fn produced_total(&self) -> i32 {
        self.produced.iter().sum()
    }
    #[inline]
    pub fn expected_total(&self) -> f64 {
        self.expected_cards.iter().sum()
    }
    /// Cards above (or below) what average dice would have given this board
    /// position. Positive means the dice were kind.
    ///
    /// Cards the bank could not pay are added back in: they were earned by the
    /// dice and denied by a supply limit, so counting them as bad luck would
    /// confound the two.
    #[inline]
    pub fn dice_luck(&self) -> f64 {
        (self.produced_total() + self.bank_shortage_loss) as f64 - self.expected_total()
    }

    /// Cards that actually reached the hand, against the same baseline.
    #[inline]
    pub fn net_production_delta(&self) -> f64 {
        self.produced_total() as f64 - self.expected_total()
    }
    /// Cards gained and lost through other players' actions against you:
    /// theft, robber blocking and monopolies.
    ///
    /// Held apart from discards on purpose. Discarding on a 7 is mostly a
    /// consequence of the player's *own* decision to sit on a large hand, so
    /// it is a style choice rather than luck; folding it in here would make
    /// the skill index adjust away a skill. Even this figure is only partly
    /// exogenous, since opponents aim the robber at whoever is winning.
    #[inline]
    pub fn theft_luck(&self) -> f64 {
        (self.stolen_from_others - self.lost_to_steal - self.robber_blocked
            + self.monopoly_gain
            - self.monopoly_loss) as f64
    }

    /// Cards thrown away on a 7. Driven by the player's own hoarding.
    #[inline]
    pub fn discarded(&self) -> f64 {
        self.lost_to_discard as f64
    }
    /// Every card that ever entered the player's hand.
    #[inline]
    pub fn total_inflow(&self) -> i32 {
        self.produced_total()
            + self.stolen_from_others
            + self.monopoly_gain
            + self.year_of_plenty_gain
            + self.bank_trade_in
            + self.player_trade_in
            + self.setup_income
    }
    /// Cards spent on anything that advances the game.
    #[inline]
    pub fn total_spent(&self) -> i32 {
        self.spent.iter().sum()
    }
    /// Victory points per 100 cards of inflow -- a raw conversion rate.
    pub fn vp_per_100_cards(&self) -> f64 {
        let i = self.total_inflow();
        if i == 0 {
            return 0.0;
        }
        100.0 * self.vp_final as f64 / i as f64
    }
    /// Average dice roll the player saw (sanity check: should approach 7).
    pub fn mean_roll(&self) -> f64 {
        if self.rolls_seen == 0 {
            0.0
        } else {
            self.dice_sum as f64 / self.rolls_seen as f64
        }
    }

    pub fn note_vp(&mut self, vp: u8, turn: u32) {
        if vp > self.last_vp_recorded {
            for v in (self.last_vp_recorded + 1)..=vp {
                self.turn_reached_vp.push((v, turn));
            }
            self.last_vp_recorded = vp;
        }
    }
}

/// A player's position at a phase boundary.
///
/// Phases are marked by the *leader's* score rather than by turn number,
/// because game lengths vary: "the game is two thirds done" is a statement
/// about how close someone is to winning, not about how many turns have
/// passed. Early ends when the leader first reaches 40% of the target, mid
/// when they reach 70%, and late runs to the end.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PhaseSnapshot {
    /// 0 = end of early, 1 = end of mid.
    pub marker: u8,
    /// Turns elapsed in the game as a whole.
    pub game_turn: u32,
    /// Turns this player has taken.
    pub own_turns: u32,
    pub vp: u8,
    /// Score visible to the table: excludes hidden victory-point cards. This
    /// is what a spectator overlay could actually read off a live game.
    pub vp_public: u8,
    /// Resource cards in hand. The count is public; the contents are not.
    pub hand_size: i16,
    /// Unplayed development cards held. Count public, faces hidden.
    pub dev_in_hand: u8,
    pub has_longest_road: bool,
    pub has_largest_army: bool,
    /// Distinct resources the player's buildings can produce.
    pub breadth: u8,
    /// Every card taken in so far.
    pub inflow: i32,
    /// Engine size: pips of their buildings.
    pub pips: u32,
    pub settlements: u8,
    pub cities: u8,
    pub roads: u8,
    pub dev_bought: u32,
    pub knights: u8,
    /// True if this player was level with the best score at the boundary.
    pub leading: bool,
}

/// Measurable properties of an opening position.
///
/// These are the regressors for valuing placement. They are recorded once
/// setup ends, before any die is rolled, so nothing that happens later can
/// have influenced them.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OpeningFeatures {
    /// Pips by resource across both opening settlements.
    pub pips: [u32; N_RES],
    pub total_pips: u32,
    /// How many of the five resources the opening touches.
    pub breadth: u8,
    /// Brick and lumber pips arriving on the same number -- the road bundle
    /// paying out in one go.
    pub paired_road: u32,
    /// Grain and ore pips arriving on the same number -- the city bundle.
    pub paired_city: u32,
    /// Any two resources arriving on the same number.
    pub paired_any: u32,
    /// A 2:1 port sits on an opening settlement, by resource.
    pub port2: [bool; N_RES],
    /// A 3:1 port sits on an opening settlement.
    pub port3: bool,
    /// Roads needed to reach the nearest port of any kind.
    pub port_dist: u8,
    /// Roads needed to reach the nearest 3:1 port.
    pub port3_dist: u8,
    /// Roads needed to reach the nearest 2:1 port, by resource.
    pub port2_dist: [u8; N_RES],
    /// The number tokens the opening touches, for ad hoc analysis.
    pub numbers: Vec<u8>,
}

/// One player's line in a finished game.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerResult {
    /// Index in the seating order (0 places first).
    pub seat: u8,
    pub policy: String,
    pub won: bool,
    pub vp: u8,
    pub metrics: PlayerMetrics,
}

/// One finished game.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GameRecord {
    pub seed: u64,
    pub turns: u32,
    pub winner: Option<u8>,
    /// True if the game hit the turn limit without a winner.
    pub timed_out: bool,
    pub players: Vec<PlayerResult>,
    /// Count of each dice outcome 2..=12, for verifying the dice.
    pub roll_histogram: [u32; 13],
}

impl GameRecord {
    pub fn total_rolls(&self) -> u32 {
        self.roll_histogram.iter().sum()
    }
}
