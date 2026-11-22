//! The bot interface.
//!
//! A policy never sees `GameState` directly. It gets a [`View`], whose
//! accessors expose only what a player at the table could know: the board,
//! every piece on it, hand *sizes*, cards played face-up, and its own hand.
//! Opponents' resource contents and unplayed development cards stay hidden.
//! Keeping this boundary means a strategy cannot accidentally win by cheating,
//! which would quietly invalidate every experiment.

use crate::action::Action;
use crate::board::{Board, PortKind};
use crate::state::{GameState, Piece};
use crate::types::*;

pub struct View<'a> {
    st: &'a GameState,
    me: u8,
}

impl<'a> View<'a> {
    pub(crate) fn new(st: &'a GameState, me: u8) -> View<'a> {
        View { st, me }
    }

    // ---- identity ----
    #[inline]
    pub fn me(&self) -> u8 {
        self.me
    }
    #[inline]
    pub fn n_players(&self) -> usize {
        self.st.n_players()
    }
    #[inline]
    pub fn turn(&self) -> u32 {
        self.st.turn
    }
    #[inline]
    pub fn vp_target(&self) -> u8 {
        self.st.vp_target
    }
    #[inline]
    pub fn hand_limit(&self) -> i16 {
        self.st.hand_limit
    }

    // ---- board (fully public) ----
    #[inline]
    pub fn board(&self) -> &Board {
        &self.st.board
    }
    #[inline]
    pub fn node(&self, n: u8) -> Option<Piece> {
        self.st.node[n as usize]
    }
    #[inline]
    pub fn edge(&self, e: u8) -> Option<u8> {
        self.st.edge[e as usize]
    }
    #[inline]
    pub fn robber(&self) -> u8 {
        self.st.board.robber
    }
    #[inline]
    pub fn bank(&self) -> &Res {
        &self.st.bank
    }
    #[inline]
    pub fn dev_deck_size(&self) -> usize {
        self.st.dev_deck.len()
    }
    #[inline]
    pub fn last_roll(&self) -> u8 {
        self.st.last_roll
    }

    // ---- own private information ----
    #[inline]
    pub fn hand(&self) -> &Res {
        &self.st.players[self.me as usize].hand
    }
    /// Playable development cards in hand, by `DevCard::idx`.
    #[inline]
    pub fn my_dev(&self) -> &[u8; 5] {
        &self.st.players[self.me as usize].dev
    }
    #[inline]
    pub fn my_dev_new(&self) -> &[u8; 5] {
        &self.st.players[self.me as usize].dev_new
    }
    #[inline]
    pub fn roads_left(&self) -> u8 {
        self.st.players[self.me as usize].roads_left
    }
    #[inline]
    pub fn settlements_left(&self) -> u8 {
        self.st.players[self.me as usize].settlements_left
    }
    #[inline]
    pub fn cities_left(&self) -> u8 {
        self.st.players[self.me as usize].cities_left
    }
    #[inline]
    pub fn free_roads(&self) -> u8 {
        self.st.free_roads
    }
    /// Exchange rate for giving away `r` (2, 3 or 4 to 1).
    #[inline]
    pub fn trade_rate(&self, r: Resource) -> u8 {
        self.st.players[self.me as usize].trade_rate(r)
    }
    pub fn my_ports(&self) -> Vec<PortKind> {
        self.st.port_kinds(self.me)
    }
    /// My true victory points, hidden cards included.
    #[inline]
    pub fn my_vp(&self) -> u8 {
        self.st.vp(self.me)
    }

    // ---- public information about everyone ----
    /// Number of resource cards held -- public, the contents are not.
    #[inline]
    pub fn hand_size(&self, p: u8) -> i16 {
        self.st.players[p as usize].hand_size()
    }
    /// Number of unplayed development cards held -- public, their faces not.
    #[inline]
    pub fn dev_count(&self, p: u8) -> u8 {
        self.st.players[p as usize].dev_in_hand()
    }
    #[inline]
    pub fn knights_played(&self, p: u8) -> u8 {
        self.st.players[p as usize].knights_played
    }
    #[inline]
    pub fn road_len(&self, p: u8) -> u8 {
        self.st.players[p as usize].road_len
    }
    #[inline]
    pub fn largest_army(&self) -> Option<u8> {
        self.st.largest_army
    }
    #[inline]
    pub fn longest_road(&self) -> Option<u8> {
        self.st.longest_road
    }
    /// Victory points an opponent is *known* to have. For yourself this still
    /// hides nothing, since `my_vp` is available.
    #[inline]
    pub fn vp_public(&self, p: u8) -> u8 {
        self.st.vp_public(p)
    }
    /// The highest public score at the table -- how close the race is.
    pub fn leader_vp(&self) -> u8 {
        (0..self.n_players() as u8)
            .map(|p| {
                if p == self.me {
                    self.my_vp()
                } else {
                    self.vp_public(p)
                }
            })
            .max()
            .unwrap_or(0)
    }

    /// Expected cards per roll from my current buildings, by resource.
    pub fn my_rate(&self) -> [f64; N_RES] {
        self.st.expected_rate(self.me)
    }
    pub fn rate_of(&self, p: u8) -> [f64; N_RES] {
        self.st.expected_rate(p)
    }

    // ---- convenience wrappers around the rules ----
    pub fn settlement_spots(&self) -> Vec<u8> {
        crate::rules::settlement_spots(self.st, self.me)
    }
    pub fn road_spots(&self) -> Vec<u8> {
        crate::rules::road_spots(self.st, self.me)
    }
    pub fn city_spots(&self) -> Vec<u8> {
        crate::rules::city_spots(self.st, self.me)
    }
    pub fn node_open(&self, n: u8) -> bool {
        crate::rules::node_open(self.st, n)
    }
    pub fn robbery_victims(&self, hex: u8) -> Vec<u8> {
        crate::rules::robbery_victims(self.st, hex, self.me)
    }
}

/// A strategy. One instance per player per game, so policies may keep state.
pub trait Policy: Send {
    fn name(&self) -> String;

    /// Pick one of `actions` (guaranteed non-empty). Return an index.
    fn choose(&mut self, view: &View, actions: &[Action]) -> usize;

    /// Choose `n` cards to discard after a 7. The engine validates and falls
    /// back to trimming the largest piles if the answer is not legal.
    fn discard(&mut self, view: &View, n: i16) -> Res {
        default_discard(view.hand(), n)
    }

    /// Respond to an offer: `offer_to_me` is what I would receive,
    /// `asked_of_me` what I would hand over. Default is to refuse.
    fn accept_trade(
        &mut self,
        _view: &View,
        _offer_to_me: &Res,
        _asked_of_me: &Res,
        _from: u8,
    ) -> bool {
        false
    }
}

/// Discard from the largest piles first, which is what the engine falls back
/// to if a policy returns something illegal.
pub fn default_discard(hand: &Res, n: i16) -> Res {
    let mut left = hand.to_owned();
    let mut out = ZERO;
    for _ in 0..n {
        let mut best = 0usize;
        for i in 1..N_RES {
            if left[i] > left[best] {
                best = i;
            }
        }
        if left[best] == 0 {
            break;
        }
        left[best] -= 1;
        out[best] += 1;
    }
    out
}
