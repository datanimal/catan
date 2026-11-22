//! A Settlers of Catan simulator built for large-scale strategy experiments.
//!
//! Modules:
//! - [`board`]  board topology (derived from hex geometry) and layout
//! - [`state`]  mutable game state
//! - [`rules`]  legality and action enumeration
//! - [`game`]   the game driver
//! - [`policy`] the bot interface and the information a bot may see
//! - [`bots`]   strategy archetypes
//! - [`metrics`] per-player accounting that separates luck from skill
//! - [`sim`]    parallel batch runner
//! - [`analysis`] aggregation, including the skill index

pub mod action;
pub mod analysis;
pub mod board;
pub mod bots;
pub mod game;
pub mod invariants;
pub mod metrics;
pub mod placement;
pub mod policy;
pub mod render;
pub mod rules;
pub mod sim;
pub mod state;
pub mod tune;
pub mod types;
