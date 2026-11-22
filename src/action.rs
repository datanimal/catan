//! Player actions. The engine enumerates every legal action; a policy picks one.

use crate::types::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Action {
    /// Setup: place a settlement on a node (free, distance rule applies).
    PlaceInitialSettlement(u8),
    /// Setup: place a road on an edge touching the settlement just placed.
    PlaceInitialRoad(u8),

    Roll,
    BuildRoad(u8),
    BuildSettlement(u8),
    BuildCity(u8),
    BuyDev,

    PlayKnight,
    PlayRoadBuilding,
    PlayYearOfPlenty(Resource, Resource),
    PlayMonopoly(Resource),

    /// Trade with the bank or a port at `rate`:1.
    BankTrade {
        give: Resource,
        want: Resource,
        rate: u8,
    },
    /// Offer `give` for `want` to the other players.
    ProposeTrade {
        give: Res,
        want: Res,
    },

    /// Move the robber and optionally rob a player with a building there.
    MoveRobber {
        hex: u8,
        victim: Option<u8>,
    },

    EndTurn,
}

impl Action {
    pub fn is_build(&self) -> bool {
        matches!(
            self,
            Action::BuildRoad(_) | Action::BuildSettlement(_) | Action::BuildCity(_)
        )
    }
    /// Short label used for action-mix statistics.
    pub fn kind(&self) -> &'static str {
        match self {
            Action::PlaceInitialSettlement(_) => "init_settlement",
            Action::PlaceInitialRoad(_) => "init_road",
            Action::Roll => "roll",
            Action::BuildRoad(_) => "road",
            Action::BuildSettlement(_) => "settlement",
            Action::BuildCity(_) => "city",
            Action::BuyDev => "buy_dev",
            Action::PlayKnight => "knight",
            Action::PlayRoadBuilding => "road_building",
            Action::PlayYearOfPlenty(..) => "year_of_plenty",
            Action::PlayMonopoly(_) => "monopoly",
            Action::BankTrade { .. } => "bank_trade",
            Action::ProposeTrade { .. } => "propose_trade",
            Action::MoveRobber { .. } => "move_robber",
            Action::EndTurn => "end_turn",
        }
    }
}
