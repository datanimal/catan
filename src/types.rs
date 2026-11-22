//! Core value types: resources, terrain, dev cards, costs, limits.

use serde::{Deserialize, Serialize};

pub const N_NODES: usize = 54;
pub const N_EDGES: usize = 72;
pub const N_HEXES: usize = 19;
pub const N_RES: usize = 5;

/// A bundle of resources, indexed by `Resource as usize`.
pub type Res = [i16; N_RES];

pub const ZERO: Res = [0; N_RES];

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(u8)]
pub enum Resource {
    Brick = 0,
    Lumber = 1,
    Wool = 2,
    Grain = 3,
    Ore = 4,
}

impl Resource {
    pub const ALL: [Resource; N_RES] = [
        Resource::Brick,
        Resource::Lumber,
        Resource::Wool,
        Resource::Grain,
        Resource::Ore,
    ];
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    #[inline]
    pub fn from_idx(i: usize) -> Resource {
        Resource::ALL[i]
    }
    pub fn short(self) -> char {
        match self {
            Resource::Brick => 'B',
            Resource::Lumber => 'L',
            Resource::Wool => 'W',
            Resource::Grain => 'G',
            Resource::Ore => 'O',
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Resource::Brick => "brick",
            Resource::Lumber => "lumber",
            Resource::Wool => "wool",
            Resource::Grain => "grain",
            Resource::Ore => "ore",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Terrain {
    Hills,
    Forest,
    Pasture,
    Fields,
    Mountains,
    Desert,
}

impl Terrain {
    #[inline]
    pub fn resource(self) -> Option<Resource> {
        match self {
            Terrain::Hills => Some(Resource::Brick),
            Terrain::Forest => Some(Resource::Lumber),
            Terrain::Pasture => Some(Resource::Wool),
            Terrain::Fields => Some(Resource::Grain),
            Terrain::Mountains => Some(Resource::Ore),
            Terrain::Desert => None,
        }
    }
    /// Official base-game terrain distribution (19 hexes).
    pub fn standard_bag() -> Vec<Terrain> {
        let mut v = Vec::with_capacity(19);
        v.extend(std::iter::repeat_n(Terrain::Forest, 4));
        v.extend(std::iter::repeat_n(Terrain::Pasture, 4));
        v.extend(std::iter::repeat_n(Terrain::Fields, 4));
        v.extend(std::iter::repeat_n(Terrain::Hills, 3));
        v.extend(std::iter::repeat_n(Terrain::Mountains, 3));
        v.push(Terrain::Desert);
        v
    }
}

/// Official base-game number tokens (18, one per non-desert hex).
pub fn standard_numbers() -> Vec<u8> {
    vec![2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
}

/// Number of ways to roll `n` with 2d6, i.e. the "pips" on the number token.
#[inline]
pub fn pips(n: u8) -> u32 {
    if n < 2 || n > 12 {
        return 0;
    }
    (6 - (7i32 - n as i32).abs()) as u32
}

/// Probability that a given hex produces on a single roll.
#[inline]
pub fn prob(n: u8) -> f64 {
    pips(n) as f64 / 36.0
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum DevCard {
    Knight,
    VictoryPoint,
    RoadBuilding,
    YearOfPlenty,
    Monopoly,
}

impl DevCard {
    pub const ALL: [DevCard; 5] = [
        DevCard::Knight,
        DevCard::VictoryPoint,
        DevCard::RoadBuilding,
        DevCard::YearOfPlenty,
        DevCard::Monopoly,
    ];
    pub fn idx(self) -> usize {
        match self {
            DevCard::Knight => 0,
            DevCard::VictoryPoint => 1,
            DevCard::RoadBuilding => 2,
            DevCard::YearOfPlenty => 3,
            DevCard::Monopoly => 4,
        }
    }
    /// Official 25-card development deck.
    pub fn standard_deck() -> Vec<DevCard> {
        let mut v = Vec::with_capacity(25);
        v.extend(std::iter::repeat_n(DevCard::Knight, 14));
        v.extend(std::iter::repeat_n(DevCard::VictoryPoint, 5));
        v.extend(std::iter::repeat_n(DevCard::RoadBuilding, 2));
        v.extend(std::iter::repeat_n(DevCard::YearOfPlenty, 2));
        v.extend(std::iter::repeat_n(DevCard::Monopoly, 2));
        v
    }
}

// ---- Costs ----
pub const COST_ROAD: Res = [1, 1, 0, 0, 0];
pub const COST_SETTLEMENT: Res = [1, 1, 1, 1, 0];
pub const COST_CITY: Res = [0, 0, 0, 2, 3];
pub const COST_DEV: Res = [0, 0, 1, 1, 1];

// ---- Piece limits per player ----
pub const MAX_ROADS: u8 = 15;
pub const MAX_SETTLEMENTS: u8 = 5;
pub const MAX_CITIES: u8 = 4;
/// Bank holds 19 of each resource.
pub const BANK_PER_RES: i16 = 19;

#[inline]
pub fn res_total(r: &Res) -> i16 {
    r[0] + r[1] + r[2] + r[3] + r[4]
}

#[inline]
pub fn can_afford(have: &Res, cost: &Res) -> bool {
    (0..N_RES).all(|i| have[i] >= cost[i])
}

#[inline]
pub fn res_sub(a: &mut Res, b: &Res) {
    for i in 0..N_RES {
        a[i] -= b[i];
    }
}

#[inline]
pub fn res_add(a: &mut Res, b: &Res) {
    for i in 0..N_RES {
        a[i] += b[i];
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Building {
    Settlement,
    City,
}

impl Building {
    pub fn yield_count(self) -> i16 {
        match self {
            Building::Settlement => 1,
            Building::City => 2,
        }
    }
    pub fn vp(self) -> u8 {
        match self {
            Building::Settlement => 1,
            Building::City => 2,
        }
    }
}
