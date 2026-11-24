//! Ranking opening intersections, using the values fitted in `placement`.
//!
//! The numbers here are not invented: they come from the randomised-opening
//! experiment, where 60,000 games were played with every opening placed
//! uniformly at random so that position was uncorrelated with strategy, and
//! luck-adjusted victory points were regressed on what the position contained.
//! See the placement section of the README.
//!
//! Everything is quoted in **adjusted pips**: one adjusted pip is one pip of
//! average production, so a spot worth 12.0 adjusted pips is worth as much as
//! 12 pips of average resources, however its own mix is made up.

use crate::board::{Board, PortKind};
use crate::types::*;

/// Value of one pip of each resource, in adjusted pips. Fitted with the tuned
/// strategy over openings of 14+ pips. Ore is the one that moves most with the
/// strategy playing (1.05 here, 1.48 for a development-card-heavy style), so
/// treat the gap between brick, ore and grain as narrower than it looks.
pub const RES_VALUE: [f64; N_RES] = [
    1.10, // brick
    0.87, // lumber
    0.85, // wool
    1.19, // grain
    1.05, // ore
];

/// Extra value per pip of brick and lumber arriving on the *same* number: a
/// road is one of each, so synchronised delivery means building instead of
/// holding half a bundle.
pub const PAIRED_ROAD: f64 = 0.42;
/// The same for grain and ore, which a city needs together.
pub const PAIRED_CITY: f64 = 0.33;
/// Per extra distinct resource, holding pip counts fixed. Measured across the
/// opening *pair*, so for a single intersection it is reported separately
/// rather than folded into the headline number.
pub const PER_DISTINCT_RESOURCE: f64 = 2.65;
/// A 2:1 port gains roughly this much per pip of its own resource. It is worth
/// nothing without production of that resource to trade.
pub const PORT_2_PER_PIP: f64 = 0.50;
/// A 3:1 port at a typical opening.
pub const PORT_3_FLAT: f64 = 1.86;

#[derive(Clone, Debug)]
pub struct SpotValue {
    pub node: u8,
    /// Every way of naming this intersection, e.g. "9:N".
    pub names: Vec<String>,
    pub pips: u32,
    pub by_res: [u32; N_RES],
    pub distinct: u8,
    pub paired_road: u32,
    pub paired_city: u32,
    pub ports: Vec<PortKind>,
    /// Production value alone, in adjusted pips.
    pub production: f64,
    /// Bonus for correlated brick/lumber or grain/ore.
    pub pairing: f64,
    pub port_value: f64,
    /// The headline: production + pairing + port.
    pub total: f64,
    /// The hexes this intersection touches, as "grain-10" and so on.
    pub touches: Vec<String>,
}

pub fn rank(board: &Board) -> Vec<SpotValue> {
    let topo = crate::board::topology();
    let mut out: Vec<SpotValue> = (0..N_NODES as u8)
        .map(|n| {
            let by_res = board.node_pips_by_res(n);
            let pips = by_res.iter().sum::<u32>();
            let distinct = by_res.iter().filter(|&&v| v > 0).count() as u8;
            let production: f64 = (0..N_RES)
                .map(|i| by_res[i] as f64 * RES_VALUE[i])
                .sum();
            let paired_road = board.paired_pips(&[n], Resource::Brick, Resource::Lumber);
            let paired_city = board.paired_pips(&[n], Resource::Grain, Resource::Ore);
            let pairing =
                PAIRED_ROAD * paired_road as f64 + PAIRED_CITY * paired_city as f64;

            let mask = board.node_port_mask[n as usize];
            let mut ports = Vec::new();
            let mut port_value = 0.0;
            for r in Resource::ALL {
                if mask & (1 << r.idx()) != 0 {
                    ports.push(PortKind::Specific(r));
                    port_value += PORT_2_PER_PIP * by_res[r.idx()] as f64;
                }
            }
            if mask & (1 << 5) != 0 {
                ports.push(PortKind::Generic);
                port_value += PORT_3_FLAT;
            }

            let touches = topo.node_hexes[n as usize]
                .iter()
                .map(|&h| match board.hex_resource(h) {
                    Some(r) => format!("{}-{}", r.name(), board.numbers[h as usize]),
                    None => "desert".to_string(),
                })
                .collect();

            SpotValue {
                node: n,
                names: crate::board::node_names(n),
                pips,
                by_res,
                distinct,
                paired_road,
                paired_city,
                ports,
                production,
                pairing,
                port_value,
                total: production + pairing + port_value,
                touches,
            }
        })
        .collect();
    out.sort_by(|a, b| b.total.partial_cmp(&a.total).unwrap());
    out
}

/// Spots that remain legal once `taken` are built on, under the distance rule.
pub fn still_open(board: &Board, taken: &[u8]) -> Vec<SpotValue> {
    let topo = crate::board::topology();
    let blocked: Vec<u8> = taken
        .iter()
        .flat_map(|&n| {
            std::iter::once(n).chain(topo.node_neighbors[n as usize].iter().copied())
        })
        .collect();
    rank(board)
        .into_iter()
        .filter(|s| !blocked.contains(&s.node))
        .collect()
}
