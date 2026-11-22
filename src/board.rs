//! Board topology and layout.
//!
//! The 54-node / 72-edge graph is *derived* from hex geometry rather than
//! hand-tabulated, so it is self-checking: see `tests` at the bottom.
//!
//! Coordinates: pointy-top hexes in axial coords (q, r). A hex centre maps to
//! the integer point (2q + r, 3r); its six corners are the centre plus
//! {(0,2),(1,1),(1,-1),(0,-2),(-1,-1),(-1,1)}. Every corner of every hex thus
//! lands on an integer lattice point, and deduplicating them yields exactly
//! the 54 intersections of a standard board.

use crate::types::*;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

/// The 19 land hexes of a standard board, in rows of 3/4/5/4/3.
pub const HEX_AXIAL: [(i32, i32); N_HEXES] = [
    (0, -2),
    (1, -2),
    (2, -2),
    (-1, -1),
    (0, -1),
    (1, -1),
    (2, -1),
    (-2, 0),
    (-1, 0),
    (0, 0),
    (1, 0),
    (2, 0),
    (-2, 1),
    (-1, 1),
    (0, 1),
    (1, 1),
    (-2, 2),
    (-1, 2),
    (0, 2),
];

const CORNER_OFFSETS: [(i32, i32); 6] = [(0, 2), (1, 1), (1, -1), (0, -2), (-1, -1), (-1, 1)];

/// Immutable board graph, shared by every game.
pub struct Topology {
    /// Node ids of each hex's 6 corners (clockwise from top).
    pub hex_nodes: [[u8; 6]; N_HEXES],
    /// Hexes touching each node (1..=3).
    pub node_hexes: Vec<Vec<u8>>,
    /// Nodes adjacent to each node (2..=3).
    pub node_neighbors: Vec<Vec<u8>>,
    /// Edges incident to each node (2..=3).
    pub node_edges: Vec<Vec<u8>>,
    /// The two endpoints of each edge.
    pub edge_nodes: [[u8; 2]; N_EDGES],
    /// Integer lattice coordinate of each node (for rendering).
    pub node_xy: Vec<(i32, i32)>,
    /// Coastal edges in perimeter-walk order (30 of them).
    pub coast_cycle: Vec<u8>,
}

static TOPOLOGY: OnceLock<Topology> = OnceLock::new();

pub fn topology() -> &'static Topology {
    TOPOLOGY.get_or_init(build_topology)
}

fn build_topology() -> Topology {
    // 1. Collect all corner coordinates and assign stable node ids.
    let mut coord_set: Vec<(i32, i32)> = Vec::new();
    for &(q, r) in HEX_AXIAL.iter() {
        let (cx, cy) = (2 * q + r, 3 * r);
        for (dx, dy) in CORNER_OFFSETS {
            coord_set.push((cx + dx, cy + dy));
        }
    }
    coord_set.sort_unstable_by_key(|&(x, y)| (y, x));
    coord_set.dedup();
    assert_eq!(coord_set.len(), N_NODES, "expected 54 intersections");
    let node_of: HashMap<(i32, i32), u8> = coord_set
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as u8))
        .collect();

    // 2. Hex -> corner node ids.
    let mut hex_nodes = [[0u8; 6]; N_HEXES];
    for (h, &(q, r)) in HEX_AXIAL.iter().enumerate() {
        let (cx, cy) = (2 * q + r, 3 * r);
        for (k, (dx, dy)) in CORNER_OFFSETS.iter().enumerate() {
            hex_nodes[h][k] = node_of[&(cx + dx, cy + dy)];
        }
    }

    // 3. Node -> hexes.
    let mut node_hexes = vec![Vec::new(); N_NODES];
    for (h, corners) in hex_nodes.iter().enumerate() {
        for &n in corners {
            node_hexes[n as usize].push(h as u8);
        }
    }
    for v in node_hexes.iter_mut() {
        v.sort_unstable();
    }

    // 4. Edges: consecutive corners around each hex, deduplicated.
    //    Also count how many hexes each edge borders (1 = coastal).
    let mut edge_set: Vec<(u8, u8)> = Vec::new();
    for corners in hex_nodes.iter() {
        for k in 0..6 {
            let (a, b) = (corners[k], corners[(k + 1) % 6]);
            edge_set.push((a.min(b), a.max(b)));
        }
    }
    edge_set.sort_unstable();
    let mut edge_hex_count: HashMap<(u8, u8), u8> = HashMap::new();
    for &e in &edge_set {
        *edge_hex_count.entry(e).or_insert(0) += 1;
    }
    edge_set.dedup();
    assert_eq!(edge_set.len(), N_EDGES, "expected 72 paths");

    let mut edge_nodes = [[0u8; 2]; N_EDGES];
    let mut edge_of: HashMap<(u8, u8), u8> = HashMap::new();
    for (i, &(a, b)) in edge_set.iter().enumerate() {
        edge_nodes[i] = [a, b];
        edge_of.insert((a, b), i as u8);
    }

    // 5. Node adjacency and incidence.
    let mut node_neighbors = vec![Vec::new(); N_NODES];
    let mut node_edges = vec![Vec::new(); N_NODES];
    for (i, &[a, b]) in edge_nodes.iter().enumerate() {
        node_neighbors[a as usize].push(b);
        node_neighbors[b as usize].push(a);
        node_edges[a as usize].push(i as u8);
        node_edges[b as usize].push(i as u8);
    }
    for v in node_neighbors.iter_mut() {
        v.sort_unstable();
    }
    for v in node_edges.iter_mut() {
        v.sort_unstable();
    }

    // 6. Walk the coastline. Coastal edges border exactly one land hex; each
    //    coastal node has exactly two coastal edges, so the walk is a cycle.
    let coastal: Vec<u8> = (0..N_EDGES as u8)
        .filter(|&e| {
            let [a, b] = edge_nodes[e as usize];
            edge_hex_count[&(a, b)] == 1
        })
        .collect();
    assert_eq!(coastal.len(), 30, "expected 30 coastal paths");
    let is_coastal = |e: u8| coastal.contains(&e);

    let start = coastal[0];
    let mut cycle = vec![start];
    let mut prev_node = edge_nodes[start as usize][0];
    let mut cur_node = edge_nodes[start as usize][1];
    let mut cur_edge = start;
    while cycle.len() < 30 {
        let next_edge = node_edges[cur_node as usize]
            .iter()
            .copied()
            .find(|&e| e != cur_edge && is_coastal(e))
            .expect("coastline must continue");
        let [a, b] = edge_nodes[next_edge as usize];
        let next_node = if a == cur_node { b } else { a };
        cycle.push(next_edge);
        prev_node = cur_node;
        cur_node = next_node;
        cur_edge = next_edge;
    }
    let _ = prev_node;

    Topology {
        hex_nodes,
        node_hexes,
        node_neighbors,
        node_edges,
        edge_nodes,
        node_xy: coord_set,
        coast_cycle: cycle,
    }
}

// ---------------------------------------------------------------- ports ----

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum PortKind {
    /// 2:1 for a specific resource.
    Specific(Resource),
    /// 3:1 for anything.
    Generic,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Port {
    pub kind: PortKind,
    /// The coastal edge the port sits on; its two nodes grant access.
    pub edge: u8,
}

/// The nine port slots, as indices into `Topology::coast_cycle`.
///
/// The physical board's sea frame alternates 2 and 1 ports per side of the
/// hexagonal frame (2+1+2+1+2+1 = 9). Each frame side spans 5 coastal edges,
/// so sides take within-side offsets {0,3} or {1}, giving gaps of 3,3,4
/// repeating around the 30-edge coastline — matching the real board, where
/// consecutive ports are two or three coastal edges apart.
pub const PORT_SLOTS: [usize; 9] = [0, 3, 6, 10, 13, 16, 20, 23, 26];

/// Canonical port kinds, in `PORT_SLOTS` order: five 2:1 ports (one per
/// resource) interleaved with four 3:1 ports.
pub fn standard_port_kinds() -> Vec<PortKind> {
    vec![
        PortKind::Generic,
        PortKind::Specific(Resource::Wool),
        PortKind::Generic,
        PortKind::Specific(Resource::Ore),
        PortKind::Specific(Resource::Grain),
        PortKind::Generic,
        PortKind::Specific(Resource::Brick),
        PortKind::Specific(Resource::Lumber),
        PortKind::Generic,
    ]
}

// ---------------------------------------------------------------- board ----

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum BoardLayout {
    /// Official "beginner"-style fixed terrain and number spiral.
    Standard,
    /// Terrain and numbers shuffled freely.
    Random,
    /// Shuffled, but rejecting boards where two 6s or 8s touch, or where two
    /// hexes with the same number touch (a common house rule / tournament
    /// setup that removes the most lopsided boards).
    RandomBalanced,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Board {
    pub terrain: [Terrain; N_HEXES],
    /// Number token per hex; 0 on the desert.
    pub numbers: [u8; N_HEXES],
    pub ports: Vec<Port>,
    /// Per node, bit i (i<5) = has 2:1 for resource i, bit 5 = has 3:1.
    pub node_port_mask: Vec<u8>,
    /// For each roll 2..=12, the hexes that produce.
    pub roll_hexes: [Vec<u8>; 13],
    pub robber: u8,
}

/// Official fixed layout: terrain and numbers laid out in the spiral printed
/// in the rulebook, written out in `HEX_AXIAL` order.
const STANDARD_TERRAIN: [Terrain; N_HEXES] = {
    use Terrain::*;
    [
        Mountains, Pasture, Forest, // row 1 (3)
        Fields, Hills, Pasture, Hills, // row 2 (4)
        Fields, Forest, Desert, Forest, Mountains, // row 3 (5)
        Forest, Mountains, Fields, Pasture, // row 4 (4)
        Hills, Fields, Pasture, // row 5 (3)
    ]
};
const STANDARD_NUMBERS: [u8; N_HEXES] = [
    10, 2, 9, //
    12, 6, 4, 10, //
    9, 11, 0, 3, 8, //
    8, 3, 4, 5, //
    5, 6, 11,
];

impl Board {
    pub fn new(layout: BoardLayout, random_ports: bool, rng: &mut impl Rng) -> Board {
        let (terrain, numbers) = match layout {
            BoardLayout::Standard => (STANDARD_TERRAIN, STANDARD_NUMBERS),
            BoardLayout::Random => Self::shuffle_land(rng),
            BoardLayout::RandomBalanced => loop {
                let cand = Self::shuffle_land(rng);
                if Self::is_balanced(&cand.1) {
                    break cand;
                }
            },
        };

        let topo = topology();
        let mut kinds = standard_port_kinds();
        let mut slots: Vec<usize> = PORT_SLOTS.to_vec();
        if random_ports {
            kinds.shuffle(rng);
            // Keep the nine canonical slots but shuffle which coastal edge
            // each occupies, so port *positions* vary too.
            let mut all: Vec<usize> = (0..30).collect();
            all.shuffle(rng);
            slots = pick_spaced_slots(&all);
        }
        let ports: Vec<Port> = slots
            .iter()
            .zip(kinds.iter())
            .map(|(&s, &k)| Port {
                kind: k,
                edge: topo.coast_cycle[s],
            })
            .collect();

        let mut node_port_mask = vec![0u8; N_NODES];
        for p in &ports {
            let bit = match p.kind {
                PortKind::Specific(r) => 1u8 << r.idx(),
                PortKind::Generic => 1u8 << 5,
            };
            for &n in topo.edge_nodes[p.edge as usize].iter() {
                node_port_mask[n as usize] |= bit;
            }
        }

        let mut roll_hexes: [Vec<u8>; 13] = Default::default();
        for h in 0..N_HEXES {
            let n = numbers[h];
            if n >= 2 && terrain[h] != Terrain::Desert {
                roll_hexes[n as usize].push(h as u8);
            }
        }

        let robber = (0..N_HEXES)
            .find(|&h| terrain[h] == Terrain::Desert)
            .unwrap() as u8;

        Board {
            terrain,
            numbers,
            ports,
            node_port_mask,
            roll_hexes,
            robber,
        }
    }

    fn shuffle_land(rng: &mut impl Rng) -> ([Terrain; N_HEXES], [u8; N_HEXES]) {
        let mut bag = Terrain::standard_bag();
        bag.shuffle(rng);
        let mut nums = standard_numbers();
        nums.shuffle(rng);
        let mut terrain = [Terrain::Desert; N_HEXES];
        let mut numbers = [0u8; N_HEXES];
        let mut ni = 0;
        for h in 0..N_HEXES {
            terrain[h] = bag[h];
            if bag[h] != Terrain::Desert {
                numbers[h] = nums[ni];
                ni += 1;
            }
        }
        (terrain, numbers)
    }

    /// No two adjacent hexes share a number, and no two 6/8 hexes touch.
    fn is_balanced(numbers: &[u8; N_HEXES]) -> bool {
        for (a, b) in hex_adjacencies() {
            let (x, y) = (numbers[a], numbers[b]);
            if x == 0 || y == 0 {
                continue;
            }
            if x == y {
                return false;
            }
            if matches!(x, 6 | 8) && matches!(y, 6 | 8) {
                return false;
            }
        }
        true
    }

    /// Resource produced by a hex, if any.
    #[inline]
    pub fn hex_resource(&self, h: u8) -> Option<Resource> {
        self.terrain[h as usize].resource()
    }

    /// Expected resource cards per roll for a settlement on `node`
    /// (multiply by 2 for a city). Ignores the robber.
    pub fn node_pips(&self, node: u8) -> u32 {
        topology().node_hexes[node as usize]
            .iter()
            .filter(|&&h| self.terrain[h as usize] != Terrain::Desert)
            .map(|&h| pips(self.numbers[h as usize]))
            .sum()
    }

    /// Pips broken down by resource for a settlement on `node`.
    pub fn node_pips_by_res(&self, node: u8) -> [u32; N_RES] {
        let mut out = [0u32; N_RES];
        for &h in topology().node_hexes[node as usize].iter() {
            if let Some(r) = self.hex_resource(h) {
                out[r.idx()] += pips(self.numbers[h as usize]);
            }
        }
        out
    }
}

/// Pairs of adjacent hex indices (sharing an edge).
pub fn hex_adjacencies() -> Vec<(usize, usize)> {
    static CACHE: OnceLock<Vec<(usize, usize)>> = OnceLock::new();
    CACHE
        .get_or_init(|| {
            let topo = topology();
            let mut out = Vec::new();
            for a in 0..N_HEXES {
                for b in (a + 1)..N_HEXES {
                    let shared = topo.hex_nodes[a]
                        .iter()
                        .filter(|n| topo.hex_nodes[b].contains(n))
                        .count();
                    if shared == 2 {
                        out.push((a, b));
                    }
                }
            }
            out
        })
        .clone()
}

/// Greedily pick 9 coastal slots at least 3 apart on the 30-cycle, preferring
/// the order given (which the caller has shuffled).
fn pick_spaced_slots(order: &[usize]) -> Vec<usize> {
    let mut chosen: Vec<usize> = Vec::new();
    for &c in order {
        let ok = chosen.iter().all(|&s| {
            let d = (c as i32 - s as i32).abs();
            let d = d.min(30 - d);
            d >= 3
        });
        if ok {
            chosen.push(c);
        }
        if chosen.len() == 9 {
            break;
        }
    }
    // Fall back to the canonical slots if the greedy pass came up short.
    if chosen.len() < 9 {
        return PORT_SLOTS.to_vec();
    }
    chosen.sort_unstable();
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_shape() {
        let t = topology();
        assert_eq!(t.node_xy.len(), N_NODES);
        assert_eq!(t.edge_nodes.len(), N_EDGES);
        // Every node touches 1..=3 hexes; incidences total 19*6.
        let total: usize = t.node_hexes.iter().map(|v| v.len()).sum();
        assert_eq!(total, N_HEXES * 6);
        assert!(t.node_hexes.iter().all(|v| (1..=3).contains(&v.len())));
        // Interior nodes (3 hexes) number 24; tips (1 hex) 18; coast (2) 12.
        let c3 = t.node_hexes.iter().filter(|v| v.len() == 3).count();
        let c2 = t.node_hexes.iter().filter(|v| v.len() == 2).count();
        let c1 = t.node_hexes.iter().filter(|v| v.len() == 1).count();
        assert_eq!((c1, c2, c3), (18, 12, 24));
        // Degrees: each node has 2 or 3 neighbours, summing to 2*72.
        let deg: usize = t.node_neighbors.iter().map(|v| v.len()).sum();
        assert_eq!(deg, 2 * N_EDGES);
        for n in 0..N_NODES {
            assert_eq!(t.node_neighbors[n].len(), t.node_edges[n].len());
        }
        assert_eq!(t.coast_cycle.len(), 30);
        // Coast cycle must be a permutation of distinct edges.
        let mut c = t.coast_cycle.clone();
        c.sort_unstable();
        c.dedup();
        assert_eq!(c.len(), 30);
    }

    #[test]
    fn hex_adjacency_count() {
        // A hexagon of 19 hexes has 42 internal borders.
        assert_eq!(hex_adjacencies().len(), 42);
    }

    #[test]
    fn standard_board_is_legal() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let b = Board::new(BoardLayout::Standard, false, &mut rng);
        let mut t = Terrain::standard_bag();
        let mut have: Vec<Terrain> = b.terrain.to_vec();
        t.sort_unstable_by_key(|x| format!("{x:?}"));
        have.sort_unstable_by_key(|x| format!("{x:?}"));
        assert_eq!(t, have);
        let mut nums: Vec<u8> = b.numbers.iter().copied().filter(|&n| n > 0).collect();
        nums.sort_unstable();
        assert_eq!(nums, standard_numbers());
        assert_eq!(b.ports.len(), 9);
        // Total pips on the board is fixed: sum over 18 tokens.
        // 18 tokens: 2 and 12 once (1 pip each), 3..6 and 8..11 twice.
        let total: u32 = b.numbers.iter().map(|&n| pips(n)).sum();
        assert_eq!(total, 58);
    }

    #[test]
    fn random_boards_are_legal() {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
        for _ in 0..200 {
            let b = Board::new(BoardLayout::RandomBalanced, true, &mut rng);
            assert_eq!(b.ports.len(), 9);
            assert_eq!(b.terrain.iter().filter(|&&t| t == Terrain::Desert).count(), 1);
            assert_eq!(b.numbers.iter().filter(|&&n| n == 0).count(), 1);
            assert!(Board::is_balanced(&b.numbers));
            // Ports never share a node under >=3 spacing.
            let topo = topology();
            let mut seen = vec![0u8; N_NODES];
            for p in &b.ports {
                for &n in topo.edge_nodes[p.edge as usize].iter() {
                    seen[n as usize] += 1;
                }
            }
            assert!(seen.iter().all(|&c| c <= 1));
        }
    }
}
