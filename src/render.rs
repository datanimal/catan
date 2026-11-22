//! Human-readable board rendering, for verifying layouts by eye.

use crate::board::{topology, Board, PortKind, HEX_AXIAL};
use crate::types::*;
use std::fmt::Write as _;

fn terrain_code(t: Terrain) -> &'static str {
    match t {
        Terrain::Hills => "brick",
        Terrain::Forest => "lumbr",
        Terrain::Pasture => "wool ",
        Terrain::Fields => "grain",
        Terrain::Mountains => "ore  ",
        Terrain::Desert => "DESRT",
    }
}

fn terrain_letter(t: Terrain) -> char {
    match t {
        Terrain::Hills => 'B',
        Terrain::Forest => 'L',
        Terrain::Pasture => 'W',
        Terrain::Fields => 'G',
        Terrain::Mountains => 'O',
        Terrain::Desert => '-',
    }
}

/// The 19 hexes laid out in rows of 3/4/5/4/3, each cell `<letter><number>`.
/// A `*` marks the robber.
pub fn board_ascii(b: &Board) -> String {
    let rows: [(usize, usize); 5] = [(0, 3), (3, 7), (7, 12), (12, 16), (16, 19)];
    let mut out = String::new();
    out.push_str("board (letter = resource, number = token, * = robber)\n");
    for &(lo, hi) in rows.iter() {
        let indent = (5 - (hi - lo)) * 3;
        out.push_str(&" ".repeat(indent));
        for h in lo..hi {
            let n = b.numbers[h];
            let num = if n == 0 {
                "  ".to_string()
            } else {
                format!("{n:>2}")
            };
            let robber = if b.robber as usize == h { '*' } else { ' ' };
            let _ = write!(out, " {}{}{}  ", terrain_letter(b.terrain[h]), num, robber);
        }
        out.push('\n');
    }
    out
}

/// Hex-by-hex listing, port listing, and the pip totals per resource.
pub fn board_table(b: &Board) -> String {
    let topo = topology();
    let mut out = String::new();

    out.push_str("\nhexes (axial q,r):\n");
    for h in 0..N_HEXES {
        let (q, r) = HEX_AXIAL[h];
        let n = b.numbers[h];
        let _ = writeln!(
            out,
            "  {:>2}  ({:>2},{:>2})  {}  {:>2}  {} pips{}",
            h,
            q,
            r,
            terrain_code(b.terrain[h]),
            n,
            pips(n),
            if b.robber as usize == h { "   <- robber" } else { "" }
        );
    }

    out.push_str("\nports (each reachable from two intersections):\n");
    for p in &b.ports {
        let [n1, n2] = topo.edge_nodes[p.edge as usize];
        // Name the port by the land it sits against.
        let mut land: Vec<String> = Vec::new();
        for &n in [n1, n2].iter() {
            for &h in topo.node_hexes[n as usize].iter() {
                let d = format!(
                    "{}{}",
                    terrain_code(b.terrain[h as usize]).trim(),
                    if b.numbers[h as usize] > 0 {
                        format!("-{}", b.numbers[h as usize])
                    } else {
                        String::new()
                    }
                );
                if !land.contains(&d) {
                    land.push(d);
                }
            }
        }
        let kind = match p.kind {
            PortKind::Generic => "3:1 any  ".to_string(),
            PortKind::Specific(r) => format!("2:1 {:<6}", r.name()),
        };
        let _ = writeln!(
            out,
            "  {}  at nodes {:>2},{:>2}  next to {}",
            kind,
            n1,
            n2,
            land.join(", ")
        );
    }

    out.push_str("\npips per resource (36ths of a roll, summed over the board):\n");
    let mut by_res = [0u32; N_RES];
    for h in 0..N_HEXES {
        if let Some(r) = b.hex_resource(h as u8) {
            by_res[r.idx()] += pips(b.numbers[h]);
        }
    }
    for r in Resource::ALL {
        let _ = writeln!(out, "  {:<7} {:>2}", r.name(), by_res[r.idx()]);
    }
    let _ = writeln!(out, "  {:<7} {:>2}", "total", by_res.iter().sum::<u32>());

    // The best intersections on this board, which is what players fight over.
    let mut best: Vec<(u32, u8)> = (0..N_NODES as u8).map(|n| (b.node_pips(n), n)).collect();
    best.sort_unstable_by(|a, c| c.0.cmp(&a.0));
    out.push_str("\nrichest intersections:\n");
    for &(p, n) in best.iter().take(6) {
        let by = b.node_pips_by_res(n);
        let mix: Vec<String> = Resource::ALL
            .iter()
            .filter(|r| by[r.idx()] > 0)
            .map(|r| format!("{}:{}", r.short(), by[r.idx()]))
            .collect();
        let port = if b.node_port_mask[n as usize] != 0 {
            "  +port"
        } else {
            ""
        };
        let _ = writeln!(out, "  node {:>2}  {:>2} pips  {}{}", n, p, mix.join(" "), port);
    }
    out
}
