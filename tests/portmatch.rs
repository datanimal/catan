#[test]
fn match_observed_ports() {
    use catan_sim::board::*;
    let topo = topology();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    use rand::prelude::*;
    let b = Board::new(BoardLayout::Standard, false, &mut rng);
    // Bearing of each standard port slot around the board centre.
    let mut slots: Vec<(f64, u8)> = b.ports.iter().map(|p| {
        let [a, c] = topo.edge_nodes[p.edge as usize];
        let (x1, y1) = topo.node_xy[a as usize];
        let (x2, y2) = topo.node_xy[c as usize];
        let x = (x1 + x2) as f64 / 2.0 * (3.0f64).sqrt() / 2.0;
        let y = (y1 + y2) as f64 / 2.0 / 2.0;
        // Lattice y grows downward, same as screen coordinates.
        ((-y).atan2(x).to_degrees().rem_euclid(360.0), p.edge)
    }).collect();
    slots.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap());

    // Ports read off the screenshot: (screen x, screen y, kind).
    let centre = (430.0f64, 1060.0f64);
    let mut obs: Vec<(f64, &str)> = vec![
        (220.0, 672.0, "ore"), (500.0, 672.0, "any"), (703.0, 785.0, "brick"),
        (85.0, 905.0, "any"), (838.0, 1020.0, "wool"), (85.0, 1140.0, "lumber"),
        (703.0, 1255.0, "any"), (220.0, 1388.0, "grain"), (500.0, 1388.0, "any"),
    ].into_iter().map(|(x, y, k)| {
        (((centre.1 - y).atan2(x - centre.0)).to_degrees().rem_euclid(360.0), k)
    }).collect();
    obs.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap());

    println!("{:<10}{:>10}{:>10}{:>9}   {}", "kind", "seen", "slot", "diff", "edge");
    for (o, s) in obs.iter().zip(slots.iter()) {
        let mut d = o.0 - s.0;
        if d > 180.0 { d -= 360.0 }
        if d < -180.0 { d += 360.0 }
        println!("{:<10}{:>9.0}deg{:>9.0}deg{:>+8.0}deg   [\"{}\", \"{}\"],",
                 o.1, o.0, s.0, d, o.1, edge_names(s.1)[0]);
        assert!(d.abs() < 20.0, "port {} is {d:.0} deg from any standard slot", o.1);
    }
}
