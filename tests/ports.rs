#[test]
fn ports_are_spread_around_the_coast() {
    use catan_sim::board::*;
    use rand::prelude::*;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
    let b = Board::new(BoardLayout::Standard, false, &mut rng);
    let topo = topology();
    let mut angles: Vec<f64> = b.ports.iter().map(|p| {
        let [a, c] = topo.edge_nodes[p.edge as usize];
        let (x1, y1) = topo.node_xy[a as usize];
        let (x2, y2) = topo.node_xy[c as usize];
        // Lattice x is scaled 2/sqrt(3) relative to y; undo for true angle.
        let x = (x1 + x2) as f64 / 2.0 * (3.0f64).sqrt() / 2.0;
        let y = (y1 + y2) as f64 / 2.0 / 2.0;
        y.atan2(x).to_degrees().rem_euclid(360.0)
    }).collect();
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("port bearings (deg): {:?}", angles.iter().map(|a| a.round() as i32).collect::<Vec<_>>());
    let mut gaps: Vec<f64> = (0..9).map(|i| {
        let d = angles[(i + 1) % 9] - angles[i];
        if d < 0.0 { d + 360.0 } else { d }
    }).collect();
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("angular gaps: {:?}", gaps.iter().map(|a| a.round() as i32).collect::<Vec<_>>());
    // Nine ports around a ring average 40 deg apart; none should be adjacent
    // or diametrically isolated.
    assert!(gaps[0] > 15.0, "two ports nearly on top of each other: {:?}", gaps);
    assert!(gaps[8] < 75.0, "a long stretch of coast with no port: {:?}", gaps);
}
