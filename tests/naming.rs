#[test]
fn corner_and_side_names_are_consistent() {
    use catan_sim::board::*;
    use catan_sim::types::*;
    let topo = topology();
    // Every (hex, corner) resolves, and every node is named at least once.
    let mut seen_nodes = vec![0usize; N_NODES];
    for h in 0..N_HEXES as u8 {
        for c in CORNER_NAMES {
            let n = corner_node(h, c).expect("corner resolves");
            seen_nodes[n as usize] += 1;
        }
    }
    assert!(seen_nodes.iter().all(|&c| c >= 1), "a node had no name");
    assert_eq!(seen_nodes.iter().sum::<usize>(), N_HEXES * 6);
    let mut seen_edges = vec![0usize; N_EDGES];
    for h in 0..N_HEXES as u8 {
        for s in SIDE_NAMES {
            let e = side_edge(h, s).expect("side resolves");
            seen_edges[e as usize] += 1;
        }
    }
    assert!(seen_edges.iter().all(|&c| c >= 1), "an edge had no name");
    assert_eq!(seen_edges.iter().sum::<usize>(), N_HEXES * 6);
    // Shared corners must agree: hex 9 is the centre, so its neighbours name
    // the same intersections it does.
    let centre_n = corner_node(9, "N").unwrap();
    let names = node_names(centre_n);
    println!("node {centre_n} is also known as {names:?}");
    assert!(names.len() >= 2, "a central node should have several names");
    for nm in &names {
        let (h, c) = nm.split_once(':').unwrap();
        assert_eq!(corner_node(h.parse().unwrap(), c).unwrap(), centre_n);
    }
    let _ = topo;
}
