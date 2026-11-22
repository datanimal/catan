//! Tests for the strategy-spec plumbing and the placement experiment, both of
//! which silently change results if they misbehave.

use catan_sim::bots::{Strategy, Weights, PARAM_KEYS};
use catan_sim::game::GameConfig;
use catan_sim::placement::{self, PlacementConfig};

#[test]
fn weight_specs_round_trip() {
    let mut w = Weights::preset("city_builder").unwrap();
    w.set("w_dev", 37.5);
    w.set("pref_ore", 2.25);
    let spec = format!("city_builder+{}", w.to_spec());
    let back = Weights::from_spec(&spec).expect("parse");
    for k in PARAM_KEYS {
        assert!(
            (w.get(k) - back.get(k)).abs() < 1e-3,
            "{k} did not survive: {} vs {}",
            w.get(k),
            back.get(k)
        );
    }
}

#[test]
fn specs_respect_bounds_and_reject_nonsense() {
    // Out-of-range values are clamped, not accepted blindly.
    let w = Weights::from_spec("balanced+w_city=99999").unwrap();
    assert!(w.w_city <= 120.0, "w_city was not clamped: {}", w.w_city);
    // Unknown keys and malformed input must fail rather than be ignored,
    // otherwise a typo in an experiment would silently run the base strategy.
    assert!(Weights::from_spec("balanced+not_a_key=1").is_none());
    assert!(Weights::from_spec("balanced+w_city=abc").is_none());
    assert!(Weights::from_spec("no_such_preset").is_none());
    // Both separators work.
    let a = Weights::from_spec("balanced+w_dev=40;port=9").unwrap();
    let b = Weights::from_spec("balanced+w_dev=40,port=9").unwrap();
    assert_eq!(a.w_dev, b.w_dev);
    assert_eq!(a.port, b.port);
}

#[test]
fn labels_keep_variants_apart() {
    // Two tuned variants of one preset must not be pooled under the base
    // name, or the comparison between them vanishes.
    let a = Strategy::parse("A:balanced+w_dev=5").unwrap();
    let b = Strategy::parse("B:balanced+w_dev=50").unwrap();
    assert_eq!(a.label(), "A");
    assert_eq!(b.label(), "B");
    assert_ne!(a.weights().unwrap().w_dev, b.weights().unwrap().w_dev);
    // A bare preset name is not mistaken for a label.
    assert_eq!(Strategy::parse("balanced").unwrap().label(), "balanced");
    assert!(Strategy::parse("random").unwrap().is_random());
}

#[test]
fn random_openings_are_actually_random() {
    // The placement experiment depends on this: if the bot still placed by
    // preference, opening position would stay entangled with strategy and the
    // valuations would be confounded.
    let mut cfg = GameConfig::default();
    cfg.n_players = 4;
    let pc = PlacementConfig {
        games: 300,
        strategy: Strategy::parse("balanced").unwrap(),
        n_players: 4,
        seed: 12,
        game: cfg,
    };
    let recs = placement::run_records(&pc);
    let pips: Vec<f64> = recs
        .iter()
        .flat_map(|g| g.players.iter())
        .map(|p| p.metrics.opening.total_pips as f64)
        .collect();
    let n = pips.len() as f64;
    let mean = pips.iter().sum::<f64>() / n;
    let sd = (pips.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    println!("random openings: mean {mean:.1} pips, sd {sd:.2}");
    // Chosen openings average about 19-20 pips with little spread. Random
    // ones must be clearly weaker and clearly more varied.
    assert!(
        mean > 10.0 && mean < 16.0,
        "random openings averaged {mean:.1} pips, which does not look random"
    );
    assert!(sd > 3.0, "openings barely varied (sd {sd:.2})");
    // And they must reach both extremes.
    assert!(pips.iter().any(|&p| p < 8.0), "no weak openings appeared");
    assert!(pips.iter().any(|&p| p > 20.0), "no strong openings appeared");
}

#[test]
fn placement_valuation_is_sane() {
    let mut cfg = GameConfig::default();
    cfg.n_players = 4;
    let pc = PlacementConfig {
        games: 4000,
        strategy: Strategy::parse("balanced").unwrap(),
        n_players: 4,
        seed: 99,
        game: cfg,
    };
    let recs = placement::run_records(&pc);
    let rep = placement::analyse(&recs, placement::REALISTIC_MIN_PIPS);
    println!(
        "lambda = {:.4} VP/pip, n = {}, R^2 = {:.3}",
        rep.lambda, rep.n, rep.full.r2
    );
    // A pip is 1/36 of a card per roll over roughly 85 turns, so it is worth
    // something, but nowhere near a victory point.
    assert!(
        rep.lambda > 0.05 && rep.lambda < 1.0,
        "implausible pip value: {}",
        rep.lambda
    );
    // Every resource must carry positive value.
    for r in catan_sim::types::Resource::ALL {
        let c = rep.full.coef_of(&format!("pips_{}", r.name()));
        assert!(c > 0.0, "{} priced at {c}", r.name());
    }
    // More pips must beat fewer pips, monotonically, in the raw buckets.
    let b = &rep.buckets;
    for w in b.windows(2) {
        assert!(
            w[1].mean_adj_vp > w[0].mean_adj_vp,
            "bucket {} scored below {}",
            w[1].lo,
            w[0].lo
        );
    }
}
