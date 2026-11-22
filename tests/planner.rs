//! Confirms the sequential planner actually runs. A feature that makes no
//! measured difference must be distinguishable from one that never fires.

use catan_sim::bots;
use catan_sim::sim::{self, SimConfig};

fn probe(spec: &str) -> (u64, u64, u64, u64) {
    bots::reset_plan_counters();
    let mut cfg = SimConfig::new(vec![spec.to_string(); 4], 300);
    cfg.base_seed = 4;
    // Single-threaded so the counters describe exactly these games.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let _ = sim::run(&cfg);
        });
    bots::plan_counters()
}

#[test]
fn the_planner_fires() {
    let (calls, found, chosen, flipped) = probe("tuned+plan_drive=1.0");
    println!(
        "300 games: {calls} decisions consulted the planner, a plan existed in {found}, \
         its first step was chosen {chosen} times, and it changed the choice {flipped} times"
    );
    assert!(calls > 1000, "planner was barely consulted: {calls}");
    assert!(
        found > 0,
        "a plan was never found in {calls} decisions -- the planner is inert"
    );
    assert!(
        chosen > 0,
        "a plan was found {found} times but never acted on"
    );
    let (affordable, better, via_rb) = bots::plan_preconditions();
    println!(
        "preconditions: a settlement was affordable in {affordable} decisions; a better \
         spot was reachable in {better}; both plus a payable route in {found}, of which \
         {via_rb} leaned on a Road Building card"
    );
}
