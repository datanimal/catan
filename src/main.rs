//! Command line front end.

use catan_sim::analysis::{self, Summary};
use catan_sim::board::{Board, BoardLayout};
use catan_sim::bots;
use catan_sim::game::{DiceMode, GameConfig};
use catan_sim::metrics::GameRecord;
use catan_sim::render;
use catan_sim::sim::{self, SimConfig};
use clap::{Parser, Subcommand, ValueEnum};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::io::Write;

#[derive(Parser)]
#[command(
    name = "catan_sim",
    about = "Settlers of Catan simulator for large-scale strategy experiments"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum LayoutArg {
    /// The fixed layout printed in the rulebook.
    Standard,
    /// Terrain and numbers shuffled freely.
    Random,
    /// Shuffled, rejecting adjacent 6/8s and adjacent equal numbers.
    Balanced,
}

#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum DiceArg {
    /// Two independent dice.
    #[value(name = "2d6")]
    TwoD6,
    /// A shuffled 36-card deck of outcomes: same distribution, less variance.
    Deck36,
}

#[derive(Subcommand)]
enum Cmd {
    /// List the available strategies.
    List,
    /// Print a board, to eyeball the layout and the port positions.
    Board {
        #[arg(long, default_value_t = 1)]
        seed: u64,
        #[arg(long, value_enum, default_value = "balanced")]
        layout: LayoutArg,
        #[arg(long)]
        random_ports: bool,
    },
    /// Play a single game and print a turn-by-turn-free summary of it.
    One {
        #[arg(long, value_delimiter = ',', default_value = "balanced,city_builder,dev_rush,expander")]
        players: Vec<String>,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Search for a stronger strategy by self-play.
    Tune {
        /// Starting point: a preset name, or `preset+key=value,...`.
        #[arg(long, default_value = "balanced")]
        base: String,
        #[arg(long, default_value_t = 18)]
        rounds: usize,
        /// Mutations tried per round.
        #[arg(long, default_value_t = 5)]
        candidates: usize,
        /// Games per candidate evaluation.
        #[arg(long, default_value_t = 1800)]
        eval_games: usize,
        #[arg(long, default_value_t = 4)]
        players: usize,
        /// Standard errors above chance a candidate must clear.
        #[arg(long, default_value_t = 1.0)]
        sigma: f64,
        #[arg(long, default_value_t = 0xC0FFEE)]
        seed: u64,
        /// Games for the final check against the preset field.
        #[arg(long, default_value_t = 40000)]
        validate: usize,
        /// Opponents to tune against, comma separated. Default is copies of
        /// the champion (self-play); naming a varied field instead optimises
        /// for general strength.
        #[arg(long, value_delimiter = ',')]
        against: Option<Vec<String>>,
    },
    #[command(name = "winprob")]
    /// Fit and score a live win-probability model, as a spectator overlay
    /// would need. Reports how well it ranks positions and whether its
    /// probabilities are honest.
    WinProb {
        #[arg(long, short = 'n', default_value_t = 40000)]
        games: usize,
        #[arg(long, value_delimiter = ',', default_value = "tuned,city_builder,dev_rush,expander")]
        players: Vec<String>,
        #[arg(long, default_value_t = 0xB0A7)]
        seed: u64,
    },
    /// Value opening positions in pips, using randomised openings.
    Placement {
        #[arg(long, short = 'n', default_value_t = 40000)]
        games: usize,
        /// Strategy every player uses after the opening.
        #[arg(long, default_value = "balanced")]
        strategy: String,
        #[arg(long, default_value_t = 4)]
        players: usize,
        #[arg(long, default_value_t = 0x9A1D)]
        seed: u64,
        #[arg(long, value_enum, default_value = "balanced")]
        layout: LayoutArg,
        /// Randomise port positions too, so port value is not tied to the
        /// nine canonical slots.
        #[arg(long)]
        random_ports: bool,
        /// Write one row per opening position here.
        #[arg(long)]
        jsonl: Option<String>,
    },
    /// Run a batch of games and report.
    Run {
        /// Strategies, one per seat, comma separated.
        #[arg(long, value_delimiter = ',', default_value = "balanced,city_builder,dev_rush,expander")]
        players: Vec<String>,
        #[arg(long, short = 'n', default_value_t = 1000)]
        games: usize,
        #[arg(long, default_value_t = 10)]
        vp: u8,
        #[arg(long, value_enum, default_value = "balanced")]
        layout: LayoutArg,
        #[arg(long)]
        random_ports: bool,
        #[arg(long, value_enum, default_value = "2d6")]
        dice: DiceArg,
        /// Switch off player-to-player trading.
        #[arg(long)]
        no_trades: bool,
        /// Keep each strategy in a fixed seat (seat advantage not cancelled).
        #[arg(long)]
        no_rotate: bool,
        #[arg(long, default_value_t = 0x5EED)]
        seed: u64,
        #[arg(long, default_value_t = 7)]
        hand_limit: i16,
        /// Write one JSON object per player-game here, for your own analysis.
        #[arg(long)]
        jsonl: Option<String>,
        /// Write the whole summary as JSON here.
        #[arg(long)]
        json: Option<String>,
    },
}

fn layout_of(l: LayoutArg) -> BoardLayout {
    match l {
        LayoutArg::Standard => BoardLayout::Standard,
        LayoutArg::Random => BoardLayout::Random,
        LayoutArg::Balanced => BoardLayout::RandomBalanced,
    }
}

fn main() {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::List => {
            println!("strategies:");
            for n in bots::available() {
                println!("  {n}");
            }
        }

        Cmd::Board {
            seed,
            layout,
            random_ports,
        } => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let b = Board::new(layout_of(layout), random_ports, &mut rng);
            println!("{}", render::board_ascii(&b));
            println!("{}", render::board_table(&b));
        }

        Cmd::One { players, seed } => {
            if let Err(e) = sim::check_lineup(&players) {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
            let mut cfg = SimConfig::new(players, 1);
            cfg.base_seed = seed;
            cfg.rotate_seats = false;
            let rec = sim::run_one(&cfg, 0);
            print_single(&rec);
        }

        Cmd::WinProb {
            games,
            players,
            seed,
        } => {
            if let Err(e) = sim::check_lineup(&players) {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
            let mut cfg = SimConfig::new(players.clone(), games);
            cfg.base_seed = seed;
            let recs = sim::run(&cfg);
            println!("=== live win probability ===");
            println!(
                "{games} games, lineup {}",
                players.join(", ")
            );
            println!(
                "models fitted on the first 70% of games and scored on the last 30%\n"
            );
            use catan_sim::winprob::{self, Info};
            for (marker, label) in [(0usize, "after setup"), (1, "end of early"), (2, "end of mid")]
            {
                println!("--- {label} ---");
                println!(
                    "{:<12} {:>9} {:>8} {:>10} {:>12} {:>14}",
                    "information", "test n", "AUC", "Brier", "vs base rate", "worst bin gap"
                );
                for info in [Info::Spectator, Info::Full] {
                    let Some(r) = winprob::evaluate(&recs, marker, info) else {
                        continue;
                    };
                    println!(
                        "{:<12} {:>9} {:>8.3} {:>10.4} {:>11.1}% {:>13.1}%",
                        if info == Info::Spectator { "spectator" } else { "full state" },
                        r.test_n,
                        r.test_auc,
                        r.test_brier,
                        100.0 * r.skill(),
                        100.0 * r.max_miscalibration
                    );
                }
                // Reliability table for the spectator model, which is the one
                // an overlay could actually run.
                if let Some(r) = winprob::evaluate(&recs, marker, Info::Spectator) {
                    print!("  calibration (spectator):");
                    for b in r.bins.iter().filter(|b| b.n >= 100) {
                        print!(
                            "  {:.0}%->{:.0}%",
                            100.0 * b.mean_predicted,
                            100.0 * b.actual
                        );
                    }
                    println!();
                }
                println!();
            }
            // What the overlay would key off.
            if let Some(r) = winprob::evaluate(&recs, 2, Info::Spectator) {
                println!("--- what drives the estimate at the end of the mid game ---");
                let mut ord: Vec<usize> = (0..r.fit.coef.len()).collect();
                ord.sort_by(|&a, &b| {
                    (r.fit.coef[b].abs() / r.fit.se[b].max(1e-9))
                        .partial_cmp(&(r.fit.coef[a].abs() / r.fit.se[a].max(1e-9)))
                        .unwrap()
                });
                for &i in ord.iter().take(8) {
                    println!(
                        "  {:<16} {:>+8.3}  (t = {:+.1})",
                        r.fit.names[i],
                        r.fit.coef[i],
                        r.fit.coef[i] / r.fit.se[i].max(1e-9)
                    );
                }
            }
        }

        Cmd::Placement {
            games,
            strategy,
            players,
            seed,
            layout,
            random_ports,
            jsonl,
        } => {
            let Some(st) = catan_sim::bots::Strategy::parse(&strategy) else {
                eprintln!("error: cannot parse strategy '{strategy}'");
                std::process::exit(2);
            };
            let mut game = GameConfig::default();
            game.n_players = players;
            game.layout = layout_of(layout);
            game.random_ports = random_ports;
            let pc = catan_sim::placement::PlacementConfig {
                games,
                strategy: st,
                n_players: players,
                seed,
                game,
            };
            let t0 = std::time::Instant::now();
            let recs = catan_sim::placement::run_records(&pc);
            let mut rep = catan_sim::placement::analyse(
                &recs,
                catan_sim::placement::REALISTIC_MIN_PIPS,
            );
            rep.wide = Some(Box::new(catan_sim::placement::analyse(&recs, 0)));
            rep.print();
            if let Some(path) = &jsonl {
                match catan_sim::placement::write_rows(path, &recs) {
                    Ok(n) => println!("\nwrote {n} opening rows to {path}"),
                    Err(e) => eprintln!("could not write {path}: {e}"),
                }
            }
            println!(
                "\n({} games in {:.1}s, post-opening strategy: {strategy})",
                games,
                t0.elapsed().as_secs_f64()
            );
        }

        Cmd::Tune {
            base,
            rounds,
            candidates,
            eval_games,
            players,
            sigma,
            seed,
            validate,
            against,
        } => {
            let Some(w0) = catan_sim::bots::Weights::from_spec(&base) else {
                eprintln!("error: cannot parse base strategy '{base}'");
                std::process::exit(2);
            };
            let mut game = GameConfig::default();
            game.n_players = players;
            let tc = catan_sim::tune::TuneConfig {
                rounds,
                candidates,
                games_per_eval: eval_games,
                n_players: players,
                seed,
                game,
                accept_sigma: sigma,
                opponents: against
                    .clone()
                    .unwrap_or_default()
                    .iter()
                    .map(|n| {
                        catan_sim::bots::Strategy::parse(n)
                            .unwrap_or_else(|| panic!("unknown strategy: {n}"))
                    })
                    .collect(),
            };
            let se = catan_sim::tune::diff_se(&tc);
            println!("=== strategy search ===");
            println!(
                "base: {base} | {rounds} rounds x {} runs x {eval_games} games = {} games",
                candidates + 1,
                rounds * (candidates + 1) * eval_games
            );
            match &against {
                Some(v) => println!("opponents: {} (fixed field)", v.join(", ")),
                None => println!("opponents: copies of the champion (self-play)"),
            }
            println!(
                "a candidate plays {players} ways; chance is {:.1}%",
                100.0 / players as f64
            );
            println!(
                "the champion is re-measured on the same seeds each round; promotion needs a \
                 margin over it"
            );
            println!(
                "of {sigma} SE, where SE of the difference is about {:.2}%\n",
                100.0 * se
            );
            println!(
                "{:<6} {:>8} {:>8} {:>8} {:>8}  {}",
                "round", "champ%", "best%", "margin", "skill", "accepted change"
            );
            let t0 = std::time::Instant::now();
            let res = catan_sim::tune::run(&tc, w0, |r, champ| {
                let ch = if r.accepted {
                    r.best
                        .changes
                        .iter()
                        .map(|(k, a, b)| format!("{k} {a:.2}->{b:.2}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                } else {
                    "-".to_string()
                };
                println!(
                    "{:<6} {:>7.2}% {:>7.2}% {:>+7.2}% {:>8.3}  {}",
                    r.round,
                    100.0 * r.champion_win_rate,
                    100.0 * r.best.win_rate,
                    100.0 * r.margin,
                    r.best.skill_index,
                    ch
                );
                let _ = champ;
            });
            println!(
                "\nsearch used {} games in {:.1}s",
                res.games_used,
                t0.elapsed().as_secs_f64()
            );
            println!("\nchampion weights:\n  {}", res.champion.to_spec());
            println!(
                "\nreproduce with:\n  catan_sim run --players \
                 'TUNED:balanced+{}',city_builder,balanced,dev_rush",
                res.champion.to_spec()
            );

            if validate > 0 {
                println!("\n=== validation: champion against the preset field ===");
                let mut lineup = vec![
                    catan_sim::bots::Strategy::Heuristic(res.champion.clone())
                        .relabel("TUNED"),
                ];
                for n in ["balanced", "city_builder", "dev_rush"] {
                    lineup.push(catan_sim::bots::Strategy::parse(n).unwrap());
                }
                let mut vc = catan_sim::sim::SimConfig::with_strategies(lineup, validate);
                vc.base_seed = seed ^ 0xABCDEF;
                let recs = catan_sim::sim::run(&vc);
                print_report(&vc, &analysis::summarize(&recs), t0.elapsed());
            }
        }

        Cmd::Run {
            players,
            games,
            vp,
            layout,
            random_ports,
            dice,
            no_trades,
            no_rotate,
            seed,
            hand_limit,
            jsonl,
            json,
        } => {
            if let Err(e) = sim::check_lineup(&players) {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
            let game = GameConfig {
                n_players: players.len(),
                vp_target: vp,
                layout: layout_of(layout),
                random_ports,
                dice: match dice {
                    DiceArg::TwoD6 => DiceMode::TwoD6,
                    DiceArg::Deck36 => DiceMode::Deck36,
                },
                turn_limit: 400,
                hand_limit,
                max_proposals: 4,
                allow_player_trades: !no_trades,
            };
            let cfg = SimConfig {
                games,
                lineup: players
                    .iter()
                    .map(|s| catan_sim::bots::Strategy::parse(s).expect("checked above"))
                    .collect(),
                rotate_seats: !no_rotate,
                base_seed: seed,
                game,
            };
            let t0 = std::time::Instant::now();
            let records = sim::run(&cfg);
            let dt = t0.elapsed();
            let summary = analysis::summarize(&records);
            print_report(&cfg, &summary, dt);

            if let Some(path) = jsonl {
                match write_jsonl(&path, &records) {
                    Ok(n) => println!("\nwrote {n} player-game rows to {path}"),
                    Err(e) => eprintln!("could not write {path}: {e}"),
                }
            }
            if let Some(path) = json {
                match serde_json::to_string_pretty(&summary)
                    .map_err(|e| e.to_string())
                    .and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string()))
                {
                    Ok(()) => println!("wrote summary to {path}"),
                    Err(e) => eprintln!("could not write {path}: {e}"),
                }
            }
        }
    }
}

fn write_jsonl(path: &str, records: &[GameRecord]) -> std::io::Result<usize> {
    let f = std::fs::File::create(path)?;
    let mut w = std::io::BufWriter::new(f);
    let mut n = 0;
    for g in records {
        for p in &g.players {
            let row = serde_json::json!({
                "seed": g.seed,
                "turns": g.turns,
                "timed_out": g.timed_out,
                "seat": p.seat,
                "policy": p.policy,
                "won": p.won,
                "vp": p.vp,
                "dice_luck": p.metrics.dice_luck(),
                "theft_luck": p.metrics.theft_luck(),
                "discarded": p.metrics.discarded(),
                "produced": p.metrics.produced_total(),
                "expected": p.metrics.expected_total(),
                "inflow": p.metrics.total_inflow(),
                "m": p.metrics,
            });
            writeln!(w, "{row}")?;
            n += 1;
        }
    }
    Ok(n)
}

fn print_single(rec: &GameRecord) {
    println!(
        "seed {} | {} turns | winner {:?}{}",
        rec.seed,
        rec.turns,
        rec.winner,
        if rec.timed_out { " (TIMED OUT)" } else { "" }
    );
    println!(
        "\n{:<14} {:>3} {:>4} {:>5} {:>6} {:>6} {:>5} {:>5} {:>4} {:>3} {:>3} {:>3} {:>4} {:>3}",
        "strategy", "vp", "prod", "exp", "luck", "theft", "disc", "flow", "pips", "S", "C", "R",
        "dev", "K"
    );
    for p in &rec.players {
        let m = &p.metrics;
        println!(
            "{:<14} {:>3} {:>4} {:>5.1} {:>+6.1} {:>+6.1} {:>5.0} {:>5} {:>4} {:>3} {:>3} {:>3} {:>4} {:>3}",
            format!("{}{}", p.policy, if p.won { "*" } else { "" }),
            p.vp,
            m.produced_total(),
            m.expected_total(),
            m.dice_luck(),
            m.theft_luck(),
            m.discarded(),
            m.total_inflow(),
            m.final_pips,
            m.settlements_built,
            m.cities_built,
            m.roads_built,
            m.dev_bought,
            m.knights_played,
        );
    }
}

fn print_report(cfg: &SimConfig, s: &Summary, dt: std::time::Duration) {
    let gps = cfg.games as f64 / dt.as_secs_f64().max(1e-9);
    println!("=== run ===");
    println!(
        "{} games, {} players, VP target {}, {} games/sec ({:.2}s)",
        s.games,
        cfg.lineup.len(),
        cfg.game.vp_target,
        gps.round() as u64,
        dt.as_secs_f64()
    );
    println!(
        "lineup: {} | seat rotation: {} | player trades: {} | dice: {:?} | layout: {:?}",
        cfg.labels().join(", "),
        if cfg.rotate_seats { "on" } else { "OFF" },
        if cfg.game.allow_player_trades { "on" } else { "OFF" },
        cfg.game.dice,
        cfg.game.layout
    );
    println!(
        "mean game length: {:.1} turns | unfinished games: {}",
        s.mean_turns, s.timeouts
    );

    println!("\n=== how much did the dice matter? ===");
    println!(
        "VP = {:.3} + {:.4} * dice_luck    (n={}, R^2={:.3})",
        s.model.intercept, s.model.beta_dice, s.model.n, s.model.r2
    );
    println!(
        "  one extra resource card from the dice is worth {:.3} VP",
        s.model.beta_dice
    );
    println!(
        "  the dice explain {:.1}% of the spread in final score; residual SD {:.2} VP",
        100.0 * s.model.r2,
        s.model.resid_sd
    );
    println!(
        "  (secondary fit: {:.4} VP per card of theft luck -- not in the headline index,",
        s.model.beta_theft
    );
    println!("   because the robber follows the leader, so it is partly earned)");

    println!("\n=== results by strategy ===");
    println!(
        "{:<14} {:>6} {:>7} {:>15} {:>6} {:>7} {:>15} {:>6}",
        "strategy", "games", "win%", "95% CI", "VP", "skill", "95% CI", "z"
    );
    let mut order: Vec<&analysis::StrategyStats> = s.strategies.iter().collect();
    order.sort_by(|a, b| b.win_rate.partial_cmp(&a.win_rate).unwrap());
    for t in &order {
        println!(
            "{:<14} {:>6} {:>6.1}% {:>7.1}-{:<7.1} {:>6.2} {:>7.2} {:>7.2}-{:<7.2} {:>+6.2}",
            t.name,
            t.games,
            100.0 * t.win_rate,
            100.0 * t.win_rate_lo,
            100.0 * t.win_rate_hi,
            t.mean_vp,
            t.skill_index,
            t.skill_index_lo,
            t.skill_index_hi,
            t.skill_index_z,
        );
    }
    println!(
        "  skill = mean VP with the dice effect removed; z = SDs above the field average"
    );

    println!("\n=== luck, placement, conversion ===");
    println!(
        "{:<14} {:>8} {:>8} {:>7} {:>8} {:>9} {:>8} {:>8} {:>8}",
        "strategy", "dice", "theft", "discard", "bankout", "cards/roll", "openpip", "inflow",
        "VP/100"
    );
    for t in &order {
        println!(
            "{:<14} {:>+8.2} {:>+8.2} {:>7.1} {:>8.2} {:>9.3} {:>8.1} {:>8.1} {:>8.2}",
            t.name,
            t.mean_dice_luck,
            t.mean_theft_luck,
            t.mean_discarded,
            t.mean_bank_shortage,
            t.mean_rate_per_roll,
            t.mean_init_pips,
            t.mean_inflow,
            t.vp_per_100_cards,
        );
    }
    println!("  dice: cards above baseline (should be ~0 across the field if unbiased)");
    println!("  theft: cards taken by opponents; discard: cards dumped on a 7 (hoarding)");
    println!("  cards/roll: expected income from their board position -- placement skill");
    println!("  VP/100: victory points per hundred cards taken in -- conversion skill");

    println!("\n=== what they actually did ===");
    println!(
        "{:<14} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} {:>7} {:>7} {:>6} {:>6}",
        "strategy", "setl", "city", "road", "dev", "knig", "bankT", "gotT", "gaveT", "LR%", "LA%"
    );
    for t in &order {
        println!(
            "{:<14} {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>5.2} {:>6.2} {:>7.2} {:>7.2} {:>5.0}% {:>5.0}%",
            t.name,
            t.mean_settlements,
            t.mean_cities,
            t.mean_roads,
            t.mean_dev_bought,
            t.mean_knights,
            t.mean_bank_trades,
            t.mean_trades_accepted,
            t.mean_trades_given,
            100.0 * t.longest_road_rate,
            100.0 * t.largest_army_rate,
        );
    }
    println!("  gotT = offers of theirs that were accepted; gaveT = offers they accepted");

    print_phases(s);

    println!("\n=== seat position ===");
    println!(
        "{:<6} {:>7} {:>7} {:>15} {:>7} {:>9}",
        "seat", "games", "win%", "95% CI", "VP", "open pips"
    );
    for t in &s.seats {
        println!(
            "{:<6} {:>7} {:>6.1}% {:>7.1}-{:<7.1} {:>7.2} {:>9.2}",
            t.seat,
            t.games,
            100.0 * t.win_rate,
            100.0 * t.win_rate_lo,
            100.0 * t.win_rate_hi,
            t.mean_vp,
            t.mean_init_pips
        );
    }
    println!("  seat 0 places first in the snake order");

    println!("\n=== opening placement ===");
    print_buckets("total pips of the two opening settlements", &s.opening_pips);
    print_buckets("resources the final position covers", &s.opening_breadth);

    // Dice sanity check.
    let total: u32 = s.roll_histogram.iter().sum();
    let worst = (2..=12u8)
        .map(|r| {
            let obs = s.roll_histogram[r as usize] as f64 / total as f64;
            ((obs / catan_sim::types::prob(r) - 1.0).abs(), r)
        })
        .fold((0.0f64, 0u8), |a, b| if b.0 > a.0 { b } else { a });
    println!(
        "\ndice: {} rolls, largest deviation from 2d6 is {:.1}% (on {})",
        total,
        100.0 * worst.0,
        worst.1
    );
}

fn print_phases(s: &Summary) {
    if s.phases.is_empty() {
        return;
    }
    println!("\n=== where each strategy gains ground ===");
    println!(
        "phases are bounded by the leader's score, not by turn number: early ends when"
    );
    println!("someone first reaches 40% of the target, mid at 70%. Rates are per own turn.");
    println!(
        "\n{:<14} {:<7} {:>7} {:>9} {:>10} {:>8} {:>8} {:>8} {:>9}",
        "strategy", "phase", "turns", "VP/turn", "cards/turn", "VP end", "pips", "lead%", "win|lead"
    );
    let mut names: Vec<&String> = s.phases.iter().map(|p| &p.name).collect();
    names.dedup();
    for n in names {
        for p in s.phases.iter().filter(|p| &p.name == n) {
            let lead = if p.lead_rate.is_nan() {
                "   -".to_string()
            } else {
                format!("{:>3.0}%", 100.0 * p.lead_rate)
            };
            let wl = if p.win_if_leading.is_nan() {
                "    -".to_string()
            } else {
                format!("{:>4.0}%", 100.0 * p.win_if_leading)
            };
            println!(
                "{:<14} {:<7} {:>7.1} {:>9.3} {:>10.2} {:>8.2} {:>8.1} {:>8} {:>9}",
                if p.phase == "early" { p.name.as_str() } else { "" },
                p.phase,
                p.turns,
                p.vp_per_turn,
                p.cards_per_turn,
                p.vp_end,
                p.pips_end,
                lead,
                wl
            );
        }
    }
    println!("  lead% = share of games level with the best score at the end of that phase");
    println!("  win|lead = win rate in those games, i.e. how well a lead converts");
}

fn print_buckets(title: &str, bs: &[analysis::Bucket]) {
    println!("  {title}:");
    for b in bs {
        if b.games == 0 {
            continue;
        }
        println!(
            "    {:<18} n={:<7} win {:>5.1}%  ({:>4.1}-{:<4.1})  VP {:.2}",
            b.label,
            b.games,
            100.0 * b.win_rate,
            100.0 * b.win_rate_lo,
            100.0 * b.win_rate_hi,
            b.mean_vp
        );
    }
}
