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
                lineup: players.clone(),
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
        cfg.lineup.join(", "),
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
