# catan_sim

A Settlers of Catan simulator built for running thousands of games and asking
whether a strategy actually works — or just got lucky.

```bash
cargo build --release
./target/release/catan_sim run -n 5000
```

Roughly **1,000 full 4-player games per second** on 4 cores, so 100,000 games
is about a minute.

## What is implemented

The full base game:

- 19 hexes, 54 intersections, 72 paths, 9 ports, derived from hex geometry
  rather than hand-tabulated (see `src/board.rs`)
- Roads, settlements, cities, with the distance rule and road-blocking
- All 25 development cards: 14 knights, 5 victory points, 2 road building,
  2 year of plenty, 2 monopoly
- Robber, the 7-card hand limit and discards, stealing
- Bank trades, 3:1 and 2:1 ports, and player-to-player trading
- Longest road and largest army, including the real tie rules (the holder
  keeps the card while tied; a tie between challengers sets it aside)
- The official bank shortage rule (if the supply cannot pay everyone owed a
  resource, nobody receives it unless exactly one player is owed)
- Piece limits: 15 roads, 5 settlements, 4 cities

Board layouts: the fixed rulebook layout, freely shuffled, or shuffled with
adjacent 6/8s and adjacent equal numbers rejected. Ports can be fixed or
randomised. Dice can be two independent d6 or a shuffled 36-outcome deck.

## Measuring skill separately from luck

This is the part the simulator is really built around. Performance is broken
into links that can be measured one at a time:

```
placement quality --> expected inflow --> dice deviation --> actual inflow
    --> conversion efficiency --> victory points
   (skill)                          (luck)                    (skill)
```

On **every** roll of the game — not just a player's own turns — the engine adds
that player's expected yield given the buildings they hold at that moment and
where the robber is. Summed over a game, `expected_cards` answers: *what should
this board position have produced, with average dice, over exactly this many
rolls?* Comparing it to what they actually collected isolates the dice:

```
dice_luck = produced - expected_cards
```

Across a large field this averages to zero (there is a test for it), so it is
a clean measure of how the dice treated a given position.

The **skill index** then regresses victory points on dice luck over every
player-game in the run and subtracts the fitted effect:

```
VP = a + b * dice_luck + e
skill_index = VP - b * (dice_luck - mean dice_luck)
```

It is in victory points, centred on the field average, so `7.8` reads as
"performed like a 7.8-point player once the dice are accounted for".
`skill_index_z` is the same thing in standard deviations.

### What is deliberately not adjusted for

An earlier version also adjusted for a combined "robber luck" term that
included cards discarded on a 7. That was wrong, and the data said so: the
fitted coefficient came out **negative**, because players who discard more tend
to score more — holding a big hand goes with having a strong engine. Discarding
is mostly a consequence of choosing to hoard, so it is a style choice, and
adjusting for it would subtract a skill rather than a luck.

So the headline index adjusts for the dice alone. Theft, robber blocking and
monopolies are reported separately as `theft_luck`, and a secondary
`skill_index_theft_adj` also removes them — flagged, because opponents aim the
robber at whoever is winning, which makes that term partly earned.

Note that over thousands of games with seat rotation on, no strategy ends up
systematically luckier, so the skill index lands very close to raw mean VP. That
is the point: it confirms a win-rate gap is not a luck artifact. The adjustment
does real work on single games and small samples, which is what the per-game
output and the JSONL export are for.

## Strategies

| name | what it does |
|---|---|
| `balanced` | all-round: cities, settlements, some development cards |
| `city_builder` | ore and grain, straight to cities, few roads |
| `dev_rush` | buys development cards relentlessly; largest army and VP cards |
| `expander` | wood and brick, sprawls outward, chases longest road |
| `trader` | leans on ports and trades with everyone |
| `hermit` | identical to `balanced` but never trades with another player |
| `random` | uniform over legal actions; the null hypothesis |

Every non-random bot shares one scoring engine and differs only by a weight
vector (`src/bots.rs`). That is deliberate: when two archetypes are compared,
the only thing that differs is their preferences, not their implementation
quality. To add a strategy, add a preset to `Weights::preset` — no engine code.

Bots see a `View`, not the game state. Opponents' hand contents and unplayed
development cards are hidden, so a strategy cannot quietly win by cheating.

## Running experiments

```bash
# archetype league, 20k games, seats rotated so position cannot confound
catan_sim run --players balanced,city_builder,dev_rush,expander -n 20000

# does trading matter? same lineup, trading switched off
catan_sim run --players trader,trader,balanced,balanced -n 20000
catan_sim run --players trader,trader,balanced,balanced -n 20000 --no-trades

# does refusing to trade hurt you, at a table that trades?
catan_sim run --players hermit,trader,trader,trader -n 20000

# board effects
catan_sim run -n 20000 --layout standard      # fixed rulebook board
catan_sim run -n 20000 --layout random        # unconstrained shuffle
catan_sim run -n 20000 --random-ports
catan_sim run -n 20000 --dice deck36          # same odds, far less variance

# 3-player, race to 8
catan_sim run --players balanced,dev_rush,expander -n 20000 --vp 8

# one game in detail, and a board to look at
catan_sim one --seed 42
catan_sim board --seed 7
```

Useful flags: `--seed`, `--no-rotate` (to *measure* seat advantage rather than
cancel it), `--hand-limit`, `--json summary.json`, and `--jsonl rows.jsonl`
which writes one row per player-game with the full metric set for analysis in
pandas or R.

Every run reports win rates with Wilson confidence intervals, so differences
can be read as significant or not rather than eyeballed.

## Trusting the results

`cargo test --release` runs two suites.

`tests/rules.rs` steps through games checking **every invariant after every
single action** — about 69,000 actions across 2-to-6-player games, all three
layouts and both dice modes. It verifies resource conservation (cards only
move, never appear), development-card accounting, piece supplies agreeing with
the board, the distance rule, that special cards are only held when earned,
that no road is orphaned, that games terminate, that one and only one player
reaches the target, and that the winner's victory points add up from their
parts. It also pins the experimental controls: `hermit` never trades, and
`--no-trades` stops all player trading while leaving bank trades working.

`tests/statistics.rs` tests the measuring apparatus, which can be biased even
when the rules are right: dice luck averages to zero, production matches
expectation to within 0.2% in aggregate, the dice pass a chi-squared test
against 2d6, the dice deck has the same mean with lower variance, seat rotation
is exactly balanced, win rates sum to one, the skill index is centred on the
field, and the heuristic bots beat random play.

## Known limitations

- Trade offers are enumerated as 1-for-1 and 2-for-1 bundles. That covers the
  large majority of real table trades but not complex multi-card deals, and
  there is no negotiation back-and-forth: an offer is made and accepted or
  refused. Conclusions about *whether* trading matters are sound; conclusions
  about sophisticated negotiation are not.
- Bots are heuristic, not optimal. They are good enough to produce realistic
  games (87 turns on average, winners at 10 VP) and to rank strategies against
  each other, but a finding about these bots is a finding about these bots. The
  shared scoring engine is what makes the comparison fair.
- The robber heuristic targets the leader, which is realistic, but means
  `theft_luck` is partly a consequence of playing well.
- **`--layout standard` is a single board, not a sample of boards.** Every
  game uses the identical fixed layout, so the only variation is the dice and
  bot tie-breaking. Confidence intervals from such a run describe *that one
  board* and are far too tight to generalise — and the strategy ranking really
  does change from board to board (on the rulebook layout `dev_rush` wins 33%
  and `expander` 11%; on random balanced boards they are 22% and 19%). Use
  `--layout random` or `balanced` for any claim about strategies in general.
- Port positions use the real board's 2-1-2-1-2-1 frame pattern; the exact
  resource assigned to each slot is a documented choice, not looked up from the
  physical board. Use `--random-ports` if that matters to a conclusion.
