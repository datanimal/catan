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
| `tuned` | found by self-play search; the strongest strategy here |
| `random` | uniform over legal actions; the null hypothesis |

Every non-random bot shares one scoring engine and differs only by a weight
vector (`src/bots.rs`). That is deliberate: when two archetypes are compared,
the only thing that differs is their preferences, not their implementation
quality. To add a strategy, add a preset to `Weights::preset` — no engine code.

Bots see a `View`, not the game state. Opponents' hand contents and unplayed
development cards are hidden, so a strategy cannot quietly win by cheating.

## Searching for a better strategy

```bash
catan_sim tune --base balanced --rounds 20 --candidates 5 --eval-games 2000
```

A (1+lambda) hill climb. Each round mutates the champion's weights several
ways and tests each variant. Two details make it trustworthy:

- **The champion is re-measured every round.** Comparing candidates against a
  fixed null of `1/n_players` is right in pure self-play by symmetry, but
  wrong against a fixed field: if the champion already beats that field at
  30%, a 26% bar promotes everything and the search becomes a random walk
  selecting on noise. (This bug cost one 180k-game run here.)
- **Common random numbers.** The champion and every candidate in a round play
  the same seeds, so they face the same boards and the same dice. They share
  the luck, and the variance of the *difference* is much smaller than the
  variance of either alone.

`--against a,b,c` tunes against a fixed field instead of self-play. Self-play
is the safer objective; a fixed field can be exploited rather than beaten in
general, so validate the result against a field it was *not* tuned on.

The `tuned` preset came out of this: ~440k games of search, and it beats every
hand-written preset (28.1% against three `city_builder`s, 30.6% against three
`balanced`, where chance is 25%). Its character versus `balanced`: trade much
more readily (+125% offer rate, −44% acceptance threshold) while guarding the
leader more (+28%), favour grain and ore over wood and brick, buy 29% fewer
development cards, and weight single-intersection resource diversity 59% less.

## Timing, reach and denial: what was missing and what it was worth

An early version scored `Roll` at a flat 1000, which made it arithmetically
impossible for any bot ever to play a knight before rolling — the maximum
knight score was about 106. That was a specification bug, not a tuning
choice. The robber changes production on *the roll that follows it*, so a
knight played before you roll affects **this** roll: it can free one of your
own hexes so it pays out immediately, or deny an opponent on your roll too.
Played afterwards, neither applies until next time round.

The pre-roll decision is now scored as a pure timing question — the knight can
still be played after the roll, so what is priced is the difference:

```
premium = (my cards freed this roll + endgame_urgency * cards denied to the leader) * 5/6
score   = knight_timing * premium - forgone_other_dev_card - preroll_risk
```

The 5/6 is the chance this roll is not a 7, which would move the robber again
and throw the placement away. Pre-roll knights went from **0% to 41-59%** of
all knights played.

Parameter sweeps, each 15k games against three copies of a bot with all four
features off, where 25.0% is break-even:

| feature | parameter | off | best | gain |
|---|---|---|---|---|
| road reach weighting | `reach_decay` 0.15 → 0.8 | 24.7% | **25.8%** | +1.1pp |
| pre-roll knights | `knight_timing` 0 → 15 | 24.7% | **25.4%** | +0.7pp |
| racing for contested spots | `contest_drive` 0 → 1.5-3 | 24.7% | 25.7% | +0.5-1.0pp, noisy |
| denying opponents spots | `block_drive` 0 → 3 | 24.7% | 24.6% | **none** |

Three results worth recording, because each is a null:

- **Looking further than two roads ahead is worthless.** `road_horizon_base`
  measured flat from 2 through 6 (26.0% / 26.0% / 25.9% / 25.9% / 25.9%). The
  gain above came from weighting the spot two roads out more heavily, not from
  a longer horizon. Distant board information is not actionable because the
  board changes before you get there.
- **Denial has no measurable value** at any setting. It is off by default. The
  hook is kept because the null may simply reflect having no lookahead.
- **"Plow with Road Building, then settle" cannot be expressed here.**
  Conditioning the card on holding settlement resources changes nothing, and
  instrumenting it shows why: Road Building is played while a settlement is
  affordable in only **2% of plays**, because whenever a settlement *is*
  affordable the bot simply builds it, which outscores the card by a wide
  margin. Capturing the play needs `play card -> lay two roads -> settle`
  scored as one package against "settle the worse spot I can already reach".
  A greedy single-action scorer cannot represent that, so this remains a known
  gap rather than a tuning problem.

Together the live features are worth about +0.7pp of win rate, and they raise
`tuned` from 31.0% to **33.3%** against the preset field. Notably they do not
raise mean victory points (7.49 against 7.52) — timing and denial are zero-sum
tempo effects that change who crosses ten first rather than how many points
anyone accumulates.

## Sequential lookahead, and why denial still does not pay

Scoring single actions greedily cannot represent "play Road Building, lay two
roads, settle the good spot", because each step on its own looks worse than
settling whatever is already in reach. A bounded planner now enumerates
settlement targets up to four roads away, prices the whole package, and lifts
the first step of the best plan onto the plan's own score.

Getting the *scale* right was the whole difficulty, and the first attempt had
it wrong. A plan's first step is usually a road, which scores about 16, while
its competitor is settling a mediocre spot now, which scores about 75 because
of the large `w_settlement` constant. A bonus of `plan_drive * excess` could
never close a 59-point gap, so the planner had no effect at any setting. Plans
are now scored the way `BuildSettlement` is scored, with each road the route
cannot get free charged at two cards — which is exactly what makes Road
Building pivotal, since it removes two of those charges.

**It still does not move the win rate, and neither does denial.** The planner
is not inert, and the counters in `bots.rs` prove it rather than leaving it to
argument. Over 300 games:

| | count |
|---|---|
| decisions that consulted the planner | 24,853 |
| a settlement was affordable at all | 3,600 |
| some better spot was reachable | 20,549 |
| **both, plus a payable route** | **197** |
| of those, leaning on a Road Building card | 64 |
| the plan's first step was chosen | 153 |
| **the plan changed the decision** | **72** |

That is roughly one decision in four games. Better spots are nearly always out
there (83% of decisions) and the binding constraint is resources: a settlement
is affordable in 14% of decisions, and only 5.5% of *those* can also pay for a
route to something better. A play that arises a quarter of a time per game
cannot shift a win rate however well it is handled.

A plausible explanation — that the 7-card hand limit suppresses it, since
plowing needs roughly 8 cards of the right types — was tested and is **wrong**:
raising the limit to 14 changes nothing (24.9%). The bot spends resources the
moment anything is affordable, so it never accumulates regardless of the limit.

Denial remains worthless at every setting, with the planner on or off. Beyond
the rarity above, there is a strategic reason to expect this in a four-player
game: blocking one opponent helps the other two as much as it helps you, so it
is a private cost paid for a public good.

## Where strategies gain ground: phase snapshots

Aggregate win rates hide *when* a strategy works. Every player's position is
recorded at three boundaries — the end of setup, and the first moments the
leader reaches 40% and 70% of the target — so the game splits into early, mid
and late. Boundaries are set by the leader's score rather than by turn number,
because game lengths vary: "two thirds done" is a statement about how close
someone is to winning, not about elapsed turns. Rates are per *own* turn, so a
strategy is not penalised for short games. The setup baseline matters: without
it the early phase starts from zero and credits everyone with the two points
their opening settlements simply came with.

What it shows (30k games, `tuned`/`city_builder`/`dev_rush`/`expander`):

| strategy | phase | VP/turn | cards/turn | pips | lead% | win if leading |
|---|---|---|---|---|---|---|
| tuned | early | 0.170 | 3.14 | 24.4 | 24% | **51%** |
| | mid | 0.301 | 4.15 | 35.6 | 27% | **69%** |
| | late | **0.384** | 5.70 | 51.2 | | |
| city_builder | early | 0.167 | 2.81 | 24.6 | 23% | 40% |
| | mid | 0.279 | 3.97 | 35.9 | 21% | 61% |
| | late | 0.337 | 5.50 | 49.2 | | |
| dev_rush | early | 0.160 | 2.90 | 20.7 | 25% | 31% |
| | mid | **0.313** | 3.50 | 24.4 | 26% | 48% |
| | late | 0.262 | 4.25 | 33.3 | | |
| expander | early | **0.179** | 2.87 | 23.9 | **27%** | **25%** |
| | mid | 0.301 | 3.42 | 32.3 | 25% | 45% |
| | late | 0.250 | 4.81 | 46.1 | | |

Three things the aggregate numbers could not show:

- **The early game is nearly symmetric.** Every strategy scores between 0.160
  and 0.179 VP per turn and leads in 23-27% of games. Nothing is decided there.
- **The ranking inverts between phases.** `dev_rush` has the *best* mid-game
  rate (0.313) and the *worst* late one (0.262); `tuned` and `city_builder`
  accelerate instead. `expander` leads most often early (27%) and converts
  worst of all.
- **Closing, not leading, is what separates them.** `expander` and `tuned`
  reach the end of the early phase in front about equally often, but `expander`
  goes on to win a quarter of those games and `tuned` half. That 2:1 gap in
  conversion, not any difference in getting ahead, is where the win rate comes
  from.

The breakdown also dates the cost of refusing to trade. `hermit` is identical
to `balanced` except that it never trades, and its deficit compounds:
**-18%** VP per turn in the early phase, **-23%** in the mid, **-31%** in the
late. Without trading you end up sitting on resources you cannot use, and that
bites hardest once cities need specific cards.

## Valuing opening positions in pips

```bash
catan_sim placement -n 60000 --strategy tuned --jsonl openings.jsonl
```

Everything about a starting position is priced in **adjusted pips**: one
adjusted pip is the value of one pip of average production, so "a 3:1 port is
worth 1.9 adjusted pips" means it is worth as much as 1.9 extra pips of
production.

The design point: in a normal game bots *choose* where to start, so opening
position is entangled with strategy and the estimates would be confounded. So
every player places uniformly at random and then plays the same strategy.
Position becomes exogenous and the coefficients can be read as the value of
the ground.

Two calibration traps this handles:

- Random openings average about 13 pips against 19-20 for chosen ones, and the
  value of a pip *falls* as pips accumulate (the squared term is strongly
  significant). Fitting over the whole range would calibrate the pip unit on
  positions nobody plays. The headline fit is therefore restricted to openings
  of 14+ pips, with the full-range fit printed alongside for comparison.
- Port intersections sit on the coast and so touch fewer hexes, making port
  access negatively correlated with raw pips. Controlling for pips by resource
  is what separates "this port is valuable" from "port spots are weak spots" —
  and it is why ports look bad in a naive comparison.

Results are checked against a second model on win probability rather than
victory points, against randomised port positions, and against a different
subject strategy. **Placement values turn out to be strategy-dependent** — ore
is worth 1.05 adjusted pips to `tuned` but 1.48 to `balanced`, which buys more
development cards — so quote them with the strategy attached.

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

Strategies can be given as `preset+key=value;key=value` to override weights,
and `label:preset+...` to name a variant so two variants of one preset are not
pooled under the base name in the report. `catan_sim list` shows the presets;
overrides are clamped to sane bounds and a typo is an error rather than a
silent fallback.

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
