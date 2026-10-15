# Hatch's Picks

**A machine-learning NFL model that projects every game's score and win probability, then prices each bet against Vegas and the major sportsbooks.**

🏈 **Live site:** https://brycemhatch-eng.github.io/Hatch-s-Picks/

The site rebuilds itself every morning with GitHub Actions. It pulls the latest play-by-play data and sportsbook odds, retrains the model, and publishes new picks.

| Page | What it shows |
|---|---|
| **This Week** | Best bets, model vs Vegas win % and projected scores for every game, and full game cards |
| **Game pages** | Stat comparison of the two teams, model vs Vegas, best price across sportsbooks, and Elo history |
| **Rankings** | Sortable power rankings (Elo, offense/defense EPA, QB) plus an offense-vs-defense chart |
| **History** | Query every game since 1999 (e.g. "opponent scored 35+", home underdogs, after a loss) and get each team's record, ATS and over/under |
| **Line Shop** | The model's side of every game priced at DraftKings, FanDuel, BetMGM, Caesars and more, with book-to-book discrepancies |
| **2026 Record** | Every flagged play this season, graded live |
| **Performance** | 20-season backtest vs Vegas, calibration, and an interactive strategy simulator |

## Results (out of sample, 2006–2025, 5,431 games)

| | Model | Vegas closing line |
|---|---|---|
| Avg margin miss | 10.38 pts | 10.23 pts |
| Picks the winner | 65.9% | 66.4% |
| Against the spread, 3+ pt edges | **55.1%** (1,393 bets) | break-even 52.4% |
| Totals, 3+ pt edges | **56.1%** (1,362 bets) | break-even 52.4% |

Vegas is slightly more accurate game by game, which is expected. The edge comes from the games where the model disagrees strongly with the line.

## How it works

1. **Data:** [nflverse](https://github.com/nflverse) play-by-play since 1999 (~1.3M plays), injury reports, snap counts, and game results with closing lines. Live odds come from [The Odds API](https://the-odds-api.com).
2. **Features (all known before kickoff):**
   - offense and defense EPA per play (pass and rush), success rate, explosive plays, points per drive, turnovers and sacks, all exponentially weighted toward recent games
   - Elo rating
   - starting-QB form
   - injury load (share of snaps held by players listed Out or Doubtful)
   - rest days, home field and weather
3. **Model:** ridge regression plus LightGBM, blended 70/30. Each predicts a team's points from its offense against the opponent's defense. A logistic layer converts the projected margin into win probability.
4. **Testing:** walk-forward by season. Each year is predicted by a model trained only on earlier years.
5. **Edge:** the gap between the model and each book's line is converted into a win probability using how often gaps of that size actually won in the backtest. That gives expected profit at the posted price.

```
pipeline/
  odds.py         sportsbook odds -> raw/book_odds.json
  run.py          one command: download -> features -> model -> export
  features.py     play-by-play -> team-game feature table
  model.py        walk-forward training + backtest vs Vegas
  export.py       game cards, edges, line shop
  site_export.py  JSON for the website (docs/data)
docs/             the website (GitHub Pages)
.github/workflows/daily.yml   daily refresh
```

Run it locally:

```bash
pip install -r requirements.txt
export ODDS_API_KEY=your_key   # optional; skips odds if missing
python pipeline/run.py         # first run downloads ~500 MB of nflverse data
cd docs && python -m http.server   # open http://localhost:8000
```

*For research and entertainment. Not betting advice.*
