"""Sportsbook odds from The Odds API -> raw/book_odds.json (one row per game x book).

Key: ODDS_API_KEY env var, or ~/.config/gridiron/odds.env (never committed or published).
If the API can't be reached, the last saved file is kept so the dashboard still builds.
"""
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "raw" / "book_odds.json"
BOOKS = "draftkings,fanduel,betmgm,williamhill_us,espnbet,betrivers,fanatics,hardrockbet"
LABEL = {"draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
         "espnbet": "ESPN BET", "betrivers": "BetRivers", "fanatics": "Fanatics", "hardrockbet": "Hard Rock"}


def key():
    if os.environ.get("ODDS_API_KEY"):
        return os.environ["ODDS_API_KEY"]
    f = Path.home() / ".config" / "gridiron" / "odds.env"
    if f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("ODDS_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def normalize(events):
    teams = pd.read_csv(ROOT / "raw" / "teams.csv")
    code = dict(zip(teams.team_name, teams.team))
    rows = []
    for ev in events:
        h, a = code.get(ev["home_team"]), code.get(ev["away_team"])
        if not h or not a:
            continue
        for bk in ev.get("bookmakers", []):
            r = {"game": f"{a}@{h}", "kickoff": ev["commence_time"], "book": LABEL.get(bk["key"], bk["title"])}
            for m in bk["markets"]:
                o = {x["name"]: x for x in m["outcomes"]}
                if m["key"] == "spreads" and ev["home_team"] in o and ev["away_team"] in o:
                    r.update(home_spread=o[ev["home_team"]]["point"], home_sp_price=o[ev["home_team"]]["price"],
                             away_sp_price=o[ev["away_team"]]["price"])
                elif m["key"] == "totals" and "Over" in o and "Under" in o:
                    r.update(total=o["Over"]["point"], over_price=o["Over"]["price"], under_price=o["Under"]["price"])
                elif m["key"] == "h2h" and ev["home_team"] in o and ev["away_team"] in o:
                    r.update(home_ml=o[ev["home_team"]]["price"], away_ml=o[ev["away_team"]]["price"])
            rows.append(r)
    return rows


def main():
    k = key()
    if not k:
        print("odds: no API key; keeping last saved book odds")
        return
    url = ("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
           f"?apiKey={k}&regions=us&markets=h2h,spreads,totals&oddsFormat=american&bookmakers={BOOKS}")
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            events = json.load(resp)
            left = resp.headers.get("x-requests-remaining")
    except Exception as e:
        print(f"odds: API unreachable ({e.__class__.__name__}); keeping last saved book odds")
        return
    rows = normalize(events)
    OUT.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                               "source": "The Odds API", "rows": rows}, indent=1))
    print(f"odds: {len(rows)} game-book rows saved; {left} API requests left this month")


if __name__ == "__main__":
    main()
