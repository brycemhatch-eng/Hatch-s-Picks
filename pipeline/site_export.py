"""Write the JSON files the Hatch's Picks website reads (docs/data/*.json).

  ml.json        model report, this season's game cards, line shop        (from export.py)
  rankings.json  power rankings: current Elo, offense/defense EPA, QB, record, Elo history
  matchups.json  per upcoming/recent game: side-by-side team stats the model used
  history.json   every out-of-sample game 2006-last season, compact, for the strategy simulator
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
SITE = ROOT / "docs" / "data"

STATS = ["elo", "off_epa_pp", "off_pass_epa_pp", "off_rush_epa_pp", "off_sr", "off_expl", "off_ppd", "off_give_rate",
         "off_sack_rate", "off_pts_pg", "def_epa_pp", "def_pass_epa_pp", "def_rush_epa_pp", "def_sr", "def_expl",
         "def_ppd", "def_give_rate", "def_sack_rate", "def_pts_pg", "qb_epa", "qb_exp", "inj_off", "inj_def", "rest"]


def r(x, n=3):
    try:
        return None if pd.isna(x) else round(float(x), n)
    except (TypeError, ValueError):
        return None


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    return v


def dump(name, obj):
    (SITE / name).write_text(json.dumps(clean(obj), separators=(",", ":"), allow_nan=False))
    print(f"  {name}: {(SITE / name).stat().st_size / 1e3:.0f} KB")


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT / "ml.json", SITE / "ml.json")
    t = pd.read_parquet(DATA / "team_games.parquet")
    t["gameday"] = pd.to_datetime(t.gameday)
    season = int(t.season.max())
    teams = pd.read_csv(ROOT / "raw" / "teams.csv").set_index("team")

    # ---------- rankings: each team's state going into its next (or last) game ----------
    cur = t[t.season == season].sort_values("gameday")
    nxt = cur[~cur.is_final].groupby("team").head(1).set_index("team")
    last = cur.groupby("team").tail(1).set_index("team")
    state = nxt.combine_first(last)
    lastp = cur[cur.is_final].groupby("team").tail(1).set_index("team")
    qb_of = lambda row: row.home_qb_name if row.side == "home" else row.away_qb_name
    played = cur[cur.is_final & (cur.game_type == "REG")]
    rec = played.assign(w=(played.points > played.opp_points).astype(int), l=(played.points < played.opp_points).astype(int),
                        tie=(played.points == played.opp_points).astype(int), pd_=played.points - played.opp_points) \
        .groupby("team")[["w", "l", "tie", "pd_"]].sum()
    hist = t[(t.season >= season - 4) & t.is_final].sort_values("gameday")
    rows = []
    for tm, s in state.iterrows():
        h = hist[hist.team == tm]
        rows.append({
            "team": tm, "name": teams.loc[tm, "team_name"], "nick": teams.loc[tm, "team_nick"],
            "conf": teams.loc[tm, "conference"], "div": teams.loc[tm, "division"],
            "color": teams.loc[tm, "color"], "color2": teams.loc[tm, "color2"],
            "w": int(rec.loc[tm, "w"]) if tm in rec.index else 0, "l": int(rec.loc[tm, "l"]) if tm in rec.index else 0,
            "t": int(rec.loc[tm, "tie"]) if tm in rec.index else 0, "pd": int(rec.loc[tm, "pd_"]) if tm in rec.index else 0,
            "qb": qb_of(s) if isinstance(qb_of(s), str) else (qb_of(lastp.loc[tm]) if tm in lastp.index else None),
            **{k: r(s[k]) for k in STATS},
            "net_epa": r(s.off_epa_pp - s.def_epa_pp),
            "elo_hist": [[str(d)[:10], r(e, 0), int(sn), int(wk)] for d, e, sn, wk in zip(h.gameday, h.elo, h.season, h.week)],
        })
    rk = pd.DataFrame(rows)
    for col, asc in (("elo", False), ("off_epa_pp", False), ("def_epa_pp", True), ("net_epa", False), ("qb_epa", False)):
        rk[f"rk_{col}"] = rk[col].rank(ascending=asc, method="min").astype(int)
    dump("rankings.json", {"season": season, "teams": rk.sort_values("elo", ascending=False).to_dict("records")})

    # ---------- matchups: home/away side-by-side for every game this season ----------
    m = {}
    for gid, grp in cur.groupby("game_id"):
        sides = {}
        for _, row in grp.iterrows():
            sides[row.side] = {k: r(row[k]) for k in STATS} | {
                "team": row.team, "qb": row.home_qb_name if row.side == "home" else row.away_qb_name}
        m[gid] = sides
    dump("matchups.json", m)

    # ---------- history: compact out-of-sample games for the simulator ----------
    pr = pd.read_parquet(OUT / "predictions.parquet")
    h = pr[pr.result.notna() & (pr.season < season)].copy()
    cols = ["season", "week", "away_team", "home_team", "pred_margin", "pred_total", "p_home", "result", "total",
            "spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds", "under_odds",
            "home_moneyline", "away_moneyline"]
    h = h[cols]
    for c in ("home_spread_odds", "away_spread_odds", "over_odds", "under_odds"):
        h[c] = h[c].fillna(-110)
    rows = [[int(x.season), int(x.week), x.away_team, x.home_team, r(x.pred_margin, 1), r(x.pred_total, 1), r(x.p_home, 3),
             int(x.result), int(x.total), r(x.spread_line, 1), r(x.total_line, 1), r(x.home_spread_odds, 0),
             r(x.away_spread_odds, 0), r(x.over_odds, 0), r(x.under_odds, 0), r(x.home_moneyline, 0), r(x.away_moneyline, 0)]
            for x in h.itertuples(index=False)]
    dump("history.json", {"cols": cols, "rows": rows})


if __name__ == "__main__":
    main()
