"""Turn the walk-forward model output into ml.json for the Gridiron ML dashboard.

Edge probabilities are *calibrated on the out-of-sample backtest*, not taken from the raw model:
  - spread / total: logistic fit of "pick won" on the size of the model-vs-line gap
  - moneyline: logistic fit of home win on logit(market) + logit(model)
so the EV numbers reflect how often edges of that size actually cashed 2006-2025.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def american_prob(o):
    o = np.asarray(o, float)
    o = np.where(o == 0, -110, o)
    return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))


def payout(o):
    o = np.asarray(o, float)
    return np.where(o > 0, o / 100, 100 / -o)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def r(x, n=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def main():
    pr = pd.read_parquet(OUT / "predictions.parquet")
    rep = json.loads((OUT / "report.json").read_text())
    current = int(pr.season.max())
    done = pr[pr.result.notna()].copy()
    hist = done[done.season < current].copy()

    for c, d in (("home_spread_odds", -110), ("away_spread_odds", -110), ("over_odds", -110), ("under_odds", -110)):
        pr[c] = pr[c].fillna(d)
        hist[c] = hist[c].fillna(d)

    # ---- calibrators ---------------------------------------------------------------------------
    sp = hist[(hist.result != hist.spread_line)]
    gap = (sp.pred_margin - sp.spread_line).abs().to_numpy()
    won = np.where(sp.pred_margin > sp.spread_line, sp.result > sp.spread_line, sp.result < sp.spread_line)
    cal_sp = LogisticRegression().fit(gap.reshape(-1, 1), won.astype(int))

    to = hist[(hist.total != hist.total_line)]
    gap_t = (to.pred_total - to.total_line).abs().to_numpy()
    won_t = np.where(to.pred_total > to.total_line, to.total > to.total_line, to.total < to.total_line)
    cal_to = LogisticRegression().fit(gap_t.reshape(-1, 1), won_t.astype(int))

    def mkt_home(df):
        ph, pa = american_prob(df.home_moneyline), american_prob(df.away_moneyline)
        return ph / (ph + pa)

    ml = hist[hist.home_moneyline.notna() & hist.away_moneyline.notna() & (hist.result != 0)].copy()
    ml["mkt"] = mkt_home(ml)
    X = np.c_[logit(ml.mkt), logit(ml.p_home)]
    cal_ml = LogisticRegression(C=100).fit(X, (ml.result > 0).astype(int))

    # ---- calibration table (model win prob vs reality) -----------------------------------------
    ml["bin"] = pd.cut(ml.p_home, np.linspace(0, 1, 11))
    calib = (ml.groupby("bin", observed=True)
               .agg(pred=("p_home", "mean"), mkt=("mkt", "mean"), actual=("result", lambda s: (s > 0).mean()), n=("p_home", "size"))
               .reset_index(drop=True))
    calib = [{k: r(v, 3) if k != "n" else int(v) for k, v in row.items()} for row in calib.to_dict("records")]

    # ---- cumulative units by season, edge >= 3 -------------------------------------------------
    def pnl_spread(df, thr=3):
        e = df.pred_margin - df.spread_line
        home = e > 0
        cov = np.sign(df.result - df.spread_line)
        w = np.where(home, cov > 0, cov < 0)
        odds = np.where(home, df.home_spread_odds, df.away_spread_odds)
        p = np.where(cov == 0, 0, np.where(w, payout(odds), -1))
        return pd.Series(np.where(e.abs() >= thr, p, 0), index=df.index)

    def pnl_total(df, thr=3):
        e = df.pred_total - df.total_line
        over = e > 0
        side = np.sign(df.total - df.total_line)
        w = np.where(over, side > 0, side < 0)
        odds = np.where(over, df.over_odds, df.under_odds)
        p = np.where(side == 0, 0, np.where(w, payout(odds), -1))
        return pd.Series(np.where(e.abs() >= thr, p, 0), index=df.index)

    hist["u_sp"], hist["u_to"] = pnl_spread(hist), pnl_total(hist)
    by = hist.groupby("season")[["u_sp", "u_to"]].sum()
    curve = [{"season": int(s), "spread": r(a), "total": r(b), "spread_cum": r(ca), "total_cum": r(cb)}
             for s, a, b, ca, cb in zip(by.index, by.u_sp, by.u_to, by.u_sp.cumsum(), by.u_to.cumsum())]

    # ---- game cards ------------------------------------------------------------------------------
    def card(g):
        sl, tl = g.spread_line, g.total_line
        e_sp = g.pred_margin - sl if pd.notna(sl) else np.nan
        e_to = g.pred_total - tl if pd.notna(tl) else np.nan
        o = {
            "id": g.game_id, "week": int(g.week), "type": g.game_type, "day": str(g.gameday)[:10], "weekday": g.weekday,
            "time": g.gametime, "home": g.home_team, "away": g.away_team,
            "home_qb": g.home_qb_name, "away_qb": g.away_qb_name, "stadium": g.stadium, "roof": g.roof,
            "pred_home": r(g.pred_home), "pred_away": r(g.pred_away), "margin": r(g.pred_margin), "total": r(g.pred_total),
            "p_home": r(g.p_home, 3), "spread_line": r(sl), "total_line": r(tl),
            "home_ml": r(g.home_moneyline, 0), "away_ml": r(g.away_moneyline, 0),
            "home_sp_odds": r(g.home_spread_odds, 0), "away_sp_odds": r(g.away_spread_odds, 0),
            "over_odds": r(g.over_odds, 0), "under_odds": r(g.under_odds, 0),
            "home_score": r(g.home_score, 0), "away_score": r(g.away_score, 0), "final": bool(pd.notna(g.result)),
            "bets": [],
        }
        if pd.notna(e_sp) and abs(e_sp) >= 0.5:
            home = e_sp > 0
            p = float(cal_sp.predict_proba([[abs(e_sp)]])[0, 1])
            odds = g.home_spread_odds if home else g.away_spread_odds
            team = g.home_team if home else g.away_team
            line = sl if home else -sl  # nfldata spread_line = expected home margin; bet line is minus that for the home side
            o["bets"].append({"market": "Spread", "pick": f"{team} {(-line):+g}" if line else f"{team} PK", "edge": r(abs(e_sp)),
                              "unit": "pts", "prob": r(p, 3), "odds": r(odds, 0),
                              "ev": r(100 * (p * float(payout(odds)) - (1 - p)), 1)})
            if o["final"]:
                cov = np.sign(g.result - sl)
                o["bets"][-1]["result"] = "P" if cov == 0 else ("W" if (cov > 0) == home else "L")
        if pd.notna(e_to) and abs(e_to) >= 0.5:
            over = e_to > 0
            p = float(cal_to.predict_proba([[abs(e_to)]])[0, 1])
            odds = g.over_odds if over else g.under_odds
            o["bets"].append({"market": "Total", "pick": f"{'Over' if over else 'Under'} {tl:g}", "edge": r(abs(e_to)),
                              "unit": "pts", "prob": r(p, 3), "odds": r(odds, 0),
                              "ev": r(100 * (p * float(payout(odds)) - (1 - p)), 1)})
            if o["final"]:
                s = np.sign(g.total - tl)
                o["bets"][-1]["result"] = "P" if s == 0 else ("W" if (s > 0) == over else "L")
        if pd.notna(g.home_moneyline) and pd.notna(g.away_moneyline):
            mk = float(mkt_home(pd.DataFrame([g]))[0])
            fair = float(cal_ml.predict_proba([[logit(mk), logit(g.p_home)]])[0, 1])
            o["mkt_home"] = r(mk, 3)
            o["fair_home"] = r(fair, 3)
            home = fair > mk
            p = fair if home else 1 - fair
            odds = g.home_moneyline if home else g.away_moneyline
            o["bets"].append({"market": "Moneyline", "pick": f"{g.home_team if home else g.away_team} ML",
                              "edge": r(100 * abs(fair - mk)), "unit": "%", "prob": r(p, 3), "odds": r(odds, 0),
                              "ev": r(100 * (p * float(payout(odds)) - (1 - p)), 1)})
            if o["final"] and g.result != 0:
                o["bets"][-1]["result"] = "W" if (g.result > 0) == home else "L"
        return o

    season = pr[pr.season == current].sort_values(["gameday", "gametime", "game_id"])
    upcoming = season[season.result.isna()]
    next_week = int(upcoming.week.min()) if len(upcoming) else None
    cards = [card(g) for g in season.itertuples(index=False)]

    # YTD record of the model's own picks this season (same rules as the backtest: edge >= 3 / >= 3 / >= 5%)
    ytd = {}
    for mk, thr in (("Spread", 3), ("Total", 3), ("Moneyline", 5)):
        res = [b for c in cards if c["final"] for b in c["bets"] if b["market"] == mk and b["edge"] >= thr and "result" in b]
        units = 0.0
        for b in res:
            units += 0 if b["result"] == "P" else (float(payout(b["odds"])) if b["result"] == "W" else -1)
        ytd[mk] = {"w": sum(b["result"] == "W" for b in res), "l": sum(b["result"] == "L" for b in res),
                   "p": sum(b["result"] == "P" for b in res), "units": round(units, 2)}

    out = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "season": current, "next_week": next_week,
        "data_through": str(done.gameday.max())[:10],
        "train_games": int(pd.read_parquet(ROOT / "data" / "games.parquet").query("is_final").shape[0]), "first_season": 1999,
        "metrics": rep["metrics"], "backtest": rep["backtest"], "ats_by_season": rep["ats_by_season"],
        "importance": rep["importance"], "calibration": calib, "curve": curve,
        "edge_curve": {
            "spread": [{"edge": e, "prob": r(cal_sp.predict_proba([[e]])[0, 1], 3)} for e in range(0, 9)],
            "total": [{"edge": e, "prob": r(cal_to.predict_proba([[e]])[0, 1], 3)} for e in range(0, 9)],
        },
        "games": cards, "ytd": ytd,
    }
    # ---- line shopping: model vs every sportsbook's own line and price -----------------------------
    bo_path = ROOT / "raw" / "book_odds.json"
    if bo_path.exists():
        bo = json.loads(bo_path.read_text())
        open_games = {f"{g.away_team}@{g.home_team}": g for g in upcoming.itertuples(index=False)}
        shop = []
        for row in bo["rows"]:
            g = open_games.get(row["game"])
            if g is None:
                continue
            a, h = row["game"].split("@")
            if row.get("home_spread") is not None:
                e = g.pred_margin + row["home_spread"]           # >0: home covers by e points in the model
                home = e > 0
                price = row["home_sp_price"] if home else row["away_sp_price"]
                line = row["home_spread"] if home else -row["home_spread"]
                p = float(cal_sp.predict_proba([[abs(e)]])[0, 1])
                shop.append({"game": row["game"], "week": int(g.week), "day": str(g.gameday)[:10], "market": "Spread",
                             "side": h if home else a, "pick": f"{h if home else a} {line:+g}" if line else f"{h if home else a} PK",
                             "line": line, "book": row["book"], "price": price, "edge": r(abs(e)), "prob": r(p, 3),
                             "ev": r(100 * (p * float(payout(price)) - (1 - p)), 1)})
            if row.get("total") is not None:
                e = g.pred_total - row["total"]
                over = e > 0
                price = row["over_price"] if over else row["under_price"]
                p = float(cal_to.predict_proba([[abs(e)]])[0, 1])
                shop.append({"game": row["game"], "week": int(g.week), "day": str(g.gameday)[:10], "market": "Total",
                             "side": "Over" if over else "Under", "pick": f"{'Over' if over else 'Under'} {row['total']:g}",
                             "line": row["total"], "book": row["book"], "price": price, "edge": r(abs(e)), "prob": r(p, 3),
                             "ev": r(100 * (p * float(payout(price)) - (1 - p)), 1)})
            if row.get("home_ml") is not None and pd.notna(g.home_moneyline):
                mk = float(mkt_home(pd.DataFrame([g]))[0])
                fair = float(cal_ml.predict_proba([[logit(mk), logit(g.p_home)]])[0, 1])
                home = fair > mk
                price = row["home_ml"] if home else row["away_ml"]
                p = fair if home else 1 - fair
                shop.append({"game": row["game"], "week": int(g.week), "day": str(g.gameday)[:10], "market": "Moneyline",
                             "side": h if home else a, "pick": f"{h if home else a} ML", "line": None, "book": row["book"],
                             "price": price, "edge": r(100 * abs(fair - mk)), "prob": r(p, 3),
                             "ev": r(100 * (p * float(payout(price)) - (1 - p)), 1)})
        out["shop"] = {"fetched_at": bo.get("fetched_at"), "source": bo.get("source"),
                       "books": sorted({x["book"] for x in bo["rows"]}), "rows": shop}

    teams = pd.read_csv(ROOT / "raw" / "teams.csv")
    out["teams"] = {t: {"name": n, "color": c} for t, n, c in zip(teams.team, teams.team_name, teams.color)}
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    (OUT / "ml.json").write_text(json.dumps(clean(out), separators=(",", ":"), allow_nan=False))
    print("ml.json:", len(cards), "games; next week", next_week, "; ytd", ytd)


if __name__ == "__main__":
    main()
