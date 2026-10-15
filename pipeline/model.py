"""Train, walk-forward test, and compare to Vegas.

Model: predict each team's points from its offense vs the opponent's defense
(plus QB, Elo, situation, weather, injuries). Two learners are blended:
  * Ridge regression on standardized features (stable, linear)
  * LightGBM gradient-boosted trees (non-linear interactions)
Game outputs: home/away score -> margin and total. Win probability comes
from the predicted margin through a logistic curve fit on training seasons.

Testing is walk-forward: to predict season S the model is trained only on
seasons before S, so every number in the backtest is out-of-sample.
Vegas lines are never model inputs, only used for evaluation.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)

FIRST_TRAIN = 2001          # 1999-2000 used only to warm up ratings
FIRST_TEST = 2006           # moneyline prices exist from 2006
RATES = ["epa_pp", "pass_epa_pp", "rush_epa_pp", "sr", "expl", "sack_rate", "give_rate", "early_epa",
         "third_rate", "ppd", "rz_td", "proe", "pass_rate", "plays_pg", "st_pg", "pen_pg", "pts_pg"]


def feature_frame(t: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=t.index)
    for r in RATES:                               # my offense vs their defense, and the reverse matchup context
        X[f"off_{r}"] = t[f"off_{r}"]
        X[f"oppdef_{r}"] = t[f"opp_def_{r}"]
    for r in ("epa_pp", "pts_pg", "sr", "plays_pg"):
        X[f"def_{r}"] = t[f"def_{r}"]            # own defense affects pace / field position
        X[f"oppoff_{r}"] = t[f"opp_off_{r}"]
    X["matchup_epa"] = t.off_epa_pp + t.opp_def_epa_pp
    X["matchup_pass"] = t.off_pass_epa_pp + t.opp_def_pass_epa_pp
    X["matchup_rush"] = t.off_rush_epa_pp + t.opp_def_rush_epa_pp
    X["elo_diff"] = t.elo - t.opp_elo
    X["qb_epa"], X["opp_qb_epa"] = t.qb_epa, t.opp_qb_epa
    X["qb_exp"], X["qb_new"] = t.qb_exp, t.qb_new
    X["home"] = t.is_home
    X["rest_diff"] = (t.rest - t.opp_rest).clip(-7, 7)
    X["short_week"] = (t.rest <= 5).astype(int)
    X["div"] = t.div_game.fillna(0)
    X["playoff"] = t.playoff
    X["dome"], X["grass"] = t.dome, t.grass
    X["temp"], X["wind"] = t.temp_f.clip(-10, 110), t.wind_mph.clip(0, 40)
    X["week"] = t.week
    X["season"] = t.season                       # scoring environment drifts over eras
    X["games_info"] = t.off_games
    X["inj_off"], X["inj_def"], X["inj_qb"] = t.inj_off, t.inj_def, t.inj_qb
    X["opp_inj_off"], X["opp_inj_def"], X["opp_inj_qb"] = t.opp_inj_off, t.opp_inj_def, t.opp_inj_qb
    X["inj_known"] = t.inj_off.notna().astype(int)
    return X


import os as _os
LGB_PARAMS = dict(objective="regression", learning_rate=float(_os.environ.get("LR", "0.015")),
                  n_estimators=int(_os.environ.get("NEST", "500")), num_leaves=int(_os.environ.get("LEAVES", "7")),
                  min_child_samples=60, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                  reg_lambda=5.0, verbose=-1)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame):
    Xtr, Xte = feature_frame(train), feature_frame(test)
    ytr = train.points.values
    w = float(_os.environ.get("RECENCY", "0.92")) ** (train.season.max() - train.season)          # recent seasons weigh more
    lin_cols = [c for c in Xtr.columns if c not in ("season", "off_proe", "oppdef_proe")]
    med = Xtr[lin_cols].median().fillna(0)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=30.0))
    ridge.fit(Xtr[lin_cols].fillna(med), ytr, ridge__sample_weight=w)
    gbm = lgb.LGBMRegressor(**LGB_PARAMS)
    gbm.fit(Xtr, ytr, sample_weight=w)
    p_r = ridge.predict(Xte[lin_cols].fillna(med))
    p_g = gbm.predict(Xte)
    tr_r = ridge.predict(Xtr[lin_cols].fillna(med))
    tr_g = gbm.predict(Xtr)
    wr = float(__import__("os").environ.get("W_RIDGE", "0.7"))
    return wr * p_r + (1 - wr) * p_g, wr * tr_r + (1 - wr) * tr_g, gbm, lin_cols


def to_games(t: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    t = t.assign(pred=pred)
    h = t[t.side == "home"].set_index("game_id")
    a = t[t.side == "away"].set_index("game_id")
    g = pd.DataFrame({"pred_home": h.pred, "pred_away": a.pred.reindex(h.index)})
    g["pred_margin"] = g.pred_home - g.pred_away
    g["pred_total"] = g.pred_home + g.pred_away
    return g


def american_prob(o):
    return np.where(o > 0, 100 / (o + 100), -o / (-o + 100))


def payout(o):
    return np.where(o > 0, o / 100, 100 / -o)


def run() -> dict:
    t = pd.read_parquet(DATA / "team_games.parquet")
    games = pd.read_parquet(DATA / "games.parquet").set_index("game_id")
    t = t[t.season >= FIRST_TRAIN]
    done = t[t.is_final]
    results, importance = [], None
    current = int(t.season.max())
    for S in range(FIRST_TEST, current + 1):
        train = done[done.season < S]
        test = t[t.season == S]
        pred, tr_pred, gbm, cols = fit_predict(train, test)
        # win-probability calibration on training seasons: P(home win) = sigmoid(a * margin + b)
        tg = to_games(train, tr_pred).join(games[["result"]])
        tg = tg[tg.result.notna() & (tg.result != 0)]
        cal = LogisticRegression(C=10).fit(tg[["pred_margin"]], (tg.result > 0).astype(int))
        resid_sd_m = float(np.std(tg.result - tg.pred_margin))
        gtot = games.loc[tg.index, "total"]
        resid_sd_t = float(np.std(gtot - tg.pred_total))
        g = to_games(test, pred)
        g["p_home"] = cal.predict_proba(g[["pred_margin"]])[:, 1]
        g["sd_margin"], g["sd_total"] = resid_sd_m, resid_sd_t
        g["season"] = S
        results.append(g)
        importance = pd.Series(gbm.booster_.feature_importance("gain"), index=feature_frame(train).columns)
        print(f"  {S}: trained on {train.game_id.nunique():,} games")
    pr = pd.concat(results).drop(columns="season").join(games)

    # ------------------------------------------------ evaluation vs Vegas (completed, out-of-sample)
    ev = pr[pr.result.notna()].copy()
    ev["home_win"] = (ev.result > 0).astype(int)
    ml_ok = ev.home_moneyline.notna() & ev.away_moneyline.notna() & (ev.result != 0)
    ph, pa = american_prob(ev.home_moneyline.fillna(-110)), american_prob(ev.away_moneyline.fillna(-110))
    ev["mkt_home"] = np.where(ml_ok, ph / (ph + pa), np.nan)
    m = ev[ml_ok & (ev.season < current)]
    ev_all = ev[ev.season < current]
    metrics = {
        "test_seasons": f"{FIRST_TEST}-{current - 1}", "games": int(len(ev_all)),
        "margin_mae": {"model": round(float((ev_all.result - ev_all.pred_margin).abs().mean()), 2),
                       "vegas": round(float((ev_all.result - ev_all.spread_line).abs().mean()), 2)},
        "total_mae": {"model": round(float((ev_all.total - ev_all.pred_total).abs().mean()), 2),
                      "vegas": round(float((ev_all.total - ev_all.total_line).abs().mean()), 2)},
        "score_mae": round(float(pd.concat([(ev_all.home_score - ev_all.pred_home).abs(),
                                            (ev_all.away_score - ev_all.pred_away).abs()]).mean()), 2),
        "winner_acc": {"model": round(float(((m.p_home > .5) == m.home_win).mean()), 4),
                       "vegas": round(float(((m.mkt_home > .5) == m.home_win).mean()), 4)},
        "brier": {"model": round(float(((m.p_home - m.home_win) ** 2).mean()), 4),
                  "vegas": round(float(((m.mkt_home - m.home_win) ** 2).mean()), 4)},
        "corr_with_vegas_spread": round(float(np.corrcoef(ev_all.pred_margin, ev_all.spread_line)[0, 1]), 3),
    }

    # ------------------------------------------------ betting backtest (flat 1 unit)
    bt = ev[ev.season < current].copy()
    bt["h_sp_odds"] = bt.home_spread_odds.fillna(-110)
    bt["a_sp_odds"] = bt.away_spread_odds.fillna(-110)
    bt["o_odds"], bt["u_odds"] = bt.over_odds.fillna(-110), bt.under_odds.fillna(-110)
    rows = []
    for thr in (0, 1, 2, 3, 4, 5):
        # spread: model margin vs line
        e = bt.pred_margin - bt.spread_line
        sel = e.abs() >= max(thr, 0.5)
        home = e > 0
        cover = np.sign(bt.result - bt.spread_line)
        won = np.where(home, cover > 0, cover < 0)
        push = cover == 0
        odds = np.where(home, bt.h_sp_odds, bt.a_sp_odds)
        pnl = np.where(push, 0, np.where(won, payout(odds), -1))
        s = sel
        rows.append({"market": "Spread", "edge": f"≥{thr} pts", "bets": int(s.sum()),
                     "win_pct": round(float(won[s & ~push].mean()), 4) if s.sum() else None,
                     "units": round(float(pnl[s].sum()), 1), "roi": round(float(pnl[s].mean()), 4) if s.sum() else None})
        # totals
        e = bt.pred_total - bt.total_line
        sel = e.abs() >= max(thr, 0.5)
        over = e > 0
        side = np.sign(bt.total - bt.total_line)
        won = np.where(over, side > 0, side < 0)
        push = side == 0
        odds = np.where(over, bt.o_odds, bt.u_odds)
        pnl = np.where(push, 0, np.where(won, payout(odds), -1))
        s = sel
        rows.append({"market": "Total", "edge": f"≥{thr} pts", "bets": int(s.sum()),
                     "win_pct": round(float(won[s & ~push].mean()), 4) if s.sum() else None,
                     "units": round(float(pnl[s].sum()), 1), "roi": round(float(pnl[s].mean()), 4) if s.sum() else None})
    mb = bt[bt.mkt_home.notna()]
    for thr in (0.0, 0.03, 0.05, 0.08, 0.10):
        eh = mb.p_home - mb.mkt_home
        pick_home = eh > 0
        edge = np.where(pick_home, eh, -eh)
        sel = edge >= max(thr, 0.005)
        won = np.where(pick_home, mb.result > 0, mb.result < 0)
        odds = np.where(pick_home, mb.home_moneyline, mb.away_moneyline)
        pnl = np.where(won, payout(odds), -1)
        rows.append({"market": "Moneyline", "edge": f"≥{int(thr*100)}%", "bets": int(sel.sum()),
                     "win_pct": round(float(won[sel].mean()), 4), "units": round(float(pnl[sel].sum()), 1),
                     "roi": round(float(pnl[sel].mean()), 4)})
    backtest = pd.DataFrame(rows)

    # season-by-season spread record at ≥3 pts
    e = bt.pred_margin - bt.spread_line
    bt["ats_pick_home"] = e > 0
    cover = np.sign(bt.result - bt.spread_line)
    bt["ats_win"] = np.where(bt.ats_pick_home, cover > 0, cover < 0)
    bt["ats_push"] = cover == 0
    by_season = (bt[(e.abs() >= 3) & ~bt.ats_push].groupby("season").ats_win.agg(["mean", "size"])
                 .rename(columns={"mean": "win_pct", "size": "bets"}).reset_index())
    by_season["win_pct"] = by_season.win_pct.round(3)

    pr.reset_index().to_parquet(OUT / "predictions.parquet", index=False)
    backtest.to_csv(OUT / "backtest.csv", index=False)
    imp = (importance / importance.sum()).sort_values(ascending=False)
    out = {"metrics": metrics, "backtest": backtest.to_dict("records"), "ats_by_season": by_season.to_dict("records"),
           "importance": imp.round(4).head(20).to_dict()}
    (OUT / "report.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["metrics"], indent=2))
    print(pd.DataFrame(r["backtest"]).to_string(index=False))
    print(pd.Series(r["importance"]).head(12).to_string())
