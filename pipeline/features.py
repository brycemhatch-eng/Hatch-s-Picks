"""Pre-game feature engineering for every NFL game since 1999.

Every feature for a game is computed ONLY from games played before it
(no leakage): exponentially-weighted team efficiency that carries across
seasons with regression to the mean, quarterback form, Elo, rest, weather,
and injury-adjusted lineup strength.

Output: data/team_games.parquet  (one row per team per game, team vs opponent)
        data/games.parquet       (one row per game with Vegas lines + results)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "raw", ROOT / "data"
OUT.mkdir(exist_ok=True)
FRANCHISE = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

HALF_LIFE = 8            # games; recent games count more
DECAY = 0.5 ** (1 / HALF_LIFE)
SEASON_CARRY = 0.45      # share of last season's evidence kept into the new season
PRIOR_WEIGHT = 4         # pseudo-games of league average blended into every rate

PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
            "play_type", "pass", "rush", "qb_dropback", "epa", "success", "yards_gained", "sack",
            "interception", "fumble_lost", "down", "yardline_100", "third_down_converted",
            "third_down_failed", "pass_oe", "passer_player_id", "passer_player_name",
            "fixed_drive", "fixed_drive_result", "posteam_score", "posteam_score_post",
            "penalty", "penalty_team", "penalty_yards", "special_teams_play", "touchdown"]


# ------------------------------------------------------------------ 1. play-by-play -> team-game stats
def team_game_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    off_rows, qb_rows = [], []
    for f in sorted(RAW.glob("pbp_*.parquet")):
        avail = pd.read_parquet(f, columns=None).columns
        cols = [c for c in PBP_COLS if c in avail]
        p = pd.read_parquet(f, columns=cols)
        for c in set(PBP_COLS) - set(cols):
            p[c] = np.nan
        p = p[p.season_type.isin(["REG", "POST"])]

        scr = p[((p["pass"] == 1) | (p["rush"] == 1)) & p.epa.notna() & p.posteam.notna()
                & ~p.play_type.isin(["qb_kneel", "qb_spike", "no_play"])].copy()
        scr["is_pass"] = (scr["pass"] == 1).astype(float)
        scr["is_rush"] = 1 - scr.is_pass
        scr["early"] = scr.down.isin([1, 2]).astype(float)
        scr["third"] = (scr.down == 3).astype(float)
        scr["explosive"] = (((scr.is_pass == 1) & (scr.yards_gained >= 20))
                            | ((scr.is_rush == 1) & (scr.yards_gained >= 10))).astype(float)
        scr["dropback"] = scr.qb_dropback.fillna(scr.is_pass)
        for c in ("success", "sack", "interception", "fumble_lost", "third_down_converted", "pass_oe"):
            scr[c] = scr[c].fillna(0)
        g = scr.groupby(["game_id", "posteam"])
        a = pd.DataFrame({
            "plays": g.size(),
            "epa": g.epa.sum(),
            "pass_plays": g.is_pass.sum(), "pass_epa": g.apply(lambda d: (d.epa * d.is_pass).sum(), include_groups=False),
            "rush_plays": g.is_rush.sum(), "rush_epa": g.apply(lambda d: (d.epa * d.is_rush).sum(), include_groups=False),
            "success": g.success.sum(),
            "explosive": g.explosive.sum(),
            "dropbacks": g.dropback.sum(), "sacks": g.sack.sum(),
            "giveaways": g.interception.sum() + g.fumble_lost.sum(),
            "early_plays": g.early.sum(), "early_epa": g.apply(lambda d: (d.epa * d.early).sum(), include_groups=False),
            "third_att": g.third.sum(), "third_conv": g.third_down_converted.sum(),
            "proe": g.pass_oe.sum(),
        }).reset_index().rename(columns={"posteam": "team"})

        # drives: points per drive and red zone TD rate
        d = p[p.posteam.notna() & p.fixed_drive.notna()].groupby(["game_id", "fixed_drive", "posteam"]).agg(
            res=("fixed_drive_result", "last"), s0=("posteam_score", "min"), s1=("posteam_score_post", "max"),
            yl=("yardline_100", "min")).reset_index()
        d["pts"] = (d.s1 - d.s0).clip(0, 8).fillna(0)
        d["rz"] = (d.yl <= 20).astype(float)
        d["rz_td"] = ((d.yl <= 20) & (d.res == "Touchdown")).astype(float)
        dd = d.groupby(["game_id", "posteam"]).agg(drives=("pts", "size"), drive_pts=("pts", "sum"),
                                                   rz_trips=("rz", "sum"), rz_tds=("rz_td", "sum")).reset_index()
        a = a.merge(dd.rename(columns={"posteam": "team"}), on=["game_id", "team"], how="left")

        # special teams EPA (kicking/punting/returns, credited to the possession team)
        st = p[(p.special_teams_play == 1) & p.epa.notna() & p.posteam.notna()]
        stt = st.groupby(["game_id", "posteam"]).epa.sum().rename("st_epa").reset_index().rename(columns={"posteam": "team"})
        a = a.merge(stt, on=["game_id", "team"], how="left")
        # penalties committed
        pen = p[(p.penalty == 1) & p.penalty_team.notna()].groupby(["game_id", "penalty_team"]).penalty_yards.sum()
        a = a.merge(pen.rename("pen_yds").reset_index().rename(columns={"penalty_team": "team"}),
                    on=["game_id", "team"], how="left")
        off_rows.append(a.fillna({"st_epa": 0, "pen_yds": 0, "drives": 0, "drive_pts": 0, "rz_trips": 0, "rz_tds": 0}))

        # quarterbacks: EPA per dropback by passer
        q = scr[scr.passer_player_id.notna() & (scr.dropback == 1)]
        qa = q.groupby(["game_id", "posteam", "passer_player_id"]).agg(
            qb_db=("epa", "size"), qb_epa=("epa", "sum"), qb_name=("passer_player_name", "last")).reset_index()
        qb_rows.append(qa.rename(columns={"posteam": "team"}))
        print(f"  {f.name}: {len(a)} team-games")
    tg = pd.concat(off_rows, ignore_index=True)
    tg["team"] = tg.team.replace(FRANCHISE)
    qb = pd.concat(qb_rows, ignore_index=True)
    qb["team"] = qb.team.replace(FRANCHISE)
    return tg, qb


# ------------------------------------------------------------------ 2. games table
def load_games() -> pd.DataFrame:
    g = pd.read_csv(RAW / "games.csv")
    for c in ("home_team", "away_team"):
        g[c] = g[c].replace(FRANCHISE)
    g["gameday"] = pd.to_datetime(g.gameday)
    g = g.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    g["is_final"] = g.home_score.notna()
    g["neutral"] = (g.location == "Neutral").astype(int)
    g["playoff"] = (g.game_type != "REG").astype(int)
    g["dome"] = g.roof.isin(["dome", "closed"]).astype(int)
    g["grass"] = g.surface.fillna("").str.contains("grass").astype(int)
    g["temp_f"] = np.where(g.dome == 1, 70, g.temp.fillna(60))
    g["wind_mph"] = np.where(g.dome == 1, 0, g.wind.fillna(8))
    return g


# ------------------------------------------------------------------ 3. rolling (pre-game) team ratings
RATE_SPECS = {  # name: (numerator col, denominator col)
    "epa_pp": ("epa", "plays"), "pass_epa_pp": ("pass_epa", "pass_plays"), "rush_epa_pp": ("rush_epa", "rush_plays"),
    "sr": ("success", "plays"), "expl": ("explosive", "plays"), "sack_rate": ("sacks", "dropbacks"),
    "give_rate": ("giveaways", "plays"), "early_epa": ("early_epa", "early_plays"),
    "third_rate": ("third_conv", "third_att"), "ppd": ("drive_pts", "drives"), "rz_td": ("rz_tds", "rz_trips"),
    "proe": ("proe", "dropbacks"), "pass_rate": ("pass_plays", "plays"),
    "plays_pg": ("plays", "one"), "st_pg": ("st_epa", "one"), "pen_pg": ("pen_yds", "one"),
    "pts_pg": ("points", "one"),
}


def rolling_ratings(games: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """For every team-game, the team's offensive and defensive rates entering that game."""
    gm = games[["game_id", "season", "gameday", "home_team", "away_team", "home_score", "away_score"]]
    tg = tg.merge(gm, on="game_id", how="inner")
    tg["opp"] = np.where(tg.team == tg.home_team, tg.away_team, tg.home_team)
    tg["points"] = np.where(tg.team == tg.home_team, tg.home_score, tg.away_score)
    tg["one"] = 1.0
    # defensive view: what the opponent did against this team
    cols = ["game_id"] + [c for c in tg.columns if c not in ("game_id", "season", "gameday", "home_team", "away_team",
                                                             "home_score", "away_score", "opp", "team")]
    allowed = tg[["game_id", "opp", "season", "gameday"] + [c for c in cols[1:] if c not in ("season", "gameday")]]
    league = {k: tg[n].sum() / tg[d].sum() for k, (n, d) in RATE_SPECS.items()}

    pw = {k: PRIOR_WEIGHT * (1.0 if d == "one" else tg[d].mean()) for k, (n, d) in RATE_SPECS.items()}
    out = []
    for side, frame, key in (("off", tg, "team"), ("def", allowed, "opp")):
        frame = frame.merge(gm[["game_id", "season", "gameday"]], on="game_id", how="left") if "season" not in frame else frame
        frame = frame.sort_values(["gameday", "game_id"])
        for team, d in frame.groupby(key, sort=False):
            st = {k: [0.0, 0.0] for k in RATE_SPECS}
            last_season, n_games = None, 0.0
            recs = []
            for r in d.itertuples(index=False):
                if last_season is not None and r.season != last_season:
                    for k in st:
                        st[k][0] *= SEASON_CARRY; st[k][1] *= SEASON_CARRY
                    n_games *= SEASON_CARRY
                rec = {"game_id": r.game_id, "team": team}
                for k, (num, den) in RATE_SPECS.items():
                    rec[f"{side}_{k}"] = (st[k][0] + league[k] * pw[k]) / (st[k][1] + pw[k])
                rec[f"{side}_games"] = n_games
                recs.append(rec)
                for k, (num, den) in RATE_SPECS.items():
                    nv, dv = getattr(r, num), getattr(r, den)
                    if pd.notna(nv) and pd.notna(dv):
                        st[k][0] = st[k][0] * DECAY + nv
                        st[k][1] = st[k][1] * DECAY + dv
                n_games = n_games * DECAY + 1
                last_season = r.season
            # state AFTER the team's latest game -> used for its next (unplayed) game
            nxt = {"game_id": "NEXT", "team": team, f"{side}_games": n_games}
            for k in RATE_SPECS:
                nxt[f"{side}_{k}"] = (st[k][0] + league[k] * pw[k]) / (st[k][1] + pw[k])
            recs.append(nxt)
            out.append(pd.DataFrame(recs))
    off = pd.concat([o for o in out if "off_epa_pp" in o], ignore_index=True)
    de = pd.concat([o for o in out if "def_epa_pp" in o], ignore_index=True)
    return off.merge(de, on=["game_id", "team"], how="outer")


# ------------------------------------------------------------------ 4. Elo (MOV-adjusted)
def elo(games: pd.DataFrame, k=20, hfa=55, revert=1 / 3) -> pd.DataFrame:
    r, seen, rows = {}, {}, []
    for g in games.itertuples():
        for t in (g.home_team, g.away_team):
            if t not in r:
                r[t], seen[t] = 1505.0, g.season
            elif seen[t] != g.season:
                r[t] = r[t] * (1 - revert) + 1505 * revert; seen[t] = g.season
        rows.append((g.game_id, r[g.home_team], r[g.away_team]))
        if not g.is_final:
            continue
        diff = r[g.home_team] - r[g.away_team] + (0 if g.neutral else hfa)
        exp = 1 / (1 + 10 ** (-diff / 400))
        m = g.home_score - g.away_score
        act = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        wd = diff if m > 0 else -diff
        mult = math.log(abs(m) + 1) * 2.2 / (wd * 0.001 + 2.2) if m else 1.0
        sh = k * mult * (act - exp)
        r[g.home_team] += sh; r[g.away_team] -= sh
    return pd.DataFrame(rows, columns=["game_id", "elo_home", "elo_away"])


# ------------------------------------------------------------------ 5. quarterback form
def qb_form(games: pd.DataFrame, qb: pd.DataFrame) -> pd.DataFrame:
    """Starter's EWMA EPA/dropback and experience entering each game. Starter = most dropbacks;
    for unplayed games, the team's most recent starter (or listed QB if nflverse provides one)."""
    qb = qb.merge(games[["game_id", "gameday", "season"]], on="game_id")
    starters = qb.sort_values("qb_db").groupby(["game_id", "team"]).tail(1)[["game_id", "team", "passer_player_id"]]
    hist = qb.sort_values(["gameday", "game_id"])
    league = hist.qb_epa.sum() / hist.qb_db.sum()
    state, career, last_season = {}, {}, {}
    pre = {}
    for gid, d in hist.groupby("game_id", sort=False):
        for r in d.itertuples():
            pre[(gid, r.passer_player_id)] = state.get(r.passer_player_id)
        for r in d.itertuples():
            pid = r.passer_player_id
            n, s = state.get(pid, (0.0, 0.0))
            if last_season.get(pid) not in (None, r.season):
                n, s = n * 0.6, s * 0.6
            state[pid] = (n * 0.93 + r.qb_db, s * 0.93 + r.qb_epa)
            career[pid] = career.get(pid, 0) + r.qb_db
            last_season[pid] = r.season
    # starter for each team-game
    rows = []
    last_starter = {}
    st_map = {(r.game_id, r.team): r.passer_player_id for r in starters.itertuples()}
    car_pre = {}
    cum = {}
    for gid, d in hist.groupby("game_id", sort=False):
        for r in d.itertuples():
            car_pre[(gid, r.passer_player_id)] = cum.get(r.passer_player_id, 0)
        for r in d.itertuples():
            cum[r.passer_player_id] = cum.get(r.passer_player_id, 0) + r.qb_db
    for g in games.itertuples():
        for side, team, listed in (("home", g.home_team, g.home_qb_id), ("away", g.away_team, g.away_qb_id)):
            pid = st_map.get((g.game_id, team))
            if pid is None:
                pid = listed if isinstance(listed, str) else last_starter.get(team)
            usual = last_starter.get(team)
            if g.is_final and (g.game_id, pid) in pre:
                stt = pre[(g.game_id, pid)]
                exp = car_pre.get((g.game_id, pid), 0)
            else:
                stt = state.get(pid)
                exp = cum.get(pid, 0)
            n, s = stt if stt else (0.0, 0.0)
            rating = (s + league * 150) / (n + 150)   # shrink toward league with 150 dropbacks of prior
            rows.append({"game_id": g.game_id, "team": team, "qb_epa": rating,
                         "qb_exp": math.log1p(exp), "qb_new": int(usual is not None and pid != usual)})
            if g.is_final and pid:
                last_starter[team] = pid
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 6. injuries (2012+ when snaps exist)
def injuries(games: pd.DataFrame) -> pd.DataFrame:
    inj = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("inj_*.parquet"))], ignore_index=True)
    snaps = pd.concat([pd.read_parquet(f) for f in sorted(RAW.glob("snap_*.parquet"))], ignore_index=True)
    inj = inj[inj.season.notna() & inj.week.notna()]
    for df in (inj, snaps):
        df["team"] = df.team.replace(FRANCHISE)
        df["season"] = df.season.astype("int64")
        df["week"] = df.week.astype("int64")
    norm = lambda s: s.str.lower().str.replace(r"[^a-z]", "", regex=True).str.replace(r"(jr|sr|ii|iii|iv)$", "", regex=True)
    snaps["key"] = norm(snaps.player)
    inj["key"] = norm(inj.full_name)
    snaps = snaps.sort_values(["season", "week"])
    # each player's average snap share over his team's previous 4 games
    snaps["share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps["unit"] = np.where(snaps.offense_pct >= snaps.defense_pct, "off", "def")
    snaps["is_qb"] = (snaps.position == "QB").astype(int)
    snaps["share_prev"] = snaps.groupby(["team", "key"]).share.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    last_share = snaps.groupby(["season", "team", "key"]).agg(share=("share", "mean"), unit=("unit", "last"),
                                                               is_qb=("is_qb", "max")).reset_index()
    out = inj[inj.report_status.isin(["Out", "Doubtful"])].copy()
    # share from the same season (prior weeks), fallback: last season
    s1 = snaps[["season", "week", "team", "key", "share_prev", "unit", "is_qb"]]
    out = out.merge(s1, on=["season", "week", "team", "key"], how="left")
    fb = last_share.assign(season=last_share.season + 1).rename(columns={"share": "share_ly", "unit": "unit_ly", "is_qb": "qb_ly"})
    out = out.merge(fb, on=["season", "team", "key"], how="left")
    # if the player didn't play this week, there's no snap row; use his season-to-date share
    # most recent snap share from weeks BEFORE this one (player may have missed the current week)
    snaps["share_roll"] = snaps.groupby(["season", "team", "key"]).share.transform(lambda s: s.rolling(4, min_periods=1).mean())
    prior = snaps[["season", "week", "team", "key", "share_roll"]].copy()
    prior["week"] = prior.week.astype("int64")
    out["week"] = out.week.astype("int64")
    out = out.sort_values("week")
    out = pd.merge_asof(out, prior.rename(columns={"week": "pw"}).sort_values("pw"), left_on="week", right_on="pw",
                        by=["season", "team", "key"], allow_exact_matches=False, direction="backward")
    out["w"] = out.share_prev.fillna(out.share_roll).fillna(out.share_ly).fillna(0)
    default_unit = pd.Series(np.where(out.position.isin(["QB", "RB", "WR", "TE", "T", "G", "C", "OL", "FB"]), "off", "def"), index=out.index)
    out["u"] = out.unit.fillna(out.unit_ly).fillna(default_unit)
    out["qb_out"] = ((out.position == "QB") & (out.w >= 0.5)).astype(int)
    agg = out.groupby(["season", "week", "team"]).apply(lambda d: pd.Series({
        "inj_off": d.loc[d.u == "off", "w"].clip(0, 1).sum(),
        "inj_def": d.loc[d.u == "def", "w"].clip(0, 1).sum(),
        "inj_qb": d.qb_out.max()}), include_groups=False).reset_index()
    gm = pd.concat([games[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"}),
                    games[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"})])
    gm = gm.merge(agg, on=["season", "week", "team"], how="left")
    gm.loc[gm.season < 2012, ["inj_off", "inj_def", "inj_qb"]] = np.nan
    return gm[["game_id", "team", "inj_off", "inj_def", "inj_qb"]]


# ------------------------------------------------------------------ 7. assemble
def build() -> None:
    print("1/6 play-by-play aggregation")
    tg, qb = team_game_stats()
    games = load_games()
    print("2/6 rolling team ratings")
    rates = rolling_ratings(games, tg)
    # ratings entering games that haven't been played yet: take each team's state after its last game
    print("3/6 Elo, QB form, injuries")
    el = elo(games)
    qf = qb_form(games, qb)
    inj = injuries(games)
    print("4/6 assembling team-perspective rows")
    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        g = games.copy()
        g["team"], g["opp"] = g[f"{side}_team"], g[f"{opp_side}_team"]
        g["is_home"] = np.where(g.neutral == 1, 0, 1 if side == "home" else -1)
        g["points"], g["opp_points"] = g[f"{side}_score"], g[f"{opp_side}_score"]
        g["rest"], g["opp_rest"] = g[f"{side}_rest"], g[f"{opp_side}_rest"]
        g["side"] = side
        rows.append(g)
    tgm = pd.concat(rows, ignore_index=True)
    tgm = tgm.merge(el, on="game_id", how="left")
    tgm["elo"] = np.where(tgm.side == "home", tgm.elo_home, tgm.elo_away)
    tgm["opp_elo"] = np.where(tgm.side == "home", tgm.elo_away, tgm.elo_home)
    tgm = tgm.drop(columns=["elo_home", "elo_away"])
    # carry each team's latest rating into unplayed games
    nxt = rates[rates.game_id == "NEXT"].drop(columns="game_id").set_index("team")
    rates = rates[rates.game_id != "NEXT"].set_index(["game_id", "team"])
    rate_cols = list(rates.columns)
    tgm["_key"] = np.where(tgm.is_final, tgm.game_id, "NEXT")
    played = tgm[tgm.is_final].join(rates, on=["game_id", "team"])
    played = played.join(rates.add_prefix("opp_"), on=["game_id", "opp"])
    future = tgm[~tgm.is_final].join(nxt, on="team").join(nxt.add_prefix("opp_"), on="opp")
    tgm = pd.concat([played, future], ignore_index=True).drop(columns="_key").sort_values(["gameday", "game_id"])
    tgm = tgm.merge(qf, on=["game_id", "team"], how="left")
    tgm = tgm.merge(qf.rename(columns={"team": "opp", "qb_epa": "opp_qb_epa", "qb_exp": "opp_qb_exp", "qb_new": "opp_qb_new"}),
                    on=["game_id", "opp"], how="left")
    tgm = tgm.merge(inj, on=["game_id", "team"], how="left")
    tgm = tgm.merge(inj.rename(columns={"team": "opp", "inj_off": "opp_inj_off", "inj_def": "opp_inj_def", "inj_qb": "opp_inj_qb"}),
                    on=["game_id", "opp"], how="left")
    for c in ("inj_off", "inj_def", "inj_qb", "opp_inj_off", "opp_inj_def", "opp_inj_qb"):
        tgm.loc[(tgm.season >= 2012) & tgm[c].isna(), c] = 0.0
    print("5/6 writing")
    tgm.to_parquet(OUT / "team_games.parquet", index=False)
    games.to_parquet(OUT / "games.parquet", index=False)
    print(f"6/6 done: {len(tgm):,} team-game rows, {tgm.shape[1]} columns, seasons {games.season.min()}-{games.season.max()}")


if __name__ == "__main__":
    build()
