/* Hatch's Picks — single-page app. Reads data/*.json written by the daily pipeline. */
(() => {
'use strict';
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const pct = (x, d = 0) => x == null ? '—' : (100 * x).toFixed(d) + '%';
const sgn = (x, d = 1) => x == null ? '—' : (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
const odds = o => o == null ? '—' : (o > 0 ? '+' + o : '−' + Math.abs(o));
const money = x => (x < 0 ? '−$' : '+$') + Math.abs(x).toFixed(Math.abs(x) < 10 ? 2 : 0);
const payout = o => o > 0 ? o / 100 : 100 / -o;
const THR = {Spread: 3, Total: 3, Moneyline: 5};
const isPlay = b => b.edge >= THR[b.market] && b.ev > 0;
const store = {get(k, d) { try { const v = localStorage.getItem('hp-' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
               set(k, v) { try { localStorage.setItem('hp-' + k, JSON.stringify(v)); } catch (e) {} }};

let ML, RK, MU, HI, TEAM = {};
let stake = store.get('stake', 100), onlyPlays = false, shopPlays = true;
const charts = [];
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

// ---------------------------------------------------------------- theme
const savedTheme = store.get('theme', null);
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
$('#theme').onclick = () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  store.set('theme', document.documentElement.dataset.theme);
  route();
};

// ---------------------------------------------------------------- helpers
const team = t => TEAM[t] || {team: t, name: t, color: '#888', nick: t};
const tb = (t, s = 34) => `<span class="tb" style="--s:${s}px;background:${team(t).color}" aria-hidden="true">${t}</span>`;
const tlink = (t, full = false) => `<a class="tlink" href="#/team/${t}">${tb(t, 26)}<span class="tn">${full ? esc(team(t).name) : t}</span></a>`;
const fmtDay = d => new Date(d + 'T12:00:00').toLocaleDateString('en-US', {weekday: 'short', month: 'short', day: 'numeric'});
const fmtTime = t => { if (!t) return ''; const [h, m] = t.split(':').map(Number); return `${(h + 11) % 12 + 1}:${String(m).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'} ET`; };
const vegasScore = g => g.spread_line == null || g.total_line == null ? null : [(g.total_line - g.spread_line) / 2, (g.total_line + g.spread_line) / 2];
const rankOf = (t, k) => { const r = RK.teams.find(x => x.team === t); return r ? r['rk_' + k] : null; };
function killCharts() { while (charts.length) charts.pop().destroy(); }
function chartDefaults() {
  if (!window.Chart) return false;
  Chart.defaults.font.family = css('--mono').split(',')[0].replace(/"/g, '') + ', monospace';
  Chart.defaults.font.size = 11;
  Chart.defaults.color = css('--muted');
  Chart.defaults.borderColor = css('--grid');
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.tooltip.backgroundColor = css('--navy-2');
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.animation = false;
  return true;
}
function setNav(r) { $$('.nav a').forEach(a => a.getAttribute('href') === '#/' + r ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')); }
const foot = () => `<footer>Hatch's Picks · data: <a href="https://github.com/nflverse" target="_blank" rel="noopener">nflverse</a> ${ML.first_season}–${ML.season} and The Odds API · model: walk-forward ridge + LightGBM ensemble · updated ${esc(ML.generated_at)} · for research and entertainment, not betting advice.</footer>`;

// ---------------------------------------------------------------- hero (home)
function hero() {
  const M = ML.metrics, BT = ML.backtest;
  const a3 = BT.find(r => r.market === 'Spread' && r.edge === '≥3 pts'), t3 = BT.find(r => r.market === 'Total' && r.edge === '≥3 pts');
  return `<section class="hero"><div class="wrap">
    <div><div class="eyebrow" style="color:var(--accent)">Machine-learning NFL model · ${ML.season} season</div>
      <h1>Hatch's <span>Picks</span></h1>
      <p>Projects every game's score and win probability from ${ML.train_games.toLocaleString()} games of play-by-play since ${ML.first_season}, then prices each bet against the Vegas line and every major sportsbook.</p>
      <div class="stamp">Updated ${esc(ML.generated_at)} · results through ${esc(ML.data_through)}</div></div>
    <div class="strip">
      <div><b>${M.margin_mae.model.toFixed(2)} <i>/ ${M.margin_mae.vegas.toFixed(2)}</i></b><small>avg margin miss, model / Vegas</small></div>
      <div><b>${pct(M.winner_acc.model, 1)}</b><small>picks the winner</small></div>
      <div><b>${pct(a3.win_pct, 1)}</b><small>ATS on 3+ pt edges</small></div>
      <div><b>${pct(t3.win_pct, 1)}</b><small>totals on 3+ pt edges</small></div>
      <div><b>${M.games.toLocaleString()}</b><small>games tested out of sample</small></div>
    </div></div></section>`;
}

// ---------------------------------------------------------------- this week
function betRow(b) {
  const ev = stake * b.ev / 100;
  const tag = b.result ? `<span class="pill ${b.result.toLowerCase()}">${{W: 'Won', L: 'Lost', P: 'Push'}[b.result]}</span>`
    : isPlay(b) ? '<span class="pill play">Play</span>' : '<span class="pill lean">Lean</span>';
  const edge = b.unit === '%' ? b.edge.toFixed(1) + '%' : b.edge.toFixed(1) + ' pts';
  return `<div class="bet"><span class="mkt">${b.market}</span><span class="pk">${esc(b.pick)}<span>${odds(b.odds)} · ${edge} edge · ${pct(b.prob)} to win</span></span>
    <span class="ev ${ev >= 0 ? 'pos' : 'neg'}">${money(ev)}</span>${tag}</div>`;
}
function gameCard(g) {
  const A = team(g.away), H = team(g.home), ph = g.p_home ?? .5, mk = g.mkt_home, v = vegasScore(g);
  const bets = [...g.bets].sort((a, b) => b.ev - a.ev), hot = !g.final && bets.some(isPlay);
  return `<a class="g${hot ? ' hot' : ''}" href="#/game/${g.id}">
    <div class="g-band"><i style="background:${A.color}"></i><i style="background:${H.color}"></i></div>
    <div class="g-top"><span>${fmtDay(g.day)}${g.time ? ' · ' + fmtTime(g.time) : ''}</span><span>${hot ? '<span class="tag">Play on board</span>' : esc(g.stadium || '')}</span></div>
    <div class="score">
      <div class="tm"><div class="ab">${tb(g.away)}${g.away}</div><div class="qb">${esc(g.away_qb || '')}</div></div>
      <div class="pts">${Math.round(g.pred_away)}–${Math.round(g.pred_home)}<small>model projection</small>${v ? `<span class="vg">Vegas ${Math.round(v[0])}–${Math.round(v[1])}</span>` : ''}${g.final ? `<span class="fin">Final ${g.away_score}–${g.home_score}</span>` : ''}</div>
      <div class="tm r"><div class="ab">${g.home}${tb(g.home)}</div><div class="qb">${esc(g.home_qb || '')}</div></div>
    </div>
    <div class="wp"><div class="wpbar"><span style="width:${100 * (1 - ph)}%;background:${A.color}"></span><span style="width:${100 * ph}%;background:${H.color}"></span>${mk != null ? `<i class="mk" style="left:calc(${100 * (1 - mk)}% - 1px)"></i>` : ''}</div>
      <div class="wpl"><span><b>${pct(1 - ph)}</b> model</span><span>${mk != null ? `Vegas ${pct(1 - mk)} / ${pct(mk)}` : ''}</span><span>model <b>${pct(ph)}</b></span></div></div>
    <div class="lines">
      <div><small>${g.home} spread</small><b>${sgn(-g.margin)}</b><em>Vegas ${g.spread_line == null ? '—' : g.spread_line === 0 ? 'PK' : sgn(-g.spread_line)}</em></div>
      <div><small>Total</small><b>${g.total.toFixed(1)}</b><em>Vegas ${g.total_line ?? '—'}</em></div>
      <div><small>Moneyline</small><b>${odds(g.away_ml)} / ${odds(g.home_ml)}</b><em>${g.away} / ${g.home}</em></div>
    </div>
    <div class="bets">${bets.length ? bets.map(betRow).join('') : '<div class="empty">No line posted yet</div>'}</div></a>`;
}
function whoWins(gs) {
  return `<div class="ww-head"><span>Away · win chance</span><span>Model bar · Vegas tick</span><span>Home · win chance</span><span>Projected score</span></div>` +
    [...gs].sort((a, b) => Math.abs(b.p_home - .5) - Math.abs(a.p_home - .5)).map(g => {
      const ph = g.p_home ?? .5, mk = g.mkt_home, A = team(g.away), H = team(g.home), v = vegasScore(g);
      const fav = ph >= .5 ? g.home : g.away, diff = mk == null ? null : (ph - mk) * 100;
      const note = mk == null ? '' : Math.abs(diff) < 3 ? 'model agrees with Vegas' : `model ${Math.abs(diff).toFixed(0)}% higher on ${diff > 0 ? g.home : g.away} than Vegas`;
      const p = (m, k) => `<span class="ww-p"><span class="m"><i class="k">Model</i>${pct(m)}</span><span><i class="k">Vegas</i>${k == null ? '—' : pct(k)}</span></span>`;
      const sc = g.final ? `<b>Final ${g.away_score}–${g.home_score}</b><span><i class="k">Model</i>${(g.home_score > g.away_score) === (ph >= .5) ? 'called it' : 'missed'}</span>`
        : `<b><i class="k">Model</i>${Math.round(g.pred_away)}–${Math.round(g.pred_home)}</b><span><i class="k">Vegas</i>${v ? Math.round(v[0]) + '–' + Math.round(v[1]) : '—'}</span>`;
      return `<a class="ww-row" href="#/game/${g.id}">
        <div class="ww-t ${fav === g.away ? 'fav' : ''}">${tb(g.away, 30)}<b>${g.away}</b>${p(1 - ph, mk == null ? null : 1 - mk)}</div>
        <div class="ww-bar"><div class="wpbar"><span style="width:${100 * (1 - ph)}%;background:${A.color}"></span><span style="width:${100 * ph}%;background:${H.color}"></span>${mk != null ? `<i class="mk" style="left:calc(${100 * (1 - mk)}% - 1px)"></i>` : ''}</div><small>${note || '&nbsp;'}</small></div>
        <div class="ww-t r ${fav === g.home ? 'fav' : ''}">${p(ph, mk)}<b>${g.home}</b>${tb(g.home, 30)}</div>
        <div class="ww-sc">${sc}</div></a>`;
    }).join('');
}
function bestBets(gs) {
  const shop = ML.shop?.rows || [];
  const all = gs.filter(g => !g.final).flatMap(g => g.bets.filter(isPlay).map(b => {
    const alt = shop.filter(r => r.game === `${g.away}@${g.home}` && r.market === b.market).sort((x, y) => y.ev - x.ev)[0];
    const best = alt && alt.ev > b.ev ? {...b, pick: alt.pick, odds: alt.price, ev: alt.ev, prob: alt.prob, book: alt.book} : b;
    return {g, b: best};
  })).sort((x, y) => y.b.ev - x.b.ev).slice(0, 6);
  if (!all.length) return '';
  return `<div><div class="head" style="margin-bottom:12px"><div><div class="eyebrow">Top of the board</div><h2>Best bets</h2></div></div>
    <div class="bb">${all.map(({g, b}) => {
      const side = b.market === 'Total' ? null : b.pick.split(' ')[0];
      return `<a href="#/game/${g.id}" style="--c:${side ? team(side).color : css('--accent')}">
        <div class="r1"><span>${g.away} @ ${g.home} · ${b.market}</span><span class="pill play">Play</span></div>
        <div class="pk">${esc(b.pick)}</div>
        <div class="r3"><span>${odds(b.odds)}${b.book ? ' · ' + esc(b.book) : ''}</span><span>${pct(b.prob)} to win</span></div>
        <div class="ev pos">${money(stake * b.ev / 100)} <small class="muted" style="font-size:11px">EV at $${stake}</small></div></a>`;
    }).join('')}</div></div>`;
}
function pageWeek(w) {
  const weeks = [...new Set(ML.games.map(g => g.week))].sort((a, b) => a - b);
  w = w || ML.next_week || weeks[weeks.length - 1];
  const gs = ML.games.filter(g => g.week === w), final = gs.length && gs.every(g => g.final);
  const plays = gs.flatMap(g => g.bets.filter(isPlay));
  const evSum = plays.filter(b => !b.result).reduce((s, b) => s + stake * b.ev / 100, 0);
  const graded = plays.filter(b => b.result), won = graded.filter(b => b.result === 'W').length, lost = graded.filter(b => b.result === 'L').length;
  const pnl = graded.reduce((s, b) => s + (b.result === 'W' ? stake * payout(b.odds) : b.result === 'L' ? -stake : 0), 0);
  const shown = (onlyPlays ? gs.filter(g => g.bets.some(isPlay)) : gs)
    .sort((a, b) => Math.max(0, ...b.bets.filter(isPlay).map(x => x.ev)) - Math.max(0, ...a.bets.filter(isPlay).map(x => x.ev)) || a.day.localeCompare(b.day));
  $('#hero').innerHTML = hero();
  $('#app').innerHTML = `
    <div class="head"><div><div class="eyebrow">${ML.season} season · Week ${w}${final ? ' · final' : w === ML.next_week ? ' · upcoming' : ''}</div><h2>${final ? `Week ${w} results` : `Week ${w} picks`}</h2>
      <p><b>Play</b> = the model's edge cleared the backtest bar (3+ points on a spread or total, 5%+ on a moneyline) and the bet has positive expected value. Click any game for the full breakdown.</p></div>
      <div class="controls">
        <label class="f">Week<select id="wk">${weeks.map(x => `<option value="${x}" ${x === w ? 'selected' : ''}>Week ${x}${x === ML.next_week ? ' · next' : ''}</option>`).join('')}</select></label>
        <label class="f">Stake per bet ($)<input id="stake" type="number" min="1" step="10" value="${stake}"></label>
        <div class="seg" role="group" aria-label="Filter"><button id="fa" aria-pressed="${!onlyPlays}">All games</button><button id="fp" aria-pressed="${onlyPlays}">Plays only</button></div>
      </div></div>
    <div class="kpis">
      <div><b class="num">${gs.length}</b><small>games</small></div>
      <div><b class="num">${plays.length}</b><small>plays flagged</small></div>
      ${graded.length ? `<div><b class="num">${won}–${lost}</b><small>plays graded</small></div><div><b class="num ${pnl >= 0 ? 'pos' : 'neg'}">${money(pnl)}</b><small>result at $${stake}/play</small></div>`
        : `<div><b class="num ${evSum >= 0 ? 'pos' : 'neg'}">${money(evSum)}</b><small>expected profit, $${stake}/play</small></div><div><b class="num">$${(stake * plays.length).toLocaleString()}</b><small>total risked</small></div>`}
    </div>
    ${final ? '' : bestBets(gs)}
    <div class="card"><h3>Who wins</h3><p class="sub"><b>Model</b> is our projection. <b>Vegas</b> is the sportsbooks' win chance with the vig removed, and the score their spread and total imply. Colored bar = model, black tick = Vegas.</p><div style="padding-top:8px">${whoWins(gs)}</div></div>
    <div class="games">${shown.length ? shown.map(gameCard).join('') : '<div class="card empty">No plays cleared the bar this week.</div>'}</div>
    ${foot()}`;
  $('#wk').onchange = e => { location.hash = '#/week/' + e.target.value; };
  $('#stake').oninput = e => { stake = Math.max(1, +e.target.value || 100); store.set('stake', stake); clearTimeout(window._st); window._st = setTimeout(() => { const y = scrollY; pageWeek(w); scrollTo(0, y); $('#stake').focus(); }, 350); };
  $('#fa').onclick = () => { onlyPlays = false; pageWeek(w); };
  $('#fp').onclick = () => { onlyPlays = true; pageWeek(w); };
}

// ---------------------------------------------------------------- game page
const CMP = [
  ['Elo rating', 'elo', 0, 'hi', 'elo'],
  ['Offense EPA / play', 'off_epa_pp', 3, 'hi', 'off_epa_pp'],
  ['Defense EPA / play allowed', 'def_epa_pp', 3, 'lo', 'def_epa_pp'],
  ['Pass offense EPA', 'off_pass_epa_pp', 3, 'hi'],
  ['Rush offense EPA', 'off_rush_epa_pp', 3, 'hi'],
  ['Pass defense EPA', 'def_pass_epa_pp', 3, 'lo'],
  ['Rush defense EPA', 'def_rush_epa_pp', 3, 'lo'],
  ['Success rate', 'off_sr', 1, 'hi', null, true],
  ['Points per drive', 'off_ppd', 2, 'hi'],
  ['Points allowed per drive', 'def_ppd', 2, 'lo'],
  ['Explosive play rate', 'off_expl', 1, 'hi', null, true],
  ['Giveaway rate', 'off_give_rate', 1, 'lo', null, true],
  ['Takeaway rate', 'def_give_rate', 1, 'hi', null, true],
  ['QB EPA / dropback', 'qb_epa', 3, 'hi', 'qb_epa'],
  ['Snaps lost to injury (off)', 'inj_off', 1, 'lo', null, true],
  ['Snaps lost to injury (def)', 'inj_def', 1, 'lo', null, true],
  ['Days of rest', 'rest', 0, 'hi'],
];
function pageGame(id) {
  const g = ML.games.find(x => x.id === id);
  if (!g) return pageWeek();
  const A = team(g.away), H = team(g.home), ph = g.p_home ?? .5, mk = g.mkt_home, v = vegasScore(g), m = MU[id] || {};
  const shop = (ML.shop?.rows || []).filter(r => r.game === `${g.away}@${g.home}`);
  $('#hero').innerHTML = `<section class="gh"><div class="gh-bg"><i style="background:${A.color}"></i><i style="background:${H.color}"></i></div><div class="wrap">
    <a class="back" href="#/week/${g.week}">← Week ${g.week} · ${fmtDay(g.day)}${g.time ? ' · ' + fmtTime(g.time) : ''} · ${esc(g.stadium || '')}</a>
    <div class="gh-main">
      <div class="gh-team">${tb(g.away, 56)}<div class="nm">${esc(A.name)}</div><div class="sub">${esc(g.away_qb || '')} · ${RK.teams.find(t => t.team === g.away)?.w ?? 0}–${RK.teams.find(t => t.team === g.away)?.l ?? 0}</div>
        <div class="wp">${pct(1 - ph)}<small>model · Vegas ${mk == null ? '—' : pct(1 - mk)}</small></div></div>
      <div class="gh-mid"><small>${g.final ? 'Final' : 'Model projection'}</small><div class="sc">${g.final ? `${g.away_score}–${g.home_score}` : `${Math.round(g.pred_away)}–${Math.round(g.pred_home)}`}</div>
        ${g.final ? `<div class="vg">Model had ${Math.round(g.pred_away)}–${Math.round(g.pred_home)}</div>` : v ? `<div class="vg">Vegas implies ${Math.round(v[0])}–${Math.round(v[1])}</div>` : ''}</div>
      <div class="gh-team r">${tb(g.home, 56)}<div class="nm">${esc(H.name)}</div><div class="sub">${esc(g.home_qb || '')} · ${RK.teams.find(t => t.team === g.home)?.w ?? 0}–${RK.teams.find(t => t.team === g.home)?.l ?? 0}</div>
        <div class="wp">${pct(ph)}<small>model · Vegas ${mk == null ? '—' : pct(mk)}</small></div></div>
    </div></div></section>`;
  const rows = CMP.map(([lbl, k, d, dir, rk, isPct]) => {
    const a = m.away?.[k], h = m.home?.[k];
    if (a == null && h == null) return '';
    const f = x => x == null ? '—' : isPct ? (100 * x).toFixed(d) + '%' : (k.includes('epa') ? sgn(x, d) : x.toFixed(d));
    const vals = RK.teams.map(t => t[k]).filter(x => x != null), lo = Math.min(...vals), hi = Math.max(...vals);
    const w = x => x == null || hi === lo ? 0 : 8 + 92 * (dir === 'hi' ? (x - lo) / (hi - lo) : (hi - x) / (hi - lo));
    const better = a == null || h == null ? null : dir === 'hi' ? (a > h ? 'a' : 'h') : (a < h ? 'a' : 'h');
    const r1 = rk ? rankOf(g.away, rk) : null, r2 = rk ? rankOf(g.home, rk) : null;
    return `<div class="cmp-row"><span class="v" style="${better === 'a' ? 'font-weight:600' : ''}">${f(a)}${r1 ? `<small>#${r1}</small>` : ''}</span>
      <div class="cbar l"><i style="width:${w(a)}%;background:${A.color};opacity:${better === 'a' ? 1 : .45}"></i></div><span class="lbl">${lbl}</span>
      <div class="cbar r"><i style="width:${w(h)}%;background:${H.color};opacity:${better === 'h' ? 1 : .45}"></i></div>
      <span class="v r" style="${better === 'h' ? 'font-weight:600' : ''}">${f(h)}${r2 ? `<small>#${r2}</small>` : ''}</span></div>`;
  }).join('');
  const groups = {};
  shop.forEach(r => (groups[r.market] ??= []).push(r));
  const books = ML.shop?.books || [];
  $('#app').innerHTML = `
    <div class="cols">
      <div class="card"><h3>The bets</h3><p class="sub">Each market priced at the model's edge. Expected profit is at your $${stake} stake.</p><div class="bets" style="padding:8px 0 4px">${[...g.bets].sort((a, b) => b.ev - a.ev).map(betRow).join('') || '<div class="empty">No line posted yet.</div>'}</div></div>
      <div class="card"><h3>Model vs Vegas</h3><div class="body tbl"><table>
        <thead><tr><th></th><th class="n">Model</th><th class="n">Vegas</th><th class="n">Gap</th></tr></thead><tbody>
        <tr><td>${g.home} win chance</td><td class="n">${pct(ph, 1)}</td><td class="n">${pct(mk, 1)}</td><td class="n">${mk == null ? '—' : sgn((ph - mk) * 100, 1) + '%'}</td></tr>
        <tr><td>${g.home} spread</td><td class="n">${sgn(-g.margin)}</td><td class="n">${g.spread_line == null ? '—' : sgn(-g.spread_line)}</td><td class="n">${g.spread_line == null ? '—' : sgn(g.margin - g.spread_line) + ' pts'}</td></tr>
        <tr><td>Total points</td><td class="n">${g.total.toFixed(1)}</td><td class="n">${g.total_line ?? '—'}</td><td class="n">${g.total_line == null ? '—' : sgn(g.total - g.total_line) + ' pts'}</td></tr>
        <tr><td>Projected score</td><td class="n">${Math.round(g.pred_away)}–${Math.round(g.pred_home)}</td><td class="n">${v ? Math.round(v[0]) + '–' + Math.round(v[1]) : '—'}</td><td></td></tr>
        </tbody></table></div></div>
    </div>
    ${shop.length ? `<div class="card"><h3>Shop the line</h3><p class="sub">The model's side at every book we track. Green = best price. Odds from ${esc(ML.shop.fetched_at)}.</p><div class="body tbl"><table>
      <thead><tr><th>Bet</th>${books.map(b => `<th>${b}</th>`).join('')}</tr></thead><tbody>${Object.entries(groups).map(([mk, rs]) => {
        const best = Math.max(...rs.map(r => r.ev));
        return `<tr><td><b>${esc(rs[0].side)}</b> <span class="muted">${mk}</span></td>${books.map(b => { const r = rs.find(q => q.book === b);
          return r ? `<td class="${r.ev === best && rs.length > 1 ? 'best' : ''}"><div class="cell"><b>${r.line == null ? 'ML' : mk === 'Total' ? r.line : sgn(r.line, 1).replace('.0', '')} ${odds(r.price)}</b><span class="${r.ev >= 0 ? 'pos' : 'neg'}">${money(stake * r.ev / 100)}</span></div></td>` : '<td>—</td>'; }).join('')}</tr>`;
      }).join('')}</tbody></table></div></div>` : ''}
    <div class="card"><h3>Tale of the tape</h3><p class="sub">What the model sees going into this game: recency-weighted team stats (#rank among 32 teams). Longer bar = better.</p>
      <div class="body"><div class="cmp-h"><span>${g.away}</span><span></span><span>Stat</span><span></span><span>${g.home}</span></div><div class="cmp">${rows || '<div class="empty">No stats yet.</div>'}</div></div></div>
    <div class="card"><h3>Elo rating, last 5 seasons</h3><p class="sub">Team strength over time. 1500 is average; the gap between the lines is the single strongest signal in the model.</p><div class="chart-box tall"><canvas id="elo"></canvas></div></div>
    ${foot()}`;
  drawElo('elo', [g.away, g.home]);
}
function drawElo(id, teams) {
  if (!chartDefaults()) return;
  const ds = teams.map(t => {
    const r = RK.teams.find(x => x.team === t);
    return {label: team(t).name, data: (r?.elo_hist || []).map(([d, e, s, w]) => ({x: d, y: e, s, w})), borderColor: team(t).color,
      backgroundColor: team(t).color, pointRadius: 0, pointHoverRadius: 4, borderWidth: 2.5, tension: .25};
  });
  const all = [...new Set(ds.flatMap(d => d.data.map(p => p.x)))].sort();
  ds.forEach(d => { const m = new Map(d.data.map(p => [p.x, p])); d.data = all.map(x => m.get(x)?.y ?? null); d._meta = all.map(x => m.get(x)); d.spanGaps = true; });
  charts.push(new Chart($('#' + id), {type: 'line', data: {labels: all, datasets: ds},
    options: {interaction: {mode: 'index', intersect: false}, scales: {
      x: {ticks: {callback: (v, i) => { const l = all[i]; return l && (i === 0 || all[i - 1].slice(0, 4) !== l.slice(0, 4)) ? l.slice(0, 4) : ''; }, autoSkip: false, maxRotation: 0}, grid: {display: false}},
      y: {grid: {color: css('--grid')}}},
      plugins: {tooltip: {callbacks: {title: it => { const m = it[0].dataset._meta[it[0].dataIndex]; return m ? `${m.s} week ${m.w}` : it[0].label; }, label: it => `${it.dataset.label}: ${Math.round(it.raw)}`}}}}}));
}

// ---------------------------------------------------------------- rankings
let rkSort = store.get('rksort', ['elo', -1]);
function pageRankings() {
  $('#hero').innerHTML = '';
  const cols = [['elo', 'Elo', 0], ['net_epa', 'Net EPA', 3], ['off_epa_pp', 'Off EPA', 3], ['def_epa_pp', 'Def EPA', 3], ['qb_epa', 'QB EPA', 3], ['pd', 'Pt diff', 0]];
  const lo = k => Math.min(...RK.teams.map(t => t[k] ?? 0)), hi = k => Math.max(...RK.teams.map(t => t[k] ?? 0));
  const bar = (t, k) => { const a = lo(k), b = hi(k), x = t[k] ?? a; let f = (x - a) / (b - a || 1); if (k === 'def_epa_pp') f = 1 - f;
    return `<span class="rkbar"><i style="width:${8 + 92 * f}%;background:${f > .66 ? css('--pos') : f < .33 ? css('--neg') : css('--faint')}"></i></span>`; };
  const spark = t => { const h = t.elo_hist.slice(-34).map(x => x[1]); if (h.length < 2) return ''; const a = 1300, b = 1750;
    const pts = h.map((y, i) => `${(i / (h.length - 1) * 80).toFixed(1)},${(22 - (y - a) / (b - a) * 22).toFixed(1)}`).join(' ');
    return `<svg width="80" height="22" viewBox="0 0 80 22" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="${t.color}" stroke-width="2"/></svg>`; };
  function render() {
    const [k, dir] = rkSort;
    const rows = [...RK.teams].sort((a, b) => dir * ((a[k] ?? -99) - (b[k] ?? -99)) || a.team.localeCompare(b.team));
    $('#rk').innerHTML = `<thead><tr><th>#</th><th>Team</th><th>Record</th>${cols.map(([c, l]) => `<th class="n sort" data-k="${c}" ${c === k ? `aria-sort="${(dir > 0) !== (c === 'def_epa_pp') ? 'ascending' : 'descending'}"` : ''}>${l}</th>`).join('')}<th>QB</th><th>Elo, last 2 seasons</th></tr></thead>
      <tbody>${rows.map((t, i) => `<tr><td class="rk">${i + 1}</td><td>${tlink(t.team, true)}</td><td class="num">${t.w}–${t.l}${t.t ? '–' + t.t : ''}</td>
        ${cols.map(([c, , d]) => `<td class="n num">${c === 'pd' ? sgn(t[c], 0) : c === 'elo' ? Math.round(t[c]) : sgn(t[c], d)}${['elo', 'net_epa', 'off_epa_pp', 'def_epa_pp'].includes(c) ? bar(t, c) : ''}</td>`).join('')}
        <td>${esc(t.qb || '—')}</td><td>${spark(t)}</td></tr>`).join('')}</tbody>`;
    $$('#rk th.sort').forEach(th => th.onclick = () => {
      const c = th.dataset.k, def = c === 'def_epa_pp' ? 1 : -1;
      rkSort = [c, rkSort[0] === c ? -rkSort[1] : def]; store.set('rksort', rkSort); render(); });
  }
  $('#app').innerHTML = `<div class="head"><div><div class="eyebrow">${RK.season} · going into Week ${ML.next_week ?? ''}</div><h2>Power rankings</h2>
    <p>Elo is the model's overall strength rating (1500 = average). EPA is expected points added per play, weighted toward recent games; for defense, lower is better. Click a column to sort, or a team for its page.</p></div></div>
    <div class="card"><div class="tbl"><table id="rk"></table></div></div>
    <div class="card"><h3>Offense vs defense</h3><p class="sub">Up and to the right is good on both sides of the ball. Hover a dot for the team.</p><div class="chart-box tall"><canvas id="sc"></canvas></div></div>
    ${foot()}`;
  render();
  if (!chartDefaults()) return;
  const quad = {id: 'quad', beforeDraw(c) { const {ctx, chartArea: a, scales: {x, y}} = c; ctx.save(); ctx.strokeStyle = css('--faint'); ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x.getPixelForValue(0), a.top); ctx.lineTo(x.getPixelForValue(0), a.bottom); ctx.moveTo(a.left, y.getPixelForValue(0)); ctx.lineTo(a.right, y.getPixelForValue(0)); ctx.stroke(); ctx.restore(); },
    afterDatasetsDraw(c) { const {ctx} = c; ctx.save(); ctx.font = `600 10px ${css('--display')}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      c.getDatasetMeta(0).data.forEach((p, i) => { ctx.fillStyle = '#fff'; ctx.fillText(RK.teams[i].team, p.x, p.y); }); ctx.restore(); }};
  charts.push(new Chart($('#sc'), {type: 'scatter', plugins: [quad], data: {datasets: [{data: RK.teams.map(t => ({x: t.off_epa_pp, y: -t.def_epa_pp, t})),
    pointRadius: 15, pointHoverRadius: 18, backgroundColor: RK.teams.map(t => t.color), borderColor: 'rgba(255,255,255,.6)', borderWidth: 1.5}]},
    options: {plugins: {legend: {display: false}, tooltip: {callbacks: {label: it => { const t = it.raw.t; return [`${t.name} (${t.w}–${t.l})`, `Offense EPA ${sgn(t.off_epa_pp, 3)} (#${t.rk_off_epa_pp})`, `Defense EPA ${sgn(t.def_epa_pp, 3)} (#${t.rk_def_epa_pp})`]; }}}},
      onClick: (e, els) => { if (els.length) location.hash = '#/team/' + RK.teams[els[0].index].team; },
      scales: {x: {title: {display: true, text: 'Offense EPA per play →'}, grid: {color: css('--grid')}}, y: {title: {display: true, text: 'Defense (EPA allowed, inverted) →'}, grid: {color: css('--grid')}}}}}));
}

// ---------------------------------------------------------------- team page
function pageTeam(t) {
  const r = RK.teams.find(x => x.team === t);
  if (!r) return pageRankings();
  const sched = ML.games.filter(g => g.home === t || g.away === t).sort((a, b) => a.day.localeCompare(b.day));
  $('#hero').innerHTML = `<section class="th" style="background:linear-gradient(110deg,${r.color},${r.color2 || r.color})"><div class="wrap">${tb(t, 74)}
    <div><div class="eyebrow" style="color:rgba(255,255,255,.85)">${esc(r.div)} · Elo rank #${r.rk_elo}</div><h1>${esc(r.name)}</h1><p>${r.w}–${r.l}${r.t ? '–' + r.t : ''} · point diff ${sgn(r.pd, 0)} · QB ${esc(r.qb || '—')}</p></div></div></section>`;
  const k = (v, l, rk) => `<div><b class="num">${v}</b><small>${l}</small>${rk ? `<em>#${rk} of 32</em>` : ''}</div>`;
  $('#app').innerHTML = `
    <div class="kpis">${k(Math.round(r.elo), 'Elo rating', r.rk_elo)}${k(sgn(r.off_epa_pp, 3), 'Offense EPA/play', r.rk_off_epa_pp)}${k(sgn(r.def_epa_pp, 3), 'Defense EPA allowed', r.rk_def_epa_pp)}${k(sgn(r.net_epa, 3), 'Net EPA', r.rk_net_epa)}${k(sgn(r.qb_epa, 3), 'QB EPA/dropback', r.rk_qb_epa)}${k(r.off_ppd?.toFixed(2) ?? '—', 'Points per drive')}</div>
    <div class="card"><h3>${ML.season} schedule & model picks</h3><div class="tbl"><table><thead><tr><th>Wk</th><th>Opponent</th><th class="n">Model win %</th><th class="n">Vegas win %</th><th class="n">Projected</th><th class="n">Result</th><th>Top bet</th></tr></thead><tbody>
      ${sched.map(g => { const home = g.home === t, opp = home ? g.away : g.home, p = home ? g.p_home : 1 - g.p_home, mk = g.mkt_home == null ? null : home ? g.mkt_home : 1 - g.mkt_home;
        const my = home ? g.pred_home : g.pred_away, th = home ? g.pred_away : g.pred_home;
        const res = g.final ? (() => { const a = home ? g.home_score : g.away_score, b = home ? g.away_score : g.home_score; return `<span class="pill ${a > b ? 'w' : a < b ? 'l' : 'p'}">${a > b ? 'W' : a < b ? 'L' : 'T'} ${a}–${b}</span>`; })() : '';
        const tb_ = [...g.bets].sort((x, y) => y.ev - x.ev)[0];
        return `<tr onclick="location.hash='#/game/${g.id}'" style="cursor:pointer"><td class="rk">${g.week}</td><td>${home ? 'vs' : '@'} ${tlink(opp)}</td><td class="n num" style="${p >= .5 ? 'font-weight:600' : ''}">${pct(p)}</td><td class="n num">${pct(mk)}</td>
          <td class="n num">${Math.round(my)}–${Math.round(th)}</td><td class="n">${res}</td><td>${tb_ && isPlay(tb_) ? `<span class="pill ${tb_.result ? tb_.result.toLowerCase() : 'play'}">${esc(tb_.pick)}</span>` : '<span class="muted">—</span>'}</td></tr>`; }).join('')}
    </tbody></table></div></div>
    <div class="card"><h3>Elo rating, last 5 seasons</h3><div class="chart-box tall"><canvas id="elo"></canvas></div></div>
    ${foot()}`;
  drawElo('elo', [t]);
}

// ---------------------------------------------------------------- line shop
function pageLines() {
  $('#hero').innerHTML = '';
  const S = ML.shop;
  if (!S || !S.rows.length) { $('#app').innerHTML = '<div class="card empty">No sportsbook odds loaded yet.</div>' + foot(); return; }
  const groups = {};
  S.rows.forEach(r => (groups[r.game + '|' + r.market] ??= []).push(r));
  const G = Object.values(groups).map(rs => { const s = [...rs].sort((a, b) => b.ev - a.ev); return {best: s[0], worst: s[s.length - 1], rows: rs}; });
  const show = G.filter(g => !shopPlays || isPlay(g.best)).sort((a, b) => b.best.ev - a.best.ev);
  const m = x => money(stake * x / 100);
  const disc = G.filter(g => g.best.ev > 0).filter(g => { const l = new Set(g.rows.map(r => r.line)), p = g.rows.map(r => r.price > 0 ? r.price - 100 : r.price + 100); return l.size > 1 || Math.max(...p) - Math.min(...p) > 8; })
    .sort((a, b) => (b.best.ev - b.worst.ev) - (a.best.ev - a.worst.ev));
  const games = [...new Set(S.rows.map(r => r.game))].sort((a, b) => groups[a + '|Spread']?.[0].day.localeCompare(groups[b + '|Spread']?.[0].day || '') || 0);
  $('#app').innerHTML = `
    <div class="head"><div><div class="eyebrow">Week ${ML.next_week} · ${S.books.join(', ')} · odds pulled ${esc(S.fetched_at)}</div><h2>Line shop</h2>
      <p>The model's side of every game, priced at each sportsbook. The same bet can be worth several dollars more at one book than another, so always take the best number.</p></div>
      <div class="controls"><div class="seg"><button id="sp" aria-pressed="${shopPlays}">Plays only</button><button id="sa" aria-pressed="${!shopPlays}">All bets</button></div></div></div>
    <div class="note"><b>Before you bet:</b> odds move all week, so confirm the price in the app. A big model edge on a line that moved hard the other way usually means news the model hasn't seen yet.</div>
    <div class="card"><h3>Best bets at the best price</h3><div class="tbl"><table><thead><tr><th>Game</th><th>Bet</th><th>Best book</th><th class="n">Price</th><th class="n">Edge</th><th class="n">Win %</th><th class="n">EV at $${stake}</th><th class="n">Worst book</th><th></th></tr></thead><tbody>
      ${show.map(({best: b, worst: w}) => { const [a, h] = b.game.split('@'); return `<tr onclick="location.hash='#/game/${ML.games.find(g => g.away === a && g.home === h && !g.final)?.id || ''}'" style="cursor:pointer"><td>${a} @ ${h}</td><td><b>${esc(b.pick)}</b></td><td><b>${b.book}</b></td><td class="n num">${odds(b.price)}</td><td class="n num">${b.edge.toFixed(1)}${b.market === 'Moneyline' ? '%' : ''}</td><td class="n num">${pct(b.prob)}</td><td class="n num ${b.ev >= 0 ? 'pos' : 'neg'}">${m(b.ev)}</td><td class="n num muted">${w.book} ${m(w.ev)}</td><td>${isPlay(b) ? '<span class="pill play">Play</span>' : '<span class="pill lean">Lean</span>'}</td></tr>`; }).join('') || '<tr><td colspan="9" class="empty">Nothing clears the bar at any book.</td></tr>'}
    </tbody></table></div></div>
    <div class="card"><h3>Where the books disagree</h3><p class="sub">Positive-value bets where books post different lines or prices more than 8 cents apart.</p><div class="body disc">${disc.map(g => `<div><b>${g.best.game.replace('@', ' @ ')} · ${g.best.market}</b>${[...g.rows].sort((a, b) => b.ev - a.ev).map(r => `<span class="${r === g.best ? 'hi' : ''}">${r.book}: ${esc(r.pick)} ${odds(r.price)} → ${m(r.ev)}</span>`).join('')}</div>`).join('') || '<div><span>All books agree.</span></div>'}</div></div>
    <div class="shopg">${games.map(gm => { const [a, h] = gm.split('@'); const mk = ['Spread', 'Total', 'Moneyline'].filter(x => groups[gm + '|' + x]);
      return `<div class="card"><h3 style="display:flex;align-items:center;gap:8px;padding-bottom:10px">${tb(a, 26)} ${a} @ ${h} ${tb(h, 26)}</h3><div class="tbl"><table><thead><tr><th>Model side</th>${S.books.map(b => `<th>${b}</th>`).join('')}</tr></thead><tbody>${mk.map(x => { const rs = groups[gm + '|' + x], best = Math.max(...rs.map(r => r.ev));
        return `<tr><td><b>${esc(rs[0].side)}</b><br><span class="muted" style="font-size:11px">${x}</span></td>${S.books.map(b => { const r = rs.find(q => q.book === b); return r ? `<td class="${r.ev === best && rs.length > 1 ? 'best' : ''}"><div class="cell"><b>${r.line == null ? 'ML' : x === 'Total' ? r.line : (r.line > 0 ? '+' : '') + r.line} ${odds(r.price)}</b><span class="${r.ev >= 0 ? 'pos' : 'neg'}">${m(r.ev)}</span></div></td>` : '<td>—</td>'; }).join('')}</tr>`; }).join('')}</tbody></table></div></div>`; }).join('')}</div>
    ${foot()}`;
  $('#sp').onclick = () => { shopPlays = true; pageLines(); };
  $('#sa').onclick = () => { shopPlays = false; pageLines(); };
}

// ---------------------------------------------------------------- 2026 record
function pageRecord() {
  $('#hero').innerHTML = '';
  const rows = ML.games.filter(g => g.final).flatMap(g => g.bets.filter(b => b.edge >= THR[b.market] && b.result).map(b => ({g, b}))).sort((a, b) => b.g.day.localeCompare(a.g.day));
  const by = m => { const r = rows.filter(x => x.b.market === m); return {w: r.filter(x => x.b.result === 'W').length, l: r.filter(x => x.b.result === 'L').length, p: r.filter(x => x.b.result === 'P').length,
    u: r.reduce((s, x) => s + (x.b.result === 'W' ? payout(x.b.odds) : x.b.result === 'L' ? -1 : 0), 0)}; };
  const mk = ['Spread', 'Total', 'Moneyline'].map(m => [m, by(m)]), tu = mk.reduce((s, [, x]) => s + x.u, 0);
  const wins = ML.games.filter(g => g.final && g.home_score !== g.away_score), su = wins.filter(g => (g.home_score > g.away_score) === (g.p_home >= .5)).length;
  $('#app').innerHTML = `<div class="head"><div><div class="eyebrow">Live test · out of sample</div><h2>${ML.season} record</h2><p>Every play the model flagged this season, graded against the final score. The model only saw games played before each kickoff.</p></div></div>
    <div class="kpis"><div><b class="num">${su}–${wins.length - su}</b><small>winners picked straight up</small><em>${pct(su / (wins.length || 1), 1)}</em></div>${mk.map(([m, x]) => `<div><b class="num">${x.w}–${x.l}${x.p ? '–' + x.p : ''}</b><small>${m} plays</small><em class="${x.u >= 0 ? 'pos' : 'neg'}">${money(x.u * stake)} at $${stake}</em></div>`).join('')}<div><b class="num ${tu >= 0 ? 'pos' : 'neg'}">${money(tu * stake)}</b><small>net at $${stake}/play</small></div></div>
    <div class="card"><h3>Graded plays</h3><div class="tbl"><table><thead><tr><th>Wk</th><th>Game</th><th>Play</th><th class="n">Odds</th><th class="n">Edge</th><th>Final</th><th>Result</th><th class="n">Profit</th></tr></thead><tbody>
      ${rows.map(({g, b}) => { const p = b.result === 'W' ? stake * payout(b.odds) : b.result === 'L' ? -stake : 0;
        return `<tr onclick="location.hash='#/game/${g.id}'" style="cursor:pointer"><td class="rk">${g.week}</td><td>${g.away} @ ${g.home}</td><td><b>${esc(b.pick)}</b></td><td class="n num">${odds(b.odds)}</td><td class="n num">${b.edge.toFixed(1)}${b.unit === '%' ? '%' : ''}</td><td class="num">${g.away_score}–${g.home_score}</td><td><span class="pill ${b.result.toLowerCase()}">${{W: 'Won', L: 'Lost', P: 'Push'}[b.result]}</span></td><td class="n num ${p >= 0 ? 'pos' : 'neg'}">${money(p)}</td></tr>`; }).join('') || '<tr><td colspan="8" class="empty">No graded plays yet.</td></tr>'}
    </tbody></table></div></div>${foot()}`;
}

// ---------------------------------------------------------------- performance + simulator
const sim = store.get('sim', {market: 'Spread', edge: 3, from: 2006, to: 2025});
function simulate() {
  const C = Object.fromEntries(HI.cols.map((c, i) => [c, i])), out = [];
  for (const r of HI.rows) {
    const s = r[C.season]; if (s < sim.from || s > sim.to) continue;
    let e, won, push, price;
    if (sim.market === 'Spread') {
      if (r[C.spread_line] == null) continue;
      e = r[C.pred_margin] - r[C.spread_line]; const home = e > 0, cov = Math.sign(r[C.result] - r[C.spread_line]);
      won = home ? cov > 0 : cov < 0; push = cov === 0; price = home ? r[C.home_spread_odds] : r[C.away_spread_odds]; e = Math.abs(e);
    } else if (sim.market === 'Total') {
      if (r[C.total_line] == null) continue;
      e = r[C.pred_total] - r[C.total_line]; const over = e > 0, sd = Math.sign(r[C.total] - r[C.total_line]);
      won = over ? sd > 0 : sd < 0; push = sd === 0; price = over ? r[C.over_odds] : r[C.under_odds]; e = Math.abs(e);
    } else {
      const hm = r[C.home_moneyline], am = r[C.away_moneyline]; if (hm == null || am == null || r[C.result] === 0) continue;
      const ip = o => o > 0 ? 100 / (o + 100) : -o / (-o + 100), mk = ip(hm) / (ip(hm) + ip(am)), d = r[C.p_home] - mk;
      e = Math.abs(d) * 100; const home = d > 0; won = home ? r[C.result] > 0 : r[C.result] < 0; push = false; price = home ? hm : am;
    }
    if (e < Math.max(sim.edge, .5)) continue;
    out.push({s, won, push, pnl: push ? 0 : won ? payout(price) : -1});
  }
  return out;
}
function pagePerformance() {
  $('#hero').innerHTML = '';
  const M = ML.metrics;
  const acc = [['Avg margin miss (pts)', M.margin_mae.model, M.margin_mae.vegas, 2], ['Avg total miss (pts)', M.total_mae.model, M.total_mae.vegas, 2], ['Brier score (win prob)', M.brier.model, M.brier.vegas, 4], ['Picks winner', M.winner_acc.model, M.winner_acc.vegas, 'p']];
  $('#app').innerHTML = `<div class="head"><div><div class="eyebrow">${M.test_seasons} · ${M.games.toLocaleString()} games out of sample</div><h2>Model performance</h2>
    <p>Walk-forward test: for every season since 2006 the model trained only on earlier seasons, then predicted the new one and was graded against Vegas closing lines.</p></div></div>
    <div class="card"><h3>Strategy simulator</h3><p class="sub">Pick a market and a minimum edge, then see how flat $${stake} bets would have done. Every number recalculates from ${HI ? HI.rows.length.toLocaleString() : ''} real games.</p>
      <div class="body controls" style="gap:22px">
        <div class="seg" id="mkt">${['Spread', 'Total', 'Moneyline'].map(m => `<button data-m="${m}" aria-pressed="${sim.market === m}">${m}</button>`).join('')}</div>
        <label class="rng">Minimum edge <output id="eo"></output><input type="range" id="edge" min="0" max="8" step="0.5" value="${sim.edge}"></label>
        <label class="rng">From season <output id="fo">${sim.from}</output><input type="range" id="from" min="2006" max="2025" value="${sim.from}"></label>
        <label class="rng">To season <output id="to_">${sim.to}</output><input type="range" id="to" min="2006" max="2025" value="${sim.to}"></label>
      </div>
      <div class="sim-out" id="so"></div>
      <div class="cols" style="padding:8px 4px 4px"><div class="chart-box"><canvas id="cum"></canvas></div><div class="chart-box"><canvas id="bys"></canvas></div></div></div>
    <div class="cols">
      <div class="card"><h3>Accuracy vs Vegas</h3><p class="sub">Vegas closing lines are the hardest benchmark in sports forecasting. Getting this close means the model is mostly seeing the same thing the market sees.</p><div class="body tbl"><table><thead><tr><th>Measure</th><th class="n">Model</th><th class="n">Vegas</th></tr></thead><tbody>
        ${acc.map(([n, a, b, d]) => { const f = x => d === 'p' ? pct(x, 1) : x.toFixed(d), lower = d !== 'p'; const aw = lower ? a < b : a > b;
          return `<tr><td>${n}</td><td class="n num" style="${aw ? 'font-weight:600' : ''}">${f(a)}</td><td class="n num" style="${!aw ? 'font-weight:600' : ''}">${f(b)}</td></tr>`; }).join('')}
        <tr><td>Correlation with Vegas spread</td><td class="n num">${M.corr_with_vegas_spread}</td><td class="n">—</td></tr></tbody></table></div></div>
      <div class="card"><h3>Win-probability calibration</h3><p class="sub">When the model says 70%, does the home team win about 70% of the time? Points on the dashed line mean yes.</p><div class="chart-box"><canvas id="cal"></canvas></div></div>
    </div>
    ${foot()}`;
  const go = () => {
    $('#eo').textContent = sim.market === 'Moneyline' ? sim.edge + '%' : sim.edge + ' pts';
    if (!HI) { $('#so').innerHTML = '<div><small>Loading history…</small></div>'; return; }
    const r = simulate(), n = r.length, nn = r.filter(x => !x.push).length, w = r.filter(x => x.won && !x.push).length, u = r.reduce((s, x) => s + x.pnl, 0);
    const seasons = [...new Set(r.map(x => x.s))].sort(), byS = seasons.map(s => { const z = r.filter(x => x.s === s); return {s, u: z.reduce((a, x) => a + x.pnl, 0), w: z.filter(x => x.won && !x.push).length / (z.filter(x => !x.push).length || 1), n: z.length}; });
    const up = byS.filter(x => x.u > 0).length, be = sim.market === 'Moneyline' ? null : .524;
    $('#so').innerHTML = [[n.toLocaleString(), 'bets'], [pct(w / (nn || 1), 1), 'win rate' + (be ? ' (52.4% break-even)' : '')], [`<span class="${u >= 0 ? 'pos' : 'neg'}">${money(u * stake)}</span>`, `profit at $${stake}/bet`],
      [`<span class="${u >= 0 ? 'pos' : 'neg'}">${sgn(100 * u / (n || 1), 1)}%</span>`, 'return on money bet'], [`${up} of ${byS.length}`, 'winning seasons']].map(([b, s]) => `<div><b class="num">${b}</b><small>${s}</small></div>`).join('');
    killCharts(); if (!chartDefaults()) return;
    let c = 0; const cum = byS.map(x => (c += x.u * stake));
    charts.push(new Chart($('#cum'), {type: 'line', data: {labels: byS.map(x => x.s), datasets: [{label: 'Cumulative profit', data: cum, borderColor: css('--accent'), backgroundColor: css('--accent-soft'), fill: true, tension: .2, pointRadius: 2}]},
      options: {plugins: {legend: {display: false}, title: {display: true, text: 'Cumulative profit ($)', align: 'start', color: css('--ink'), font: {size: 12}}, tooltip: {callbacks: {label: it => money(it.raw)}}}, scales: {y: {grid: {color: css('--grid')}, ticks: {callback: v => '$' + v.toLocaleString()}}, x: {grid: {display: false}}}}}));
    charts.push(new Chart($('#bys'), {type: 'bar', data: {labels: byS.map(x => x.s), datasets: [{label: 'Win rate', data: byS.map(x => 100 * x.w), backgroundColor: byS.map(x => be == null ? (x.u >= 0 ? css('--pos') : css('--neg')) : (x.w > be ? css('--pos') : css('--neg'))), borderRadius: 3}]},
      options: {plugins: {legend: {display: false}, title: {display: true, text: 'Win rate by season (%)', align: 'start', color: css('--ink'), font: {size: 12}}, tooltip: {callbacks: {label: it => `${it.raw.toFixed(1)}% on ${byS[it.dataIndex].n} bets · ${money(byS[it.dataIndex].u * stake)}`}}},
        scales: {y: {min: 30, max: 75, grid: {color: css('--grid')}}, x: {grid: {display: false}}}}}));
    const cal = ML.calibration.filter(d => d.n >= 30);
    charts.push(new Chart($('#cal'), {type: 'scatter', data: {datasets: [
      {label: 'Model', data: cal.map(d => ({x: 100 * d.pred, y: 100 * d.actual, n: d.n})), backgroundColor: css('--model'), pointRadius: cal.map(d => Math.max(4, Math.min(11, Math.sqrt(d.n) / 3)))},
      {label: 'Vegas', data: cal.map(d => ({x: 100 * d.mkt, y: 100 * d.actual, n: d.n})), backgroundColor: css('--book'), pointRadius: 4},
      {type: 'line', label: 'Perfect', data: [{x: 0, y: 0}, {x: 100, y: 100}], borderColor: css('--faint'), borderDash: [4, 4], pointRadius: 0, borderWidth: 1}]},
      options: {plugins: {tooltip: {callbacks: {label: it => it.dataset.label === 'Perfect' ? '' : `${it.dataset.label} said ${it.raw.x.toFixed(0)}% → won ${it.raw.y.toFixed(0)}% (${it.raw.n} games)`}}},
        scales: {x: {min: 0, max: 100, title: {display: true, text: 'Predicted home win %'}, grid: {color: css('--grid')}}, y: {min: 0, max: 100, title: {display: true, text: 'Actual home win %'}, grid: {color: css('--grid')}}}}}));
  };
  $$('#mkt button').forEach(b => b.onclick = () => { sim.market = b.dataset.m; if (sim.market === 'Moneyline' && sim.edge > 8) sim.edge = 5; store.set('sim', sim); $$('#mkt button').forEach(x => x.setAttribute('aria-pressed', x === b)); go(); });
  $('#edge').oninput = e => { sim.edge = +e.target.value; store.set('sim', sim); go(); };
  $('#from').oninput = e => { sim.from = Math.min(+e.target.value, sim.to); $('#fo').textContent = sim.from; store.set('sim', sim); go(); };
  $('#to').oninput = e => { sim.to = Math.max(+e.target.value, sim.from); $('#to_').textContent = sim.to; store.set('sim', sim); go(); };
  go();
  if (!HI) fetch('data/history.json').then(r => r.json()).then(d => { HI = d; if (location.hash.startsWith('#/performance')) pagePerformance(); });
}

// ---------------------------------------------------------------- about
function pageAbout() {
  $('#hero').innerHTML = '';
  const LBL = {matchup_epa: 'Offense EPA vs opponent defense', elo_diff: 'Elo rating gap', qb_epa: 'Starting QB EPA per dropback', home: 'Home field', matchup_pass: 'Pass EPA matchup', off_pts_pg: 'Points per game', qb_exp: 'QB experience', wind: 'Wind', oppdef_pts_pg: 'Opponent points allowed', off_ppd: 'Points per drive', qb_new: 'New starting QB', opp_inj_def: 'Opponent defensive injuries', oppdef_expl: 'Opponent explosive plays allowed', oppdef_pen_pg: 'Opponent penalties', oppdef_plays_pg: 'Opponent plays faced', matchup_rush: 'Rush EPA matchup', off_inj: 'Own offensive injuries', rest_diff: 'Rest advantage', temp: 'Temperature', off_sr: 'Success rate'};
  const imp = Object.entries(ML.importance).slice(0, 15), mx = imp[0][1];
  $('#app').innerHTML = `<div class="head"><div><div class="eyebrow">Method</div><h2>How the model works</h2></div></div>
    <div class="cols"><div class="card"><h3>Pipeline</h3><div class="body prose">
      <p><b>Data.</b> nflverse play-by-play for every game since 1999 (about 1.3 million plays), weekly injury reports since 2009, snap counts since 2012, and results with Vegas spreads, totals and moneylines. Sportsbook prices come from The Odds API.</p>
      <p><b>Features, all known before kickoff:</b></p><ul>
        <li>Offense and defense EPA per play (pass and rush), success rate, explosive plays, points per drive, red-zone scoring, sacks, turnovers, third downs, pass rate over expected, special teams, penalties</li>
        <li>Each weighted toward recent games (half-life about 8 games), carried 45% into a new season and pulled toward league average early on</li>
        <li>Elo rating with a margin-of-victory adjustment</li>
        <li>Starting QB's recent EPA per dropback and experience, plus a new-starter flag</li>
        <li>Injury load: share of snaps held by players listed Out or Doubtful</li>
        <li>Rest, home field, division game, playoffs, dome, wind and temperature</li></ul>
      <p><b>Learning.</b> A ridge regression and a LightGBM gradient-boosted model each predict a team's points from its offense against the opponent's defense, blended 70/30. Home and away predictions give the score, margin and total; a logistic layer turns margin into win probability.</p>
      <p><b>Testing.</b> Walk-forward by season, retrained every year on past seasons only, with recent seasons weighted more.</p>
      <p><b>Edge.</b> The gap between model and line becomes a win probability based on how often gaps that size actually cashed in the backtest, then expected profit at each book's price.</p></div></div>
    <div class="card"><h3>What the model leans on</h3><p class="sub">Share of LightGBM split gain in the latest training run.</p><div class="body imp">${imp.map(([k, v]) => `<div><span title="${k}">${LBL[k] || k}</span><span class="b" style="width:${100 * v / mx}%"></span><span class="num">${(100 * v).toFixed(1)}%</span></div>`).join('')}</div></div></div>
    <div class="card"><h3>Read before betting</h3><div class="body prose"><ul>
      <li>The backtest uses closing lines. Earlier lines are softer, but they also move against you when you're right.</li>
      <li>About 55% against the spread over 1,400 bets beats chance, but single seasons have ranged from 41% to 67%. Expect losing years.</li>
      <li>The model doesn't read news. A late QB scratch or injury after the morning refresh isn't in these numbers.</li>
      <li>Keep each bet to 1–2% of a bankroll you can afford to lose. This is a research and portfolio project, not financial advice.</li></ul></div></div>
    ${foot()}`;
}

// ---------------------------------------------------------------- router
function route() {
  killCharts();
  const [r, arg] = location.hash.replace(/^#\/?/, '').split('/');
  setNav(r === 'week' || r === 'game' ? '' : r === 'team' ? 'rankings' : r || '');
  ({'': () => pageWeek(), week: () => pageWeek(+arg), game: () => pageGame(arg), rankings: pageRankings, team: () => pageTeam(arg),
    lines: pageLines, record: pageRecord, performance: pagePerformance, about: pageAbout}[r] || (() => pageWeek()))();
  if (!route.keep) scrollTo(0, 0);
}
addEventListener('hashchange', route);
Promise.all(['ml', 'rankings', 'matchups'].map(f => fetch(`data/${f}.json`).then(r => { if (!r.ok) throw new Error(f); return r.json(); })))
  .then(([a, b, c]) => { ML = a; RK = b; MU = c; RK.teams.forEach(t => TEAM[t.team] = t); Object.entries(ML.teams).forEach(([k, v]) => TEAM[k] ??= {team: k, ...v}); route(); })
  .catch(e => { $('#app').innerHTML = `<div class="card empty">Couldn't load the model data (${esc(e.message)}). Try refreshing.</div>`; });
})();
