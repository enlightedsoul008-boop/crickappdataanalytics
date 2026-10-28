// Prediction engine: learns from completed matches and combines several
// "mini models" whose weights adapt to how well each one has actually performed.
// Works for any number of experts (the app uses 10).
// A prediction value is 0 (team 1), 1 (team 2), null (paid / hidden) or -1 (expert not tracked for that match).
// m.preds    = what was known when the match was entered (drives the prediction + situation stats)
// m.revealed = paid picks that became known only later (used only to learn expert records)
(function (root) {
  'use strict';

  const ETA = 0.35;   // learning rate for model weights (higher = faster shift to winners)
  const SHARE = 0.02; // keeps every model alive so the mix can re-adapt when patterns change
  const CAL_LR = 0.05, CAL_MIN = 0.3, CAL_MAX = 3; // online confidence calibration

  const MODELS = [
    { id: 'maj',  name: 'Simple Majority' },
    { id: 'acc',  name: 'Expert Accuracy' },
    { id: 'role', name: 'Majority / Minority Role' },
    { id: 'pat',  name: 'Pattern / Group Memory' },
    { id: 'form', name: 'Recent Form' },
    { id: 'team', name: 'Team-specific' },
    { id: 'h2h',  name: 'Head-to-Head' },
  ];

  const BUCKETS = [
    { lab: 'STRONG', min: 0.72 },
    { lab: 'GOOD',   min: 0.62 },
    { lab: 'LEAN',   min: 0.55 },
    { lab: 'RISKY',  min: 0 },
  ];

  const cnt = () => ({ c: 0, n: 0 });
  const get = (o, k) => o[k] || (o[k] = cnt());
  const rate = (c, n, prior = 0.5, k = 2) => (c + prior * k) / (n + k);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const logit = p => Math.log(p / (1 - p));
  const sig = z => 1 / (1 + Math.exp(-z));
  const norm = s => String(s || '').trim().toLowerCase();
  const isDone = m => m.result === 0 || m.result === 1;
  const isOpen = p => p === 0 || p === 1;
  const isPaid = p => p === null;
  const pc = x => Math.round(x * 100) + '%';

  // Pads/truncates a match to n experts; missing experts become -1 (not tracked).
  function fit(m, n) {
    const P = m.preds || [], V = m.revealed || [];
    if (P.length === n && V.length === n) return m;
    const preds = [], revealed = [];
    for (let i = 0; i < n; i++) {
      const p = P[i];
      preds.push(isOpen(p) || isPaid(p) ? p : -1);
      revealed.push(isPaid(p) && isOpen(V[i]) ? V[i] : null);
    }
    return { ...m, preds, revealed };
  }

  // Role of an open expert among the open votes.
  // key = exact situation ("3/4" = agrees with 2 others out of 4 open), cat = broad bucket.
  function roleOf(P, i) {
    const k = P.filter(isOpen).length, agree = P.filter(x => x === P[i]).length;
    const cat = k === 1 ? 'solo' : agree * 2 > k ? 'maj' : agree * 2 < k ? 'min' : 'tie';
    return { key: agree + '/' + k, cat, agree, k };
  }

  function splitOf(P) {
    const v0 = P.filter(x => x === 0).length, v1 = P.filter(x => x === 1).length;
    return {
      key: Math.max(v0, v1) + '-' + Math.min(v0, v1),
      majTeam: v0 > v1 ? 0 : v1 > v0 ? 1 : null,
      v0, v1, k: v0 + v1, paid: P.filter(isPaid).length,
    };
  }

  // Split keys ordered: more open votes first, then more one-sided first.
  function splitSort(keys) {
    const parse = k => k.split('-').map(Number);
    return keys.slice().sort((x, y) => {
      const [a1, b1] = parse(x), [a2, b2] = parse(y);
      return (a2 + b2) - (a1 + b1) || (a2 - b2) - (a1 - b1);
    });
  }

  // Grouping of experts, normalised so the first open expert is always on side "0"; "x" = not open.
  function patternKey(P) {
    const f = P.find(isOpen);
    return P.map(x => (isOpen(x) ? String(x ^ f) : 'x')).join('');
  }

  function patternSides(key, names) {
    const a = [], b = [], off = [];
    key.split('').forEach((ch, i) => (ch === '0' ? a : ch === '1' ? b : off).push(names[i]));
    return { a, b, off };
  }

  function patternLabel(key, names) {
    const { a, b } = patternSides(key, names);
    return b.length ? a.join('+') + ' vs ' + b.join('+') : a.join('+') + (a.length > 1 ? ' (sab same)' : ' (akela open)');
  }

  // Decision-time picks with later-revealed paid picks filled in.
  function fullPreds(m) {
    return m.preds.map((p, i) => (isOpen(p) ? p : isPaid(p) && m.revealed && isOpen(m.revealed[i]) ? m.revealed[i] : null));
  }

  // Applies an edit to a stored match. data.preds must already be in the new team order.
  function mergeEdit(old, data) {
    const swapped = norm(data.t1) === norm(old.t2) && norm(data.t2) === norm(old.t1);
    const m = { ...old, ...data };
    if (swapped && isDone(old)) m.result = 1 - old.result;
    m.revealed = data.preds.map((p, i) => (isPaid(p) && data.revealed && isOpen(data.revealed[i]) ? data.revealed[i] : null));
    return m;
  }

  const grid = n => Array.from({ length: n }, () => Array.from({ length: n }, cnt));

  function newStats(n) {
    const ex = [];
    for (let i = 0; i < n; i++) ex.push({ c: 0, n: 0, paid: 0, paidRev: cnt(), hist: [], streak: 0, teams: {}, role: {}, cat: {} });
    // pair[i][j]: i right when i and j disagreed; agree[i][j]: both right when they agreed
    return { size: n, n: 0, ex, pair: grid(n), agree: grid(n), split: {}, pattern: {} };
  }

  function update(S, m) {
    const P = m.preds, F = fullPreds(m), r = m.result, n = S.size;
    F.forEach((p, i) => {
      const e = S.ex[i];
      if (isPaid(P[i])) e.paid++;
      if (!isOpen(p)) return;
      const ok = p === r ? 1 : 0;
      e.c += ok; e.n++;
      e.hist.push(ok);
      e.streak = ok ? (e.streak >= 0 ? e.streak + 1 : 1) : (e.streak <= 0 ? e.streak - 1 : -1);
      [norm(m.t1), norm(m.t2)].forEach(t => { const tc = get(e.teams, t); tc.c += ok; tc.n++; });
      if (!isOpen(P[i])) { e.paidRev.c += ok; e.paidRev.n++; return; }
      // role records describe the situation that was visible at decision time
      const ro = roleOf(P, i);
      const a = get(e.role, ro.key); a.c += ok; a.n++;
      const b = get(e.cat, ro.cat); b.c += ok; b.n++;
    });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i === j || !isOpen(F[i]) || !isOpen(F[j])) continue;
      const cell = F[i] !== F[j] ? S.pair[i][j] : S.agree[i][j];
      cell.n++; if (F[i] === r) cell.c++;
    }
    const sp = splitOf(P), s = get(S.split, sp.key);
    s.n++; if (sp.majTeam !== null && sp.majTeam === r) s.c++;
    const pt = get(S.pattern, patternKey(P));
    pt.n++; if (P.find(isOpen) === r) pt.c++;
    S.n++;
  }

  // Naive-Bayes style vote: each open expert pushes toward their pick by log-odds of their accuracy.
  // Damped because experts are rarely fully independent; final calibration fixes the rest.
  function nb(P, accs, damp) {
    let z = 0;
    P.forEach((p, i) => { if (isOpen(p)) z += (p === 0 ? 1 : -1) * logit(clamp(accs[i], 0.1, 0.9)); });
    return sig(z * damp);
  }

  // Every model returns P(team 1 wins).
  function modelProbs(S, m) {
    const P = m.preds, out = {}, n = S.size;
    const sp = splitOf(P);
    const overall = S.ex.map(e => rate(e.c, e.n, 0.5, 4));

    out.maj = clamp(0.5 + 0.3 * (sp.v0 - sp.v1) / n, 0.05, 0.95);

    out.acc = nb(P, overall, 0.6);

    // exact situation record, backed off to the broad role bucket, backed off to overall accuracy
    out.role = nb(P, S.ex.map((e, i) => {
      if (!isOpen(P[i])) return 0.5;
      const ro = roleOf(P, i), c = e.cat[ro.cat] || cnt(), r = e.role[ro.key] || cnt();
      return rate(r.c, r.n, rate(c.c, c.n, overall[i], 3), 3);
    }), 0.6);

    // Group memory: how often each agreeing pair was right together; the exact grouping's own
    // history (when it has repeated) refines that.
    let za = 0, ka = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (!isOpen(P[i]) || P[i] !== P[j]) continue;
      const g = S.agree[i][j];
      za += (P[i] === 0 ? 1 : -1) * logit(clamp(rate(g.c, g.n, (overall[i] + overall[j]) / 2, 3), 0.1, 0.9));
      ka++;
    }
    const group = ka ? sig((za / ka) * 1.5) : out.acc;
    const first = P.find(isOpen);
    const pt = S.pattern[patternKey(P)] || cnt();
    const side0 = rate(pt.c, pt.n, first === 0 ? group : 1 - group, 3);
    out.pat = first === 0 ? side0 : 1 - side0;

    out.form = nb(P, S.ex.map((e, i) => {
      const h = e.hist.slice(-8);
      return rate(h.reduce((a, b) => a + b, 0), h.length, overall[i], 2);
    }), 0.6);

    const t1 = norm(m.t1), t2 = norm(m.t2);
    out.team = nb(P, S.ex.map((e, i) => {
      const a = e.teams[t1] || cnt(), b = e.teams[t2] || cnt();
      return rate(a.c + b.c, a.n + b.n, overall[i], 3);
    }), 0.6);

    let z = 0, k = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (!isOpen(P[i]) || !isOpen(P[j]) || P[i] === P[j]) continue;
      const pr = S.pair[i][j];
      z += (P[i] === 0 ? 1 : -1) * logit(clamp(rate(pr.c, pr.n, 0.5, 2), 0.1, 0.9));
      k++;
    }
    out.h2h = k ? sig((z / k) * 1.5) : out.maj;

    return out;
  }

  function weightsFrom(lw) {
    const ids = Object.keys(lw), mx = Math.max(...ids.map(k => lw[k]));
    const w = {}; let s = 0;
    ids.forEach(k => { w[k] = Math.exp(lw[k] - mx); s += w[k]; });
    ids.forEach(k => { w[k] /= s; });
    return w;
  }

  const mix = (probs, w) => MODELS.reduce((p, md) => p + w[md.id] * probs[md.id], 0);
  // Scales confidence (never flips the pick): cal > 1 sharpens, < 1 softens.
  const calibrate = (pRaw, cal) => sig(cal * logit(clamp(pRaw, 1e-4, 1 - 1e-4)));

  function finalize(pRaw, cal, probs, w, m, S) {
    let pick;
    if (Math.abs(pRaw - 0.5) > 1e-9) pick = pRaw > 0.5 ? 0 : 1;
    else {
      const sp = splitOf(m.preds);
      if (sp.majTeam !== null) pick = sp.majTeam;
      else {
        let best = -1;
        S.ex.forEach((e, i) => {
          if (isOpen(m.preds[i]) && (best < 0 || rate(e.c, e.n) > rate(S.ex[best].c, S.ex[best].n))) best = i;
        });
        pick = m.preds[best];
      }
    }
    const p = calibrate(pRaw, cal);
    const conf = Math.max(p, 1 - p);
    const verdict = BUCKETS.find(b => conf >= b.min).lab;
    return { p, pRaw, pick, conf, verdict, probs, w: { ...w }, n: S.n };
  }

  function insights(S, m, names, pick) {
    const P = m.preds, out = [], tn = [m.t1, m.t2];
    const recOf = r => (r && r.n ? `${r.c}/${r.n} sahi (${pc(r.c / r.n)})` : 'abhi tak koi record nahi');

    const sp = splitOf(P);
    const paidNames = names.filter((_, i) => isPaid(P[i]));
    if (paidNames.length) out.push(`💰 Paid: ${paidNames.join(', ')} — system sirf ${sp.k} open prediction se decide kar raha hai.`);

    if (sp.k === 1) {
      const i = P.findIndex(isOpen), e = S.ex[i];
      out.push(`Sirf ${names[i]} open hai (${tn[P[i]]}). Overall: ${recOf(e)}. Akela open hone pe: ${recOf(e.cat.solo)}.`);
      if (pick !== undefined && pick !== P[i]) out.push(`⚠️ System ${names[i]} ke OPPOSITE ja raha hai, kyunki uska record 50% se kam hai.`);
    } else if (sp.majTeam !== null) {
      const s = S.split[sp.key];
      out.push(`${sp.key} split (majority: ${tn[sp.majTeam]}) — pehle ${s && s.n ? `majority ${s.c}/${s.n} baar jeeti (${pc(s.c / s.n)})` : 'is split ka koi data nahi'}.`);
    } else {
      out.push(`${sp.key} split — majority nahi hai, system experts ki history se decide karega.`);
    }

    const key = patternKey(P), pt = S.pattern[key];
    if (sp.k > 1 && pt && pt.n) {
      const f = P.find(isOpen);
      const side0 = names.filter((_, i) => P[i] === f).join('+');
      out.push(`Ye exact grouping pehle ${pt.n} baar aayi — ${side0} side ${pt.c}/${pt.n} baar sahi (${pc(pt.c / pt.n)}).`);
    }

    // strongest agreeing pair with enough history
    let bestPair = null;
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      if (!isOpen(P[i]) || P[i] !== P[j]) continue;
      const g = S.agree[i][j];
      if (g.n >= 5 && (!bestPair || g.c / g.n > bestPair.r)) bestPair = { i, j, r: g.c / g.n, g };
    }
    if (bestPair) out.push(`${names[bestPair.i]} + ${names[bestPair.j]} (${tn[P[bestPair.i]]}) — jab ye dono saath hote hain: ${recOf(bestPair.g)}.`);

    P.forEach((p, i) => {
      if (!isOpen(p)) return;
      const e = S.ex[i], ro = roleOf(P, i);
      if (ro.cat === 'min' && ro.agree === 1) out.push(`${names[i]} akela opposite (${tn[p]}) — minority me hote hue ${recOf(e.cat.min)}.`);
      if (e.streak >= 3) out.push(`🔥 ${names[i]} lagatar ${e.streak} match se sahi.`);
      if (e.streak <= -3) out.push(`❄️ ${names[i]} lagatar ${-e.streak} match se galat.`);
    });

    if (S.n >= 5 && sp.k > 1) {
      let best = -1;
      S.ex.forEach((e, i) => { if (isOpen(P[i]) && e.n && (best < 0 || e.c / e.n > S.ex[best].c / S.ex[best].n)) best = i; });
      if (best >= 0) out.push(`Open experts me best: ${names[best]} (${pc(S.ex[best].c / S.ex[best].n)}) ne ${tn[P[best]]} bola.`);
    }
    return out;
  }

  // Walk-forward: every match is predicted using ONLY the matches before it,
  // so the accuracy shown for past matches is honest, not hindsight.
  function run(matches, names) {
    const n = names.length, S = newStats(n), lw = {};
    let cal = 1;
    MODELS.forEach(md => { lw[md.id] = 0; });
    const rows = [];
    for (const raw of matches) {
      const m = fit(raw, n);
      if (!m.preds.some(isOpen)) continue; // nothing to predict from
      const probs = modelProbs(S, m), w = weightsFrom(lw);
      const pred = finalize(mix(probs, w), cal, probs, w, m, S);
      pred.insights = insights(S, m, names, pred.pick);
      rows.push({ m, pred });
      if (isDone(m)) {
        MODELS.forEach(md => {
          const pa = m.result === 0 ? probs[md.id] : 1 - probs[md.id];
          lw[md.id] += ETA * Math.log(clamp(pa, 0.02, 0.98));
        });
        const nw = weightsFrom(lw);
        MODELS.forEach(md => { lw[md.id] = Math.log((1 - SHARE) * nw[md.id] + SHARE / MODELS.length); });
        const y = m.result === 0 ? 1 : 0;
        cal = clamp(cal - CAL_LR * (pred.p - y) * logit(clamp(pred.pRaw, 1e-4, 1 - 1e-4)), CAL_MIN, CAL_MAX);
        update(S, m);
      }
    }
    return { rows, S, w: weightsFrom(lw), cal, size: n };
  }

  function predictWith(R, raw, names) {
    const m = fit(raw, R.size);
    const probs = modelProbs(R.S, m);
    const pred = finalize(mix(probs, R.w), R.cal, probs, R.w, m, R.S);
    pred.insights = insights(R.S, m, names, pred.pick);
    return pred;
  }

  function evaluate(rows) {
    const ev = {
      n: 0, sys: 0, decisive: 0, majC: 0, sysOnDecisive: 0,
      bySplit: {}, models: {}, buckets: BUCKETS.map(b => ({ ...b, c: 0, n: 0 })), series: [],
    };
    MODELS.forEach(md => { ev.models[md.id] = cnt(); });
    let cumM = 0;
    rows.forEach(({ m, pred }) => {
      if (!isDone(m)) return;
      const ok = pred.pick === m.result ? 1 : 0;
      ev.n++; ev.sys += ok;
      const sp = splitOf(m.preds), bs = get(ev.bySplit, sp.key);
      bs.n++; bs.c += ok;
      if (sp.majTeam !== null) {
        ev.decisive++; ev.sysOnDecisive += ok;
        if (sp.majTeam === m.result) ev.majC++;
      }
      cumM += sp.majTeam === null ? 0.5 : sp.majTeam === m.result ? 1 : 0;
      ev.series.push({ s: ev.sys / ev.n, m: cumM / ev.n });
      MODELS.forEach(md => {
        const p = pred.probs[md.id], mc = ev.models[md.id];
        mc.n++;
        mc.c += Math.abs(p - 0.5) < 1e-9 ? 0.5 : ((p > 0.5 ? 0 : 1) === m.result ? 1 : 0);
      });
      const b = ev.buckets.find(x => x.lab === pred.verdict);
      b.n++; b.c += ok;
    });
    return ev;
  }

  const api = { MODELS, BUCKETS, run, predictWith, evaluate, roleOf, splitOf, splitSort, fit,
                patternKey, patternLabel, patternSides, isDone, isOpen, isPaid, norm, fullPreds, mergeEdit };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Engine = api;
})(this);
