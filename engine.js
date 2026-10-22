// Prediction engine: learns from completed matches and combines several
// "mini models" whose weights adapt to how well each one has actually performed.
// A prediction value is 0 (team 1), 1 (team 2) or null (paid / hidden).
(function (root) {
  'use strict';

  const ETA = 0.35;   // learning rate for model weights (higher = faster shift to winners)
  const SHARE = 0.02; // keeps every model alive so the mix can re-adapt when patterns change

  const MODELS = [
    { id: 'maj',  name: 'Simple Majority',          desc: 'Open predictions me jis team ko zyada vote mile' },
    { id: 'acc',  name: 'Expert Accuracy',          desc: 'Har open expert ke overall hit-rate se weighted vote' },
    { id: 'role', name: 'Majority / Minority Role', desc: 'Expert majority, minority ya akela open — us situation me uska record' },
    { id: 'pat',  name: 'Pattern Memory',           desc: 'Exact grouping (jaise A+B vs D, C paid) ka purana result' },
    { id: 'form', name: 'Recent Form',              desc: 'Har expert ke last 8 matches ka form' },
    { id: 'team', name: 'Team-specific',            desc: 'In dono teams ke matches me expert ka record' },
    { id: 'h2h',  name: 'Head-to-Head',             desc: 'Jab do expert opposite hote hain tab kaun sahi nikalta hai' },
  ];

  const BUCKETS = [
    { lab: 'STRONG', min: 0.72 },
    { lab: 'GOOD',   min: 0.62 },
    { lab: 'LEAN',   min: 0.55 },
    { lab: 'RISKY',  min: 0 },
  ];

  const SPLIT_ORDER = ['4-0', '3-1', '2-2', '3-0', '2-1', '2-0', '1-1', '1-0'];

  const cnt = () => ({ c: 0, n: 0 });
  const get = (o, k) => o[k] || (o[k] = cnt());
  const rate = (c, n, prior = 0.5, k = 2) => (c + prior * k) / (n + k);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const logit = p => Math.log(p / (1 - p));
  const sig = z => 1 / (1 + Math.exp(-z));
  const norm = s => String(s || '').trim().toLowerCase();
  const isDone = m => m.result === 0 || m.result === 1;
  const isOpen = p => p === 0 || p === 1;
  const pc = x => Math.round(x * 100) + '%';

  // Role of a visible expert among the visible votes.
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
      v0, v1, k: v0 + v1, paid: 4 - v0 - v1,
    };
  }

  // Grouping of experts, normalised so the first open expert is always on side "0"; "x" = paid.
  function patternKey(P) {
    const f = P.find(isOpen);
    return P.map(x => (isOpen(x) ? String(x ^ f) : 'x')).join('');
  }

  function patternSides(key, names) {
    const a = [], b = [], paid = [];
    key.split('').forEach((ch, i) => (ch === '0' ? a : ch === '1' ? b : paid).push(names[i]));
    return { a, b, paid };
  }

  function patternLabel(key, names) {
    const { a, b, paid } = patternSides(key, names);
    let s = b.length ? a.join('+') + ' vs ' + b.join('+') : a.join('+') + (a.length > 1 ? ' (sab same)' : ' (akela open)');
    if (paid.length) s += ' · paid: ' + paid.join('+');
    return s;
  }

  function newStats() {
    const ex = [], pair = [];
    for (let i = 0; i < 4; i++) {
      ex.push({ c: 0, n: 0, paid: 0, hist: [], streak: 0, teams: {}, role: {}, cat: {} });
      pair.push([cnt(), cnt(), cnt(), cnt()]);
    }
    return { n: 0, ex, pair, split: {}, pattern: {} };
  }

  function update(S, m) {
    const P = m.preds, r = m.result;
    P.forEach((p, i) => {
      const e = S.ex[i];
      if (!isOpen(p)) { e.paid++; return; }
      const ok = p === r ? 1 : 0;
      e.c += ok; e.n++;
      e.hist.push(ok);
      e.streak = ok ? (e.streak >= 0 ? e.streak + 1 : 1) : (e.streak <= 0 ? e.streak - 1 : -1);
      const ro = roleOf(P, i);
      const a = get(e.role, ro.key); a.c += ok; a.n++;
      const b = get(e.cat, ro.cat); b.c += ok; b.n++;
      [norm(m.t1), norm(m.t2)].forEach(t => { const tc = get(e.teams, t); tc.c += ok; tc.n++; });
    });
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      if (i !== j && isOpen(P[i]) && isOpen(P[j]) && P[i] !== P[j]) { S.pair[i][j].n++; if (P[i] === r) S.pair[i][j].c++; }
    }
    const sp = splitOf(P), s = get(S.split, sp.key);
    s.n++; if (sp.majTeam !== null && sp.majTeam === r) s.c++;
    const pt = get(S.pattern, patternKey(P));
    pt.n++; if (P.find(isOpen) === r) pt.c++;
    S.n++;
  }

  // Naive-Bayes style vote: each open expert pushes toward their pick by log-odds of their accuracy.
  // Damped because experts are correlated, which plain NB would over-count.
  function nb(P, accs, damp) {
    let z = 0;
    P.forEach((p, i) => { if (isOpen(p)) z += (p === 0 ? 1 : -1) * logit(clamp(accs[i], 0.1, 0.9)); });
    return sig(z * damp);
  }

  // Every model returns P(team 1 wins).
  function modelProbs(S, m) {
    const P = m.preds, out = {};
    const sp = splitOf(P);
    const overall = S.ex.map(e => rate(e.c, e.n, 0.5, 4));

    out.maj = 0.5 + (sp.v0 - sp.v1) * 0.075;

    out.acc = nb(P, overall, 0.6);

    // exact situation record, backed off to the broad role bucket, backed off to overall accuracy
    out.role = nb(P, S.ex.map((e, i) => {
      if (!isOpen(P[i])) return 0.5;
      const ro = roleOf(P, i), c = e.cat[ro.cat] || cnt(), r = e.role[ro.key] || cnt();
      return rate(r.c, r.n, rate(c.c, c.n, overall[i], 3), 3);
    }), 0.6);

    const first = P.find(isOpen);
    let prior = 0.5;
    if (sp.majTeam !== null) {
      const s = S.split[sp.key] || cnt();
      const majRate = rate(s.c, s.n, 0.5 + 0.25 * Math.abs(sp.v0 - sp.v1) / 4, 3);
      prior = first === sp.majTeam ? majRate : 1 - majRate;
    }
    const pt = S.pattern[patternKey(P)] || cnt();
    const side0 = rate(pt.c, pt.n, prior, 3);
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
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) {
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

  function finalize(p, probs, w, m, S) {
    let pick;
    if (Math.abs(p - 0.5) > 1e-9) pick = p > 0.5 ? 0 : 1;
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
    const conf = Math.max(p, 1 - p);
    const verdict = BUCKETS.find(b => conf >= b.min).lab;
    return { p, pick, conf, verdict, probs, w: { ...w }, n: S.n };
  }

  function insights(S, m, names, pick) {
    const P = m.preds, out = [], tn = [m.t1, m.t2];
    const recOf = r => (r && r.n ? `${r.c}/${r.n} sahi (${pc(r.c / r.n)})` : 'abhi tak koi record nahi');
    if (S.n < 8) out.push(`Abhi sirf ${S.n} completed match ka data hai — system seekh raha hai, prediction kam reliable.`);

    const sp = splitOf(P);
    const paidNames = names.filter((_, i) => !isOpen(P[i]));
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
    if (sp.k > 1) {
      const lbl = patternLabel(key, names);
      const f = P.find(isOpen);
      const side0 = names.filter((_, i) => P[i] === f).join('+');
      out.push(pt && pt.n
        ? `Pattern "${lbl}" pehle ${pt.n} baar aaya — ${side0} side ${pt.c}/${pt.n} baar sahi (${pc(pt.c / pt.n)}).`
        : `Pattern "${lbl}" pehli baar aaya hai.`);
    }

    P.forEach((p, i) => {
      if (!isOpen(p)) return;
      const e = S.ex[i], ro = roleOf(P, i);
      if (ro.cat === 'min') {
        out.push(`${names[i]} ${ro.agree === 1 ? 'akela opposite' : 'minority me'} (${tn[p]}) — minority me hote hue ${recOf(e.cat.min)}.`);
      }
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
    const S = newStats(), lw = {};
    MODELS.forEach(md => { lw[md.id] = 0; });
    const rows = [];
    for (const m of matches) {
      if (!m.preds.some(isOpen)) continue; // nothing to predict from
      const probs = modelProbs(S, m), w = weightsFrom(lw);
      let p = 0;
      MODELS.forEach(md => { p += w[md.id] * probs[md.id]; });
      const pred = finalize(p, probs, w, m, S);
      pred.insights = insights(S, m, names, pred.pick);
      rows.push({ m, pred });
      if (isDone(m)) {
        MODELS.forEach(md => {
          const pa = m.result === 0 ? probs[md.id] : 1 - probs[md.id];
          lw[md.id] += ETA * Math.log(clamp(pa, 0.02, 0.98));
        });
        const nw = weightsFrom(lw);
        MODELS.forEach(md => { lw[md.id] = Math.log((1 - SHARE) * nw[md.id] + SHARE / MODELS.length); });
        update(S, m);
      }
    }
    return { rows, S, w: weightsFrom(lw) };
  }

  function predictWith(R, m, names) {
    const probs = modelProbs(R.S, m);
    let p = 0;
    MODELS.forEach(md => { p += R.w[md.id] * probs[md.id]; });
    const pred = finalize(p, probs, R.w, m, R.S);
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

  const api = { MODELS, BUCKETS, SPLIT_ORDER, run, predictWith, evaluate, roleOf, splitOf,
                patternKey, patternLabel, patternSides, isDone, isOpen, norm };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Engine = api;
})(this);
