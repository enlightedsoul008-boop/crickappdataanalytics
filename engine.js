// Prediction engine: learns from completed matches and combines several
// "mini models" whose weights adapt to how well each one has actually performed.
(function (root) {
  'use strict';

  const ETA = 0.35;   // learning rate for model weights (higher = faster shift to winners)
  const SHARE = 0.02; // keeps every model alive so the mix can re-adapt when patterns change

  const MODELS = [
    { id: 'maj',  name: 'Simple Majority',          desc: 'Jis team ko zyada experts ne pick kiya' },
    { id: 'acc',  name: 'Expert Accuracy',          desc: 'Har expert ke overall hit-rate se weighted vote' },
    { id: 'role', name: 'Majority / Minority Role', desc: 'Expert majority me hai ya minority me, us role me uska record' },
    { id: 'pat',  name: 'Pattern Memory',           desc: 'Exact grouping (jaise A+B+C vs D) ka purana result' },
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

  const cnt = () => ({ c: 0, n: 0 });
  const rate = (c, n, prior = 0.5, k = 2) => (c + prior * k) / (n + k);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const logit = p => Math.log(p / (1 - p));
  const sig = z => 1 / (1 + Math.exp(-z));
  const norm = s => String(s || '').trim().toLowerCase();
  const isDone = m => m.result === 0 || m.result === 1;
  const pc = x => Math.round(x * 100) + '%';

  function roleOf(P, i) {
    const agree = P.filter(x => x === P[i]).length;
    return agree === 4 ? 'maj4' : agree === 3 ? 'maj3' : agree === 2 ? 'tie' : 'min';
  }

  function splitOf(P) {
    const v0 = P.filter(x => x === 0).length;
    const top = Math.max(v0, 4 - v0);
    return {
      key: top === 4 ? '4-0' : top === 3 ? '3-1' : '2-2',
      majTeam: v0 > 2 ? 0 : v0 < 2 ? 1 : null,
      v0,
    };
  }

  // Grouping of experts, normalised so expert #1 is always on side "0".
  function patternKey(P) { return P.map(x => x ^ P[0]).join(''); }

  function patternLabel(key, names) {
    const a = [], b = [];
    key.split('').forEach((bit, i) => (bit === '0' ? a : b).push(names[i]));
    return b.length ? a.join('+') + ' vs ' + b.join('+') : a.join('+') + ' (sab same)';
  }

  function newStats() {
    const ex = [], pair = [];
    for (let i = 0; i < 4; i++) {
      ex.push({ c: 0, n: 0, hist: [], streak: 0, teams: {},
        role: { maj4: cnt(), maj3: cnt(), tie: cnt(), min: cnt() } });
      pair.push([cnt(), cnt(), cnt(), cnt()]);
    }
    return { n: 0, ex, pair, split: { '4-0': cnt(), '3-1': cnt(), '2-2': cnt() }, pattern: {} };
  }

  function update(S, m) {
    const P = m.preds, r = m.result;
    P.forEach((p, i) => {
      const e = S.ex[i], ok = p === r ? 1 : 0;
      e.c += ok; e.n++;
      e.hist.push(ok);
      e.streak = ok ? (e.streak >= 0 ? e.streak + 1 : 1) : (e.streak <= 0 ? e.streak - 1 : -1);
      const ro = e.role[roleOf(P, i)]; ro.c += ok; ro.n++;
      [norm(m.t1), norm(m.t2)].forEach(t => {
        const tc = e.teams[t] || (e.teams[t] = cnt()); tc.c += ok; tc.n++;
      });
    });
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
      if (i !== j && P[i] !== P[j]) { S.pair[i][j].n++; if (P[i] === r) S.pair[i][j].c++; }
    }
    const sp = splitOf(P), s = S.split[sp.key];
    s.n++; if (sp.majTeam !== null && sp.majTeam === r) s.c++;
    const key = patternKey(P), pt = S.pattern[key] || (S.pattern[key] = cnt());
    pt.n++; if (P[0] === r) pt.c++;
    S.n++;
  }

  // Naive-Bayes style vote: each expert pushes toward their pick by log-odds of their accuracy.
  // Damped because experts are correlated, which plain NB would over-count.
  function nb(P, accs, damp) {
    let z = 0;
    P.forEach((p, i) => { z += (p === 0 ? 1 : -1) * logit(clamp(accs[i], 0.1, 0.9)); });
    return sig(z * damp);
  }

  // Every model returns P(team 1 wins).
  function modelProbs(S, m) {
    const P = m.preds, out = {};
    const sp = splitOf(P);
    const overall = S.ex.map(e => rate(e.c, e.n, 0.5, 4));

    out.maj = 0.5 + (sp.v0 - 2) * 0.15;

    out.acc = nb(P, overall, 0.6);

    out.role = nb(P, S.ex.map((e, i) => {
      const r = e.role[roleOf(P, i)];
      return rate(r.c, r.n, overall[i], 3);
    }), 0.6);

    let prior = 0.5;
    if (sp.key !== '2-2') {
      const s = S.split[sp.key];
      const majRate = rate(s.c, s.n, sp.key === '4-0' ? 0.75 : 0.62, 3);
      prior = roleOf(P, 0) === 'min' ? 1 - majRate : majRate;
    }
    const pt = S.pattern[patternKey(P)] || cnt();
    const side0 = rate(pt.c, pt.n, prior, 3);
    out.pat = P[0] === 0 ? side0 : 1 - side0;

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
      if (P[i] === P[j]) continue;
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
        let best = 0;
        S.ex.forEach((e, i) => { if (rate(e.c, e.n) > rate(S.ex[best].c, S.ex[best].n)) best = i; });
        pick = m.preds[best];
      }
    }
    const conf = Math.max(p, 1 - p);
    const verdict = BUCKETS.find(b => conf >= b.min).lab;
    return { p, pick, conf, verdict, probs, w: { ...w }, n: S.n };
  }

  function insights(S, m, names) {
    const P = m.preds, out = [], tn = [m.t1, m.t2];
    if (S.n < 8) out.push(`Abhi sirf ${S.n} completed match ka data hai — system seekh raha hai, prediction kam reliable.`);

    const sp = splitOf(P);
    if (sp.key !== '2-2') {
      const s = S.split[sp.key];
      out.push(`${sp.key} split (majority: ${tn[sp.majTeam]}) — pehle ${s.n ? `majority ${s.c}/${s.n} baar jeeti (${pc(s.c / s.n)})` : 'is split ka koi data nahi'}.`);
    } else {
      out.push('2-2 split — majority nahi hai, system experts ki history se decide karega.');
    }

    const key = patternKey(P), pt = S.pattern[key];
    const lbl = patternLabel(key, names);
    const side0 = names.filter((_, i) => P[i] === P[0]).join('+');
    out.push(pt && pt.n
      ? `Pattern "${lbl}" pehle ${pt.n} baar aaya — ${side0} side ${pt.c}/${pt.n} baar sahi (${pc(pt.c / pt.n)}).`
      : `Pattern "${lbl}" pehli baar aaya hai.`);

    P.forEach((p, i) => {
      const e = S.ex[i], role = roleOf(P, i);
      if (role === 'min') {
        const r = e.role.min;
        out.push(`${names[i]} akela opposite (${tn[p]}) — minority me hote hue ${r.n ? `${r.c}/${r.n} baar sahi (${pc(r.c / r.n)})` : 'abhi tak koi record nahi'}.`);
      }
      if (e.streak >= 3) out.push(`🔥 ${names[i]} lagatar ${e.streak} match se sahi.`);
      if (e.streak <= -3) out.push(`❄️ ${names[i]} lagatar ${-e.streak} match se galat.`);
    });

    if (S.n >= 5) {
      let best = 0, worst = 0;
      S.ex.forEach((e, i) => {
        if (e.c / e.n > S.ex[best].c / S.ex[best].n) best = i;
        if (e.c / e.n < S.ex[worst].c / S.ex[worst].n) worst = i;
      });
      out.push(`Overall best: ${names[best]} (${pc(S.ex[best].c / S.ex[best].n)}) ne ${tn[P[best]]} bola. Weakest: ${names[worst]} (${pc(S.ex[worst].c / S.ex[worst].n)}).`);
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
      const probs = modelProbs(S, m), w = weightsFrom(lw);
      let p = 0;
      MODELS.forEach(md => { p += w[md.id] * probs[md.id]; });
      const pred = finalize(p, probs, w, m, S);
      pred.insights = insights(S, m, names);
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
    pred.insights = insights(R.S, m, names);
    return pred;
  }

  function evaluate(rows) {
    const ev = {
      n: 0, sys: 0, decisive: 0, majC: 0, sysOnDecisive: 0,
      bySplit: { '4-0': cnt(), '3-1': cnt(), '2-2': cnt() },
      models: {}, buckets: BUCKETS.map(b => ({ ...b, c: 0, n: 0 })), series: [],
    };
    MODELS.forEach(md => { ev.models[md.id] = cnt(); });
    let cumM = 0;
    rows.forEach(({ m, pred }) => {
      if (!isDone(m)) return;
      const ok = pred.pick === m.result ? 1 : 0;
      ev.n++; ev.sys += ok;
      const sp = splitOf(m.preds);
      ev.bySplit[sp.key].n++; ev.bySplit[sp.key].c += ok;
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

  const api = { MODELS, BUCKETS, run, predictWith, evaluate, roleOf, splitOf, patternKey, patternLabel, isDone, norm };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Engine = api;
})(this);
