(() => {
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmt = (x) => (x == null ? "–" : x.toFixed(3));
  const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

  function setupTabs(container, onSelect) {
    const buttons = [...container.querySelectorAll("button")];
    buttons.forEach((b) =>
      b.addEventListener("click", () => {
        buttons.forEach((o) => o.setAttribute("aria-selected", o === b ? "true" : "false"));
        onSelect(b);
      })
    );
  }

  // ---------------------------------------------------------------- per-instance scores
  // Public test instances 301-310. Values are k / (number of goal literals).
  const INSTANCES = [301, 302, 303, 304, 305, 306, 307, 308, 309, 310];
  const t3 = (ks) => ks.map((k) => (k == null ? null : k / 3));
  const t13 = (ks) => ks.map((k) => (k == null ? null : k / 13));
  const SCORES = {
    trash: {
      literals: 3,
      series: [
        { name: "Ours: BDDL", color: "--accent", mix: 55, values: t3([2, 3, 2, 0, 2, 3, 0, 3, 3, 2]) },
        { name: "Ours: BDDL + clamp", color: "--accent", mix: 100, values: t3([3, 3, 2, 2, 2, 2, 0, 3, 2, 3]) },
        { name: "2025 checkpoint (partial run)", color: "--blue", mix: 100, values: t3([3, 2, null, null, null, null, null, null, null, null]) },
      ],
    },
    veg: {
      literals: 13,
      series: [
        { name: "Ours: BDDL", color: "--accent", mix: 100, values: t13([4, 3, 4, 7, 10, 1, 1, 7, 4, 5]) },
        { name: "2025 checkpoint (partial run)", color: "--blue", mix: 100, values: t13([5, 0, null, 4, 4, 7, 6, null, null, null]) },
      ],
    },
  };

  function seriesColor(s) {
    const c = css(s.color);
    return s.mix === 100 ? c : `color-mix(in srgb, ${c} ${s.mix}%, ${css("--bg")})`;
  }

  function drawInstances(key) {
    const data = SCORES[key];
    const host = document.getElementById("inst-chart");
    host.innerHTML = "";
    const W = 900, H = 260, m = { l: 44, r: 8, t: 12, b: 34 };
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Per-instance q-score" }, host);
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const y = (v) => m.t + ih * (1 - v);
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => {
      el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: css("--border"), "stroke-width": 1 }, svg);
      el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end", "font-size": 12, fill: css("--fg-faint") }, svg).textContent = v.toFixed(2);
    });
    const gw = iw / INSTANCES.length, ns = data.series.length;
    const bw = Math.min(22, (gw - 14) / ns);
    INSTANCES.forEach((inst, i) => {
      const gx = m.l + gw * i + (gw - bw * ns - 2 * (ns - 1)) / 2;
      data.series.forEach((s, j) => {
        const v = s.values[i];
        const x = gx + j * (bw + 2);
        if (v == null) {
          el("line", { x1: x + 2, x2: x + bw - 2, y1: y(0) - 3, y2: y(0) - 3, stroke: css("--fg-faint"), "stroke-width": 1, "stroke-dasharray": "2 2" }, svg);
          return;
        }
        const h = Math.max(ih * v, 2);
        const r = el("rect", { x, y: y(0) - h, width: bw, height: h, rx: 2, fill: seriesColor(s) }, svg);
        el("title", {}, r).textContent = `${s.name} · instance ${inst}: q = ${fmt(v)} (${Math.round(v * data.literals)}/${data.literals} literals)`;
      });
      el("text", { x: m.l + gw * i + gw / 2, y: H - 12, "text-anchor": "middle", "font-size": 12, fill: css("--fg-muted") }, svg).textContent = inst;
    });

    document.getElementById("inst-legend").innerHTML = data.series
      .map((s) => `<span><i style="background:${seriesColor(s)};height:10px"></i>${s.name}</span>`)
      .join("") + `<span><i style="background:none;border-top:1px dashed var(--fg-faint);height:0"></i>not run yet</span>`;

    // Means over all instances, plus a matched comparison where the baseline exists.
    const base = data.series[data.series.length - 1];
    const matched = INSTANCES.map((_, i) => i).filter((i) => base.values[i] != null);
    const full = data.series
      .filter((s) => s.values.every((v) => v != null))
      .map((s) => `${s.name} ${fmt(mean(s.values))}`);
    const onMatched = data.series.map((s) => `${s.name} ${fmt(mean(matched.map((i) => s.values[i])))}`);
    document.getElementById("inst-note").textContent =
      `Mean q over all 10: ${full.join(", ")}. On the ${matched.length} instances with a baseline run: ${onMatched.join(", ")}.`;
  }

  let instKey = "trash";
  setupTabs(document.getElementById("inst-tabs"), (b) => drawInstances((instKey = b.dataset.task)));

  // ---------------------------------------------------------------- videos
  const VIDEOS = {
    success: [
      { f: "trash_309", task: "picking_up_trash", inst: 309, q: "3/3", kind: "ok", speed: 8 },
      { f: "trash_308", task: "picking_up_trash", inst: 308, q: "3/3", kind: "ok", speed: 8 },
      { f: "trash_302", task: "picking_up_trash", inst: 302, q: "3/3", kind: "ok", speed: 8 },
      { f: "trash_306", task: "picking_up_trash", inst: 306, q: "3/3", kind: "ok", speed: 8 },
    ],
    long: [
      { f: "veg_305", task: "sorting_vegetables", inst: 305, q: "10/13", kind: "partial", speed: 10, note: "Best long-horizon episode." },
      { f: "veg_301", task: "sorting_vegetables", inst: 301, q: "4/13", kind: "partial", speed: 10, note: "Tracker ends at stage 12 while the true stage is 4." },
    ],
    fail: [
      { f: "trash_304", task: "picking_up_trash", inst: 304, q: "0/3", kind: "fail", speed: 8, note: "Stage prediction flips 70 times in one episode." },
      { f: "veg_309", task: "sorting_vegetables", inst: 309, q: "4/13", kind: "fail", speed: 10, note: "Tracker reaches the terminal stage 13 at true stage 4." },
    ],
  };
  const LABEL = { ok: "success", partial: "partial", fail: "failure" };

  function showVideos(set) {
    document.getElementById("vgrid").innerHTML = VIDEOS[set]
      .map(
        (v) => `<div class="vid">
          <video src="static/videos/${v.f}.mp4" poster="static/videos/${v.f}.jpg" muted loop playsinline controls preload="none"></video>
          <div class="meta"><span><code>${v.task}</code> · ${v.inst} · ${v.speed}&times;</span>
            <span><span class="q">q ${v.q}</span> <span class="pill ${v.kind}">${LABEL[v.kind]}</span></span></div>
          ${v.note ? `<div class="caption">${v.note}</div>` : ""}
        </div>`
      )
      .join("");
    // Autoplay whichever clips are on screen.
    const io = new IntersectionObserver((entries) =>
      entries.forEach((e) => (e.isIntersecting ? e.target.play().catch(() => {}) : e.target.pause()))
    );
    document.querySelectorAll("#vgrid video").forEach((v) => io.observe(v));
  }
  setupTabs(document.getElementById("vid-tabs"), (b) => showVideos(b.dataset.set));
  showVideos("success");

  // ---------------------------------------------------------------- stage traces
  let traces = null, traceIdx = 0;

  function drawTrace() {
    if (!traces) return;
    const task = traces.sorting_vegetables;
    const ep = task.episodes[traceIdx];
    const N = task.n_literals;
    const secPerQuery = task.step_interval / task.fps;
    const host = document.getElementById("trace-chart");
    host.innerHTML = "";
    const W = 900, H = 320, m = { l: 40, r: 12, t: 14, b: 40 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const T = ep.argmax.length * secPerQuery;
    const x = (i) => m.l + (iw * i * secPerQuery) / T;
    const y = (s) => m.t + ih * (1 - s / N);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Stage trace, instance ${ep.instance}` }, host);

    for (let s = 0; s <= N; s++) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(s), y2: y(s), stroke: css("--border"), "stroke-width": s % 2 ? 0.5 : 1 }, svg);
      if (s % 2 === 0 || s === N)
        el("text", { x: m.l - 8, y: y(s) + 4, "text-anchor": "end", "font-size": 12, fill: css("--fg-faint") }, svg).textContent = s;
    }
    for (let t = 0; t <= T; t += 60) {
      const px = m.l + (iw * t) / T;
      el("text", { x: px, y: H - 20, "text-anchor": "middle", "font-size": 12, fill: css("--fg-faint") }, svg).textContent = `${t / 60} min`;
    }
    el("text", { x: 12, y: m.t + ih / 2, transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle", "font-size": 12, fill: css("--fg-muted") }, svg).textContent = "stage";

    // Argmax: one dot per query, slightly transparent so dense runs read as bands.
    const blue = css("--blue");
    const g = el("g", { fill: blue, "fill-opacity": 0.35 }, svg);
    ep.argmax.forEach((a, i) => el("circle", { cx: x(i), cy: y(a), r: 1.8 }, g));

    // Ground truth: the stage the episode ended in. Monotone labels make this the maximum reached.
    el("line", { x1: m.l, x2: W - m.r, y1: y(ep.true_final), y2: y(ep.true_final), stroke: css("--green"), "stroke-width": 2.5, "stroke-dasharray": "7 4" }, svg);

    // Tracker as a step line.
    let d = `M${x(0)},${y(ep.tracker[0])}`;
    ep.tracker.forEach((s, i) => { if (i) d += `H${x(i)}V${y(s)}`; });
    d += `H${W - m.r}`;
    el("path", { d, fill: "none", stroke: css("--accent"), "stroke-width": 2.5, "stroke-linejoin": "round" }, svg);

    const changes = ep.argmax.reduce((c, a, i) => c + (i && a !== ep.argmax[i - 1] ? 1 : 0), 0);
    const above = ep.argmax.filter((a) => a > ep.true_final).length / ep.argmax.length;
    document.getElementById("trace-note").textContent =
      `Instance ${ep.instance}: reached stage ${ep.true_final}/${N} (q = ${fmt(ep.q)}); tracker ended at ${ep.tracker[ep.tracker.length - 1]}; ` +
      `prediction changed ${changes}× over ${ep.argmax.length} queries; ${Math.round(above * 100)}% of predictions above the stage actually reached.`;
  }

  fetch("static/data/stage_traces.json")
    .then((r) => r.json())
    .then((d) => {
      traces = d;
      const tabs = document.getElementById("trace-tabs");
      tabs.innerHTML = d.sorting_vegetables.episodes
        .map((e, i) => `<button role="tab" aria-selected="${i === 0}" data-i="${i}">${e.instance}</button>`)
        .join("");
      setupTabs(tabs, (b) => { traceIdx = +b.dataset.i; drawTrace(); });
      drawTrace();
    })
    .catch(() => {
      document.getElementById("trace-note").textContent = "Could not load stage traces (serve this page over HTTP, not file://).";
    });

  // ---------------------------------------------------------------- misc
  const redraw = () => { drawInstances(instKey); drawTrace(); };
  redraw();
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", redraw);

  document.getElementById("copy-bib").addEventListener("click", (e) => {
    navigator.clipboard.writeText(document.getElementById("bibtex").textContent).then(() => {
      e.target.textContent = "Copied";
      setTimeout(() => (e.target.textContent = "Copy"), 1500);
    });
  });
})();
