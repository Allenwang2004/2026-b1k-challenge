(() => {
  const NS = "http://www.w3.org/2000/svg";
  // Colors are CSS custom properties so the charts follow the page theme without a redraw.
  const v = (name) => `var(${name})`;
  const el = (tag, attrs = {}, parent) => {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  const fmt = (x) => (x == null ? "–" : x.toFixed(3));

  function setupTabs(container, onSelect) {
    const buttons = [...container.querySelectorAll("button")];
    buttons.forEach((b) =>
      b.addEventListener("click", () => {
        buttons.forEach((o) => o.setAttribute("aria-selected", o === b ? "true" : "false"));
        onSelect(b);
      })
    );
  }

  // ---------------------------------------------------------------- teaser: predicted stage over the clip
  // Raw stage-head argmax from the logged rollout, as [simulator step, predicted stage] change points.
  // The clip runs at 4x and holds its last frame; the episode ended at step 2572 when the third can went in,
  // so stage 3 (task complete) is the simulator's verdict, not a prediction.
  const TEASER = { speed: 4, fps: 30, endStep: 2572, doneStage: 3, changes: [[0, 0], [1760, 1], [2200, 2]] };
  (function teaser() {
    const video = document.getElementById("teaser-video");
    if (!video) return;
    const cells = [...document.querySelectorAll("#stage-cells li")];
    const segs = document.getElementById("stage-segs");
    const head = document.getElementById("stage-playhead");
    TEASER.changes.forEach(([step, stage], i) => {
      const next = i + 1 < TEASER.changes.length ? TEASER.changes[i + 1][0] : TEASER.endStep;
      const seg = document.createElement("div");
      seg.className = `seg s${stage}`;
      seg.style.flex = String(next - step);
      segs.appendChild(seg);
    });

    let shown = -1;
    const render = () => {
      const step = video.currentTime * TEASER.speed * TEASER.fps;
      let stage = 0;
      for (const [s, st] of TEASER.changes) if (step >= s) stage = st;
      if (step >= TEASER.endStep - 1) stage = TEASER.doneStage;
      if (stage !== shown) {
        cells.forEach((c, i) => {
          c.classList.toggle("active", i === stage);
          c.classList.toggle("past", i < stage);
        });
        shown = stage;
      }
      head.style.left = `${Math.min(step / TEASER.endStep, 1) * 100}%`;
    };
    const tick = () => { render(); if (!video.paused) requestAnimationFrame(tick); };
    video.addEventListener("play", tick);
    video.addEventListener("timeupdate", render);
    video.addEventListener("seeked", render);
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
      video.removeAttribute("autoplay");
      video.pause();
      video.controls = true;
    }
    render();
  })();

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
      el("line", { x1: m.l, x2: W - m.r, y1: y(s), y2: y(s), style: `stroke:${v("--border")}`, "stroke-width": s % 2 ? 0.5 : 1 }, svg);
      if (s % 2 === 0 || s === N)
        el("text", { x: m.l - 8, y: y(s) + 4, "text-anchor": "end", "font-size": 12, style: `fill:${v("--fg-faint")}` }, svg).textContent = s;
    }
    for (let t = 0; t <= T; t += 60) {
      const px = m.l + (iw * t) / T;
      el("text", { x: px, y: H - 20, "text-anchor": "middle", "font-size": 12, style: `fill:${v("--fg-faint")}` }, svg).textContent = `${t / 60} min`;
    }
    el("text", { x: 12, y: m.t + ih / 2, transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle", "font-size": 12, style: `fill:${v("--fg-muted")}` }, svg).textContent = "stage";

    // Argmax: one dot per query, slightly transparent so dense runs read as bands.
    const g = el("g", { style: `fill:${v("--blue")}`, "fill-opacity": 0.35 }, svg);
    ep.argmax.forEach((a, i) => el("circle", { cx: x(i), cy: y(a), r: 1.8 }, g));

    // Ground truth: the stage the episode ended in. Monotone labels make this the maximum reached.
    el("line", { x1: m.l, x2: W - m.r, y1: y(ep.true_final), y2: y(ep.true_final), style: `stroke:${v("--green")}`, "stroke-width": 2.5, "stroke-dasharray": "7 4" }, svg);

    // Tracker as a step line.
    let d = `M${x(0)},${y(ep.tracker[0])}`;
    ep.tracker.forEach((s, i) => { if (i) d += `H${x(i)}V${y(s)}`; });
    d += `H${W - m.r}`;
    el("path", { d, fill: "none", style: `stroke:${v("--accent")}`, "stroke-width": 2.5, "stroke-linejoin": "round" }, svg);

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

  // ---------------------------------------------------------------- BDDL dataset: charts + task explorer
  const tipFor = (host) => {
    const tip = document.createElement("div");
    tip.className = "chart-tip";
    tip.hidden = true;
    host.appendChild(tip);
    return {
      show(html, evt) {
        tip.innerHTML = html;
        tip.hidden = false;
        const r = host.getBoundingClientRect();
        const x = Math.min(evt.clientX - r.left + 12, r.width - tip.offsetWidth - 4);
        tip.style.left = `${Math.max(0, x)}px`;
        tip.style.top = `${evt.clientY - r.top - tip.offsetHeight - 10}px`;
      },
      hide() { tip.hidden = true; },
    };
  };
  // Bar with a 4px rounded data end and a square baseline.
  const colPath = (x, y, w, h) => {
    const r = Math.min(4, w / 2, h);
    return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
  };
  const rowPath = (x, y, w, h) => {
    const r = Math.min(4, h / 2, w);
    return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
  };
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  function drawLiteralHist(tasks) {
    const host = document.getElementById("lit-hist");
    const maxL = Math.max(...tasks.map((t) => t.literals));
    const bins = Array.from({ length: maxL }, (_, i) => tasks.filter((t) => t.literals === i + 1));
    const maxN = Math.max(...bins.map((b) => b.length));
    const W = 440, H = 220, m = { l: 30, r: 6, t: 10, b: 30 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const slot = iw / maxL, bw = Math.min(24, slot - 2);
    const yMax = Math.ceil(maxN / 5) * 5;
    const y = (n) => m.t + ih * (1 - n / yMax);
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Histogram of goal literals per task" }, host);
    for (let n = 0; n <= yMax; n += 5) {
      el("line", { x1: m.l, x2: W - m.r, y1: y(n), y2: y(n), style: `stroke:${v("--border")}`, "stroke-width": 1 }, svg);
      el("text", { x: m.l - 6, y: y(n) + 4, "text-anchor": "end", "font-size": 11, style: `fill:${v("--fg-faint")}` }, svg).textContent = n;
    }
    [1, 5, 10, 15, 20, 25].filter((k) => k <= maxL).forEach((k) => {
      el("text", { x: m.l + slot * (k - 0.5), y: H - 12, "text-anchor": "middle", "font-size": 11, style: `fill:${v("--fg-faint")}` }, svg).textContent = k;
    });
    el("text", { x: m.l + iw / 2, y: H - 0.5, "text-anchor": "middle", "font-size": 11, style: `fill:${v("--fg-muted")}` }, svg).textContent = "goal literals";
    const tip = tipFor(host);
    bins.forEach((b, i) => {
      const x0 = m.l + slot * i;
      if (b.length) el("path", { d: colPath(x0 + (slot - bw) / 2, y(b.length), bw, y(0) - y(b.length)), style: `fill:${v("--accent")}` }, svg);
      const hit = el("rect", { x: x0, y: m.t, width: slot, height: ih, fill: "transparent" }, svg);
      const names = b.map((t) => t.name);
      const html = `<b>${i + 1} literal${i ? "s" : ""}</b>: ${b.length} task${b.length === 1 ? "" : "s"}` +
        (b.length ? `<br>${esc(names.slice(0, 4).join(", "))}${names.length > 4 ? `, +${names.length - 4} more` : ""}` : "");
      hit.addEventListener("mousemove", (e) => tip.show(html, e));
      hit.addEventListener("mouseleave", () => tip.hide());
    });
  }

  function drawPredicates(tasks) {
    const host = document.getElementById("pred-bars");
    const lit = {}, used = {};
    tasks.forEach((t) => Object.entries(t.predicates).forEach(([p, n]) => { lit[p] = (lit[p] || 0) + n; used[p] = (used[p] || 0) + 1; }));
    const rows = Object.keys(lit).sort((a, b) => lit[b] - lit[a]);
    const total = rows.reduce((a, p) => a + lit[p], 0);
    const rowH = 22, bh = 14;
    const W = 440, m = { l: 86, r: 40, t: 4, b: 4 };
    const H = m.t + m.b + rowH * rows.length;
    const iw = W - m.l - m.r, maxV = lit[rows[0]];
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Goal literals by predicate" }, host);
    el("line", { x1: m.l, x2: m.l, y1: m.t, y2: H - m.b, style: `stroke:${v("--border")}`, "stroke-width": 1 }, svg);
    const tip = tipFor(host);
    rows.forEach((p, i) => {
      const yy = m.t + rowH * i, w = Math.max((iw * lit[p]) / maxV, 2);
      el("text", { x: m.l - 8, y: yy + rowH / 2 + 4, "text-anchor": "end", "font-size": 11.5, style: `fill:${v("--fg-muted")};font-family:${v("--mono")}` }, svg).textContent = p;
      el("path", { d: rowPath(m.l, yy + (rowH - bh) / 2, w, bh), style: `fill:${v("--accent")}` }, svg);
      el("text", { x: m.l + w + 6, y: yy + rowH / 2 + 4, "font-size": 11.5, style: `fill:${v("--fg")}` }, svg).textContent = lit[p];
      const hit = el("rect", { x: 0, y: yy, width: W, height: rowH, fill: "transparent" }, svg);
      const html = `<b>${esc(p)}</b>: ${lit[p]} literals (${((100 * lit[p]) / total).toFixed(0)}%)<br>used by ${used[p]} task${used[p] === 1 ? "" : "s"}`;
      hit.addEventListener("mousemove", (e) => tip.show(html, e));
      hit.addEventListener("mouseleave", () => tip.hide());
    });
  }

  function setupExplorer(tasks) {
    const body = document.getElementById("task-rows");
    const search = document.getElementById("task-search");
    const count = document.getElementById("task-count");
    const buttons = [...document.querySelectorAll("#task-explorer thead button")];
    const maxL = Math.max(...tasks.map((t) => t.literals));
    let sortKey = "id", asc = true;
    const open = new Set();

    const detail = (t) => {
      const chips = Object.entries(t.predicates).map(([p, n]) => `<span class="chip">${esc(p)} × ${n}</span>`).join("");
      const note = t.options > 1 ? `<span class="chip">one of ${t.options} solution options</span>` : "";
      return `<tr class="detail"><td colspan="6"><div class="chips">${chips}${note}</div>
        <ul class="literals">${t.goal.map((g) => `<li>${esc(g)}</li>`).join("")}</ul></td></tr>`;
    };
    const render = () => {
      const q = search.value.trim().toLowerCase();
      const list = tasks
        .filter((t) => !q || t.name.toLowerCase().includes(q) || Object.keys(t.predicates).some((p) => p.includes(q)))
        .sort((a, b) => {
          const d = typeof a[sortKey] === "string" ? a[sortKey].localeCompare(b[sortKey]) : a[sortKey] - b[sortKey];
          return asc ? d : -d;
        });
      body.innerHTML = list.map((t) => {
        const isOpen = open.has(t.id);
        return `<tr class="row" tabindex="0" data-id="${t.id}" aria-expanded="${isOpen}">
          <td class="idx">${t.id}</td>
          <td class="name"><code>${esc(t.name)}</code></td>
          <td class="num"><span class="bar-mini" style="width:${Math.round((48 * t.literals) / maxL)}px"></span>${t.literals}</td>
          <td class="num">${t.options}</td>
          <td class="num">${t.minutes.toFixed(1)} min</td>
          <td class="num">${Math.round(t.monotone_pct)}%</td>
        </tr>${isOpen ? detail(t) : ""}`;
      }).join("");
      count.textContent = `${list.length} of ${tasks.length} tasks`;
      buttons.forEach((b) => {
        if (b.dataset.sort === sortKey) b.setAttribute("aria-sort", asc ? "ascending" : "descending");
        else b.removeAttribute("aria-sort");
      });
    };
    const toggle = (tr) => {
      const id = +tr.dataset.id;
      open.has(id) ? open.delete(id) : open.add(id);
      render();
      body.querySelector(`tr.row[data-id="${id}"]`)?.focus();
    };
    body.addEventListener("click", (e) => { const tr = e.target.closest("tr.row"); if (tr) toggle(tr); });
    body.addEventListener("keydown", (e) => {
      const tr = e.target.closest("tr.row");
      if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggle(tr); }
    });
    buttons.forEach((b) => b.addEventListener("click", () => {
      if (sortKey === b.dataset.sort) asc = !asc;
      else { sortKey = b.dataset.sort; asc = sortKey === "id" || sortKey === "name"; }
      render();
    }));
    search.addEventListener("input", render);
    render();
  }

  fetch("static/data/bddl_tasks.json")
    .then((r) => r.json())
    .then((d) => {
      drawLiteralHist(d.tasks);
      drawPredicates(d.tasks);
      setupExplorer(d.tasks);
    })
    .catch(() => {
      document.getElementById("task-count").textContent = "Could not load the task table (serve this page over HTTP, not file://).";
    });
})();
