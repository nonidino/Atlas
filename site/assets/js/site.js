// Atlas proposal site: reveals, count-ups, the sticky steps, tabs and copy buttons.
// Everything here is progressive: without the script every section is visible and
// every number is already written in the page.
(function () {
  "use strict";
  var root = document.documentElement;
  root.classList.remove("no-js");
  root.classList.add("js");
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (reduced) root.classList.add("reduced");

  function onView(els, cb, opts) {
    if (!("IntersectionObserver" in window)) { els.forEach(function (e) { cb(e, true); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { cb(en.target, en.isIntersecting, en); });
    }, opts || { rootMargin: "0px 0px -12% 0px", threshold: 0.12 });
    els.forEach(function (e) { io.observe(e); });
    return io;
  }
  var $$ = function (sel, ctx) { return Array.prototype.slice.call((ctx || document).querySelectorAll(sel)); };

  // reveals: once
  onView($$(".reveal, .scenario, .stage-n"), function (el, inView) {
    if (inView) el.classList.add("in");
  });

  // count-ups: the number counts from zero once, then the page's own text is restored
  function countUp(el) {
    if (el.dataset.counted) return;
    el.dataset.counted = "1";
    var html = el.innerHTML;
    var text = el.textContent;
    var m = text.match(/^([0-9][0-9,]*)(\.(\d+))?/);
    if (!m || reduced) return;
    var target = parseFloat(m[0].replace(/,/g, ""));
    var dec = m[3] ? m[3].length : 0;
    var rest = text.slice(m[0].length);
    var t0 = null, dur = 1300;
    function fmt(v) {
      var s = v.toFixed(dec);
      if (m[1].indexOf(",") >= 0) s = Number(s).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec });
      return s;
    }
    function frame(t) {
      if (!t0) t0 = t;
      var p = Math.min(1, (t - t0) / dur);
      var e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * e) + rest;
      if (p < 1) requestAnimationFrame(frame); else el.innerHTML = html;
    }
    requestAnimationFrame(frame);
  }
  onView($$(".count"), function (el, inView) { if (inView) countUp(el); }, { threshold: 0.6 });

  // the sticky steps: the step nearest the middle of the screen is active
  var steps = $$(".step");
  if (steps.length) {
    onView(steps, function (el, inView) {
      if (!inView) return;
      steps.forEach(function (s) { s.classList.toggle("active", s === el); });
      document.dispatchEvent(new CustomEvent("atlas:step", { detail: { name: el.dataset.step } }));
    }, { rootMargin: "-45% 0px -45% 0px", threshold: 0 });
    steps[0].classList.add("active");
  }

  // the proofs' dependency graph: nodes fill in dependency order as it scrolls in
  var graph = document.querySelector("#dep-graph");
  if (graph) {
    var nodes = $$(".dep-node", graph).sort(function (a, b) { return (+a.dataset.layer) - (+b.dataset.layer); });
    if (reduced) nodes.forEach(function (n) { n.classList.add("on"); });
    else onView([graph], function (el, inView) {
      if (!inView || el.dataset.done) return;
      el.dataset.done = "1";
      nodes.forEach(function (n, i) { setTimeout(function () { n.classList.add("on"); }, 40 * i + 120 * (+n.dataset.layer)); });
    }, { threshold: 0.25 });
  }

  // tabs (install commands)
  $$("[role=tablist]").forEach(function (list) {
    var tabs = $$("[role=tab]", list);
    function select(tab) {
      tabs.forEach(function (t) {
        var on = t === tab;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        var panel = document.getElementById(t.getAttribute("aria-controls"));
        if (panel) panel.hidden = !on;
      });
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () { select(t); });
      t.addEventListener("keydown", function (e) {
        if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          var j = (i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
          tabs[j].focus(); select(tabs[j]);
        }
      });
    });
    var guess = /Mac/.test(navigator.platform) ? "mac" : /Linux/.test(navigator.platform) ? "linux" : "win";
    var pick = tabs.filter(function (t) { return t.dataset.os === guess; })[0] || tabs[0];
    select(pick);
  });

  // copy buttons
  $$(".copy").forEach(function (b) {
    b.addEventListener("click", function () {
      var pre = b.parentNode.querySelector("pre");
      var txt = pre ? pre.textContent : "";
      var done = function () { var o = b.textContent; b.textContent = "Copied"; setTimeout(function () { b.textContent = o; }, 1600); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(done, function () {});
    });
  });

  // vision: show the port types on tap
  $$(".ports-btn").forEach(function (b) {
    b.addEventListener("click", function () {
      var f = b.closest(".frame");
      var on = f.classList.toggle("show-ports");
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
  });
})();
