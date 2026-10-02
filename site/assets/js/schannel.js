// "Cut. Solve. Stitch." drawn from the workbench's real S-channel case
// (data/s-channel.json, exported by scripts/site_export_schannel.py): its cells,
// the four windows its layout generated from its shape, the seams where they
// overlap, and the temperature field solved by the decomposed run.
(function () {
  "use strict";
  var canvas = document.getElementById("schannel");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var D = null, img = null, off = null, stage = "cut", t0 = 0, raf = 0;
  var SCALE = 4;
  // window tints (light) and the temperature ramp: one hue, light to dark
  var TINT = [[186, 230, 245], [214, 236, 205], [252, 224, 196], [226, 214, 245]];
  var RAMP = [[255, 243, 232], [244, 160, 104], [176, 52, 20], [110, 22, 8]];
  var CYAN = [25, 195, 240];

  function ramp(q) {
    var x = Math.max(0, Math.min(1, q / 1000)) * (RAMP.length - 1);
    var i = Math.min(RAMP.length - 2, Math.floor(x)), f = x - i, a = RAMP[i], b = RAMP[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  }
  function popcount(b) { var n = 0; while (b) { n += b & 1; b >>= 1; } return n; }
  function firstWin(b) { for (var k = 0; k < 8; k++) if (b & (1 << k)) return k; return -1; }

  function draw(now) {
    if (!D) return;
    var nx = D.grid.nx, ny = D.grid.ny, q = D.field_q, bits = D.window_bits, nw = D.windows.length;
    var el = (now - t0) / 1000;
    var data = img.data;
    var fillUpTo = stage === "solve" ? (reduced ? nw : Math.min(nw, el / 0.55)) : (stage === "cut" ? -1 : nw);
    var pulse = 0.5 + 0.5 * Math.sin(el * 3.2);
    for (var j = 0; j < ny; j++) {
      for (var i = 0; i < nx; i++) {
        var c = j * nx + i, o = ((ny - 1 - j) * nx + i) * 4, v = q[c];
        if (v < 0) { data[o] = 255; data[o + 1] = 255; data[o + 2] = 255; data[o + 3] = 0; continue; }
        var b = bits[c], k = firstWin(b), col;
        var filled = k >= 0 && k < fillUpTo;
        if (stage === "cut" || !filled) {
          col = k >= 0 ? TINT[k % TINT.length] : [233, 236, 240];
          if (stage === "solve" && k >= 0 && k < fillUpTo + 1 && !filled) {
            var f = Math.max(0, fillUpTo - k); var r = ramp(v);
            col = [col[0] + (r[0] - col[0]) * f, col[1] + (r[1] - col[1]) * f, col[2] + (r[2] - col[2]) * f];
          }
        } else {
          col = ramp(v);
        }
        var seam = popcount(b) >= 2;
        if (seam && stage !== "check-off") {
          var a = stage === "cut" ? 0.55 : stage === "solve" ? 0.25 : stage === "stitch" ? (reduced ? 0.55 : 0.35 + 0.45 * pulse) : 0.0;
          col = [col[0] + (CYAN[0] - col[0]) * a, col[1] + (CYAN[1] - col[1]) * a, col[2] + (CYAN[2] - col[2]) * a];
        }
        data[o] = col[0]; data[o + 1] = col[1]; data[o + 2] = col[2]; data[o + 3] = 255;
      }
    }
    var octx = off.getContext("2d");
    octx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, nx * SCALE, ny * SCALE);
    var busy = !reduced && (stage === "stitch" || (stage === "solve" && el < nw * 0.55 + 0.6));
    if (busy) raf = requestAnimationFrame(draw); else raf = 0;
  }

  function go(name) {
    stage = name || "cut";
    t0 = performance.now();
    if (!raf) raf = requestAnimationFrame(draw);
  }

  document.addEventListener("atlas:step", function (e) { go(e.detail.name); });

  fetch(canvas.dataset.src).then(function (r) { return r.json(); }).then(function (d) {
    D = d;
    canvas.width = d.grid.nx * SCALE;
    canvas.height = d.grid.ny * SCALE;
    off = document.createElement("canvas");
    off.width = d.grid.nx; off.height = d.grid.ny;
    img = off.getContext("2d").createImageData(d.grid.nx, d.grid.ny);
    var active = document.querySelector(".step.active");
    go(active ? active.dataset.step : "cut");
  }).catch(function () {
    var p = document.createElement("p");
    p.className = "fine";
    p.textContent = "The drawing needs the page to be served over http (it loads data/s-channel.json).";
    canvas.parentNode.appendChild(p);
  });
})();
