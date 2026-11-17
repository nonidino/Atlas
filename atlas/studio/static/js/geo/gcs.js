// Dimensions and constraints, solved by planegcs (FreeCAD's 2-D solver, in
// WebAssembly; plan §3.3).  The drawing becomes a sketch: a rectangle is four corners
// joined by two horizontal and two vertical lines (4 degrees of freedom: x, y, w, h),
// an ellipse a centre and two radius points (4), a path its points (2 each) and its
// curve handles (2 each).  An arc's bulge, a corner radius and a channel's width are
// numbers the sketch does not move, so they are not counted.  Every field a formula drives
// (`bind`, a dimension) is a driving constraint; `constraints` adds fixes, coincidence,
// horizontal and vertical edges, parallel, perpendicular and equal lines, and
// distances.  The count of what can still move is planegcs's own (`dof`); zero is
// "fully defined".  A path whose points are constrained is solved again while one of
// its points is dragged, so the constraints hold as it moves.

let modPromise = null;

/** the solver, loaded once (484 KB of WebAssembly) */
export function solver(base = new URL('../../vendor/planegcs/dist/', import.meta.url).href) {
  if (!modPromise) {
    modPromise = import(base + 'index.js').then((m) => ({ make: () => m.make_gcs_wrapper(base + 'planegcs_dist/planegcs.wasm') }));
  }
  return modPromise;
}

const P = (id, x, y, fixed = false) => ({ id, type: 'point', x, y, fixed });

/** the sketch of a drawing: primitives and constraints for planegcs; and notes */
export function sketch(geom, only = null) {
  const prims = [];
  const notes = [];
  const binds = (f, k) => f.bind && f.bind[k] !== undefined;
  const fixed = new Set((geom.constraints || []).filter((c) => c.type === 'fix').flatMap((c) => c.refs));
  for (const f of geom.features) {
    if (only && f.id !== only) continue;
    const id = f.id;
    if (f.kind === 'rect') {
      const { x, y, w, h } = f;
      prims.push(P(`${id}:c0`, x, y), P(`${id}:c1`, x + w, y), P(`${id}:c2`, x + w, y + h), P(`${id}:c3`, x, y + h));
      for (let k = 0; k < 4; k++) prims.push({ id: `${id}:e${k}`, type: 'line', p1_id: `${id}:c${k}`, p2_id: `${id}:c${(k + 1) % 4}` });
      prims.push({ id: `${id}:h0`, type: 'horizontal_l', l_id: `${id}:e0` }, { id: `${id}:v1`, type: 'vertical_l', l_id: `${id}:e1` },
        { id: `${id}:h2`, type: 'horizontal_l', l_id: `${id}:e2` }, { id: `${id}:v3`, type: 'vertical_l', l_id: `${id}:e3` });
      if (binds(f, 'w')) prims.push({ id: `${id}:dw`, type: 'difference', param1: { o_id: `${id}:c0`, prop: 'x' }, param2: { o_id: `${id}:c1`, prop: 'x' }, difference: w });
      if (binds(f, 'h')) prims.push({ id: `${id}:dh`, type: 'difference', param1: { o_id: `${id}:c0`, prop: 'y' }, param2: { o_id: `${id}:c3`, prop: 'y' }, difference: h });
      if (binds(f, 'x') || fixed.has(id)) prims.push({ id: `${id}:fx`, type: 'coordinate_x', p_id: `${id}:c0`, x });
      if (binds(f, 'y') || fixed.has(id)) prims.push({ id: `${id}:fy`, type: 'coordinate_y', p_id: `${id}:c0`, y });
    } else if (f.kind === 'ellipse') {
      prims.push(P(`${id}:c`, f.cx, f.cy), P(`${id}:rx`, f.cx + f.rx, f.cy), P(`${id}:ry`, f.cx, f.cy + f.ry));
      prims.push({ id: `${id}:hx`, type: 'horizontal_pp', p1_id: `${id}:c`, p2_id: `${id}:rx` },
        { id: `${id}:vy`, type: 'vertical_pp', p1_id: `${id}:c`, p2_id: `${id}:ry` });
      if (binds(f, 'rx')) prims.push({ id: `${id}:drx`, type: 'difference', param1: { o_id: `${id}:c`, prop: 'x' }, param2: { o_id: `${id}:rx`, prop: 'x' }, difference: f.rx });
      if (binds(f, 'ry')) prims.push({ id: `${id}:dry`, type: 'difference', param1: { o_id: `${id}:c`, prop: 'y' }, param2: { o_id: `${id}:ry`, prop: 'y' }, difference: f.ry });
      if (binds(f, 'cx') || fixed.has(id)) prims.push({ id: `${id}:fx`, type: 'coordinate_x', p_id: `${id}:c`, x: f.cx });
      if (binds(f, 'cy') || fixed.has(id)) prims.push({ id: `${id}:fy`, type: 'coordinate_y', p_id: `${id}:c`, y: f.cy });
    } else if (f.kind === 'path' || f.kind === 'channel') {
      const n = f.points.length;
      const fix = fixed.has(id);
      f.points.forEach(([x, y], k) => prims.push(P(`${id}:p${k}`, x, y, fix)));
      const m = f.closed ? n : n - 1;
      for (let k = 0; k < m; k++) {
        const kind = (f.edges || [])[k] || 'line';
        if (kind === 'line') prims.push({ id: `${id}:e${k}`, type: 'line', p1_id: `${id}:p${k}`, p2_id: `${id}:p${(k + 1) % n}` });
        else if (kind === 'cubic' && f.handles[k]) {
          const hd = f.handles[k];
          prims.push(P(`${id}:h${k}a`, hd[0], hd[1], fix), P(`${id}:h${k}b`, hd[2], hd[3], fix));
        }
      }
    } else if (f.kind === 'scatter') {
      // a generator: its box is four numbers; its inclusions follow its seed
      prims.push(P(`${id}:c0`, f.x, f.y, fixed.has(id)), P(`${id}:c2`, f.x + f.w, f.y + f.h, fixed.has(id)));
    }
  }
  for (const c of geom.constraints || []) {
    if (only && !c.refs.every((r) => r.startsWith(`${only}:`))) continue;
    // a dimension between two shapes is held by the binding it wrote; it is drawn only
    if (c.type === 'distance' && c.param && c.value == null) continue;
    const r = c.refs;
    if (c.type === 'coincident' && r.length === 2) prims.push({ id: c.id, type: 'p2p_coincident', p1_id: r[0], p2_id: r[1] });
    else if (c.type === 'horizontal' && r.length === 1) prims.push({ id: c.id, type: 'horizontal_l', l_id: r[0] });
    else if (c.type === 'vertical' && r.length === 1) prims.push({ id: c.id, type: 'vertical_l', l_id: r[0] });
    else if (c.type === 'parallel' && r.length === 2) prims.push({ id: c.id, type: 'parallel', l1_id: r[0], l2_id: r[1] });
    else if (c.type === 'perpendicular' && r.length === 2) prims.push({ id: c.id, type: 'perpendicular_ll', l1_id: r[0], l2_id: r[1] });
    else if (c.type === 'equal' && r.length === 2) prims.push({ id: c.id, type: 'equal_length', l1_id: r[0], l2_id: r[1] });
    else if (c.type === 'distance' && r.length === 2 && c.value > 0) prims.push({ id: c.id, type: 'p2p_distance', p1_id: r[0], p2_id: r[1], distance: c.value });
    else if (c.type !== 'fix') notes.push(`${c.type} on ${r.join(', ')} is not solved here`);
  }
  // only references that exist: a constraint on a deleted shape is dropped, and said
  const ids = new Set(prims.filter((p) => p.id).map((p) => p.id));
  const refsOk = (p) => ['p1_id', 'p2_id', 'l_id', 'l1_id', 'l2_id', 'p_id'].every((k) => !(k in p) || ids.has(p[k]));
  const kept = prims.filter((p) => { const ok = refsOk(p); if (!ok) notes.push(`${p.id} refers to something gone`); return ok; });
  return { prims: kept, notes };
}

function load(wrapper, prims) {
  wrapper.push_primitives_and_params(prims);
}

/** the drawing's degrees of freedom: {dof, status, notes}; 0 is "fully defined" */
export async function freedom(geom, mod = null) {
  const m = mod || await solver();
  const { prims, notes } = sketch(geom);
  if (!prims.length) return { dof: 0, status: 'empty', notes };
  const w = await m.make();
  try {
    load(w, prims);
    const st = w.solve();
    const dof = w.gcs.dof();
    return { dof, status: st, conflicting: w.has_gcs_conflicting_constraints(), redundant: w.has_gcs_redundant_constraints(), notes };
  } finally {
    w.destroy_gcs_module();
  }
}

/** a path's points after one of them moves to (x, y), its constraints kept; null when
 * the path has none to keep or the solver cannot satisfy them */
export async function dragPoint(geom, f, k, x, y, mod = null) {
  const own = (geom.constraints || []).filter((c) => c.refs.length && c.refs.every((r) => r.startsWith(`${f.id}:`)));
  if (!own.length) return null;
  const m = mod || await solver();
  const g = { features: [f], constraints: own };
  const { prims } = sketch(g, f.id);
  prims.push({ id: 'drag:x', type: 'coordinate_x', p_id: `${f.id}:p${k}`, x, temporary: true },
    { id: 'drag:y', type: 'coordinate_y', p_id: `${f.id}:p${k}`, y, temporary: true });
  const w = await m.make();
  try {
    load(w, prims);
    const st = w.solve();
    if (st !== 0 && st !== 1) return null;
    w.apply_solution();
    const out = f.points.map((p) => p.slice());
    for (const p of w.sketch_index.get_primitives()) {
      const mm = /^(.+):p(\d+)$/.exec(p.id || '');
      if (p.type === 'point' && mm && mm[1] === f.id) out[+mm[2]] = [p.x, p.y];
    }
    return out;
  } finally {
    w.destroy_gcs_module();
  }
}
