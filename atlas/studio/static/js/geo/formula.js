// The formula language of atlas/studio/formula.py, for the page's previews: numbers,
// names, + - * / ^ (also **, ×, ÷, −), unary minus, brackets, pi and e, and the
// functions exp log sqrt abs sin cos tan tanh min max.  Parsed into a tree and
// walked; nothing is evaluated as code.  A test runs this file in Node beside the
// server's parser and checks they agree.

const FUNCTIONS = {
  exp: [1, Math.exp], log: [1, Math.log], sqrt: [1, Math.sqrt], abs: [1, Math.abs],
  sin: [1, Math.sin], cos: [1, Math.cos], tan: [1, Math.tan], tanh: [1, Math.tanh],
  min: [2, Math.min], max: [2, Math.max],
};
const CONSTANTS = { pi: Math.PI, e: Math.E };
const TOKEN = /\s*(?:(\d+\.\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)?)|([A-Za-z_][A-Za-z0-9_]*)|(\*\*|[-+*/^(),]))/y;

export class FormulaError extends Error {}

export function tokens(text) {
  const t = text.replace(/−/g, '-').replace(/×/g, '*').replace(/÷/g, '/').replace(/·/g, '*');
  const out = [];
  let pos = 0;
  while (pos < t.length) {
    if (!t.slice(pos).trim()) break;
    TOKEN.lastIndex = pos;
    const m = TOKEN.exec(t);
    if (!m || TOKEN.lastIndex === pos) throw new FormulaError(`cannot read '${t.slice(pos).trim().slice(0, 12)}'`);
    if (m[1]) out.push(['num', m[1]]);
    else if (m[2]) out.push(['name', m[2]]);
    else out.push(['op', m[3] === '**' ? '^' : m[3]]);
    pos = TOKEN.lastIndex;
  }
  return out;
}

export function parse(text) {
  if (typeof text !== 'string' || !text.trim()) throw new FormulaError('an empty formula');
  const t = tokens(text);
  let i = 0;
  const peek = () => t[i] || ['end', ''];
  const is = (k, v) => peek()[0] === k && peek()[1] === v;
  const take = (k, v) => {
    const tok = peek();
    if ((k && tok[0] !== k) || (v && tok[1] !== v)) throw new FormulaError(`expected '${v || k}', found '${tok[1] || 'the end'}'`);
    i++;
    return tok;
  };
  const expr = () => {
    let n = term();
    while (is('op', '+') || is('op', '-')) { const op = take()[1]; n = [op, n, term()]; }
    return n;
  };
  const term = () => {
    let n = unary();
    while (is('op', '*') || is('op', '/')) { const op = take()[1]; n = [op, n, unary()]; }
    return n;
  };
  const unary = () => {
    if (is('op', '-')) { take(); return ['neg', unary()]; }
    if (is('op', '+')) { take(); return unary(); }
    return power();
  };
  const power = () => {
    const base = atom();
    if (is('op', '^')) { take(); return ['^', base, unary()]; }
    return base;
  };
  const atom = () => {
    const [k, v] = peek();
    if (k === 'num') { take(); return ['num', parseFloat(v)]; }
    if (k === 'name') {
      take();
      if (is('op', '(')) {
        if (!FUNCTIONS[v]) throw new FormulaError(`no function '${v}'`);
        take('op', '(');
        const args = [expr()];
        while (is('op', ',')) { take(); args.push(expr()); }
        take('op', ')');
        if (args.length !== FUNCTIONS[v][0]) throw new FormulaError(`${v} takes ${FUNCTIONS[v][0]} argument${FUNCTIONS[v][0] === 1 ? '' : 's'}`);
        return ['call', v, args];
      }
      return ['name', v];
    }
    if (k === 'op' && v === '(') { take(); const n = expr(); take('op', ')'); return n; }
    throw new FormulaError(`unexpected '${v || 'end'}'`);
  };
  const node = expr();
  if (peek()[0] !== 'end') throw new FormulaError(`unexpected '${peek()[1]}'`);
  return node;
}

export function names(node) {
  if (node[0] === 'name') return new Set([node[1]]);
  if (node[0] === 'num') return new Set();
  if (node[0] === 'call') return new Set(node[2].flatMap((a) => [...names(a)]));
  if (node[0] === 'neg') return names(node[1]);
  return new Set([...names(node[1]), ...names(node[2])]);
}

export function evaluate(node, env) {
  switch (node[0]) {
    case 'num': return node[1];
    case 'name':
      if (node[1] in env) return env[node[1]];
      if (node[1] in CONSTANTS) return CONSTANTS[node[1]];
      throw new FormulaError(`'${node[1]}' is not defined here`);
    case 'neg': return -evaluate(node[1], env);
    case 'call': return FUNCTIONS[node[1]][1](...node[2].map((a) => evaluate(a, env)));
    case '+': return evaluate(node[1], env) + evaluate(node[2], env);
    case '-': return evaluate(node[1], env) - evaluate(node[2], env);
    case '*': return evaluate(node[1], env) * evaluate(node[2], env);
    case '/': return evaluate(node[1], env) / evaluate(node[2], env);
    case '^': return Math.pow(evaluate(node[1], env), evaluate(node[2], env));
    default: throw new FormulaError(`unknown operation '${node[0]}'`);
  }
}

export function value(text, env) {
  const v = evaluate(parse(text), env);
  if (!Number.isFinite(v)) throw new FormulaError(`'${text}' is not a finite number here`);
  return v;
}

/** parameters' values, formulas after what they read; a circle is refused */
export function resolveParameters(params) {
  const defs = Object.fromEntries(params.map((p) => [p.name, p]));
  const out = {};
  const visiting = [];
  const get = (n) => {
    if (n in out) return out[n];
    if (visiting.includes(n)) throw new FormulaError(`the parameters depend on each other in a circle: ${[...visiting.slice(visiting.indexOf(n)), n].join(' -> ')}`);
    const p = defs[n];
    if (!p.expr) { out[n] = Number(p.value); return out[n]; }
    visiting.push(n);
    const tree = parse(p.expr);
    const env = {};
    for (const m of names(tree)) if (m in defs) env[m] = get(m);
    out[n] = value(p.expr, env);
    visiting.pop();
    return out[n];
  };
  for (const n of Object.keys(defs)) get(n);
  return out;
}
