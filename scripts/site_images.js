// Resize and compress an image to WebP in headless Chromium (no image library needed).
// Usage: node scripts/site_images.js <input.png|jpg|webp> <output.webp> <max-width> [quality 0-1]
// Run with Playwright available (the cloud container's global install, or `npm i playwright`).
const path = require('path');
let pw;
try { pw = require('playwright'); } catch (e) { pw = require('/opt/node-tools/node_modules/playwright'); }
const fs = require('fs');

(async () => {
  const [inp, out, maxW, q] = process.argv.slice(2);
  const data = fs.readFileSync(inp).toString('base64');
  const mime = inp.endsWith('.png') ? 'image/png' : inp.endsWith('.webp') ? 'image/webp' : 'image/jpeg';
  const browser = await pw.chromium.launch();
  const page = await browser.newPage();
  const res = await page.evaluate(async ({ data, mime, maxW, q }) => {
    const img = new Image();
    img.src = `data:${mime};base64,${data}`;
    await img.decode();
    const s = Math.min(1, maxW / img.naturalWidth);
    const c = document.createElement('canvas');
    c.width = Math.round(img.naturalWidth * s);
    c.height = Math.round(img.naturalHeight * s);
    const ctx = c.getContext('2d');
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, 0, 0, c.width, c.height);
    return { url: c.toDataURL('image/webp', q), w: c.width, h: c.height };
  }, { data, mime, maxW: Number(maxW), q: Number(q || 0.82) });
  fs.writeFileSync(out, Buffer.from(res.url.split(',')[1], 'base64'));
  console.log(`${path.basename(out)}: ${res.w}x${res.h}, ${fs.statSync(out).size} bytes`);
  await browser.close();
})();
