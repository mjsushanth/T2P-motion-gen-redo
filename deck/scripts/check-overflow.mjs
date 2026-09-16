// Programmatic overflow check for every slide, per
// MJS_LIB_TOOLS/codex-skills/build-project-deck/references/rendering-and-qa.md section 1.
//
// A screenshot cannot detect this: a fixed-canvas slide that clips overflowing content
// (`.slidev-slide-container` here) makes anything past its true bottom edge simply
// absent -- no scrollbar, no visual artifact, nothing "looks wrong". The only real check
// is comparing the deepest-rendered element's bottom edge against the canvas's actual
// measured height, in pixels, not against a guessed buffer.
//
// Usage:
//   node scripts/check-overflow.mjs [baseUrl] [slideCount]
//   node scripts/check-overflow.mjs http://localhost:3030 13
//   node scripts/check-overflow.mjs https://mjsushanth.github.io/T2P-motion-gen-redo 13
//
// Exits non-zero if any slide overflows (overflowPx > 0), so it can gate CI later.

import { chromium } from 'playwright';

const baseUrl = process.argv[2] || 'http://localhost:3030';
const slideCount = parseInt(process.argv[3] || '13', 10);

const measure = () => {
  const canvas = document.querySelector('.slidev-slide-container');
  if (!canvas) return { error: 'no .slidev-slide-container found' };
  const canvasRect = canvas.getBoundingClientRect();
  const trueHeight = canvasRect.height;

  let deepestBottom = 0;
  let deepestEl = null;
  for (const el of canvas.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (getComputedStyle(el).display === 'none') continue;
    const relativeBottom = r.bottom - canvasRect.top;
    if (relativeBottom > deepestBottom) {
      deepestBottom = relativeBottom;
      deepestEl = el.tagName + (el.className ? '.' + String(el.className).split(' ').join('.') : '');
    }
  }

  return {
    trueHeight,
    deepestBottom,
    overflowPx: deepestBottom - trueHeight,
    deepestEl,
  };
};

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

let anyOverflow = false;
const results = [];

for (let i = 1; i <= slideCount; i++) {
  const url = `${baseUrl}/${i}`;
  await page.goto(url, { waitUntil: 'networkidle' });
  await page.waitForTimeout(700); // let entrance animations / fonts settle before measuring

  const r = await page.evaluate(measure);
  if (r.error) {
    console.error(`slide ${i}: ${r.error}`);
    anyOverflow = true;
    continue;
  }

  const status = r.overflowPx > 0 ? 'OVERFLOW' : 'ok';
  if (r.overflowPx > 0) anyOverflow = true;
  results.push({ slide: i, ...r, status });
  console.log(
    `slide ${i}: canvas=${r.trueHeight.toFixed(1)}px deepest=${r.deepestBottom.toFixed(1)}px ` +
    `overflow=${r.overflowPx.toFixed(1)}px [${status}]` +
    (r.overflowPx > 0 ? ` -- deepest element: ${r.deepestEl}` : '')
  );
}

await browser.close();

console.log('');
console.log(anyOverflow
  ? `FAIL: ${results.filter(r => r.status === 'OVERFLOW').length}/${slideCount} slide(s) overflow`
  : `PASS: all ${slideCount} slides fit within the true canvas height`);

process.exit(anyOverflow ? 1 : 0);
