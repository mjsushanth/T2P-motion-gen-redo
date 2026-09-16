// Programmatic overflow check for every slide, at every reachable click state, per
// MJS_LIB_TOOLS/codex-skills/build-project-deck/references/rendering-and-qa.md sections 1 and 4.
//
// A screenshot cannot detect this: a fixed-canvas slide that clips overflowing content
// (`.slidev-slide-container` here) makes anything past its true bottom edge simply
// absent -- no scrollbar, no visual artifact, nothing "looks wrong". The only real check
// is comparing the deepest-rendered element's bottom edge against the canvas's actual
// measured height, in pixels, not against a guessed buffer.
//
// This version fixes two defects a peer review caught in the first draft (both real,
// verified by re-tracing the arithmetic and by empirical probing, not just accepted):
//
// 1. The first draft's `overflowPx > 0` test read a broken measurement (every element
//    zero-size, e.g. because the page is hidden/backgrounded and got zero-size layout --
//    the exact failure mode rendering-and-qa.md section 2/3 describes) as a PASS: the
//    skip-zero-size guard would skip every element, deepestBottom would stay 0, and
//    `0 - trueHeight` is negative, which reads as "no overflow". A checker whose failure
//    mode is a silent green is worse than no checker. Fixed by asserting the measurement
//    itself is sane (document not hidden, trueHeight > 0, deepestBottom > 0 whenever the
//    slide has any rendered content) before trusting overflowPx at all -- a distinct
//    INVALID status, never folded into "ok".
// 2. The first draft measured only each slide's resting state (`/{slide}`, no click
//    param), which rendering-and-qa.md section 4 explicitly says is not enough --
//    content revealed by a later click can push a diagram off the canvas that was fine
//    at rest. Fixed by walking every click state via Slidev's real routing convention,
//    confirmed empirically (not assumed from docs) by probing with Playwright against a
//    live dev server: `/{slide}?clicks={n}`, not `/{slide}/{n}`. Requesting a clicks value
//    past the slide's real maximum clamps back to that maximum (also confirmed
//    empirically) -- this script relies on that clamping to find each slide's real max
//    click count by probing upward until the URL stops advancing, and separately
//    verifies the clamp holds (asking for an absurdly high value doesn't produce a new
//    phantom state).
//
// Usage:
//   node scripts/check-overflow.mjs [baseUrl] [slideCount]
//   node scripts/check-overflow.mjs http://localhost:3030 13
//   node scripts/check-overflow.mjs https://mjsushanth.github.io/T2P-motion-gen-redo 13
//
// Exits non-zero if any slide/state overflows OR if any measurement is invalid -- an
// invalid measurement is not evidence of a passing slide, and must not be silently
// treated as one.

import { chromium } from 'playwright';

const baseUrl = process.argv[2] || 'http://localhost:3030';
const slideCount = parseInt(process.argv[3] || '13', 10);
const MAX_CLICKS_TO_PROBE = 20; // generous ceiling; real decks won't have more per slide

const measure = () => {
  const canvas = document.querySelector('.slidev-slide-container');
  if (!canvas) return { invalid: 'no .slidev-slide-container found' };
  if (document.hidden) return { invalid: 'document.hidden is true -- measurement is not trustworthy' };

  const canvasRect = canvas.getBoundingClientRect();
  const trueHeight = canvasRect.height;
  if (trueHeight === 0) return { invalid: 'canvas trueHeight measured as 0' };

  let deepestBottom = 0;
  let deepestEl = null;
  let sawAnyElement = false;
  for (const el of canvas.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    if (getComputedStyle(el).display === 'none') continue;
    sawAnyElement = true;
    const relativeBottom = r.bottom - canvasRect.top;
    if (relativeBottom > deepestBottom) {
      deepestBottom = relativeBottom;
      deepestEl = el.tagName + (el.className ? '.' + String(el.className).split(' ').join('.') : '');
    }
  }

  // A slide always renders at least a title/body wrapper with nonzero geometry. If we
  // walked the canvas and found nothing with real size, that is the broken-instrument
  // symptom (e.g. a hidden/backgrounded page returning zero-size layout for everything),
  // not evidence the slide is empty -- never let this read as "fits".
  if (!sawAnyElement || deepestBottom === 0) {
    return { invalid: 'every element measured zero-size -- broken measurement, not a fitting slide' };
  }

  return {
    trueHeight,
    deepestBottom,
    overflowPx: deepestBottom - trueHeight,
    deepestEl,
  };
};

// Probe a slide's real maximum click count by requesting increasing ?clicks=N and
// reading back the URL Slidev actually settles on -- confirmed empirically that
// requesting a value past the real max clamps the URL back down, so the first N whose
// resulting URL no longer matches N is the maximum (that N-1 is real, N onward clamps).
async function findMaxClicks(page, slideUrl) {
  let lastConfirmed = 0;
  for (let n = 1; n <= MAX_CLICKS_TO_PROBE; n++) {
    await page.goto(`${slideUrl}?clicks=${n}`, { waitUntil: 'networkidle' });
    await page.waitForTimeout(150);
    const url = new URL(page.url());
    const got = parseInt(url.searchParams.get('clicks') || '0', 10);
    if (got === n) {
      lastConfirmed = n;
    } else {
      // Clamped below what we asked for -- n is past the real max.
      break;
    }
  }
  return lastConfirmed;
}

// Verify the clamp actually holds (rendering-and-qa.md section 4's "verify the state
// machine itself is faithful" requirement) rather than assuming it from the one probe
// sequence above: ask for something absurdly past any plausible max and confirm it lands
// on the same maximum, not a new phantom state.
async function verifyClampHolds(page, slideUrl, knownMax) {
  await page.goto(`${slideUrl}?clicks=${MAX_CLICKS_TO_PROBE * 5}`, { waitUntil: 'networkidle' });
  await page.waitForTimeout(150);
  const url = new URL(page.url());
  const got = parseInt(url.searchParams.get('clicks') || '0', 10);
  return got === knownMax;
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

let anyProblem = false;
const results = [];

for (let i = 1; i <= slideCount; i++) {
  const slideUrl = `${baseUrl}/${i}`;

  const maxClicks = await findMaxClicks(page, slideUrl);
  const clampOk = await verifyClampHolds(page, slideUrl, maxClicks);
  if (!clampOk) {
    console.error(`slide ${i}: FAIL -- clamping check did not hold (state machine not faithful)`);
    anyProblem = true;
  }

  for (let c = 0; c <= maxClicks; c++) {
    const url = maxClicks === 0 ? slideUrl : `${slideUrl}?clicks=${c}`;
    await page.goto(url, { waitUntil: 'networkidle' });
    await page.waitForTimeout(700); // let entrance animations / fonts settle before measuring

    const r = await page.evaluate(measure);
    const label = `slide ${i}${maxClicks > 0 ? ` click ${c}/${maxClicks}` : ''}`;

    if (r.invalid) {
      console.error(`${label}: INVALID -- ${r.invalid}`);
      anyProblem = true;
      results.push({ slide: i, click: c, status: 'INVALID', reason: r.invalid });
      continue;
    }

    const status = r.overflowPx > 0 ? 'OVERFLOW' : 'ok';
    if (r.overflowPx > 0) anyProblem = true;
    results.push({ slide: i, click: c, ...r, status });
    console.log(
      `${label}: canvas=${r.trueHeight.toFixed(1)}px deepest=${r.deepestBottom.toFixed(1)}px ` +
      `overflow=${r.overflowPx.toFixed(1)}px [${status}]` +
      (r.overflowPx > 0 ? ` -- deepest element: ${r.deepestEl}` : '')
    );
  }
}

await browser.close();

const invalidCount = results.filter(r => r.status === 'INVALID').length;
const overflowCount = results.filter(r => r.status === 'OVERFLOW').length;

console.log('');
if (anyProblem) {
  console.log(`FAIL: ${overflowCount} overflowing state(s), ${invalidCount} invalid/untrustworthy measurement(s) out of ${results.length} checked`);
} else {
  console.log(`PASS: all ${results.length} slide/click states fit within the true canvas height (measurements verified sane)`);
}

process.exit(anyProblem ? 1 : 0);
