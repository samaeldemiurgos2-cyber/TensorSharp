#!/usr/bin/env node
// Real Chromium regression for Generate reference removal. No browser packages or running model host required.
// Serves the unmodified shared Studio JS/CSS and deterministic API fixtures, opens the public #studio entry,
// and uses trusted CDP pointer events (never HTMLElement.click()). This validates client UI behavior only.
// Example: node eng/studio/validate_reference_removal.mjs --output artifacts/studio-reference-removal-browser
// --source path/to/studio.js can compare a saved source snapshot; --expect-failure records a known regression.
// --allow-legacy-choices isolates pointer races when testing the old always-visible reference grid.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const args = Object.fromEntries(process.argv.slice(2).reduce((out, arg, i, all) => {
  if (arg.startsWith('--')) out.push([arg.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]);
  return out;
}, []));
const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const output = path.resolve(String(args.output || 'artifacts/studio-reference-removal-browser'));
if (!['artifacts', 'docs/validation'].some(folder => output.startsWith(path.join(repository, folder) + path.sep)))
  throw new Error('--output must be below ignored artifacts/ or docs/validation/.');
mkdirSync(output, { recursive: true });
const sourcePath = path.resolve(String(args.source || path.join(repository, 'TensorSharp.Chat/WebUi/studio.js')));
const source = readFileSync(sourcePath);
const css = readFileSync(path.join(repository, 'TensorSharp.Chat/WebUi/studio.css'));
writeFileSync(path.join(output, 'studio-source.js'), source);
const report = { source: sourcePath, sourceSha256: createHash('sha256').update(source).digest('hex'),
  coverage: 'Real Chromium shared client UI with deterministic mock API; no inference, native device, or host integration claims.',
  passed: false, steps: [], requests: [], consoleErrors: [], exceptions: [], screenshots: [] };
const projectId = 'e'.repeat(32);
const image = id => ({ id, kind: 'image', name: id + '.png', width: 640, height: 384 });
const video = id => ({ id, kind: 'video', name: id + '.mp4', width: 640, height: 384, durationSeconds: 3, hasAudio: false });
const project = { id: projectId, name: 'Reference removal browser fixture', revision: 1, width: 1280, height: 720,
  frameRate: { numerator: 24, denominator: 1 }, durationFrames: 0, tracks: [], clips: [], captions: [],
  assets: [image('portrait-a'), image('portrait-b'), image('portrait-c'), video('motion-a'), video('motion-b')] };
const model = { modelId: 'reference-fixture', displayName: 'Reference test model', family: 'minimax-h3', available: true,
  modes: ['t2v', 'i2v', 'fl2v', 'ref'], maxReferences: 2, referenceVideos: true, maxReferenceVideos: 1,
  referencesWithFirstFrame: false, spatialMultiple: 32, temporalMultiple: 1, temporalOffset: 0,
  maxDimension: 2048, maxFrames: 257, defaultWidth: 640, defaultHeight: 384, defaultFrameCount: 49,
  defaultFps: 24, defaultSteps: 8, defaultCfgScale: 1, fixedCfgScale: 1 };
let jobs = [];
function initialShot(use) { return { modelId: model.modelId, use, first: '', last: '', firstFrom: null, guides: [], refs: [], videoRefs: [],
  prompt: 'Keep this prompt while changing references.', negativePrompt: '', seed: 42, width: 640, height: 384,
  frameCount: 49, fps: 24, steps: 8, cfgScale: 1, generateAudio: false, sizeSource: 'custom', samplingRecipe: 'native' }; }
const svg = '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect width="320" height="180" fill="#316675"/><path d="M0 180L160 20L320 180" fill="#6eb5a1"/></svg>';
const server = createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  if (pathname === '/studio.js') { res.writeHead(200, { 'Content-Type': 'text/javascript' }); res.end(source); return; }
  if (pathname === '/studio.css') { res.writeHead(200, { 'Content-Type': 'text/css' }); res.end(css); return; }
  if (pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
  if (pathname.includes('/assets/') || pathname.includes('/thumbnail/')) { res.writeHead(200, { 'Content-Type': 'image/svg+xml' }); res.end(svg); return; }
  if (pathname === '/api/studio') {
    let body = ''; for await (const chunk of req) body += chunk;
    const call = JSON.parse(body); report.requests.push(call);
    let result;
    switch (call.action) {
      case 'capabilities': result = { models: [model], externalProcesses: true, importAvailable: true, renderAvailable: true }; break;
      case 'projects/list': result = [project]; break;
      case 'projects/get': result = project; break;
      case 'jobs/list': result = jobs; break;
      case 'proposals/list': case 'generation-drafts/list': result = []; break;
      default: res.writeHead(400, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ message: 'Unmocked action ' + call.action })); return;
    }
    res.writeHead(200, { 'Content-Type': 'application/json' }); res.end(JSON.stringify(result)); return;
  }
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="/studio.css"></head><body style="margin:0;background:#10151b"><div class="header-controls"></div><script src="/studio.js"></script></body></html>');
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const base = 'http://127.0.0.1:' + server.address().port;
const profile = mkdtempSync(path.join(os.tmpdir(), 'studio-remove-'));
const chromePath = String(args.chrome || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');
if (!existsSync(chromePath)) throw new Error('Chrome missing; use --chrome.');
const chrome = spawn(chromePath, ['--headless=new', '--remote-debugging-port=0', '--user-data-dir=' + profile,
  '--no-first-run', '--no-default-browser-check', '--window-size=1440,1000', 'about:blank'], { stdio: 'ignore' });
let socket, nextId = 0, touchMode = false; const pending = new Map();
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function cdp(method, params = {}) {
  const id = ++nextId; socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject, method }));
}
async function evaluate(expression) {
  const r = await cdp('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error('In page: ' + (r.exceptionDetails.exception?.description || r.exceptionDetails.text));
  return r.result.value;
}
async function until(expression, what, timeout = 10000) {
  const end = Date.now() + timeout;
  while (Date.now() < end) { if (await evaluate(expression)) return; await pause(50); }
  throw new Error('Timed out waiting for ' + what);
}
async function screenshot(name) {
  const shot = await cdp('Page.captureScreenshot', { format: 'png' });
  writeFileSync(path.join(output, name + '.png'), Buffer.from(shot.data, 'base64')); report.screenshots.push(name + '.png');
}
// Deliberately separate mouse press/release. A real person can hold a button past focusout timers.
async function pointerClick(selector, hold = 80, duringPress = null) {
  const pos = await evaluate(`(() => { const n = document.querySelector(${JSON.stringify(selector)}); if (!n) throw new Error('No element: ' + ${JSON.stringify(selector)});
    if (n.disabled) throw new Error('Disabled: ' + ${JSON.stringify(selector)}); n.scrollIntoView({block:'center'});
    const r=n.getBoundingClientRect(); if (!r.width || !r.height) throw new Error('Invisible: ' + ${JSON.stringify(selector)});
    return {x:r.x+r.width/2,y:r.y+r.height/2}; })()`);
  if (touchMode) await cdp('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ ...pos, id: 1 }] });
  else {
    await cdp('Input.dispatchMouseEvent', { type: 'mouseMoved', ...pos });
    await cdp('Input.dispatchMouseEvent', { type: 'mousePressed', ...pos, button: 'left', clickCount: 1 });
  }
  if (duringPress) await duringPress();
  await pause(hold);
  if (touchMode) await cdp('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
  else await cdp('Input.dispatchMouseEvent', { type: 'mouseReleased', ...pos, button: 'left', clickCount: 1 });
  await pause(100);
}
const key = value => '[data-focus-key=' + JSON.stringify(value) + ']';
async function state(name) {
  const value = await evaluate(`(() => ({
    status: document.querySelector('.ts-status')?.textContent,
    statusError: document.querySelector('.ts-status')?.dataset.error,
    selected: [...document.querySelectorAll('.ts-ref-item')].map(n=>n.textContent.trim()),
    selectedButtons: [...document.querySelectorAll('.ts-ref-item button')].map(n=>n.dataset.focusKey),
    tiles: [...document.querySelectorAll('.ts-refs [data-asset-id]')].map(n=>({id:n.dataset.assetId,pressed:n.getAttribute('aria-pressed'),disabled:n.disabled})),
    firstSlot: document.querySelector('.ts-slot[data-slot=first]')?.textContent,
    draft: JSON.parse(localStorage.getItem('tensorsharp.studio.shot.${projectId}') || 'null'),
    prompt: document.querySelector('textarea[data-focus-key=prompt]')?.value,
    focus: document.activeElement?.dataset.focusKey,
    summary: document.querySelector('.ts-gen-summary')?.textContent,
    alwaysVisibleChoiceCount: document.querySelectorAll('.ts-studio-inspector .ts-refs [data-asset-id]').length,
    pickerOpen: !!document.querySelector('.ts-studio-inspector .ts-picker'),
    events: window.__referenceValidationEvents.splice(0)
  }))()`);
  report.steps.push({ name, ...value }); console.log(name, JSON.stringify({ selected: value.selected, refs: value.draft?.refs, videoRefs: value.draft?.videoRefs, status: value.status }));
  return value;
}
async function verify(name, refs, videoRefs) {
  const value = await state(name);
  assert.deepEqual(value.draft.refs, refs, name + ': persisted photos');
  assert.deepEqual(value.draft.videoRefs, videoRefs, name + ': persisted videos');
  assert.equal(value.selectedButtons.length, refs.length + videoRefs.length, name + ': DOM selected row count');
  assert.equal(value.draft.prompt, initialShot().prompt, name + ': prompt preserved');
  assert.equal(value.draft.seed, 42, name + ': seed preserved');
  assert.notEqual(value.statusError, 'true', name + ': no runtime status error');
  assert.equal(value.pickerOpen, false, name + ': choices close after selection');
  if (!args['allow-legacy-choices']) assert.equal(value.alwaysVisibleChoiceCount, 0, name + ': unselected reference thumbnails appear only in the Add picker');
}
async function boot(use) {
  jobs = [];
  await evaluate(`localStorage.setItem('tensorsharp.studio.project', ${JSON.stringify(projectId)}); localStorage.setItem('tensorsharp.studio.shot.${projectId}', ${JSON.stringify(JSON.stringify(initialShot(use)))}); true`);
  await cdp('Page.navigate', { url: base + '/run-' + use + (touchMode ? '-touch' : '') + '/#studio' });
  await until(`!!document.querySelector('.ts-studio-library .ts-asset')`, 'project boot');
  await pointerClick(key(touchMode ? 'mobile-generate' : 'tab-generate'));
  await until(`!!document.querySelector('[data-focus-key=generate]')`, 'fully built Generate form');
  await state(use + (touchMode ? '-touch' : '') + '-opened');
}
async function addPhoto(use, id) {
  if (await evaluate(`!!document.querySelector('[data-focus-key=refs-add]')`)) { await pointerClick(key('refs-add')); await pointerClick('.ts-picker ' + key('tile-' + id)); }
  else await pointerClick('.ts-refs ' + key('tile-' + id));
}
async function addVideo(id) { await pointerClick(key('refs-video-add')); await pointerClick('.ts-picker ' + key('video-' + id)); }
async function scenario(use, deferPoll, race = false) {
  const label = use + (touchMode ? '-touch' : '') + (race ? '-capability-race' : '');
  await boot(use);
  await addPhoto(use, 'portrait-a'); await addPhoto(use, 'portrait-b'); await addVideo('motion-a');
  // Choosing a full video opens reference mode; return through the real radio to test the frame view's selected list.
  if (use === 'frames') await pointerClick(key('use-frames'));
  await verify(label + '-selected', ['portrait-a', 'portrait-b'], ['motion-a']);
  await screenshot(label + '-before');
  if (deferPoll) {
    // A newly completed generation asks refreshProject to redraw while the person types. Studio defers that redraw.
    await pointerClick(key('prompt'));
    const count = report.requests.filter(r => r.action === 'projects/get').length;
    jobs = [{ id: 'completed-fixture-job', projectId, kind: 'generate', state: 'completed', projectRevision: 1 }];
    const end = Date.now() + 6000;
    while (report.requests.filter(r => r.action === 'projects/get').length === count && Date.now() < end) await pause(100);
    assert.ok(report.requests.filter(r => r.action === 'projects/get').length > count, 'poll completed while prompt focused');
  }
  if (race) {
    // Public host notification arrives between the trusted press and release. A redraw must keep that button alive.
    await pause(1100);
    await pointerClick(key('ref-portrait-a-remove'), 100, async () => {
      model.defaultSteps++;
      const count = report.requests.filter(r => r.action === 'capabilities').length;
      await evaluate(`window.dispatchEvent(new Event('tensorsharp:model-changed')); true`);
      const end = Date.now() + 3000;
      while (report.requests.filter(r => r.action === 'capabilities').length === count && Date.now() < end) await pause(30);
      assert.ok(report.requests.filter(r => r.action === 'capabilities').length > count, 'capabilities refreshed during trusted pointer press');
      await pause(100);
      await state(label + '-held-through-capability-refresh');
    });
  } else await pointerClick(key('ref-portrait-a-remove'));
  await verify(label + '-photo-a-removed', ['portrait-b'], ['motion-a']);
  await pointerClick(key('ref-portrait-b-remove'));
  await verify(label + '-photo-b-removed', [], ['motion-a']);
  await pointerClick(key('video-ref-motion-a-remove'));
  await verify(label + '-video-removed', [], []);
  await screenshot(label + '-after');
  await addPhoto(use, 'portrait-c'); await addVideo('motion-b');
  await verify(label + '-replacement-selected', ['portrait-c'], ['motion-b']);
  await screenshot(label + '-replacement');
  await cdp('Page.reload');
  await until(`!!document.querySelector('.ts-studio-library .ts-asset')`, 'reopened project');
  await pointerClick(key(touchMode ? 'mobile-generate' : 'tab-generate'));
  await verify(label + '-replacement-restored-after-reload', ['portrait-c'], ['motion-b']);
}

try {
  let port;
  for (let i = 0; i < 100 && !port; i++) { const f = path.join(profile, 'DevToolsActivePort'); if (existsSync(f)) port = readFileSync(f, 'utf8').split('\n')[0]; else await pause(100); }
  if (!port) throw new Error('Chrome did not open a DevTools port.');
  const page = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(t => t.type === 'page');
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const m = JSON.parse(event.data), p = m.id && pending.get(m.id);
    if (p) { pending.delete(m.id); if (m.error) p.reject(new Error(p.method + ': ' + m.error.message)); else p.resolve(m.result); }
    else if (m.method === 'Runtime.exceptionThrown') report.exceptions.push(m.params.exceptionDetails);
    else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') report.consoleErrors.push(m.params.args.map(a => a.value || a.description));
    else if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error') report.consoleErrors.push(m.params.entry);
  };
  await cdp('Page.enable'); await cdp('Runtime.enable'); await cdp('Log.enable'); await cdp('DOM.enable');
  report.browser = await cdp('Browser.getVersion');
  await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
  await cdp('Page.addScriptToEvaluateOnNewDocument', { source: `window.__referenceValidationEvents=[];
    for (const type of ['pointerdown','pointerup','mousedown','mouseup','click','focusout']) document.addEventListener(type,e=>{
      if (e.target.closest?.('.ts-studio-inspector')) window.__referenceValidationEvents.push({type,trusted:e.isTrusted,key:e.target.closest('[data-focus-key]')?.dataset.focusKey});
    },true);` });
  await cdp('Page.navigate', { url: base });
  await until('!!window.TensorSharpStudio', 'public Studio API');
  await scenario('ref', false);
  await scenario('frames', true);
  await scenario('ref', false, true);
  touchMode = true;
  await cdp('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await cdp('Emulation.setTouchEmulationEnabled', { enabled: true });
  await scenario('ref', false);
  await scenario('frames', true);
  assert.equal(report.exceptions.length, 0, 'no uncaught browser exceptions');
  assert.equal(report.consoleErrors.length, 0, 'no browser console errors');
  report.passed = true;
  if (args['expect-failure']) { report.unexpectedPass = true; process.exitCode = 1; }
  console.log('BROWSER PASSED', report.steps.length, 'states; screenshots in', output);
} catch (error) {
  report.error = String(error.stack || error);
  report.expectedFailure = !!args['expect-failure'] && error.code === 'ERR_ASSERTION';
  try { await state('failure-state'); await screenshot('zz-failure'); } catch (_) { }
  console.error('BROWSER FAILED', error);
  if (!report.expectedFailure) process.exitCode = 1;
} finally {
  writeFileSync(path.join(output, 'report.json'), JSON.stringify(report, null, 2));
  try { socket?.close(); } catch (_) { }
  chrome.kill(); server.close();
  await pause(300); rmSync(profile, { recursive: true, force: true });
}
