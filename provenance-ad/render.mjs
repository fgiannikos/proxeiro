import { chromium } from '/opt/node22/lib/node_modules/playwright/index.mjs';
import { spawn } from 'node:child_process';
import path from 'node:path';
const mode = process.argv[2] || 'stills';
const scene = process.env.SCENE || 'scene.html';
import fs from 'node:fs';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
if (fs.existsSync('painting.jpg')) {
  const uri = 'data:image/jpeg;base64,' + fs.readFileSync('painting.jpg').toString('base64');
  await page.addInitScript(u => { window.PAINTING = u; }, uri);
}
if (fs.existsSync('memory-frames')) {
  const frames = fs.readdirSync('memory-frames').filter(f => f.endsWith('.jpg')).sort()
    .map(f => 'data:image/jpeg;base64,' + fs.readFileSync('memory-frames/' + f).toString('base64'));
  await page.addInitScript(fr => { window.MEMFRAMES = fr; }, frames);
}
await page.goto('file://' + path.resolve(scene));
await page.evaluate(() => window.ready);
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(500);
if (mode === 'stills') {
  for (const t of (process.argv[3] || '2,5,8,9.5,11.5,15').split(',').map(Number)) {
    await page.evaluate(t => renderAt(t), t);
    await page.screenshot({ path: `stills/${scene.replace('.html', '')}-t${t}.png` });
  }
} else {
  const fps = 30, dur = Number(process.env.DUR || 15);
  const ff = spawn('ffmpeg', ['-y', '-f', 'image2pipe', '-framerate', String(fps), '-i', '-',
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'slow', '-movflags', '+faststart', mode], { stdio: ['pipe', 'inherit', 'inherit'] });
  for (let f = 0; f < fps * dur; f++) {
    await page.evaluate(t => renderAt(t), f / fps);
    const buf = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
}
await browser.close();
