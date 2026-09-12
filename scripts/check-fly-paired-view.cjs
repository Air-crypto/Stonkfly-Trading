/* Run against a recording server: NODE_PATH=<playwright module path> node scripts/check-fly-paired-view.cjs.
   Set FLY_VIEW_URL, CHROMIUM_PATH, and optional FLY_VIEW_SCREENSHOT. No model jobs are submitted. */
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
 const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];let writes=0;
 page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
 const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8765';
 const current='marketrestore01-restore_10704',other='marketrestore01-trained_frozen';
 await page.goto(`${base}/?run=${current}&step=1&neuron=10704&compare=${other}`);
 await page.waitForFunction(()=>document.querySelectorAll('#paired-traces svg').length===3);
 const views=await page.evaluate(async({current,other})=>Promise.all([current,other].map(async r=>await(await fetch('/api/view?run='+r)).json())),{current,other});
 const [a,b]=views,plastic=a.edges.filter(e=>e.plastic_index!==null);
 const deltas=plastic.map(e=>{const oe=b.edges.find(o=>o.id===e.id);if(!oe)return -1;return Math.abs(a.frames[1].plastic_weights[0][a.plastic_selection.indexOf(e.plastic_index)]-b.frames[1].plastic_weights[0][b.plastic_selection.indexOf(oe.plastic_index)]);});
 const ei=deltas.indexOf(Math.max(...deltas));assert(deltas[ei]>0);await page.selectOption('#edge',String(ei));
 let note=await page.locator('#paired-note').innerText();assert(note.includes('Input identical: yes'));assert(note.includes('Body 10704'));assert(note.includes('10 ms into this observation'));
 const paths=await page.locator('#paired-weight svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));assert.equal(paths.length,2);assert.notEqual(paths[0],paths[1]);
 await page.locator('#bin').fill('37');await page.locator('#bin').dispatchEvent('input');assert((await page.locator('#paired-note').innerText()).includes('380 ms into this observation'));
 if(process.env.FLY_VIEW_SCREENSHOT)await page.locator('#paired-traces').screenshot({path:process.env.FLY_VIEW_SCREENSHOT});
 await page.click('#refresh');assert.equal(await page.inputValue('#compare'),other);assert(await page.locator('#paired-traces').isVisible());
 const originalPaths=await page.locator('#paired-traces svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));
 // Compare permuted display indices: body IDs and edge endpoints must still match.
 const reordered=structuredClone(b),n=reordered.nodes.length;reordered.nodes.reverse();
 reordered.edges.forEach(e=>{e.source=n-1-e.source;e.target=n-1-e.target;});reordered.edges.reverse();
 reordered.frames.forEach(f=>{f.counts.forEach(row=>row.reverse());f.voltage.forEach(row=>row.reverse());});
 let fixture=reordered;
 await page.route('**/api/view?run='+other,r=>r.fulfill({json:fixture}));
 await page.selectOption('#compare','');assert(!(await page.locator('#paired-traces').isVisible()));
 await page.selectOption('#compare',other);await page.waitForFunction(()=>document.querySelectorAll('#paired-traces svg').length===3);
 assert((await page.locator('#paired-note').innerText()).includes('Body 10704'));
 assert.deepEqual(await page.locator('#paired-traces svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d'))),originalPaths);
 const pairedBefore=await page.locator('#paired-spikes svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));
 assert((await page.locator('#paired-note').innerText()).includes('Recorded input prefix: identical'));
 // A matching current image must not imply the preceding recorded input history matched.
 fixture=structuredClone(reordered);fixture.frames[0].event.input_sha256='f'.repeat(64);
 await page.selectOption('#compare','');await page.selectOption('#compare',other);
 await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('Recorded input prefix: different'));
 assert((await page.locator('#paired-note').innerText()).includes('Input identical: yes'));
 fixture=structuredClone(reordered);delete fixture.frames[0].event.input_sha256;
 await page.selectOption('#compare','');await page.selectOption('#compare',other);
 await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('Recorded input prefix: unverified'));
 fixture=structuredClone(reordered);fixture.frames.shift();
 await page.selectOption('#compare','');await page.selectOption('#compare',other);
 await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('Recorded input prefix: different'));
 assert.equal(await page.locator('#paired-traces svg').count(),3);
 fixture=structuredClone(reordered);fixture.frames.push(structuredClone(fixture.frames[1]));
 await page.selectOption('#compare','');await page.selectOption('#compare',other);
 await page.waitForFunction(()=>document.querySelector('#comparison').textContent.includes('ambiguous observation keys excluded'));
 assert.equal(await page.locator('#paired-traces svg').count(),0);
 assert((await page.locator('#paired-note').innerText()).includes('No shared observation'));

 fixture=structuredClone(reordered);fixture.frames.forEach(f=>f.event.market_decision_ts+=1);
 await page.selectOption('#compare','');await page.selectOption('#compare',other);
 await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('No shared observation'));assert.equal(await page.locator('#paired-traces svg').count(),0);
 fixture=structuredClone(reordered);fixture.frames.forEach(f=>f.times_ms[0]+=1);
 await page.selectOption('#compare','');await page.selectOption('#compare',other);
 await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('different time bins'));assert.equal(await page.locator('#paired-traces svg').count(),0);
 fixture=reordered;await page.selectOption('#compare','');await page.selectOption('#compare',other);await page.waitForFunction(()=>document.querySelectorAll('#paired-traces svg').length===3);
 assert.deepEqual(await page.locator('#paired-spikes svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d'))),pairedBefore);
 await page.selectOption('#runs','marketrestore01-restore_11402');await page.waitForFunction(()=>document.querySelector('#paired-traces').hidden);assert.equal(await page.inputValue('#compare'),'');
 await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({checks:'paired traces, identity mapping, cursor, refresh, input prefix, ambiguous timestamps, absent timestamps, mismatched bins, clearing, mobile',errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
