/* Read-only browser QA. FLY_VIEW_URL, CHROMIUM_PATH and FLY_DECODER_SCREENSHOT are optional. */
const assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];let writes=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8765',name='market06-pool1-trained_frozen';
  await page.goto(`${base}/?run=${name}&step=1&bin=49`);await page.waitForFunction(()=>document.querySelectorAll('#decoder-path svg').length===2);
  const read=()=>page.locator('#decoder-path-note').innerText();
  const reopen=async()=>{await page.goto(`${base}/?run=${name}&step=1&bin=49`);await page.waitForFunction(()=>document.querySelector('#decoder-path-note').textContent.length>0);};
  let note=await read();assert(note.includes('At 500 ms'));assert(note.includes('left contribution 34 Hz, right 36 Hz, right − left 2 Hz; 3 gate spikes'));assert(note.includes('One spike contributes 2 Hz'));
  const original=await page.evaluate(async name=>await(await fetch('/api/view?run='+name)).json(),name);
  const paths=await page.locator('#decoder-path svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));
  assert.equal(paths.length,4);
  if(process.env.FLY_DECODER_SCREENSHOT)await page.locator('#decoder-path').screenshot({path:process.env.FLY_DECODER_SCREENSHOT});
  await page.locator('#bin').fill('0');await page.locator('#bin').dispatchEvent('input');
  const f=original.frames[1],ids=original.nodes.map(n=>n.id),sum=key=>f.event.cell_ids[key].reduce((a,id)=>a+f.counts[0][ids.indexOf(id)],0);
  note=await read();assert(note.includes('At 10 ms'));assert(note.includes(`left contribution ${sum('left')*2} Hz, right ${sum('right')*2} Hz`));assert(note.includes(`${sum('gate')} gate spikes so far`));
  let fixture=structuredClone(original);fixture.nodes.reverse();fixture.edges.forEach(e=>{e.source=fixture.nodes.length-1-e.source;e.target=fixture.nodes.length-1-e.target;});fixture.frames.forEach(f=>{f.counts.forEach(r=>r.reverse());f.voltage.forEach(r=>r.reverse());});
  await page.route('**/api/view?run='+name,r=>r.fulfill({json:fixture}));await reopen();
  assert.deepEqual(await page.locator('#decoder-path svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d'))),paths);
  fixture=structuredClone(original);fixture.frames[1].event.cell_ids.left=['missing'];await reopen();
  assert((await read()).includes('missing or ambiguous'));assert.equal(await page.locator('#decoder-path svg').count(),0);
  fixture=structuredClone(original);fixture.frames[1].event.left_hz+=2;await reopen();
  assert((await read()).includes('do not reconcile'));assert.equal(await page.locator('#decoder-path svg').count(),0);
  fixture=structuredClone(original);fixture.frames[1].times_ms[0]+=1;await reopen();
  assert((await read()).includes('unsupported observation time grid'));assert.equal(await page.locator('#decoder-path svg').count(),0);
  fixture=original;await reopen();assert.equal(await page.locator('#decoder-path svg').count(),2);
  await page.selectOption('#step','2');assert((await read()).includes('Final direction: 4 Hz'));assert((await read()).includes('0 gate spikes so far'));
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));assert.deepEqual(errors,[]);assert.equal(writes,0);
  console.log(JSON.stringify({checks:'recorded endpoint rates, intermediate counts, ID remapping, missing output, inconsistent rates, time grid, clearing, mobile',errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
