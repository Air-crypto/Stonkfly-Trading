/* Inspect the new online connection selection; no model submissions. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766';
 const root='examples/fly-debugger',prefix='creditdivergence01-trained_online_recorded_';
 const views=Object.fromEntries(['carry','reset_rates'].map(k=>[k,JSON.parse(fs.readFileSync(path.join(root,prefix+k,'view.json')))]));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];let writes=0,bins=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
  for(const side of ['carry','reset_rates']){
   const other=side==='carry'?'reset_rates':'carry';
   await page.goto(`${base}/?run=${prefix+side}&step=1&bin=0&neuron=11402&edge=4110156&compare=${prefix+other}`);
   await page.waitForFunction(()=>document.querySelector('#paired-weight svg')&&document.querySelector('#credit-chart svg'));
   assert.deepEqual(await page.evaluate(async n=>await(await fetch('/api/view?run='+n)).json(),prefix+side),views[side]);
   for(let i=0;i<3;i++){
    await page.selectOption('#step',String(i));
    for(const b of [0,4,13,37,49]){
     await page.locator('#bin').fill(String(b));await page.locator('#bin').dispatchEvent('input');
     const frame=views[side].frames[i],otherFrame=views[other].frames[i];
     const ni=views[side].nodes.findIndex(n=>n.id==='11402'),oi=views[other].nodes.findIndex(n=>n.id==='11402');
     const pi=views[side].plastic_selection.indexOf(8),op=views[other].plastic_selection.indexOf(8);
     const paired=await page.locator('#paired-note').innerText();
     assert(paired.includes(`Body 11402: ${frame.counts[b][ni]} vs ${otherFrame.counts[b][oi]} spikes in this bin`));
     assert(paired.includes(`${Number(frame.plastic_weights[b][pi].toFixed(6))} vs ${Number(otherFrame.plastic_weights[b][op].toFixed(6))}`));
     assert((await page.locator('#edge-detail').innerText()).includes('Edge 4110156 · 18540 → 11402'));
     assert.equal(await page.locator('#credit-chart svg path').count(),3);
     assert(!(await page.locator('#paired-memory-note').innerText()).includes('unavailable'));
     assert(!(await page.locator('#credit-note').innerText()).includes('unavailable'));
     assert.equal(await page.locator('#decision').innerText(),views[side].frames[i].event.side);bins++;
    }
   }
  }
  await page.goto(`${base}/?run=${prefix}reset_rates&step=1&bin=4&neuron=11402&edge=4110156&compare=${prefix}carry`);
  await page.waitForFunction(()=>document.querySelector('#paired-voltage svg'));
  if(process.env.FLY_DIVERGENCE_FIRST_SCREENSHOT)await page.locator('#paired-traces').screenshot({path:process.env.FLY_DIVERGENCE_FIRST_SCREENSHOT});
  await page.goto(`${base}/?run=${prefix}reset_rates&step=2&bin=37&neuron=10527&edge=4110156&compare=${prefix}carry`);
  await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('Body 10527: 0 vs 1 spikes in this bin'));
  assert.equal(await page.locator('#decision').innerText(),'HOLD');
  if(process.env.FLY_DIVERGENCE_SCREENSHOT)await page.locator('#paired-traces').screenshot({path:process.env.FLY_DIVERGENCE_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);
  console.log(JSON.stringify({recordings:2,observations:6,selectedBins:bins,addedConnection:'4110156',gateBin:37,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
