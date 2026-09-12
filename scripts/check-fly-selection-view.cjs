/* Check extended recorded selections against the actual server; never run a model. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const root=process.env.FLY_SELECTION_RECORDINGS||'runs/research-market-10/connection-selection';
 const views=Object.fromEntries(['trained','pristine'].map(k=>[k,JSON.parse(fs.readFileSync(path.join(root,k,'view.json')))]));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050},locale:'en-US'}),errors=[];let writes=0,bins=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766',prefix=process.env.FLY_SELECTION_PREFIX||'market10-edge4110156-';
  const open=async(memory,step=0)=>{const other=memory==='trained'?'pristine':'trained';await page.goto(`${base}/?run=${prefix+memory}&step=${step}&bin=49&neuron=18540&edge=4110156&compare=${prefix+other}`);await page.waitForFunction(()=>document.querySelector('#paired-w svg'));};
  for(const memory of ['trained','pristine']){
   await open(memory);const v=views[memory],other=views[memory==='trained'?'pristine':'trained'];
   const fetched=await page.evaluate(async name=>await(await fetch('/api/view?run='+name)).json(),prefix+memory);assert.deepEqual(fetched,v);
   const n=v.nodes.findIndex(n=>n.id==='18540'),on=other.nodes.findIndex(n=>n.id==='18540');assert(n>=0&&on>=0);
   await page.locator('details:has(#provenance) > summary').click();
   assert((await page.locator('#provenance').innerText()).includes('selection_extension'));
   await page.locator('details:has(#provenance) > summary').click();
   for(let i=0;i<v.frames.length;i++){
    await page.selectOption('#step',String(i));const f=v.frames[i],o=other.frames[i];
    for(const b of new Set([0,Math.max(0,f.counts.findIndex(row=>row[n]>0)),49])){
     await page.locator('#bin').fill(String(b));await page.locator('#bin').dispatchEvent('input');
     assert((await page.locator('#paired-note').innerText()).includes(`Body 18540: ${f.counts[b][n]} vs ${o.counts[b][on]} spikes in this bin`));
     assert((await page.locator('#paired-note').innerText()).includes('Edge 4110156 (18540 → 11402)'));
     for(const id of ['paired-voltage','paired-spikes','paired-weight','paired-u','paired-w'])assert.equal(await page.locator('#'+id+' svg path').count(),2);
     assert.equal(await page.locator('#network [data-edge-id="4110156"]').getAttribute('data-source-spikes'),String(f.counts[b][n]));
     assert((await page.locator('#activity-neuron-state').innerText()).includes('Body 18540 at this boundary'));
     bins++;
    }
    const first=f.counts.findIndex((row,b)=>row[n]!==o.counts[b][on]);
    if(first>=0){assert.equal(await page.locator('#first-difference').getAttribute('data-bin'),String(first));if(first!==49)await page.locator('#first-difference').click();assert.equal(await page.locator('#bin').inputValue(),String(first));}
   }
  }
  await open('trained',1);
  if(process.env.FLY_SELECTION_SCREENSHOT)await page.locator('#paired-traces').screenshot({path:process.env.FLY_SELECTION_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({recordings:2,observations:6,selectedBins:bins,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
