/* Verify the silent-source update and contemporaneous DAN spikes in the saved view. */
const assert=require('node:assert/strict'),{chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000},locale:'en-US'}),errors=[];let writes=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8765',prefix='creditdivergence01-trained_online_recorded_';
  for(const side of ['carry','reset_rates']){
   const other=side==='carry'?'reset_rates':'carry';
   await page.goto(`${base}/?run=${prefix+side}&step=1&bin=0&neuron=18540&edge=4110156&compare=${prefix+other}`);
   await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('Body 18540: 0 vs 0 spikes in this bin'));
   const edge=page.locator('#network [data-edge-id="4110156"]');
   assert.equal(await edge.getAttribute('data-source-spikes'),'0');assert.equal(await edge.getAttribute('stroke-dasharray'),'none');
   assert.equal(await edge.getAttribute('stroke'),'#ffbf69');
   const drive=await page.locator('#credit-note').innerText();
   assert(drive.includes(`total drive: ${side==='carry'?'-0.526448':'0'} u/s`));
   assert(drive.includes(`Recorded weight step: ${side==='carry'?'-0.013908':'-0.003061'}`));
   const ids=await page.evaluate(async run=>(await(await fetch('/api/view?run='+run)).json()).nodes.map(n=>n.id),prefix+side);
   for(const neuron of ['11327','11900']){
    await page.selectOption('#node',String(ids.indexOf(neuron)));
    assert((await page.locator('#neuron-detail').innerText()).includes(`${neuron} · 1 spikes in this bin`));
    assert.equal(await edge.getAttribute('data-source-spikes'),'0');
    const link=new URL(await page.locator('#moment-link').getAttribute('href'));
    assert.equal(link.searchParams.get('neuron'),neuron);assert.equal(link.searchParams.get('edge'),'4110156');
   }
  }
  assert.deepEqual(errors,[]);assert.equal(writes,0);
  console.log(JSON.stringify({recordings:2,quietSource:'18540',updatingEdge:'4110156',activeDANs:['11327','11900'],checks:'source activity distinct from changing weight, drive and weight-step values, independent neuron/edge selection',errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
