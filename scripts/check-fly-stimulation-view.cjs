/* Check actual audited recordings in the UI; never submit a model invocation. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const root=process.env.FLY_STIMULATION_RECORDINGS||'runs/recipient-stimulation-01/cloud',views=new Map();
 for(const name of fs.readdirSync(root).filter(n=>/^pool\d+-/.test(n))){const file=path.join(root,name,'view.json');if(fs.existsSync(file))views.set(name,JSON.parse(fs.readFileSync(file)));}
 assert.equal(views.size,12);
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050},locale:'en-US'}),errors=[];let writes=0,observations=0;
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/api/**',r=>{const u=new URL(r.request().url());
   if(u.pathname==='/api/run'){writes++;return r.abort();}
   if(u.pathname==='/api/runs')return r.fulfill({json:{runs:[...views.keys()],can_run:false}});
   if(u.pathname==='/api/status')return r.fulfill({json:{status:'idle',run:null}});
   if(u.pathname==='/api/view'){const view=views.get(u.searchParams.get('run'));assert(view);return r.fulfill({json:view});}
   errors.push('Unexpected API request: '+u.pathname);return r.abort();
  });
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766';
  for(const [name,v] of views){
   assert.equal(v.report.source,'matched_market_stimulus_assay');
   await page.goto(`${base}/?run=${name}`);await page.waitForFunction(n=>document.querySelector('#status').textContent.startsWith('Loaded '+n),name);
   assert.equal(await page.locator('#source-badge').innerText(),'Historical inputs · neural diagnostic');assert(await page.locator('#market-panel').isHidden());
   for(let i=0;i<3;i++){
    await page.selectOption('#step',String(i));const f=v.frames[i],e=f.event;
    assert.equal(await page.locator('#decision').innerText(),e.side);
    const note=await page.locator('#phase-note').innerText();assert(note.includes(`Diagnostic current ${e.stimulation.current} to cells 10704, 11402 for 500 ms`));assert(note.includes('weights and memory frozen'));assert(note.includes('No fills or returns recomputed'));
    assert((await page.locator('#input-info').innerText()).includes('news available at quote time'));
    for(const id of ['10704','11402']){
     const ix=v.nodes.findIndex(n=>n.id===id);assert(ix>=0);await page.selectOption('#node',String(ix));
     const bin=f.counts.findIndex(r=>r[ix]>0);await page.locator('#bin').fill(String(Math.max(bin,0)));await page.locator('#bin').dispatchEvent('input');
     assert((await page.locator('#neuron-detail').innerText()).includes(`${f.counts[Math.max(bin,0)][ix]} spikes in this bin`));
    }
    observations++;
   }
  }
  await page.goto(base+'/?run=pool0-trained-current10&step=0&bin=3&neuron=11402&edge=4110156&compare=pool0-pristine-current10&pristine=1');
  await page.waitForFunction(()=>document.querySelector('#paired-spikes svg'));
  assert((await page.locator('#paired-note').innerText()).includes('Input identical: yes'));
  assert((await page.locator('#paired-note').innerText()).includes('Applied diagnostic current: blue 10 to cells 10704, 11402 for 500 ms; pink 10 to cells 10704, 11402 for 500 ms'));
  if(process.env.FLY_STIMULATION_PUBLISHED){
   await page.unroute('**/api/**');await page.route('**/api/run',r=>{writes++;return r.abort();});
   await page.goto(base+'/?run=stimulation01-pool0-trained-current10&step=0&bin=3&neuron=11402&edge=4110156&pristine=1&compare=stimulation01-pool0-pristine-current10');
   await page.waitForFunction(()=>document.querySelector('#paired-spikes svg'));
   assert((await page.locator('#paired-note').innerText()).includes('Body 11402: 0 vs 1 spikes in this bin'));
  }
  if(process.env.FLY_STIMULATION_SCREENSHOT)await page.locator('#paired-traces').screenshot({path:process.env.FLY_STIMULATION_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({recordings:views.size,observations,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
