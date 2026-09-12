/* Verify edge styling against real recorded source spikes; never submit model jobs. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const root=process.env.FLY_SPIKE_RECORDINGS||'examples/fly-debugger';
 const cases=[['market08-pool0-trained_frozen','changed'],['marketrestore01-restore_10704','restored']];
 const views=new Map(cases.map(([name])=>[name,JSON.parse(fs.readFileSync(path.join(root,name,'view.json')))]));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];let writes=0;const verified=[];
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/api/**',r=>{const u=new URL(r.request().url());
   if(u.pathname==='/api/run'){writes++;return r.abort();}
   if(u.pathname==='/api/runs')return r.fulfill({json:{runs:[...views.keys()],can_run:false}});
   if(u.pathname==='/api/status')return r.fulfill({json:{status:'idle',run:null}});
   if(u.pathname==='/api/view'){const v=views.get(u.searchParams.get('run'));assert(v);return r.fulfill({json:v});}
   errors.push('Unexpected API request: '+u.pathname);return r.abort();
  });
  for(const [name,kind] of cases){
   const v=views.get(name),plastic=v.edges.filter(e=>e.plastic_index!==null),restored=new Set((v.report.restoration?.edge_ids||[]).map(String));let pick;
   for(let ei=0;ei<plastic.length&&!pick;ei++){
    const e=plastic[ei],pi=v.plastic_selection.indexOf(e.plastic_index);
    for(let step=0;step<v.frames.length&&!pick;step++){
     const f=v.frames[step],active=f.counts.findIndex(row=>row[e.source]>0),quiet=f.counts.findIndex(row=>row[e.source]===0);
     if(active<0||quiet<0)continue;
     if(kind==='restored'?!restored.has(String(e.id)):restored.has(String(e.id))||Math.abs(f.plastic_weights[active][pi]-e.weight)<=1e-9)continue;
     pick={e,ei,step,active,quiet,pi};
    }
   }
   assert(pick,'Need an actual active/quiet connection with the requested memory state');
   const {e,ei,step,active,quiet,pi}=pick;
   await page.goto(`${process.env.FLY_VIEW_URL||'http://127.0.0.1:8765'}/?run=${name}&step=${step}&bin=${active}&edge=${e.id}&neuron=${v.nodes[e.source].id}`);
   await page.waitForFunction(name=>document.querySelector('#status').textContent.startsWith('Loaded '+name),name);
   await page.check('#pristine');
   const edgePath=()=>page.locator(`#network path[data-edge-id="${e.id}"]`);
   const expectedColor=kind==='restored'?'#c4a1ff':'#ffbf69';
   for(const bin of [active,quiet]){
    await page.locator('#bin').fill(String(bin));await page.locator('#bin').dispatchEvent('input');
    const count=v.frames[step].counts[bin][e.source];
    assert.equal(await edgePath().getAttribute('data-source-spikes'),String(count));
    assert.equal(await edgePath().getAttribute('stroke'),expectedColor);
    assert.equal(await edgePath().getAttribute('stroke-dasharray'),count>0?'4 2':'none');
    assert((await edgePath().locator('title').textContent()).includes(`source ${count} spikes in this bin`));
    assert.equal(await page.locator('#network path[data-selected-edge]').getAttribute('data-selected-edge'),String(e.id));
    assert.equal(await page.locator('#network path[data-edge-id]').count(),v.edges.length);
   }
   await page.locator('#bin').fill(String(active));await page.locator('#bin').dispatchEvent('input');
   const moment=await page.locator('#moment-link').getAttribute('href');assert.equal(new URL(moment).searchParams.get('pristine'),'1');
   await page.goto(moment);await page.waitForFunction(name=>document.querySelector('#status').textContent.startsWith('Loaded '+name),name);
   assert(await page.isChecked('#pristine'));assert.equal(await edgePath().getAttribute('stroke'),expectedColor);
   assert.equal(await edgePath().getAttribute('stroke-dasharray'),'4 2');
   if(process.env.FLY_SPIKE_SCREENSHOTS){fs.mkdirSync(process.env.FLY_SPIKE_SCREENSHOTS,{recursive:true});await page.locator('.circuit').screenshot({path:path.join(process.env.FLY_SPIKE_SCREENSHOTS,`fly-source-spike-${kind}.png`)});}
   if(kind==='changed'&&v.frames[step].plastic_weights[active][pi]===v.frames[step].plastic_initial[pi]){
    await page.uncheck('#pristine');assert.equal(await edgePath().getAttribute('stroke'),'#67e8cf');
    assert.equal(await edgePath().getAttribute('stroke-dasharray'),'4 2');
   }
   const another=(ei+1)%plastic.length;await page.selectOption('#edge',String(another));
   assert.equal(await page.locator('#network path[data-selected-edge]').count(),1);
   assert.equal(await page.locator('#network path[data-selected-edge]').getAttribute('data-selected-edge'),String(plastic[another].id));
   verified.push({name,edge:e.id,source:v.nodes[e.source].id,step,activeBin:active,quietBin:quiet});
  }
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({verified,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
