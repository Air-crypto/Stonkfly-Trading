/* Read-only browser check of the two retained, independently audited controls. */
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {chromium}=require('playwright');

(async()=>{
 const root=process.env.FLY_CONTROL_ROOT||'runs/cloud-debugger';
 const names=['online11-development-pool0-pristine_frozen','online11-development-pool0-trained_frozen'];
 const views=names.map(name=>JSON.parse(fs.readFileSync(path.join(root,name,'view.json'))));
 for(const view of views){
  assert.equal(view.validation_only,false);assert.equal(view.report.config.learning,false);
  assert.equal(view.frames.length,24);assert.equal(view.report.study,'11');
 }
 const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766';
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];let writes=0,bins=0;
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/api/run',r=>{writes++;return r.abort();});
  // All read requests use the actual server and installed recordings.
  for(let run=0;run<names.length;run++){
   const view=views[run],node=view.nodes.findIndex(n=>n.id==='10527');assert(node>=0);
   await page.goto(base+'/?run='+names[run]+'&step=0&bin=0&neuron=10527');
   await page.waitForFunction(name=>document.querySelector('#status').textContent.startsWith('Loaded '+name+' · 24 observations'),names[run]);
   assert.equal(await page.locator('#source-badge').innerText(),'Market replay · simulated fills');
   assert((await page.locator('#phase-note').innerText()).includes('frozen throughout'));
   assert(!(await page.locator('#phase-note').innerText()).includes('updated online'));
   assert((await page.locator('#market-coverage').innerText()).includes('24 / 24 decision slots observed'));
   assert.equal(await page.locator('#market-timeline tr').count(),26);
   for(let step=0;step<24;step++){
    await page.selectOption('#step',String(step));
    assert.equal(await page.locator('#decision').innerText(),view.frames[step].event.side);
    for(const bin of [0,19,20,49]){
     await page.locator('#bin').fill(String(bin));await page.locator('#bin').dispatchEvent('input');
     assert((await page.locator('#neuron-detail').innerText()).includes(view.frames[step].counts[bin][node]+' spikes in this bin'));
     bins++;
    }
   }
  }
  const file=path.join(root,names[1],'view.json');
  await page.locator('#import').setInputFiles(file);
  await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Loaded view.json · 24 observations'));
  await page.selectOption('#step','23');
  assert.equal(await page.locator('#decision').innerText(),views[1].frames[23].event.side);
  assert.equal(await page.locator('#error').innerText(),'');
  const step=views[0].frames.findIndex((f,i)=>f.event.side!==views[1].frames[i].event.side);assert(step>=0);
  let neuron=null,bin=null;
  for(const id of ['10527','555871']){
   const indices=views.map(v=>v.nodes.findIndex(n=>n.id===id));
   if(indices.every(i=>i>=0)){
    const different=views[0].frames[step].counts.findIndex((row,b)=>row[indices[0]]!==views[1].frames[step].counts[b][indices[1]]);
    if(different>=0){neuron=id;bin=different;break;}
   }
  }
  assert(neuron!==null);
  const url=base+'/?run='+names[1]+'&compare='+names[0]+'&step='+step+'&bin='+bin+'&neuron='+neuron+'&pristine=1';
  await page.goto(url);
  await page.waitForFunction(()=>document.querySelector('#comparison').textContent.includes('24 shared decision timestamps'));
  assert.equal(await page.locator('#comparison table tr').count(),25);
  assert.equal(await page.locator('#decision').innerText(),views[1].frames[step].event.side);
  assert.equal(await page.locator('#pristine').isChecked(),true);
  assert.equal(await page.locator('#paired-traces').isVisible(),true);
  if(process.env.FLY_CONTROL_SCREENSHOT)await page.screenshot({path:process.env.FLY_CONTROL_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});
  assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);
  console.log(JSON.stringify({status:'real_control_browser_check_passed',runs:names,observations:48,
   selected_bins:bins,paired_decisions:24,large_import:true,mobile_overflow:false,errors,writes,url}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
