/* Verify six audited intervention recordings on the actual local server. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const root=process.env.FLY_RESET_ROOT||'runs/credit-reset-01/cloud';
 const audit=JSON.parse(fs.readFileSync(path.join(root,'audit.json'))),arms=Object.keys(audit.arms);
 const views=Object.fromEntries(arms.map(k=>[k,JSON.parse(fs.readFileSync(path.join(root,k,'view.json')))]));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100},locale:'en-US'}),errors=[];let writes=0,bins=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766',prefix=process.env.FLY_RESET_PREFIX||'creditreset01-';
  for(const name of arms){
   const v=views[name],a=audit.protocol.arms[name],compare=a.reference_arm+(a.boundary==='carry'?'_reset_rates':'_carry');
   await page.goto(`${base}/?run=${prefix+name}&step=2&bin=49&neuron=10527&compare=${prefix+compare}`);
   await page.waitForFunction(()=>document.querySelector('#credit-chart svg')&&document.querySelector('#paired-voltage svg'));
   assert.deepEqual(await page.evaluate(async n=>await(await fetch('/api/view?run='+n)).json(),prefix+name),v);
   const target=v.nodes.findIndex(n=>n.id==='10527'),other=views[compare],on=other.nodes.findIndex(n=>n.id==='10527');
   for(let i=0;i<3;i++){
    await page.selectOption('#step',String(i));const frame=v.frames[i];
    const note=await page.locator('#activity-boundary-note').innerText();
    assert(note.includes(i===0?'Fresh dynamics':a.boundary==='carry'?'Carry all activity':'Reset KC and DAN learning traces only'));
    assert((await page.locator('#activity-neuron-state').innerText()).includes('Body 10527'));
    assert.equal(await page.locator('#decision').innerText(),audit.arms[name][i].side);
    for(const b of [0,19,20,49]){
     await page.locator('#bin').fill(String(b));await page.locator('#bin').dispatchEvent('input');
     assert.equal(await page.locator('#credit-chart svg path').count(),3);
     const text=await page.locator('#credit-note').innerText();assert(!text.includes('unavailable'));
     if(i===0||a.boundary==='reset_rates')assert(text.includes('Earlier-image traces: 0;'));
     assert((await page.locator('#paired-note').innerText()).includes(`Body 10527: ${frame.counts[b][target]} vs ${other.frames[i].counts[b][on]} spikes in this bin`));
     bins++;
    }
   }
  }
  const name='trained_online_recorded_reset_rates';
  await page.goto(`${base}/?run=${prefix+name}&step=2&bin=37&neuron=10527&edge=10516644&compare=${prefix}trained_online_recorded_carry`);
  await page.waitForFunction(()=>document.querySelector('#paired-voltage svg'));
  if(process.env.FLY_RESET_SCREENSHOT)await page.locator('#paired-traces').screenshot({path:process.env.FLY_RESET_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({recordings:arms.length,observations:arms.length*3,selectedBins:bins,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
