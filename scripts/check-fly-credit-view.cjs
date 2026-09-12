/* Exercise actual recorded drive values; block all model submissions. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const arms=['trained_online_recorded','trained_online_none','trained_frozen_recorded'];
 const root=process.env.FLY_CREDIT_ROOT||'examples/fly-debugger';
 const views=Object.fromEntries(arms.map(k=>[k,JSON.parse(fs.readFileSync(path.join(root,'credit01-'+k,'view.json')))]));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050},locale:'en-US'}),errors=[];let writes=0,checks=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766';
  const open=async arm=>{await page.goto(`${base}/?run=credit01-${arm}&step=2&bin=39&edge=11483030`);await page.waitForFunction(()=>document.querySelector('#credit-chart svg'));};
  const precise=x=>x===0?'0':Math.abs(x)<.00001?x.toExponential(3):Number(x).toLocaleString('en-US',{maximumFractionDigits:6});
  for(const arm of arms){
   await open(arm);const v=views[arm];
   assert.deepEqual(await page.evaluate(async name=>await(await fetch('/api/view?run='+name)).json(),'credit01-'+arm),v);
   await page.locator('details:has(#provenance)>summary').click();assert((await page.locator('#provenance').innerText()).includes('credit_audit'));await page.locator('details:has(#provenance)>summary').click();
   const plastic=v.edges.filter(e=>e.plastic_index!==null),es=[0,plastic.findIndex(e=>e.id==='11483030')].filter(i=>i>=0);
   for(const ei of new Set(es)){
    await page.selectOption('#edge',String(ei));const pi=v.plastic_selection.indexOf(plastic[ei].plastic_index);
    for(let s=0;s<3;s++){
     await page.selectOption('#step',String(s));const c=v.frames[s].plastic_credit;
     for(const b of [0,19,20,39,49]){
      await page.locator('#bin').fill(String(b));await page.locator('#bin').dispatchEvent('input');
      const note=await page.locator('#credit-note').innerText();
      assert(note.includes(`Earlier-image traces: ${precise(c.earlier[b][pi])}`));
      assert(note.includes(`current-image traces: ${precise(c.current[b][pi])}`));
      assert(note.includes(`total drive: ${precise(c.total[b][pi])} u/s`));
      assert(note.includes(`weight step: ${precise(c.weight_step[b][pi])}`));
      if(!c.learning)assert(note.includes('Memory frozen'));
      assert.equal(await page.locator('#credit-chart svg path').count(),3);checks++;
     }
    }
   }
  }
  const original=views.trained_online_recorded;
  for(const mutation of ['missing','selection','time','sum','nonfinite','weight','learning','provenance']){
   const v=structuredClone(original),c=v.frames[0].plastic_credit;
   if(mutation==='missing')delete v.frames[0].plastic_credit;
   if(mutation==='selection')c.plastic_selection[0]+=1;
   if(mutation==='time')c.times_ms[0]+=1;
   if(mutation==='sum')c.total[0][0]+=1;
   if(mutation==='nonfinite')c.current[0][0]=null;
   if(mutation==='weight')c.weight_step[0][0]+=1;
   if(mutation==='learning')c.learning=false;
   if(mutation==='provenance')delete v.credit_audit;
   await page.evaluate(v=>load(v),v);
   assert((await page.locator('#credit-note').innerText()).includes('unavailable'),mutation);
   assert.equal(await page.locator('#credit-chart svg').count(),0);
  }
  await open('trained_online_recorded');
  // This observed bin has positive drive but a negative actual weight step.
  const note=await page.locator('#credit-note').innerText();assert(note.includes('6.39312 u/s'));assert(note.includes('weight step: -0.004844'));
  if(process.env.FLY_CREDIT_SCREENSHOT)await page.locator('#credit-panel').screenshot({path:process.env.FLY_CREDIT_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({recordings:3,observations:9,selectedBins:checks,invalidCases:8,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
