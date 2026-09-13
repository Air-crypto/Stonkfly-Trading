/* Verify the actual five-condition viewer without submitting model work. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');
(async()=>{
 const root=process.env.FLY_SCALE_ROOT||'runs/learning-scale-01/cloud';
 const out=process.env.FLY_VIEW_CHECK_OUT||'runs/learning-scale-browser-01';
 const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8770';
 const full=process.env.FLY_REQUIRE_FULL_RECORDINGS!=='0';
 const audit=JSON.parse(fs.readFileSync(path.join(root,'audit.json'))),receipt=JSON.parse(fs.readFileSync(path.join(root,'cloud-call.json')));
 assert.equal(audit.status,'learning_scale_audited');assert.equal(receipt.status,'completed');
 assert.equal(audit.verification.observations,15);
 assert(!fs.existsSync(out),'Preserve previous browser evidence');fs.mkdirSync(out,{recursive:true});
 const views=Object.fromEntries(Object.keys(audit.arms).map(n=>[n,JSON.parse(fs.readFileSync(path.join(root,n,'view.json')))]));
 const labels={carry:'Carry all activity forward',reset_rates:'Reset KC and DAN learning traces only',
  reset_kc:'Reset KC learning traces only',reset_dan:'Reset DAN learning traces only'};
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[],checks=[];let submissions=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{submissions++;return r.abort();});
  const config=await(await page.request.get(base+'/api/runs')).json();assert.equal(config.can_run,false);
  assert.deepEqual(new Set(config.runs),new Set(Object.keys(views)));
  for(const [name,v] of Object.entries(views)){
   assert.deepEqual(await(await page.request.get(base+'/api/view?run='+name)).json(),v);
   const arm=audit.protocol.arms[name],compare=name.startsWith('trained_low_eta_')?arm.reference_arm+'_carry':name==='trained_frozen_recorded_carry'?'trained_online_recorded_carry':name.replace('trained_online_','trained_low_eta_');
   const other=views[compare],neuron='10527',vi=v.nodes.findIndex(n=>n.id===neuron),oi=other.nodes.findIndex(n=>n.id===neuron);
   assert(vi>=0&&oi>=0);
   const edge=v.edges.find(e=>e.id===10516644&&other.edges.some(x=>x.id===e.id))||
    v.edges.find(e=>e.plastic_index!==null&&other.edges.some(x=>x.id===e.id&&x.plastic_index!==null));
   assert(edge,'A recorded plastic connection shared by the pair is required');
   for(let step=0;step<3;step++){
    const url=`${base}/?run=${name}&compare=${compare}&step=${step}&bin=0&neuron=${neuron}&edge=${edge.id}&pristine=1`;
    await page.goto(url);
    await page.waitForFunction(({name,compare})=>document.querySelector('#status').textContent.includes('Loaded '+name)
      &&document.querySelector('#comparison').textContent.includes('Other: '+compare)
      &&document.querySelector('#paired-voltage svg'),{name,compare});
    assert(await page.locator('#run').isDisabled());
    assert.equal(await page.locator('#decision').innerText(),audit.arms[name][step].side);
    if(arm.learning)assert((await page.locator('#phase-note').innerText()).includes(`eta ${arm.eta.toFixed(4)}`));
    assert((await page.locator('#activity-boundary-note').innerText()).includes(step===0?'Fresh dynamics':labels[arm.boundary]));
    for(const bin of step===2?[0,37,49]:[0,49]){
     await page.locator('#bin').fill(String(bin));await page.locator('#bin').dispatchEvent('input');
     const note=await page.locator('#paired-note').innerText();
     assert(note.includes(`Body ${neuron}: ${v.frames[step].counts[bin][vi]} vs ${other.frames[step].counts[bin][oi]} spikes in this bin`));
     assert(note.includes('Input identical: yes'));
     assert(!(await page.locator('#paired-memory-note').innerText()).includes('unavailable'));
     assert.equal(await page.locator('#credit-chart svg path').count(),3);
     const moment=new URL(await page.locator('#moment-link').getAttribute('href'),base);
     assert.equal(moment.searchParams.get('run'),name);assert.equal(moment.searchParams.get('compare'),compare);
     assert.equal(moment.searchParams.get('step'),String(step));assert.equal(moment.searchParams.get('bin'),String(bin));
     checks.push({name,compare,step,bin,neuron,edge:edge.id,action:audit.arms[name][step].side});
     if(step===2&&bin===37&&name.startsWith('trained_low_eta_'))
      await page.locator('#paired-traces').screenshot({path:path.join(out,arm.pulses+'-pair.png')});
    }
   }
   // This exercises full-neuron NPZ lookup, separately from the selected view.
   const response=await page.request.get(`${base}/api/neuron?run=${name}&id=10527`),raw=await response.json();
   if(full){
    assert.equal(raw.id,'10527');assert.equal(raw.frames.length,3);
    for(let step=0;step<3;step++){
     assert.deepEqual(raw.frames[step].times_ms,v.frames[step].times_ms);
     assert.deepEqual(raw.frames[step].counts,v.frames[step].counts.map(row=>row[vi]));
    }
   }else{
    assert.equal(response.status(),400);assert.equal(raw.error,'Full neuron recordings unavailable in this run');
   }
  }
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.equal(submissions,0);assert.deepEqual(errors,[]);
  const result={status:'learning_scale_views_checked',served_views:5,observations:15,selected_bins:checks.length,
   full_neuron_lookups:full?5:0,explicit_missing_recording_responses:full?0:5,checks,model_submissions:submissions,errors,mobile_overflow:false};
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify({served_views:5,observations:15,selected_bins:checks.length,full_neuron_lookups:full?5:0,model_submissions:submissions,errors}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
