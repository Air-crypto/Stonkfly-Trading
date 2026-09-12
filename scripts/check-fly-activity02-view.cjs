/* Saved recordings only. Intercept every job submission during browser QA. */
const assert=require('node:assert/strict'),fs=require('fs'),{chromium}=require('playwright');
(async()=>{const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});try{
 const page=await browser.newPage({viewport:{width:1440,height:1050},locale:'en-US'}),errors=[];let writes=0;
 page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
 const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8765',names=fs.readdirSync('examples/fly-debugger').filter(n=>n.startsWith('activity02-'));assert.equal(names.length,12);
 const read=n=>JSON.parse(fs.readFileSync(`examples/fly-debugger/${n}/view.json`)),num=x=>Number(x).toLocaleString('en-US',{maximumFractionDigits:5});
 async function checkState(v,step,id){
  const ix=v.nodes.findIndex(n=>n.id===id);assert(ix>=0);await page.selectOption('#node',String(ix));
  const s=v.frames[step].activity_state,note=await page.locator('#activity-neuron-state').innerText();
  assert(note.includes('Body '+id+' at this boundary'));assert(note.includes(`voltage ${num(s.before_v[ix])} → ${num(s.after_v[ix])} mV`));assert(note.includes(`g ${num(s.before_g[ix])} → ${num(s.after_g[ix])} (model units)`));
  await page.locator('#bin').fill('49');await page.locator('#bin').dispatchEvent('input');assert.equal(await page.locator('#activity-neuron-state').innerText(),note);
 }
 for(const name of names){
  const v=read(name);assert.equal(v.frames.length,3);
  await page.goto(`${base}/?run=${name}&step=2&bin=49`);await page.waitForFunction(()=>document.querySelector('#step').value==='2'&&!document.querySelector('#activity-boundary').hidden);
  for(let i=0;i<3;i++){
   await page.selectOption('#step',String(i));const e=v.frames[i].event,b=e.activity_boundary,note=await page.locator('#activity-boundary-note').innerText();
   assert.equal(await page.locator('#decision').innerText(),e.side);assert(note.includes(`${num(b.before_clock.sim_ms)} → ${num(b.after_clock.sim_ms)} ms`));assert(note.includes('Target fields: '+(b.target_fields.join(', ')||'none')));
   assert(note.includes(b.memory_sha256.slice(0,16)));if(b.target_ids.length)assert(note.includes('neurons '+b.target_ids.join(', ')));
   for(const id of ['10527','555871','11402'])await checkState(v,i,id);
  }
 }
 const name='activity02-trained_gate_voltage_conductance',v=read(name);
 await page.goto(`${base}/?run=${name}&step=2&neuron=10527&compare=activity02-trained_carry`);await page.waitForFunction(()=>document.querySelector('#paired-spikes svg'));
 assert((await page.locator('#paired-note').innerText()).includes('Input identical: yes'));await checkState(v,2,'10527');
 // Mock only the existing lookup transport with values from the actual saved compact recording.
 await page.route('**/api/neuron?*',r=>{const id=new URL(r.request().url()).searchParams.get('id'),ix=v.nodes.findIndex(n=>n.id===id);assert(ix>=0);return r.fulfill({json:{id,frames:v.frames.map(f=>({times_ms:f.times_ms,counts:f.counts.map(row=>row[ix]),voltage:f.voltage.map(row=>row[ix]),plastic_edges:[]}))}});});
 await page.locator('#lookup').fill('10059');await page.locator('#lookup-button').click();await page.waitForFunction(()=>document.querySelector('#activity-neuron-state').textContent.includes('Body 10059'));
 const ix=v.nodes.findIndex(n=>n.id==='10059');assert((await page.locator('#activity-neuron-state').innerText()).includes(`voltage ${num(v.frames[2].activity_state.before_v[ix])} → ${num(v.frames[2].activity_state.after_v[ix])}`));
 await checkState(v,2,'10527');
 if(process.env.FLY_ACTIVITY_SCREENSHOTS){await page.locator('#activity-boundary').screenshot({path:process.env.FLY_ACTIVITY_SCREENSHOTS+'/fly-activity02-boundary.png'});await page.locator('#paired-traces').screenshot({path:process.env.FLY_ACTIVITY_SCREENSHOTS+'/fly-activity02-gate.png'});}
 const recovered='activity02-trained_conductance',rv=read(recovered),cv=read('activity02-trained_carry');
 await page.goto(`${base}/?run=${recovered}&step=2&bin=2&neuron=10527&edge=10020213&compare=activity02-trained_carry`);await page.waitForFunction(()=>document.querySelector('#paired-spikes svg'));
 const counts=w=>w.frames[2].counts.map(row=>row[w.nodes.findIndex(n=>n.id==='10527')]);assert.notDeepEqual(counts(rv),counts(cv));
 assert.equal(await page.locator('#paired-traces svg').count(),5);
 const curves=await page.locator('#paired-spikes svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));assert.notEqual(curves[0],curves[1]);
 assert((await page.locator('#paired-note').innerText()).includes('Input identical: yes'));
 if(process.env.FLY_ACTIVITY_SCREENSHOTS)await page.locator('#paired-traces').screenshot({path:process.env.FLY_ACTIVITY_SCREENSHOTS+'/fly-activity02-recovered.png'});
 await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 await page.goto(base+'/?run=activity01-trained_full');await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Loaded activity01-'));assert((await page.locator('#activity-neuron-state').innerText()).includes('unavailable'));
 await page.goto(base+'/?run=market07-pool1-trained_frozen');await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Loaded market07-'));assert(await page.locator('#activity-boundary').isHidden());assert.equal(await page.locator('#activity-neuron-state').innerText(),'');
 assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({checks:'36 boundaries, 108 neuron states, fixed boundary values while moving time, gate scope, lookup identity, paired traces, mobile and legacy clearing',errors,writes}));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
