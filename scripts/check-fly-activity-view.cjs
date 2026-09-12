/* Run from the repository root; requires a recording server and Playwright. No jobs are submitted. */
const assert=require('node:assert/strict'),fs=require('fs'),{chromium}=require('playwright');
(async()=>{const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});try{
 const page=await browser.newPage({viewport:{width:1440,height:1050}}),errors=[];let writes=0;
 page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
 const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8765',names=fs.readdirSync('examples/fly-debugger').filter(n=>n.startsWith('activity01-'));assert.equal(names.length,8);
 for(const name of names){
  const v=JSON.parse(fs.readFileSync(`examples/fly-debugger/${name}/view.json`));assert.equal(v.frames.length,3);
  await page.goto(`${base}/?run=${name}&step=2&bin=49`);await page.waitForFunction(()=>document.querySelector('#step').value==='2'&&!document.querySelector('#activity-boundary').hidden);
  for(let i=0;i<3;i++){
   await page.selectOption('#step',String(i));await page.locator('#bin').fill('49');await page.locator('#bin').dispatchEvent('input');
   const e=v.frames[i].event,b=e.activity_boundary,note=await page.locator('#activity-boundary-note').innerText();
   assert.equal(await page.locator('#decision').innerText(),e.side);assert(note.includes(`${b.before_clock.sim_ms.toLocaleString('en-US')} → ${b.after_clock.sim_ms.toLocaleString('en-US')} ms`));
   assert(note.includes('Target fields: '+(b.target_fields.join(', ')||'none')));assert(note.includes(b.memory_sha256.slice(0,16)));assert.equal(await page.locator('#decoder-path svg').count(),2);
   assert.equal(await page.locator('#time').innerText(),e.brain_ms.toLocaleString('en-US')+' ms');
  }
 }
 const current='activity01-trained_full',other='activity01-trained_carry';
 await page.goto(`${base}/?run=${current}&step=2&bin=49&neuron=10527&compare=${other}`);await page.waitForFunction(()=>document.querySelector('#paired-spikes svg'));
 assert((await page.locator('#paired-note').innerText()).includes('Input identical: yes'));
 const [a,b]=[current,other].map(n=>JSON.parse(fs.readFileSync(`examples/fly-debugger/${n}/view.json`)));
 const counts=v=>v.frames[2].counts.map(row=>row[v.nodes.findIndex(n=>n.id==='10527')]);
 const curves=await page.locator('#paired-spikes svg path').evaluateAll(ns=>ns.map(n=>n.getAttribute('d')));
 if(JSON.stringify(counts(a))===JSON.stringify(counts(b)))assert.equal(curves[0],curves[1]);else assert.notEqual(curves[0],curves[1]);
 assert((await page.locator('#activity-boundary-note').innerText()).includes('500 → 0 ms'));
 const trigger=counts(a).findIndex((n,i)=>n>counts(b)[i]);if(trigger>=0){await page.locator('#bin').fill(String(trigger));await page.locator('#bin').dispatchEvent('input');}
 if(process.env.FLY_ACTIVITY_SCREENSHOTS){await page.locator('#activity-boundary').screenshot({path:process.env.FLY_ACTIVITY_SCREENSHOTS+'/fly-activity-boundary.png'});await page.locator('#paired-traces').screenshot({path:process.env.FLY_ACTIVITY_SCREENSHOTS+'/fly-activity-gate.png'});await page.locator('.circuit').screenshot({path:process.env.FLY_ACTIVITY_SCREENSHOTS+'/fly-activity-circuit.png'});}
 await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
 await page.goto(base+'/?run=market07-pool1-trained_frozen');await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('Loaded market07-'));
 assert(await page.locator('#activity-boundary').isHidden());assert.equal(await page.locator('#activity-boundary-note').innerText(),'');
 assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({checks:'all 24 state boundaries, exact clocks, native outputs, reset fields, preserved memory identity, paired gate counts, mobile, legacy clearing',errors,writes}));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1;});
