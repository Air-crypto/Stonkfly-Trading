/* Check recorded state-difference navigation with real portable traces and explicit corruption fixtures. */
const assert=require('node:assert/strict'),{chromium}=require('playwright');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1200},locale:'en-US'}),errors=[];let writes=0;
  page.on('pageerror',e=>errors.push(String(e)));await page.route('**/api/run',r=>{writes++;return r.abort();});
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8765';
  const prefix='creditdivergence01-trained_online_recorded_',current=prefix+'reset_rates',other=prefix+'carry';
  const open=async(a=current,b=other)=>{
   await page.goto(`${base}/?run=${a}&step=1&bin=49&neuron=11402&edge=4110156&compare=${b}`);
   await page.waitForFunction(()=>document.querySelectorAll('#paired-state-differences tbody tr').length===4);
  };
  const button=(field,kind)=>page.locator(`#paired-state-differences tr[data-field="${field}"] button[data-jump="${kind}"]`);
  const delta=field=>page.locator(`#paired-state-differences [data-delta="${field}"]`);
  await open();
  const views=await page.evaluate(async names=>Promise.all(names.map(async name=>await(await fetch('/api/view?run='+name)).json())),[current,other]);
  const columns=views.map(v=>({neuron:v.nodes.findIndex(n=>n.id==='11402'),plastic:v.plastic_selection.indexOf(8)}));
  assert.equal(await button('voltage','first').innerText(),'50 ms');
  for(const field of ['weight','u','w'])assert.equal(await button(field,'first').innerText(),'10 ms');
  await button('voltage','first').click();assert.equal(await page.inputValue('#bin'),'4');
  const firstDelta=views[0].frames[1].voltage[4][columns[0].neuron]-views[1].frames[1].voltage[4][columns[1].neuron];
  assert.equal(Number(await delta('voltage').innerText()),Number(firstDelta.toFixed(6)));
  const next=views[0].frames[1].voltage.findIndex((row,i)=>i>4&&row[columns[0].neuron]!==views[1].frames[1].voltage[i][columns[1].neuron]);
  await button('voltage','next').click();assert.equal(Number(await page.inputValue('#bin')),next);
  const link=new URL(await page.locator('#moment-link').getAttribute('href'));
  assert.equal(link.searchParams.get('bin'),String(next));assert.equal(link.searchParams.get('step'),'1');
  await button('weight','first').click();assert.equal(await page.inputValue('#bin'),'0');
  const weightDelta=Number(await delta('weight').innerText());assert(weightDelta>0);
  await button('weight','next').click();assert.equal(await page.inputValue('#bin'),'1');
  await page.click('#play');await button('weight','first').click();
  assert.equal(await page.locator('#play').innerText(),'Play bins');assert.equal(await page.inputValue('#bin'),'0');
  await page.selectOption('#step','0');
  for(const field of ['voltage','weight','u','w']){
   assert.equal(await button(field,'first').innerText(),'None');assert(await button(field,'first').isDisabled());
   assert.equal(await delta(field).innerText(),'0');
  }
  await open(other,current);await button('weight','first').click();assert.equal(Number(await delta('weight').innerText()),-weightDelta);
  await open();await button('voltage','first').click();
  await page.evaluate(()=>document.activeElement.blur());
  if(process.env.FLY_STATE_NAVIGATION_SCREENSHOT)await page.locator('#paired-traces').screenshot({path:process.env.FLY_STATE_NAVIGATION_SCREENSHOT});
  const original=await page.locator('#paired-state-differences').innerText();
  let fixture=structuredClone(views[1]);
  // Reorder both identity maps: array positions are not neuron/connection identities.
  const size=fixture.nodes.length;fixture.nodes.reverse();fixture.edges.forEach(e=>{e.source=size-1-e.source;e.target=size-1-e.target;});
  fixture.plastic_selection.reverse();fixture.frames.forEach(f=>{
   for(const key of ['voltage','counts','plastic_weights','plastic_u','plastic_w'])f[key].forEach(row=>row.reverse());
   f.plastic_initial.reverse();
  });
  await page.route('**/api/view?run='+other,r=>r.fulfill({json:fixture}));
  const compare=async()=>{
   await page.selectOption('#compare','');assert.equal(await page.locator('#paired-state-differences tbody tr').count(),0);
   await page.selectOption('#compare',other);await page.waitForFunction(()=>document.querySelectorAll('#paired-state-differences tbody tr').length===4);
  };
  await compare();assert.equal(await page.locator('#paired-state-differences').innerText(),original);
  fixture=structuredClone(views[1]);fixture.frames.forEach(f=>{delete f.plastic_u;delete f.plastic_w;});
  await compare();
  for(const field of ['u','w']){assert.equal(await button(field,'first').innerText(),'Not recorded');assert(await button(field,'first').isDisabled());assert.equal(await delta(field).innerText(),'Not recorded');}
  assert.equal(await button('weight','first').innerText(),'10 ms');
  fixture=structuredClone(views[1]);const changed=fixture.edges.find(e=>e.id===4110156);changed.source=changed.target;
  await compare();
  for(const field of ['weight','u','w'])assert.equal(await button(field,'first').innerText(),'Not recorded');
  assert.equal(await button('voltage','first').innerText(),'50 ms');
  fixture=structuredClone(views[1]);fixture.frames.forEach(f=>f.event.market_decision_ts+=1);
  await page.selectOption('#compare','');await page.selectOption('#compare',other);
  await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('No shared observation'));
  assert.equal(await page.locator('#paired-state-differences tbody tr').count(),0);
  fixture=structuredClone(views[1]);fixture.frames.forEach(f=>f.times_ms[0]+=1);
  await page.selectOption('#compare','');await page.selectOption('#compare',other);
  await page.waitForFunction(()=>document.querySelector('#paired-note').textContent.includes('different time bins'));
  assert.equal(await page.locator('#paired-state-differences tbody tr').count(),0);
  fixture=views[1];await compare();await page.setViewportSize({width:390,height:844});
  assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);
  console.log(JSON.stringify({recordings:2,checks:['first and next voltage/weight differences','memory first differences','cursor link and playback pause','identical first observations','signed deltas','reordered neuron and plastic indices','missing memory','conflicting connection endpoints','unmatched timestamps','mismatched bins','mobile'],errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
