/* Check saved market/reset recordings through the real UI. Never submit model jobs. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const root=process.env.FLY_INPUT_RECORDINGS||'runs/research-market-08',views=new Map();
 for(const name of fs.readdirSync(root).filter(n=>/^pool\d+-/.test(n))){const file=path.join(root,name,'view.json');if(fs.existsSync(file))views.set(name,JSON.parse(fs.readFileSync(file)));}
 assert(views.size>0,'No completed market recordings');
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050},locale:'en-US'}),errors=[];let writes=0,boundaries=0,slots=0;
  page.on('pageerror',e=>errors.push(String(e)));
  // Only the transport is replaced: every event, quote, fill and neuron value comes from the saved view.json.
  await page.route('**/api/**',r=>{const u=new URL(r.request().url());
   if(u.pathname==='/api/run'){writes++;return r.abort();}
   if(u.pathname==='/api/runs')return r.fulfill({json:{runs:[...views.keys()],can_run:false}});
   if(u.pathname==='/api/status')return r.fulfill({json:{status:'idle',run:null}});
   if(u.pathname==='/api/view'){const view=views.get(u.searchParams.get('run'));assert(view);return r.fulfill({json:view});}
   errors.push('Unexpected API request: '+u.pathname);return r.abort();
  });
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8765',num=x=>Number(x).toLocaleString('en-US',{maximumFractionDigits:5});
  for(const [name,v] of views){
   assert.equal(v.report.source,'sealed_retrospective_market_replay');assert(['carry','conductance'].includes(v.report.config.activity_reset));
   const timeline=v.report.market_timeline,observed=timeline.filter(r=>r.observation==='observed');assert.equal(observed.length,v.frames.length);
   await page.goto(`${base}/?run=${name}`);await page.waitForFunction(name=>document.querySelector('#status').textContent.startsWith('Loaded '+name),name);
   if(v.report.memory_origin){
    const note=await page.locator('#phase-note').innerText();
    assert(note.includes(v.report.evaluation_phase==='development'?'Development evaluation':'Test evaluation'));
    assert(note.includes(v.report.memory_origin==='paper_trained'?'paper-trained connection memory':'pristine connection memory'));
    if(v.report.training_exposure){const x=v.report.training_exposure;assert(note.includes(`${num(x.observations)} observations; ${num(x.positive_rewards)} positive and ${num(x.negative_rewards)} negative paper rewards`));}
    assert((await page.locator('#input-info').innerText()).includes('news available at quote time'));
   }
   assert(await page.locator('#run').isDisabled());assert(!(await page.locator('#market-panel').isHidden()));
   assert((await page.locator('#market-coverage').innerText()).includes(`${observed.length} / ${timeline.filter(r=>!r.terminal).length}`));
   const rows=await page.locator('#market-timeline tr:has(td)').allInnerTexts();assert.equal(rows.length,timeline.length);
   for(let i=0;i<rows.length;i++){
    const row=timeline[i];assert(rows[i].includes(row.observation.replaceAll('_',' ')));assert(rows[i].includes('$'+Number(row.equity).toFixed(2)));
    assert(rows[i].includes(row.fill_reason?`${row.fill_status}: ${row.fill_reason}`:row.fill_status));slots++;
   }
   for(let i=0;i<v.frames.length;i++){
    await page.selectOption('#step',String(i));const f=v.frames[i],e=f.event,b=e.activity_boundary;
    assert.equal(e.market_decision_ts,observed[i].decision_ts);assert.equal(await page.locator('#decision').innerText(),e.side);
    if(e.stimulation)assert((await page.locator('#phase-note').innerText()).includes(`Applied diagnostic current ${e.stimulation.current} to cells 10704, 11402 throughout each 500 ms image`));
    assert.equal(b.observation,i+1);assert.equal(b.mode,i===0?'initial':v.report.config.activity_reset);
    const note=await page.locator('#activity-boundary-note').innerText();assert(note.includes(`${num(b.before_clock.sim_ms)} → ${num(b.after_clock.sim_ms)} ms`));
    const n=v.nodes.findIndex(n=>n.id==='10527');assert(n>=0);await page.selectOption('#node',String(n));
    const s=f.activity_state,value=await page.locator('#activity-neuron-state').innerText();assert(value.includes(`voltage ${num(s.before_v[n])} → ${num(s.after_v[n])} mV`));assert(value.includes(`g ${num(s.before_g[n])} → ${num(s.after_g[n])}`));
    await page.locator('#bin').fill('49');await page.locator('#bin').dispatchEvent('input');assert.equal(await page.locator('#activity-neuron-state').innerText(),value);
    assert.equal(await page.locator('#time').innerText(),num(e.brain_ms)+' ms');assert.equal(await page.locator('#decoder-path svg').count(),2);boundaries++;
   }
  }
  for(const [name,v] of views){if(!/-trained_input_reset(?:-development)?$/.test(name))continue;
   const other=name.replace(/-trained_input_reset(?=-development$|$)/,'-trained_frozen');if(!views.has(other))continue;
   await page.goto(`${base}/?run=${name}&step=${v.frames.length-1}&bin=49&neuron=10527&compare=${other}`);
   await page.waitForFunction(()=>document.querySelector('#paired-spikes svg'));assert((await page.locator('#paired-note').innerText()).includes('Input identical: yes'));
   assert((await page.locator('#paired-note').innerText()).includes('Recorded input prefix: identical'));
   if(v.report.memory_origin){await page.locator('#comparison summary').click();assert((await page.locator('#comparison pre').innerText()).includes('training_exposure'));}
  }
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({recordings:views.size,slots,boundaries,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
