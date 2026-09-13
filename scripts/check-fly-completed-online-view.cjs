/* Check the actual completed-study server against its audited view files. */
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {chromium}=require('playwright');

(async()=>{
 const root=process.env.FLY_STUDY_ROOT||'runs/online-complete-audit-11';
 const out=process.env.FLY_VIEW_CHECK_OUT||'runs/online-complete-browser-11';
 const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8767';
 const report=JSON.parse(fs.readFileSync(path.join(root,'report.json')));
 assert.equal(report.status,'paper_online_completion_audited');assert.equal(report.audited,true);
 assert.equal(Object.keys(report.audit_sha256).length,16);
 assert(!fs.existsSync(out),'Preserve previous browser evidence');fs.mkdirSync(out,{recursive:true});
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[],checks=[];let submissions=0;
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/api/run',r=>{submissions++;return r.abort();});
  const views={};
  for(const name of Object.keys(report.audit_sha256)){
   views[name]=JSON.parse(fs.readFileSync(path.join(root,'views',name,'view.json')));
   const response=await page.request.get(base+'/api/view?run='+encodeURIComponent(name));
   assert.equal(response.status(),200);assert.deepEqual(await response.json(),views[name]);
  }
  for(const stage of ['development','test'])for(const pool of [0,1]){
   const name=`${stage}-pool${pool}-trained_online_reset_rates`,other=`${stage}-pool${pool}-trained_online_carry`;
   const v=views[name],o=views[other],neuron='10527';
   assert.deepEqual(v.frames.map(f=>f.event.market_decision_ts),o.frames.map(f=>f.event.market_decision_ts));
   assert.deepEqual(v.frames.map(f=>f.event.input_sha256),o.frames.map(f=>f.event.input_sha256));
   const vi=v.nodes.findIndex(n=>n.id===neuron),oi=o.nodes.findIndex(n=>n.id===neuron);assert(vi>=0&&oi>=0);
   const common=v.edges.filter(e=>e.plastic_index!==null&&o.edges.some(x=>x.id===e.id&&x.plastic_index!==null));
   assert(common.length);const edge=(common.find(e=>e.id===4110156)||common[0]).id;
   const changed=v.frames.findIndex((f,i)=>f.event.side!==o.frames[i].event.side);
   const steps=[...new Set([0,changed<0?0:changed,v.frames.length-1])];
   for(const step of steps){
    const f=v.frames[step],of=o.frames[step];let bin=f.counts.findIndex((row,i)=>row[vi]!==of.counts[i][oi]);if(bin<0)bin=49;
    const url=`${base}/?run=${name}&compare=${other}&step=${step}&bin=${bin}&neuron=${neuron}&edge=${edge}&pristine=1`;
    await page.goto(url);
    await page.waitForFunction(({name,other,n})=>document.querySelector('#status').textContent.includes('Loaded '+name)
      &&document.querySelector('#comparison').textContent.includes('Other: '+other)
      &&document.querySelector('#comparison').textContent.includes(n+' shared decision timestamps'),{name,other,n:v.frames.length});
    assert.equal(await page.locator('#decision').innerText(),f.event.side);
    assert.equal(await page.locator('#source-badge').innerText(),'Market replay · simulated fills');
    assert(await page.locator('#run').isDisabled());
    assert((await page.locator('#phase-note').innerText()).startsWith(stage==='test'?'Test evaluation:':'Development evaluation:'));
    const note=await page.locator('#paired-note').innerText();
    assert(note.includes(`Body ${neuron}: ${f.counts[bin][vi]} vs ${of.counts[bin][oi]} spikes in this bin`));
    assert(note.includes('Input identical: yes'));
    assert(!(await page.locator('#paired-memory-note').innerText()).includes('unavailable'));
    assert.equal(await page.locator('#credit-chart svg path').count(),3);
    const linked=new URL(await page.locator('#moment-link').getAttribute('href'),base);
    assert.equal(linked.searchParams.get('run'),name);assert.equal(linked.searchParams.get('compare'),other);
    assert.equal(linked.searchParams.get('step'),String(step));assert.equal(linked.searchParams.get('bin'),String(bin));
    if(step===(changed<0?0:changed))await page.locator('#paired-traces').screenshot({path:path.join(out,`${stage}-pool${pool}-pair.png`)});
    checks.push({url,stage,pool,step,bin,neuron,edge,current_action:f.event.side,comparison_action:of.event.side});
   }
  }
  await page.setViewportSize({width:390,height:844});
  assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.equal(submissions,0);assert.deepEqual(errors,[]);
  fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({status:'complete_study_views_checked',
   served_views:16,selected_observations:checks.length,checks,model_submissions:submissions,errors,mobile_overflow:false},null,2)+'\n');
  console.log(JSON.stringify({served_views:16,selected_observations:checks.length,model_submissions:submissions,errors}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
