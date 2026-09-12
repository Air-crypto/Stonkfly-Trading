/* Inspect recorded pulse boundaries without running a model or sending orders. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const root='examples/fly-debugger',views=new Map();
 for(const name of fs.readdirSync(root).filter(n=>/^(pulse01-|stimulation01-)/.test(n)))views.set(name,JSON.parse(fs.readFileSync(path.join(root,name,'view.json'))));
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1050},locale:'en-US'}),errors=[];let writes=0,checked=0,fixture=null;
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/api/**',r=>{const u=new URL(r.request().url());
   if(u.pathname==='/api/run'){writes++;return r.abort();}
   if(u.pathname==='/api/runs')return r.fulfill({json:{runs:[...views.keys()],can_run:false}});
   if(u.pathname==='/api/status')return r.fulfill({json:{status:'idle',run:null}});
   if(u.pathname==='/api/view'){const v=fixture||views.get(u.searchParams.get('run'));assert(v);return r.fulfill({json:v});}
   errors.push('Unexpected API request '+u.pathname);return r.abort();
  });
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766';
  const open=async(name,step=0,bin=0)=>{await page.goto(`${base}/?run=${name}&step=${step}&bin=${bin}`);await page.waitForFunction(name=>document.querySelector('#status').textContent.startsWith('Loaded '+name),name);};
  for(const [name,v] of views){
   await open(name);
   for(let i=0;i<v.frames.length;i++){
    await page.selectOption('#step',String(i));const e=v.frames[i].event;
    for(const b of [0,19,20,49]){
     await page.locator('#bin').fill(String(b));await page.locator('#bin').dispatchEvent('input');
     assert.equal(await page.locator('#pulse-timing-track [data-input-active]').getAttribute('data-input-active'),String(e.stimulus!=='none'&&b*10<e.stimulus_ms));
     assert((await page.locator('#pulse-timing-note').innerText()).includes(`Selected bin ${b*10}–${b*10+10} ms`));
     if(e.stimulation){
      assert.equal(await page.locator('#current-timing-track [data-input-active]').getAttribute('data-input-active'),String(e.stimulation.current!==0));
      assert((await page.locator('#current-timing-note').innerText()).includes('cells '+e.stimulation.target_ids.join(', ')));
     }else{assert.equal(await page.locator('#current-timing-track svg').count(),0);assert((await page.locator('#current-timing-note').innerText()).includes('unavailable'));}
     checked++;
    }
   }
  }
  const name='pulse01-trained_online_recorded',original=views.get(name);
  for(const corrupt of [f=>delete f.event.stimulus_ms,f=>f.event.stimulus_ms=600,f=>f.event.stimulus='none',f=>f.times_ms[0]+=1]){
   fixture=structuredClone(original);corrupt(fixture.frames[1]);await open(name,1,19);
   assert.equal(await page.locator('#pulse-timing-track svg').count(),0);assert((await page.locator('#pulse-timing-note').innerText()).includes('unavailable'));
  }
  fixture=null;await open(name,1,19);
  assert((await page.locator('#pulse-timing-note').innerText()).includes('Aversive pulse · 0–200 ms. Selected bin 190–200 ms: input applied'));
  if(process.env.FLY_INPUT_TIMING_SCREENSHOT)await page.locator('#input-timing').screenshot({path:process.env.FLY_INPUT_TIMING_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);console.log(JSON.stringify({recordings:views.size,binChecks:checked,invalidTimingChecks:4,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
