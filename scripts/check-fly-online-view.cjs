/* Inspect a complete long recording; browser requests cannot submit compute. */
const assert=require('node:assert/strict'),fs=require('fs'),path=require('path'),{chromium}=require('playwright');
(async()=>{
 const file=process.env.FLY_ONLINE_VIEW||'runs/online-native-check-02/offline-audit/view.json';
 const v=JSON.parse(fs.readFileSync(file));assert.equal(v.validation_only,true);
 const frozen=JSON.parse(fs.readFileSync('examples/fly-debugger/market10-pool0-trained_frozen/view.json'));
 assert(v.frames.length>=18);assert(fs.statSync(file).size>15000000);
 const browser=await chromium.launch({headless:true,...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{})});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];let writes=0,bins=0;
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/api/run',r=>{writes++;return r.abort();});
  await page.route('**/api/runs',r=>r.fulfill({json:{runs:['synthetic-online-validation','frozen-control'],can_run:false,backend:'recordings'}}));
  await page.route('**/api/status',r=>r.fulfill({json:{run:null,status:'idle'}}));
  await page.route('**/api/view?*',r=>r.fulfill({json:new URL(r.request().url()).searchParams.get('run')==='frozen-control'?frozen:v}));
  const base=process.env.FLY_VIEW_URL||'http://127.0.0.1:8766';
  await page.goto(base+'/?run=synthetic-online-validation&step=0&bin=0&neuron=10527');
  await page.waitForFunction(()=>document.querySelector('#credit-chart svg'));
  assert.equal(await page.locator('#source-badge').innerText(),'Synthetic prices · implementation check');
  assert((await page.locator('#market-coverage').innerText()).includes(v.frames.length+' / 24 decision slots observed'));
  assert.equal(await page.locator('#market-timeline tr').count(),26);
  for(let i=0;i<v.frames.length;i++){
   await page.selectOption('#step',String(i));
   const note=await page.locator('#phase-note').innerText();
   assert(note.includes('updated online using this phase’s paper equity feedback'));
   assert(note.includes('only KC/DAN rate traces cleared'));
   assert(!note.includes('frozen throughout'));
   assert.equal(await page.locator('#decision').innerText(),v.frames[i].event.side);
   for(const b of [0,19,20,49]){
    await page.locator('#bin').fill(String(b));await page.locator('#bin').dispatchEvent('input');
    assert.equal(await page.locator('#credit-chart svg path').count(),3);
    assert((await page.locator('#credit-note').innerText()).includes('Earlier-image traces: 0;'));bins++;
   }
  }
  // The file input previously rejected this >15 MB recording.
  await page.locator('#import').setInputFiles(file);
  // Existing options still belong to the server-loaded recording while file.text()
  // is pending. Wait for this import's identity before selecting its final frame.
  await page.waitForFunction(({name,count})=>document.querySelector('#status').textContent
   .startsWith(`Loaded ${name} · ${count} observations`),{name:path.basename(file),count:v.frames.length});
  await page.selectOption('#step',String(v.frames.length-1));
  assert.equal(await page.locator('#error').innerText(),'');
  assert.equal(await page.locator('#decision').innerText(),v.frames.at(-1).event.side);
  await page.goto(base+'/?run=frozen-control&step=0&bin=0');
  await page.waitForFunction(()=>document.querySelector('#phase-note').textContent.includes('frozen throughout'));
  assert(!(await page.locator('#phase-note').innerText()).includes('updated online'));
  await page.goto(base+'/?run=synthetic-online-validation&step=20&bin=49&neuron=10527');
  await page.waitForFunction(()=>document.querySelector('#credit-chart svg'));
  if(process.env.FLY_ONLINE_SCREENSHOT)await page.screenshot({path:process.env.FLY_ONLINE_SCREENSHOT});
  await page.setViewportSize({width:390,height:844});
  assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)));
  assert.deepEqual(errors,[]);assert.equal(writes,0);
  console.log(JSON.stringify({observations:v.frames.length,selectedBins:bins,decisionSlots:24,terminalMarks:1,largeImport:true,frozenRegression:true,errors,writes}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
