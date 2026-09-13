/* Read-only browser check for the raw-verified study 11 baton trade. */
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {chromium}=require('playwright');
(async()=>{
 const folder='runs/trade-trace-11',caseFile=folder+'/selected-case.json';
 const evidence=JSON.parse(fs.readFileSync(caseFile));
 const name='test-pool1-trained_online_reset_rates',other='test-pool1-trained_online_carry';
 const current=evidence.chunks[name],comparison=evidence.chunks[other];
 assert.equal(current.side,'BUY');assert.equal(comparison.side,'HOLD');
 assert.equal(current.decoder_counts_per_bin['10527'][43],1);
 assert.equal(comparison.decoder_counts_per_bin['10527'][43],0);
 const url=`http://127.0.0.1:8767/?run=${name}&compare=${other}&step=3&bin=43&neuron=10527`;
 const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}});let submissions=0;const errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  await page.route('**/api/run',r=>{submissions++;return r.abort();});
  await page.goto(url);
  await page.waitForFunction(({name,other})=>document.querySelector('#status').textContent.includes('Loaded '+name)
    &&document.querySelector('#comparison').textContent.includes('Other: '+other),{name,other});
  assert.equal(await page.locator('#decision').innerText(),'BUY');
  assert.equal(await page.locator('#step').inputValue(),'3');
  assert.equal(await page.locator('#bin').inputValue(),'43');
  assert(await page.locator('#run').isDisabled());
  const note=await page.locator('#paired-note').innerText();
  assert(note.includes('Body 10527: 1 vs 0 spikes in this bin'));
  assert(note.includes('Input identical: yes'));
  const decoder=await page.locator('#decoder').innerText();
  assert(decoder.includes('42')&&decoder.includes('44'));
  const timeline=await page.locator('#market-panel').innerText();
  assert(timeline.includes('22:50:00')&&timeline.includes('22:53:47'));
  await page.locator('#paired-traces').screenshot({path:folder+'/selected-trade-pair.png'});
  assert.equal(submissions,0);assert.deepEqual(errors,[]);
  const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
  fs.writeFileSync(folder+'/browser.json',JSON.stringify({status:'selected_trade_view_checked',url,
    case_sha256:sha(caseFile),screenshot_sha256:sha(folder+'/selected-trade-pair.png'),
    script_sha256:sha(__filename),current_action:'BUY',comparison_action:'HOLD',
    current_gate_bin_count:1,comparison_gate_bin_count:0,bin:43,observation:3,
    model_submissions:submissions,errors},null,2)+'\n');
  console.log('Selected trade, paired gate bin, decoder and execution timeline verified; no model submissions.');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
