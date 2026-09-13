'use strict';
const $=id=>document.getElementById(id);
const historical=report=>['sealed_retrospective_market_replay','matched_market_stimulus_assay'].includes(report.source);
const phaseKey=(f,i)=>`${f.event.phase||'assay'}:${f.event.phase_step??i+1}`;
const inputHash=f=>typeof f.event.input_sha256==='string'&&/^[a-f0-9]{64}$/.test(f.event.input_sha256)?f.event.input_sha256:null;
function observationPairs(current,other,market){
 const keys=view=>view.frames.map((f,i)=>market?(Number.isFinite(f.event.market_decision_ts)?f.event.market_decision_ts:null):phaseKey(f,i));
 const a=keys(current),b=keys(other),counts=keys=>{const result=new Map();for(const k of keys)if(k!==null)result.set(k,(result.get(k)||0)+1);return result;};
 const ac=counts(a),bc=counts(b),indices=new Map(b.map((k,i)=>[k,i])),matched=[],ambiguous=new Set();
 for(let i=0;i<a.length;i++){
  const key=a[i];if(key===null||!indices.has(key))continue;
  if(ac.get(key)!==1||bc.get(key)!==1){ambiguous.add(key);continue;}
  const j=indices.get(key),prefixA=current.frames.slice(0,i+1),prefixB=other.frames.slice(0,j+1);
  const unknown=prefixA.some((f,k)=>!inputHash(f)||a[k]===null||ac.get(a[k])!==1)||prefixB.some((f,k)=>!inputHash(f)||b[k]===null||bc.get(b[k])!==1);
  const history=unknown?'unverified':i===j&&prefixA.every((f,k)=>a[k]===b[k]&&inputHash(f)===inputHash(prefixB[k]))?'identical':'different';
  matched.push([current.frames[i],other.frames[j],i,history]);
 }
 return {matched,ambiguous:ambiguous.size};
}
let data=null, step=0, bin=0, node=0, edge=0, timer=null, runName='', positions=[], imported=false, viewRequest=0, externalNeuron=false, paired=null;
const colors={visual:'#8fbcff',KC:'#b2bfd1',DAN:'#e6a8e8',MBON:'#ffbf69',output:'#67e8cf',other:'#b2bfd1'};
const num=(x,d=3)=>Number(x).toLocaleString(undefined,{maximumFractionDigits:d});
const fmt=x=>x===null?'Not defined':typeof x==='number'?num(x):String(x);
function error(e){$('error').textContent=e?.message||String(e);}
async function api(path, options){const r=await fetch(path,options);const body=await r.json();if(!r.ok)throw new Error(body.error||r.statusText);return body;}
function options(el, values){el.replaceChildren(...values.map(([v,label])=>{const o=document.createElement('option');o.value=v;o.textContent=label;return o;}));}
function dl(el,rows){el.replaceChildren();for(const [k,v] of rows){let dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=k;dd.textContent=fmt(v);el.append(dt,dd);}}
function chart(id, xs, series, unit, cursor, xLabel='Neural time (ms)'){
 const el=$(id),w=Math.max(250,el.clientWidth),h=175,left=60,right=w-16,top=16,bottom=137;
 const values=series.flatMap(s=>s.values);let ymin=Math.min(...values),ymax=Math.max(...values);if(ymin===ymax){ymin-=1;ymax+=1;}const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
 const xmin=xs[0],xmax=xs.at(-1),X=x=>left+(x-xmin)/(xmax-xmin||1)*(right-left),Y=y=>bottom-(y-ymin)/(ymax-ymin)*(bottom-top);
 const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.setAttribute('role','img');svg.setAttribute('aria-label',`${unit} over neural time in milliseconds`);
 const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;};
 add('rect',{x:left,y:top,width:right-left,height:bottom-top,fill:'none',stroke:'#35445c'});
 const tickDigits=Math.min(6,Math.max(2,Math.ceil(-Math.log10((ymax-ymin)/2))+1));
 for(let k=0;k<3;k++){const v=ymin+(ymax-ymin)*k/2;add('text',{x:left-8,y:Y(v)+4,'text-anchor':'end',fill:'#b2bfd1','font-size':11},num(Math.abs(v)<.5*10**(-tickDigits)?0:v,tickDigits));}
 for(let k=0;k<3;k++){const x=xmin+(xmax-xmin)*k/2;add('text',{x:X(x),y:bottom+17,'text-anchor':k===0?'start':k===2?'end':'middle',fill:'#b2bfd1','font-size':11},num(x));}
 add('text',{x:left,y:12,fill:'#b2bfd1','font-size':11},unit);add('text',{x:(left+right)/2,y:173,'text-anchor':'middle',fill:'#b2bfd1','font-size':11},xLabel);
 for(const s of series)add('path',{d:s.values.map((v,i)=>`${i?'L':'M'}${X(xs[i])},${Y(v)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':1.7});
 if(cursor!==undefined)add('line',{x1:X(xs[cursor]),x2:X(xs[cursor]),y1:top,y2:bottom,stroke:'#e8edf6','stroke-dasharray':'3 4'});
 const legend=document.createElement('p');legend.className='small';legend.textContent=series.map(s=>s.name).join(' / ');el.replaceChildren(svg,legend);
}
function restoredEdges(f){return f.event.phase==='probe'?data.report.probe_boundary?.restoration?.edge_ids||[]:data.report.restoration?.edge_ids||[];}
function network(f){
 const svg=$('network'),rect=svg.getBoundingClientRect(),w=rect.width,h=rect.height;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.replaceChildren();
 const ns='http://www.w3.org/2000/svg',add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);return n;};
 const groups=['visual','KC','DAN','MBON','output','other'].filter(g=>data.nodes.some(n=>n.group===g));positions=Array(data.nodes.length);
 for(const[gidx,g]of groups.entries()){const members=data.nodes.map((n,i)=>n.group===g?i:-1).filter(i=>i>=0);const x=25+(w-50)*gidx/(groups.length-1||1);add('text',{x,y:16,fill:colors[g],'font-size':11,'text-anchor':gidx===0?'start':gidx===groups.length-1?'end':'middle'},g==='visual'?'Visual':g==='output'?'Decoder':g);members.forEach((i,j)=>positions[i]=[x+(j%2?4:-4),40+(h-60)*(j+.5)/members.length]);}
 const pmap=new Map(data.plastic_selection.map((v,i)=>[v,i]));
 const restored=new Set(restoredEdges(f)),selected=data.edges.filter(e=>e.plastic_index!==null)[edge];
 // Keep memory color and source-spike highlighting independent. Draw the selected
 // connection last so it remains inspectable among overlapping connections.
 const ordered=selected?[...data.edges.filter(e=>e!==selected),selected]:data.edges;
 for(const e of ordered){
  const [x,y]=positions[e.source],[tx,ty]=positions[e.target],pi=pmap.get(e.plastic_index);
  const changed=pi!==undefined&&Math.abs(f.plastic_weights[bin][pi]-($('pristine').checked?e.weight:f.plastic_initial[pi]))>1e-9,sourceSpikes=f.counts[bin][e.source],active=sourceSpikes>0,isRestored=restored.has(String(e.id));
  const d=`M${x},${y} Q${(x+tx)/2+12},${(y+ty)/2-8} ${tx},${ty}`;
  if(e===selected)add('path',{d,fill:'none',stroke:'#e8edf6',opacity:.8,'stroke-width':4.5,'data-selected-edge':e.id,'pointer-events':'none'});
  const path=add('path',{d,fill:'none',stroke:isRestored?'#c4a1ff':changed?'#ffbf69':active?'#67e8cf':'#35445c',opacity:e===selected?.95:active?.8:isRestored?.85:changed?.65:.2,'stroke-width':e===selected?2:active?1.7:isRestored?1.4:changed?1.1:.6,'stroke-dasharray':active?'4 2':'none','data-edge-id':e.id,'data-source-spikes':sourceSpikes});
  const title=document.createElementNS(ns,'title');title.textContent=`Edge ${e.id}: ${data.nodes[e.source].id} → ${data.nodes[e.target].id}; source ${sourceSpikes} spikes in this bin${changed?'; weight differs from selected baseline':''}${isRestored?'; restored before replay/probe':''}. Source activity does not establish target transmission.`;path.append(title);
 }
 data.nodes.forEach((n,i)=>{const[x,y]=positions[i],active=f.counts[bin][i]>0;const circle=add('circle',{cx:x,cy:y,r:active?4.5:2.5,fill:active?'#67e8cf':colors[n.group],opacity:active?1:.6});const title=document.createElementNS(ns,'title');title.textContent=`${n.type} ${n.id}: ${f.counts[bin][i]} spikes`;circle.append(title);if(i===node)add('circle',{cx:x,cy:y,r:7,fill:'none',stroke:'#fff','stroke-width':1.5});});
}
function spikeBin(direction){
 if(!data)return -1;
 const counts=data.frames[step].counts;
 for(let i=bin+direction;i>=0&&i<counts.length;i+=direction)if(counts[i][node]>0)return i;
 return -1;
}
function jumpSpike(direction){const target=spikeBin(direction);if(target<0)return;clearInterval(timer);timer=null;$('play').textContent='Play bins';bin=target;draw();}
function momentLink(){
 const link=$('moment-link');link.hidden=imported||externalNeuron||!runName;if(link.hidden)return;
 const url=new URL(location.href);url.search='';url.searchParams.set('run',runName);url.searchParams.set('step',step);url.searchParams.set('bin',bin);url.searchParams.set('neuron',data.nodes[node].id);
 const selectedEdge=data.edges.filter(e=>e.plastic_index!==null)[edge];if(selectedEdge)url.searchParams.set('edge',selectedEdge.id);
 if($('pristine').checked)url.searchParams.set('pristine','1');
 if($('compare').value)url.searchParams.set('compare',$('compare').value);link.href=url.href;
}
function hasMemory(f,pi){return ['plastic_u','plastic_w'].every(key=>Array.isArray(f[key])&&f[key].length===f.times_ms.length&&f[key].every(row=>Array.isArray(row)&&Number.isFinite(row[pi])));}
function edgeMemory(f,pi,e){
 const note=$('edge-memory-note');$('edge-memory-chart').replaceChildren();
 if(!hasMemory(f,pi)){note.textContent='Per-connection u/w are unavailable in this recording. They are not inferred from the weight or treated as zero.';return;}
 const u=f.plastic_u[bin][pi],w=f.plastic_w[bin][pi];
 note.textContent=`Edge ${e.id}: u ${num(u,6)} · w ${num(w,6)} · efficacy fraction (1 + w) ${num(1+w,6)} · u − w ${num(u-w,6)}. u is the stored learning state; w is its filtered efficacy deviation. The current synaptic weight already includes this efficacy.`;
 chart('edge-memory-chart',f.times_ms,[{name:'u: stored learning state (blue)',values:f.plastic_u.map(v=>v[pi]),color:'#8fbcff'},{name:'w: filtered efficacy deviation (pink)',values:f.plastic_w.map(v=>v[pi]),color:'#e6a8e8'}],'Memory deviation (model units)',bin);
}
function edgeCredit(f,pi,e){
 const c=f.plastic_credit,note=$('credit-note');$('credit-chart').replaceChildren();$('credit-interpretation').textContent='';
 const valid=data.credit_audit?.schema===1&&c?.schema===1&&c.units==='u per second'
  &&typeof c.learning==='boolean'&&c.learning===f.event.diagnostics.plasticity_enabled
  &&f.times_ms.length===50&&f.times_ms.every((t,i)=>t===f.event.brain_ms-500+(i+1)*10)
  &&JSON.stringify(c.plastic_selection)===JSON.stringify(data.plastic_selection)
  &&JSON.stringify(c.times_ms)===JSON.stringify(f.times_ms)
  &&['earlier','current','total','weight_step'].every(k=>Array.isArray(c[k])&&c[k].length===f.times_ms.length
   &&c[k].every(row=>Array.isArray(row)&&row.length===data.plastic_selection.length&&row.every(Number.isFinite)))
  &&c.total.every((row,i)=>row.every((v,j)=>Math.abs(v-c.earlier[i][j]-c.current[i][j])<=1e-10*(1+Math.abs(v))
   &&(c.learning||(v===0&&c.earlier[i][j]===0&&c.current[i][j]===0))
   &&Math.abs(c.weight_step[i][j]-(f.plastic_weights[i][j]-(i?f.plastic_weights[i-1][j]:f.plastic_initial[j])))<1e-10));
 if(!valid){note.textContent='Verified trace-origin decomposition is unavailable in this recording. No learning drive is inferred from weights or treated as zero.';return;}
 const precise=x=>x===0?'0':Math.abs(x)<.00001?x.toExponential(3):num(x,6);
 note.textContent=`Edge ${e.id} · ${num(f.times_ms[bin]-(f.event.brain_ms-500)-10)}–${num(f.times_ms[bin]-(f.event.brain_ms-500))} ms in image. Earlier-image traces: ${precise(c.earlier[bin][pi])}; current-image traces: ${precise(c.current[bin][pi])}; total drive: ${precise(c.total[bin][pi])} u/s. Recorded weight step: ${precise(c.weight_step[bin][pi])}. ${c.learning?'Updates enabled.':'Memory frozen: applied drive and weight steps are zero.'}`;
 chart('credit-chart',f.times_ms.map(t=>t-(f.event.brain_ms-500)),[
  {name:'Earlier-image traces (amber)',values:c.earlier.map(v=>v[pi]),color:'#ffbf69'},
  {name:'Current-image traces (teal)',values:c.current.map(v=>v[pi]),color:'#67e8cf'},
  {name:'Total drive (white)',values:c.total.map(v=>v[pi]),color:'#e8edf6'}],'Learning drive (u / s)',bin,'Time in observation (ms)');
 $('credit-interpretation').textContent='Current run only. Earlier-image traces are the rate history present at this image boundary; current-image traces accumulate afterward. They sum before memory filtering and clipping. This is an algebraic split of recorded activity, not a gradient or proof of which image or trade caused learning. Weight steps also depend on stored u/w.';
}
function jumpPairedBin(target){
 if(!paired||!Number.isInteger(target)||target<0||target>=data.frames[step].times_ms.length)return;
 clearInterval(timer);timer=null;$('play').textContent='Play bins';bin=target;draw();
}
function pairedDifferenceTable(xs,f,o,n,ni,e,pi,oi,sameEnds){
 const target=$('paired-state-differences');
 const column=(rows,index,width)=>index>=0&&Array.isArray(rows)&&rows.length===xs.length
  &&rows.every(row=>Array.isArray(row)&&row.length===width&&Number.isFinite(row[index]))?rows.map(row=>row[index]):null;
 const currentVoltage=column(f.voltage,node,data.nodes.length),otherVoltage=column(o.voltage,ni,paired.data.nodes.length);
 const memory=(frame,index,width,field)=>sameEnds?column(frame[field],index,width):null;
 const rows=[['voltage',`Body ${n.id}: voltage (mV)`,currentVoltage,otherVoltage],
  ...[['weight','plastic_weights'],['u','plastic_u'],['w','plastic_w']].map(([label,field])=>[
   label,`Edge ${e?.id??'unselected'}: ${label}`,
   memory(f,pi,data.plastic_selection.length,field),memory(o,oi,paired.data.plastic_selection.length,field)])];
 const body=document.createElement('tbody'),table=document.createElement('table'),head=document.createElement('thead'),tr=document.createElement('tr');
 for(const title of ['Recorded quantity','First different bin','Next after cursor','Current − comparison']){const th=document.createElement('th');th.scope='col';th.textContent=title;tr.append(th);}head.append(tr);table.append(head,body);
 const precise=x=>x===0?'0':Math.abs(x)<.00001?x.toExponential(3):num(x,6);
 for(const [key,label,left,right] of rows){
  const row=document.createElement('tr');row.dataset.field=key;
  const name=document.createElement('th');name.scope='row';name.textContent=label;row.append(name);
  const differences=left&&right?left.flatMap((v,i)=>v!==right[i]?[i]:[]):null;
  for(const [kind,index] of [['first',differences?.[0]],['next',differences?.find(i=>i>bin)]]){
   const cell=document.createElement('td'),button=document.createElement('button');button.type='button';button.dataset.jump=kind;
   button.textContent=differences===null?'Not recorded':index===undefined?'None':`${num(xs[index])} ms`;
   button.disabled=index===undefined||index===bin;button.setAttribute('aria-label',`${kind==='first'?'First difference':'Next difference'} for ${label}: ${button.textContent}`);
   if(index!==undefined){button.dataset.bin=index;button.onclick=()=>jumpPairedBin(index);}cell.append(button);row.append(cell);
  }
  const delta=document.createElement('td');delta.dataset.delta=key;delta.textContent=left&&right?precise(left[bin]-right[bin]):'Not recorded';row.append(delta);body.append(row);
 }
 target.append(table);
}
function pairedTraces(){
 const panel=$('paired-traces');panel.hidden=!paired;
 for(const id of ['first-difference','next-difference']){$(id).disabled=true;delete $(id).dataset.bin;}
 $('paired-difference-note').textContent='';
 $('paired-state-differences').replaceChildren();
 for(const id of ['paired-voltage','paired-spikes','paired-weight','paired-u','paired-w'])$(id).replaceChildren();
 $('paired-memory-note').textContent='';if(!paired)return;
 const f=data.frames[step],o=paired.frames.get(step),note=$('paired-note');
 if(!o){note.textContent='No shared observation for the current decision timestamp or phase. No traces are aligned.';return;}
 const elapsed=frame=>frame.times_ms.map(t=>t-(frame.event.brain_ms-500));
 const xs=elapsed(f),otherXs=elapsed(o);
 if(xs.length!==otherXs.length||xs.some((t,i)=>Math.abs(t-otherXs[i])>1e-6)){
  note.textContent='The recordings have different time bins. No traces are interpolated or aligned.';return;
 }
 const identical=!!inputHash(f)&&inputHash(f)===inputHash(o);
 note.textContent=`Displayed selections at ${num(xs[bin])} ms into this observation. Blue: current; pink: comparison. Input identical: ${identical?'yes':'no'}. Bins are 10 ms, not exact spike times.`;
 if(f.event.stimulation||o.event.stimulation){
  const applied=e=>e.stimulation?`${e.stimulation.current} to cells ${e.stimulation.target_ids.join(', ')} for ${e.stimulation.duration_ms} ms`:'unrecorded';
  note.textContent+=` Applied diagnostic current: blue ${applied(f.event)}; pink ${applied(o.event)}.`;
  const memory=report=>report.memory_origin??report.config.memory??'unrecorded';
  note.textContent+=` Stored memory: blue ${memory(data.report)}; pink ${memory(paired.data.report)}.`;
 }
 const history=paired.histories.get(step)||'unverified';
 note.textContent+=` Recorded input prefix: ${history}. This covers the images saved here through this observation, not earlier training or activity resets.`;
 const build=data.report.native_build?.binary_sha256,otherBuild=paired.data.report.native_build?.binary_sha256;
 if(!build||!otherBuild||build!==otherBuild)note.textContent+=' Matching native build is unverified; differences may include build effects.';
 const n=data.nodes[node],ni=paired.data.nodes.findIndex(v=>String(v.id)===String(n.id));
 const series=(a,b)=>[{name:'Current (blue)',values:a,color:'#8fbcff'},{name:'Comparison (pink)',values:b,color:'#e6a8e8'}];
 if(ni>=0){
  chart('paired-voltage',xs,series(f.voltage.map(v=>v[node]),o.voltage.map(v=>v[ni])),`Body ${n.id}: voltage (mV)`,bin,'Time in observation (ms)');
  chart('paired-spikes',xs,series(f.counts.map(v=>v[node]),o.counts.map(v=>v[ni])),`Body ${n.id}: spikes / 10 ms`,bin,'Time in observation (ms)');
  note.textContent+=` Body ${n.id}: ${f.counts[bin][node]} vs ${o.counts[bin][ni]} spikes in this bin.`;
  const differences=f.counts.flatMap((r,i)=>r[node]!==o.counts[i][ni]?[i]:[]);
  $('paired-difference-note').textContent=`Selected body ${n.id}: ${differences.length} of ${xs.length} bins have different spike counts. This compares this neuron only; it does not locate the first difference in the full brain or establish a causal path.`;
  for(const [id,target] of [['first-difference',differences[0]],['next-difference',differences.find(i=>i>bin)]])if(target!==undefined){$(id).dataset.bin=target;$(id).disabled=target===bin;}
 }else $('paired-voltage').textContent=`Body ${n.id} is absent from the comparison's displayed subset.`;
 const e=data.edges.filter(v=>v.plastic_index!==null)[edge];
 const oe=e&&paired.data.edges.find(v=>String(v.id)===String(e.id)&&v.plastic_index!==null);
 const pi=e?data.plastic_selection.indexOf(e.plastic_index):-1,oi=oe?paired.data.plastic_selection.indexOf(oe.plastic_index):-1;
 const sameEnds=!!oe&&String(data.nodes[e.source].id)===String(paired.data.nodes[oe.source].id)&&String(data.nodes[e.target].id)===String(paired.data.nodes[oe.target].id);
 pairedDifferenceTable(xs,f,o,n,ni,e,pi,oi,sameEnds);
 if(!oe){$('paired-weight').textContent='Selected connection is absent from the comparison subset.';return;}
 if(!sameEnds||pi<0||oi<0){$('paired-weight').textContent='Connection identity does not reconcile between recordings.';return;}
 chart('paired-weight',xs,series(f.plastic_weights.map(v=>v[pi]),o.plastic_weights.map(v=>v[oi])),`Edge ${e.id}: weight`,bin,'Time in observation (ms)');
 if(hasMemory(f,pi)&&hasMemory(o,oi)){
  chart('paired-u',xs,series(f.plastic_u.map(v=>v[pi]),o.plastic_u.map(v=>v[oi])),`Edge ${e.id}: u`,bin,'Time in observation (ms)');
  chart('paired-w',xs,series(f.plastic_w.map(v=>v[pi]),o.plastic_w.map(v=>v[oi])),`Edge ${e.id}: w`,bin,'Time in observation (ms)');
  $('paired-memory-note').textContent=`u: ${num(f.plastic_u[bin][pi],6)} vs ${num(o.plastic_u[bin][oi],6)} · w: ${num(f.plastic_w[bin][pi],6)} vs ${num(o.plastic_w[bin][oi],6)}. Blue is current; pink is comparison. These are recorded memory states, not gradients.`;
 }else $('paired-memory-note').textContent='Per-connection memory comparison is unavailable for this observation in one or both recordings.';
 note.textContent+=` Edge ${e.id} (${data.nodes[e.source].id} → ${data.nodes[e.target].id}): ${num(f.plastic_weights[bin][pi],6)} vs ${num(o.plastic_weights[bin][oi],6)}; current minus comparison ${num(f.plastic_weights[bin][pi]-o.plastic_weights[bin][oi],6)}.`;
}
function decoderPath(f){
 const note=$('decoder-path-note');for(const id of ['decoder-direction-chart','decoder-gate-chart'])$(id).replaceChildren();
 const groups=f.event.cell_ids,ids=data.nodes.map(n=>String(n.id));
 const unavailable=reason=>{note.textContent='Decoder count trace unavailable: '+reason;};
 if(!groups||!['left','right','gate'].every(k=>Array.isArray(groups[k])&&groups[k].length))return unavailable('missing output identities.');
 const wanted=['left','right','gate'].flatMap(k=>groups[k].map(String));
 if(new Set(ids).size!==ids.length||new Set(wanted).size!==wanted.length||wanted.some(id=>!ids.includes(id)))return unavailable('missing or ambiguous output neurons.');
 // Captures use 50 bins of 10 ms. Do not infer a 490 ms observation from last minus first.
 const start=f.event.brain_ms-500,xs=f.times_ms.map(t=>t-start);
 if(xs.length!==50||xs.some((x,i)=>Math.abs(x-(i+1)*10)>1e-6))return unavailable('unsupported observation time grid.');
 if(!Array.isArray(f.counts)||f.counts.length!==xs.length||f.counts.some(row=>!Array.isArray(row)||row.length!==ids.length))return unavailable('invalid count dimensions.');
 const series={};
 for(const key of ['left','right','gate']){
  const columns=groups[key].map(id=>ids.indexOf(String(id)));let total=0;series[key]=[];
  for(const row of f.counts){const values=columns.map(i=>row[i]);if(values.some(v=>!Number.isInteger(v)||v<0))return unavailable('invalid recorded counts.');total+=values.reduce((a,b)=>a+b,0);series[key].push(key==='gate'?total:total/columns.length/.5);}
 }
 if(series.left.length!==xs.length||Math.abs(series.left.at(-1)-f.event.left_hz)>1e-6||Math.abs(series.right.at(-1)-f.event.right_hz)>1e-6||series.gate.at(-1)!==f.event.gate_spikes)return unavailable('counts do not reconcile to the recorded output.');
 const difference=series.right.map((r,i)=>r-series.left[i]);
 if(Math.abs(difference.at(-1)-f.event.difference_hz)>1e-6)return unavailable('direction does not reconcile.');
 note.textContent=`At ${num(xs[bin])} ms: left contribution ${num(series.left[bin])} Hz, right ${num(series.right[bin])} Hz, right − left ${num(difference[bin])} Hz; ${num(series.gate[bin])} gate spikes so far. Final direction: ${num(f.event.difference_hz)} Hz. One spike contributes ${num(2/groups.left.length)} Hz on the left or ${num(2/groups.right.length)} Hz on the right. Left: ${groups.left.join(', ')}; right: ${groups.right.join(', ')}; gate: ${groups.gate.join(', ')}.`;
 chart('decoder-direction-chart',xs,[{name:'Left contribution (blue)',values:series.left,color:'#8fbcff'},{name:'Right contribution (pink)',values:series.right,color:'#e6a8e8'},{name:'Right − left (green)',values:difference,color:'#67e8cf'}],'Accumulated contribution (Hz)',bin,'Time in observation (ms)');
 chart('decoder-gate-chart',xs,[{name:'Accumulated gate spikes',values:series.gate,color:'#ffbf69'}],'Gate spikes',bin,'Time in observation (ms)');
}
function inputTiming(f){
 const e=f.event,elapsed=f.times_ms.map(t=>t-(e.brain_ms-500));
 const validGrid=Number.isFinite(e.brain_ms)&&elapsed.length===50&&elapsed.every((t,i)=>Math.abs(t-(i+1)*10)<1e-6);
 const start=bin*10,end=start+10;
 for(const kind of ['pulse','current']){$(kind+'-timing-track').replaceChildren();$(kind+'-timing-note').textContent=`${kind==='pulse'?'Reinforcement':'Diagnostic current'} timing unavailable${validGrid?' in this recording.':': unsupported observation time grid.'}`;}
 if(!validGrid)return;
 const drawTrack=(kind,duration,enabled,label,color)=>{
  const active=enabled&&start<duration;
  $(kind+'-timing-note').textContent=`${label}. Selected bin ${start}–${end} ms: ${active?'input applied':'input off'}.`;
  const el=$(kind+'-timing-track'),w=Math.max(250,el.clientWidth),left=12,right=w-12,X=t=>left+t/500*(right-left);
  const ns='http://www.w3.org/2000/svg',svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${w} 52`);svg.setAttribute('role','img');svg.setAttribute('aria-label',$(kind+'-timing-note').textContent);
  const add=(tag,attrs,text)=>{const n=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;svg.append(n);};
  add('rect',{x:left,y:5,width:right-left,height:20,fill:'#202e44'});
  if(enabled&&duration>0)add('rect',{x:left,y:5,width:X(duration)-left,height:20,fill:color,'data-input-duration':duration});
  add('rect',{x:X(start),y:4,width:X(end)-X(start),height:22,fill:'none',stroke:'#e8edf6','stroke-width':2,'data-input-active':String(active),'data-bin-start':start});
  for(const t of [0,250,500])add('text',{x:X(t),y:44,fill:'#b2bfd1','font-size':11,'text-anchor':t===0?'start':t===500?'end':'middle'},`${t} ms`);
  el.append(svg);
 };
 if(['none','reward','aversive'].includes(e.stimulus)&&Number.isFinite(e.stimulus_ms)&&e.stimulus_ms>=0&&e.stimulus_ms<=500&&e.stimulus_ms%10===0&&((e.stimulus==='none')===(e.stimulus_ms===0))){
  drawTrack('pulse',e.stimulus_ms,e.stimulus!=='none',e.stimulus==='none'?'No reinforcement pulse':`${e.stimulus==='reward'?'Reward':'Aversive'} pulse · 0–${e.stimulus_ms} ms`,'#e6a8e8');
 }
 const s=e.stimulation;
 if(s&&Number.isFinite(s.current)&&s.duration_ms===500&&Array.isArray(s.target_ids)&&s.target_ids.length&&s.target_ids.every(id=>typeof id==='string'&&/^\d+$/.test(id))){
  drawTrack('current',s.duration_ms,s.current!==0,`Diagnostic current ${num(s.current)} · cells ${s.target_ids.join(', ')} · 0–500 ms`,'#ffbf69');
 }
}
function activityBoundary(f, selectedIndex=node){
 const boundary=f.event.activity_boundary,panel=$('activity-boundary');panel.hidden=!boundary;$('activity-boundary-note').textContent='';$('activity-neuron-state').textContent='';if(!boundary)return;
 const label={initial:'Fresh dynamics; no between-observation intervention yet',carry:'Carry all activity forward',reset_rates:'Reset KC and DAN learning traces only',reset_kc:'Reset KC learning traces only',reset_dan:'Reset DAN learning traces only',full:'Reset every recorded dynamic field',visual_filters:'Reset visual filters only',adaptation:'Reset intrinsic adaptation only',voltage:'Reset membrane voltage',conductance:'Reset synaptic input state',voltage_conductance:'Reset voltage and synaptic input',gate_voltage_conductance:'Reset voltage and synaptic input only at gate neurons'};
 $('activity-boundary-note').textContent=`${label[boundary.mode]||boundary.mode}. Neural clock: ${num(boundary.before_clock.sim_ms)} → ${num(boundary.after_clock.sim_ms)} ms before this image. Target fields: ${boundary.target_fields.join(', ')||'none'}. Scope: ${boundary.target_ids?.length?'neurons '+boundary.target_ids.join(', '):'full target arrays'}. Fields whose values changed: ${boundary.changed_fields.join(', ')||'none'}. Synaptic memory fingerprint: ${boundary.memory_sha256.slice(0,16)}.`;
 const state=f.activity_state;
 if(state&&['before_v','after_v','before_g','after_g'].every(k=>Array.isArray(state[k])&&state[k].length===data.nodes.length&&Number.isFinite(state[k][selectedIndex]))){
  $('activity-neuron-state').textContent=`Body ${data.nodes[selectedIndex].id} at this boundary, before the image: voltage ${num(state.before_v[selectedIndex],5)} → ${num(state.after_v[selectedIndex],5)} mV; synaptic input state g ${num(state.before_g[selectedIndex],5)} → ${num(state.after_g[selectedIndex],5)} (model units). These boundary values do not move with the within-image time cursor.`;
 }else $('activity-neuron-state').textContent='Per-neuron boundary values are unavailable in this recording; no values are inferred.';
}
function draw(){if(!data)return;externalNeuron=false;document.querySelector('.legend .changed').parentElement.lastChild.textContent=$('pristine').checked?'Plastic edge differs from pristine state':'Plastic edge changed since observation start';const f=data.frames[step],n=data.nodes[node];bin=Math.min(bin,f.times_ms.length-1);$('bin').max=f.times_ms.length-1;$('bin').value=bin;$('time').textContent=num(f.times_ms[bin])+' ms';network(f);decoderPath(f);activityBoundary(f);inputTiming(f);
 document.querySelector('.legend .restored').parentElement.lastChild.textContent=data.report.restoration?'Restored before replay':'Restored before probe';
 $('previous-spike').disabled=spikeBin(-1)<0;$('next-spike').disabled=spikeBin(1)<0;momentLink();
 $('neuron-detail').textContent=`${n.type||'Unannotated'} · ${n.id} · ${f.counts[bin][node]} spikes in this bin · ${num(f.voltage[bin][node])} mV`;
 chart('neuron-chart',f.times_ms,[{name:'Selected neuron voltage',values:f.voltage.map(v=>v[node]),color:'#8fbcff'}],'Voltage (mV)',bin);
 $('input').src='data:image/png;base64,'+f.input_png;$('input-info').textContent=`${f.event.phase||'assay'} · ${f.event.input_preset||data.report.config.preset} · ${data.report.config.view} ${f.event.input_amplitude===undefined?"":"· movement ×"+num(f.event.input_amplitude)} · ${(f.event.input_news||data.report.config.news)==='sealed_timestamped'?'news available at quote time':(f.event.input_news||data.report.config.news)+' news'}`;$('encoding-note').textContent=data.report.config.view==='fixed_returns'?'Fixed return scale, referenced to the first price in the window. Bounds: 0.01×–100× that price; amber points mark clipping. See provenance for the exact transform.':'';
 $('phase-note').textContent=f.event.phase==='probe'?`Frozen probe: neural activity was reset; ${data.report.probe_boundary.restoration?.mode&&data.report.probe_boundary.restoration.mode!=='none'?`${num(data.report.probe_boundary.restoration.edge_count)} connections restored to pristine memory (${data.report.probe_boundary.restoration.mode})`:'trained weights were retained'}. Weight L2 from pristine: ${num(data.report.probe_boundary.weight_delta_from_pristine_l2,6)}. No reinforcement or extra current.`:data.report.source==='matched_market_stimulus_assay'?`Historical replay: ${data.report.config.memory} memory; ${data.report.config.learning?'updates enabled':'weights and memory frozen'}; pulses ${data.report.config.pulses}.${f.event.stimulation?` Diagnostic current ${f.event.stimulation.current} to cells ${f.event.stimulation.target_ids.join(', ')} for ${f.event.stimulation.duration_ms} ms.`:''}${data.report.restoration?.post_ids?.length?` Restored ${num(data.report.restoration.edge_count)} incoming connections to body IDs ${data.report.restoration.post_ids.join(', ')} before replay.`:''} No fills or returns recomputed.`:data.report.restoration?.post_ids?.length?`Market replay: ${num(data.report.restoration.edge_count)} incoming connections to body IDs ${data.report.restoration.post_ids.join(', ')} restored to pristine before inference. Training memory is retained elsewhere; inference is frozen.`:'';$('decision').textContent=f.event.side;dl($('decoder'),[['Left DNp20 (Hz)',f.event.left_hz],['Right DNp20 (Hz)',f.event.right_hz],['Right − left (Hz)',f.event.difference_hz],['Gate spikes',f.event.gate_spikes],['Reinforcement',f.event.stimulus]]);
 if(data.report.memory_origin){
  const exposure=data.report.training_exposure;
  const learning=data.report.config.learning===true?'updated online using this phase’s paper equity feedback':'frozen throughout';
  const boundary=data.report.config.activity_reset==='conductance'?'synaptic input state g reset before each later image':data.report.config.activity_reset==='reset_rates'?'only KC/DAN rate traces cleared before each later image; other activity and memory carried forward':'activity carried between images';
  $('phase-note').textContent=`${data.report.evaluation_phase==='development'?'Development':'Test'} evaluation: ${data.report.memory_origin==='paper_trained'?'paper-trained':'pristine'} connection memory, ${learning}. Fresh neural activity at phase start; ${boundary}.${f.event.stimulation?` Applied diagnostic current ${f.event.stimulation.current} to cells ${f.event.stimulation.target_ids.join(', ')} throughout each 500 ms image.`:''}${exposure?` Training exposure: ${num(exposure.observations)} observations; ${num(exposure.positive_rewards)} positive and ${num(exposure.negative_rewards)} negative paper rewards.`:''} Simulated fills; this does not establish profitable learning.`;
 }
 dl($('metrics'),[['Plasticity enabled',f.event.diagnostics.plasticity_enabled??data.report.config.learning],[$('pristine').checked?'Changed edges vs pristine':'Changed edges at cursor',($('pristine').checked?f.changed_from_pristine:f.changed_edges)[bin]],[$('pristine').checked?'Weight Δ L2 vs pristine':'Weight Δ L2 at cursor',($('pristine').checked?f.weight_from_pristine_l2:f.weight_delta_l2)[bin]],['Memory u L2',f.memory_u_l2[bin]],['Memory w L2',f.memory_w_l2[bin]],['Loss',null],['Backprop gradient',null],['Clipped edges (end)',f.event.diagnostics.clipped_edges]]);
 chart('spike-chart',f.times_ms,[{name:'All retained neurons',values:f.total_spikes,color:'#67e8cf'}],'Spikes / 10 ms',bin);
 chart('weight-chart',f.times_ms,[{name:$('pristine').checked?'All plastic edges: L2 from pristine':'All plastic edges: L2 from observation start',values:$('pristine').checked?f.weight_from_pristine_l2:f.weight_delta_l2,color:'#ffbf69'}],'Weight Δ L2',bin);
 chart('trace-chart',f.times_ms,[{name:'KC trace',values:f.kc_mean_hz,color:'#8fbcff'},{name:'DAN trace',values:f.dan_mean_hz,color:'#e6a8e8'}],'Mean trace (Hz)',bin);
 const plastic=data.edges.filter(e=>e.plastic_index!==null),e=plastic[edge];if(e){const pi=data.plastic_selection.indexOf(e.plastic_index);$('edge-detail').textContent=`Edge ${e.id} · ${data.nodes[e.source].id} → ${data.nodes[e.target].id} · weight ${num(f.plastic_weights[bin][pi],6)} · Δ ${num(f.plastic_weights[bin][pi]-($('pristine').checked?e.weight:f.plastic_initial[pi]),6)} · source spikes ${f.counts[bin][e.source]}${restoredEdges(f).includes(String(e.id))?` · restored to pristine memory before ${f.event.phase==='probe'?'probe':'replay'}`:''}`;edgeMemory(f,pi,e);edgeCredit(f,pi,e);chart('edge-chart',f.times_ms,[{name:'Synaptic weight',values:f.plastic_weights.map(v=>v[pi]),color:'#ffbf69'}],'Weight (model units)',bin);}
 pairedTraces();
}
function table(headers, rows){const t=document.createElement('table'),head=document.createElement('tr');for(const title of headers){const th=document.createElement('th');th.textContent=title;head.append(th);}t.append(head);for(const values of rows){const row=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;row.append(td);}t.append(row);}return t;}
function utc(ts){return new Date(ts*1000).toISOString().slice(11,19);}
function marketTimeline(report){
 const rows=report.market_timeline;$('market-panel').hidden=report.source!=='sealed_retrospective_market_replay';$('market-timeline').replaceChildren();
 if(!rows){$('market-coverage').textContent='This older recording has no decision timeline. Consult its study ledger for skipped observations and fills.';return;}
 const decisions=rows.filter(r=>!r.terminal),observed=decisions.filter(r=>r.observation==='observed').length,fills=rows.filter(r=>r.fill_status==='filled').length;
 $('market-coverage').textContent=`${observed} / ${decisions.length} decision slots observed · ${fills} simulated fills · ${decisions.length-observed} skipped. Times are UTC.`;
 $('market-timeline').append(table(['Decision UTC','Quote UTC','Observation','Output','Fill from prior decision','Sleeve equity'],rows.map(r=>[utc(r.decision_ts),utc(r.quote_ts),r.observation.replaceAll('_',' '),r.action??'—',r.fill_reason?`${r.fill_status}: ${r.fill_reason}`:r.fill_status,'$'+Number(r.equity).toFixed(2)])));
}
function load(d){if(!d.report||!d.nodes?.length||!d.frames?.length)throw new Error('Not a fly view.json recording');data=d;paired=null;$('pristine').checked=false;$('pristine').disabled=!d.frames.every(f=>f.weight_from_pristine_l2&&f.changed_from_pristine);marketTimeline(d.report);const market=d.report.source==='sealed_retrospective_market_replay',mechanics=d.report.source==='matched_market_stimulus_assay';$('source-badge').textContent=d.validation_only===true?'Synthetic prices · implementation check':mechanics?'Historical inputs · neural diagnostic':market?'Market replay · simulated fills':'Synthetic assay · paper only';$('decoder-note').textContent=mechanics?'Recorded market images with controlled memory, updates, and stimulation. This is a neural diagnostic; no account, fills, or returns are recomputed.':market?'Whole-observation result (500 ms). Simulated fills are recorded in the study ledger. This viewer sends no orders.':'Whole-observation result (500 ms). BUY/SELL are engineered mappings of downstream firing rates; this assay executes no trades.';step=bin=node=edge=0;clearInterval(timer);timer=null;$('play').textContent='Play bins';options($('step'),d.frames.map((f,i)=>[i,`${f.event.phase?f.event.phase+' '+f.event.phase_step:i+1} · ${f.event.side}`]));options($('node'),d.nodes.map((n,i)=>[i,`${n.group}: ${n.type||n.id} · ${n.id}`]));options($('edge'),d.edges.filter(e=>e.plastic_index!==null).map((e,i)=>[i,`${e.id} · ${d.nodes[e.source].type} → ${d.nodes[e.target].type}`]));$('graph-size').textContent=`${num(d.report.graph.neurons)} neurons / ${num(d.report.graph.edges)} connections simulated`;$('selection').textContent=`Showing ${d.nodes.length} neurons and ${d.edges.length} connections. ${d.selection}`;$('provenance').textContent=JSON.stringify({...d.report,...(d.memory_enrichment?{memory_enrichment:d.memory_enrichment}:{}),...(d.selection_extension?{selection_extension:d.selection_extension}:{}),...(d.credit_audit?{credit_audit:d.credit_audit}:{})},null,2);$('status').textContent=`Loaded ${runName||'imported trace'} · ${d.frames.length} observations · ${num(d.report.seconds,1)} s capture time`;$('comparison').replaceChildren();$('compare').value='';draw();}
async function loadRun(name){const ticket=++viewRequest;$('error').textContent='';const recording=await api('/api/view?run='+encodeURIComponent(name));if(ticket!==viewRequest)return;runName=name;$('runs').value=name;imported=false;load(recording);}
async function refresh(){const r=await api('/api/runs');const prior=$('runs').value,priorCompare=$('compare').value;options($('runs'),r.runs.map(n=>[n,n]));options($('compare'),[['','Choose a run'],...r.runs.map(n=>[n,n])]);if(r.runs.includes(priorCompare))$('compare').value=priorCompare;else{paired=null;$('comparison').replaceChildren();pairedTraces();} $('run').disabled=!r.can_run;$('run').textContent=!r.can_run?'Read-only recordings':r.backend==='modal'?'Run on Modal (cloud)':'Run full-network assay';if(r.runs.includes(prior))$('runs').value=prior;if(!data&&r.runs.length){const requested=new URLSearchParams(location.search).get('run');await loadRun(r.runs.includes(requested)?requested:r.runs.at(-1));}if(!r.runs.length&&!data)$('status').textContent='No recordings yet. Run an assay or import a view.json.';}
async function status(initial=false){const s=await api('/api/status');if(!s.run||(initial&&['succeeded','not_started'].includes(s.status)))return;const log=await api('/api/log?run='+encodeURIComponent(s.run));$('log').textContent=log.log;$('status').textContent=`${s.run}: ${s.status}`;if(s.status==='running'){setTimeout(()=>status().catch(error),2000);}else{$('run').disabled=false;if(s.status==='succeeded'){await refresh();$('runs').value=s.run;await loadRun(s.run);}else error(s.error||('Assay '+s.status+'. See worker log or Modal call.'));}}
$('assay').onsubmit=async e=>{e.preventDefault();$('error').textContent='';const f=new FormData(e.target);const config=Object.fromEntries(f);config.steps=Number(f.get('steps'));config.probe_steps=Number(f.get('probe_steps'));config.amplitude=Number(f.get('amplitude'));config.probe_amplitude=Number(f.get('probe_amplitude'));config.eta=Number(f.get('eta'));config.current=Number(f.get('current'));config.learning=f.has('learning');config.reinforcement_only=f.has('reinforcement_only');config.probe_edges=String(f.get('probe_edges')).split(',').map(x=>x.trim()).filter(Boolean);config.neurons=String(f.get('neurons')).split(',').map(x=>x.trim()).filter(Boolean);try{$('run').disabled=true;await api('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});await status();}catch(e){$('run').disabled=false;error(e);}};
$('runs').onchange=e=>loadRun(e.target.value).catch(error);$('refresh').onclick=()=>refresh().catch(error);$('step').onchange=e=>{step=Number(e.target.value);bin=0;draw();};$('bin').oninput=e=>{bin=Number(e.target.value);draw();};$('node').onchange=e=>{node=Number(e.target.value);draw();};$('edge').onchange=e=>{edge=Number(e.target.value);draw();};
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='Play bins';return;}if(!data)return;$('play').textContent='Pause';timer=setInterval(()=>{bin++;if(bin>=data.frames[step].times_ms.length){bin=0;step=(step+1)%data.frames.length;$('step').value=step;}draw();},150);};
$('pristine').onchange=()=>draw();
$('previous-spike').onclick=()=>jumpSpike(-1);$('next-spike').onclick=()=>jumpSpike(1);
for(const id of ['first-difference','next-difference'])$(id).onclick=()=>jumpPairedBin(Number($(id).dataset.bin));
$('network').onclick=e=>{const r=$('network').getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let best=22;positions.forEach(([nx,ny],i)=>{const d=Math.hypot(x-nx,y-ny);if(d<best){best=d;node=i;}});$('node').value=node;draw();};
$('lookup-button').onclick=async()=>{try{if(imported)throw new Error('Full-neuron lookup needs the recording server and its NPZ files.');const id=$('lookup').value.trim(),d=await api('/api/neuron?run='+encodeURIComponent(runName)+'&id='+encodeURIComponent(id)),f=d.frames[step];$('neuron-detail').textContent=`Full recording: ${d.id} · ${num(f.counts.reduce((a,b)=>a+b,0))} spikes in observation · ${f.plastic_edges.length} adjacent plastic edges`;chart('neuron-chart',f.times_ms,[{name:'Queried neuron voltage',values:f.voltage,color:'#8fbcff'}],'Voltage (mV)',bin);$('lookup-button').title=JSON.stringify(f.plastic_edges);externalNeuron=true;activityBoundary(data.frames[step],data.nodes.findIndex(n=>String(n.id)===String(d.id)));$('previous-spike').disabled=$('next-spike').disabled=true;momentLink();}catch(e){error(e);}};
$('import').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>64*1024*1024)throw new Error('Trace JSON must be under 64 MiB');runName=file.name;imported=true;load(JSON.parse(await file.text()));}catch(e){error(e);}};
$('compare').onchange=async e=>{try{
 if(!data)return;paired=null;pairedTraces();$('comparison').replaceChildren();momentLink();if(!e.target.value){++viewRequest;return;}
 const current=data,currentName=runName,otherName=e.target.value,request=++viewRequest,other=await api('/api/view?run='+encodeURIComponent(otherName));
 if(request!==viewRequest||current!==data)return;
 const market=historical(current.report),otherMarket=historical(other.report);
 if(market!==otherMarket){$('comparison').textContent='A synthetic assay and a market replay cannot be paired by observation index.';return;}
 const {matched,ambiguous}=observationPairs(current,other,market);
 const wrap=document.createElement('div');wrap.className='table-wrap';
 wrap.append(table([market?'Decision UTC':'Phase / observation','Current action','Other action','Current R−L Hz','Other R−L Hz','Current gate spikes','Other gate spikes','Input identical','Recorded input prefix'],matched.map(([f,o,i,history])=>[market?utc(f.event.market_decision_ts):phaseKey(f,i),f.event.side,o.event.side,num(f.event.difference_hz),num(o.event.difference_hz),f.event.gate_spikes,o.event.gate_spikes,inputHash(f)&&inputHash(f)===inputHash(o)?'Yes':'No / unverified',history])));
 const names=document.createElement('p');names.textContent=`Current: ${currentName} · Other: ${otherName}`;const note=document.createElement('p');note.textContent=market?`${matched.length} shared decision timestamps. Unmatched slots are excluded; inspect each timeline for gaps.`:`${matched.length} observations paired by phase and within-phase index.`;
 if(ambiguous)note.textContent+=` ${ambiguous} ambiguous observation keys excluded; duplicated timestamps or phase indices cannot be aligned.`;
 note.textContent+=' The input prefix compares only saved image hashes and observation keys. It does not establish matching earlier training or dynamic state.';
 if([current,other].some(d=>d.report.source==='matched_market_stimulus_assay'))note.textContent+=' Neural responses only: the diagnostic does not recompute fills or returns.';
 const currentBuild=current.report.native_build?.binary_sha256,otherBuild=other.report.native_build?.binary_sha256;if(!currentBuild||!otherBuild)note.textContent+=' Native build provenance is missing; controlled pairing is unverified.';else if(currentBuild!==otherBuild)note.textContent+=' Native binaries differ; this comparison includes build/platform effects.';const config=document.createElement('pre');config.textContent=JSON.stringify(Object.fromEntries([['current',current],['comparison',other]].map(([name,view])=>[name,{configuration:view.report.config,evaluation_phase:view.report.evaluation_phase,memory_origin:view.report.memory_origin,training_exposure:view.report.training_exposure}])),null,2);const details=document.createElement('details'),heading=document.createElement('summary');heading.textContent='Compared configurations';details.append(heading,config);$('comparison').replaceChildren(names,note,wrap,details);paired={data:other,frames:new Map(matched.map(([f,o,i])=>[i,o])),histories:new Map(matched.map(([f,o,i,history])=>[i,history]))};pairedTraces();
 }catch(e){error(e);}};
window.addEventListener('resize',()=>draw());refresh().then(async()=>{
 const q=new URLSearchParams(location.search);
 const index=(name,length)=>{const value=q.get(name);if(!value||!/^\d+$/.test(value))return -1;const n=Number(value);return Number.isSafeInteger(n)&&n<length?n:-1;};
 if(data){
  const selected=index('step',data.frames.length);if(selected>=0){step=selected;$('step').value=step;}
  const selectedBin=index('bin',data.frames[step].times_ms.length);if(selectedBin>=0)bin=selectedBin;
  const selectedNode=data.nodes.findIndex(n=>String(n.id)===q.get('neuron'));if(selectedNode>=0){node=selectedNode;$('node').value=node;}
  const selectedEdge=data.edges.filter(e=>e.plastic_index!==null).findIndex(e=>String(e.id)===q.get('edge'));if(selectedEdge>=0){edge=selectedEdge;$('edge').value=edge;}
  if(q.get('pristine')==='1'&&!$('pristine').disabled)$('pristine').checked=true;
  draw();
 }
 const comparison=q.get('compare');if(comparison&&[...$('compare').options].some(o=>o.value===comparison)){$('compare').value=comparison;await $('compare').onchange({target:$('compare')});}await status(true);}).catch(error);

$('restore-edge').onclick=()=>{if(!data)return;const e=data.edges.filter(e=>e.plastic_index!==null)[edge];if(!e)return;const form=$('assay').elements,ids=form.probe_edges.value.split(',').map(s=>s.trim()).filter(Boolean);if(!ids.includes(String(e.id)))ids.push(String(e.id));if(ids.length>64){error('At most 64 restoration edges');return;}form.probe_edges.value=ids.join(', ');form.probe_restore.value='selected';if(Number(form.probe_steps.value)===0)form.probe_steps.value='1';form.probe_restore.closest('details').open=true;form.probe_restore.focus();};
