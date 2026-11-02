'use strict';
let token = '', pickerTarget = '', browseDirectory = '', busy = false, ordinaryDefaults = null, exampleInputs = null;
function byId(id) { return document.getElementById(id); }
const numeric = ['duration_s', 'manning_n', 'infiltration_mm_h', 'interval_s', 'checkpoint_interval_s'];
function config() {
  const data = Object.fromEntries(new FormData(byId('run-form')));
  for (const name of numeric) data[name] = Number(data[name]);
  return data;
}
async function api(path, data) {
  const options = data === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json','X-FloodWarp-Token':token},body:JSON.stringify(data)};
  const response = await fetch(path, options), result = await response.json();
  if (!response.ok) throw new Error(result.error || response.statusText);
  return result;
}
function visibility() {
  byId('domain-field').hidden = byId('domain_mode').value === 'all';
  byId('multiplier-field').hidden = byId('distribution').value !== 'map';
  const roughness=byId('roughness_mode').value, infiltration=byId('infiltration_mode').value;
  byId('roughness-uniform').hidden=roughness!=='uniform'; byId('manning_n').required=roughness==='uniform';
  byId('roughness-map-field').hidden=roughness!=='map';
  byId('infiltration-uniform').hidden=infiltration!=='uniform'; byId('infiltration_mm_h').required=infiltration==='uniform';
  byId('infiltration-map-field').hidden=infiltration!=='map';
  byId('exponential-fields').hidden=infiltration!=='exponential';
  byId('infiltration-units-field').hidden=infiltration==='uniform';
  byId('boundary-file-field').hidden=byId('boundary').value!=='custom';
}
function textNode(tag, text) { const node = document.createElement(tag); node.textContent = text; return node; }
function chart(summary) {
  const svg = byId('rain-chart'), times = (summary.rainfall_preview_times_s || summary.rainfall_times_s).map(function hours(t) { return t/3600; }), rates = (summary.rainfall_preview_rates_m_s || summary.rainfall_rates_m_s).map(function millimetresPerHour(r) { return r*3.6e6; });
  svg.replaceChildren();
  const peak = Math.max(...rates, 1), duration = times[times.length-1];
  const line = document.createElementNS('http://www.w3.org/2000/svg','polyline');
  line.setAttribute('points', times.map(function chartPoint(t,i) { return `${45+430*t/duration},${220-190*rates[i]/peak}`; }).join(' '));
  line.setAttribute('fill','none'); line.setAttribute('stroke','#08796e'); line.setAttribute('stroke-width','3'); svg.append(line);
  for (const [x,y,text] of [[5,20,`${peak.toPrecision(3)} mm/h`],[45,245,'0 h'],[380,245,`${duration.toPrecision(3)} h`],[5,222,'0']]) {
    const label = document.createElementNS('http://www.w3.org/2000/svg','text'); label.setAttribute('x',x); label.setAttribute('y',y); label.setAttribute('font-size','13'); label.textContent=text; svg.append(label);
  }
  byId('rain-label').textContent = summary.rainfall+' · Scaling multiplier: ' + summary.multiplier_range.map(formatMapValue).join(' to ');
}
function review(result) {
  const s = result.summary;
  byId('review').hidden = false;
  const roughness=s.roughness_mode==='map'?'spatial Manning n '+s.roughness_range.map(formatMapValue).join('–'):'Manning n '+s.manning_n;
  const infiltration=s.infiltration_mode==='exponential'?'spatial exponential infiltration (f0/fd/k)':s.infiltration_mode==='map'?'spatial constant infiltration '+s.infiltration_range_mm_h.map(formatMapValue).join('–')+' mm/h':'constant infiltration '+s.infiltration_mm_h+' mm/h';
  const boundary=s.boundary==='free'?'FREE exterior':s.boundary==='closed'?'Closed':'Explicit faces';
  byId('summary').textContent = `FV1 · FP64 · ${s.execution} · resolved device ${s.device} · ${boundary} (${s.free_faces} FREE, ${s.stage_faces} stage faces) · ${roughness} · ${infiltration} · ${s.active_cells.toLocaleString()} active cells · ${s.cell_size_m.join(' × ')} m · ${s.duration_s} s · ${s.numerical_profile} · CFL ${s.numerical_defaults.cfl} · dt ${s.numerical_defaults.minimum_dt_s}–${s.numerical_defaults.maximum_dt_s} s · ${s.numerical_defaults.stage_boundary_momentum}`;
  if (s.stage_range_m) byId('summary').textContent += ' · absolute stage '+s.stage_range_m.map(formatMapValue).join('–')+' m '+s.vertical_datum;
  if (s.exponential_ranges) byId('summary').textContent += ' · f0 '+s.exponential_ranges.f0.map(formatMapValue).join('–')+' m/s · fd '+s.exponential_ranges.fd.map(formatMapValue).join('–')+' m · k '+s.exponential_ranges.conductivity.map(formatMapValue).join('–')+' m/s';
  byId('terrain-preview').src = 'data:image/png;base64,'+result.terrain_png;
  chart(s); showTerrainMap(result);
}
async function validate(submitted) {
  if (!byId('run-form').reportValidity()) return;
  byId('message').textContent = 'Validating local inputs…';
  try { const result = await api('/validate', submitted instanceof Event || submitted === undefined ? config() : submitted); review(result); byId('message').textContent = 'Inputs valid.'; return result; }
  catch(error) { byId('message').textContent = error.message; }
}
async function start(event) {
  event.preventDefault();
  if (busy) return;
  busy = true; byId('run').disabled = true;
  try {
    const submitted = config();
    const result = await validate(submitted);
    if (!result) return;
    clearResultMaps(); await api('/run', submitted);
    byId('message').textContent = 'Simulation started.';
    await refresh();
  } catch(error) { byId('message').textContent = error.message; }
  finally { busy = false; }
}
async function refresh() {
  try {
    const s = await api('/status');
    byId('run').disabled = busy || s.state === 'running';
    if (s.state === 'idle') return;
    byId('progress').hidden = false;
    byId('state').textContent = s.state.charAt(0).toUpperCase()+s.state.slice(1);
    byId('bar').max = s.end_time_s; byId('bar').value = s.time_s;
    byId('timing').textContent = `Simulated ${s.time_s} / ${s.end_time_s} s · elapsed ${s.elapsed_s.toFixed(1)} s`;
    byId('output').textContent = s.output;
    byId('run-error').textContent = s.error || '';
    byId('downloads').replaceChildren();
    for (const name of s.files || []) {
      if (name === 'status.json' || name === 'request.json' || name.endsWith('.png')) continue;
      const link = textNode('a',name); link.href = '/result?name='+encodeURIComponent(name); byId('downloads').append(link);
    }
    byId('maximum-preview').hidden = s.state !== 'completed';
    if (s.state === 'completed' && byId('maximum-preview').dataset.run !== s.output) {byId('maximum-preview').src = '/result?name=maximum_depth.png'; byId('maximum-preview').dataset.run=s.output;}
    if (s.state === 'running') byId('maximum-preview').removeAttribute('src');
    await showResultMap(s);
  } catch(error) { byId('message').textContent = 'Progress unavailable: '+error.message; }
}
async function browse(path) {
  try {
    const result = await api('/browse',{path}); browseDirectory=result.path;
    byId('browse-path').value=result.path; byId('entries').replaceChildren(); byId('browse-error').textContent='';
    const parent=textNode('button','↰ Parent directory'); parent.onclick=function parentDirectory() { browse(result.parent); }; byId('entries').append(parent);
    for (const entry of result.entries) {
      const button=textNode('button',(entry.directory?'▸ ':'')+entry.name);
      button.onclick=function selectPath() { if(entry.directory) browse(entry.path); else {byId(pickerTarget).value=entry.path; byId('picker').close();} };
      byId('entries').append(button);
    }
  } catch(error) {byId('browse-error').textContent=error.message;}
}
function setForm(values) {
  for (const [name,value] of Object.entries(values)) if (byId(name)) byId(name).value=value;
  visibility();
}
function loadSmallExample() {
  const parent=byId('output_parent').value;
  setForm(ordinaryDefaults); setForm(exampleInputs);
  byId('output_parent').value=parent; byId('domain_mode').value='raster';
  clearResultMaps(); visibility(); byId('preset-message').textContent='Small synthetic control loaded · 120 s. Press Validate & preview or Run.';
}
async function loadOrton() {
  byId('preset-message').textContent='Checking original inputs and preparing the full-domain setup…';
  try {
    const result=await api('/preset/orton',{directory:byId('orton_directory').value,output_parent:byId('output_parent').value});
    clearResultMaps(); setForm(result.config);
    const checked=await validate(result.config);
    if (!checked) throw new Error(byId('message').textContent);
    byId('preset-message').textContent='Orton Bradley full domain loaded · '+result.manifest.active_cells.toLocaleString()+' active cells · '+result.manifest.coastal_stage_faces+' explicit coastal contacts · 72-hour event. Press Run to simulate.';
  } catch(error) {byId('preset-message').textContent=error.message;}
}
async function init() {
  try {
    const b=await api('/bootstrap'); token=b.token; browseDirectory=b.directory; ordinaryDefaults=b.ordinary_defaults; exampleInputs=b.example;
    byId('orton_directory').value=b.orton_directory;
    for (const [name,value] of Object.entries(b.defaults)) if (byId(name)) byId(name).value=value;
    for (const device of b.devices) {const option=textNode('option',device); option.value=device; byId('device').append(option);}
    if (b.validated) review(b.validated);
    byId('example').onclick=loadSmallExample; byId('orton').onclick=loadOrton;
    for(const name of ['domain_mode','distribution','roughness_mode','infiltration_mode','boundary']) byId(name).onchange=visibility;
    byId('validate').onclick=validate; byId('run-form').onsubmit=start;
    for(const button of document.querySelectorAll('[data-browse]')) button.onclick=function openPicker() {pickerTarget=button.dataset.browse; byId('choose-directory').hidden=!['output_parent','orton_directory'].includes(pickerTarget); byId('picker').showModal(); browse(byId(pickerTarget).value || browseDirectory);};
    byId('browse-go').onclick=function goToPath() { browse(byId('browse-path').value); };
    byId('choose-directory').onclick=function chooseDirectory() {byId(pickerTarget).value=browseDirectory; byId('picker').close();};
    byId('close-picker').onclick=function closePicker() { byId('picker').close(); };
    visibility(); await refresh();
    if (!b.has_configuration && b.orton_directory) await loadOrton();
    byId('controls').disabled=false; setInterval(refresh,1500);
  } catch(error) {byId('message').textContent=error.message;}
}
init();
