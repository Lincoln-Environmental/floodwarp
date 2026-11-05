import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const element = (id) => document.getElementById(id);
const state = { index: -1, request: 0, playing: false, frame: null, cache: new Map(), cacheLimit: 6, timer: null, ready: false };
let manifest, renderer, scene, camera, controls, water, terrainIndices, waterIndices, depthAttribute;
let canvasWidth = 1, canvasHeight = 1, modelBounds;
const frameInterval = 300;
const chartPadding = { left: 34, right: 9, top: 8, bottom: 23 };

function setLoading(message) {
  element('loading').textContent = message;
  element('loading').hidden = !message;
}
function fail(error) {
  console.error(error);
  setLoading('');
  element('fatal').hidden = false;
  element('fatal').textContent = `The viewer could not load this data. ${error.message}. Use a current browser with WebGL2 and start the local HTTP viewer.`;
}
async function loadBinary(resource, signal) {
  const response = await fetch(resource.path, { signal });
  if (!response.ok) throw new Error(`HTTP ${response.status} loading ${resource.path}`);
  let buffer = await response.arrayBuffer();
  const magic = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  if (magic[0] === 0x1f && magic[1] === 0x8b) {
    const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
    buffer = await new Response(stream).arrayBuffer();
  }
  if (buffer.byteLength !== resource.raw_bytes) throw new Error('Unexpected data length');
  const digest = await crypto.subtle.digest('SHA-256', buffer);
  const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
  if (hash !== resource.raw_sha256) throw new Error('Display data checksum differs');
  return buffer;
}
function buildTerrainIndices(valid, ny, nx) {
  const indices = new Uint32Array((ny - 1) * (nx - 1) * 6);
  let length = 0;
  for (let row = 0; row < ny - 1; row++) {
    for (let col = 0; col < nx - 1; col++) {
      const a = row * nx + col, b = a + 1, c = a + nx, d = c + 1;
      if (valid[a] && valid[b] && valid[c]) { indices[length++] = a; indices[length++] = b; indices[length++] = c; }
      if (valid[b] && valid[d] && valid[c]) { indices[length++] = b; indices[length++] = d; indices[length++] = c; }
    }
  }
  return indices.slice(0, length);
}
function buildWaterIndices(wet) {
  let length = 0;
  for (let j = 0; j < terrainIndices.length; j += 3) {
    const a = terrainIndices[j], b = terrainIndices[j + 1], c = terrainIndices[j + 2];
    if (wet[a] && wet[b] && wet[c]) { waterIndices[length++] = a; waterIndices[length++] = b; waterIndices[length++] = c; }
  }
  return length;
}
function setupScene(buffer) {
  const [ny, nx] = manifest.grid.shape;
  const count = ny * nx;
  const positions = new Float32Array(buffer, 0, count * 3);
  const colours = new Uint8Array(buffer, count * 12, count * 3);
  const valid = new Uint8Array(buffer, count * 15, count);
  terrainIndices = buildTerrainIndices(valid, ny, nx);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3, true));
  geometry.setIndex(new THREE.BufferAttribute(terrainIndices, 1));
  geometry.computeBoundingSphere();
  modelBounds = geometry.boundingSphere.clone();
  const terrainMaterial = new THREE.ShaderMaterial({
    vertexShader: `#include <common>
      #include <logdepthbuf_pars_vertex>
      attribute vec3 color; varying vec3 vColor;
      void main(){vColor=color;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);
      #include <logdepthbuf_vertex>
      }`,
    fragmentShader: `#include <logdepthbuf_pars_fragment>
      varying vec3 vColor; void main(){gl_FragColor=vec4(vColor,1.0);
      #include <logdepthbuf_fragment>
      }`
  });
  scene = new THREE.Scene();
  scene.add(new THREE.Mesh(geometry, terrainMaterial));
  const waterGeometry = new THREE.BufferGeometry();
  waterGeometry.setAttribute('position', geometry.getAttribute('position'));
  depthAttribute = new THREE.BufferAttribute(new Float32Array(count), 1).setUsage(THREE.DynamicDrawUsage);
  waterGeometry.setAttribute('depth', depthAttribute);
  waterIndices = new Uint32Array(terrainIndices.length);
  waterGeometry.setIndex(new THREE.BufferAttribute(waterIndices, 1).setUsage(THREE.DynamicDrawUsage));
  waterGeometry.setDrawRange(0, 0);
  const waterMaterial = new THREE.ShaderMaterial({
    uniforms: { opacity: { value: Number(element('opacity').value) }, minimum: { value: manifest.configuration.wet_threshold_m }, maximum: { value: manifest.configuration.depth_maximum_m } },
    vertexShader: `#include <common>
      #include <logdepthbuf_pars_vertex>
      attribute float depth; varying float vDepth;
      void main(){vec3 p=position;p.y+=depth;vDepth=depth;gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.0);
      #include <logdepthbuf_vertex>
      }`,
    fragmentShader: `#include <logdepthbuf_pars_fragment>
      varying float vDepth;uniform float opacity;uniform float minimum;uniform float maximum;
      void main(){float t=clamp(log(max(vDepth,minimum)/minimum)/log(maximum/minimum),0.0,1.0);
      vec3 a=vec3(88.,223.,235.)/255.;vec3 b=vec3(43.,173.,224.)/255.;vec3 c=vec3(22.,119.,198.)/255.;vec3 d=vec3(9.,59.,140.)/255.;
      vec3 color=t<1./3.?mix(a,b,t*3.):t<2./3.?mix(b,c,t*3.-1.):mix(c,d,t*3.-2.);gl_FragColor=vec4(color,opacity);
      #include <logdepthbuf_fragment>
      }`,
    transparent: true, depthWrite: false, depthTest: true
  });
  water = new THREE.Mesh(waterGeometry, waterMaterial);
  water.frustumCulled = false;
  scene.add(water);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true, reversedDepthBuffer: true, logarithmicDepthBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  element('scene').appendChild(renderer.domElement);
  camera = new THREE.PerspectiveCamera(38, 1, 2, 40000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.minDistance = 30;
  controls.maxDistance = 22000;
  controls.maxPolarAngle = Math.PI / 2 - 0.025;
  controls.addEventListener('change', drawScene);
  resetView(false);
  new ResizeObserver(resizeScene).observe(element('scene'));
  renderer.setAnimationLoop(() => { if (controls.update()) drawScene(); });
  renderer.domElement.addEventListener('webglcontextlost', (event) => { event.preventDefault(); fail(new Error('WebGL context lost; reload the viewer')); });
  resizeScene();
}
function resetView(plan) {
  const elevation = manifest.elevation_range_m;
  const target = new THREE.Vector3(0, elevation[0] + .2 * (elevation[1] - elevation[0]), 0);
  controls.target.copy(target);
  if (plan) camera.position.set(0, 10000, 1);
  else camera.position.set(-4300, 5800, -7400);
  controls.update();
  drawScene();
}
function resizeScene() {
  const rectangle = element('scene').getBoundingClientRect();
  canvasWidth = Math.max(1, rectangle.width); canvasHeight = Math.max(1, rectangle.height);
  renderer.setSize(canvasWidth, canvasHeight);
  camera.aspect = canvasWidth / canvasHeight;
  camera.updateProjectionMatrix();
  drawScene();
  if (state.frame) drawCharts(state.frame.time_s);
}
function drawScene() {
  const distance = camera.position.distanceTo(modelBounds.center);
  camera.near = Math.max(.1, distance - modelBounds.radius * 1.05);
  camera.far = Math.max(camera.near + 100, distance + modelBounds.radius * 1.05);
  camera.updateProjectionMatrix();
  renderer.render(scene, camera);
  const bearing = Math.atan2(camera.position.x - controls.target.x, camera.position.z - controls.target.z);
  element('north-arrow').style.transform = `rotate(${-bearing * 180 / Math.PI}deg)`;
}
function formatClock(seconds) {
  return `${Math.floor(seconds / 3600).toString().padStart(2, '0')}:${Math.floor(seconds % 3600 / 60).toString().padStart(2, '0')}`;
}
function drawChart(id, values, seconds, colour) {
  const canvas = element(id), bounds = canvas.getBoundingClientRect(), ratio = Math.min(devicePixelRatio, 2);
  canvas.width = Math.round(bounds.width * ratio); canvas.height = Math.round(bounds.height * ratio);
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  const { left, right, top, bottom } = chartPadding;
  const width = bounds.width - left - right, height = bounds.height - top - bottom;
  const maximum = Math.max(1, ...values) * 1.08;
  const x = (s) => left + s / 259200 * width;
  const y = (v) => top + height * (1 - v / maximum);
  context.fillStyle = '#10212b'; context.fillRect(left, top, width, height);
  context.lineWidth = 1; context.font = '9px system-ui';
  for (let tick = 0; tick <= 3; tick++) {
    const v = maximum * tick / 3, py = y(v);
    context.strokeStyle = '#29414b'; context.beginPath(); context.moveTo(left, py); context.lineTo(left + width, py); context.stroke();
    context.fillStyle = '#91adb9'; context.textAlign = 'right'; context.fillText(v >= 10 ? v.toFixed(0) : v.toFixed(1), left - 6, py + 3);
  }
  for (let hour = 0; hour <= 72; hour += 12) {
    const px = x(hour * 3600);
    context.strokeStyle = '#203640'; context.beginPath(); context.moveTo(px, top); context.lineTo(px, top + height); context.stroke();
    context.fillStyle = '#91adb9'; context.textAlign = 'center'; context.fillText(`${hour}`, px, top + height + 15);
  }
  const times = manifest.diagnostics.time_s;
  context.beginPath(); context.moveTo(x(0), y(0));
  for (let i = 0; i < values.length; i++) context.lineTo(x(times[i]), y(values[i]));
  context.lineTo(x(259200), y(0)); context.closePath(); context.fillStyle = `${colour}18`; context.fill();
  context.beginPath();
  for (let i = 0; i < values.length; i++) { if (!i) context.moveTo(x(times[i]), y(values[i])); else context.lineTo(x(times[i]), y(values[i])); }
  context.strokeStyle = colour; context.lineWidth = 1.5; context.stroke();
  context.strokeStyle = '#f0c67b'; context.lineWidth = 1.3; context.beginPath(); context.moveTo(x(seconds), top); context.lineTo(x(seconds), top + height); context.stroke();
  const selected = values[Math.round(seconds / 60)];
  context.beginPath(); context.arc(x(seconds), y(selected), 3, 0, 2 * Math.PI); context.fillStyle = '#f0c67b'; context.fill();
  canvas.setAttribute('aria-valuenow', String(seconds / 3600));
  canvas.setAttribute('aria-valuetext', `${formatClock(seconds)} elapsed`);
}
function drawCharts(seconds) {
  drawChart('hydrograph', manifest.diagnostics.discharge_m3_s, seconds, '#63c7ed');
  drawChart('hyetograph', manifest.diagnostics.rainfall_mm_h, seconds, '#76bdb1');
}
async function selectFrame(index) {
  index = Math.max(0, Math.min(manifest.frames.length - 1, Math.round(index)));
  const request = ++state.request;
  if (state.abort) state.abort.abort();
  if (index === state.index && state.frame) {
    element('time').value = String(index);
    element('hours').value = Number((state.frame.time_s / 3600).toFixed(6));
    setLoading('');
    return true;
  }
  state.abort = new AbortController();
  const record = manifest.frames[index];
  setLoading(`Loading ${formatClock(record.time_s)}…`);
  try {
    let buffer = state.cache.get(index);
    if (!buffer) buffer = await loadBinary(record, state.abort.signal);
    if (request !== state.request) return false;
    state.cache.delete(index); state.cache.set(index, buffer);
    while (state.cache.size > state.cacheLimit) state.cache.delete(state.cache.keys().next().value);
    const count = manifest.grid.shape[0] * manifest.grid.shape[1];
    depthAttribute.array.set(new Float32Array(buffer, 0, count));
    depthAttribute.needsUpdate = true;
    const wet = new Uint8Array(buffer, count * 4, count);
    const length = buildWaterIndices(wet);
    water.geometry.index.needsUpdate = true;
    water.geometry.setDrawRange(0, length);
    state.index = index; state.frame = record;
    const second = record.time_s, diagnostic = Math.round(second / 60);
    element('clock').textContent = formatClock(second);
    element('time').value = String(index);
    element('hours').value = Number((second / 3600).toFixed(6));
    element('discharge').textContent = `${manifest.diagnostics.discharge_m3_s[diagnostic].toFixed(2)} m³/s`;
    element('rainfall').textContent = `${manifest.diagnostics.rainfall_mm_h[diagnostic].toFixed(2)} mm/h`;
    element('wet-area').textContent = `${(record.wet_cells * manifest.grid.dx * manifest.grid.dy / 1e6).toFixed(3)} km²`;
    element('maximum-depth').textContent = `${record.native_maximum_depth_m.toFixed(3)} m`;
    drawCharts(second); drawScene(); setLoading('');
    element('back').disabled = index === 0; element('forward').disabled = index === manifest.frames.length - 1;
    history.replaceState(null, '', `#t=${second}`);
    return true;
  } catch (error) {
    if (request !== state.request || error.name === 'AbortError') return false;
    pause(); setLoading(''); fail(error); return false;
  }
}
function pause() {
  state.playing = false; clearTimeout(state.timer);
  element('play').textContent = '▶ Play'; element('play').setAttribute('aria-label', 'Play the event');
}
async function playback() {
  if (!state.playing) return;
  if (state.index >= manifest.frames.length - 1) { pause(); return; }
  await selectFrame(state.index + 1);
  if (state.playing) state.timer = setTimeout(playback, 1000 / Number(element('speed').value));
}
function selectUserFrame(index) { pause(); selectFrame(index); }
function bindChart(id) {
  const canvas = element(id);
  const choose = (event) => {
    const rect = canvas.getBoundingClientRect();
    const fraction = (event.clientX - rect.left - chartPadding.left) / (rect.width - chartPadding.left - chartPadding.right);
    selectUserFrame(fraction * (manifest.frames.length - 1));
  };
  canvas.addEventListener('pointerdown', (event) => { canvas.setPointerCapture(event.pointerId); choose(event); });
  canvas.addEventListener('pointermove', (event) => { if (canvas.hasPointerCapture(event.pointerId)) choose(event); });
  canvas.addEventListener('pointerup', (event) => { if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId); });
  canvas.addEventListener('keydown', (event) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') { event.preventDefault(); selectUserFrame(state.index + (event.key === 'ArrowLeft' ? -1 : 1)); }
    if (event.key === 'Home' || event.key === 'End') { event.preventDefault(); selectUserFrame(event.key === 'Home' ? 0 : manifest.frames.length - 1); }
  });
}
function bindControls() {
  element('time').max = manifest.frames.length - 1;
  element('time').addEventListener('input', (event) => selectUserFrame(Number(event.target.value)));
  element('hours').addEventListener('change', (event) => { const hour = Number(event.target.value); if (Number.isFinite(hour)) selectUserFrame(hour * 3600 / frameInterval); });
  element('back').addEventListener('click', () => selectUserFrame(state.index - 1));
  element('forward').addEventListener('click', () => selectUserFrame(state.index + 1));
  element('start').addEventListener('click', () => selectUserFrame(0));
  element('end').addEventListener('click', () => selectUserFrame(manifest.frames.length - 1));
  element('discharge-peak').addEventListener('click', () => selectUserFrame(manifest.initial_frame));
  element('inundation-peak').addEventListener('click', () => selectUserFrame(manifest.peak_inundation_frame));
  element('play').addEventListener('click', () => { if (state.playing) pause(); else { state.playing = true; element('play').textContent = 'Ⅱ Pause'; element('play').setAttribute('aria-label', 'Pause the event'); if (state.index === manifest.frames.length - 1) selectFrame(0).then(playback); else playback(); } });
  element('reset').addEventListener('click', () => resetView(false));
  element('overhead').addEventListener('click', () => resetView(true));
  const zoom = (factor) => { camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target); controls.update(); drawScene(); };
  element('zoom-in').addEventListener('click', () => zoom(.8)); element('zoom-out').addEventListener('click', () => zoom(1.25));
  element('opacity').addEventListener('input', (event) => { water.material.uniforms.opacity.value = Number(event.target.value); drawScene(); });
  element('fullscreen').addEventListener('click', async () => { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch (error) { console.warn(error); } });
  bindChart('hydrograph'); bindChart('hyetograph');
}
async function start() {
  const response = await fetch('./data/manifest.json');
  if (!response.ok) throw new Error('Build the browser dataset before starting the viewer');
  manifest = await response.json();
  const terrain = await loadBinary(manifest.terrain);
  setupScene(terrain); bindControls();
  element('precision').textContent = `Measured maximum browser depth rounding: ${manifest.maximum_display_depth_error_m.toExponential(2)} m. Native FP64 wet/dry classification is preserved.`;
  element('data-credit').textContent = manifest.attribution;
  const bookmark = new URLSearchParams(location.hash.slice(1)).get('t');
  const initial = bookmark !== null && Number.isFinite(Number(bookmark)) ? Number(bookmark) / frameInterval : manifest.initial_frame;
  if (!await selectFrame(initial)) throw new Error('Initial result could not be loaded');
  state.ready = true;
  document.documentElement.dataset.viewerReady = 'true';
  // Read-only diagnostics support repeatable browser QA without exposing solver actions.
  window.floodwarpViewer = { inspect: () => ({ ready: state.ready, index: state.index, time_s: state.frame?.time_s, wet_triangles: water.geometry.drawRange.count / 3, cached_frames: state.cache.size, camera: camera.position.toArray(), target: controls.target.toArray(), rendered_frames: renderer.info.render.frame, depth_buffer: renderer.capabilities.reversedDepthBuffer ? 'reversed' : 'logarithmic' }) };
}
start().catch(fail);
