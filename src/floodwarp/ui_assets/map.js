'use strict';
let modelMap = null, osmTiles = null, visibleOverlay = null, boundaryLines = null;
let terrainOverlay = null, resultOverlays = {}, mappedResult = '';
function initializeMap() {
  if (modelMap) return;
  modelMap = L.map('model-map', {zoomControl:true, attributionControl:true, fadeAnimation:false, zoomAnimation:false});
  osmTiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom:19, updateWhenIdle:true,
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
  });
  osmTiles.on('tileerror', function tileFailure() {document.getElementById('map-message').textContent='OpenStreetMap tiles unavailable. Local overlays and simulation outputs remain available.';});
  osmTiles.on('load', function tilesReady() {document.getElementById('map-message').textContent='';});
  if (document.getElementById('basemap').checked) osmTiles.addTo(modelMap);
  L.control.scale({imperial:false}).addTo(modelMap);
  document.getElementById('basemap').onchange = function toggleBasemap() {
    if (this.checked) osmTiles.addTo(modelMap); else modelMap.removeLayer(osmTiles);
  };
  document.getElementById('map-layer').onchange = chooseMapOverlay;
  document.getElementById('map-opacity').oninput = function overlayOpacity() {
    if (visibleOverlay) visibleOverlay.setOpacity(Number(this.value));
  };
}
function boundaryStyle(feature) {return {color:feature.properties.closure==='stage'?'#00bfe7':'#ef8123',weight:4};}
function describeBoundary(feature, layer) {
  const p=feature.properties;
  layer.bindTooltip(`${p.axis}[${p.row},${p.col}] · ${p.closure}`);
}
function showTerrainMap(result) {
  document.getElementById('map-panel').hidden=false;
  initializeMap(); terrainOverlay=result.terrain_map; clearResultMaps(true);
  if (boundaryLines) modelMap.removeLayer(boundaryLines);
  boundaryLines=L.geoJSON(result.boundary_geojson,{style:boundaryStyle,onEachFeature:describeBoundary}).addTo(modelMap);
  if (!resultOverlays.maximum) document.getElementById('map-layer').value='terrain';
  chooseMapOverlay(); modelMap.invalidateSize({animate:false});
  modelMap.fitBounds(terrainOverlay.bounds,{padding:[15,15],animate:false});
}
function chooseMapOverlay() {
  const choice=document.getElementById('map-layer').value;
  const overlay=choice==='terrain'?terrainOverlay:resultOverlays[choice];
  if (!overlay || !modelMap) return;
  if (visibleOverlay) modelMap.removeLayer(visibleOverlay);
  const url=overlay.png?'data:image/png;base64,'+overlay.png:'/result?name='+encodeURIComponent(overlay.file);
  visibleOverlay=L.imageOverlay(url,overlay.bounds,{opacity:Number(document.getElementById('map-opacity').value)}).addTo(modelMap);
  if (boundaryLines) boundaryLines.bringToFront();
  document.getElementById('map-legend').textContent=overlay.label+' · '+overlay.range.map(formatMapValue).join(' to ')+' · display reprojection only';
}
function formatMapValue(value) {return Number(value).toPrecision(5);}
async function showResultMap(status) {
  if (status.state!=='completed' || !status.files.includes('map.json') || mappedResult===status.output) return;
  const metadata=await api('/result?name=map.json');
  resultOverlays=metadata; mappedResult=status.output;
  if (metadata.terrain) terrainOverlay=metadata.terrain;
  if (metadata.boundaries && modelMap) {
    if (boundaryLines) modelMap.removeLayer(boundaryLines);
    boundaryLines=L.geoJSON(metadata.boundaries,{style:boundaryStyle,onEachFeature:describeBoundary}).addTo(modelMap);
  }
  const select=document.getElementById('map-layer');
  for (const name of ['maximum','final']) {
    if (!select.querySelector('option[value="'+name+'"]')) {
      const option=document.createElement('option'); option.value=name; option.textContent=metadata[name].label; select.append(option);
    }
  }
  select.value='maximum'; chooseMapOverlay();
}
function clearResultMaps(keepResultIdentity=false) {
  resultOverlays={}; if(!keepResultIdentity) mappedResult='';
  const select=document.getElementById('map-layer');
  for (const option of Array.from(select.options)) if(option.value!=='terrain') option.remove();
  select.value='terrain'; chooseMapOverlay();
}
