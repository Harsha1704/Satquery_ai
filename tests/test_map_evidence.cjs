const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync('frontend/static/js/map.js', 'utf8');
const records = [
  {analysis_id:'job', path:'imagery/gee_2024_test.tif', georeferenced:true, crs:'EPSG:4326', bounds:[74,29,75,30], transform:[.001,0,74,0,-.001,30]},
  {analysis_id:'job', path:'outputs/evidence/scene_2024_rgb.png', georeferenced:false},
  {analysis_id:'job', path:'mask.png', georeferenced:true, crs:'EPSG:4326', bounds:[74,29,75,30]},
  {analysis_id:'job', path:'sheet.png', georeferenced:false},
  {analysis_id:'job', path:'difference.png', georeferenced:false},
  {analysis_id:'job', path:'preview.png', georeferenced:false},
];
const job = {job_id:'job', result:{provenance:{evidence_identity:records}}};
let added, cleared = 0, comparison = 0, preview = 0;
const context = {
  currentAnalysisId:'job', ANALYSIS_SOURCE:'source', ANALYSIS_LAYER:'layer', AOI_LINE:'aoi',
  map:{isStyleLoaded:()=>true, addSource:(id,value)=>{added=value;}, addLayer:()=>{}, getLayer:()=>true},
  els:{analysisLayerToggle:{}, activeLayerLabel:{}, mapHint:{}},
  clearAnalysisOverlay:()=>{cleared++;}, artifactUrl:(id,path)=>path,
  taskLayerStats:()=>({artifact:'mask.png'}), comparisonData:()=>({comparison_artifact:'sheet.png', change_artifact:'preview.png'}),
  evidenceLabel:v=>v, evidenceRole:()=>'', renderAnalysisLegend:()=>{},
  document:{querySelectorAll:()=>[]}, openComparison:()=>{comparison++;},
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function evidenceBounds('),source.indexOf('function showTemporalChangePolygons(')), context);
context.openEvidencePreview = ()=>{preview++;};
context.showEvidenceOnMap(job,'outputs/evidence/scene_2024_rgb.png');
assert.equal(added.url,'outputs/evidence/scene_2024_rgb.png');
assert.equal(JSON.stringify(added.coordinates),'[[74,30],[75,30],[75,29],[74,29]]');
context.showEvidenceOnMap(job,'preview.png');
assert.equal(added.url,'mask.png');
context.showEvidenceOnMap(job,'sheet.png');
assert.equal(comparison,1);
context.showEvidenceOnMap(job,'difference.png');
assert.equal(preview,1);
assert.equal(cleared,2, 'Preview must preserve geographic overlay');
context.showEvidenceOnMap({...job,job_id:'stale'},'mask.png');
assert.equal(cleared,2, 'Stale analysis must not replace the overlay');
records[0].bounds[0]=null;
assert.equal(context.evidenceBounds(job,'outputs/evidence/scene_2024_rgb.png'),null);
records[0].bounds[0]=74;
records[0].transform[1]=.1;
assert.equal(context.evidenceBounds(job,'outputs/evidence/scene_2024_rgb.png'),null);
console.log('Map evidence switching, preview routing, invalid bounds and stale-result checks passed.');
