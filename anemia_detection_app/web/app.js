const $ = (selector) => document.querySelector(selector);
let activeSplit = 'training';
let resultData = { metrics: null, history: {}, charts: {} };
const tooltip = $('#tooltip');

function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char])); }
async function api(url, options = {}) { const response = await fetch(url, options); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Request failed.'); return data; }
function showTooltip(event, text) { tooltip.textContent = text; tooltip.hidden = false; tooltip.style.left = `${event.clientX + 12}px`; tooltip.style.top = `${event.clientY + 12}px`; }
function hideTooltip() { tooltip.hidden = true; }
function empty(target, message) { target.innerHTML = `<div class="chart-empty">${escapeHtml(message)}</div>`; }
function fmt(value) { return typeof value === 'number' ? value.toFixed(3) : '—'; }

function renderMetrics() {
  const target = $('#metrics'); const metrics = resultData.metrics;
  if (!metrics) { target.innerHTML = '<div class="empty-state">Train or load a model to see its evaluation.</div>'; return; }
  target.innerHTML = ['training', 'validation', 'testing'].map(split => {
    const m = metrics[split] || {}; return `<article class="metric-card"><small>${split[0].toUpperCase() + split.slice(1)} accuracy</small><strong>${m.accuracy == null ? '—' : `${(m.accuracy * 100).toFixed(1)}%`}</strong><span>F1 score ${fmt(m.f1_score)}</span></article>`;
  }).join('');
}
function renderConfusion() {
  const target = $('#confusion-chart'); const m = resultData.metrics?.[activeSplit]; const chart = resultData.charts?.[activeSplit] || {};
  const values = chart.confusion_matrix || m?.confusion_matrix; const labels = chart.class_names || ['anemic', 'non_anemic'];
  $('#matrix-title').textContent = `${activeSplit[0].toUpperCase() + activeSplit.slice(1)} confusion matrix`;
  if (!values?.length) return empty(target, 'Confusion-matrix data is not available for this run.');
  const max = Math.max(1, ...values.flat());
  const headers = labels.map(x => `<th scope="col">${escapeHtml(x)}</th>`).join('');
  const rows = values.map((row, r) => `<tr><th scope="row">${escapeHtml(labels[r] || `Class ${r + 1}`)}</th>${row.map((value, c) => { const alpha = .13 + .75 * (value / max); return `<td tabindex="0" style="background:rgba(23,123,115,${alpha})" data-cell="Actual: ${labels[r] || r}; predicted: ${labels[c] || c}; ${value} image${value === 1 ? '' : 's'}">${value}</td>`; }).join('')}</tr>`).join('');
  target.innerHTML = `<table class="matrix"><caption>Rows are actual classes; columns are predicted classes.</caption><thead><tr><th></th>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
  target.querySelectorAll('[data-cell]').forEach(cell => { cell.addEventListener('mouseenter', event => showTooltip(event, cell.dataset.cell)); cell.addEventListener('mousemove', event => showTooltip(event, cell.dataset.cell)); cell.addEventListener('mouseleave', hideTooltip); cell.addEventListener('focus', event => showTooltip(event, cell.dataset.cell)); cell.addEventListener('blur', hideTooltip); });
}
function svgShell(width=520, height=280) { return { width, height, left:50, right:18, top:16, bottom:38, plotWidth:width-68, plotHeight:height-54 }; }
function lineChart(target, series, title, xLabel, yLabel, fixedDomain = null) {
  if (!series.length || !series.some(s => s.points?.length)) return empty(target, 'No curve data was saved for this run.');
  const dim = svgShell(); const all = series.flatMap(s => s.points); const xs = all.map(p => p.x); const ys = all.map(p => p.y); const x0 = fixedDomain?.[0] ?? Math.min(...xs); const x1 = fixedDomain?.[1] ?? Math.max(...xs); const y0 = fixedDomain?.[2] ?? Math.min(...ys); const y1 = fixedDomain?.[3] ?? Math.max(...ys); const x = v => dim.left + ((v-x0)/((x1-x0)||1))*dim.plotWidth; const y = v => dim.top + dim.plotHeight - ((v-y0)/((y1-y0)||1))*dim.plotHeight;
  const ticks = Array.from({length:5}, (_,i) => y0 + (i*(y1-y0)/4)); const grids = ticks.map(t => `<line class="grid" x1="${dim.left}" y1="${y(t)}" x2="${dim.left+dim.plotWidth}" y2="${y(t)}"/><text class="axis-label" x="${dim.left-8}" y="${y(t)+4}" text-anchor="end">${t.toFixed(2)}</text>`).join('');
  const xTicks = Array.from({length:5}, (_,i) => x0 + (i*(x1-x0)/4)).map(t => `<text class="axis-label" x="${x(t)}" y="${dim.top+dim.plotHeight+20}" text-anchor="middle">${Number.isInteger(t) ? t : t.toFixed(2)}</text>`).join('');
  const paths = series.map(s => `<path class="${s.className}" d="M ${s.points.map(p => `${x(p.x)},${y(p.y)}`).join(' L ')}"/>${s.points.map(p => `<circle class="point" fill="${s.color}" cx="${x(p.x)}" cy="${y(p.y)}" r="4" data-tip="${escapeHtml(`${s.name}: ${p.y.toFixed(3)} at ${xLabel.toLowerCase()} ${p.x}`)}"/>`).join('')}`).join('');
  target.innerHTML = `<svg viewBox="0 0 ${dim.width} ${dim.height}" role="img" aria-label="${escapeHtml(title)}"><title>${escapeHtml(title)}</title>${grids}<line class="axis" x1="${dim.left}" y1="${dim.top}" x2="${dim.left}" y2="${dim.top+dim.plotHeight}"/><line class="axis" x1="${dim.left}" y1="${dim.top+dim.plotHeight}" x2="${dim.left+dim.plotWidth}" y2="${dim.top+dim.plotHeight}"/>${xTicks}${paths}<text class="axis-label" x="${dim.left+dim.plotWidth/2}" y="${dim.height-3}" text-anchor="middle">${escapeHtml(xLabel)}</text><text class="axis-label" transform="translate(13 ${dim.top+dim.plotHeight/2}) rotate(-90)" text-anchor="middle">${escapeHtml(yLabel)}</text></svg>`;
  target.querySelectorAll('[data-tip]').forEach(point => { point.addEventListener('mouseenter', event => showTooltip(event, point.dataset.tip)); point.addEventListener('mousemove', event => showTooltip(event, point.dataset.tip)); point.addEventListener('mouseleave', hideTooltip); });
}
function renderRoc() { const points = resultData.charts?.[activeSplit]?.roc || []; $('#roc-title').textContent = `${activeSplit[0].toUpperCase() + activeSplit.slice(1)} ROC curve`; lineChart($('#roc-chart'), [{name:'ROC',className:'line-train',color:'#177b73',points:points.map(p=>({x:p.false_positive_rate,y:p.true_positive_rate}))},{name:'Chance',className:'line-valid',color:'#e76f51',points:[{x:0,y:0},{x:1,y:1}]}], 'Receiver operating characteristic curve', 'False positive rate', 'True positive rate', [0,1,0,1]); }
function renderHistory() { const h = resultData.history || {}; const accuracy = []; const loss = []; if (h.accuracy) accuracy.push({name:'Training accuracy',className:'line-train',color:'#177b73',points:h.accuracy.map((y,i)=>({x:i+1,y}))}); if(h.val_accuracy) accuracy.push({name:'Validation accuracy',className:'line-valid',color:'#e76f51',points:h.val_accuracy.map((y,i)=>({x:i+1,y}))}); if(!accuracy.length && h.loss) loss.push({name:'Training loss',className:'line-train',color:'#177b73',points:h.loss.map((y,i)=>({x:i+1,y}))}); if(!accuracy.length && h.val_loss) loss.push({name:'Validation loss',className:'line-valid',color:'#e76f51',points:h.val_loss.map((y,i)=>({x:i+1,y}))}); lineChart($('#history-chart'), accuracy.length ? accuracy : loss, 'Fine-tuning curves', 'Epoch', accuracy.length ? 'Accuracy' : 'Loss'); }
function renderFigures() { renderConfusion(); renderRoc(); renderHistory(); }
async function refresh() { try { const status = await api('/api/status'); $('#dataset-root').value ||= status.default_dataset || ''; $('#log-box').textContent = status.logs.join('\n'); $('#log-box').scrollTop = $('#log-box').scrollHeight; const busy = status.training; $('#train-button').disabled = busy; $('#train-button').textContent = busy ? 'Training in progress…' : 'Start training →'; $('#predict-button').disabled = !status.model_loaded || busy; $('#run-status').classList.toggle('busy', busy); const accelerator = status.accelerator; const device = accelerator?.available ? `GPU ready · ${accelerator.devices.join(', ')}` : 'CPU mode · GPU unavailable'; $('#run-status').lastElementChild.textContent = busy ? `Training on ${accelerator?.available ? 'GPU' : 'CPU'}` : (status.model_loaded ? `Model loaded · ${device}` : device); if (status.model_loaded && !resultData.metrics) await loadResults(); } catch (error) { console.error(error); } }
async function loadResults() { resultData = await api('/api/results'); renderMetrics(); renderFigures(); }
$('#train-button').addEventListener('click', async () => { try { await api('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({dataset_root:$('#dataset-root').value,reducer:$('#reducer').value,classifier:$('#classifier').value,fine_tune:$('#fine-tune').checked})}); resultData={metrics:null,history:{},charts:{}}; renderMetrics(); renderFigures(); await refresh(); } catch(error) { alert(error.message); } });
$('#load-button').addEventListener('click', async () => { try { await api('/api/load-model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_directory:$('#model-directory').value})}); await loadResults(); await refresh(); } catch(error) { alert(error.message); } });
$('#predict-button').addEventListener('click', async () => { const image = $('#image-input').files[0]; if(!image) return alert('Choose an image first.'); const data = new FormData(); data.append('image',image); try { const result = await api('/api/predict',{method:'POST',body:data}); $('#prediction-result').innerHTML = `<strong>${escapeHtml(result.class_name)}</strong> · ${(result.confidence*100).toFixed(1)}% confidence`; await refresh(); } catch(error) { alert(error.message); } });
document.querySelectorAll('[data-split]').forEach(button => button.addEventListener('click', () => { activeSplit=button.dataset.split; document.querySelectorAll('[data-split]').forEach(b=>b.setAttribute('aria-selected',String(b===button))); renderFigures(); }));
$('#clear-log').addEventListener('click',()=>$('#log-box').textContent='');
refresh(); setInterval(refresh, 2500);
