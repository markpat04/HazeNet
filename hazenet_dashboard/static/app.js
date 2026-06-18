// HazeNet Mission Control — frontend (vanilla JS, no deps)
const $  = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);
const fmt = (x, d=1) => (x==null||isNaN(x)) ? "—" : Number(x).toFixed(d);

async function api(path){
  try{ const r = await fetch(path); return await r.json(); }
  catch(e){ return {error:String(e)}; }
}

// ---------- nav ----------
$$("#nav button").forEach(b => b.onclick = () => {
  $$("#nav button").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  $$(".section").forEach(s => s.classList.remove("active"));
  $("#sec-" + b.dataset.sec).classList.add("active");
  if(b.dataset.sec === "attrib") loadAttribMeta();   // โหลดโมเดลครั้งแรกที่เปิดแท็บ
});

// ---------- tiny charts ----------
function line(canvas, series, opts={}){
  const c = canvas, ctx = c.getContext("2d");
  const W = c.width = c.clientWidth*2, H = c.height = (opts.h||190)*2;
  ctx.scale(1,1); ctx.clearRect(0,0,W,H);
  const pad = {l:54, r:14, t:14, b:28};
  const all = series.flatMap(s => s.y).filter(v=>v!=null&&!isNaN(v));
  if(!all.length){ ctx.fillStyle="#9aa7b4"; ctx.font="22px Segoe UI"; ctx.fillText("no data", pad.l, H/2); return; }
  let ymin = opts.ymin!=null?opts.ymin:Math.min(...all), ymax = Math.max(...all);
  if(opts.log){ ymin = Math.max(ymin, 1e-6); }
  const n = Math.max(...series.map(s=>s.y.length));
  const X = i => pad.l + (W-pad.l-pad.r) * (n<2?0:i/(n-1));
  const Yv = v => { if(opts.log){ const a=Math.log10(Math.max(v,1e-6)),b=Math.log10(Math.max(ymin,1e-6)),d=Math.log10(ymax)-b; return H-pad.b-(H-pad.t-pad.b)*(d?(a-b)/d:0);} return H-pad.b-(H-pad.t-pad.b)*((v-ymin)/((ymax-ymin)||1)); };
  // grid + y labels
  ctx.strokeStyle="#30363d"; ctx.fillStyle="#9aa7b4"; ctx.font="20px Segoe UI"; ctx.lineWidth=1;
  for(let g=0; g<=4; g++){ const v=ymin+(ymax-ymin)*g/4, y=Yv(v);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
    ctx.fillText(opts.log?Math.round(v): (Math.abs(v)>=1000?(v/1000).toFixed(0)+"k":fmt(v,0)), 4, y+6); }
  // series
  series.forEach(s=>{
    ctx.strokeStyle=s.color; ctx.lineWidth=s.w||3; ctx.beginPath(); let started=false;
    s.y.forEach((v,i)=>{ if(v==null||isNaN(v)){started=false;return;} const px=X(i),py=Yv(v);
      if(!started){ctx.moveTo(px,py);started=true;} else ctx.lineTo(px,py); });
    ctx.stroke();
  });
}
function bars(canvas, labels, values, colors, opts={}){
  const c=canvas, ctx=c.getContext("2d");
  const W=c.width=c.clientWidth*2, H=c.height=(opts.h||200)*2;
  ctx.clearRect(0,0,W,H);
  const pad={l:50,r:14,t:14,b:64};
  const vmax=Math.max(...values.filter(v=>v!=null&&!isNaN(v)),1);
  const bw=(W-pad.l-pad.r)/values.length;
  ctx.fillStyle="#9aa7b4"; ctx.font="19px Segoe UI";
  for(let g=0;g<=4;g++){const v=vmax*g/4,y=H-pad.b-(H-pad.t-pad.b)*(g/4);
    ctx.strokeStyle="#30363d";ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();
    ctx.fillText(fmt(v,0),6,y+6);}
  values.forEach((v,i)=>{ if(v==null||isNaN(v))return;
    const h=(H-pad.t-pad.b)*(v/vmax), x=pad.l+bw*i+bw*0.15, y=H-pad.b-h;
    ctx.fillStyle=colors[i]||"#58a6ff"; ctx.fillRect(x,y,bw*0.7,h);
    ctx.fillStyle="#e6edf3"; ctx.font="20px Segoe UI"; ctx.textAlign="center";
    ctx.fillText(fmt(v,0), x+bw*0.35, y-8);
    ctx.save(); ctx.translate(x+bw*0.35, H-pad.b+10); ctx.rotate(-Math.PI/5.5);
    ctx.fillStyle="#9aa7b4"; ctx.font="18px Segoe UI"; ctx.textAlign="right";
    ctx.fillText((labels[i]||"").slice(0,22), 0, 0); ctx.restore(); ctx.textAlign="left";
  });
}

// ---------- renderers ----------
function badge(s){ return `<span class="badge b-${s}">${s.toUpperCase()}</span>`; }

async function loadPipeline(){
  const d = await api("/api/pipeline");
  if(d.error){ $("#pipe").innerHTML=`<div class="err">${d.error}</div>`; return; }
  $("#scanned").textContent = "scanned: " + d.scanned;
  let ok=0, stale=0, miss=0;
  const html = d.stages.map((s,i)=>{
    if(s.status==="ok")ok++; else if(s.status==="stale")stale++; else miss++;
    const outs = s.outputs.slice(0,3).map(o=>`<div title="${o.path}">📄 ${o.path.split('/').pop()} · ${o.size} · ${o.ago}</div>`).join("") || `<div class="muted">— ยังไม่มี output —</div>`;
    const scr = s.scripts.map(x=>`<span class="pill">${x.name}</span>`).join(" ");
    const arrow = i<d.stages.length-1 ? '<div class="arrow">→</div>' : '';
    return `<div class="node"><div style="display:flex;justify-content:space-between;align-items:center">
      <span class="t">${s.title}</span>${badge(s.status)}</div>
      <div class="files">${outs}</div>
      <div class="meta">${scr}</div></div>${arrow}`;
  }).join("");
  $("#pipe").innerHTML = html;
  $("#kpis").innerHTML = [
    ["🟢 ready", ok, "var(--green)"],
    ["🟡 stale (re-run)", stale, "var(--amber)"],
    ["🔴 missing", miss, "var(--red)"],
    ["stages", d.stages.length, "var(--cyan)"],
  ].map(([l,v,c])=>`<div class="kpi"><div class="v" style="color:${c}">${v}</div><div class="l">${l}</div></div>`).join("");

  // code list + Run buttons (run = whole stage via hazenet CLI)
  $("#codeList").innerHTML = d.stages.map(s=>`<div class="card" style="margin-bottom:12px">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <h3 style="margin:0">${s.title} ${badge(s.status)}</h3>
      ${s.run?`<button class="refresh" style="padding:4px 14px;background:var(--green);color:#0d1117;font-weight:bold" onclick="runScript('${s.run}')">▶ run ${s.run}</button>`:`<span class="muted" style="font-size:11px">รันผ่าน CLI (ต้องมี key/CDS)</span>`}</div>
    <table style="margin-top:8px"><tr><th>module</th><th>doc</th><th>แก้ล่าสุด</th></tr>
    ${s.scripts.map(x=>`<tr><td class="mono">${x.path||x.name}${x.exists?"":" <span style='color:var(--red)'>(missing)</span>"}</td><td>${x.doc||"—"}</td><td>${x.ago}</td></tr>`).join("")}</table>
    <div class="muted" style="font-size:12px;margin-top:8px">outputs: ${s.outputs.map(o=>o.path.split('/').pop()).join(", ")||"—"}</div>
  </div>`).join("");
}

// ---------- run-script (subprocess + live log) ----------
let _pollTimer = null;
async function runScript(name){
  if(!confirm("รันสคริปต์ "+name+" ?\n(จะรันด้วย Python env hazenet — บางตัวหนัก เช่น train/download)")) return;
  showLogPanel(name);
  const r = await api("/api/run_script?name="+encodeURIComponent(name));
  if(r.error){ $("#logBody").textContent = "ERROR: "+r.error; return; }
  pollScript(name);
}
function showLogPanel(name){
  let p = $("#logPanel");
  if(!p){
    p = document.createElement("div"); p.id="logPanel";
    p.style.cssText="position:fixed;right:18px;bottom:18px;width:560px;max-width:92vw;height:330px;background:#0a0e13;border:1px solid var(--border);border-radius:12px;box-shadow:0 8px 40px #000a;z-index:50;display:flex;flex-direction:column";
    p.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 12px;border-bottom:1px solid var(--border)">
      <b id="logTitle" style="font-size:13px;color:var(--cyan)"></b>
      <span><span id="logState" class="pill"></span>
      <button class="refresh" style="padding:2px 8px" onclick="document.getElementById('logPanel').remove()">✕</button></span></div>
      <pre id="logBody" style="margin:0;flex:1;overflow:auto;font-size:11.5px;padding:10px 12px;background:#0a0e13;border:none;border-radius:0"></pre>`;
    document.body.appendChild(p);
  }
  $("#logTitle").textContent = "▶ "+name;
  $("#logBody").textContent = "starting…";
}
async function pollScript(name){
  if(_pollTimer) clearInterval(_pollTimer);
  const upd = async ()=>{
    const s = await api("/api/script_status?name="+encodeURIComponent(name));
    if(!$("#logPanel")) { clearInterval(_pollTimer); return; }
    $("#logBody").textContent = s.log || "(no output yet)";
    $("#logBody").scrollTop = $("#logBody").scrollHeight;
    const st = $("#logState");
    st.textContent = s.state + (s.elapsed?(" · "+s.elapsed+"s"):"");
    st.style.color = s.state==="running"?"var(--amber)":(s.state==="done"?"var(--green)":"var(--red)");
    if(s.state!=="running"){ clearInterval(_pollTimer); loadPipeline(); loadGallery(); }
  };
  upd(); _pollTimer = setInterval(upd, 1500);
}

async function loadData(){
  const d = await api("/api/eda");
  if(d.error){ $("#dataKpis").innerHTML=`<div class="err">${d.error}</div>`; return; }
  const cu=d.cube||{}, pm=d.pm25||{};
  $("#dataKpis").innerHTML = [
    [`${cu.H||"?"}×${cu.W||"?"}`, "กริด (ช่อง/แถว)"],
    [(cu.G||0).toLocaleString(), "ช่องกริดรวม G"],
    [cu.T||"?", "วัน (ก.พ.–เม.ย.)"],
    [pm.stations||"?", `สถานี PM2.5 (ไทย ${pm.thai||0}/อื่น ${pm.other||0})`],
  ].map(([v,l])=>`<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join("");

  if(d.pm25_series) line($("#pmChart"), [{y:d.pm25_series.value, color:"#56d4dd"}], {h:190});
  if(d.frp_series)  line($("#frpChart"), [{y:d.frp_series.value, color:"#e3b341"}], {h:190});

  if(d.channels){
    $("#chanTable").innerHTML = `<table><tr><th>channel</th><th>min</th><th>max</th><th>mean</th><th>NaN%</th></tr>`+
      d.channels.map(c=>`<tr><td class="mono">${c.channel}</td><td>${fmt(c.min)}</td><td>${fmt(c.max)}</td><td>${fmt(c.mean)}</td><td>${fmt(c.nan_pct,1)}</td></tr>`).join("")+`</table>`;
  }
  if(d.pm25_by_year){
    $("#yearTable").innerHTML = `<table><tr><th>ปี</th><th>เฉลี่ย</th><th>สูงสุด</th><th>n</th></tr>`+
      d.pm25_by_year.map(y=>`<tr><td>${y.year}</td><td><b>${fmt(y.mean)}</b></td><td>${fmt(y.max)}</td><td>${y.n}</td></tr>`).join("")+`</table>`;
  }
}

async function loadTraining(){
  const runs = await api("/api/runs");
  const cur  = await api("/api/run/latest");
  const ep = (cur.epochs)||[];
  const meta = cur.meta||{};
  $("#trainKpis").innerHTML = [
    [meta.status||"—", "สถานะ run ล่าสุด"],
    [meta.model||"—", "โมเดล"],
    [ep.length? ep[ep.length-1].epoch : "—", "epoch ปัจจุบัน"],
    [ep.length&&ep[ep.length-1].test!=null? fmt(ep[ep.length-1].test,4):"—", "test loss ล่าสุด"],
  ].map(([v,l])=>`<div class="kpi"><div class="v" style="font-size:18px">${v}</div><div class="l">${l}</div></div>`).join("");
  $("#trainTitle").textContent = cur.id ? `Loss curve — run ${cur.id} (${meta.status||""})` : "Loss curve";
  line($("#lossChart"), [
    {y: ep.map(e=>e.train), color:"#58a6ff"},
    {y: ep.map(e=>e.test ?? e.val), color:"#e3b341"},
  ], {h:240, log:true});

  if(runs && runs.length){
    $("#noRun").style.display="none";
    $("#runTable").innerHTML = `<table><tr><th>run id</th><th>โมเดล</th><th>สถานะ</th><th>เมื่อ</th></tr>`+
      runs.map(r=>`<tr><td class="mono">${r.id}</td><td>${r.model}</td><td>${r.status}</td><td>${r.ago}</td></tr>`).join("")+`</table>`;
  }
}

async function loadExperiments(){
  const e = await api("/api/experiments");
  if(!Array.isArray(e)) return;
  $("#expTable").innerHTML = `<table><tr><th>exp</th><th>model</th><th>domain</th><th>split</th><th>MAE↓</th><th>RMSE↓</th><th>tag</th><th>note</th></tr>`+
    e.map(r=>`<tr class="${r.tag||''}"><td>${r.exp||r._run||'-'}</td><td><b>${r.model||'-'}</b></td><td>${r.domain||'-'}</td><td>${r.split||'-'}</td>
      <td>${r.MAE!=null?fmt(r.MAE):(r.metrics&&r.metrics.MAE!=null?fmt(r.metrics.MAE):'—')}</td>
      <td>${r.RMSE!=null?fmt(r.RMSE):(r.metrics&&r.metrics.RMSE!=null?fmt(r.metrics.RMSE):'—')}</td>
      <td><span class="tag ${r.tag||''}">${r.tag||'-'}</span></td><td class="muted" style="font-size:11.5px">${r.note||''}</td></tr>`).join("")+`</table>`;
  // bar chart of MAE for rows that have it
  const withMae = e.filter(r=> (r.MAE!=null) || (r.metrics&&r.metrics.MAE!=null));
  bars($("#expChart"),
    withMae.map(r=>`${r.model}`),
    withMae.map(r=> r.MAE!=null? r.MAE : r.metrics.MAE),
    withMae.map(r=> r.tag==="ours"? "#3fb950" : "#58a6ff"),
    {h:200});
}

async function loadGallery(){
  const figs = await api("/api/figures");
  if(!Array.isArray(figs)) return;
  const now = Date.now()/1000;
  $("#gallery").innerHTML = figs.map(f=>{
    const isNew = (now - f.mtime) < 3600;
    return `<div class="fig"><img src="${f.url}?t=${Math.floor(f.mtime)}" loading="lazy">
      <div class="cap"><span>${f.name} ${isNew?'<span class="new">NEW</span>':''}</span><span>${f.size} · ${f.ago}</span></div></div>`;
  }).join("") || '<div class="muted">ยังไม่มีรูปใน figures/</div>';
}

// ---------- attribution explorer ----------
let _attribLoaded = false;
async function loadAttribMeta(){
  if(_attribLoaded) return; _attribLoaded = true;
  $("#aMsg").textContent = "loading model…";
  const m = await api("/api/attrib/meta");
  if(m.error){ $("#aMsg").textContent = "error: "+m.error; _attribLoaded=false; return; }
  // stations sorted by name
  const sta = m.stations.slice().sort((a,b)=>a.name.localeCompare(b.name));
  $("#aSta").innerHTML = sta.map(s=>`<option value="${s.idx}">${s.name} (${s.lat.toFixed(2)},${s.lon.toFixed(2)})</option>`).join("");
  // days sorted by mean_pm desc (วันวิกฤตขึ้นก่อน)
  const days = m.days.filter(d=>d.mean_pm!=null).sort((a,b)=>b.mean_pm-a.mean_pm);
  $("#aDay").innerHTML = days.map(d=>`<option value="${d.idx}">${d.date} · เฉลี่ย ${d.mean_pm}</option>`).join("");
  // default: Chiang Mai-ish station nearest, worst day
  $("#aDay").value = m.default_day;
  $("#aMsg").textContent = "พร้อม — เลือกแล้วกดคำนวณ";
  window._attribH = m.H; window._attribW = m.W;
}
async function runAttrib(){
  const sid = $("#aSta").value, did = $("#aDay").value;
  $("#aMsg").textContent = "กำลังคำนวณ…";
  const r = await api(`/api/attrib?station=${sid}&day=${did}`);
  if(r.error){ $("#aMsg").textContent = "error: "+r.error; return; }
  $("#aMsg").textContent = "";
  $("#aKpis").innerHTML = [
    [r.station.name.slice(0,22), "สถานี"],
    [r.date, "วันที่"],
    [r.pred_pm25, "พยากรณ์ PM2.5"],
    [r.obs_pm25!=null?r.obs_pm25:"—", "วัดจริง PM2.5"],
  ].map(([v,l])=>`<div class="kpi"><div class="v" style="font-size:18px">${v}</div><div class="l">${l}</div></div>`).join("");
  // sectors bar
  const order=["N","NE","E","SE","S","SW","W","NW"];
  const sm = {}; r.sectors.forEach(s=>sm[s.dir]=s.pct);
  bars($("#aSect"), order, order.map(d=>sm[d]||0), order.map(()=>"#56d4dd"), {h:220});
  $("#aNearFar").innerHTML = `📍 ใกล้ (&lt;60กม.): <b style="color:var(--amber)">${r.near_pct}%</b> &nbsp;·&nbsp; ไกล (&gt;60กม.): <b style="color:var(--cyan)">${r.far_pct}%</b>`;
  // top cells
  $("#aTop").innerHTML = `<table><tr><th>#</th><th>พิกัด (lat,lon)</th><th>ทิศจากสถานี</th><th>ระยะ (กม.)</th><th>สัดส่วน %</th></tr>`+
    r.top_cells.map((c,i)=>`<tr><td>${i+1}</td><td class="mono">${c.lat}, ${c.lon}</td><td>${c.dir}</td><td>${c.km}</td><td><b>${c.pct}%</b></td></tr>`).join("")+`</table>`;
  drawHeat($("#aHeat"), r.grid, r.extent, r.station);
}
function drawHeat(canvas, grid, extent, station){
  const ctx = canvas.getContext("2d");
  const H = grid.length, W = grid[0].length;
  const cw = canvas.width = canvas.clientWidth*2;
  const ch = canvas.height = 300*2;
  ctx.fillStyle="#0a0e13"; ctx.fillRect(0,0,cw,ch);
  let vmax=0; for(const row of grid) for(const v of row) if(v>vmax) vmax=v;
  vmax = vmax||1;
  const px=cw/W, py=ch/H;
  for(let i=0;i<H;i++)for(let j=0;j<W;j++){
    const v=grid[i][j]; if(v<=0) continue;
    const t=Math.min(1, Math.sqrt(v/vmax));   // sqrt → เห็น tail ชัด
    // hot colormap
    const r=Math.min(255,t*3*255), g=Math.min(255,Math.max(0,(t*3-1))*255), b=Math.min(255,Math.max(0,(t*3-2))*255);
    ctx.fillStyle=`rgb(${r|0},${g|0},${b|0})`;
    // origin lower → flip i
    ctx.fillRect(j*px, (H-1-i)*py, px+1, py+1);
  }
  // station marker
  const [lon0,lon1,lat0,lat1]=extent;
  const sx=( (station.lon-lon0)/(lon1-lon0) )*cw;
  const sy=( 1-(station.lat-lat0)/(lat1-lat0) )*ch;
  ctx.fillStyle="#56d4dd"; ctx.font="34px Segoe UI"; ctx.textAlign="center";
  ctx.fillText("★", sx, sy+12); ctx.textAlign="left";
}

// ---------- loop ----------
async function loadHealth(){
  const h = await api("/api/health");
  if(h.ok){ $("#hstatus").textContent="connected"; $("#hdot").style.background="var(--green)"; $("#hroot").textContent=h.root; }
  else { $("#hstatus").textContent="server down"; $("#hdot").style.background="var(--red)"; }
}
function tick(){ $("#clock").textContent = new Date().toLocaleTimeString(); }

async function refreshAll(force){
  tick(); loadHealth();
  loadPipeline(); loadTraining(); loadGallery();
  if(force){ // เคลียร์ cache ฝั่ง client ของ EDA โดยโหลดใหม่
    loadData(); loadExperiments();
  }
}

// initial
loadHealth(); loadPipeline(); loadData(); loadTraining(); loadExperiments(); loadGallery();
setInterval(tick, 1000);
setInterval(()=>{ loadPipeline(); loadTraining(); loadGallery(); loadHealth(); }, 3000); // live ทุก 3 วิ
setInterval(()=>{ loadData(); loadExperiments(); }, 30000); // EDA หนักกว่า → 30 วิ
