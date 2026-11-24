"""
damage_map.py - local web app: building damage predictions + tweets on a map.

    python damage_map.py              # opens http://localhost:8780 in your browser

Needs map_data/ from map_export.py. Tweets: every CSV in tweets/ is loaded
(mock_tweets.csv is ignored once any real tweet CSV is present). Columns are found by
common names - edit TWEET_COLUMNS below if yours differ.
"""

import argparse
import csv
import io
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

DATA = Path("map_data")
TWEETS = Path("tweets")

# Possible column names (lower-case) for each tweet field - first match wins.
TWEET_COLUMNS = {
    "lat": ["latitude", "lat", "y"],
    "lon": ["longitude", "lon", "lng", "long", "x"],
    "label": ["label", "prediction", "predicted", "class", "category", "critical"],
    "text": ["text", "tweet", "tweet_text", "full_text", "content", "message"],
    "user": ["user", "username", "user_name", "screen_name", "handle", "author"],
    "time": ["created_at", "timestamp", "time", "date", "datetime"],
    "hurricane": ["hurricane", "event", "disaster", "storm"],
    "is_mock": ["is_mock"],
}


def find_col(fields, names):
    low = {f.lower().strip(): f for f in fields}
    return next((low[n] for n in names if n in low), None)


def norm_label(v):
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "critical", "crit"):
        return "Critical"
    if s in ("0", "false", "no", "noncritical", "non-critical", "non_critical", "not critical"):
        return "NonCritical"
    return "Critical" if s.startswith("crit") else "NonCritical"


def load_tweets(images, hurricanes):
    files = sorted(TWEETS.glob("*.csv")) if TWEETS.is_dir() else []
    real = [f for f in files if f.name != "mock_tweets.csv"]
    files = real or files
    centers = [(sum(c[1] for c in im["corners"]) / 4, sum(c[0] for c in im["corners"]) / 4, im["h"])
               for im in images if im.get("corners")]
    tweets, mock = [], False
    for f in files:
        with open(f, newline="", encoding="utf-8-sig", errors="replace") as fh:
            rd = csv.DictReader(fh)
            cols = {k: find_col(rd.fieldnames or [], v) for k, v in TWEET_COLUMNS.items()}
            if not cols["lat"] or not cols["lon"]:
                print(f"[tweets] {f.name}: no latitude/longitude column found - skipped "
                      f"(columns: {rd.fieldnames})")
                continue
            for r in rd:
                try:
                    lat, lon = float(r[cols["lat"]]), float(r[cols["lon"]])
                except (TypeError, ValueError):
                    continue
                h = None
                if cols["hurricane"] and r.get(cols["hurricane"]):
                    hv = r[cols["hurricane"]].lower()
                    h = next((i for i, n in enumerate(hurricanes) if n.split("-")[-1] in hv), None)
                if h is None and centers:          # nearest imagery tile, within ~50 km
                    d, hh = min(((lat - a) ** 2 + ((lon - b) * math.cos(math.radians(lat))) ** 2, hh)
                                for a, b, hh in centers)
                    h = hh if math.sqrt(d) * 111 < 50 else None
                is_mock = bool(cols["is_mock"]) and str(r.get(cols["is_mock"], "")).lower() in ("true", "1")
                mock |= is_mock
                tweets.append({
                    "lat": lat, "lon": lon, "h": -1 if h is None else h,
                    "label": norm_label(r[cols["label"]]) if cols["label"] else "NonCritical",
                    "text": (r.get(cols["text"]) or "") if cols["text"] else "",
                    "user": (r.get(cols["user"]) or "") if cols["user"] else "",
                    "time": (r.get(cols["time"]) or "") if cols["time"] else "",
                })
        print(f"[tweets] loaded {f.name}")
    return tweets, mock


class App:
    def __init__(self):
        if not (DATA / "meta.json").is_file():
            raise SystemExit("map_data/ not found - run `python map_export.py` first.")
        self.meta = json.loads((DATA / "meta.json").read_text())
        self.images = json.loads((DATA / "images.json").read_text())
        self.polys = json.loads((DATA / "polygons.json").read_text())
        tweets, mock = load_tweets(self.images, self.meta["hurricanes"])
        self.payload = json.dumps({
            "meta": {**self.meta, "mock_tweets": mock},
            "buildings": json.loads((DATA / "buildings.json").read_text()),
            "images": self.images,
            "tweets": tweets,
        }, separators=(",", ":")).encode()
        self.tile_cache = {}
        print(f"{self.meta['n_buildings']} buildings, {len(self.images)} images, {len(tweets)} tweets"
              f"{' (MOCK)' if mock else ''}, model: {self.meta.get('model') or 'none (ground truth)'}")

    def tile(self, name):
        if name not in self.tile_cache:
            from PIL import Image
            src = Path(self.meta["xbd_images"]) / f"{name}.png"
            im = Image.open(src).convert("RGB")
            im.thumbnail((768, 768))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=85)
            if len(self.tile_cache) > 200:
                self.tile_cache.clear()
            self.tile_cache[name] = buf.getvalue()
        return self.tile_cache[name]


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = unquote(self.path.split("?")[0])
            try:
                if p in ("/", "/index.html"):
                    return self.send(PAGE.encode(), "text/html; charset=utf-8")
                if p == "/data.json":
                    return self.send(app.payload, "application/json")
                if p.startswith("/poly/"):
                    k = p.split("/")[-1]
                    return self.send(json.dumps(app.polys.get(k, [])).encode(), "application/json")
                if p.startswith("/tile/"):
                    name = Path(p).name
                    if not any(name in (im["name"], im["pre"]) for im in app.images):
                        return self.send(b"unknown tile", "text/plain", 404)
                    return self.send(app.tile(name), "image/jpeg")
            except Exception as e:
                return self.send(str(e).encode(), "text/plain", 500)
            self.send(b"not found", "text/plain", 404)
    return H


PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Hurricane Damage Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
 *{box-sizing:border-box} body{margin:0;font:13px/1.4 system-ui,Segoe UI,Arial,sans-serif;display:flex;height:100vh;color:#1d1d1f}
 #side{width:330px;overflow-y:auto;padding:12px 14px;background:#f6f6f4;border-right:1px solid #ddd}
 #map{flex:1} h1{font-size:16px;margin:0 0 2px} .sub{color:#666;font-size:12px;margin-bottom:8px}
 h2{font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#555;margin:14px 0 6px}
 label{display:block;cursor:pointer;margin:2px 0} select,input[type=range]{width:100%}
 .sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
 #stats{display:grid;grid-template-columns:1fr 1fr;gap:6px}
 .st{background:#fff;border:1px solid #e2e2e2;border-radius:6px;padding:6px 8px}
 .st b{display:block;font-size:17px} .st span{color:#666;font-size:11px}
 #note{font-size:11px;color:#7a5b00;background:#fff6d6;border:1px solid #f0dc9a;border-radius:6px;padding:6px 8px;margin-top:8px}
 #mockbanner{position:absolute;z-index:1000;top:10px;left:50%;transform:translateX(-50%);background:#b00020;color:#fff;
   padding:5px 14px;border-radius:14px;font-weight:600;font-size:12px;display:none;box-shadow:0 1px 4px #0004}
 .feed{background:#fff;border:1px solid #e6e6e6;border-left:3px solid #c62828;border-radius:5px;padding:6px 8px;margin-bottom:6px;cursor:pointer}
 .feed:hover{background:#fff4f4} .feed .m{color:#777;font-size:11px}
 .tcard{max-width:220px;white-space:normal;font-size:11px;border-left:3px solid #c62828}
 .popt{font-weight:600} .imgbtn{margin-top:6px;padding:3px 8px;cursor:pointer}
</style></head><body>
<div id="side">
 <h1>Hurricane Damage Map</h1>
 <div class="sub" id="modelinfo">loading...</div>

 <h2>Damage source</h2>
 <select id="mode">
  <option value="val">Model predictions: held-out images only</option>
  <option value="all">Model predictions: all images</option>
  <option value="gt">Ground truth labels</option>
 </select>
 <div id="note"></div>

 <h2>Hurricanes</h2><div id="hurs"></div>
 <h2>Damage levels</h2><div id="cls"></div>
 <div id="confbox"><h2>Min model confidence: <span id="confv">0.00</span></h2>
  <input id="conf" type="range" min="0" max="0.95" step="0.05" value="0"></div>

 <h2>Layers</h2>
 <label><input type="checkbox" id="lb" checked> Buildings</label>
 <label><input type="checkbox" id="lh"> Damage heatmap</label>
 <label><input type="checkbox" id="lt" checked> Tweets</label>
 <label><input type="checkbox" id="ltc" checked> Show critical tweet text on map (zoom in)</label>
 <label><input type="checkbox" id="lth"> Critical tweet heatmap</label>
 <h2>Tweets</h2>
 <label><input type="checkbox" class="tl" value="Critical" checked> <span class="sw" style="background:#d50000"></span>Critical</label>
 <label><input type="checkbox" class="tl" value="NonCritical" checked> <span class="sw" style="background:#9e9e9e"></span>Non-critical</label>

 <h2>Summary (current filters)</h2><div id="stats"></div>
 <h2>Satellite imagery (xBD tiles)</h2>
 <label><input type="checkbox" id="lall"> Show all tiles in view (zoom in)</label>
 <label><input type="radio" name="when" value="post" checked> After storm</label>
 <label><input type="radio" name="when" value="pre"> Before storm</label>
 <label><input type="radio" name="when" value="none"> Hide imagery</label>
 <label>Opacity <input type="range" id="opac" min="0.2" max="1" step="0.1" value="0.9"></label>
 <div class="sub" id="imgname"></div>
 <h2>Critical tweet feed</h2><div id="feed"></div>
</div>
<div id="map"></div><div id="mockbanner">MOCK TWEETS - placeholder data, not real reports</div>
<script>
const CLS=["No damage","Minor","Major","Destroyed"], CCOL=["#2e7d32","#f9a825","#ef6c00","#c62828"];
const map=L.map('map',{preferCanvas:true,zoomControl:true}).setView([30,-85],5);
const sat=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
 {maxZoom:20,maxNativeZoom:19,attribution:'Imagery &copy; Esri'}).addTo(map);
const osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:20,maxNativeZoom:19,attribution:'&copy; OpenStreetMap'});
L.control.layers({"Satellite (today)":sat,"Streets":osm},null,{position:'topright'}).addTo(map);
const rend=L.canvas({padding:0.5});
let D, bLayer=L.layerGroup().addTo(map), tLayer=L.layerGroup().addTo(map), cardLayer=L.layerGroup().addTo(map),
    heat=null, theat=null, polyLayer=L.layerGroup().addTo(map), ovs={}, curImg=null, fitted=false;
const MAXTILES=15, TILEZOOM=14;
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const nice=h=>D.meta.hurricanes[h]?D.meta.hurricanes[h].replace('hurricane-','').replace(/^\w/,c=>c.toUpperCase()):'Unknown';

fetch('/data.json').then(r=>r.json()).then(d=>{D=d; init(); render();});

function init(){
 const m=D.meta;
 $('modelinfo').textContent=(m.model?`Classifier: ${m.model}`:'No classifier - ground truth only')+` · ${m.n_buildings.toLocaleString()} buildings`;
 if(!m.model){$('mode').value='gt';}
 if(m.mock_tweets) $('mockbanner').style.display='block';
 $('hurs').innerHTML=m.hurricanes.map((h,i)=>`<label><input type="checkbox" class="hu" value="${i}" checked> ${nice(i)}</label>`).join('');
 $('cls').innerHTML=CLS.map((c,i)=>`<label><input type="checkbox" class="cl" value="${i}" checked> <span class="sw" style="background:${CCOL[i]}"></span>${c}</label>`).join('');
 document.querySelectorAll('input,select').forEach(e=>e.addEventListener('change',()=>{if(e.name==='when'||e.id==='opac'||e.id==='lall')return refreshTiles();render();}));
 $('conf').addEventListener('input',()=>{$('confv').textContent=(+$('conf').value).toFixed(2)});
 $('opac').addEventListener('input',()=>Object.values(ovs).forEach(o=>o.setOpacity(+$('opac').value)));
 map.on('zoomend',renderCards);
}
const checked=cls=>new Set([...document.querySelectorAll('.'+cls+':checked')].map(e=>e.value));

function render(){
 const B=D.buildings, mode=$('mode').value, H=checked('hu'), C=checked('cl'), minc=+$('conf').value;
 const useGT=mode==='gt';
 $('confbox').style.display=useGT?'none':'block';
 $('note').innerHTML= mode==='val'?'Honest view: only buildings from images the model <b>never trained on</b>.'
   : mode==='all'?'&#9888; Includes the 80% of images the model <b>trained on</b> - looks better than real performance.'
   :'Actual labels from the xBD dataset (what the model is graded against).';
 bLayer.clearLayers(); if(heat){map.removeLayer(heat);heat=null;}
 const counts=[0,0,0,0], hp=[], bounds=[];
 for(let i=0;i<B.lat.length;i++){
  if(mode==='val'&&!B.val[i]) continue;
  if(!H.has(String(B.h[i]))) continue;
  const c=useGT?B.true[i]:B.pred[i];
  if(!C.has(String(c))) continue;
  if(!useGT&&B.conf[i]<minc) continue;
  counts[c]++; bounds.push([B.lat[i],B.lon[i]]);
  if($('lh').checked) hp.push([B.lat[i],B.lon[i],useGT?[0.05,0.35,0.7,1][c]:Math.max(B.psev[i],0.05)]);
  if($('lb').checked){
   const mk=L.circleMarker([B.lat[i],B.lon[i]],{renderer:rend,radius:4,weight:0.6,color:'#fff',fillColor:CCOL[c],fillOpacity:0.9});
   mk.on('click',()=>{
    const correct=B.pred[i]===B.true[i];
    mk.bindPopup(`<div class="popt">${useGT?'Label':'Predicted'}: ${CLS[c]}</div>`+
     (useGT?'':`Confidence ${(B.conf[i]*100).toFixed(0)}% · P(major+destroyed) ${(B.psev[i]*100).toFixed(0)}%<br>True label: ${CLS[B.true[i]]} ${correct?'&#10004;':'&#10008;'}<br>`)+
     `${nice(B.h[i])} · ${B.val[i]?'held-out image':'training image'}<br>`+
     `<button class="imgbtn" onclick="pickImage(${B.img[i]})">Show before/after satellite tile</button>`).openPopup();
   });
   bLayer.addLayer(mk);
  }
 }
 if(hp.length) heat=L.heatLayer(hp,{radius:18,blur:22,maxZoom:16,minOpacity:0.3}).addTo(map);
 renderTweets(H);
 const tot=counts.reduce((a,b)=>a+b,0);
 $('stats').innerHTML=[['Buildings shown',tot.toLocaleString()],...CLS.map((c,i)=>[c,counts[i].toLocaleString()]),
   ['Critical tweets',D._crit],['All tweets shown',D._tw]].map(([a,b])=>`<div class="st"><b>${b}</b><span>${a}</span></div>`).join('');
 if(!fitted&&bounds.length){map.fitBounds(bounds,{maxZoom:14});fitted=true;}
}

function renderTweets(H){
 tLayer.clearLayers(); if(theat){map.removeLayer(theat);theat=null;}
 const L_=checked('tl'), crit=[], hp=[]; let shown=0;
 D._vis=[];
 D.tweets.forEach((t,i)=>{
  if(!L_.has(t.label)) return;
  if(t.h>=0&&!H.has(String(t.h))) return;
  shown++; D._vis.push(i);
  const isC=t.label==='Critical';
  if(isC){crit.push(t); hp.push([t.lat,t.lon,1]);}
  if($('lt').checked){
   const mk=L.circleMarker([t.lat,t.lon],{renderer:rend,radius:isC?7:5,weight:2,color:'#fff',fillColor:isC?'#d50000':'#9e9e9e',fillOpacity:0.95});
   mk.bindTooltip(tweetHtml(t),{direction:'top',className:'tcard',opacity:0.97});
   tLayer.addLayer(mk); t._mk=mk;
  }
 });
 if($('lth').checked&&hp.length) theat=L.heatLayer(hp,{radius:30,blur:25,gradient:{0.3:'#ffcdd2',0.6:'#e53935',1:'#7f0000'}}).addTo(map);
 crit.sort((a,b)=>(b.time||'').localeCompare(a.time||''));
 D._crit=crit.length; D._tw=shown;
 $('feed').innerHTML=crit.length?crit.slice(0,150).map(t=>`<div class="feed" data-i="${D.tweets.indexOf(t)}">${esc(t.text)}<div class="m">${esc(t.user)} · ${esc(t.time)} · ${nice(t.h)}</div></div>`).join('')
   :'<i>No critical tweets for these filters.</i>';
 document.querySelectorAll('.feed').forEach(e=>e.onclick=()=>{const t=D.tweets[+e.dataset.i];map.flyTo([t.lat,t.lon],17);
   setTimeout(()=>t._mk&&t._mk.openTooltip(),1200);});
 renderCards();
}
function tweetHtml(t){return `<b>${t.label==='Critical'?'&#9888; CRITICAL':'Non-critical'}</b><br>${esc(t.text)}<br><span style="color:#777">${esc(t.user)} · ${esc(t.time)}</span>`;}
function renderCards(){
 cardLayer.clearLayers();
 if(!D||!$('ltc').checked||!$('lt').checked||map.getZoom()<14) return;
 const vb=map.getBounds().pad(0.1);
 D._vis.map(i=>D.tweets[i]).filter(t=>t.label==='Critical'&&vb.contains([t.lat,t.lon])).slice(0,60).forEach(t=>{
  cardLayer.addLayer(L.tooltip({permanent:true,direction:'right',className:'tcard',offset:[8,0]}).setLatLng([t.lat,t.lon]).setContent(esc(t.text)));
 });
}
map.on('moveend',()=>{renderCards();refreshTiles();});

function pickImage(k){curImg=k; refreshTiles();
 polyLayer.clearLayers();
 fetch('/poly/'+k).then(r=>r.json()).then(ps=>ps.forEach(p=>polyLayer.addLayer(L.polygon(p.map(([lo,la])=>[la,lo]),{color:'#00e5ff',weight:1,fill:false}))));}
function tb(im){ if(!im._b&&im.corners){const la=im.corners.map(c=>c[1]),lo=im.corners.map(c=>c[0]);
   im._b={s:Math.min(...la),n:Math.max(...la),w:Math.min(...lo),e:Math.max(...lo)};} return im._b;}
function refreshTiles(){
 if(!D) return;
 const w=document.querySelector('input[name=when]:checked').value, want=new Set();
 let msg='';
 if(w!=='none'){
  if(curImg!==null) want.add(curImg);
  if($('lall').checked){
   if(map.getZoom()<TILEZOOM) msg='Zoom in further to load tiles in view.';
   else{
    const mb=map.getBounds(), S=mb.getSouth(),N=mb.getNorth(),W=mb.getWest(),E=mb.getEast(), cy=(S+N)/2, cx=(W+E)/2;
    D.images.map((im,k)=>[k,tb(im)]).filter(([k,b])=>b&&b.s<N&&b.n>S&&b.w<E&&b.e>W)
     .map(([k,b])=>[k,((b.s+b.n)/2-cy)**2+((b.w+b.e)/2-cx)**2]).sort((a,b)=>a[1]-b[1])
     .slice(0,MAXTILES).forEach(([k])=>want.add(k));
    msg=want.size?`Showing ${want.size} tile(s) in view (max ${MAXTILES}).`:'No xBD tiles in this view.';
   }
  }
 }
 const keys=new Set([...want].map(k=>k+'_'+w));
 for(const key of Object.keys(ovs)) if(!keys.has(key)){map.removeLayer(ovs[key]);delete ovs[key];}
 for(const k of want){const key=k+'_'+w, im=D.images[k], b=tb(im); if(ovs[key]||!b) continue;
  ovs[key]=L.imageOverlay('/tile/'+(w==='pre'?im.pre:im.name),[[b.s,b.w],[b.n,b.e]],{opacity:+$('opac').value}).addTo(map);
  ovs[key].bringToBack();}
 $('imgname').textContent=(curImg!==null?'Inspecting: '+D.images[curImg].name+'. ':'')+msg;
}
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    app = App()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    url = f"http://localhost:{args.port}"
    print(f"Damage map running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
