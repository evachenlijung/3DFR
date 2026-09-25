// Side-by-side mesh comparison: one camera, one set of controls, two scenes.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { OBJLoader } from "three/addons/OBJLoader.js";
import { MTLLoader } from "three/addons/MTLLoader.js";
import { mergeVertices } from "three/addons/BufferGeometryUtils.js";

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem("meshcmp." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("meshcmp." + k, JSON.stringify(v)); } catch { /* private mode */ } },
};

const state = {
  mode: store.get("mode", "unlit"), wire: store.get("wire", false), flat: store.get("flat", false),
  space: store.get("space", "camera"), layout: store.get("layout", "h"), align: store.get("align", "shared"),
  flip: store.get("flip", false),
};

// ---------------------------------------------------------------- renderer / camera
const canvas = $("c");
const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x1b2220, 1);
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 1e6);
camera.position.set(0, 0, 500);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

const texLoader = new THREE.TextureLoader();
const texCache = new Map();
function loadTexture(url) {
  if (!texCache.has(url)) {
    texCache.set(url, new Promise((resolve) => texLoader.load(url, (t) => {
      t.colorSpace = THREE.SRGBColorSpace;
      t.wrapS = t.wrapT = THREE.RepeatWrapping; // UDIM tiles keep their integer UV offset
      t.anisotropy = renderer.capabilities.getMaxAnisotropy();
      resolve(t);
    }, undefined, () => resolve(null))));
  }
  return texCache.get(url);
}

// ---------------------------------------------------------------- panes
function makePane(i) {
  const scene = new THREE.Scene();
  const ambient = new THREE.AmbientLight(0xffffff, 0.35);
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  scene.add(ambient, key, key.target);
  const outer = new THREE.Group(); // flip
  const inner = new THREE.Group(); // centring offset
  outer.add(inner);
  scene.add(outer);
  return { i, scene, ambient, key, outer, inner, model: null, id: null, loading: 0,
           el: $("pane" + i), sel: $("s" + i), q: $("q" + i), info: $("i" + i) };
}
const panes = [makePane(0), makePane(1)];

// ---------------------------------------------------------------- materials
const CLAY = 0xcfc9c2;
function materialFor(tex, textured) {
  const mode = (!textured && (state.mode === "unlit" || state.mode === "lit")) ? "clay" : state.mode;
  const common = { side: THREE.DoubleSide, flatShading: state.flat };
  if (mode === "unlit") return new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide });
  if (mode === "lit") return new THREE.MeshStandardMaterial({ map: tex, roughness: 0.75, metalness: 0, ...common });
  if (mode === "normal") return new THREE.MeshNormalMaterial(common);
  return new THREE.MeshStandardMaterial({ color: CLAY, roughness: 0.6, metalness: 0, ...common });
}
const wireMat = new THREE.MeshBasicMaterial({ color: 0x9fe0cf, wireframe: true, transparent: true, opacity: 0.28, depthWrite: false });

function applyMaterials(p) {
  if (!p.model) return;
  for (const part of p.model.parts) {
    const old = part.mesh.material;
    for (const m of [].concat(old)) m.dispose();
    const mats = part.textures.map((t) => materialFor(t, p.model.textured));
    part.mesh.material = mats.length === 1 ? mats[0] : mats;
    part.wire.visible = state.wire;
  }
  describe(p);
}

// ---------------------------------------------------------------- model building
function buildModel(parts, info) {
  const box = new THREE.Box3();
  for (const part of parts) {
    part.geometry.computeBoundingBox();
    box.union(part.geometry.boundingBox);
    part.mesh = new THREE.Mesh(part.geometry, new THREE.MeshBasicMaterial());
    part.wire = new THREE.Mesh(part.geometry, wireMat);
    part.wire.renderOrder = 1;
  }
  const textured = parts.some((p) => p.textures.some(Boolean));
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  return { parts, box, center: sphere.center, radius: sphere.radius, textured, ...info };
}

async function fetchServerModel(id, onProgress) {
  const res = await fetch("api/mesh?id=" + encodeURIComponent(id));
  if (!res.ok) throw new Error(await res.text());
  const total = +res.headers.get("Content-Length") || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    onProgress(total ? `下載中 ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB` : `下載中 ${(got / 1e6).toFixed(0)} MB`);
  }
  const buf = new Uint8Array(got);
  let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  const dv = new DataView(buf.buffer);
  if (String.fromCharCode(...buf.slice(0, 4)) !== "FMV1") throw new Error("伺服器回傳的格式不對");
  const hlen = dv.getUint32(4, true);
  const h = JSON.parse(new TextDecoder().decode(buf.slice(8, 8 + hlen)));
  let off = 8 + hlen;
  const take = (Type, n) => { const a = new Type(buf.buffer.slice(off, off + n * Type.BYTES_PER_ELEMENT)); off += n * Type.BYTES_PER_ELEMENT; return a; };
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(take(Float32Array, h.nv * 3), 3));
  g.setAttribute("normal", new THREE.BufferAttribute(take(Float32Array, h.nv * 3), 3));
  if (h.hasUV) g.setAttribute("uv", new THREE.BufferAttribute(take(Float32Array, h.nv * 2), 2));
  g.setIndex(new THREE.BufferAttribute(take(Uint32Array, h.nf * 3), 1));
  h.groups.forEach((gr, k) => g.addGroup(gr.start, gr.count, k));
  onProgress("讀取貼圖中…");
  const textures = await Promise.all(h.groups.map((gr) => gr.texture ? loadTexture(gr.texture) : null));
  return buildModel([{ geometry: g, textures }], { label: h.path, vertices: h.vertices, faces: h.nf });
}

async function loadDroppedFiles(files) {
  const urls = new Map();
  let objFile = null;
  for (const f of files) {
    urls.set(f.name.toLowerCase(), URL.createObjectURL(f));
    if (f.name.toLowerCase().endsWith(".obj")) objFile = f;
  }
  if (!objFile) throw new Error("拖進來的檔案裡沒有 .obj");
  const manager = new THREE.LoadingManager();
  manager.setURLModifier((url) => urls.get(decodeURIComponent(url.split(/[\\/]/).pop()).toLowerCase()) || url);
  const text = await objFile.text();
  const obj = new OBJLoader(manager);
  const mtlName = (text.match(/^mtllib\s+(.+)$/m) || [])[1];
  if (mtlName && urls.has(mtlName.trim().split(/[\\/]/).pop().toLowerCase())) {
    const mtlText = await (await fetch(urls.get(mtlName.trim().split(/[\\/]/).pop().toLowerCase()))).text();
    const mats = new MTLLoader(manager).parse(mtlText, "");
    mats.preload();
    obj.setMaterials(mats);
  }
  const group = obj.parse(text);
  const parts = [];
  const pending = [];
  group.traverse((m) => {
    if (!m.isMesh) return;
    const g = m.geometry;
    g.deleteAttribute("normal");
    const merged = mergeVertices(g);
    merged.computeVertexNormals();
    const mats = [].concat(m.material);
    const textures = mats.map((mt) => mt.map || null);
    for (const t of textures) if (t) { t.colorSpace = THREE.SRGBColorSpace; t.wrapS = t.wrapT = THREE.RepeatWrapping; }
    if (!merged.groups.length) merged.addGroup(0, merged.index.count, 0);
    parts.push({ geometry: merged, textures });
    for (const t of textures) if (t && !t.image) pending.push(new Promise((r) => { const iv = setInterval(() => { if (t.image) { clearInterval(iv); r(); } }, 50); setTimeout(() => { clearInterval(iv); r(); }, 20000); }));
  });
  await Promise.all(pending);
  const faces = parts.reduce((s, p) => s + p.geometry.index.count / 3, 0);
  const vertices = parts.reduce((s, p) => s + p.geometry.attributes.position.count, 0);
  return buildModel(parts, { label: objFile.name + "（拖放）", vertices, faces });
}

function setModel(p, model) {
  if (p.model) for (const part of p.model.parts) { p.inner.remove(part.mesh, part.wire); }
  p.model = model;
  if (model) for (const part of model.parts) p.inner.add(part.mesh, part.wire);
  applyMaterials(p);
  arrange();
}

async function loadInto(p, source) {
  const token = ++p.loading;
  p.info.classList.remove("warn");
  p.info.textContent = typeof source === "string" ? "伺服器轉換中…（第一次開大網格要幾秒）" : "讀取拖放的檔案中…";
  try {
    const model = typeof source === "string"
      ? await fetchServerModel(source, (msg) => { if (token === p.loading) p.info.textContent = msg; })
      : await loadDroppedFiles(source);
    if (token !== p.loading) return;
    const first = !panes.some((q) => q.model);
    setModel(p, model);
    if (first) fitView();
  } catch (e) {
    if (token !== p.loading) return;
    p.info.classList.add("warn");
    p.info.textContent = "載入失敗：" + e.message;
  }
}

function describe(p) {
  if (!p.model) return;
  const m = p.model;
  const note = (!m.textured && (state.mode === "unlit" || state.mode === "lit")) ? "　⚠ 沒有貼圖，改用無貼圖＋光照" : "";
  p.info.classList.toggle("warn", Boolean(note));
  p.info.textContent = `${m.label}\n${m.vertices.toLocaleString()} 頂點 · ${m.faces.toLocaleString()} 三角形 · ${m.textured ? "有貼圖" : "無貼圖"}${note}`;
  p.info.style.whiteSpace = "pre-line";
}

// ---------------------------------------------------------------- placement / view
function reference() { return panes[0].model || panes[1].model; }
function arrange() {
  const ref = reference();
  for (const p of panes) {
    p.outer.rotation.set(state.flip ? Math.PI : 0, 0, 0);
    const c = state.align === "own" && p.model ? p.model.center : ref ? ref.center : new THREE.Vector3();
    p.inner.position.copy(c).multiplyScalar(-1);
  }
}
function fitView() {
  const r = Math.max(...panes.filter((p) => p.model).map((p) => p.model.radius), 1);
  const d = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.05;
  camera.near = r / 200; camera.far = r * 200; camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, d);
  camera.up.set(0, 1, 0);
  controls.update();
}

// ---------------------------------------------------------------- lights
const az = $("az"), el = $("el"), inten = $("int"), amb = $("amb");
const tmp = new THREE.Vector3();
function updateLights() {
  const a = THREE.MathUtils.degToRad(+az.value), e = THREE.MathUtils.degToRad(+el.value);
  tmp.set(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e));
  if (state.space === "camera") tmp.applyQuaternion(camera.quaternion);
  const R = camera.position.distanceTo(controls.target) * 2 + 1;
  for (const p of panes) {
    p.key.position.copy(controls.target).addScaledVector(tmp, R);
    p.key.target.position.copy(controls.target);
    p.key.intensity = +inten.value / 100;
    p.ambient.intensity = +amb.value / 100;
  }
  $("az-v").textContent = az.value + "°";
  $("el-v").textContent = el.value + "°";
}

// ---------------------------------------------------------------- layout + render loop
function rects() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (state.layout === "v") {
    const h0 = Math.floor(h / 2);
    return [{ x: 0, y: 0, w, h: h0 }, { x: 0, y: h0, w, h: h - h0 }];
  }
  const w0 = Math.floor(w / 2);
  return [{ x: 0, y: 0, w: w0, h }, { x: w0, y: 0, w: w - w0, h }];
}
function placeOverlays() {
  const rs = rects();
  panes.forEach((p, i) => Object.assign(p.el.style, { left: rs[i].x + "px", top: rs[i].y + "px", width: rs[i].w + "px", height: rs[i].h + "px" }));
  const d = $("divider");
  Object.assign(d.style, state.layout === "v"
    ? { left: "0", right: "0", top: rs[1].y + "px", height: "1px", width: "auto", bottom: "auto" }
    : { top: "0", bottom: "0", left: rs[1].x + "px", width: "1px", height: "auto", right: "auto" });
}
function resize() {
  renderer.setSize(stage.clientWidth, stage.clientHeight, false);
  placeOverlays();
}
new ResizeObserver(resize).observe(stage);

function renderAll() {
  const H = stage.clientHeight;
  renderer.setScissorTest(true);
  rects().forEach((r, i) => {
    camera.aspect = r.w / Math.max(r.h, 1);
    camera.updateProjectionMatrix();
    const y = H - r.y - r.h; // WebGL origin is bottom-left
    renderer.setViewport(r.x, y, r.w, r.h);
    renderer.setScissor(r.x, y, r.w, r.h);
    renderer.render(panes[i].scene, camera);
  });
}
function loop() {
  controls.update();
  updateLights();
  renderAll();
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- model list
let models = [];
function fillSelect(p) {
  const words = p.q.value.toLowerCase().split(/\s+/).filter(Boolean);
  const keep = models.filter((m) => words.every((w) => m.rel.toLowerCase().includes(w)));
  const current = p.id;
  p.sel.innerHTML = "";
  const none = document.createElement("option");
  none.value = ""; none.textContent = keep.length ? `（${keep.length} 個模型，選一個）` : "（沒有符合的模型）";
  p.sel.appendChild(none);
  const byGroup = new Map();
  for (const m of keep) {
    const parts = m.rel.split("/");
    const g = (models.rootsCount > 1 ? `[${m.id.split(":")[0]}] ` : "") + (parts.length > 1 ? parts[0] : "（根目錄）");
    if (!byGroup.has(g)) byGroup.set(g, []);
    byGroup.get(g).push(m);
  }
  for (const [g, list] of byGroup) {
    const og = document.createElement("optgroup");
    og.label = g;
    for (const m of list) {
      const o = document.createElement("option");
      o.value = m.id;
      o.textContent = `${m.rel}   ${(m.size / 1e6).toFixed(0)} MB${m.textured ? " · 貼圖" : ""}`;
      o.title = m.root + "/" + m.rel;
      og.appendChild(o);
    }
    p.sel.appendChild(og);
  }
  if (current && keep.some((m) => m.id === current)) p.sel.value = current;
}
async function loadList(refresh) {
  const res = await fetch("api/models" + (refresh ? "?refresh=1" : ""));
  const data = await res.json();
  models = data.models;
  models.rootsCount = data.roots.length;
  $("lib").textContent = `${models.length} 個 OBJ · ${data.roots.join("　")}`;
  $("lib").title = data.roots.join("\n");
  panes.forEach(fillSelect);
}

// ---------------------------------------------------------------- UI wiring
function pressed(groupId, attr, value) {
  for (const b of $(groupId).querySelectorAll("button")) b.setAttribute("aria-pressed", String(b.dataset[attr] === value));
}
function setMode(mode) { state.mode = mode; store.set("mode", mode); pressed("shading", "mode", mode); panes.forEach(applyMaterials); }
function toggle(name, btnId) {
  state[name] = !state[name]; store.set(name, state[name]);
  $(btnId).setAttribute("aria-pressed", String(state[name]));
  panes.forEach(applyMaterials);
}
function swap() {
  const [a, b] = panes;
  const ma = a.model, mb = b.model, ia = a.id, ib = b.id;
  setModel(a, null); setModel(b, null);
  a.id = ib; b.id = ia;
  setModel(a, mb); setModel(b, ma);
  a.sel.value = ib || ""; b.sel.value = ia || "";
  if (!mb) a.info.textContent = "選一個模型，或把 OBJ 拖進來";
  if (!ma) b.info.textContent = "選一個模型，或把 OBJ 拖進來";
  store.set("sel0", a.id); store.set("sel1", b.id);
}
function screenshot() {
  renderAll();
  canvas.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `compare_${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
}

for (const b of $("shading").querySelectorAll("button")) b.addEventListener("click", () => setMode(b.dataset.mode));
$("wire").addEventListener("click", () => toggle("wire", "wire"));
$("flat").addEventListener("click", () => toggle("flat", "flat"));
for (const b of $("lightspace").querySelectorAll("button")) b.addEventListener("click", () => {
  state.space = b.dataset.space; store.set("space", state.space); pressed("lightspace", "space", state.space);
});
for (const b of $("layout").querySelectorAll("button")) b.addEventListener("click", () => {
  state.layout = b.dataset.layout; store.set("layout", state.layout); pressed("layout", "layout", state.layout); placeOverlays();
});
for (const b of $("align").querySelectorAll("button")) b.addEventListener("click", () => {
  state.align = b.dataset.align; store.set("align", state.align); pressed("align", "align", state.align); arrange();
});
$("flipx").addEventListener("click", () => { state.flip = !state.flip; store.set("flip", state.flip); $("flipx").setAttribute("aria-pressed", String(state.flip)); arrange(); });
$("swap").addEventListener("click", swap);
$("reset").addEventListener("click", fitView);
$("shot").addEventListener("click", screenshot);
$("refresh").addEventListener("click", () => loadList(true));
for (const s of [az, el, inten, amb]) s.addEventListener("input", () => store.set("light", [az.value, el.value, inten.value, amb.value]));

panes.forEach((p) => {
  p.sel.addEventListener("change", () => {
    p.id = p.sel.value || null; store.set("sel" + p.i, p.id);
    if (p.id) loadInto(p, p.id); else setModel(p, null);
  });
  p.q.addEventListener("input", () => fillSelect(p));
});

// drag & drop onto either half
function paneAt(ev) {
  const r = stage.getBoundingClientRect();
  return state.layout === "v" ? (ev.clientY - r.top > r.height / 2 ? 1 : 0) : (ev.clientX - r.left > r.width / 2 ? 1 : 0);
}
stage.addEventListener("dragover", (ev) => { ev.preventDefault(); const k = paneAt(ev); panes.forEach((p) => p.el.classList.toggle("drop", p.i === k)); });
stage.addEventListener("dragleave", (ev) => { if (!stage.contains(ev.relatedTarget)) panes.forEach((p) => p.el.classList.remove("drop")); });
stage.addEventListener("drop", (ev) => {
  ev.preventDefault();
  panes.forEach((p) => p.el.classList.remove("drop"));
  const p = panes[paneAt(ev)];
  p.id = null; p.sel.value = "";
  loadInto(p, [...ev.dataTransfer.files]);
});

window.addEventListener("keydown", (ev) => {
  if (ev.target.closest("input, select, textarea") || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ev.key.toLowerCase();
  if ("1234".includes(k) && k) setMode(["unlit", "lit", "clay", "normal"][+k - 1]);
  else if (k === "w") toggle("wire", "wire");
  else if (k === "f") toggle("flat", "flat");
  else if (k === "r") fitView();
  else if (k === "s") swap();
});

// ---------------------------------------------------------------- start
const light = store.get("light", null);
if (light) [az.value, el.value, inten.value, amb.value] = light;
pressed("shading", "mode", state.mode);
pressed("lightspace", "space", state.space);
pressed("layout", "layout", state.layout);
pressed("align", "align", state.align);
$("wire").setAttribute("aria-pressed", String(state.wire));
$("flat").setAttribute("aria-pressed", String(state.flat));
$("flipx").setAttribute("aria-pressed", String(state.flip));
resize();
requestAnimationFrame(loop);
loadList(false).then(() => {
  panes.forEach((p) => {
    const id = store.get("sel" + p.i, null);
    if (id && models.some((m) => m.id === id)) { p.id = id; p.sel.value = id; loadInto(p, id); }
  });
}).catch(() => { $("lib").textContent = "連不到 serve.py，請用 python viewer/serve.py 開啟（也可以直接把 OBJ 拖進來）"; });
