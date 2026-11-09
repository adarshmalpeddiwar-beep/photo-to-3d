import * as THREE from "three";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const STAGES = [
  ["segment", "Find object"],
  ["reconstruct", "Build shape"],
  ["scale", "Scale"],
  ["clean", "Clean mesh"],
  ["check", "Print checks"],
  ["export", "Export files"],
];
const STAGE_ORDER = ["queued", ...STAGES.map(([key]) => key), "done"];
const STATUS_LABELS = {
  draft: "Draft", queued: "Queued", running: "Running", done: "Done", failed: "Failed",
};
const TOKEN_KEY = "photo3d.token";
const POLL_MS = 2000;

const $ = (sel) => document.querySelector(sel);
const els = {
  form: $("#new-job-form"),
  submit: $("#submit"),
  formError: $("#form-error"),
  photos: $("#photos"),
  name: $("#name"),
  reference: $("#reference"),
  list: $("#job-list"),
  listEmpty: $("#jobs-empty"),
  emptyState: $("#empty-state"),
  detail: $("#detail"),
  title: $("#detail-title"),
  status: $("#detail-status"),
  meta: $("#detail-meta"),
  stages: $("#stages"),
  errorBox: $("#error-box"),
  notes: $("#notes"),
  resultCard: $("#result-card"),
  maskImg: $("#mask-img"),
  viewer: $("#viewer"),
  dlStl: $("#dl-stl"),
  dl3mf: $("#dl-3mf"),
  checksBody: $("#checks-body"),
  dims: $("#dims"),
};

const view = {
  jobs: new Map(),
  selected: null,
  renderedFor: null,
  objectUrls: [],
  viewer: null,
};

// ----- HTTP helpers ----------------------------------------------------------

function authHeaders(headers = {}) {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { ...headers, Authorization: `Bearer ${token}` } : { ...headers };
}

async function api(path, opts = {}) {
  const { method = "GET", body = null, headers = {}, raw = false, retried = false } = opts;
  const res = await fetch(path, { method, body, headers: authHeaders(headers) });
  if (res.status === 401 && !retried) {
    const token = window.prompt("This instance needs an access token:");
    if (!token) throw new Error("An access token is required.");
    localStorage.setItem(TOKEN_KEY, token.trim());
    return api(path, { ...opts, retried: true });
  }
  if (raw) {
    if (!res.ok) throw new Error(`Request failed (${res.status}).`);
    return res;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status}).`);
  return data;
}

async function fetchBlobUrl(jobId, name) {
  const res = await api(`/api/jobs/${jobId}/files/${name}`, { raw: true });
  const url = URL.createObjectURL(await res.blob());
  view.objectUrls.push(url);
  return url;
}

async function fetchBytes(jobId, name) {
  const res = await api(`/api/jobs/${jobId}/files/${name}`, { raw: true });
  return res.arrayBuffer();
}

async function download(jobId, name, filename) {
  const res = await api(`/api/jobs/${jobId}/files/${name}`, { raw: true });
  const url = URL.createObjectURL(await res.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function revokeObjectUrls() {
  for (const url of view.objectUrls) URL.revokeObjectURL(url);
  view.objectUrls = [];
}

function slugFor(job) {
  const base = (job.name || job.id).replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  return base.slice(0, 60) || job.id;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// ----- job list and polling --------------------------------------------------

async function refreshJobs() {
  try {
    const { jobs } = await api("/api/jobs");
    view.jobs = new Map(jobs.map((job) => [job.id, job]));
    renderJobList();
    if (view.selected && view.jobs.has(view.selected)) renderDetail(view.jobs.get(view.selected));
  } catch (err) {
    console.warn("Could not refresh jobs:", err);
  }
}

function renderJobList() {
  els.list.replaceChildren();
  for (const job of view.jobs.values()) {
    const item = element("li");
    const button = element("button", "job-item" + (job.id === view.selected ? " selected" : ""));
    button.type = "button";
    button.append(
      element("span", "name", job.name || `Untitled (${job.id})`),
      element("span", `pill ${job.status}`, STATUS_LABELS[job.status] || job.status),
    );
    button.addEventListener("click", () => selectJob(job.id));
    item.append(button);
    els.list.append(item);
  }
  els.listEmpty.hidden = view.jobs.size > 0;
}

function selectJob(jobId) {
  if (view.selected !== jobId) {
    view.selected = jobId;
    view.renderedFor = null;
    els.resultCard.hidden = true;
  }
  renderJobList();
  const job = view.jobs.get(jobId);
  if (job) renderDetail(job);
}

// ----- job detail ------------------------------------------------------------

function renderDetail(job) {
  els.emptyState.hidden = true;
  els.detail.hidden = false;

  els.title.textContent = job.name || "Untitled job";
  els.status.textContent = STATUS_LABELS[job.status] || job.status;
  els.status.className = `pill ${job.status}`;

  const reference = job.reference_width_mm
    ? `reference width ${job.reference_width_mm} mm`
    : "no reference (unscaled)";
  els.meta.textContent = `${job.photos} photo${job.photos === 1 ? "" : "s"} · ${reference} · updated ${new Date(job.updated_at).toLocaleString()}`;

  renderStages(job);

  if (job.status === "failed") {
    els.errorBox.hidden = false;
    els.errorBox.textContent = `Failed${job.error_stage ? ` at "${job.error_stage}"` : ""}: ${job.error}`;
  } else {
    els.errorBox.hidden = true;
  }

  els.notes.replaceChildren();
  if (job.status === "done" && job.scale) els.notes.append(element("p", null, job.scale.accuracy));
  for (const note of job.notes || []) els.notes.append(element("p", null, note));

  if (job.status === "done") {
    els.resultCard.hidden = false;
    if (view.renderedFor !== job.id) renderResult(job);
  } else {
    els.resultCard.hidden = true;
  }
}

function renderStages(job) {
  const current = STAGE_ORDER.indexOf(job.stage);
  els.stages.replaceChildren();
  for (const [key, label] of STAGES) {
    const idx = STAGE_ORDER.indexOf(key);
    let state = "todo";
    if (job.status === "done" || idx < current) state = "done";
    else if (idx === current && job.status === "running") state = "active";
    else if (idx === current && job.status === "failed") state = "failed";
    els.stages.append(element("li", state, label));
  }
}

async function renderResult(job) {
  view.renderedFor = job.id;
  revokeObjectUrls();
  try {
    const [maskUrl, meshBytes] = await Promise.all([
      fetchBlobUrl(job.id, "mask.png"),
      fetchBytes(job.id, "mesh.stl"),
    ]);
    if (view.selected !== job.id) return;

    els.maskImg.src = maskUrl;
    showMesh(meshBytes);

    els.dlStl.onclick = () => download(job.id, "mesh.stl", `${slugFor(job)}.stl`).catch(showDownloadError);
    els.dl3mf.onclick = () => download(job.id, "model.3mf", `${slugFor(job)}.3mf`).catch(showDownloadError);

    renderChecks(job.report);
  } catch (err) {
    view.renderedFor = null;
    els.errorBox.hidden = false;
    els.errorBox.textContent = `Could not load the result: ${err.message}`;
  }
}

function showDownloadError(err) {
  els.errorBox.hidden = false;
  els.errorBox.textContent = `Download failed: ${err.message}`;
}

function renderChecks(report) {
  els.checksBody.replaceChildren();
  for (const check of report.checks) {
    const row = element("tr");
    row.append(
      element("td", null, check.name.replace(/_/g, " ")),
      element("td", null),
      element("td", null, check.detail),
    );
    row.children[1].append(element("span", `pill ${check.status}`, check.status.replace("_", " ")));
    els.checksBody.append(row);
  }
  const [x, y, z] = report.dimensions_mm.map((d) => d.toFixed(1));
  els.dims.textContent = `Size ${x} × ${y} × ${z} mm · volume ${Math.round(report.volume_mm3)} mm³ · ${report.triangles} triangles`;
}

// ----- 3D viewer -------------------------------------------------------------

function ensureViewer() {
  if (view.viewer) return view.viewer;
  const container = els.viewer;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  container.replaceChildren(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f5f7);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 2.0;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x9999aa, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(200, 400, 300);
  scene.add(sun);
  scene.add(new THREE.GridHelper(220, 22, 0x6b7280, 0xc7cbd3));

  const resize = () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(container);
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  view.viewer = { renderer, scene, camera, controls, mesh: null };
  const turn = document.getElementById("turn-toggle");
  if (turn) {
    turn.addEventListener("click", () => {
      controls.autoRotate = !controls.autoRotate;
      turn.textContent = controls.autoRotate ? "Pause turn" : "Start turn";
    });
  }
  resize();
  return view.viewer;
}

function showMesh(buffer) {
  const v = ensureViewer();
  const geometry = new STLLoader().parse(buffer);
  geometry.computeBoundingBox();
  const b = geometry.boundingBox;

  if (v.mesh) {
    v.scene.remove(v.mesh);
    v.mesh.geometry.dispose();
    v.mesh.material.dispose();
  }
  const material = new THREE.MeshStandardMaterial({ color: 0x5b8fd6, roughness: 0.55, metalness: 0.05 });
  const mesh = new THREE.Mesh(geometry, material);
  // STL is Z-up; three.js is Y-up. Rotating -90° about X maps (x, y, z) to (x, z, -y).
  mesh.rotation.x = -Math.PI / 2;
  v.scene.add(mesh);
  v.mesh = mesh;

  const size = Math.max(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z, 1);
  const cx = (b.min.x + b.max.x) / 2;
  const cy = (b.min.z + b.max.z) / 2;
  const cz = -(b.min.y + b.max.y) / 2;
  v.controls.target.set(cx, cy, cz);
  v.camera.position.set(cx + size * 1.6, cy + size * 1.2, cz + size * 1.6);
  v.camera.near = size / 100;
  v.camera.far = size * 100;
  v.camera.updateProjectionMatrix();
  v.controls.update();
}

// ----- new job form ----------------------------------------------------------

function showFormError(message) {
  els.formError.hidden = false;
  els.formError.textContent = message;
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  els.formError.hidden = true;

  const files = [...els.photos.files];
  if (files.length === 0) return showFormError("Choose at least one photo.");
  if (files.length > 5) return showFormError("Choose up to 5 photos.");

  const referenceText = els.reference.value.trim();
  const reference = referenceText ? Number(referenceText) : null;
  if (reference !== null && !(reference >= 1 && reference <= 2000)) {
    return showFormError("Reference width must be between 1 and 2000 mm.");
  }

  els.submit.disabled = true;
  els.submit.textContent = "Uploading…";
  try {
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference_width_mm: reference, name: els.name.value.trim() || null }),
    });
    for (const file of files) {
      await api(`/api/jobs/${job.id}/photos`, {
        method: "POST",
        headers: { "Content-Type": file.type || "application/octet-stream" },
        body: file,
      });
    }
    await api(`/api/jobs/${job.id}/start`, { method: "POST" });
    els.form.reset();
    view.selected = job.id;
    view.renderedFor = null;
    await refreshJobs();
  } catch (err) {
    showFormError(err.message);
  } finally {
    els.submit.disabled = false;
    els.submit.textContent = "Create model";
  }
});

// ----- start ------------------------------------------------------------------

refreshJobs();
setInterval(refreshJobs, POLL_MS);
