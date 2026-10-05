// YVP — capture guidée pour la photo de profil LinkedIn (composant st.components.v2).
// La caméra s'ouvre au centre de l'écran ; MediaPipe Face Landmarker (servi en
// local depuis webui/static/mediapipe) suit le visage et donne des consignes
// (« rapproche-toi », « souris », « tourne la tête »…). Chaque pose est prise
// automatiquement dès qu'elle est tenue, puis recadrée sur le visage : c'est ce
// que le module d'identité de PhotoMaker voit le mieux. Les photos sont envoyées
// à Python (trigger « photos »), rien ne quitte le PC.
//
// Streamlit rappelle cette fonction à chaque rerun sur le même parentElement :
// l'état vit dans CAPTURES et seule la première invocation construit l'interface.

const CAPTURES = new WeakMap();
const ASSETS = `${window.location.origin}/app/static/mediapipe/`;
const MODEL = "face_landmarker.task";
const DETECT_EVERY_MS = 66;   // ~15 analyses par seconde suffisent
const HOLD_MS = 800;          // pose tenue avant la prise
const STABLE_MOVE = 0.008;    // déplacement max du nez entre deux analyses (fraction de l'image)
const DARK_LUMA = 60;         // visage trop sombre en dessous (0-255)
const CROP_SCALE = 2.0;       // carré = 2× le visage, comme le worker
const MAX_SIDE = 1024;

// Repères du maillage MediaPipe : bout du nez et bords gauche/droit du visage.
const NOSE = 1, CHEEK_A = 234, CHEEK_B = 454;

// Poses demandées. size = hauteur du visage (front → menton) / hauteur de l'image
// affichée : ~0,3 devant la webcam d'un portable, ~0,4 pour un selfie au téléphone ;
// turn : front (face), side (tourné), other (tourné de l'autre côté), slight (un peu).
const STEPS = [
  { id: "front", size: [0.26, 0.5], turn: "front", smile: "any" },
  { id: "smile", size: [0.26, 0.5], turn: "front", smile: "smile" },
  { id: "side", size: [0.24, 0.5], turn: "side", smile: "any" },
  { id: "other", size: [0.24, 0.5], turn: "other", smile: "any" },
  { id: "closer", size: [0.45, 0.85], turn: "front", smile: "any" },
  { id: "farther", size: [0.16, 0.25], turn: "front", smile: "any" },
  { id: "slight_smile", size: [0.24, 0.5], turn: "slight", smile: "smile" },
  { id: "natural", size: [0.26, 0.5], turn: "front", smile: "neutral" },
];
const TURN = { front: [0, 0.07], side: [0.13, 0.32], other: [0.13, 0.32], slight: [0.07, 0.18] };

let landmarkerPromise = null;

function loadLandmarker() {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await import(`${ASSETS}vision_bundle.js`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${ASSETS}wasm`);
      const options = (delegate) => ({
        baseOptions: { modelAssetPath: `${ASSETS}${MODEL}`, delegate },
        runningMode: "VIDEO",
        numFaces: 2,
        outputFaceBlendshapes: true,
      });
      try {
        return await vision.FaceLandmarker.createFromOptions(fileset, options("GPU"));
      } catch (error) {
        console.warn("YVP headshot: GPU face tracking unavailable, using CPU", error);
        return vision.FaceLandmarker.createFromOptions(fileset, options("CPU"));
      }
    })();
    landmarkerPromise.catch(() => { landmarkerPromise = null; });
  }
  return landmarkerPromise;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) node.appendChild(child);
  return node;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result);
      resolve(url.slice(url.indexOf(",") + 1));
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

// Mesures du visage dans la zone affichée (coordonnées 0-1 de cette zone).
export function measureFace(landmarks, blendshapes, crop) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const toCrop = (x, y) => [(x - crop.x) / crop.w, (y - crop.y) / crop.h];
  const [x0, y0] = toCrop(minX, minY);
  const [x1, y1] = toCrop(maxX, maxY);
  const a = landmarks[CHEEK_A], b = landmarks[CHEEK_B], nose = landmarks[NOSE];
  const width = Math.abs(b.x - a.x) || 1e-6;
  const score = (name) => blendshapes?.find((c) => c.categoryName === name)?.score ?? 0;
  return {
    box: { minX, minY, maxX, maxY },             // image caméra entière
    cx: (x0 + x1) / 2, cy: (y0 + y1) / 2,
    size: y1 - y0,
    turn: (nose.x - (a.x + b.x) / 2) / width,    // 0 de face, ± quand la tête tourne
    nose: toCrop(nose.x, nose.y),
    smile: (score("mouthSmileLeft") + score("mouthSmileRight")) / 2,
    blink: Math.max(score("eyeBlinkLeft"), score("eyeBlinkRight")),
  };
}

// Première consigne non respectée pour cette pose, ou "" si tout est bon.
export function checkPose(step, face, context) {
  if (!face) return "hint_no_face";
  if (context.faces > 1) return "hint_one_person";
  if (face.cx < 0.3 || face.cx > 0.7 || face.cy < 0.25 || face.cy > 0.72) return "hint_center";
  if (face.size < step.size[0]) return "hint_closer";
  if (face.size > step.size[1]) return "hint_farther";
  const turn = Math.abs(face.turn);
  const [low, high] = TURN[step.turn];
  if (step.turn === "front" && turn > high) return "hint_look_camera";
  if (step.turn !== "front") {
    if (step.turn === "other" && context.sideSign && Math.sign(face.turn) === context.sideSign && turn > 0.05) {
      return "hint_other_side";
    }
    if (turn < low) return "hint_turn_more";
    if (turn > high) return "hint_turn_less";
  }
  if (step.smile === "smile" && face.smile < 0.45) return "hint_smile";
  if (step.smile === "neutral" && face.smile > 0.25) return "hint_neutral";
  if (face.blink > 0.45) return "hint_eyes_open";
  if (context.dark) return "hint_light";
  if (context.moving) return "hint_hold_still";
  return "";
}

function createCapture(root, component) {
  const state = {
    component,
    labels: {},
    mode: "idle",        // idle | live | sending
    stream: null,
    landmarker: null,
    raf: 0,
    lastDetect: 0,
    stepIndex: 0,
    shots: [],           // { step, blob, url }
    sideSign: 0,
    holdSince: 0,
    lastNose: null,
    capturing: false,
    hint: "",
    face: null,
    flashUntil: 0,
  };
  const L = (key) => state.labels[key] || key;

  const ui = {};
  ui.canvas = el("canvas", { class: "hs-canvas", width: "720", height: "960" });
  ui.instruction = el("div", { class: "hs-instruction" });
  ui.hint = el("div", { class: "hs-hint" });
  ui.progress = el("div", { class: "hs-progress" });
  ui.thumbs = el("div", { class: "hs-thumbs" });
  ui.status = el("div", { class: "hs-status" });
  ui.start = el("button", { type: "button", class: "btn primary" });
  ui.skip = el("button", { type: "button", class: "btn" });
  ui.finish = el("button", { type: "button", class: "btn primary" });
  ui.cancel = el("button", { type: "button", class: "btn" });
  ui.stage = el("div", { class: "hs-stage" }, [ui.canvas, ui.instruction, ui.progress]);
  ui.panel = el("div", { class: "hs-panel" }, [
    ui.stage, ui.hint, ui.thumbs, ui.status,
    el("div", { class: "hs-actions" }, [ui.cancel, ui.skip, ui.finish]),
  ]);
  ui.overlay = el("div", { class: "hs-overlay" }, [ui.panel]);
  ui.container = el("div", { class: "hs" }, [ui.start, ui.overlay]);
  root.appendChild(ui.container);

  const ctx = ui.canvas.getContext("2d");
  // Le suivi analyse seulement la zone affichée (3:4) : dans l'image paysage
  // entière, un visage un peu éloigné devient trop petit pour le détecteur.
  const detectCanvas = document.createElement("canvas");
  detectCanvas.width = 480;
  detectCanvas.height = 640;
  const detectCtx = detectCanvas.getContext("2d");
  const lumaCanvas = document.createElement("canvas");
  lumaCanvas.width = lumaCanvas.height = 24;
  const lumaCtx = lumaCanvas.getContext("2d", { willReadFrequently: true });
  const video = el("video", { playsinline: "", muted: "" });
  video.muted = true;

  ui.start.addEventListener("click", () => start());
  ui.cancel.addEventListener("click", () => stop());
  ui.skip.addEventListener("click", () => nextStep());
  ui.finish.addEventListener("click", () => send());
  const onKeyDown = (event) => { if (event.key === "Escape" && state.mode === "live") stop(); };
  document.addEventListener("keydown", onKeyDown);

  function render() {
    ui.start.textContent = L("start");
    ui.cancel.textContent = L("cancel");
    ui.skip.textContent = L("skip");
    ui.finish.textContent = L("finish").replace("{count}", String(state.shots.length));
    ui.finish.disabled = state.shots.length < (state.minShots || 5) || state.mode !== "live";
    ui.skip.disabled = state.mode !== "live" || state.stepIndex >= STEPS.length;
    ui.overlay.classList.toggle("open", state.mode !== "idle");
    const step = STEPS[state.stepIndex];
    ui.instruction.textContent = step ? L(`step_${step.id}`) : L("done");
    ui.progress.textContent = `${Math.min(state.stepIndex + 1, STEPS.length)} / ${STEPS.length}`;
    ui.hint.textContent = state.hint ? L(state.hint) : (step ? L("hint_hold_still") : "");
    ui.hint.classList.toggle("ok", !state.hint);
  }

  function setStatus(key, kind = "") {
    ui.status.textContent = key ? L(key) : "";
    ui.status.className = `hs-status ${kind}`;
  }

  async function start() {
    if (!navigator.mediaDevices?.getUserMedia) { setStatus("error_camera", "error"); return; }
    clearShots();
    state.mode = "live";
    state.stepIndex = 0;
    state.sideSign = 0;
    render();
    setStatus("loading");
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: "user" },
        audio: false,
      });
      video.srcObject = state.stream;
      await video.play();
      state.landmarker = await loadLandmarker();
      setStatus("");
      cancelAnimationFrame(state.raf);
      state.raf = requestAnimationFrame(loop);
    } catch (error) {
      console.error("YVP headshot:", error);
      stopTracks();
      setStatus(error?.name === "NotAllowedError" ? "error_permission" : "error_camera", "error");
    }
  }

  function stopTracks() {
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }

  function stop() {
    cancelAnimationFrame(state.raf);
    stopTracks();
    state.mode = "idle";
    state.hint = "";
    render();
  }

  function clearShots() {
    state.shots.forEach((shot) => URL.revokeObjectURL(shot.url));
    state.shots = [];
    ui.thumbs.replaceChildren();
  }

  // Zone affichée : portrait 3:4 au centre de l'image caméra.
  function cropRect() {
    const vw = video.videoWidth, vh = video.videoHeight;
    const w = Math.min(vw, vh * 3 / 4), h = Math.min(vh, vw * 4 / 3);
    return { x: (vw - w) / 2 / vw, y: (vh - h) / 2 / vh, w: w / vw, h: h / vh, px: { w, h, x: (vw - w) / 2, y: (vh - h) / 2 } };
  }

  function faceLuma(box) {
    const vw = video.videoWidth, vh = video.videoHeight;
    lumaCtx.drawImage(video, box.minX * vw, box.minY * vh, (box.maxX - box.minX) * vw,
      (box.maxY - box.minY) * vh, 0, 0, 24, 24);
    const pixels = lumaCtx.getImageData(0, 0, 24, 24).data;
    let sum = 0;
    for (let i = 0; i < pixels.length; i += 4) sum += 0.299 * pixels[i] + 0.587 * pixels[i + 1] + 0.114 * pixels[i + 2];
    return sum / (pixels.length / 4);
  }

  function loop(timestamp) {
    if (state.mode !== "live") return;
    state.raf = requestAnimationFrame(loop);
    if (!video.videoWidth) return;
    const crop = cropRect();
    if (state.landmarker && timestamp - state.lastDetect >= DETECT_EVERY_MS) {
      state.lastDetect = timestamp;
      analyse(timestamp, crop);
    }
    draw(crop, timestamp);
  }

  function analyse(timestamp, crop) {
    const step = STEPS[state.stepIndex];
    let result;
    try {
      const { w, h, x, y } = crop.px;
      detectCtx.drawImage(video, x, y, w, h, 0, 0, detectCanvas.width, detectCanvas.height);
      result = state.landmarker.detectForVideo(detectCanvas, timestamp);
    } catch (error) {
      console.error("YVP headshot: face tracking failed", error);
      return;
    }
    const faces = result.faceLandmarks || [];
    // Repères en coordonnées de la zone affichée : on les ramène à l'image caméra.
    const toVideo = (points) => points.map((p) => ({ x: crop.x + p.x * crop.w, y: crop.y + p.y * crop.h }));
    const face = faces.length
      ? measureFace(toVideo(faces[0]), result.faceBlendshapes?.[0]?.categories, crop) : null;
    state.face = face;
    if (!step || state.capturing) return;
    let moving = false;
    if (face) {
      if (state.lastNose) {
        moving = Math.hypot(face.nose[0] - state.lastNose[0], face.nose[1] - state.lastNose[1]) > STABLE_MOVE;
      }
      state.lastNose = face.nose;
    }
    const hint = checkPose(step, face, {
      faces: faces.length, sideSign: state.sideSign, moving,
      dark: face ? faceLuma(face.box) < DARK_LUMA : false,
    });
    if (hint) {
      state.holdSince = 0;
    } else if (!state.holdSince) {
      state.holdSince = timestamp;
    } else if (timestamp - state.holdSince >= HOLD_MS) {
      state.holdSince = 0;
      if (step.turn === "side") state.sideSign = Math.sign(face.turn);
      takeShot(step, face);
    }
    if (hint !== state.hint) { state.hint = hint; render(); }
  }

  function draw(crop, timestamp) {
    const { w, h, x, y } = crop.px;
    const cw = ui.canvas.width, ch = ui.canvas.height;
    ctx.save();
    // Miroir : on se voit comme dans une glace (les photos, elles, ne sont pas inversées).
    ctx.translate(cw, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, x, y, w, h, 0, 0, cw, ch);
    ctx.restore();
    // Ovale guide : vert quand la pose est bonne, avec la progression du maintien.
    const ok = state.face && !state.hint;
    ctx.lineWidth = 6;
    ctx.strokeStyle = ok ? "#30A46C" : "rgba(255,255,255,0.85)";
    ctx.setLineDash(ok ? [] : [14, 10]);
    ctx.beginPath();
    ctx.ellipse(cw / 2, ch * 0.46, cw * 0.27, ch * 0.29, 0, 0, Math.PI * 2);
    ctx.stroke();
    if (ok && state.holdSince) {
      const progress = Math.min(1, (timestamp - state.holdSince) / HOLD_MS);
      ctx.setLineDash([]);
      ctx.lineWidth = 10;
      ctx.strokeStyle = "#30A46C";
      ctx.beginPath();
      ctx.ellipse(cw / 2, ch * 0.46, cw * 0.27 + 10, ch * 0.29 + 10, 0, -Math.PI / 2, -Math.PI / 2 + progress * Math.PI * 2);
      ctx.stroke();
    }
    if (timestamp < state.flashUntil) {
      ctx.fillStyle = `rgba(255,255,255,${(state.flashUntil - timestamp) / 250})`;
      ctx.fillRect(0, 0, cw, ch);
    }
  }

  // Carré centré sur le visage, dans l'image caméra d'origine (pleine définition).
  async function takeShot(step, face) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const { minX, minY, maxX, maxY } = face.box;
    const fw = (maxX - minX) * vw, fh = (maxY - minY) * vh;
    const side = Math.min(CROP_SCALE * Math.max(fw, fh), vw, vh);
    const left = Math.min(Math.max((minX + maxX) / 2 * vw - side / 2, 0), vw - side);
    const top = Math.min(Math.max((minY + maxY) / 2 * vh - side / 2, 0), vh - side);
    const out = document.createElement("canvas");
    out.width = out.height = Math.round(Math.min(side, MAX_SIDE));
    out.getContext("2d").drawImage(video, left, top, side, side, 0, 0, out.width, out.height);
    state.flashUntil = performance.now() + 250;
    // Pas de nouvelle analyse tant que la photo n'est pas encodée.
    state.capturing = true;
    const blob = await new Promise((resolve) => out.toBlob(resolve, "image/jpeg", 0.92));
    state.capturing = false;
    if (state.mode !== "live") return;
    if (blob) {
      const url = URL.createObjectURL(blob);
      state.shots.push({ step: step.id, blob, url });
      ui.thumbs.appendChild(el("img", { src: url, alt: step.id }));
      setStatus("");
    }
    nextStep();
  }

  function nextStep() {
    state.stepIndex += 1;
    state.holdSince = 0;
    state.hint = "";
    if (state.stepIndex >= STEPS.length) {
      if (state.shots.length >= (state.minShots || 5)) {
        render();
        setTimeout(send, 400);
        return;
      }
      // Pas assez de photos : on revient aux poses passées plutôt que d'abandonner.
      const taken = new Set(state.shots.map((shot) => shot.step));
      const missing = STEPS.findIndex((step) => !taken.has(step.id));
      if (missing >= 0) {
        state.stepIndex = missing;
        setStatus("too_few");
      }
    }
    render();
  }

  async function send() {
    if (state.mode !== "live" || !state.shots.length) return;
    state.mode = "sending";
    cancelAnimationFrame(state.raf);
    stopTracks();
    render();
    setStatus("sending");
    try {
      const photos = await Promise.all(state.shots.map(async (shot) => ({
        step: shot.step, data: await blobToBase64(shot.blob),
      })));
      state.component.setTriggerValue("photos", { photos });
      setStatus("");
    } catch (error) {
      console.error("YVP headshot: send failed", error);
      setStatus("error_camera", "error");
    }
    state.mode = "idle";
    render();
  }

  function update(component) {
    state.component = component;
    const data = component.data || {};
    state.labels = data.labels || state.labels;
    state.minShots = data.min_shots || state.minShots;
    render();
  }

  function destroy() {
    document.removeEventListener("keydown", onKeyDown);
    cancelAnimationFrame(state.raf);
    stopTracks();
    clearShots();
  }

  render();
  // Diagnostic (console du navigateur) : dernières mesures du visage et consigne.
  window.__yvpHeadshotStats = () => ({ step: STEPS[state.stepIndex]?.id, hint: state.hint, face: state.face,
    shots: state.shots.length, camera: `${video.videoWidth}x${video.videoHeight}` });
  return { update, destroy };
}

export default function (component) {
  const root = component.parentElement;
  let capture = CAPTURES.get(root);
  if (!capture) {
    capture = createCapture(root, component);
    CAPTURES.set(root, capture);
  }
  capture.update(component);
  return () => {
    capture.destroy();
    CAPTURES.delete(root);
  };
}
