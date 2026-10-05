// YVP — studio caméra (composant st.components.v2).
// Tout se passe dans le navigateur : caméra + micro, détourage de la personne
// (MediaPipe Selfie Segmentation, servi en local depuis webui/static/mediapipe),
// fond remplacé, enregistrement du rendu. La prise est envoyée à Python
// (trigger « recording ») qui la convertit en MP4.
//
// Streamlit rappelle cette fonction à chaque rerun sur le même parentElement :
// l'état vit dans STUDIOS et seule la première invocation construit l'interface.

const STUDIOS = new WeakMap();
const ASSETS = `${window.location.origin}/app/static/mediapipe/`;
const SIZES = { landscape: [1280, 720], portrait: [720, 1280] };
const PRESETS = [
  { id: "none", kind: "none" },
  { id: "blur", kind: "blur" },
  { id: "violet", kind: "gradient", colors: ["#8C70FF", "#5B38F0"] },
  { id: "sunset", kind: "gradient", colors: ["#FF8A5B", "#FF4F8B"] },
  { id: "ocean", kind: "gradient", colors: ["#2BC0E4", "#1E5799"] },
  { id: "night", kind: "color", color: "#14121F" },
  { id: "light", kind: "color", color: "#F4F2FA" },
  { id: "green", kind: "color", color: "#00B140" },
];

// Modèles de détourage (servis en local). « multiclass » distingue cheveux,
// visage, corps et vêtements : meilleurs contours, mais plus lourd.
const MODELS = {
  selfie: { file: "selfie_segmenter.tflite", personFrom: "foreground" },
  multiclass: { file: "selfie_multiclass_256x256.tflite", personFrom: "background" },
};
// Multiclass par défaut ; repli automatique sur « selfie » si le PC est lent.
const DEFAULT_MODEL = MODELS[window.__yvpSegModel] ? window.__yvpSegModel : "multiclass";
const SLOW_SEGMENT_MS = 30;     // au-delà : détourage une image sur deux
const TOO_SLOW_SEGMENT_MS = 45; // au-delà (en moyenne) : modèle léger
// Bord du masque : en dessous de LOW c'est du fond, au-dessus de HIGH c'est
// la personne. Une bande étroite et un peu haute supprime le liseré clair.
const EDGE = window.__yvpEdge || [0.45, 0.75];
const ZOOM_MIN = 1, ZOOM_MAX = 2.5, ZOOM_SOFT_LIMIT = 1.5;

// Un détourage par modèle et par contexte WebGL (le canvas du studio, ou aucun).
const segmenterPromises = new Map();

// --- Composition sur la carte graphique ------------------------------------
// Pour chaque pixel de sortie : le masque basse définition (256 px) est
// « recalé » sur les vrais contours de l'image (filtre bilatéral joint, comme
// Google Meet), le bord est resserré, puis la couleur des pixels de bord est
// reprise vers l'intérieur de la personne pour effacer le halo de l'ancien fond.
const VERTEX_SHADER = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uVideo;
uniform sampler2D uMask;
uniform sampler2D uBg;
uniform vec4 uCrop;        // zone de la caméra affichée (x, y, w, h en uv)
uniform vec2 uMaskTexel;   // taille d'un point du masque, en uv caméra
uniform vec2 uEdge;        // bord bas / haut du masque
uniform int uReplace;      // 0 : image d'origine, 1 : fond remplacé
in vec2 vUv;
out vec4 outColor;

vec3 cam(vec2 p) { return texture(uVideo, p).rgb; }
float mask(vec2 p) { return texture(uMask, p).r; }

void main() {
  vec2 p = uCrop.xy + vUv * uCrop.zw;
  vec3 color = cam(p);
  if (uReplace == 0) { outColor = vec4(color, 1.0); return; }

  // Filtre bilatéral joint : les points du masque dont la couleur ressemble à
  // ce pixel comptent davantage, le bord suit donc les contours réels.
  float sum = 0.0, weights = 0.0;
  for (int j = -2; j <= 2; j++) {
    for (int i = -2; i <= 2; i++) {
      vec2 q = p + vec2(float(i), float(j)) * uMaskTexel;
      vec3 d = cam(q) - color;
      float w = exp(-float(i * i + j * j) / 4.0 - dot(d, d) * 40.0);
      sum += w * mask(q);
      weights += w;
    }
  }
  float alpha = smoothstep(uEdge.x, uEdge.y, sum / max(weights, 1e-5));

  // Décontamination : un pixel de bord prend la couleur de la personne un peu
  // plus à l'intérieur, au lieu de celle de l'ancien fond (le liseré blanc).
  vec3 person = color;
  if (alpha > 0.001 && alpha < 0.999) {
    vec2 g = vec2(mask(p + vec2(uMaskTexel.x, 0.0)) - mask(p - vec2(uMaskTexel.x, 0.0)),
                  mask(p + vec2(0.0, uMaskTexel.y)) - mask(p - vec2(0.0, uMaskTexel.y)));
    float len = length(g);
    if (len > 1e-4) person = mix(cam(p + g / len * uMaskTexel * 1.5), color, alpha);
  }
  vec3 background = texture(uBg, vUv).rgb;
  outColor = vec4(mix(background, person, alpha), 1.0);
}`;

// Lissage dans le temps, sur la carte graphique : seulement quand ça bouge
// peu, pour éviter un « fantôme » derrière les mouvements rapides.
const SMOOTH_SHADER = `#version 300 es
precision highp float;
uniform sampler2D uNew;
uniform sampler2D uPrev;
uniform int uInvert;
uniform int uHasPrev;
out vec4 outColor;
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy);
  float v = texelFetch(uNew, c, 0).r;
  if (uInvert == 1) v = 1.0 - v;
  float prev = uHasPrev == 1 ? texelFetch(uPrev, c, 0).r : v;
  float k = abs(v - prev) > 0.3 ? 1.0 : 0.6;
  outColor = vec4(vec3(prev + (v - prev) * k), 1.0);
}`;

function createCompositor(canvas) {
  if (window.__yvpNoGL) return null;
  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, alpha: false, antialias: false });
  if (!gl) return null;
  const compile = (type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    return shader;
  };
  const link = (fragment) => {
    const program = gl.createProgram();
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, fragment));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  };
  let program, smoothProgram;
  try {
    program = link(FRAGMENT_SHADER);
    smoothProgram = link(SMOOTH_SHADER);
  } catch (error) {
    console.warn("YVP camera: WebGL compositor unavailable", error);
    return null;
  }
  // VAO à nous : MediaPipe partage ce contexte et a ses propres attributs.
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  for (const p of [program, smoothProgram]) {
    const loc = gl.getAttribLocation(p, "aPos");
    if (loc >= 0) { gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0); }
  }
  gl.bindVertexArray(null);
  const uniform = (name) => gl.getUniformLocation(program, name);
  const u = {
    video: uniform("uVideo"), mask: uniform("uMask"), bg: uniform("uBg"), crop: uniform("uCrop"),
    texel: uniform("uMaskTexel"), edge: uniform("uEdge"), replace: uniform("uReplace"),
  };
  const su = {
    fresh: gl.getUniformLocation(smoothProgram, "uNew"), prev: gl.getUniformLocation(smoothProgram, "uPrev"),
    invert: gl.getUniformLocation(smoothProgram, "uInvert"), hasPrev: gl.getUniformLocation(smoothProgram, "uHasPrev"),
  };
  const texture = () => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };
  const tex = { video: texture(), bg: texture() };
  // Masque lissé : deux textures en alternance (précédent / nouveau).
  const smooth = { width: 0, height: 0, targets: [], current: 0, ready: false };
  let bgKey = null;

  // MediaPipe modifie l'état WebGL entre deux images : on remet le nôtre.
  const resetState = () => {
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST); gl.disable(gl.SCISSOR_TEST); gl.disable(gl.CULL_FACE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.bindVertexArray(vao);
  };
  const bind = (unit, texture) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, texture); };

  return {
    gl,
    // Nouveau masque (texture MediaPipe, valable seulement pendant son rappel) :
    // lissé dans le temps vers une texture à nous, réutilisable ensuite.
    pushMaskTexture(maskTexture, width, height, invert) {
      resetState();
      if (smooth.width !== width || smooth.height !== height) {
        smooth.targets.forEach(({ texture, fbo }) => { gl.deleteTexture(texture); gl.deleteFramebuffer(fbo); });
        smooth.targets = [0, 1].map(() => {
          const t = texture();
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
          const fbo = gl.createFramebuffer();
          gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
          gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
          return { texture: t, fbo };
        });
        Object.assign(smooth, { width, height, current: 0, ready: false });
      }
      const next = 1 - smooth.current;
      gl.useProgram(smoothProgram);
      gl.bindFramebuffer(gl.FRAMEBUFFER, smooth.targets[next].fbo);
      gl.viewport(0, 0, width, height);
      bind(3, maskTexture);
      bind(4, smooth.targets[smooth.current].texture);
      gl.uniform1i(su.fresh, 3); gl.uniform1i(su.prev, 4);
      gl.uniform1i(su.invert, invert ? 1 : 0); gl.uniform1i(su.hasPrev, smooth.ready ? 1 : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.bindVertexArray(null);
      smooth.current = next;
      smooth.ready = true;
    },
    render({ video, crop, mask, background, backgroundKey, width, height }) {
      // MediaPipe redimensionne ce canvas à la taille de la caméra : on lui
      // rend la taille de sortie juste avant de dessiner.
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      resetState();
      gl.useProgram(program);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, width, height);
      bind(0, tex.video);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
      gl.uniform1i(u.video, 0); gl.uniform1i(u.mask, 1); gl.uniform1i(u.bg, 2);
      gl.uniform4f(u.crop, crop.sx / crop.vw, crop.sy / crop.vh, crop.sw / crop.vw, crop.sh / crop.vh);
      const useMask = mask && background && smooth.ready;
      gl.uniform1i(u.replace, useMask ? 1 : 0);
      if (useMask) {
        bind(1, smooth.targets[smooth.current].texture);
        gl.uniform2f(u.texel, 1 / mask.width, 1 / mask.height);
        gl.uniform2f(u.edge, EDGE[0], EDGE[1]);
        bind(2, tex.bg);
        if (backgroundKey !== bgKey || backgroundKey === null) {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, background);
          bgKey = backgroundKey;
        }
      }
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    },
  };
}

// Avec « canvas », MediaPipe travaille dans le contexte WebGL du studio : le
// masque reste une texture sur la carte graphique, sans relecture bloquante.
// Par défaut MediaPipe redimensionne le canvas partagé à la taille de la
// caméra, ce qui casse le format de sortie (portrait affiché en petit, MP4 en
// paysage). Son moteur interne expose setAutoResizeCanvas : on le coupe.
function keepCanvasSize(segmenter) {
  const seen = new Set();
  const visit = (object, depth) => {
    if (!object || typeof object !== "object" || seen.has(object) || depth > 2) return false;
    seen.add(object);
    if (typeof object.setAutoResizeCanvas === "function") {
      object.setAutoResizeCanvas(false);
      return true;
    }
    return Object.values(object).some((value) => visit(value, depth + 1));
  };
  if (!visit(segmenter, 0)) console.warn("YVP camera: could not disable MediaPipe canvas resizing");
}

function loadSegmenter(model, canvas = null) {
  const key = `${model}|${canvas ? "shared" : "own"}`;
  const cache = canvas ? (canvas.__yvpSegmenters ||= new Map()) : segmenterPromises;
  if (!cache.has(key)) {
    const promise = (async () => {
      const vision = await import(`${ASSETS}vision_bundle.js`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${ASSETS}wasm`);
      const options = (delegate) => ({
        ...(canvas ? { canvas } : {}),
        baseOptions: { modelAssetPath: `${ASSETS}${MODELS[model].file}`, delegate },
        runningMode: "VIDEO",
        outputCategoryMask: false,
        outputConfidenceMasks: true,
      });
      let segmenter;
      try {
        segmenter = await vision.ImageSegmenter.createFromOptions(fileset, options("GPU"));
      } catch (error) {
        console.warn("YVP camera: GPU segmentation unavailable, using CPU", error);
        segmenter = await vision.ImageSegmenter.createFromOptions(fileset, options("CPU"));
      }
      if (canvas) keepCanvasSize(segmenter);
      return segmenter;
    })();
    promise.catch(() => cache.delete(key));
    cache.set(key, promise);
  }
  return cache.get(key);
}

function pickMimeType() {
  const candidates = [
    "video/mp4;codecs=avc1.42E01F,mp4a.40.2",
    "video/webm;codecs=vp9,opus",
    "video/webm;codecs=vp8,opus",
    "video/webm",
  ];
  return candidates.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
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

function formatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    // « data:video/webm;codecs=vp9,opus;base64,… » : le type contient lui-même
    // une virgule, on coupe donc après le marqueur base64.
    reader.onload = () => {
      const url = String(reader.result);
      const marker = url.indexOf(";base64,");
      resolve(marker >= 0 ? url.slice(marker + 8) : "");
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

function createStudio(root, component) {
  const state = {
    component,
    labels: {},
    maxSeconds: 180,
    maxBytes: 150 * 1024 * 1024,
    aspect: "portrait",
    background: PRESETS[2],
    photo: null,          // ImageBitmap du fond importé
    stream: null,
    segmenter: null,
    segmenterModel: DEFAULT_MODEL,
    segmenterFailed: false,
    segmentMs: 0,         // durée moyenne d'un détourage
    segmentFrames: 0,
    raf: 0,
    mask: null,           // { width, height } (+ prev, canvas en repli 2D)
    zoom: 1,              // zoom visé (1 = pas de zoom)
    zoomShown: 1,         // zoom affiché, rejoint « zoom » en douceur
    panX: 0, panY: 0,     // recadrage quand on est zoomé (-1 … 1)
    recorder: null,
    chunks: [],
    recordStart: 0,
    recordedBlob: null,
    recordedUrl: "",
    mode: "idle",         // idle | live | countdown | recording | review | sending
  };
  const L = (key) => state.labels[key] || key;

  // --- Interface -----------------------------------------------------------
  const ui = {};
  ui.format = el("div", { class: "seg" });
  ui.backgrounds = el("div", { class: "chips" });
  ui.photoInput = el("input", { type: "file", accept: "image/*", hidden: "" });
  ui.cameraSelect = el("select", { class: "select", hidden: "" });
  ui.micSelect = el("select", { class: "select" });
  ui.micLevel = el("span", { class: "mic-level" }, [el("span", { class: "mic-level__bar" })]);
  ui.micControl = el("div", { class: "mic" }, [ui.micSelect, ui.micLevel]);
  ui.canvas = el("canvas", { class: "stage-canvas" });
  ui.review = el("video", { class: "stage-video", controls: "", playsinline: "", hidden: "" });
  ui.placeholder = el("div", { class: "placeholder" });
  ui.countdown = el("div", { class: "countdown", hidden: "" });
  ui.timer = el("div", { class: "timer", hidden: "" });
  ui.stage = el("div", { class: "stage portrait" }, [ui.canvas, ui.review, ui.placeholder, ui.countdown, ui.timer]);
  ui.zoomOut = el("button", { type: "button", class: "btn icon", text: "−" });
  ui.zoomIn = el("button", { type: "button", class: "btn icon", text: "+" });
  ui.zoomValue = el("button", { type: "button", class: "zoom-value", text: "100 %" });
  ui.zoomHint = el("span", { class: "zoom-hint" });
  ui.zoom = el("div", { class: "zoom" }, [ui.zoomOut, ui.zoomValue, ui.zoomIn, ui.zoomHint]);
  ui.status = el("div", { class: "status" });
  ui.warning = el("div", { class: "status error", hidden: "" });
  ui.actions = el("div", { class: "actions" });

  const row = (labelKey, control) => {
    const label = el("div", { class: "row-label" });
    label.dataset.label = labelKey;
    return el("div", { class: "row" }, [label, control]);
  };
  ui.panel = el("div", { class: "panel" }, [
    row("format", ui.format),
    row("background", ui.backgrounds),
    row("camera", ui.cameraSelect),
    row("microphone", ui.micControl),
    ui.stage,
    ui.zoom,
    ui.warning,
    ui.status,
    ui.actions,
  ]);
  // Caméra allumée : le panneau flotte au centre de l'écran (regard au centre).
  ui.overlay = el("div", { class: "overlay" }, [ui.panel]);
  ui.container = el("div", { class: "studio" }, [ui.overlay, ui.photoInput]);
  root.appendChild(ui.container);

  const onKeyDown = (event) => {
    if (state.mode === "idle") return;
    if (event.key === "Escape" && state.mode === "live") stopCamera();
    else if (event.key === "+" || event.key === "=") setZoom(state.zoom * 1.1);
    else if (event.key === "-" || event.key === "_") setZoom(state.zoom / 1.1);
  };
  document.addEventListener("keydown", onKeyDown);

  // Composition sur la carte graphique ; repli en 2D si WebGL2 est absent.
  // Le rendu WebGL (partagé avec MediaPipe) se fait sur un canvas caché ;
  // chaque image finie est copiée sur le canvas affiché et enregistré, qui
  // garde ainsi toujours la taille exacte du format choisi (720×1280 ou 1280×720).
  const glCanvas = document.createElement("canvas");
  const gpu = createCompositor(glCanvas);
  const canvasCtx = ui.canvas.getContext("2d");
  const bgCanvas = document.createElement("canvas");
  const bgCtx = bgCanvas.getContext("2d");
  const personCanvas = document.createElement("canvas");
  const personCtx = personCanvas.getContext("2d");
  const video = el("video", { playsinline: "", muted: "" });
  video.muted = true;

  function button(labelKey, onClick, variant = "") {
    const b = el("button", { type: "button", class: `btn ${variant}` });
    b.dataset.label = labelKey;
    b.addEventListener("click", onClick);
    return b;
  }

  function setStatus(key, kind = "") {
    ui.status.textContent = key ? L(key) : "";
    ui.status.className = `status ${kind}`;
  }

  function renderLabels() {
    ui.container.querySelectorAll("[data-label]").forEach((node) => {
      node.textContent = L(node.dataset.label);
    });
    if (ui.cameraSelect.parentElement) ui.cameraSelect.parentElement.hidden = ui.cameraSelect.hidden;
    if (ui.micControl.parentElement) ui.micControl.parentElement.hidden = ui.micSelect.hidden;
  }

  function renderFormat() {
    ui.format.replaceChildren(
      ...["portrait", "landscape"].map((aspect) => {
        const b = button(aspect, () => {
          if (state.mode === "recording" || state.mode === "countdown") return;
          state.aspect = aspect;
          state.aspectChosen = true;
          renderFormat();
          resizeStage();
        }, state.aspect === aspect ? "on" : "");
        b.disabled = state.mode === "recording" || state.mode === "countdown" || state.mode === "sending";
        return b;
      }),
    );
    renderLabels();
  }

  function renderBackgrounds() {
    const chips = PRESETS.map((preset) => {
      const chip = el("button", { type: "button", class: `chip ${state.background === preset ? "on" : ""}` });
      chip.dataset.bg = preset.id;
      chip.title = L(`bg_${preset.id}`);
      if (preset.kind === "gradient") chip.style.background = `linear-gradient(135deg, ${preset.colors[0]}, ${preset.colors[1]})`;
      else if (preset.kind === "color") chip.style.background = preset.color;
      else chip.appendChild(el("span", { text: preset.id === "none" ? "⌀" : "≋" }));
      chip.addEventListener("click", () => { state.background = preset; renderBackgrounds(); });
      return chip;
    });
    const photoChip = el("button", { type: "button", class: `chip photo ${state.background?.kind === "photo" ? "on" : ""}` });
    photoChip.title = L("bg_photo");
    if (state.photoUrl) photoChip.style.backgroundImage = `url("${state.photoUrl}")`;
    else photoChip.appendChild(el("span", { text: "+" }));
    photoChip.addEventListener("click", () => ui.photoInput.click());
    ui.backgrounds.replaceChildren(...chips, photoChip);
  }

  function renderActions() {
    const items = [];
    const mode = state.mode;
    if (mode === "idle") {
      items.push(button("start_camera", startCamera, "primary"));
    } else if (mode === "live") {
      items.push(button("record", startCountdown, "rec"));
      items.push(button("stop_camera", stopCamera));
    } else if (mode === "countdown" || mode === "recording") {
      items.push(button("stop", stopRecording, "rec"));
    } else if (mode === "review") {
      items.push(button("keep", sendRecording, "primary"));
      items.push(button("retake", retake));
    }
    ui.actions.replaceChildren(...items);
    renderLabels();
    renderFormat();
  }

  function setZoom(value) {
    state.zoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, value));
    if (state.zoom <= 1.001) { state.panX = 0; state.panY = 0; }
    renderZoom();
  }

  function renderZoom() {
    ui.zoomValue.textContent = `${Math.round(state.zoom * 100)} %`;
    ui.zoomOut.disabled = state.zoom <= ZOOM_MIN + 0.001;
    ui.zoomIn.disabled = state.zoom >= ZOOM_MAX - 0.001;
    ui.zoomOut.title = L("zoom_out");
    ui.zoomIn.title = L("zoom_in");
    ui.zoomValue.title = L("zoom_reset");
    const degraded = state.zoom > ZOOM_SOFT_LIMIT + 0.001;
    ui.zoomHint.textContent = degraded ? L("zoom_quality") : L("zoom_hint");
    ui.zoomHint.classList.toggle("warn", degraded);
  }

  ui.zoomIn.addEventListener("click", () => setZoom(state.zoom + 0.1));
  ui.zoomOut.addEventListener("click", () => setZoom(state.zoom - 0.1));
  ui.zoomValue.addEventListener("click", () => setZoom(1));
  ui.stage.addEventListener("dblclick", () => setZoom(1));
  ui.stage.addEventListener("wheel", (event) => {
    if (state.mode === "idle" || ui.canvas.hidden) return;
    event.preventDefault();
    setZoom(state.zoom * (event.deltaY < 0 ? 1.08 : 1 / 1.08));
  }, { passive: false });
  // Glisser l'aperçu pour recadrer quand on est zoomé.
  let drag = null;
  ui.stage.addEventListener("pointerdown", (event) => {
    if (state.zoom <= 1.001 || ui.canvas.hidden) return;
    drag = { x: event.clientX, y: event.clientY, panX: state.panX, panY: state.panY };
    ui.stage.setPointerCapture(event.pointerId);
    ui.stage.classList.add("dragging");
  });
  ui.stage.addEventListener("pointermove", (event) => {
    const crop = state.lastCrop;
    if (!drag || !crop) return;
    // L'image suit le pointeur : déplacement à l'écran → pixels caméra →
    // fraction de la marge de recadrage disponible de chaque côté.
    const rect = ui.canvas.getBoundingClientRect();
    const clamp = (v) => Math.min(1, Math.max(-1, v));
    const slackX = (crop.vw - crop.sw) / 2, slackY = (crop.vh - crop.sh) / 2;
    if (slackX > 1) state.panX = clamp(drag.panX - (event.clientX - drag.x) / rect.width * crop.sw / slackX);
    if (slackY > 1) state.panY = clamp(drag.panY - (event.clientY - drag.y) / rect.height * crop.sh / slackY);
  });
  const endDrag = () => { drag = null; ui.stage.classList.remove("dragging"); };
  ui.stage.addEventListener("pointerup", endDrag);
  ui.stage.addEventListener("pointercancel", endDrag);

  function setMode(mode) {
    state.mode = mode;
    const reviewing = mode === "review" || mode === "sending";
    ui.review.hidden = !reviewing;
    ui.canvas.hidden = reviewing || mode === "idle";
    ui.placeholder.hidden = true;
    ui.stage.hidden = mode === "idle";
    ui.overlay.classList.toggle("open", mode !== "idle");
    ui.timer.hidden = mode !== "recording";
    ui.stage.classList.toggle("recording", mode === "recording");
    ui.zoom.hidden = mode === "idle" || reviewing;
    ui.cameraSelect.disabled = ui.micSelect.disabled = mode !== "live";
    renderActions();
    renderZoom();
  }

  function resizeStage() {
    const [w, h] = SIZES[state.aspect];
    ui.canvas.width = personCanvas.width = bgCanvas.width = w;
    ui.canvas.height = personCanvas.height = bgCanvas.height = h;
    state.bgKey = null;   // redimensionner efface le fond déjà dessiné
    ui.stage.classList.toggle("portrait", state.aspect === "portrait");
    ui.stage.classList.toggle("landscape", state.aspect === "landscape");
  }

  // --- Caméra --------------------------------------------------------------
  // Caméras et micros (noms visibles seulement après l'autorisation).
  async function listDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const fill = (select, kind, current, fallback) => {
        const list = devices.filter((d) => d.kind === kind && d.deviceId !== "communications");
        select.replaceChildren(...list.map((d, i) => {
          const option = el("option", { value: d.deviceId, text: d.label || `${fallback} ${i + 1}` });
          if (d.deviceId === current) option.selected = true;
          return option;
        }));
        return list.length;
      };
      const camera = state.stream?.getVideoTracks()[0]?.getSettings().deviceId;
      const mic = state.stream?.getAudioTracks()[0]?.getSettings().deviceId;
      ui.cameraSelect.hidden = fill(ui.cameraSelect, "videoinput", camera, "Camera") < 2;
      // Le micro reste visible dès qu'il y en a un : on voit lequel enregistre.
      ui.micSelect.hidden = fill(ui.micSelect, "audioinput", mic, "Micro") < 1;
      renderLabels();
    } catch (_) { /* liste facultative */ }
  }

  // Vumètre du micro : pour vérifier qu'on a choisi le bon avant d'enregistrer.
  function watchMicLevel(stream) {
    state.audioMeter?.close?.();
    state.audioMeter = null;
    const track = stream.getAudioTracks()[0];
    const bar = ui.micLevel.firstChild;
    bar.style.width = "0%";
    if (!track || !window.AudioContext) return;
    try {
      const context = new AudioContext();
      const analyser = context.createAnalyser();
      analyser.fftSize = 512;
      context.createMediaStreamSource(new MediaStream([track])).connect(analyser);
      const samples = new Uint8Array(analyser.fftSize);
      state.audioMeter = context;
      const tick = () => {
        if (state.audioMeter !== context) return;
        analyser.getByteTimeDomainData(samples);
        let peak = 0;
        for (const v of samples) peak = Math.max(peak, Math.abs(v - 128));
        bar.style.width = `${Math.min(100, (peak / 128) * 160)}%`;
        requestAnimationFrame(tick);
      };
      tick();
    } catch (_) { /* vumètre facultatif */ }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) { setStatus("error_unsupported", "error"); return; }
    setStatus("starting");
    try {
      const videoConstraints = { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } };
      if (state.videoDeviceId) videoConstraints.deviceId = { exact: state.videoDeviceId };
      const audio = { echoCancellation: true, noiseSuppression: true, autoGainControl: true };
      if (state.audioDeviceId) audio.deviceId = { exact: state.audioDeviceId };
      let stream;
      let noMic = false;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio });
      } catch (error) {
        if (error?.name === "NotAllowedError") throw error;
        // Pas de micro (ou micro occupé) : on filme quand même, sans le son.
        stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints });
        noMic = true;
      }
      stopTracks();
      state.stream = stream;
      ui.warning.textContent = noMic ? L("no_microphone") : "";
      ui.warning.hidden = !noMic;
      video.srcObject = stream;
      await video.play();
      resizeStage();
      setMode("live");
      setStatus("");
      listDevices();
      watchMicLevel(stream);
      cancelAnimationFrame(state.raf);
      state.raf = requestAnimationFrame(drawFrame);
      ensureSegmenter();
    } catch (error) {
      console.error("YVP camera:", error);
      setMode("idle");
      setStatus(error?.name === "NotAllowedError" ? "error_permission"
        : error?.name === "NotFoundError" ? "error_no_camera" : "error_camera", "error");
    }
  }

  function ensureSegmenter() {
    if (state.segmenter || state.segmenterFailed) return;
    setStatus("loading_ai");
    const model = state.segmenterModel;
    loadSegmenter(model, gpu ? glCanvas : null).then((segmenter) => {
      state.segmenter = segmenter;
      state.mask = null;
      state.segmentMs = 0; state.segmentFrames = 0;
      if (ui.status.textContent === L("loading_ai")) setStatus("");
    }).catch((error) => {
      console.error(`YVP camera: segmentation model ${model} failed to load`, error);
      if (model !== "selfie") { state.segmenterModel = "selfie"; ensureSegmenter(); return; }
      state.segmenterFailed = true;
      setStatus("error_ai", "error");
    });
  }

  // PC trop lent pour le modèle multiclass : on passe au modèle léger.
  function downgradeIfTooSlow() {
    if (state.segmenterModel === "selfie" || state.segmentFrames < 100) return;
    if (state.segmentMs < TOO_SLOW_SEGMENT_MS) return;
    console.warn(`YVP camera: ${state.segmentMs.toFixed(0)} ms per frame, switching to the light model`);
    state.segmenterModel = "selfie";
    loadSegmenter("selfie", gpu ? glCanvas : null).then((segmenter) => {
      state.segmenter = segmenter;
      state.mask = null;
      state.segmentMs = 0; state.segmentFrames = 0;
    }).catch((error) => console.error("YVP camera: light model failed", error));
  }

  function stopTracks() {
    state.stream?.getTracks().forEach((track) => track.stop());
    state.stream = null;
    state.audioMeter?.close?.();
    state.audioMeter = null;
  }

  function stopCamera() {
    cancelAnimationFrame(state.raf);
    stopTracks();
    video.srcObject = null;
    setMode("idle");
  }

  // Changer d'appareil : on libère l'ancien flux avant d'ouvrir le nouveau
  // (certaines webcams refusent d'être ouvertes deux fois).
  const switchDevice = (kind, value) => {
    if (state.mode !== "live") return;
    state[kind] = value;
    stopTracks();
    startCamera();
  };
  ui.cameraSelect.addEventListener("change", () => switchDevice("videoDeviceId", ui.cameraSelect.value));
  ui.micSelect.addEventListener("change", () => switchDevice("audioDeviceId", ui.micSelect.value));

  // --- Rendu image par image ----------------------------------------------
  function cropRect() {
    const vw = video.videoWidth, vh = video.videoHeight;
    const [w, h] = SIZES[state.aspect];
    const target = w / h;
    let sw = vw, sh = vh;
    if (vw / vh > target) sw = vh * target; else sh = vw / target;
    // Zoom numérique : on garde une zone plus petite de l'image caméra,
    // déplacée par le recadrage (panX/panY) sans jamais sortir du cadre.
    state.zoomShown += (state.zoom - state.zoomShown) * 0.25;
    const z = Math.abs(state.zoom - state.zoomShown) < 0.002 ? state.zoom : state.zoomShown;
    sw /= z; sh /= z;
    const cx = vw / 2 + state.panX * (vw - sw) / 2;
    const cy = vh / 2 + state.panY * (vh - sh) / 2;
    state.lastCrop = { sx: cx - sw / 2, sy: cy - sh / 2, sw, sh, vw, vh };
    return state.lastCrop;
  }

  function updateMask(timestamp) {
    const segmenter = state.segmenter;
    if (!segmenter) return false;
    let ok = false;
    const started = performance.now();
    segmenter.segmentForVideo(video, timestamp, (result) => {
      const confidence = result.confidenceMasks?.[0];
      if (!confidence) return;
      const width = confidence.width, height = confidence.height;
      const invertMask = MODELS[state.segmenterModel].personFrom === "background";
      if (gpu) {
        // Le masque reste sur la carte graphique : pas de relecture bloquante.
        gpu.pushMaskTexture(confidence.getAsWebGLTexture(), width, height, invertMask);
        state.mask = { width, height };
        ok = true;
        return;
      }
      if (!state.mask || state.mask.width !== width || state.mask.height !== height) {
        const canvas = document.createElement("canvas");
        canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext("2d");
        state.mask = { width, height, prev: new Float32Array(width * height), canvas, ctx, image: ctx.createImageData(width, height) };
      }
      const values = confidence.getAsFloat32Array();
      const { prev } = state.mask;
      for (let i = 0; i < values.length; i++) {
        const v = invertMask ? 1 - values[i] : values[i];
        // Lissage dans le temps seulement quand ça bouge peu : pas de
        // « fantôme » qui traîne derrière un mouvement rapide.
        const k = Math.abs(v - prev[i]) > 0.3 ? 1 : 0.6;
        prev[i] += (v - prev[i]) * k;
      }
      // Repli 2D (pas de WebGL2) : masque en canal alpha d'un petit canvas.
      const pixels = state.mask.image.data;
      for (let i = 0; i < prev.length; i++) {
        let a = (prev[i] - EDGE[0]) / (EDGE[1] - EDGE[0]);
        a = a <= 0 ? 0 : a >= 1 ? 1 : a * a * (3 - 2 * a);
        const j = i * 4;
        pixels[j] = pixels[j + 1] = pixels[j + 2] = 255;
        pixels[j + 3] = a * 255;
      }
      state.mask.ctx.putImageData(state.mask.image, 0, 0);
      ok = true;
    });
    const spent = performance.now() - started;
    // Les premières images (compilation des shaders du modèle) ne comptent pas.
    state.segmentFrames += 1;
    if (state.segmentFrames > 10) {
      state.segmentMs = state.segmentFrames === 11 ? spent : state.segmentMs * 0.95 + spent * 0.05;
    }
    downgradeIfTooSlow();
    return ok;
  }

  function drawCover(ctx, source, sw, sh, w, h) {
    const scale = Math.max(w / sw, h / sh);
    const dw = sw * scale, dh = sh * scale;
    ctx.drawImage(source, (w - dw) / 2, (h - dh) / 2, dw, dh);
  }

  function drawBackground(ctx, crop, w, h) {
    const bg = state.background;
    if (bg.kind === "color") {
      ctx.fillStyle = bg.color; ctx.fillRect(0, 0, w, h);
    } else if (bg.kind === "gradient") {
      const g = ctx.createLinearGradient(0, 0, w, h);
      g.addColorStop(0, bg.colors[0]); g.addColorStop(1, bg.colors[1]);
      ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    } else if (bg.kind === "photo" && state.photo) {
      drawCover(ctx, state.photo, state.photo.width, state.photo.height, w, h);
    } else if (bg.kind === "blur") {
      ctx.filter = `blur(${Math.round(w / 60)}px)`;
      ctx.drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, -20, -20, w + 40, h + 40);
      ctx.filter = "none";
    }
  }

  function backgroundFor(crop, w, h) {
    const bg = state.background;
    if (bg.kind === "blur") {
      // Flou : petite image floutée à chaque image, agrandie par le GPU.
      bgCanvas.width = Math.round(w / 6); bgCanvas.height = Math.round(h / 6);
      bgCtx.filter = "blur(3px)";
      bgCtx.drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, -4, -4, bgCanvas.width + 8, bgCanvas.height + 8);
      bgCtx.filter = "none";
      state.bgKey = null;
      return { canvas: bgCanvas, key: null };
    }
    const key = `${bg.id}|${w}x${h}|${bg.kind === "photo" ? state.photoUrl : ""}`;
    if (state.bgKey !== key) {
      bgCanvas.width = w; bgCanvas.height = h;
      drawBackground(bgCtx, crop, w, h);
      state.bgKey = key;
    }
    return { canvas: bgCanvas, key };
  }

  function drawFrame(timestamp) {
    state.raf = requestAnimationFrame(drawFrame);
    if (!state.stream || video.readyState < 2 || !video.videoWidth) return;
    const [w, h] = SIZES[state.aspect];
    const crop = cropRect();
    const ctx = canvasCtx;
    // PC lent : détourage une image sur deux, le masque précédent sert entre-temps.
    state.frameNo = (state.frameNo || 0) + 1;
    const wantMask = state.background.kind !== "none" && state.segmenter;
    const skip = state.mask && state.segmentMs > SLOW_SEGMENT_MS && state.frameNo % 2 === 1;
    const replace = wantMask && (skip || updateMask(timestamp)) && state.mask;
    if (gpu) {
      const bg = replace ? backgroundFor(crop, w, h) : null;
      gpu.render({
        video, crop, width: w, height: h,
        mask: replace ? state.mask : null, background: bg?.canvas, backgroundKey: bg?.key ?? null,
      });
      ctx.drawImage(glCanvas, 0, 0, w, h);
    } else if (!replace) {
      ctx.drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, w, h);
    } else {
      drawBackground(ctx, crop, w, h);
      const m = state.mask.canvas;
      const kx = m.width / crop.vw, ky = m.height / crop.vh;
      personCtx.globalCompositeOperation = "source-over";
      personCtx.clearRect(0, 0, w, h);
      personCtx.drawImage(video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, w, h);
      personCtx.globalCompositeOperation = "destination-in";
      personCtx.drawImage(m, crop.sx * kx, crop.sy * ky, crop.sw * kx, crop.sh * ky, 0, 0, w, h);
      personCtx.globalCompositeOperation = "source-over";
      ctx.drawImage(personCanvas, 0, 0);
    }
    if (state.mode === "recording") {
      const elapsed = (performance.now() - state.recordStart) / 1000;
      ui.timer.textContent = `● ${formatTime(elapsed)} / ${formatTime(state.maxSeconds)}`;
      if (elapsed >= state.maxSeconds) stopRecording();
    }
  }

  // --- Enregistrement ------------------------------------------------------
  function startCountdown() {
    if (!window.MediaRecorder) { setStatus("error_unsupported", "error"); return; }
    setMode("countdown");
    let n = 3;
    ui.countdown.hidden = false;
    ui.countdown.textContent = String(n);
    const tick = setInterval(() => {
      n -= 1;
      if (state.mode !== "countdown") { clearInterval(tick); ui.countdown.hidden = true; return; }
      if (n > 0) { ui.countdown.textContent = String(n); return; }
      clearInterval(tick);
      ui.countdown.hidden = true;
      startRecording();
    }, 1000);
  }

  function startRecording() {
    const mimeType = pickMimeType();
    const output = ui.canvas.captureStream(30);
    state.stream.getAudioTracks().forEach((track) => output.addTrack(track));
    const [w] = SIZES[state.aspect];
    try {
      state.recorder = new MediaRecorder(output, {
        ...(mimeType ? { mimeType } : {}),
        videoBitsPerSecond: w >= 1280 ? 4_000_000 : 3_000_000,
        audioBitsPerSecond: 128_000,
      });
    } catch (error) {
      console.error("YVP camera:", error);
      setStatus("error_unsupported", "error");
      setMode("live");
      return;
    }
    state.chunks = [];
    state.recorder.ondataavailable = (event) => { if (event.data?.size) state.chunks.push(event.data); };
    state.recorder.onstop = () => {
      const type = state.recorder.mimeType || mimeType || "video/webm";
      state.recordedBlob = new Blob(state.chunks, { type });
      state.chunks = [];
      if (state.recordedUrl) URL.revokeObjectURL(state.recordedUrl);
      state.recordedUrl = URL.createObjectURL(state.recordedBlob);
      ui.review.src = state.recordedUrl;
      setMode("review");
      setStatus("");
    };
    state.recordStart = performance.now();
    state.recorder.start(1000);
    setMode("recording");
    setStatus("keep_visible");
  }

  function stopRecording() {
    if (state.mode === "countdown") { setMode("live"); return; }
    if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
  }

  function retake() {
    ui.review.pause();
    ui.review.removeAttribute("src");
    if (state.recordedUrl) URL.revokeObjectURL(state.recordedUrl);
    state.recordedUrl = "";
    state.recordedBlob = null;
    setMode(state.stream ? "live" : "idle");
    setStatus("");
  }

  async function sendRecording() {
    const blob = state.recordedBlob;
    if (!blob) return;
    if (blob.size > state.maxBytes) { setStatus("error_too_big", "error"); return; }
    setMode("sending");
    setStatus("sending");
    try {
      const data = await blobToBase64(blob);
      const seconds = ui.review.duration && Number.isFinite(ui.review.duration) ? ui.review.duration : null;
      state.component.setTriggerValue("recording", { data, mime: blob.type, aspect: state.aspect, seconds });
      // Prise envoyée : on coupe la caméra, Python affiche où la vidéo est rangée.
      retake();
      stopCamera();
      setStatus("");
    } catch (error) {
      console.error("YVP camera:", error);
      setMode("review");
      setStatus("error_send", "error");
    }
  }

  // --- Fond : photo importée ----------------------------------------------
  ui.photoInput.addEventListener("change", async () => {
    const file = ui.photoInput.files?.[0];
    ui.photoInput.value = "";
    if (!file) return;
    try {
      const bitmap = await createImageBitmap(file);
      state.photo?.close?.();
      state.photo = bitmap;
      if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
      state.photoUrl = URL.createObjectURL(file);
      state.background = { id: "photo", kind: "photo" };
      renderBackgrounds();
    } catch (error) {
      console.error("YVP camera: bad image", error);
      setStatus("error_photo", "error");
    }
  });

  function update(component) {
    state.component = component;
    const data = component.data || {};
    state.labels = data.labels || state.labels;
    if (data.max_seconds) state.maxSeconds = data.max_seconds;
    if (data.max_bytes) state.maxBytes = data.max_bytes;
    // Format par défaut = celui de la vidéo à monter, tant que l'utilisateur
    // n'a pas choisi lui-même (et jamais pendant un enregistrement).
    if (!state.aspectChosen && SIZES[data.default_aspect] && state.mode === "idle"
        && state.aspect !== data.default_aspect) {
      state.aspect = data.default_aspect;
      resizeStage();
    }
    ui.placeholder.textContent = L("placeholder");
    renderBackgrounds();
    renderActions();
  }

  function destroy() {
    document.removeEventListener("keydown", onKeyDown);
    cancelAnimationFrame(state.raf);
    if (state.recorder && state.recorder.state !== "inactive") {
      state.recorder.onstop = null;
      state.recorder.stop();
    }
    stopTracks();
    if (state.recordedUrl) URL.revokeObjectURL(state.recordedUrl);
    if (state.photoUrl) URL.revokeObjectURL(state.photoUrl);
  }

  resizeStage();
  setMode("idle");
  // Diagnostic (console du navigateur) : modèle utilisé et coût du détourage.
  window.__yvpCameraStats = () => ({
    model: state.segmenterModel, segmentMs: Math.round(state.segmentMs), frames: state.segmentFrames,
    renderer: gpu ? "webgl2" : "2d", camera: `${video.videoWidth}x${video.videoHeight}`, crop: state.lastCrop,
  });
  return { update, destroy };
}

export default function (component) {
  const root = component.parentElement;
  let studio = STUDIOS.get(root);
  if (!studio) {
    studio = createStudio(root, component);
    STUDIOS.set(root, studio);
  }
  studio.update(component);
  return () => {
    studio.destroy();
    STUDIOS.delete(root);
  };
}
