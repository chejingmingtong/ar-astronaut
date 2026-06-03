const video = document.querySelector("#camera");
const canvas = document.querySelector("#arCanvas");
const ctx = canvas.getContext("2d");
const startButton = document.querySelector("#startButton");
const statusPill = document.querySelector("#statusPill");
const statusText = document.querySelector("#statusText");
const qrPanel = document.querySelector("#qrPanel");
const shareButton = document.querySelector("#shareButton");
const closeQrButton = document.querySelector("#closeQrButton");
const shareUrl = document.querySelector("#shareUrl");
const qrImage = document.querySelector("#qrImage");
const startLabel = document.querySelector("#startLabel");
const captureButton = document.querySelector("#captureButton");
const switchCameraButton = document.querySelector("#switchCameraButton");
const stage = document.querySelector(".stage");
const realSuitImage = new Image();
const partSuitImages = {
  body: new Image(),
  leftArm: new Image(),
  rightArm: new Image(),
  leftLeg: new Image(),
  rightLeg: new Image(),
};

let poseLandmarker;
let running = false;
let lastVideoTime = -1;
let hasPose = false;
let smoothedPose = null;
let lastGoodPoseAt = 0;
let lastStatusText = "";
let realSuitReady = false;
let cameraFacingMode = "environment";
let currentStream = null;

const poseHoldMs = 900;
const smoothing = 0.72;
const previewMode = new URLSearchParams(window.location.search).get("preview") === "1";
const assetVersion = "drawn-camera-v1";

realSuitImage.onload = () => {
  realSuitReady = true;
};
realSuitImage.src = `./assets/astronaut-suit-real.png?v=${assetVersion}`;

function loadPartSuitImage(key, src) {
  const image = partSuitImages[key];
  image.onload = () => {
    render();
  };
  image.src = `${src}?v=${assetVersion}`;
}

loadPartSuitImage("body", "./assets/parts/astronaut-body.png");
loadPartSuitImage("leftArm", "./assets/parts/astronaut-arm-approved.png");
loadPartSuitImage("rightArm", "./assets/parts/astronaut-arm-other.png");
loadPartSuitImage("leftLeg", "./assets/parts/astronaut-leg-left-from-real.png");
loadPartSuitImage("rightLeg", "./assets/parts/astronaut-leg-right-from-real.png");

const suitPart = {
  torso: { sx: 250, sy: 100, sw: 540, sh: 780 },
  leftUpperArm: { sx: 75, sy: 210, sw: 280, sh: 430 },
  leftForearm: { sx: 50, sy: 520, sw: 275, sh: 360 },
  leftGlove: { sx: 45, sy: 655, sw: 255, sh: 245 },
  rightUpperArm: { sx: 685, sy: 210, sw: 280, sh: 430 },
  rightForearm: { sx: 710, sy: 520, sw: 275, sh: 360 },
  rightGlove: { sx: 755, sy: 655, sw: 255, sh: 245 },
  leftThigh: { sx: 300, sy: 710, sw: 230, sh: 425 },
  leftShin: { sx: 300, sy: 1000, sw: 230, sh: 360 },
  leftBoot: { sx: 245, sy: 1220, sw: 300, sh: 285 },
  rightThigh: { sx: 530, sy: 710, sw: 230, sh: 425 },
  rightShin: { sx: 530, sy: 1000, sw: 230, sh: 360 },
  rightBoot: { sx: 505, sy: 1220, sw: 300, sh: 285 },
};

const suitThemes = {
  classic: {
    shell: "#edf3f8",
    shade: "#aebdca",
    trim: "#233145",
    visor: "#1d3148",
    accent: "#34d6c5",
  },
  orange: {
    shell: "#f47f32",
    shade: "#9d4326",
    trim: "#2e2c33",
    visor: "#172b3c",
    accent: "#ffe58a",
  },
  lunar: {
    shell: "#cdd5dc",
    shade: "#7f8d98",
    trim: "#27313d",
    visor: "#101820",
    accent: "#bdf5ff",
  },
};

function setStatus(text, mode = "idle") {
  if (text === lastStatusText) {
    return;
  }

  lastStatusText = text;
  statusText.textContent = text;
  statusPill.classList.toggle("is-live", mode === "live");
  statusPill.classList.toggle("is-error", mode === "error");
}

function setupShare() {
  const url = new URL(window.location.href);
  const isArEntry = url.searchParams.get("ar") === "1";
  url.searchParams.set("ar", "1");

  shareUrl.value = url.href;

  if (window.qrcode) {
    const qr = window.qrcode(0, "M");
    qr.addData(url.href);
    qr.make();
    qrImage.src = qr.createDataURL(8, 18);
  } else {
    qrImage.src = `https://api.qrserver.com/v1/create-qr-code/?size=720x720&margin=18&data=${encodeURIComponent(url.href)}`;
  }

  if (isArEntry || window.matchMedia("(pointer: coarse)").matches) {
    qrPanel.classList.add("is-hidden");
  }
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.floor(canvas.clientWidth * dpr);
  const height = Math.floor(canvas.clientHeight * dpr);

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

async function loadPoseModel() {
  if (poseLandmarker) return poseLandmarker;

  try {
    setStatus("正在加载人体识别模型");
    const vision = await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14");
    const fileset = await vision.FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );

    poseLandmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });

    return poseLandmarker;
  } catch (error) {
    console.warn(error);
    setStatus("人体识别模型加载失败，请检查网络后刷新页面", "error");
    return null;
  }
}

function stopCameraStream() {
  if (!currentStream) return;
  currentStream.getTracks().forEach((track) => track.stop());
  currentStream = null;
}

async function startCamera() {
  try {
    const model = await loadPoseModel();

    if (!model) {
      return;
    }

    stopCameraStream();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: cameraFacingMode },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    currentStream = stream;
    video.srcObject = stream;
    await video.play();

    running = true;
    startLabel.textContent = "AR 识别中";
    startButton.disabled = true;
    stage.classList.toggle("is-rear-camera", cameraFacingMode === "environment");
    setStatus(cameraFacingMode === "environment" ? "后置摄像头已开启，请对准人体" : "前置摄像头已开启，请站入画面", "live");
    requestAnimationFrame(render);
  } catch (error) {
    console.warn(error);
    setStatus("无法打开摄像头，请检查权限或使用 HTTPS 地址", "error");
    running = true;
    requestAnimationFrame(render);
  }
}

function getVideoDisplayRect(width, height) {
  const videoWidth = video.videoWidth || width;
  const videoHeight = video.videoHeight || height;
  const scale = Math.max(width / videoWidth, height / videoHeight);
  const displayWidth = videoWidth * scale;
  const displayHeight = videoHeight * scale;

  return {
    x: (width - displayWidth) / 2,
    y: (height - displayHeight) / 2,
    width: displayWidth,
    height: displayHeight,
  };
}

function mapPoint(point, rect) {
  const shouldMirror = cameraFacingMode === "user";
  return {
    x: rect.x + (shouldMirror ? 1 - point.x : point.x) * rect.width,
    y: rect.y + point.y * rect.height,
    score: point.visibility ?? 1,
  };
}

function midpoint(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function pointConfidence(...points) {
  return points.reduce((sum, point) => sum + point.score, 0) / points.length;
}

function poseFromLandmarks(landmarks, width, height) {
  const rect = getVideoDisplayRect(width, height);
  const p = (index) => mapPoint(landmarks[index], rect);
  const leftShoulder = p(11);
  const rightShoulder = p(12);
  const leftHip = p(23);
  const rightHip = p(24);
  const leftElbow = p(13);
  const rightElbow = p(14);
const leftWrist = p(15);
const rightWrist = p(16);
const leftKnee = p(25);
const rightKnee = p(26);
  const leftAnkle = p(27);
  const rightAnkle = p(28);
  const coreConfidence = pointConfidence(leftShoulder, rightShoulder, leftHip, rightHip);
  const shoulderWidth = distance(leftShoulder, rightShoulder);
  const torsoHeight = distance(midpoint(leftShoulder, rightShoulder), midpoint(leftHip, rightHip));

  if (coreConfidence < 0.32 || shoulderWidth < width * 0.08 || torsoHeight < height * 0.08) {
    return null;
  }

  return {
    head: p(0),
    neck: midpoint(leftShoulder, rightShoulder),
    leftShoulder,
    rightShoulder,
    leftElbow,
    rightElbow,
    leftWrist,
    rightWrist,
    leftHip,
    rightHip,
    leftKnee,
    rightKnee,
    leftAnkle,
    rightAnkle,
  };
}

function previewPose(width, height) {
  const cx = width / 2;
  const shoulderY = height * 0.33;
  const hipY = height * 0.58;
  const shoulderW = Math.min(width * 0.34, height * 0.22);
  const hipW = shoulderW * 0.72;

  return {
    head: { x: cx, y: height * 0.2, score: 1 },
    neck: { x: cx, y: shoulderY - height * 0.03, score: 1 },
    leftShoulder: { x: cx - shoulderW / 2, y: shoulderY, score: 1 },
    rightShoulder: { x: cx + shoulderW / 2, y: shoulderY, score: 1 },
    leftElbow: { x: cx - shoulderW * 0.72, y: height * 0.46, score: 1 },
    rightElbow: { x: cx + shoulderW * 0.72, y: height * 0.46, score: 1 },
    leftWrist: { x: cx - shoulderW * 0.72, y: height * 0.58, score: 1 },
    rightWrist: { x: cx + shoulderW * 0.72, y: height * 0.58, score: 1 },
    leftHip: { x: cx - hipW / 2, y: hipY, score: 1 },
    rightHip: { x: cx + hipW / 2, y: hipY, score: 1 },
    leftKnee: { x: cx - hipW * 0.42, y: height * 0.78, score: 1 },
    rightKnee: { x: cx + hipW * 0.42, y: height * 0.78, score: 1 },
    leftAnkle: { x: cx - hipW * 0.42, y: height * 0.92, score: 1 },
    rightAnkle: { x: cx + hipW * 0.42, y: height * 0.92, score: 1 },
  };
}

function mixPoint(previous, next, amount) {
  if (!previous) return next;

  return {
    x: previous.x * amount + next.x * (1 - amount),
    y: previous.y * amount + next.y * (1 - amount),
    score: next.score,
  };
}

function smoothPose(previous, next) {
  if (!previous) return next;

  return Object.fromEntries(
    Object.entries(next).map(([key, point]) => [key, mixPoint(previous[key], point, smoothing)])
  );
}

function roundedRect(x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

function drawFabricGrain(x, y, width, height, density, alpha = 0.12) {
  const count = Math.max(8, Math.floor((width * height * density) / 900));
  const step = Math.max(6, Math.sqrt((width * height) / count));

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = "#7d93a5";
  ctx.lineWidth = 1;
  for (let i = 0; i < count; i += 1) {
    const col = i % Math.max(1, Math.floor(width / step));
    const row = Math.floor(i / Math.max(1, Math.floor(width / step)));
    const px = x + ((col * step + row * 3.7) % width);
    const py = y + ((row * step + col * 2.3) % height);
    const len = 2 + ((i * 7) % 5);
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + len, py + (((i * 11) % 18) - 9) / 10);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSoftShadow(pathFn, blur, alpha) {
  ctx.save();
  ctx.shadowColor = `rgba(10, 20, 32, ${alpha})`;
  ctx.shadowBlur = blur;
  ctx.shadowOffsetY = blur * 0.24;
  pathFn();
  ctx.fillStyle = "rgba(10, 20, 32, 0.08)";
  ctx.fill();
  ctx.restore();
}

function capsule(start, end, width, color, shade, options = {}) {
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const len = distance(start, end);
  const cuffColor = options.cuffColor || "#2f6cae";

  ctx.save();
  ctx.translate(start.x, start.y);
  ctx.rotate(angle);
  drawSoftShadow(() => roundedRect(0, -width / 2, len, width, width / 2), width * 0.22, 0.18);

  const gradient = ctx.createLinearGradient(0, -width / 2, 0, width / 2);
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(0.18, "#f8fbfd");
  gradient.addColorStop(0.58, color);
  gradient.addColorStop(0.88, "#d4e0e8");
  gradient.addColorStop(1, shade);
  ctx.fillStyle = gradient;
  roundedRect(0, -width / 2, len, width, width / 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(75, 96, 113, 0.46)";
  ctx.lineWidth = Math.max(2, width * 0.045);
  ctx.stroke();

  ctx.strokeStyle = "rgba(112, 132, 148, 0.44)";
  ctx.lineWidth = Math.max(1.5, width * 0.028);
  for (let i = 0.18; i <= 0.84; i += 0.16) {
    ctx.beginPath();
    ctx.moveTo(len * i, -width * 0.4);
    ctx.quadraticCurveTo(len * (i + 0.025), 0, len * i, width * 0.4);
    ctx.stroke();
  }

  if (options.cuffs !== false) {
    ctx.fillStyle = cuffColor;
    roundedRect(len * 0.02, -width * 0.52, Math.max(5, width * 0.14), width * 1.04, width * 0.07);
    ctx.fill();
    roundedRect(len - Math.max(5, width * 0.16), -width * 0.52, Math.max(5, width * 0.14), width * 1.04, width * 0.07);
    ctx.fill();
  }

  ctx.globalAlpha = 0.38;
  drawFabricGrain(0, -width * 0.44, len, width * 0.88, 0.45, 0.4);
  ctx.restore();
}

function bandOnLimb(start, end, at, bandWidth, limbWidth, color) {
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const len = distance(start, end);

  ctx.save();
  ctx.translate(start.x, start.y);
  ctx.rotate(angle);
  ctx.fillStyle = color;
  roundedRect(len * at - bandWidth / 2, -limbWidth * 0.53, bandWidth, limbWidth * 1.06, bandWidth * 0.45);
  ctx.fill();
  ctx.strokeStyle = "rgba(28, 64, 112, 0.55)";
  ctx.lineWidth = Math.max(1.5, limbWidth * 0.035);
  ctx.stroke();
  ctx.restore();
}

function drawShoulderPatch(point, width, flip) {
  ctx.save();
  ctx.translate(point.x, point.y);
  ctx.rotate(flip * 0.22);
  ctx.fillStyle = "#f28b45";
  roundedRect(-width * 0.24, -width * 0.52, width * 0.5, width * 0.34, width * 0.08);
  ctx.fill();
  ctx.strokeStyle = "rgba(143, 76, 42, 0.52)";
  ctx.lineWidth = Math.max(1.5, width * 0.035);
  ctx.stroke();
  ctx.restore();
}

function drawHose(from, to, width, flip = 1) {
  const controlA = {
    x: from.x + flip * width * 0.5,
    y: from.y + width * 0.15,
  };
  const controlB = {
    x: to.x + flip * width * 0.4,
    y: to.y - width * 0.3,
  };

  ctx.save();
  ctx.strokeStyle = "rgba(78, 96, 112, 0.86)";
  ctx.lineWidth = Math.max(5, width * 0.09);
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.bezierCurveTo(controlA.x, controlA.y, controlB.x, controlB.y, to.x, to.y);
  ctx.stroke();

  ctx.strokeStyle = "rgba(238, 246, 250, 0.8)";
  ctx.lineWidth = Math.max(2, width * 0.035);
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.bezierCurveTo(controlA.x, controlA.y, controlB.x, controlB.y, to.x, to.y);
  ctx.stroke();
  ctx.restore();
}

function drawBoot(point, width, theme, flip = 1) {
  ctx.save();
  ctx.translate(point.x, point.y);
  ctx.scale(flip, 1);
  drawSoftShadow(() => roundedRect(-width * 0.58, -width * 0.36, width, width * 0.84, width * 0.22), width * 0.18, 0.2);
  const bootGradient = ctx.createLinearGradient(0, -width * 0.36, 0, width * 0.55);
  bootGradient.addColorStop(0, "#ffffff");
  bootGradient.addColorStop(0.58, "#dce7ee");
  bootGradient.addColorStop(1, "#9baebe");
  ctx.fillStyle = bootGradient;
  roundedRect(-width * 0.58, -width * 0.36, width * 0.9, width * 0.84, width * 0.22);
  ctx.fill();
  ctx.fillStyle = "#455a6b";
  roundedRect(-width * 0.52, width * 0.28, width * 1.05, width * 0.3, width * 0.1);
  ctx.fill();
  ctx.strokeStyle = "#6f8291";
  ctx.lineWidth = Math.max(2, width * 0.035);
  for (let i = 0; i < 4; i += 1) {
    ctx.beginPath();
    ctx.moveTo(-width * 0.38 + i * width * 0.18, width * 0.34);
    ctx.lineTo(-width * 0.3 + i * width * 0.18, width * 0.5);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(35, 49, 69, 0.5)";
  ctx.lineWidth = Math.max(2, width * 0.05);
  ctx.stroke();
  ctx.restore();
}

function drawGlove(point, width, theme) {
  ctx.save();
  const gloveGradient = ctx.createRadialGradient(point.x - width * 0.14, point.y - width * 0.18, width * 0.1, point.x, point.y, width * 0.62);
  gloveGradient.addColorStop(0, "#ffffff");
  gloveGradient.addColorStop(0.58, "#e4edf4");
  gloveGradient.addColorStop(1, "#9caebb");
  ctx.fillStyle = gloveGradient;
  ctx.beginPath();
  ctx.arc(point.x, point.y, width * 0.48, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(35, 49, 69, 0.58)";
  ctx.lineWidth = Math.max(2, width * 0.05);
  ctx.stroke();
  ctx.fillStyle = "#9fb0bd";
  roundedRect(point.x - width * 0.35, point.y - width * 0.12, width * 0.7, width * 0.24, width * 0.1);
  ctx.fill();
  ctx.strokeStyle = "rgba(93, 111, 126, 0.42)";
  ctx.lineWidth = Math.max(1, width * 0.026);
  for (let i = -0.2; i <= 0.2; i += 0.14) {
    ctx.beginPath();
    ctx.moveTo(point.x + width * i, point.y - width * 0.34);
    ctx.lineTo(point.x + width * (i + 0.08), point.y - width * 0.02);
    ctx.stroke();
  }
  ctx.restore();
}

function drawSuitSeams(torsoX, torsoY, suitWidth, torsoHeight, theme) {
  ctx.save();
  ctx.strokeStyle = "rgba(86, 110, 129, 0.34)";
  ctx.lineWidth = Math.max(2, suitWidth * 0.012);
  ctx.beginPath();
  for (let i = 0.22; i <= 0.88; i += 0.16) {
    ctx.moveTo(torsoX + suitWidth * 0.18, torsoY + torsoHeight * i);
    ctx.bezierCurveTo(
      torsoX + suitWidth * 0.34,
      torsoY + torsoHeight * (i + 0.025),
      torsoX + suitWidth * 0.66,
      torsoY + torsoHeight * (i - 0.025),
      torsoX + suitWidth * 0.82,
      torsoY + torsoHeight * i
    );
  }
  ctx.stroke();

  ctx.strokeStyle = "rgba(35, 49, 69, 0.28)";
  ctx.lineWidth = Math.max(2, suitWidth * 0.015);
  ctx.beginPath();
  ctx.moveTo(torsoX + suitWidth * 0.5, torsoY + torsoHeight * 0.12);
  ctx.lineTo(torsoX + suitWidth * 0.5, torsoY + torsoHeight * 1.02);
  ctx.stroke();
  ctx.restore();
}

function drawMissionPatch(x, y, size) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, size, 0, Math.PI * 2);
  ctx.fillStyle = "#235aa8";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(2, size * 0.16);
  ctx.stroke();

  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(1.5, size * 0.1);
  ctx.beginPath();
  ctx.moveTo(x - size * 0.55, y + size * 0.12);
  ctx.quadraticCurveTo(x, y - size * 0.58, x + size * 0.55, y + size * 0.12);
  ctx.stroke();

  ctx.fillStyle = "#ff5b5b";
  ctx.beginPath();
  ctx.moveTo(x, y - size * 0.55);
  ctx.lineTo(x + size * 0.17, y + size * 0.42);
  ctx.lineTo(x - size * 0.17, y + size * 0.42);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawLifeSupportPack(center, suitWidth, torsoHeight, theme) {
  const packWidth = suitWidth * 0.66;
  const packHeight = torsoHeight * 0.92;

  ctx.save();
  drawSoftShadow(
    () => roundedRect(center.x - packWidth / 2, center.y - packHeight * 0.12, packWidth, packHeight, packWidth * 0.22),
    suitWidth * 0.08,
    0.2
  );
  roundedRect(center.x - packWidth / 2, center.y - packHeight * 0.12, packWidth, packHeight, packWidth * 0.22);
  const packGradient = ctx.createLinearGradient(center.x - packWidth / 2, center.y, center.x + packWidth / 2, center.y + packHeight);
  packGradient.addColorStop(0, "rgba(238, 245, 249, 0.86)");
  packGradient.addColorStop(0.58, "rgba(176, 190, 201, 0.78)");
  packGradient.addColorStop(1, "rgba(110, 132, 149, 0.78)");
  ctx.fillStyle = packGradient;
  ctx.fill();
  ctx.strokeStyle = "rgba(78, 96, 112, 0.55)";
  ctx.lineWidth = Math.max(2, suitWidth * 0.018);
  ctx.stroke();

  ctx.strokeStyle = "rgba(238, 246, 250, 0.64)";
  ctx.lineWidth = Math.max(3, suitWidth * 0.026);
  ctx.beginPath();
  ctx.moveTo(center.x - packWidth * 0.34, center.y + packHeight * 0.02);
  ctx.lineTo(center.x - packWidth * 0.34, center.y + packHeight * 0.62);
  ctx.moveTo(center.x + packWidth * 0.34, center.y + packHeight * 0.02);
  ctx.lineTo(center.x + packWidth * 0.34, center.y + packHeight * 0.62);
  ctx.stroke();

  ctx.fillStyle = "rgba(58, 75, 90, 0.22)";
  roundedRect(center.x - packWidth * 0.18, center.y + packHeight * 0.46, packWidth * 0.36, packHeight * 0.18, packWidth * 0.06);
  ctx.fill();
  ctx.restore();
}

function drawChestUnit(center, suitWidth, torsoHeight, theme) {
  const unitWidth = suitWidth * 0.48;
  const unitHeight = torsoHeight * 0.34;
  const x = center.x - unitWidth / 2;
  const y = center.y - unitHeight / 2;

  ctx.save();
  drawSoftShadow(() => roundedRect(x, y, unitWidth, unitHeight, unitWidth * 0.12), suitWidth * 0.035, 0.18);
  roundedRect(x, y, unitWidth, unitHeight, unitWidth * 0.12);
  const unitGradient = ctx.createLinearGradient(x, y, x + unitWidth, y + unitHeight);
  unitGradient.addColorStop(0, "#ffffff");
  unitGradient.addColorStop(0.48, "#e9f0f5");
  unitGradient.addColorStop(1, "#b8c8d4");
  ctx.fillStyle = unitGradient;
  ctx.fill();
  ctx.strokeStyle = "rgba(74, 92, 108, 0.68)";
  ctx.lineWidth = Math.max(2, suitWidth * 0.016);
  ctx.stroke();

  roundedRect(x + unitWidth * 0.12, y + unitHeight * 0.16, unitWidth * 0.22, unitHeight * 0.36, 6);
  ctx.fillStyle = "#79a9df";
  ctx.fill();
  ctx.strokeStyle = "#3d6796";
  ctx.stroke();

  ctx.fillStyle = "#26394c";
  ctx.beginPath();
  ctx.arc(x + unitWidth * 0.52, y + unitHeight * 0.32, unitWidth * 0.055, 0, Math.PI * 2);
  ctx.arc(x + unitWidth * 0.7, y + unitHeight * 0.32, unitWidth * 0.055, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#6f8291";
  ctx.lineWidth = Math.max(2, suitWidth * 0.012);
  ctx.beginPath();
  ctx.moveTo(x + unitWidth * 0.14, y + unitHeight * 0.72);
  ctx.lineTo(x + unitWidth * 0.86, y + unitHeight * 0.72);
  ctx.stroke();

  ctx.fillStyle = "#e85f44";
  ctx.beginPath();
  ctx.arc(x + unitWidth * 0.82, y + unitHeight * 0.3, unitWidth * 0.045, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#f4c84f";
  ctx.beginPath();
  ctx.arc(x + unitWidth * 0.82, y + unitHeight * 0.52, unitWidth * 0.04, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawCanvasSuitHardware(torsoX, torsoY, suitWidth, torsoHeight) {
  const strapWidth = Math.max(7, suitWidth * 0.055);
  const beltY = torsoY + torsoHeight * 0.92;
  const collarY = torsoY - torsoHeight * 0.03;

  ctx.save();

  ctx.strokeStyle = "rgba(82, 96, 108, 0.72)";
  ctx.lineWidth = strapWidth;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(torsoX + suitWidth * 0.2, torsoY + torsoHeight * 0.08);
  ctx.lineTo(torsoX + suitWidth * 0.38, torsoY + torsoHeight * 0.74);
  ctx.moveTo(torsoX + suitWidth * 0.8, torsoY + torsoHeight * 0.08);
  ctx.lineTo(torsoX + suitWidth * 0.62, torsoY + torsoHeight * 0.74);
  ctx.stroke();

  ctx.strokeStyle = "#1e6db4";
  ctx.lineWidth = Math.max(4, strapWidth * 0.42);
  ctx.beginPath();
  ctx.moveTo(torsoX + suitWidth * 0.23, torsoY + torsoHeight * 0.14);
  ctx.lineTo(torsoX + suitWidth * 0.4, torsoY + torsoHeight * 0.72);
  ctx.moveTo(torsoX + suitWidth * 0.77, torsoY + torsoHeight * 0.14);
  ctx.lineTo(torsoX + suitWidth * 0.6, torsoY + torsoHeight * 0.72);
  ctx.stroke();

  ctx.fillStyle = "rgba(236, 242, 246, 0.96)";
  roundedRect(torsoX + suitWidth * 0.16, beltY, suitWidth * 0.68, torsoHeight * 0.11, suitWidth * 0.035);
  ctx.fill();
  ctx.strokeStyle = "rgba(70, 88, 104, 0.58)";
  ctx.lineWidth = Math.max(2, suitWidth * 0.012);
  ctx.stroke();

  ctx.fillStyle = "#1f72b8";
  roundedRect(torsoX + suitWidth * 0.28, beltY + torsoHeight * 0.018, suitWidth * 0.12, torsoHeight * 0.07, 4);
  ctx.fill();
  roundedRect(torsoX + suitWidth * 0.6, beltY + torsoHeight * 0.018, suitWidth * 0.12, torsoHeight * 0.07, 4);
  ctx.fill();

  ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
  roundedRect(torsoX + suitWidth * 0.62, torsoY + torsoHeight * 0.55, suitWidth * 0.2, torsoHeight * 0.23, 10);
  ctx.fill();
  ctx.strokeStyle = "rgba(94, 113, 128, 0.38)";
  ctx.stroke();

  ctx.strokeStyle = "rgba(115, 132, 145, 0.26)";
  ctx.lineWidth = Math.max(1.2, suitWidth * 0.006);
  for (let i = 0; i < 8; i += 1) {
    const y = torsoY + torsoHeight * (0.16 + i * 0.095);
    ctx.beginPath();
    ctx.moveTo(torsoX + suitWidth * 0.13, y);
    ctx.quadraticCurveTo(torsoX + suitWidth * 0.5, y + torsoHeight * 0.025, torsoX + suitWidth * 0.87, y);
    ctx.stroke();
  }

  const metal = ctx.createLinearGradient(torsoX, collarY, torsoX, collarY + torsoHeight * 0.16);
  metal.addColorStop(0, "#f8fbfc");
  metal.addColorStop(0.45, "#8999a5");
  metal.addColorStop(1, "#eff5f8");
  ctx.fillStyle = metal;
  roundedRect(torsoX + suitWidth * 0.33, collarY, suitWidth * 0.34, torsoHeight * 0.12, torsoHeight * 0.055);
  ctx.fill();
  ctx.strokeStyle = "rgba(43, 59, 72, 0.65)";
  ctx.lineWidth = Math.max(2, suitWidth * 0.014);
  ctx.stroke();

  ctx.restore();
}

function drawGlassHelmet(center, radius, theme) {
  const gradient = ctx.createRadialGradient(
    center.x - radius * 0.35,
    center.y - radius * 0.45,
    radius * 0.12,
    center.x,
    center.y,
    radius
  );

  gradient.addColorStop(0, "rgba(255, 255, 255, 0.58)");
  gradient.addColorStop(0.42, "rgba(202, 244, 255, 0.2)");
  gradient.addColorStop(1, "rgba(60, 130, 160, 0.16)");

  ctx.save();
  ctx.globalCompositeOperation = "source-over";
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();

  ctx.strokeStyle = "rgba(236, 243, 247, 0.98)";
  ctx.lineWidth = Math.max(6, radius * 0.1);
  ctx.stroke();

  ctx.strokeStyle = "rgba(42, 69, 92, 0.82)";
  ctx.lineWidth = Math.max(3, radius * 0.045);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 0.88, Math.PI * 0.08, Math.PI * 0.92);
  ctx.stroke();

  ctx.strokeStyle = "rgba(20, 35, 53, 0.26)";
  ctx.lineWidth = Math.max(2, radius * 0.028);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 0.72, Math.PI * 0.15, Math.PI * 0.85);
  ctx.stroke();

  ctx.strokeStyle = "rgba(255, 255, 255, 0.72)";
  ctx.lineWidth = Math.max(3, radius * 0.04);
  ctx.beginPath();
  ctx.arc(center.x - radius * 0.16, center.y - radius * 0.16, radius * 0.56, Math.PI * 1.05, Math.PI * 1.48);
  ctx.stroke();

  ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
  ctx.beginPath();
  ctx.ellipse(center.x - radius * 0.34, center.y - radius * 0.42, radius * 0.14, radius * 0.07, -0.45, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawNeckSeal(center, radius, theme) {
  ctx.save();
  const ringWidth = radius * 1.34;
  const ringHeight = radius * 0.38;
  const x = center.x - ringWidth / 2;
  const y = center.y - ringHeight * 0.18;

  roundedRect(x, y, ringWidth, ringHeight, ringHeight * 0.45);
  ctx.fillStyle = "#f4f7f9";
  ctx.fill();
  ctx.strokeStyle = "rgba(68, 86, 102, 0.66)";
  ctx.lineWidth = Math.max(3, radius * 0.05);
  ctx.stroke();

  ctx.strokeStyle = "#246ab8";
  ctx.lineWidth = Math.max(2, radius * 0.035);
  ctx.beginPath();
  ctx.moveTo(x + ringWidth * 0.18, y + ringHeight * 0.5);
  ctx.lineTo(x + ringWidth * 0.82, y + ringHeight * 0.5);
  ctx.stroke();
  ctx.restore();
}

function drawSuit(pose, width, height) {
  const theme = suitThemes.classic;
  const shoulderCenter = midpoint(pose.leftShoulder, pose.rightShoulder);
  const hipCenter = midpoint(pose.leftHip, pose.rightHip);
  const shoulderWidth = distance(pose.leftShoulder, pose.rightShoulder);
  const torsoHeight = Math.max(distance(shoulderCenter, hipCenter), height * 0.14);
  const suitWidth = shoulderWidth * 1.42;
  const headRadius = Math.max(shoulderWidth * 0.42, 48);
  const armWidth = Math.max(shoulderWidth * 0.22, 28);
  const legWidth = Math.max(shoulderWidth * 0.24, 32);

  ctx.save();
  ctx.globalAlpha = 0.96;

  drawLifeSupportPack({ x: shoulderCenter.x, y: shoulderCenter.y + torsoHeight * 0.18 }, suitWidth, torsoHeight, theme);

  capsule(pose.leftShoulder, pose.leftElbow, armWidth, theme.shell, theme.shade);
  capsule(pose.leftElbow, pose.leftWrist, armWidth * 0.92, theme.shell, theme.shade);
  capsule(pose.rightShoulder, pose.rightElbow, armWidth, theme.shell, theme.shade);
  capsule(pose.rightElbow, pose.rightWrist, armWidth * 0.92, theme.shell, theme.shade);
  capsule(pose.leftHip, pose.leftKnee, legWidth, theme.shell, theme.shade);
  capsule(pose.rightHip, pose.rightKnee, legWidth, theme.shell, theme.shade);
  capsule(pose.leftKnee, { x: pose.leftKnee.x, y: pose.leftKnee.y + legWidth * 1.15 }, legWidth * 0.82, theme.shell, theme.shade);
  capsule(pose.rightKnee, { x: pose.rightKnee.x, y: pose.rightKnee.y + legWidth * 1.15 }, legWidth * 0.82, theme.shell, theme.shade);

  drawShoulderPatch(pose.leftShoulder, armWidth, -1);
  drawShoulderPatch(pose.rightShoulder, armWidth, 1);
  bandOnLimb(pose.leftShoulder, pose.leftElbow, 0.48, armWidth * 0.22, armWidth, "#246ab8");
  bandOnLimb(pose.rightShoulder, pose.rightElbow, 0.48, armWidth * 0.22, armWidth, "#246ab8");
  bandOnLimb(pose.leftElbow, pose.leftWrist, 0.78, armWidth * 0.2, armWidth * 0.92, "#246ab8");
  bandOnLimb(pose.rightElbow, pose.rightWrist, 0.78, armWidth * 0.2, armWidth * 0.92, "#246ab8");
  bandOnLimb(pose.leftHip, pose.leftKnee, 0.7, legWidth * 0.2, legWidth, "#246ab8");
  bandOnLimb(pose.rightHip, pose.rightKnee, 0.7, legWidth * 0.2, legWidth, "#246ab8");

  drawGlove(pose.leftWrist, armWidth, theme);
  drawGlove(pose.rightWrist, armWidth, theme);
  drawBoot({ x: pose.leftKnee.x, y: pose.leftKnee.y + legWidth * 1.55 }, legWidth * 1.18, theme, -1);
  drawBoot({ x: pose.rightKnee.x, y: pose.rightKnee.y + legWidth * 1.55 }, legWidth * 1.18, theme, 1);

  const torsoX = shoulderCenter.x - suitWidth / 2;
  const torsoY = shoulderCenter.y - torsoHeight * 0.12;
  const torsoGradient = ctx.createLinearGradient(torsoX, torsoY, torsoX + suitWidth, torsoY + torsoHeight);
  torsoGradient.addColorStop(0, "#ffffff");
  torsoGradient.addColorStop(0.48, theme.shell);
  torsoGradient.addColorStop(1, "#c7d4dd");
  drawSoftShadow(() => roundedRect(torsoX, torsoY, suitWidth, torsoHeight * 1.16, suitWidth * 0.26), suitWidth * 0.06, 0.16);
  roundedRect(torsoX, torsoY, suitWidth, torsoHeight * 1.16, suitWidth * 0.26);
  ctx.fillStyle = torsoGradient;
  ctx.fill();
  ctx.strokeStyle = "rgba(74, 92, 108, 0.58)";
  ctx.lineWidth = Math.max(3, suitWidth * 0.022);
  ctx.stroke();
  drawSuitSeams(torsoX, torsoY, suitWidth, torsoHeight, theme);
  drawCanvasSuitHardware(torsoX, torsoY, suitWidth, torsoHeight);

  ctx.fillStyle = "rgba(255, 255, 255, 0.42)";
  roundedRect(torsoX + suitWidth * 0.1, torsoY + torsoHeight * 0.1, suitWidth * 0.18, torsoHeight * 0.76, 14);
  ctx.fill();
  drawFabricGrain(torsoX + suitWidth * 0.08, torsoY + torsoHeight * 0.08, suitWidth * 0.84, torsoHeight * 0.92, 0.34, 0.12);

  const chestCenter = {
    x: shoulderCenter.x,
    y: torsoY + torsoHeight * 0.39,
  };
  drawChestUnit(chestCenter, suitWidth, torsoHeight, theme);
  drawHose({ x: chestCenter.x - suitWidth * 0.18, y: chestCenter.y + torsoHeight * 0.1 }, pose.leftHip, suitWidth, -1);
  drawHose({ x: chestCenter.x + suitWidth * 0.18, y: chestCenter.y + torsoHeight * 0.1 }, pose.rightHip, suitWidth, 1);

  drawMissionPatch(torsoX + suitWidth * 0.75, torsoY + torsoHeight * 0.22, suitWidth * 0.08);

  const helmetY = pose.head.y - headRadius * 0.18;
  drawNeckSeal({ x: pose.head.x, y: helmetY + headRadius * 0.76 }, headRadius, theme);
  roundedRect(pose.head.x - headRadius * 0.72, helmetY + headRadius * 0.62, headRadius * 1.44, headRadius * 0.44, 16);
  ctx.fillStyle = theme.shell;
  ctx.fill();
  ctx.strokeStyle = "rgba(74, 92, 108, 0.68)";
  ctx.stroke();
  ctx.strokeStyle = "rgba(210, 224, 233, 0.8)";
  ctx.lineWidth = Math.max(4, headRadius * 0.06);
  ctx.beginPath();
  ctx.arc(pose.head.x, helmetY + headRadius * 0.74, headRadius * 0.72, Math.PI * 1.05, Math.PI * 1.95);
  ctx.stroke();

  drawGlassHelmet({ x: pose.head.x, y: helmetY }, headRadius, theme);

  ctx.globalAlpha = 1;
  ctx.restore();
}

function drawImagePart(part, x, y, drawWidth, drawHeight, options = {}) {
  if (!realSuitReady) return;

  ctx.save();
  ctx.globalAlpha = options.alpha ?? 0.98;
  if (options.shadow) {
    ctx.shadowColor = "rgba(10, 20, 32, 0.2)";
    ctx.shadowBlur = Math.max(6, drawWidth * 0.05);
    ctx.shadowOffsetY = Math.max(2, drawWidth * 0.012);
  }
  ctx.drawImage(realSuitImage, part.sx, part.sy, part.sw, part.sh, x, y, drawWidth, drawHeight);
  ctx.restore();
}

function drawImagePartBetween(part, start, end, drawWidth, options = {}) {
  if (!realSuitReady) return;

  const len = distance(start, end);
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const topOverlap = options.topOverlap ?? drawWidth * 0.16;
  const bottomOverlap = options.bottomOverlap ?? drawWidth * 0.18;
  const drawHeight = len + topOverlap + bottomOverlap;
  const anchorX = options.anchorX ?? 0.5;

  ctx.save();
  ctx.translate(start.x, start.y);
  ctx.rotate(angle - Math.PI / 2);
  ctx.globalAlpha = options.alpha ?? 0.98;
  if (options.shadow) {
    ctx.shadowColor = "rgba(10, 20, 32, 0.18)";
    ctx.shadowBlur = Math.max(6, drawWidth * 0.05);
    ctx.shadowOffsetY = Math.max(2, drawWidth * 0.012);
  }
  ctx.drawImage(
    realSuitImage,
    part.sx,
    part.sy,
    part.sw,
    part.sh,
    -drawWidth * anchorX,
    -topOverlap,
    drawWidth,
    drawHeight
  );
  ctx.restore();
}

function extendPoint(from, to, amount) {
  const len = distance(from, to) || 1;
  return {
    x: to.x + ((to.x - from.x) / len) * amount,
    y: to.y + ((to.y - from.y) / len) * amount,
    score: to.score,
  };
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function partSuitReady() {
  return Object.values(partSuitImages).every((image) => image.complete && image.naturalWidth > 0);
}

function drawPartAsset(image, x, y, drawWidth, drawHeight, options = {}) {
  if (!image.complete || !image.naturalWidth) return;

  ctx.save();
  ctx.globalAlpha = options.alpha ?? 0.98;
  if (options.shadow) {
    ctx.shadowColor = "rgba(10, 20, 32, 0.2)";
    ctx.shadowBlur = Math.max(6, drawWidth * 0.05);
    ctx.shadowOffsetY = Math.max(2, drawWidth * 0.012);
  }
  ctx.drawImage(image, x, y, drawWidth, drawHeight);
  ctx.restore();
}

function drawPartAssetBetween(image, start, end, drawWidth, options = {}) {
  if (!image.complete || !image.naturalWidth) return;

  const len = distance(start, end);
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const topOverlap = options.topOverlap ?? drawWidth * 0.12;
  const bottomOverlap = options.bottomOverlap ?? drawWidth * 0.16;
  const drawHeight = len + topOverlap + bottomOverlap;
  const anchorX = options.anchorX ?? 0.5;

  ctx.save();
  ctx.translate(start.x, start.y);
  ctx.rotate(angle - Math.PI / 2);
  ctx.globalAlpha = options.alpha ?? 0.98;
  if (options.shadow) {
    ctx.shadowColor = "rgba(10, 20, 32, 0.18)";
    ctx.shadowBlur = Math.max(6, drawWidth * 0.05);
    ctx.shadowOffsetY = Math.max(2, drawWidth * 0.012);
  }
  ctx.drawImage(image, -drawWidth * anchorX, -topOverlap, drawWidth, drawHeight);
  ctx.restore();
}

function drawPartSuit(pose, width, height) {
  const theme = suitThemes.classic;
  const shoulderCenter = midpoint(pose.leftShoulder, pose.rightShoulder);
  const hipCenter = midpoint(pose.leftHip, pose.rightHip);
  const shoulderWidth = distance(pose.leftShoulder, pose.rightShoulder);
  const torsoHeight = Math.max(distance(shoulderCenter, hipCenter), height * 0.14);
  const body = partSuitImages.body;
  const leftArm = partSuitImages.leftArm;
  const rightArm = partSuitImages.rightArm;
  const leftLeg = partSuitImages.leftLeg;
  const rightLeg = partSuitImages.rightLeg;
  const stableShoulderWidth = clamp(shoulderWidth, width * 0.18, width * 0.42);
  const bodyWidth = clamp(stableShoulderWidth * 1.7, width * 0.34, width * 0.66);
  const bodyHeight = bodyWidth * (body.naturalHeight / body.naturalWidth);
  const bodyX = shoulderCenter.x - bodyWidth / 2;
  const bodyY = shoulderCenter.y - bodyHeight * 0.33;
  const armWidth = clamp(bodyWidth * 0.2, 52, 112);
  const legWidth = clamp(bodyWidth * 0.2, 52, 110);
  const leftWrist = pose.leftWrist?.score > 0.25 ? pose.leftWrist : extendPoint(pose.leftShoulder, pose.leftElbow, armWidth * 2.2);
  const rightWrist = pose.rightWrist?.score > 0.25 ? pose.rightWrist : extendPoint(pose.rightShoulder, pose.rightElbow, armWidth * 2.2);
  const leftWristEnd = extendPoint(pose.leftElbow, leftWrist, armWidth * 0.46);
  const rightWristEnd = extendPoint(pose.rightElbow, rightWrist, armWidth * 0.46);
  const leftAnkle = pose.leftAnkle?.score > 0.2 ? pose.leftAnkle : extendPoint(pose.leftHip, pose.leftKnee, legWidth * 1.9);
  const rightAnkle = pose.rightAnkle?.score > 0.2 ? pose.rightAnkle : extendPoint(pose.rightHip, pose.rightKnee, legWidth * 1.9);
  const leftFootEnd = extendPoint(pose.leftKnee, leftAnkle, legWidth * 0.34);
  const rightFootEnd = extendPoint(pose.rightKnee, rightAnkle, legWidth * 0.34);
  const faceToShoulder = Math.max(distance(pose.head, shoulderCenter), height * 0.1);
  const headRadius = clamp(faceToShoulder * 0.46, 48, bodyWidth * 0.36);
  const helmetCenter = {
    x: pose.head.x,
    y: pose.head.y - headRadius * 0.06,
  };

  drawPartAssetBetween(leftLeg, pose.leftHip, leftFootEnd, legWidth, {
    shadow: true,
    topOverlap: legWidth * 0.32,
    bottomOverlap: legWidth * 0.18,
    anchorX: 0.48,
  });
  drawPartAssetBetween(rightLeg, pose.rightHip, rightFootEnd, legWidth, {
    shadow: true,
    topOverlap: legWidth * 0.32,
    bottomOverlap: legWidth * 0.18,
    anchorX: 0.52,
  });
  drawPartAsset(body, bodyX, bodyY, bodyWidth, Math.max(bodyHeight, torsoHeight * 1.6), { shadow: true });
  drawPartAssetBetween(leftArm, pose.leftShoulder, leftWristEnd, armWidth, {
    shadow: true,
    topOverlap: armWidth * 0.18,
    bottomOverlap: armWidth * 0.16,
    anchorX: 0.5,
    alpha: 0.96,
  });
  drawPartAssetBetween(rightArm, pose.rightShoulder, rightWristEnd, armWidth, {
    shadow: true,
    topOverlap: armWidth * 0.18,
    bottomOverlap: armWidth * 0.16,
    anchorX: 0.5,
    alpha: 0.96,
  });

  drawNeckSeal({ x: pose.head.x, y: helmetCenter.y + headRadius * 0.77 }, headRadius, theme);
  drawGlassHelmet(helmetCenter, headRadius, theme);
}

function drawSegmentedRealSuit(pose, width, height) {
  const theme = suitThemes.classic;
  const shoulderCenter = midpoint(pose.leftShoulder, pose.rightShoulder);
  const hipCenter = midpoint(pose.leftHip, pose.rightHip);
  const shoulderWidth = distance(pose.leftShoulder, pose.rightShoulder);
  const torsoHeight = Math.max(distance(shoulderCenter, hipCenter), height * 0.14);
  const torsoWidth = Math.min(width * 0.88, shoulderWidth * 1.95);
  const torsoDrawHeight = Math.max(torsoHeight * 1.72, torsoWidth * (suitPart.torso.sh / suitPart.torso.sw));
  const torsoX = shoulderCenter.x - torsoWidth / 2;
  const torsoY = pose.neck.y - torsoDrawHeight * 0.12;
  const armWidth = Math.max(shoulderWidth * 0.48, 54);
  const forearmWidth = armWidth * 0.92;
  const thighWidth = Math.max(shoulderWidth * 0.52, 58);
  const shinWidth = thighWidth * 0.9;
  const leftAnkle = pose.leftAnkle?.score > 0.2 ? pose.leftAnkle : extendPoint(pose.leftHip, pose.leftKnee, shinWidth * 1.35);
  const rightAnkle = pose.rightAnkle?.score > 0.2 ? pose.rightAnkle : extendPoint(pose.rightHip, pose.rightKnee, shinWidth * 1.35);
  const leftHandEnd = extendPoint(pose.leftElbow, pose.leftWrist, forearmWidth * 0.62);
  const rightHandEnd = extendPoint(pose.rightElbow, pose.rightWrist, forearmWidth * 0.62);
  const headRadius = Math.max(shoulderWidth * 0.43, 48);
  const helmetCenter = {
    x: pose.head.x,
    y: pose.head.y - headRadius * 0.16,
  };

  drawImagePartBetween(suitPart.leftUpperArm, pose.leftShoulder, pose.leftElbow, armWidth, {
    shadow: true,
    anchorX: 0.48,
  });
  drawImagePartBetween(suitPart.rightUpperArm, pose.rightShoulder, pose.rightElbow, armWidth, {
    shadow: true,
    anchorX: 0.52,
  });
  drawImagePartBetween(suitPart.leftThigh, pose.leftHip, pose.leftKnee, thighWidth, {
    shadow: true,
    topOverlap: thighWidth * 0.2,
    bottomOverlap: thighWidth * 0.12,
  });
  drawImagePartBetween(suitPart.rightThigh, pose.rightHip, pose.rightKnee, thighWidth, {
    shadow: true,
    topOverlap: thighWidth * 0.2,
    bottomOverlap: thighWidth * 0.12,
  });

  drawImagePart(suitPart.torso, torsoX, torsoY, torsoWidth, torsoDrawHeight, { shadow: true });

  drawImagePartBetween(suitPart.leftForearm, pose.leftElbow, pose.leftWrist, forearmWidth, {
    shadow: true,
    topOverlap: forearmWidth * 0.16,
    bottomOverlap: forearmWidth * 0.25,
  });
  drawImagePartBetween(suitPart.rightForearm, pose.rightElbow, pose.rightWrist, forearmWidth, {
    shadow: true,
    topOverlap: forearmWidth * 0.16,
    bottomOverlap: forearmWidth * 0.25,
  });
  drawImagePartBetween(suitPart.leftShin, pose.leftKnee, leftAnkle, shinWidth, {
    shadow: true,
    topOverlap: shinWidth * 0.12,
    bottomOverlap: shinWidth * 0.08,
  });
  drawImagePartBetween(suitPart.rightShin, pose.rightKnee, rightAnkle, shinWidth, {
    shadow: true,
    topOverlap: shinWidth * 0.12,
    bottomOverlap: shinWidth * 0.08,
  });

  drawImagePartBetween(suitPart.leftGlove, pose.leftWrist, leftHandEnd, forearmWidth * 1.05, {
    shadow: true,
    topOverlap: forearmWidth * 0.18,
    bottomOverlap: forearmWidth * 0.28,
    anchorX: 0.48,
  });
  drawImagePartBetween(suitPart.rightGlove, pose.rightWrist, rightHandEnd, forearmWidth * 1.05, {
    shadow: true,
    topOverlap: forearmWidth * 0.18,
    bottomOverlap: forearmWidth * 0.28,
    anchorX: 0.52,
  });

  const leftFootEnd = extendPoint(pose.leftKnee, leftAnkle, shinWidth * 0.58);
  const rightFootEnd = extendPoint(pose.rightKnee, rightAnkle, shinWidth * 0.58);
  drawImagePartBetween(suitPart.leftBoot, leftAnkle, leftFootEnd, shinWidth * 1.42, {
    shadow: true,
    topOverlap: shinWidth * 0.2,
    bottomOverlap: shinWidth * 0.42,
    anchorX: 0.54,
  });
  drawImagePartBetween(suitPart.rightBoot, rightAnkle, rightFootEnd, shinWidth * 1.42, {
    shadow: true,
    topOverlap: shinWidth * 0.2,
    bottomOverlap: shinWidth * 0.42,
    anchorX: 0.46,
  });

  drawNeckSeal({ x: pose.head.x, y: helmetCenter.y + headRadius * 0.77 }, headRadius, theme);
  drawGlassHelmet(helmetCenter, headRadius, theme);
}

function drawRealSuit(pose, width, height) {
  const theme = suitThemes.classic;
  const shoulderCenter = midpoint(pose.leftShoulder, pose.rightShoulder);
  const hipCenter = midpoint(pose.leftHip, pose.rightHip);
  const shoulderWidth = distance(pose.leftShoulder, pose.rightShoulder);
  const torsoHeight = Math.max(distance(shoulderCenter, hipCenter), height * 0.14);
  const imageRatio = realSuitImage.height / realSuitImage.width;
  const suitWidth = Math.min(width * 0.92, Math.max(shoulderWidth * 2.45, torsoHeight * 1.38));
  const suitHeight = suitWidth * imageRatio;
  const suitX = shoulderCenter.x - suitWidth / 2;
  const suitY = pose.neck.y - suitHeight * 0.095;
  const headRadius = Math.max(shoulderWidth * 0.43, 48);
  const helmetCenter = {
    x: pose.head.x,
    y: pose.head.y - headRadius * 0.16,
  };

  ctx.save();
  ctx.globalAlpha = 0.98;
  ctx.shadowColor = "rgba(10, 20, 32, 0.22)";
  ctx.shadowBlur = Math.max(8, suitWidth * 0.045);
  ctx.shadowOffsetY = Math.max(3, suitWidth * 0.012);
  ctx.drawImage(realSuitImage, suitX, suitY, suitWidth, suitHeight);
  ctx.restore();

  drawNeckSeal({ x: pose.head.x, y: helmetCenter.y + headRadius * 0.77 }, headRadius, theme);
  drawGlassHelmet(helmetCenter, headRadius, theme);
}

function clearStage() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
}

function render() {
  resizeCanvas();
  clearStage();

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  let pose = null;
  const now = performance.now();

  if (previewMode) {
    pose = previewPose(width, height);
  } else if (poseLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = poseLandmarker.detectForVideo(video, now);
    if (result.landmarks?.[0]) {
      pose = poseFromLandmarks(result.landmarks[0], width, height);
    }
  }

  if (pose) {
    smoothedPose = smoothPose(smoothedPose, pose);
    lastGoodPoseAt = now;
  }

  const poseToDraw = smoothedPose && now - lastGoodPoseAt < poseHoldMs ? smoothedPose : null;

  if (!poseToDraw) {
    if (hasPose) {
      setStatus("未检测到完整人体，请露出上半身", "idle");
      hasPose = false;
      smoothedPose = null;
    }
  } else {
    if (!hasPose) {
      setStatus("已检测到人体，正在贴合宇航服", "live");
      hasPose = true;
    }
    drawSuit(poseToDraw, width, height);
  }

  if (running) {
    requestAnimationFrame(render);
  }
}

captureButton.addEventListener("click", () => {
  if (!hasPose) {
    setStatus("请先让摄像头检测到人体", "idle");
    return;
  }

  setStatus("已生成当前宇航服造型", "live");
  document.body.classList.add("flash");
  window.setTimeout(() => document.body.classList.remove("flash"), 180);
});

startButton.addEventListener("click", startCamera);
switchCameraButton.addEventListener("click", async () => {
  cameraFacingMode = cameraFacingMode === "user" ? "environment" : "user";
  stage.classList.toggle("is-rear-camera", cameraFacingMode === "environment");

  if (currentStream || running) {
    startButton.disabled = true;
    setStatus("正在切换摄像头", "live");
    await startCamera();
  } else {
    setStatus(cameraFacingMode === "environment" ? "已选择后置摄像头" : "已选择前置摄像头", "idle");
  }
});
shareButton.addEventListener("click", () => qrPanel.classList.remove("is-hidden"));
closeQrButton.addEventListener("click", () => qrPanel.classList.add("is-hidden"));
window.addEventListener("resize", resizeCanvas);

setupShare();
resizeCanvas();
if (previewMode) {
  qrPanel.classList.add("is-hidden");
  setStatus("宇航服效果预览", "live");
}
render();
