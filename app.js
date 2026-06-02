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

let poseLandmarker;
let running = false;
let lastVideoTime = -1;
let hasPose = false;
let smoothedPose = null;
let lastGoodPoseAt = 0;
let lastStatusText = "";

const poseHoldMs = 900;
const smoothing = 0.72;

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

async function startCamera() {
  try {
    const model = await loadPoseModel();

    if (!model) {
      return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    video.srcObject = stream;
    await video.play();

    running = true;
    startLabel.textContent = "AR 识别中";
    startButton.disabled = true;
    setStatus("请站入画面，系统会自动贴合宇航服", "live");
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
  return {
    x: rect.x + (1 - point.x) * rect.width,
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

function capsule(start, end, width, color, shade) {
  const angle = Math.atan2(end.y - start.y, end.x - start.x);
  const len = distance(start, end);

  ctx.save();
  ctx.translate(start.x, start.y);
  ctx.rotate(angle);
  const gradient = ctx.createLinearGradient(0, -width / 2, 0, width / 2);
  gradient.addColorStop(0, "#ffffff");
  gradient.addColorStop(0.52, color);
  gradient.addColorStop(1, shade);
  ctx.fillStyle = gradient;
  roundedRect(0, -width / 2, len, width, width / 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(86, 110, 129, 0.52)";
  ctx.lineWidth = Math.max(2, width * 0.045);
  ctx.stroke();

  ctx.strokeStyle = "rgba(118, 139, 155, 0.48)";
  ctx.lineWidth = Math.max(1.5, width * 0.032);
  for (let i = 0.24; i <= 0.76; i += 0.26) {
    ctx.beginPath();
    ctx.moveTo(len * i, -width * 0.38);
    ctx.quadraticCurveTo(len * (i + 0.03), 0, len * i, width * 0.38);
    ctx.stroke();
  }
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
  ctx.fillStyle = "#e6edf3";
  roundedRect(-width * 0.5, -width * 0.32, width * 0.82, width * 0.78, width * 0.2);
  ctx.fill();
  ctx.fillStyle = "#4f6070";
  roundedRect(-width * 0.42, width * 0.28, width * 0.92, width * 0.28, width * 0.1);
  ctx.fill();
  ctx.strokeStyle = "rgba(35, 49, 69, 0.5)";
  ctx.lineWidth = Math.max(2, width * 0.05);
  ctx.stroke();
  ctx.restore();
}

function drawGlove(point, width, theme) {
  ctx.save();
  ctx.fillStyle = "#eef4f8";
  ctx.beginPath();
  ctx.arc(point.x, point.y, width * 0.48, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(35, 49, 69, 0.58)";
  ctx.lineWidth = Math.max(2, width * 0.05);
  ctx.stroke();
  ctx.fillStyle = "#9fb0bd";
  roundedRect(point.x - width * 0.35, point.y - width * 0.12, width * 0.7, width * 0.24, width * 0.1);
  ctx.fill();
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
  roundedRect(center.x - packWidth / 2, center.y - packHeight * 0.12, packWidth, packHeight, packWidth * 0.22);
  ctx.fillStyle = "rgba(182, 194, 204, 0.72)";
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
  ctx.restore();
}

function drawChestUnit(center, suitWidth, torsoHeight, theme) {
  const unitWidth = suitWidth * 0.48;
  const unitHeight = torsoHeight * 0.34;
  const x = center.x - unitWidth / 2;
  const y = center.y - unitHeight / 2;

  ctx.save();
  roundedRect(x, y, unitWidth, unitHeight, unitWidth * 0.12);
  ctx.fillStyle = "#eef3f7";
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
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = gradient;
  ctx.fill();
  ctx.strokeStyle = "rgba(231, 249, 255, 0.96)";
  ctx.lineWidth = Math.max(4, radius * 0.07);
  ctx.stroke();

  ctx.strokeStyle = "rgba(52, 214, 197, 0.54)";
  ctx.lineWidth = Math.max(2, radius * 0.026);
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius * 0.82, Math.PI * 0.08, Math.PI * 0.92);
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
  roundedRect(torsoX, torsoY, suitWidth, torsoHeight * 1.16, suitWidth * 0.26);
  ctx.fillStyle = torsoGradient;
  ctx.fill();
  ctx.strokeStyle = "rgba(74, 92, 108, 0.58)";
  ctx.lineWidth = Math.max(3, suitWidth * 0.022);
  ctx.stroke();
  drawSuitSeams(torsoX, torsoY, suitWidth, torsoHeight, theme);

  ctx.fillStyle = "rgba(255, 255, 255, 0.42)";
  roundedRect(torsoX + suitWidth * 0.1, torsoY + torsoHeight * 0.1, suitWidth * 0.18, torsoHeight * 0.76, 14);
  ctx.fill();

  const chestCenter = {
    x: shoulderCenter.x,
    y: torsoY + torsoHeight * 0.39,
  };
  drawChestUnit(chestCenter, suitWidth, torsoHeight, theme);
  drawHose({ x: chestCenter.x - suitWidth * 0.18, y: chestCenter.y + torsoHeight * 0.1 }, pose.leftHip, suitWidth, -1);
  drawHose({ x: chestCenter.x + suitWidth * 0.18, y: chestCenter.y + torsoHeight * 0.1 }, pose.rightHip, suitWidth, 1);

  drawMissionPatch(torsoX + suitWidth * 0.75, torsoY + torsoHeight * 0.22, suitWidth * 0.08);

  const helmetY = pose.head.y - headRadius * 0.18;
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

  if (poseLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
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
shareButton.addEventListener("click", () => qrPanel.classList.remove("is-hidden"));
closeQrButton.addEventListener("click", () => qrPanel.classList.add("is-hidden"));
window.addEventListener("resize", resizeCanvas);

setupShare();
resizeCanvas();
render();
