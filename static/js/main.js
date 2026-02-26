"use strict";

// ── DOM refs ───────────────────────────────────────────────────────────────
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const imageUrl = document.getElementById("imageUrl");
const clearUrlBtn = document.getElementById("clearUrlBtn");
const dzIdle = document.getElementById("dzIdle");
const dzPreview = document.getElementById("dzPreview");
const previewIcon = document.getElementById("previewIcon");
const previewName = document.getElementById("previewName");
const previewSize = document.getElementById("previewSize");
const browseLink = document.getElementById("browseLink");
const clearBtn = document.getElementById("clearBtn");
const generateBtn = document.getElementById("generateBtn");
const errorBanner = document.getElementById("errorBanner");
const errorMsg = document.getElementById("errorMsg");
const mediaTitle = document.getElementById("mediaTitle");

// Image elements
const resultInput = document.getElementById("resultInput");
const rgbPlaceholder = document.getElementById("rgbPlaceholder");
const resultOutput = document.getElementById("resultOutput");
const irPlaceholder = document.getElementById("irPlaceholder");
const irLoading = document.getElementById("irLoading");
const downloadBtn = document.getElementById("downloadBtn");

// Video elements
const videoInput = document.getElementById("videoInput");
const videoOutput = document.getElementById("videoOutput");
const videoInputLoading = document.getElementById("videoInputLoading");
const videoInputStage = document.getElementById("videoInputStage");
const videoProgress = document.getElementById("videoProgress");
const progressBar = document.getElementById("progressBar");
const progressCounter = document.getElementById("progressCounter");
const progressStage = document.getElementById("progressStage");

// Detection results
const detCountBadge = document.getElementById("detCountBadge");
const detHuman = document.getElementById("detHuman");
const detRaw = document.getElementById("detRaw");
const detRawPre = document.getElementById("detRawPre");
const detectionWarning = document.getElementById("detectionWarning");
const detectionWarningMsg = document.getElementById("detectionWarningMsg");
const rawToggleTrack = document.getElementById("rawToggleTrack");
const rawToggleThumb = document.getElementById("rawToggleThumb");

// ── State ──────────────────────────────────────────────────────────────────
let selectedFile = null;
let isVideoMode = false;
let rawEnabled = false;
let activeJobId = null;
let activeSSE = null;

const VIDEO_EXTS = new Set([".mp4", ".avi", ".mov", ".mkv", ".webm"]);

// ── Helpers ────────────────────────────────────────────────────────────────

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorMsg.textContent = "";
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function toggleRaw() {
  rawEnabled = !rawEnabled;
  if (rawEnabled) {
    rawToggleTrack.style.backgroundColor = "#7c3aed";
    rawToggleThumb.style.transform = "translateX(16px)";
    detHuman.hidden = true;
    detRaw.hidden = false;
  } else {
    rawToggleTrack.style.backgroundColor = "";
    rawToggleThumb.style.transform = "";
    detHuman.hidden = false;
    detRaw.hidden = true;
  }
}

// ── Media panel helpers ───────────────────────────────────────────────────

const IMG_ICON_SVG = `<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
  <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909M3 12V6.75A2.25 2.25 0 015.25 4.5h13.5A2.25 2.25 0 0121 6.75V17.25A2.25 2.25 0 0118.75 19.5H5.25A2.25 2.25 0 013 17.25V12z"/>
</svg>`;

const VID_ICON_SVG = `<svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
  <path stroke-linecap="round" stroke-linejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"/>
</svg>`;

function setImageModeUI() {
  mediaTitle.textContent = "Images";
  generateBtn.textContent = "Generate IR Image";
  // Show image elements, hide video
  resultInput.hidden = true;
  videoInput.hidden = true;
  videoInputLoading.hidden = true;
  resultOutput.hidden = true;
  videoOutput.hidden = true;
  videoProgress.hidden = true;
  rgbPlaceholder.hidden = false;
  irPlaceholder.hidden = false;
  irLoading.hidden = true;
}

function setVideoModeUI() {
  mediaTitle.textContent = "Videos";
  generateBtn.textContent = "Process Video";
  // Show video loading state, hide images
  resultInput.hidden = true;
  resultOutput.hidden = true;
  irLoading.hidden = true;
  rgbPlaceholder.hidden = false;
  irPlaceholder.hidden = false;
  videoInput.hidden = true;
  videoOutput.hidden = true;
  videoInputLoading.hidden = true;
  videoProgress.hidden = true;
}

function updateProgress(phase, frame, total) {
  progressCounter.textContent = `${frame} / ${total}`;
  const pct = total > 0 ? Math.round((frame / total) * 100) : 0;
  progressBar.style.width = pct + "%";
}

// ── Upload handling ────────────────────────────────────────────────────────

function applyPreview(file) {
  const ext = "." + file.name.split(".").pop().toLowerCase();
  isVideoMode = VIDEO_EXTS.has(ext) || file.type.startsWith("video/");

  previewIcon.innerHTML = isVideoMode ? VID_ICON_SVG : IMG_ICON_SVG;
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  dzIdle.hidden = true;
  dzPreview.hidden = false;

  if (isVideoMode) {
    setVideoModeUI();
    // Show local video preview in input panel
    const url = URL.createObjectURL(file);
    videoInput.src = url;
    videoInput.hidden = false;
    rgbPlaceholder.hidden = true;
  } else {
    setImageModeUI();
    const url = URL.createObjectURL(file);
    resultInput.src = url;
    resultInput.hidden = false;
    rgbPlaceholder.hidden = true;
  }

  generateBtn.disabled = false;
  hideError();
}

function clearUpload() {
  // Stop any active SSE
  if (activeSSE) {
    activeSSE.close();
    activeSSE = null;
  }
  if (activeJobId) {
    fetch(`/video_cleanup/${activeJobId}`, { method: "DELETE" }).catch(() => { });
    activeJobId = null;
  }

  selectedFile = null;
  isVideoMode = false;
  fileInput.value = "";
  imageUrl.value = "";

  imageUrl.disabled = false;
  imageUrl.classList.remove("opacity-50", "cursor-not-allowed");
  clearUrlBtn.hidden = true;

  dropZone.style.pointerEvents = "auto";
  dropZone.style.opacity = "1";

  previewIcon.innerHTML = "";
  dzPreview.hidden = true;
  dzIdle.hidden = false;

  // Reset image panels
  resultInput.src = "";
  resultInput.hidden = true;
  resultOutput.src = "";
  resultOutput.hidden = true;
  irLoading.hidden = true;
  rgbPlaceholder.hidden = false;
  irPlaceholder.hidden = false;

  // Reset video panels
  videoInput.src = "";
  videoInput.hidden = true;
  videoOutput.src = "";
  videoOutput.hidden = true;
  videoInputLoading.hidden = true;
  videoProgress.hidden = true;
  progressBar.style.width = "0%";

  downloadBtn.hidden = true;
  generateBtn.disabled = true;
  mediaTitle.textContent = "Images";
  generateBtn.textContent = "Generate IR Image";

  // Reset detection panel
  detCountBadge.textContent = "0";
  detHuman.innerHTML = '<p style="font-size:14px;color:#9ca3af;text-align:center;padding:16px 0;">No detections yet. Generate an IR image to see results.</p>';
  detRawPre.textContent = "No data";
  detectionWarning.hidden = true;

  hideError();
}

function handleFile(file) {
  if (!file) return;
  const ext = "." + file.name.split(".").pop().toLowerCase();
  const isVid = VIDEO_EXTS.has(ext) || file.type.startsWith("video/");

  if (!isVid && file.size > 16 * 1024 * 1024) {
    showError("Image too large. Maximum size is 16 MB.");
    return;
  }
  if (isVid && file.size > 200 * 1024 * 1024) {
    showError("Video too large. Maximum size is 200 MB.");
    return;
  }

  selectedFile = file;

  imageUrl.disabled = true;
  imageUrl.value = "";
  imageUrl.classList.add("opacity-50", "cursor-not-allowed");
  clearUrlBtn.hidden = true;

  applyPreview(file);
}

function handleUrlInput() {
  const url = imageUrl.value.trim();
  if (url.length > 0) {
    clearUrlBtn.hidden = false;
    dropZone.style.pointerEvents = "none";
    dropZone.style.opacity = "0.5";
    generateBtn.disabled = false;
    selectedFile = null;
    fileInput.value = "";
    isVideoMode = false;
    hideError();
  } else {
    clearUrlBtn.hidden = true;
    dropZone.style.pointerEvents = "auto";
    dropZone.style.opacity = "1";
    generateBtn.disabled = true;
  }
}

// ── Event listeners ────────────────────────────────────────────────────────

browseLink.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", () => { if (!dzIdle.hidden) fileInput.click(); });
dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
clearBtn.addEventListener("click", (e) => { e.stopPropagation(); clearUpload(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", (e) => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove("drag-over"); });
dropZone.addEventListener("drop", (e) => { e.preventDefault(); dropZone.classList.remove("drag-over"); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => e.preventDefault());

imageUrl.addEventListener("input", handleUrlInput);
clearUrlBtn.addEventListener("click", () => { imageUrl.value = ""; handleUrlInput(); });

// ── Detection rendering ────────────────────────────────────────────────────

const DET_COLORS = {
  person: "#7c3aed",
  car: "#2563eb",
  bicycle: "#059669",
  dog: "#d97706",
  truck: "#dc2626",
  bus: "#db2777",
};
const DET_DEFAULT = "#6b7280";

function getDetColor(cls) {
  return DET_COLORS[cls.toLowerCase()] || DET_DEFAULT;
}

function renderDetections(data) {
  const predictions = data.predictions || [];
  const count = data.detection_count || 0;
  const errMsg = data.detection_error || null;

  detCountBadge.textContent = String(count);
  detRawPre.textContent = JSON.stringify(predictions, null, 2);

  if (errMsg) {
    detectionWarningMsg.textContent = errMsg;
    detectionWarning.hidden = false;
  } else {
    detectionWarning.hidden = true;
  }

  if (predictions.length === 0) {
    detHuman.innerHTML = '<p style="font-size:14px;color:#9ca3af;text-align:center;padding:16px 0;">No objects detected in the generated IR image.</p>';
    return;
  }

  const classCounts = {};
  predictions.forEach(function (p) { classCounts[p.class_name] = (classCounts[p.class_name] || 0) + 1; });

  let html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;">';
  Object.entries(classCounts).forEach(function ([cls, cnt]) {
    var c = getDetColor(cls);
    html += '<span style="font-size:12px;font-weight:700;padding:2px 10px;border-radius:4px;border:1px solid ' + c + '44;background:' + c + '11;color:' + c + ';text-transform:uppercase;letter-spacing:0.04em;">' + cls + ' &times;' + cnt + '</span>';
  });
  html += '</div>';

  predictions.forEach(function (pred) {
    var c = getDetColor(pred.class_name);
    var conf = Math.round(pred.confidence * 100);
    var w = pred.x2 - pred.x1;
    var h = pred.y2 - pred.y1;
    html +=
      '<div style="background:#faf9ff;border:1px solid #ede9fe;border-radius:6px;padding:10px 12px;margin-bottom:8px;">' +
      '<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px;">' +
      '<span style="font-size:16px;font-weight:600;color:' + c + ';">' + pred.class_name + '</span>' +
      '<span style="font-size:14px;color:#6b7280;font-weight:500;">' + conf + '%</span>' +
      '</div>' +
      '<div style="height:4px;background:#ede9fe;border-radius:4px;overflow:hidden;margin-bottom:7px;">' +
      '<div style="height:100%;width:' + conf + '%;background:' + c + ';border-radius:4px;transition:width 0.4s ease;"></div>' +
      '</div>' +
      '<div style="font-size:12px;color:#9ca3af;font-family:Consolas,monospace;">[' + pred.x1 + ', ' + pred.y1 + '] &rarr; [' + pred.x2 + ', ' + pred.y2 + '] &nbsp;&middot;&nbsp; ' + w + '&times;' + h + 'px</div>' +
      '</div>';
  });

  detHuman.innerHTML = html;
}

// ── Image generate ────────────────────────────────────────────────────────

async function generateImage() {
  const urlVal = imageUrl.value.trim();
  if (!selectedFile && urlVal.length === 0) return;

  hideError();
  generateBtn.disabled = true;
  irPlaceholder.hidden = true;
  irLoading.hidden = false;
  resultOutput.hidden = true;
  downloadBtn.hidden = true;

  if (urlVal.length > 0) {
    resultInput.hidden = true;
    rgbPlaceholder.hidden = false;
  }

  const formData = new FormData();
  if (selectedFile) {
    formData.append("image", selectedFile);
  } else {
    formData.append("image_url", urlVal);
  }

  const detectionCheckbox = document.getElementById("detectionCheckbox");
  formData.append("enable_detection", detectionCheckbox && detectionCheckbox.checked ? "true" : "false");

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || ("Server error (" + response.status + ")"));
      irPlaceholder.hidden = false;
      irLoading.hidden = true;
      return;
    }

    resultInput.src = "data:image/png;base64," + data.input;
    resultInput.hidden = false;
    rgbPlaceholder.hidden = true;

    const outputSrc = "data:image/png;base64," + data.output;
    resultOutput.src = outputSrc;
    resultOutput.hidden = false;
    irLoading.hidden = true;

    downloadBtn.href = outputSrc;
    downloadBtn.download = selectedFile
      ? "ir_" + selectedFile.name.replace(/\.[^.]+$/, "") + ".png"
      : "ir_output.png";
    downloadBtn.hidden = false;

    renderDetections(data);

  } catch (err) {
    showError("Request failed: " + err.message);
    irPlaceholder.hidden = false;
    irLoading.hidden = true;
  } finally {
    generateBtn.disabled = false;
  }
}

// ── Video generate ────────────────────────────────────────────────────────

async function generateVideo() {
  if (!selectedFile) return;

  hideError();
  generateBtn.disabled = true;
  downloadBtn.hidden = true;

  // Show loading spinners — hide the local video preview while processing
  videoInput.hidden = true;
  videoOutput.hidden = true;
  irPlaceholder.hidden = false;
  videoInputLoading.hidden = false;
  videoInputStage.textContent = "Uploading video...";
  videoProgress.hidden = false;
  progressBar.style.width = "0%";
  progressStage.textContent = "Uploading...";
  progressCounter.textContent = "0 / 0";

  // Detection in videos is applied per-frame
  const detectionCheckbox = document.getElementById("detectionCheckbox");
  const enableDet = detectionCheckbox && detectionCheckbox.checked ? "true" : "false";

  // POST the video
  const formData = new FormData();
  formData.append("video", selectedFile);
  formData.append("enable_detection", enableDet);

  let jobId;
  try {
    const res = await fetch("/predict_video", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || "Upload failed");
      resetVideoLoadingUI();
      generateBtn.disabled = false;
      return;
    }
    jobId = data.job_id;
    activeJobId = jobId;
  } catch (err) {
    showError("Upload failed: " + err.message);
    resetVideoLoadingUI();
    generateBtn.disabled = false;
    return;
  }

  // Listen for SSE progress
  const evtSource = new EventSource(`/video_status/${jobId}`);
  activeSSE = evtSource;

  evtSource.onmessage = function (e) {
    const event = JSON.parse(e.data);

    if (event.type === "stage") {
      progressStage.textContent = event.stage;
      videoInputStage.textContent = event.stage;

    } else if (event.type === "total") {
      progressCounter.textContent = `0 / ${event.total}`;

    } else if (event.type === "progress") {
      updateProgress(event.phase, event.frame, event.total);
      // Adjust bar offset based on phase
      let baseOffset = 0;
      if (event.phase === "detect") baseOffset = 50;
      const pct = baseOffset + Math.round((event.frame / event.total) * 50);
      progressBar.style.width = Math.min(pct, 99) + "%";

    } else if (event.type === "done") {
      evtSource.close();
      activeSSE = null;
      progressBar.style.width = "100%";
      progressStage.textContent = "Done!";

      // Load the result videos — show input video back, hide spinner
      videoInputLoading.hidden = true;
      videoInput.hidden = false;
      rgbPlaceholder.hidden = true;
      // The source was already set to local file for input preview
      // Output IR video:
      videoOutput.src = `/video_result/${jobId}/ir?t=${Date.now()}`;
      videoOutput.load();
      videoOutput.hidden = false;
      irPlaceholder.hidden = true;

      // Hide progress bar after short delay
      setTimeout(() => { videoProgress.hidden = true; }, 2000);

      generateBtn.disabled = false;

      // Detection panel — no per-frame JSON in video mode, just clear it
      detCountBadge.textContent = "N/A";
      detHuman.innerHTML = '<p style="font-size:14px;color:#9ca3af;text-align:center;padding:16px 0;">Per-frame detections are baked into the video output.</p>';

    } else if (event.type === "error") {
      evtSource.close();
      activeSSE = null;
      showError(event.message || "Video processing failed.");
      resetVideoLoadingUI();
      generateBtn.disabled = false;
    }
  };

  evtSource.onerror = function () {
    evtSource.close();
    activeSSE = null;
    showError("Lost connection to server while processing video.");
    resetVideoLoadingUI();
    generateBtn.disabled = false;
  };
}

function resetVideoLoadingUI() {
  videoInput.hidden = false;  // Show local preview again
  videoInputLoading.hidden = true;
  videoProgress.hidden = true;
  irPlaceholder.hidden = false;
  progressBar.style.width = "0%";
}

// ── Main button click ───────────────────────────────────────────────────────

generateBtn.addEventListener("click", function () {
  if (isVideoMode) {
    generateVideo();
  } else {
    generateImage();
  }
});
