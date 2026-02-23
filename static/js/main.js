"use strict";

const dropZone      = document.getElementById("dropZone");
const fileInput     = document.getElementById("fileInput");
const dzIdle        = document.getElementById("dzIdle");
const dzPreview     = document.getElementById("dzPreview");
const previewImg    = document.getElementById("previewImg");
const generateBtn   = document.getElementById("generateBtn");
const errorBanner   = document.getElementById("errorBanner");
const errorMsg      = document.getElementById("errorMsg");
const loadingOverlay= document.getElementById("loadingOverlay");

const resultCard    = document.getElementById("resultCard");
const resultInput   = document.getElementById("resultInput");
const resultOutput  = document.getElementById("resultOutput");
const downloadBtn   = document.getElementById("downloadBtn");
const resetBtn      = document.getElementById("resetBtn");

let selectedFile = null;

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden = false;
}

function hideError() {
  errorBanner.hidden = true;
  errorMsg.textContent = "";
}

function setLoading(active) {
  loadingOverlay.hidden = !active;
  generateBtn.disabled  = active;
}

function applyPreview(file) {
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  dzIdle.hidden   = true;
  dzPreview.hidden = false;
  generateBtn.disabled = false;
  hideError();
}

function resetUpload() {
  selectedFile = null;
  fileInput.value = "";
  previewImg.src  = "";
  dzPreview.hidden = true;
  dzIdle.hidden    = false;
  generateBtn.disabled = true;
  hideError();
}

function resetAll() {
  resultCard.hidden = true;
  resultInput.src   = "";
  resultOutput.src  = "";
  downloadBtn.href  = "#";

  
  const badge   = document.getElementById("detCountBadge");
  const content = document.getElementById("detectionContent");
  const warning = document.getElementById("detectionWarning");
  badge.textContent = "0";
  badge.classList.remove("active");
  content.innerHTML = "";
  warning.hidden    = true;

  resetUpload();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function handleFile(file) {
  if (!file) return;

  const allowed = ["image/jpeg", "image/png", "image/bmp", "image/tiff",
                   "image/tif", "image/webp"];
  if (!file.type.startsWith("image/") && !allowed.includes(file.type)) {
    showError(`Unsupported file type: ${file.type || "unknown"}. Please upload an image.`);
    return;
  }
  if (file.size > 16 * 1024 * 1024) {
    showError("File too large. Maximum size is 16 MB.");
    return;
  }

  selectedFile = file;
  applyPreview(file);
}

const DET_COLORS = {
  person:  "#ff376e",
  car:     "#37b4ff",
  bicycle: "#50f078",
  dog:     "#ffc837",
  truck:   "#b437ff",
  bus:     "#ffa500",
};
const DET_DEFAULT_COLOR = "#ff6b35";

function getDetColor(className) {
  return DET_COLORS[className.toLowerCase()] || DET_DEFAULT_COLOR;
}

function renderDetections(data) {
  const badge   = document.getElementById("detCountBadge");
  const content = document.getElementById("detectionContent");
  const warning = document.getElementById("detectionWarning");
  const warnMsg = document.getElementById("detectionWarningMsg");

  const predictions = data.predictions    || [];
  const count       = data.detection_count || 0;
  const errMsg      = data.detection_error || null;

  
  badge.textContent = count;
  badge.classList.toggle("active", count > 0);

  
  if (errMsg) {
    warnMsg.textContent = errMsg;
    warning.hidden = false;
  } else {
    warning.hidden = true;
  }

  
  if (predictions.length === 0) {
    content.innerHTML =
      '<p class="detection-empty">No objects detected in the generated IR image.</p>';
    return;
  }

  
  const list = document.createElement("div");
  list.className = "detection-list";

  predictions.forEach(function (pred) {
    const color = getDetColor(pred.class_name);
    const conf  = Math.round(pred.confidence * 100);
    const w     = pred.x2 - pred.x1;
    const h     = pred.y2 - pred.y1;

    const item = document.createElement("div");
    item.className = "detection-item";
    item.innerHTML =
      '<div class="detection-item-top">' +
        '<span class="detection-class-chip" style="background:' + color + '22; border-color:' + color + '; color:' + color + '">' +
          pred.class_name +
        '</span>' +
        '<span class="detection-conf-text">' + conf + '% confidence</span>' +
      '</div>' +
      '<div class="detection-conf-bar-wrap">' +
        '<div class="detection-conf-bar" style="width:' + conf + '%; background:' + color + '"></div>' +
      '</div>' +
      '<div class="detection-coords">' +
        '[' + pred.x1 + ', ' + pred.y1 + '] &rarr; [' + pred.x2 + ', ' + pred.y2 + ']' +
        '&nbsp;&middot;&nbsp;' + w + '&times;' + h + ' px' +
      '</div>';

    list.appendChild(item);
  });

  content.innerHTML = "";
  content.appendChild(list);
}

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", (e) => {
  if (!dropZone.contains(e.relatedTarget)) {
    dropZone.classList.remove("drag-over");
  }
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

document.addEventListener("dragover",  (e) => e.preventDefault());
document.addEventListener("drop",      (e) => e.preventDefault());

generateBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hideError();
  setLoading(true);
  resultCard.hidden = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || `Server error (${response.status})`);
      return;
    }

    
    const inputSrc  = `data:image/png;base64,${data.input}`;
    const outputSrc = `data:image/png;base64,${data.output}`;

    resultInput.src  = inputSrc;
    resultOutput.src = outputSrc;

    
    downloadBtn.href     = outputSrc;
    downloadBtn.download = `ir_${selectedFile.name.replace(/\.[^.]+$/, "")}.png`;

    
    renderDetections(data);

    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showError(`Request failed: ${err.message}`);
  } finally {
    setLoading(false);
  }
});

resetBtn.addEventListener("click", resetAll);
