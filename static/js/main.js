"use strict";

const dropZone         = document.getElementById("dropZone");
const fileInput        = document.getElementById("fileInput");
const dzIdle           = document.getElementById("dzIdle");
const dzPreview        = document.getElementById("dzPreview");
const previewThumb     = document.getElementById("previewThumb");
const previewName      = document.getElementById("previewName");
const previewSize      = document.getElementById("previewSize");
const browseLink       = document.getElementById("browseLink");
const clearBtn         = document.getElementById("clearBtn");
const generateBtn      = document.getElementById("generateBtn");
const errorBanner      = document.getElementById("errorBanner");
const errorMsg         = document.getElementById("errorMsg");

const resultInput      = document.getElementById("resultInput");
const rgbPlaceholder   = document.getElementById("rgbPlaceholder");
const resultOutput     = document.getElementById("resultOutput");
const irPlaceholder    = document.getElementById("irPlaceholder");
const irLoading        = document.getElementById("irLoading");
const downloadBtn      = document.getElementById("downloadBtn");

const detCountBadge       = document.getElementById("detCountBadge");
const detHuman            = document.getElementById("detHuman");
const detRaw              = document.getElementById("detRaw");
const detRawPre           = document.getElementById("detRawPre");
const detectionWarning    = document.getElementById("detectionWarning");
const detectionWarningMsg = document.getElementById("detectionWarningMsg");
const rawToggleTrack      = document.getElementById("rawToggleTrack");
const rawToggleThumb      = document.getElementById("rawToggleThumb");

let selectedFile = null;
let rawEnabled   = false;

function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.hidden   = false;
}

function hideError() {
  errorBanner.hidden   = true;
  errorMsg.textContent = "";
}

function formatBytes(bytes) {
  if (bytes < 1024)          return bytes + " B";
  if (bytes < 1024 * 1024)  return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function toggleRaw() {
  rawEnabled = !rawEnabled;
  if (rawEnabled) {
    rawToggleTrack.style.backgroundColor = "#7c3aed";
    rawToggleThumb.style.transform       = "translateX(16px)";
    detHuman.hidden = true;
    detRaw.hidden   = false;
  } else {
    rawToggleTrack.style.backgroundColor = "";
    rawToggleThumb.style.transform       = "";
    detHuman.hidden = false;
    detRaw.hidden   = true;
  }
}

function applyPreview(file) {
  const url = URL.createObjectURL(file);
  previewThumb.src        = url;
  previewName.textContent = file.name;
  previewSize.textContent = formatBytes(file.size);
  dzIdle.hidden    = true;
  dzPreview.hidden = false;

  resultInput.src           = url;
  resultInput.hidden        = false;
  rgbPlaceholder.hidden     = true;

  generateBtn.disabled = false;
  hideError();
}

function clearUpload() {
  selectedFile     = null;
  fileInput.value  = "";
  previewThumb.src = "";
  dzPreview.hidden = true;
  dzIdle.hidden    = false;

  resultInput.hidden    = true;
  resultInput.src       = "";
  rgbPlaceholder.hidden = false;

  resultOutput.hidden  = true;
  resultOutput.src     = "";
  irPlaceholder.hidden = false;
  irLoading.hidden     = true;

  downloadBtn.hidden   = true;
  generateBtn.disabled = true;

  detCountBadge.textContent = "0";
  detHuman.innerHTML = '<p style="font-size:14px;color:#9ca3af;text-align:center;padding:16px 0;">No detections yet. Generate an IR image to see results.</p>';
  detRawPre.textContent   = "No data";
  detectionWarning.hidden = true;

  hideError();
}

function handleFile(file) {
  if (!file) return;
  if (file.size > 16 * 1024 * 1024) {
    showError("File too large. Maximum size is 16 MB.");
    return;
  }
  selectedFile = file;
  applyPreview(file);
}

browseLink.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", () => { if (!dzIdle.hidden) fileInput.click(); });
dropZone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
clearBtn.addEventListener("click", (e) => { e.stopPropagation(); clearUpload(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); });

dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", (e) => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove("drag-over"); });
dropZone.addEventListener("drop", (e) => { e.preventDefault(); dropZone.classList.remove("drag-over"); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop",     (e) => e.preventDefault());

const DET_COLORS = {
  person:  "#7c3aed",
  car:     "#2563eb",
  bicycle: "#059669",
  dog:     "#d97706",
  truck:   "#dc2626",
  bus:     "#db2777",
};
const DET_DEFAULT = "#6b7280";

function getDetColor(cls) {
  return DET_COLORS[cls.toLowerCase()] || DET_DEFAULT;
}

function renderDetections(data) {
  const predictions = data.predictions     || [];
  const count       = data.detection_count || 0;
  const errMsg      = data.detection_error || null;

  detCountBadge.textContent = String(count);
  detRawPre.textContent     = JSON.stringify(predictions, null, 2);

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
  predictions.forEach(function(p) { classCounts[p.class_name] = (classCounts[p.class_name] || 0) + 1; });

  let html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:14px;">';
  Object.entries(classCounts).forEach(function([cls, cnt]) {
    var c = getDetColor(cls);
    html += '<span style="font-size:12px;font-weight:700;padding:2px 10px;border-radius:4px;border:1px solid ' + c + '44;background:' + c + '11;color:' + c + ';text-transform:uppercase;letter-spacing:0.04em;">' + cls + ' &times;' + cnt + '</span>';
  });
  html += '</div>';

  predictions.forEach(function(pred) {
    var c    = getDetColor(pred.class_name);
    var conf = Math.round(pred.confidence * 100);
    var w    = pred.x2 - pred.x1;
    var h    = pred.y2 - pred.y1;
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

generateBtn.addEventListener("click", async function() {
  if (!selectedFile) return;

  hideError();
  generateBtn.disabled = true;
  irPlaceholder.hidden = true;
  irLoading.hidden     = false;
  resultOutput.hidden  = true;
  downloadBtn.hidden   = true;

  const formData = new FormData();
  formData.append("image", selectedFile);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data     = await response.json();

    if (!response.ok || data.error) {
      showError(data.error || ("Server error (" + response.status + ")"));
      irPlaceholder.hidden = false;
      irLoading.hidden     = true;
      return;
    }

    resultInput.src       = "data:image/png;base64," + data.input;
    resultInput.hidden    = false;
    rgbPlaceholder.hidden = true;

    const outputSrc      = "data:image/png;base64," + data.output;
    resultOutput.src     = outputSrc;
    resultOutput.hidden  = false;
    irLoading.hidden     = true;

    downloadBtn.href     = outputSrc;
    downloadBtn.download = "ir_" + selectedFile.name.replace(/\.[^.]+$/, "") + ".png";
    downloadBtn.hidden   = false;

    renderDetections(data);

  } catch (err) {
    showError("Request failed: " + err.message);
    irPlaceholder.hidden = false;
    irLoading.hidden     = true;
  } finally {
    generateBtn.disabled = false;
  }
});
