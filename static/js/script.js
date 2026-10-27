(function () {
  const MAX_BYTES = 5 * 1024 * 1024;
  const TYPES = ["image/jpeg", "image/png", "image/webp"];
  const $ = (id) => document.getElementById(id);

  const dropzone = $("dropzone"), input = $("file-input");
  const previewBox = $("preview-box"), preview = $("preview");
  const removeBtn = $("remove-btn"), classifyBtn = $("classify-btn");
  const errorEl = $("error"), results = $("results");
  let file = null;

  function showError(msg) { errorEl.textContent = msg; errorEl.hidden = !msg; }

  function setFile(f) {
    showError("");
    results.hidden = true;
    if (!f) return;
    if (!TYPES.includes(f.type)) return showError("Unsupported file. Please choose a JPG, PNG or WEBP image.");
    if (f.size > MAX_BYTES) return showError("File is too large. Maximum size is 5 MB.");
    file = f;
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.src = URL.createObjectURL(f);
    previewBox.hidden = false;
    dropzone.hidden = true;
    classifyBtn.disabled = false;
  }

  function clearFile() {
    file = null;
    input.value = "";
    preview.removeAttribute("src");
    previewBox.hidden = true;
    dropzone.hidden = false;
    classifyBtn.disabled = true;
    results.hidden = true;
    showError("");
  }

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
  removeBtn.addEventListener("click", clearFile);

  function render(data) {
    $("category").textContent = data.category;
    $("confidence").textContent = data.confidence.toFixed(1) + "% confidence";
    $("recommendation").textContent = data.recommendation;
    const list = $("top-list");
    list.innerHTML = "";
    data.top_predictions.forEach((p) => {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.textContent = p.label;
      const bar = document.createElement("div");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = p.confidence + "%";
      bar.appendChild(fill);
      const pct = document.createElement("span");
      pct.className = "pct";
      pct.textContent = p.confidence.toFixed(1) + "%";
      li.append(name, bar, pct);
      list.appendChild(li);
    });
    results.hidden = false;
    results.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  classifyBtn.addEventListener("click", async () => {
    if (!file) return showError("Please select an image first.");
    showError("");
    results.hidden = true;
    classifyBtn.disabled = true;
    classifyBtn.classList.add("loading");
    classifyBtn.textContent = "Classifying";
    try {
      const body = new FormData();
      body.append("image", file);
      const res = await fetch("/predict", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || "Something went wrong. Please try again.");
      render(data);
    } catch (err) {
      showError(err.message === "Failed to fetch" ? "Cannot reach the server. Is Flask running?" : err.message);
    } finally {
      classifyBtn.classList.remove("loading");
      classifyBtn.textContent = "Classify Waste";
      classifyBtn.disabled = !file;
    }
  });
})();
