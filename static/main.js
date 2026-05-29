const form = document.querySelector("#analysis-form");
const fileInput = document.querySelector("#file");
const preview = document.querySelector("#preview");
const statusEl = document.querySelector("#status");
const runButton = document.querySelector("#run");

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return "-";
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 0.001 || abs > 10000)) return Number(value).toExponential(3);
  return Number(value).toFixed(digits);
}

function tableFor(peaks) {
  if (!peaks || peaks.length === 0) return "<p class='status'>No peaks found.</p>";
  const rows = peaks.map((p) => `
    <tr>
      <td>${p.label}</td>
      <td>${fmt(p.period, 4)}</td>
      <td>${fmt(p.period_error, 4)}</td>
      <td>${fmt(p.frequency, 6)}</td>
      <td>${fmt(p.power, 6)}</td>
      <td>${fmt(p.fap, 3)}</td>
      <td>${p.kind}</td>
    </tr>
  `).join("");
  return `
    <table>
      <thead>
        <tr>
          <th>Peak</th>
          <th>P</th>
          <th>σP</th>
          <th>f</th>
          <th>Power</th>
          <th>FAP</th>
          <th>Type</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const text = await file.text();
  const lines = text.split(/\r?\n/).filter((line) => line.trim()).slice(0, 8);
  preview.textContent = lines.join("\n");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = new FormData(form);
  runButton.disabled = true;
  setStatus("Running Lomb-Scargle, sampling-window analysis, bootstrap, prewhitening, and folding...");

  try {
    const response = await fetch("/analyze", { method: "POST", body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Analysis failed");

    document.querySelector("#periodogram").src = result.plots.periodogram;
    document.querySelector("#window").src = result.plots.window;
    document.querySelector("#prewhitened").src = result.plots.prewhitened;
    document.querySelector("#folded").src = result.plots.folded;

    document.querySelector("#peaks-table").innerHTML = tableFor(result.peaks);
    document.querySelector("#residual-table").innerHTML = tableFor(result.residual_peaks);

    const maxima = result.folded_maxima.map((m) => fmt(m.phase, 4)).join(", ");
    document.querySelector("#summary").innerHTML = `
      <div class="summary-card"><span>Rows used</span><strong>${result.n_points}</strong></div>
      <div class="summary-card"><span>Baseline</span><strong>${fmt(result.baseline, 1)} d</strong></div>
      <div class="summary-card"><span>Primary period</span><strong>${fmt(result.primary_period, 4)} d</strong></div>
      <div class="summary-card"><span>Folded maxima</span><strong>${maxima || "-"}</strong></div>
    `;
    setStatus("Done.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    runButton.disabled = false;
  }
});
