"use strict";

(() => {
  const viewerRoot = document.getElementById("molstar-viewer");
  const targetSelect = document.getElementById("target-select");
  const poseButton = document.getElementById("show-pose");
  const ligandButton = document.getElementById("show-ligand");
  const downloadReceptor = document.getElementById("download-receptor");
  const downloadPose = document.getElementById("download-pose");
  const interactionStatus = document.getElementById("interaction-status");
  const status = document.getElementById("viewer-status");
  const fallback = document.getElementById("viewer-fallback");
  let viewer;
  let reportData;
  let viewMode = "pose";

  function setStatus(message) {
    status.textContent = message;
  }

  function selectedTarget() {
    return reportData.targets.find((target) => target.target_id === targetSelect.value);
  }

  function updateAssetLinks(target) {
    if (!target) {
      throw new Error("Selected target is absent from report metadata");
    }
    downloadReceptor.href = target.receptor.path;
    downloadPose.href = target.pose.path;
  }

  async function clearViewer() {
    if (!viewer.plugin || typeof viewer.plugin.clear !== "function") {
      throw new Error("Mol* viewer does not expose the required clear API");
    }
    await viewer.plugin.clear();
  }

  function structureFormat(asset) {
    if (!asset || !asset.path || !asset.format) {
      throw new Error("Report asset metadata is incomplete");
    }
    return asset.format;
  }

  async function canvasReadyForPixelVerification() {
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await new Promise((resolve) => setTimeout(resolve, 1200));
    const canvas = viewerRoot.querySelector("canvas");
    if (!canvas || canvas.width < 2 || canvas.height < 2) {
      return false;
    }
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) {
      return false;
    }
    if (gl.isContextLost()) {
      return false;
    }
    const sample = document.createElement("canvas");
    sample.width = 64;
    sample.height = 64;
    const context = sample.getContext("2d", { willReadFrequently: true });
    if (!context) {
      return false;
    }
    try {
      context.drawImage(canvas, 0, 0, sample.width, sample.height);
      const pixels = context.getImageData(0, 0, sample.width, sample.height).data;
      const first = [pixels[0], pixels[1], pixels[2], pixels[3]];
      let distinctPixels = 0;
      for (let index = 4; index < pixels.length; index += 4) {
        const difference =
          Math.abs(pixels[index] - first[0]) +
          Math.abs(pixels[index + 1] - first[1]) +
          Math.abs(pixels[index + 2] - first[2]) +
          Math.abs(pixels[index + 3] - first[3]);
        if (difference > 24) {
          distinctPixels += 1;
          if (distinctPixels >= 8) {
            return true;
          }
        }
      }
    } catch (_error) {
      return false;
    }
    return false;
  }

  async function loadInteraction(target) {
    const response = await fetch(target.interaction.path, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Interaction metadata failed to load (${response.status})`);
    }
    const payload = await response.json();
    const score = payload.docking && payload.docking.vina_score;
    const scoreText = Number.isFinite(score) ? `${score.toFixed(3)} kcal/mol` : "not available";
    interactionStatus.textContent = `${payload.interaction_status}; docking score ${scoreText}`;
  }

  async function loadSelection() {
    const target = selectedTarget();
    if (!target) {
      throw new Error("Selected target is absent from report metadata");
    }
    viewerRoot.dataset.viewerLoaded = "false";
    viewerRoot.dataset.canvasNonblank = "false";
    setStatus(`Loading ${target.target_id}...`);
    await clearViewer();
    await viewer.loadStructureFromUrl(
      target.receptor.path,
      structureFormat(target.receptor),
      false,
    );
    const ligandAsset = viewMode === "pose" ? target.pose : reportData.original_ligand;
    await viewer.loadStructureFromUrl(
      ligandAsset.path,
      structureFormat(ligandAsset),
      false,
    );
    await loadInteraction(target);
    updateAssetLinks(target);
    const nonblank = await canvasReadyForPixelVerification();
    viewerRoot.dataset.canvasNonblank = String(nonblank);
    if (!nonblank) {
      throw new Error("Mol* canvas was unavailable for pixel verification");
    }
    viewerRoot.dataset.activeTarget = target.target_id;
    viewerRoot.dataset.activeView = viewMode;
    viewerRoot.dataset.viewerLoaded = "true";
    setStatus(`${target.target_id}: ${viewMode === "pose" ? "docked pose" : "input ligand"}`);
  }

  function setViewMode(mode) {
    viewMode = mode;
    poseButton.setAttribute("aria-pressed", String(mode === "pose"));
    ligandButton.setAttribute("aria-pressed", String(mode === "ligand"));
  }

  async function runSelfTest(initialTarget) {
    if (new URLSearchParams(window.location.search).get("selftest") !== "1") {
      return;
    }
    if (reportData.targets.length > 1) {
      targetSelect.value = reportData.targets[1].target_id;
      await loadSelection();
      targetSelect.value = initialTarget;
      await loadSelection();
      viewerRoot.dataset.targetSwitchTest = "passed";
    } else {
      viewerRoot.dataset.targetSwitchTest = "not-applicable";
    }
    setViewMode("ligand");
    await loadSelection();
    setViewMode("pose");
    await loadSelection();
    viewerRoot.dataset.poseSwitchTest = "passed";
  }

  async function boot() {
    if (!window.molstar || !window.molstar.Viewer || typeof window.molstar.Viewer.create !== "function") {
      throw new Error("Local Mol* bundle did not expose Viewer.create");
    }
    const response = await fetch("targets.json", { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Target metadata failed to load (${response.status})`);
    }
    reportData = await response.json();
    if (!Array.isArray(reportData.targets) || reportData.targets.length === 0) {
      throw new Error("Report contains no target assets");
    }
    for (const target of reportData.targets) {
      const option = document.createElement("option");
      option.value = target.target_id;
      option.textContent = target.label;
      targetSelect.appendChild(option);
    }
    targetSelect.value = reportData.top_target_id;
    const initialTarget = targetSelect.value;
    updateAssetLinks(selectedTarget());
    viewer = await window.molstar.Viewer.create("molstar-viewer", {
      layoutIsExpanded: false,
      layoutShowControls: true,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      collapseRightPanel: true,
      viewportShowExpand: false,
      viewportShowSelectionMode: false,
    });
    if (typeof viewer.loadStructureFromUrl !== "function") {
      throw new Error("Mol* viewer does not expose loadStructureFromUrl");
    }
    setViewMode("pose");
    await loadSelection();
    targetSelect.addEventListener("change", () => loadSelection().catch(showFailure));
    poseButton.addEventListener("click", () => {
      setViewMode("pose");
      loadSelection().catch(showFailure);
    });
    ligandButton.addEventListener("click", () => {
      setViewMode("ligand");
      loadSelection().catch(showFailure);
    });
    await runSelfTest(initialTarget);
  }

  function showFailure(error) {
    const message = error instanceof Error ? error.message : String(error);
    document.body.dataset.viewerError = message;
    viewerRoot.dataset.viewerLoaded = "false";
    fallback.hidden = false;
    fallback.textContent = `3D viewer unavailable: ${message}. Molecular files remain downloadable.`;
    setStatus("3D rendering failed");
  }

  boot().catch(showFailure);
})();
