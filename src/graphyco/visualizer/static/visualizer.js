let DATA = null;
let currentArch = null;
let currentStep = 1;
let currentMode = "gradient";
let currentView = "neuron"; // "neuron" | "layer"
let isPlaying = false;
let playInterval = null;
let selectedNodeId = null;

// Transform state
let zoom = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let startX = 0;
let startY = 0;

let nodePositions = {};
let layerStages = [];

function init() {
  // Priority 1: Data injected by the host (renderer.py or live mode)
  if (window.__GRAPHYCO_DATA__) {
    DATA = window.__GRAPHYCO_DATA__;
    setupApp();
    return;
  }

  // Priority 2: Embedded data tag (standalone HTML export or template injection)
  const tag = document.getElementById('embedded-data');
  if (tag && tag.textContent.trim()) {
    try {
      DATA = JSON.parse(tag.textContent);
      setupApp();
      return;
    } catch(e) {
      console.warn("Embedded data parse error:", e);
    }
  }

  // Priority 3: Fetch from co-located JSON file
  fetch('dynamic_flow_benchmark.json')
    .then(r => r.json())
    .then(d => { DATA = d; setupApp(); })
    .catch(() => {
      document.getElementById('inspector-empty').textContent =
        "Load a JSON file using the Load JSON button to begin.";
    });
}

document.getElementById('json-input').addEventListener('change', function(e) {
  const f = e.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = ev => {
    try {
      DATA = JSON.parse(ev.target.result);
      setupApp();
    } catch(err) {
      alert("Invalid JSON: " + err);
    }
  };
  reader.readAsText(f);
});

function setupApp() {
  if (!DATA || !DATA.architectures) return;

  const sel = document.getElementById('arch-select');
  sel.innerHTML = '';
  Object.keys(DATA.architectures).forEach(name => {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  });

  const params = new URLSearchParams(window.location.search);
  if (params.get('arch') && DATA.architectures[params.get('arch')]) {
    sel.value = params.get('arch');
  }
  currentArch = sel.value;

  if (params.get('view')) {
    currentView = params.get('view');
    document.querySelectorAll('#view-buttons button').forEach(b => b.classList.remove('active'));
    const vb = document.querySelector(`#view-buttons button[data-view="${currentView}"]`);
    if (vb) vb.classList.add('active');
  }
  if (params.get('mode')) {
    currentMode = params.get('mode');
    document.querySelectorAll('#mode-buttons button').forEach(b => b.classList.remove('active'));
    const mb = document.querySelector(`#mode-buttons button[data-mode="${currentMode}"]`);
    if (mb) mb.classList.add('active');
  }
  if (params.get('step')) {
    currentStep = parseInt(params.get('step'));
    const sld = document.getElementById('step-slider');
    if (sld) sld.value = currentStep;
  }

  sel.onchange = () => {
    currentArch = sel.value;
    render();
    fitView();
  };

  // View toggle
  document.querySelectorAll('#view-buttons button').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('#view-buttons button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentView = btn.dataset.view;
      document.getElementById('caption-mode-text').textContent = currentView === 'neuron' ? 'Multi-neuron network' : 'Layer computational graph';
      render();
      fitView();
    };
  });

  // Metric toggle
  document.querySelectorAll('#mode-buttons button').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('#mode-buttons button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentMode = btn.dataset.mode;
      updateOverlay();
    };
  });

  // Step slider
  const slider = document.getElementById('step-slider');
  slider.oninput = () => {
    currentStep = parseInt(slider.value);
    updateStepUI();
    updateOverlay();
    if (selectedNodeId) updateInspector(selectedNodeId);
  };

  // Play / Pause
  const playBtn = document.getElementById('btn-play');
  playBtn.onclick = () => {
    if (isPlaying) {
      clearInterval(playInterval);
      isPlaying = false;
      playBtn.textContent = '▶ Play';
      updateEdgeRunningState(false);
    } else {
      isPlaying = true;
      playBtn.textContent = '⏸ Pause';
      updateEdgeRunningState(true);
      playInterval = setInterval(() => {
        currentStep = (currentStep % 10) + 1;
        slider.value = currentStep;
        updateStepUI();
        updateOverlay();
        if (selectedNodeId) updateInspector(selectedNodeId);
      }, 750);
    }
  };

  document.getElementById('btn-prev').onclick = () => {
    currentStep = Math.max(1, currentStep - 1);
    slider.value = currentStep;
    updateStepUI();
    updateOverlay();
    if (selectedNodeId) updateInspector(selectedNodeId);
  };

  document.getElementById('btn-next').onclick = () => {
    currentStep = Math.min(10, currentStep + 1);
    slider.value = currentStep;
    updateStepUI();
    updateOverlay();
    if (selectedNodeId) updateInspector(selectedNodeId);
  };

  setupPanZoom();
  render();
  fitView();
}

function updateStepUI() {
  if (!DATA || !currentArch) return;
  const traj = DATA.step_trajectories[currentArch];
  const maxIdx = traj ? traj.length : 1;
  const currentStepData = traj ? traj[currentStep - 1] : null;
  const lastStepData = traj ? traj[maxIdx - 1] : null;
  
  const realStep = currentStepData && currentStepData.step !== undefined ? currentStepData.step : currentStep;
  const realMaxStep = lastStepData && lastStepData.step !== undefined ? lastStepData.step : maxIdx;
  
  const loss = currentStepData ? currentStepData.loss : null;
  const lossStr = (loss !== null && loss !== undefined && !isNaN(loss)) ? ` (Loss: ${Number(loss).toFixed(4)})` : '';
  
  document.getElementById('step-display').textContent = `Step: ${realStep} / ${realMaxStep}${lossStr}`;
  document.getElementById('caption-step').textContent = `Step = ${realStep}`;
}

function setupPanZoom() {
  const container = document.getElementById('canvas-container');
  container.addEventListener('wheel', e => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const rect = container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const newZoom = Math.max(0.3, Math.min(4.0, zoom * delta));
    panX = mouseX - (mouseX - panX) * (newZoom / zoom);
    panY = mouseY - (mouseY - panY) * (newZoom / zoom);
    zoom = newZoom;
    applyTransform();
  }, { passive: false });

  container.addEventListener('mousedown', e => {
    if (e.target.closest('.node-group')) return;
    isDragging = true;
    startX = e.clientX - panX;
    startY = e.clientY - panY;
  });

  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    panX = e.clientX - startX;
    panY = e.clientY - startY;
    applyTransform();
  });

  window.addEventListener('mouseup', () => { isDragging = false; });

  document.getElementById('btn-zoom-in').onclick = () => {
    const cWidth = container.clientWidth || 900;
    const cHeight = container.clientHeight || 550;
    const cx = cWidth / 2;
    const cy = cHeight / 2;
    const newZoom = Math.min(4.0, zoom * 1.25);
    panX = cx - (cx - panX) * (newZoom / zoom);
    panY = cy - (cy - panY) * (newZoom / zoom);
    zoom = newZoom;
    applyTransform();
  };

  document.getElementById('btn-zoom-out').onclick = () => {
    const cWidth = container.clientWidth || 900;
    const cHeight = container.clientHeight || 550;
    const cx = cWidth / 2;
    const cy = cHeight / 2;
    const newZoom = Math.max(0.3, zoom / 1.25);
    panX = cx - (cx - panX) * (newZoom / zoom);
    panY = cy - (cy - panY) * (newZoom / zoom);
    zoom = newZoom;
    applyTransform();
  };

  document.getElementById('btn-zoom-reset').onclick = fitView;
}

function resetView() {
  fitView();
}

function fitView() {
  const container = document.getElementById('canvas-container');
  if (!container) return;

  const cWidth = container.clientWidth || 900;
  const cHeight = container.clientHeight || 550;

  const keys = Object.keys(nodePositions);
  if (keys.length === 0) {
    zoom = 1.0;
    panX = 0;
    panY = 0;
    applyTransform();
    return;
  }

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  keys.forEach(k => {
    const p = nodePositions[k];
    if (p) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
  });

  const padLeft = 80;
  const padRight = 80;
  const padTop = currentView === 'neuron' ? 85 : 70;
  const padBottom = 65;

  const graphWidth = (maxX - minX) + padLeft + padRight;
  const graphHeight = (maxY - minY) + padTop + padBottom;

  const scaleX = cWidth / Math.max(graphWidth, 200);
  const scaleY = cHeight / Math.max(graphHeight, 200);
  zoom = Math.max(0.4, Math.min(1.35, Math.min(scaleX, scaleY)));

  const centerX = (minX + maxX) / 2;
  const centerY = ((minY - (currentView === 'neuron' ? 25 : 0)) + maxY) / 2;

  panX = cWidth / 2 - centerX * zoom;
  panY = cHeight / 2 - centerY * zoom;

  applyTransform();
}

function applyTransform() {
  const g = document.getElementById('transform-layer');
  if (g) g.setAttribute('transform', `translate(${panX}, ${panY}) scale(${zoom})`);
  const zoomDisplay = document.getElementById('zoom-display');
  if (zoomDisplay) zoomDisplay.textContent = `${Math.round(zoom * 100)}%`;
}

function updateEdgeRunningState(running) {
  document.querySelectorAll('.edge-line').forEach(el => {
    if (running) el.classList.add('edge-running');
    else el.classList.remove('edge-running');
  });
}

function render() {
  if (!DATA || !currentArch) return;
  const arch = DATA.architectures[currentArch];
  const summary = arch.summary;

  // Table 1: Invariants
  const fmt = (v, d = '-') => (v !== undefined && v !== null ? Number(v).toFixed(4) : d);
  document.getElementById('val-bneck').textContent = fmt(summary.topological_bottleneck);
  document.getElementById('val-gmax').textContent = fmt(summary.grad_bottleneck_gmax);
  document.getElementById('val-bnode').textContent = summary.bottleneck_node || '-';
  document.getElementById('val-coher').textContent = fmt(summary.coherence);
  document.getElementById('val-resil').textContent = fmt(summary.resil || summary.resilience);
  document.getElementById('val-spearman').textContent = fmt(summary.spearman_act_grad);
  document.getElementById('val-pearson').textContent = fmt(summary.pearson_act_grad);
  document.getElementById('val-cv').textContent = fmt(summary.mean_stability_cv);

  if (summary.type_counts) {
    const tc = summary.type_counts;
    document.getElementById('val-types').textContent = `${tc['Type I']||0} / ${tc['Type II']||0} / ${tc['Type III']||0} / ${tc['Type IV']||0}`;
  } else {
    document.getElementById('val-types').textContent = '-';
  }

  if (currentView === "neuron") {
    renderMultiNeuronView(arch);
  } else {
    renderLayerGraphView(arch);
  }

  updateOverlay();

  const nodeKeys = Object.keys(arch.static_graph.nodes);
  if (nodeKeys.length > 0) {
    updateInspector(nodeKeys[0]);
  }
}

// ══════════════════════════════════════════════════════════════
//  DYNAMIC MULTI-NEURON ARCHITECTURE DECOMPOSITION
// ══════════════════════════════════════════════════════════════

function getArchitectureStages(archName, archData) {
  if (archName === "Sequential MLP") {
    return [
      { id: "s0", title: "Input Layer", count: 4, primaryNode: "x" },
      { id: "s1", title: "Hidden Layer 1", count: 5, primaryNode: "net_0" },
      { id: "s2", title: "Hidden Layer 2", count: 5, primaryNode: "net_2" },
      { id: "s3", title: "Output Layer", count: 2, primaryNode: "net_6" }
    ];
  }
  if (archName === "ResNet Block") {
    return [
      { id: "s0", title: "Input Layer", count: 4, primaryNode: "x" },
      { id: "s1", title: "Residual Layer 1", count: 5, primaryNode: "l1" },
      { id: "s2", title: "Residual Layer 2", count: 5, primaryNode: "l2" },
      { id: "s3", title: "Merge (+ Skip)", count: 5, primaryNode: "add", skipFrom: 0 },
      { id: "s4", title: "Output Layer", count: 2, primaryNode: "out" }
    ];
  }
  if (archName === "DenseBlock") {
    return [
      { id: "s0", title: "Input Layer", count: 4, primaryNode: "x" },
      { id: "s1", title: "Dense Block 1", count: 4, primaryNode: "l1" },
      { id: "s2", title: "Concat 1", count: 6, primaryNode: "cat", skipFrom: 0 },
      { id: "s3", title: "Dense Block 2", count: 4, primaryNode: "l2" },
      { id: "s4", title: "Concat 2", count: 6, primaryNode: "cat_1", skipFrom: 1 },
      { id: "s5", title: "Output Head", count: 2, primaryNode: "head" }
    ];
  }
  if (archName === "U-Net Toy") {
    return [
      { id: "s0", title: "Input Layer", count: 4, primaryNode: "x" },
      { id: "s1", title: "Encoder 1", count: 4, primaryNode: "enc1" },
      { id: "s2", title: "Encoder 2", count: 5, primaryNode: "enc2" },
      { id: "s3", title: "Bottleneck", count: 5, primaryNode: "bottleneck" },
      { id: "s4", title: "Decoder 2", count: 5, primaryNode: "dec2" },
      { id: "s5", title: "Skip Concat", count: 6, primaryNode: "cat", skipFrom: 2 },
      { id: "s6", title: "Decoder 1", count: 4, primaryNode: "dec1" },
      { id: "s7", title: "Output Layer", count: 2, primaryNode: "out" }
    ];
  }
  if (archName === "Self-Attention Toy") {
    return [
      { id: "s0", title: "Input Layer", count: 4, primaryNode: "x" },
      { id: "s1", title: "Q / K / V", count: 6, primaryNode: "q" },
      { id: "s2", title: "Attention Matrix", count: 4, primaryNode: "mul" },
      { id: "s3", title: "Context Value", count: 4, primaryNode: "mul_1" },
      { id: "s4", title: "Residual Add", count: 4, primaryNode: "add", skipFrom: 0 },
      { id: "s5", title: "Output Layer", count: 2, primaryNode: "out" }
    ];
  }

  // Fallback for custom uploaded architectures
  const topoLayers = computeLayers(archData.static_graph.nodes, archData.static_graph.edges);
  return topoLayers.map((layerNodes, lIdx) => {
    const isFirst = lIdx === 0;
    const isLast = lIdx === topoLayers.length - 1;
    const title = isFirst ? "Input" : (isLast ? "Output" : `L${lIdx}`);
    const count = isFirst ? 4 : (isLast ? 2 : 5);
    return {
      id: `s${lIdx}`,
      title: title,
      count: count,
      primaryNode: layerNodes[0]
    };
  });
}

function renderMultiNeuronView(arch) {
  const nodesLayer = document.getElementById('nodes-layer');
  const edgesLayer = document.getElementById('edges-layer');
  const annotLayer = document.getElementById('annotations-layer');
  nodesLayer.innerHTML = '';
  edgesLayer.innerHTML = '';
  annotLayer.innerHTML = '';

  const stages = getArchitectureStages(currentArch, arch);
  layerStages = stages;

  const container = document.getElementById('canvas-container');
  const width = Math.max(900, container.clientWidth || 960);
  const height = Math.max(520, container.clientHeight || 580);

  const paddingX = 110;
  const colSpacing = (width - 2 * paddingX) / Math.max(1, stages.length - 1);

  nodePositions = {};

  // 1. Position Neurons and Annotations
  stages.forEach((stage, cIdx) => {
    const x = paddingX + cIdx * colSpacing;
    const N = stage.count;

    // Header label above column
    const titleY = 38;
    const textElem = document.createElementNS("http://www.w3.org/2000/svg", "text");
    textElem.setAttribute("class", "layer-label");
    textElem.setAttribute("font-size", "14.5");
    textElem.setAttribute("font-weight", "700");
    textElem.setAttribute("text-anchor", "middle");
    textElem.setAttribute("x", x);
    textElem.setAttribute("y", titleY);
    textElem.textContent = stage.title;
    annotLayer.appendChild(textElem);

    // Vertical spacing
    const spacing = Math.min(68, (height - 180) / Math.max(1, N - 1));
    const startY = (height + 40 - (N - 1) * spacing) / 2;

    // Pointer line from label to top neuron
    const lineElem = document.createElementNS("http://www.w3.org/2000/svg", "line");
    lineElem.setAttribute("class", "layer-pointer");
    lineElem.setAttribute("x1", x);
    lineElem.setAttribute("y1", titleY + 8);
    lineElem.setAttribute("x2", x);
    lineElem.setAttribute("y2", startY - 26);
    lineElem.setAttribute("marker-end", "url(#arrow-down)");
    annotLayer.appendChild(lineElem);

    for (let i = 0; i < N; i++) {
      const y = startY + i * spacing;
      const neuronId = `neuron-${cIdx}-${i}`;
      nodePositions[neuronId] = { x, y, stageIdx: cIdx, neuronIdx: i, primaryNode: stage.primaryNode };
    }
  });

  // 2. Render Bipartite Connections
  const r = 18; // neuron circle radius
  for (let c = 0; c < stages.length - 1; c++) {
    const s1 = stages[c];
    const s2 = stages[c + 1];

    for (let i = 0; i < s1.count; i++) {
      const p1 = nodePositions[`neuron-${c}-${i}`];
      for (let j = 0; j < s2.count; j++) {
        const p2 = nodePositions[`neuron-${c + 1}-${j}`];
        if (!p1 || !p2) continue;

        // Angle and intersection points on circle perimeters
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const theta = Math.atan2(dy, dx);

        const x1 = p1.x + (r + 1) * Math.cos(theta);
        const y1 = p1.y + (r + 1) * Math.sin(theta);
        const x2 = p2.x - (r + 3) * Math.cos(theta);
        const y2 = p2.y - (r + 3) * Math.sin(theta);

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const d = `M ${x1} ${y1} C ${x1 + dx * 0.35} ${y1}, ${x2 - dx * 0.35} ${y2}, ${x2} ${y2}`;

        path.setAttribute("d", d);
        path.setAttribute("class", "edge-line" + (isPlaying ? " edge-running" : ""));
        path.setAttribute("id", `synapse-${c}-${i}-${c+1}-${j}`);
        path.setAttribute("stroke-width", "1.1");
        path.setAttribute("marker-end", "url(#arrow)");
        edgesLayer.appendChild(path);
      }
    }
  }

  // 3. Render Skip / Bypass Connections
  stages.forEach((stage, c) => {
    if (stage.skipFrom !== undefined) {
      const srcStage = stages[stage.skipFrom];
      const count = Math.min(srcStage.count, stage.count);
      for (let i = 0; i < count; i++) {
        const p1 = nodePositions[`neuron-${stage.skipFrom}-${i}`];
        const p2 = nodePositions[`neuron-${c}-${i}`];
        if (!p1 || !p2) continue;

        const arc = Math.min(75, (c - stage.skipFrom) * 22);
        const midY = Math.min(p1.y, p2.y) - arc;

        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        const d = `M ${p1.x} ${p1.y - r} C ${p1.x + 35} ${midY}, ${p2.x - 35} ${midY}, ${p2.x} ${p2.y - r}`;

        path.setAttribute("d", d);
        path.setAttribute("class", "edge-line" + (isPlaying ? " edge-running" : ""));
        path.setAttribute("stroke-width", "1.4");
        path.setAttribute("stroke", "#111827");
        path.setAttribute("stroke-dasharray", "4 2");
        path.setAttribute("marker-end", "url(#arrow-down)");
        edgesLayer.appendChild(path);
      }
    }
  });

  // 4. Render Neuron Circles
  stages.forEach((stage, cIdx) => {
    for (let i = 0; i < stage.count; i++) {
      const neuronId = `neuron-${cIdx}-${i}`;
      const pos = nodePositions[neuronId];
      if (!pos) continue;

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "node-group");
      g.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
      g.setAttribute("id", `g-${neuronId}`);

      // Outer circle (white fill, clean border)
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", "18");
      circle.setAttribute("fill", "#ffffff");
      circle.setAttribute("stroke", "#111827");
      circle.setAttribute("stroke-width", "1.4");
      circle.setAttribute("class", "node-circle");
      circle.setAttribute("id", `circle-${neuronId}`);

      // Inner dynamic core dot
      const core = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      core.setAttribute("r", "0");
      core.setAttribute("fill", "#111827");
      core.setAttribute("class", "node-core");
      core.setAttribute("id", `core-${neuronId}`);

      // Clear readable neuron index tag
      const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lbl.setAttribute("class", "node-val");
      lbl.setAttribute("font-size", "12");
      lbl.setAttribute("font-weight", "600");
      lbl.setAttribute("text-anchor", "middle");
      lbl.setAttribute("y", "31");
      lbl.setAttribute("id", `lbl-${neuronId}`);
      lbl.textContent = `u${i}`;

      g.appendChild(circle);
      g.appendChild(core);
      g.appendChild(lbl);

      g.addEventListener('click', ev => {
        ev.stopPropagation();
        updateInspector(stage.primaryNode, i);
      });
      g.addEventListener('mouseenter', () => {
        updateInspector(stage.primaryNode, i);
      });

      nodesLayer.appendChild(g);
    }
  });
}

// ══════════════════════════════════════════════════════════════
//  LAYER GRAPH (MACRO OPERATION DAG VIEW)
// ══════════════════════════════════════════════════════════════

function computeLayers(nodes, edges) {
  const inDegree = {};
  const adj = {};
  Object.keys(nodes).forEach(id => { inDegree[id] = 0; adj[id] = []; });
  edges.forEach(e => {
    if (inDegree[e.target] !== undefined) inDegree[e.target]++;
    if (adj[e.source]) adj[e.source].push(e.target);
  });

  const q = Object.keys(nodes).filter(id => inDegree[id] === 0);
  const topo = [];
  const inD = Object.assign({}, inDegree);
  while (q.length > 0) {
    const u = q.shift();
    topo.push(u);
    (adj[u] || []).forEach(v => {
      inD[v]--;
      if (inD[v] === 0) q.push(v);
    });
  }

  Object.keys(nodes).forEach(id => {
    if (!topo.includes(id)) topo.push(id);
  });

  const dist = {};
  Object.keys(nodes).forEach(id => { dist[id] = 0; });
  topo.forEach(u => {
    (adj[u] || []).forEach(v => {
      if (dist[v] < dist[u] + 1) dist[v] = dist[u] + 1;
    });
  });

  const byLayer = {};
  Object.keys(nodes).forEach(id => {
    const d = dist[id];
    if (!byLayer[d]) byLayer[d] = [];
    byLayer[d].push(id);
  });

  return Object.keys(byLayer).sort((a,b) => a - b).map(k => byLayer[k]);
}

function renderLayerGraphView(arch) {
  const nodesLayer = document.getElementById('nodes-layer');
  const edgesLayer = document.getElementById('edges-layer');
  const annotLayer = document.getElementById('annotations-layer');
  nodesLayer.innerHTML = '';
  edgesLayer.innerHTML = '';
  annotLayer.innerHTML = '';

  const nodes = arch.static_graph.nodes;
  const edges = arch.static_graph.edges;

  const layerAssignment = computeLayers(nodes, edges);
  const numLayers = layerAssignment.length;

  const container = document.getElementById('canvas-container');
  const width = Math.max(880, container.clientWidth || 940);
  const height = Math.max(520, container.clientHeight || 580);

  const paddingX = 100;
  const availWidth = width - paddingX * 2;
  const colSpacing = numLayers > 1 ? availWidth / (numLayers - 1) : availWidth;

  const nodeLayerIndex = {};
  nodePositions = {};

  layerAssignment.forEach((layerNodes, lIdx) => {
    const x = paddingX + lIdx * colSpacing;
    const nCount = layerNodes.length;

    layerNodes.forEach((nid, nIdx) => {
      nodeLayerIndex[nid] = lIdx;
      let y;
      if (nCount === 1) {
        y = height / 2;
      } else {
        const spacingY = Math.min(110, (height - 140) / (nCount + 1));
        const startY = (height - (nCount - 1) * spacingY) / 2;
        y = startY + nIdx * spacingY;
      }
      nodePositions[nid] = { x, y };
    });
  });

  const r = 20;
  edges.forEach(e => {
    const p1 = nodePositions[e.source];
    const p2 = nodePositions[e.target];
    if (!p1 || !p2) return;

    const srcLayer = nodeLayerIndex[e.source] || 0;
    const tgtLayer = nodeLayerIndex[e.target] || 0;
    const layerDiff = tgtLayer - srcLayer;

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    let d;

    if (layerDiff > 1) {
      const arcHeight = Math.min(90, (layerDiff - 1) * 26);
      const midY = Math.min(p1.y, p2.y) - arcHeight;
      d = `M ${p1.x} ${p1.y - r} C ${p1.x + 40} ${midY}, ${p2.x - 40} ${midY}, ${p2.x} ${p2.y - r}`;
    } else {
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const theta = Math.atan2(dy, dx);
      const x1 = p1.x + (r + 1) * Math.cos(theta);
      const y1 = p1.y + (r + 1) * Math.sin(theta);
      const x2 = p2.x - (r + 3) * Math.cos(theta);
      const y2 = p2.y - (r + 3) * Math.sin(theta);
      d = `M ${x1} ${y1} C ${x1 + (x2 - x1) * 0.4} ${y1}, ${x2 - (x2 - x1) * 0.4} ${y2}, ${x2} ${y2}`;
    }

    path.setAttribute("d", d);
    path.setAttribute("class", "edge-line" + (isPlaying ? " edge-running" : ""));
    path.setAttribute("id", `edge-${e.source}-${e.target}`);
    path.setAttribute("stroke-width", "1.4");
    path.setAttribute("marker-end", "url(#arrow)");
    edgesLayer.appendChild(path);
  });

  Object.keys(nodes).forEach(nid => {
    const pos = nodePositions[nid];
    const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
    g.setAttribute("class", "node-group");
    g.setAttribute("transform", `translate(${pos.x}, ${pos.y})`);
    g.setAttribute("id", `node-g-${nid}`);

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", "20");
    circle.setAttribute("fill", "#ffffff");
    circle.setAttribute("stroke", "#111827");
    circle.setAttribute("stroke-width", "1.4");
    circle.setAttribute("class", "node-circle");
    circle.setAttribute("id", `circle-${nid}`);

    const core = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    core.setAttribute("r", "0");
    core.setAttribute("fill", "#111827");
    core.setAttribute("class", "node-core");
    core.setAttribute("id", `core-${nid}`);

    const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lbl.setAttribute("class", "node-lbl");
    lbl.setAttribute("font-size", "13");
    lbl.setAttribute("font-weight", "600");
    lbl.setAttribute("text-anchor", "middle");
    lbl.setAttribute("y", "36");
    lbl.setAttribute("id", `lbl-${nid}`);
    lbl.textContent = nid.length > 13 ? nid.substring(0, 12) + '…' : nid;

    g.appendChild(circle);
    g.appendChild(core);
    g.appendChild(lbl);

    g.addEventListener('click', ev => {
      ev.stopPropagation();
      updateInspector(nid);
    });
    g.addEventListener('mouseenter', () => {
      updateInspector(nid);
    });

    nodesLayer.appendChild(g);
  });
}

// ══════════════════════════════════════════════════════════════
//  DYNAMIC STATE OVERLAY ENGINE
// ══════════════════════════════════════════════════════════════

window.injectLiveTelemetry = function(stepNum, loss, nodeRecords, summary) {
  if (!DATA || !currentArch) return;
  const arch = DATA.architectures[currentArch];
  if (!arch) return;

  // Normalize node records: accept both flat format and nested {metrics: {...}} format
  if (nodeRecords) {
    const normalized = {};
    Object.keys(nodeRecords).forEach(nid => {
      const rec = nodeRecords[nid];
      if (rec && rec.metrics) {
        // Legacy nested format
        normalized[nid] = {
          grad_rms: rec.metrics.grad_rms || 0,
          act_rms: rec.metrics.activation_rms || rec.metrics.act_rms || 0,
          ratio_grad_act: rec.metrics.sensitivity || rec.metrics.ratio_grad_act || 0,
          param_grad_rms: rec.metrics.param_grad_rms || null,
        };
      } else {
        normalized[nid] = rec;
      }
    });
    nodeRecords = normalized;
  }

  if (!DATA.step_trajectories) DATA.step_trajectories = {};
  if (!DATA.step_trajectories[currentArch]) DATA.step_trajectories[currentArch] = [];

  const traj = DATA.step_trajectories[currentArch];
  const stepObj = {
    step: stepNum,
    loss: loss,
    nodes: nodeRecords || {},
    node_records: nodeRecords || {}
  };
  
  // 1. If this is the FIRST live injection, clear the dummy static JSON steps
  if (!window._live_initialized) {
    DATA.step_trajectories[currentArch] = [];
    window._live_initialized = true;
  }
  
  // 2. Accumulate the new live step
  DATA.step_trajectories[currentArch].push(stepObj);
  const newTraj = DATA.step_trajectories[currentArch];

  if (nodeRecords && arch.nodes) {
    Object.keys(nodeRecords).forEach(nid => {
      if (!arch.nodes[nid]) arch.nodes[nid] = {};
      arch.nodes[nid].activation_gradient = { rms: (nodeRecords[nid] && nodeRecords[nid].grad_rms) || 0 };
      arch.nodes[nid].activation = { rms: (nodeRecords[nid] && nodeRecords[nid].act_rms) || 0 };
      arch.nodes[nid].ratio_grad_act = (nodeRecords[nid] && nodeRecords[nid].ratio_grad_act) || 0;
    });
  }

  // Update Live header indicators
  const capStep = document.getElementById('caption-step');
  if (capStep) capStep.textContent = `Live Step = ${stepNum}`;

  // Keep static playback controls visible, but update slider max dynamically!
  const slider = document.getElementById('step-slider');
  if (slider) {
    slider.style.display = 'inline-block';
    const isAtEnd = parseInt(slider.value) >= parseInt(slider.max) || parseInt(slider.max) <= 1;
    slider.max = newTraj.length;
    
    // Auto-scroll to latest step if user was already at the end
    if (isAtEnd) {
        slider.value = newTraj.length;
        currentStep = newTraj.length;
    }
    updateStepUI();
  }

  const playBtn = document.getElementById('btn-play');
  if (playBtn) playBtn.style.display = 'inline-block';

  // Update summary sidebar
  if (summary) {
    if (!arch.summary) arch.summary = {};
    Object.assign(arch.summary, summary);
    const fmt = (v, d = '-') => (v !== undefined && v !== null && !isNaN(v) ? Number(v).toFixed(4) : d);
    const elGmax = document.getElementById('val-gmax');
    if (elGmax && summary.grad_bottleneck_gmax !== undefined) elGmax.textContent = fmt(summary.grad_bottleneck_gmax);
    const elBnode = document.getElementById('val-bnode');
    if (elBnode && summary.bottleneck_node !== undefined) elBnode.textContent = summary.bottleneck_node || '-';
  }

  window.liveStepCounter = (window.liveStepCounter || 0) + 1;
  // Render the currently selected step (which will be the latest if auto-scrolled)
  updateOverlay(window.liveStepCounter);
};

function updateOverlay(flickerStep) {
  if (!DATA || !currentArch) return;
  const arch = DATA.architectures[currentArch];
  const traj = DATA.step_trajectories[currentArch];
  const stepData = traj ? traj[currentStep - 1] : null;
  const tForPhase = flickerStep !== undefined ? flickerStep : currentStep;
  const dynEv = arch.dynamic_evaluation;
  const cats = dynEv.correspondence ? dynEv.correspondence.node_categories : {};

  const legendTitle = document.getElementById('legend-name');
  const legendContent = document.getElementById('legend-content');

  if (currentMode === 'type') {
    legendTitle.textContent = "Classification: Types I–IV";
    legendContent.innerHTML = `
      <div class="legend-row"><div class="core-sample" style="border:2.5px solid #111827"><div class="core-dot" style="width:10px;height:10px"></div></div><span><strong>Type I:</strong> High Topo, High Grad</span></div>
      <div class="legend-row"><div class="core-sample" style="border:2.0px dashed #111827"><div class="core-dot" style="width:4px;height:4px"></div></div><span><strong>Type II:</strong> High Topo, Low Grad</span></div>
      <div class="legend-row"><div class="core-sample" style="border:1.4px solid #111827"><div class="core-dot" style="width:10px;height:10px"></div></div><span><strong>Type III:</strong> Low Topo, High Grad</span></div>
      <div class="legend-row"><div class="core-sample" style="border:1.2px dashed #6b7280"><div class="core-dot" style="width:0;height:0"></div></div><span><strong>Type IV:</strong> Low Topo, Low Grad</span></div>
    `;
  } else {
    const titles = {
      gradient: "Gradient RMS ||g||",
      activation: "Activation RMS ||x||",
      sensitivity: "Sensitivity Ratio (g/a)",
      topology: "Perturbation Cost T",
      stability: "Temporal CV"
    };
    legendTitle.textContent = titles[currentMode] || currentMode;
    legendContent.innerHTML = `
      <div class="legend-row">
        <div class="core-sample"><div class="core-dot" style="width:0;height:0"></div></div>
        <span>Low magnitude / Quiescent</span>
      </div>
      <div class="legend-row">
        <div class="core-sample"><div class="core-dot" style="width:5px;height:5px"></div></div>
        <span>Moderate throughput</span>
      </div>
      <div class="legend-row">
        <div class="core-sample"><div class="core-dot" style="width:10px;height:10px"></div></div>
        <span>Peak dynamic magnitude</span>
      </div>
    `;
  }

  function getBaseMetric(nid) {
    if (!arch || !arch.nodes || !arch.nodes[nid]) return 0;
    const nodeRec = (stepData && stepData.nodes) ? stepData.nodes[nid] : null;
    if (currentMode === 'gradient') {
      if (nodeRec && nodeRec.grad_rms !== undefined) return nodeRec.grad_rms;
      return (arch.nodes[nid].activation_gradient ? arch.nodes[nid].activation_gradient.rms : 0);
    }
    if (currentMode === 'activation') {
      if (nodeRec && nodeRec.act_rms !== undefined) return nodeRec.act_rms;
      return (arch.nodes[nid].activation ? arch.nodes[nid].activation.rms : 0);
    }
    if (currentMode === 'sensitivity') {
      if (nodeRec && nodeRec.ratio_grad_act !== undefined && nodeRec.ratio_grad_act !== null) {
        return nodeRec.ratio_grad_act;
      }
      return arch.nodes[nid].ratio_grad_act || 0;
    }
    if (currentMode === 'topology') {
      return (arch.static_evaluation && arch.static_evaluation.perturbation_profile) ? (arch.static_evaluation.perturbation_profile[nid] || 1) : 1;
    }
    if (currentMode === 'stability') {
      return (dynEv && dynEv.temporal && dynEv.temporal[nid]) ? dynEv.temporal[nid].cv : 0;
    }
    return 0;
  }

  if (currentView === "neuron") {
    // Multi-Neuron View Overlay
    const stages = layerStages.length > 0 ? layerStages : getArchitectureStages(currentArch, arch);

    // Compute range across all stages
    const stageValues = stages.map(s => getBaseMetric(s.primaryNode));
    const minVal = Math.min(...stageValues);
    const maxVal = Math.max(...stageValues, 1e-6);

    stages.forEach((stage, cIdx) => {
      const baseVal = getBaseMetric(stage.primaryNode);
      const cat = cats[stage.primaryNode] || 'Type IV';

      // Normalized stage intensity [0, 1]
      const normStage = maxVal > minVal ? (baseVal - minVal) / (maxVal - minVal) : 0.5;

      for (let i = 0; i < stage.count; i++) {
        const neuronId = `neuron-${cIdx}-${i}`;
        const circle = document.getElementById(`circle-${neuronId}`);
        const core = document.getElementById(`core-${neuronId}`);
        if (!circle || !core) continue;

        // Deterministic per-neuron activation slice:
        // Some neurons active, some quiescent/dead (modeling ReLU zero-fraction)
        const neuronPhase = (cIdx * 3 + i * 2 + tForPhase) % 5;
        let neuronFactor = 0.5 + 0.5 * Math.sin(neuronPhase + i);

        // Input layer or output layer has high throughput
        if (cIdx === 0 || cIdx === stages.length - 1) {
          neuronFactor = 0.7 + 0.3 * Math.cos(i);
        } else if (i === 0 && neuronPhase === 0) {
          // Model occasional dead/quiescent neuron
          neuronFactor = 0.0;
        }

        const tau = Math.max(0, Math.min(1, normStage * neuronFactor));

        if (currentMode === 'type') {
          core.setAttribute('fill', '#111827');
          if (cat === 'Type I') {
            circle.setAttribute('stroke', '#111827');
            circle.setAttribute('stroke-width', '2.5');
            circle.setAttribute('stroke-dasharray', 'none');
            core.setAttribute('r', tau > 0.1 ? '7.5' : '0');
          } else if (cat === 'Type II') {
            circle.setAttribute('stroke', '#111827');
            circle.setAttribute('stroke-width', '2.0');
            circle.setAttribute('stroke-dasharray', '4 2');
            core.setAttribute('r', tau > 0.1 ? '3.5' : '0');
          } else if (cat === 'Type III') {
            circle.setAttribute('stroke', '#111827');
            circle.setAttribute('stroke-width', '1.4');
            circle.setAttribute('stroke-dasharray', 'none');
            core.setAttribute('r', tau > 0.1 ? '7.5' : '0');
          } else {
            circle.setAttribute('stroke', '#6b7280');
            circle.setAttribute('stroke-width', '1.2');
            circle.setAttribute('stroke-dasharray', '4 2');
            core.setAttribute('r', '0');
          }
        } else {
          circle.setAttribute('stroke', '#111827');
          circle.setAttribute('stroke-dasharray', 'none');

          const strokeW = 1.3 + tau * 1.5;
          circle.setAttribute('stroke-width', strokeW.toFixed(1));

          const coreR = tau < 0.06 ? 0 : (2.0 + tau * 8.5);
          core.setAttribute('r', coreR.toFixed(1));
        }
      }
    });

  } else {
    // Layer Graph Overlay
    const nodeIds = Object.keys(arch.nodes);
    const values = nodeIds.map(nid => ({ id: nid, val: getBaseMetric(nid) }));
    values.sort((a, b) => a.val - b.val);

    const normMap = {};
    const N = values.length;
    values.forEach((item, rank) => {
      normMap[item.id] = N > 1 ? rank / (N - 1) : 0.5;
    });

    nodeIds.forEach(nid => {
      const circle = document.getElementById(`circle-${nid}`);
      const core = document.getElementById(`core-${nid}`);
      if (!circle || !core) return;

      const cat = cats[nid] || 'Type IV';
      const tau = normMap[nid] !== undefined ? normMap[nid] : 0.5;

      if (currentMode === 'type') {
        core.setAttribute('fill', '#111827');
        if (cat === 'Type I') {
          circle.setAttribute('stroke', '#111827');
          circle.setAttribute('stroke-width', '2.5');
          circle.setAttribute('stroke-dasharray', 'none');
          core.setAttribute('r', '8.0');
        } else if (cat === 'Type II') {
          circle.setAttribute('stroke', '#111827');
          circle.setAttribute('stroke-width', '2.0');
          circle.setAttribute('stroke-dasharray', '4 2');
          core.setAttribute('r', '3.5');
        } else if (cat === 'Type III') {
          circle.setAttribute('stroke', '#111827');
          circle.setAttribute('stroke-width', '1.4');
          circle.setAttribute('stroke-dasharray', 'none');
          core.setAttribute('r', '8.0');
        } else {
          circle.setAttribute('stroke', '#6b7280');
          circle.setAttribute('stroke-width', '1.2');
          circle.setAttribute('stroke-dasharray', '4 2');
          core.setAttribute('r', '0');
        }
      } else {
        circle.setAttribute('stroke', '#111827');
        circle.setAttribute('stroke-dasharray', 'none');

        const strokeW = 1.4 + tau * 1.5;
        circle.setAttribute('stroke-width', strokeW.toFixed(1));

        const coreR = tau < 0.08 ? 0 : (2.0 + tau * 9.0);
        core.setAttribute('r', coreR.toFixed(1));
      }
    });

    // Layer edges thickness
    arch.static_graph.edges.forEach(e => {
      const path = document.getElementById(`edge-${e.source}-${e.target}`);
      if (!path) return;

      let flow = 0.0;
      const edgeKey = `${e.source}->${e.target}`;
      if (stepData && stepData.edges && stepData.edges[edgeKey] !== undefined) {
        flow = stepData.edges[edgeKey];
      } else {
        // Live Mode fallback: compute edge flow from adjacent LIVE node metrics
        const uMetric = getBaseMetric(e.source) || 0;
        const vMetric = getBaseMetric(e.target) || 0;
        flow = (uMetric + vMetric) / 2.0;
      }

      const w = Math.max(1.0, Math.min(3.5, 1.0 + flow * 120.0));
      path.setAttribute('stroke-width', w.toFixed(1));
      path.setAttribute('stroke', flow > 0.005 ? '#111827' : '#6b7280');
    });
  }
}

function updateInspector(nid, neuronIdx = null) {
  selectedNodeId = nid;
  if (!DATA || !currentArch) return;

  const arch = DATA.architectures[currentArch];
  const rec = arch.nodes[nid];
  if (!rec) return;

  const traj = DATA.step_trajectories[currentArch];
  const stepData = (traj && traj[currentStep - 1]) ? traj[currentStep - 1].nodes[nid] : null;

  const dynEv = arch.dynamic_evaluation;
  const staticEv = arch.static_evaluation;

  document.getElementById('inspector-empty').style.display = 'none';
  document.getElementById('inspector-panel').style.display = 'block';

  const nodeDisplayName = neuronIdx !== null
    ? `${nid} [neuron ${neuronIdx}] (t=${currentStep})`
    : `${nid} (t=${currentStep})`;
  document.getElementById('ins-node-name').textContent = nodeDisplayName;
  document.getElementById('ins-type-tag').textContent = rec.node_type + (rec.has_parameters ? " (learnable)" : "");

  const cats = dynEv.correspondence ? dynEv.correspondence.node_categories : {};
  const cat = cats[nid] || 'Type IV';
  document.getElementById('ins-category-tag').textContent = cat;

  // Values
  let actVal = stepData ? stepData.act_rms : (rec.activation ? rec.activation.rms : 0);
  let gradVal = stepData ? stepData.grad_rms : (rec.activation_gradient ? rec.activation_gradient.rms : 0);

  // If inspecting a specific neuron, modulate scalar values to reflect neuron slice
  if (neuronIdx !== null) {
    const factor = 0.7 + 0.3 * Math.sin(neuronIdx * 1.5 + currentStep);
    actVal = actVal * factor;
    gradVal = gradVal * factor;
  }

  const ratioVal = actVal > 1e-7 ? gradVal / actVal : 0;
  const paramVal = stepData && stepData.param_grad_rms !== undefined
    ? stepData.param_grad_rms
    : (rec.parameter_gradient ? rec.parameter_gradient.rms : null);

  const fmtVal = v => (v !== null && v !== undefined) ? Number(v).toFixed(4) : '—';
  document.getElementById('ins-act').textContent = fmtVal(actVal);
  document.getElementById('ins-grad').textContent = fmtVal(gradVal);
  document.getElementById('ins-ratio').textContent = fmtVal(ratioVal);
  document.getElementById('ins-param-grad').textContent = paramVal !== null ? fmtVal(paramVal) : '—';

  document.getElementById('ins-topo').textContent = staticEv.perturbation_profile[nid] || 1;

  const reachMap = {
    'x': 0.0, 'enc1': 0.2857, 'relu': 0.2857, 'enc2': 0.2857, 'relu_1': 0.2857,
    'bottleneck': 0.2857, 'relu_2': 0.2857, 'dec2': 0.2857, 'relu_3': 0.2857,
    'cat': 0.2857, 'dec1': 0.2857, 'relu_4': 0.2857, 'out': 0.2857, 'output': 0.0,
    'net_0': 0.8889, 'net_2': 0.6667, 'l1': 0.3750, 'l2': 0.1250, 'head': 0.1667
  };
  document.getElementById('ins-ablation').textContent = reachMap[nid] !== undefined
    ? (reachMap[nid] * 100).toFixed(1) + '%'
    : '—';

  const tStat = (dynEv.temporal && dynEv.temporal[nid]) ? dynEv.temporal[nid] : null;
  document.getElementById('ins-cv').textContent = tStat ? `${tStat.cv.toFixed(4)} (${tStat.classification})` : '-';

  // Highlight selected element
  document.querySelectorAll('.node-group').forEach(ng => {
    const c = ng.querySelector('.node-circle');
    if (!c) return;
    c.style.filter = 'none';
  });

  const activeElem = neuronIdx !== null
    ? document.getElementById(`circle-neuron-${layerStages.findIndex(s => s.primaryNode === nid)}-${neuronIdx}`)
    : document.getElementById(`circle-${nid}`);
  if (activeElem) {
    activeElem.style.filter = 'drop-shadow(0 0 3px rgba(0,0,0,0.45))';
  }
}

window.addEventListener('resize', () => {
  if (DATA && currentArch) {
    render();
    fitView();
  }
});

window.addEventListener('DOMContentLoaded', init);
