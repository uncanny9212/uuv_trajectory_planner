const statusText = document.getElementById("statusText");
const simulationBadge = document.getElementById("simulationBadge");
const targetRows = document.getElementById("targetRows");
const addTargetBtn = document.getElementById("addTargetBtn");
const discoveryRangeInput = document.getElementById("discoveryRangeInput");
const dialogInput = document.getElementById("dialogInput");
const runSimulationBtn = document.getElementById("runSimulationBtn");
const trajectoryCanvas = document.getElementById("trajectoryCanvas");
const bearingCanvas = document.getElementById("bearingCanvas");
const agentThoughts = document.getElementById("agentThoughts");
const pilotReport = document.getElementById("pilotReport");

const SIMULATION_DEFAULTS = {
  startPosition: [0, 0, -50],
  stepDistance: 180,
  approachRange: 50,
  bearingNoise: 1,
  orbitTurns: 5,
  orbitRadius: 10,
  maxIterations: 100,
};
const WORLD = {
  min: 0,
  max: 2000,
  resolution: 1,
};
const ROUTE_COLOR = "#087c89";

let targets = [{ x: "", y: "", depth: "" }];
let trajectoryAnimation = null;
let lastResult = null;
let lastNarrativeKey = "";

addTargetBtn.addEventListener("click", () => {
  targets.push({ x: "", y: "", depth: "" });
  renderTargetRows();
});

runSimulationBtn.addEventListener("click", runClosedLoopSimulation);
window.addEventListener("resize", () => {
  if (lastResult) {
    renderResult(lastResult, { animate: false });
  } else {
    drawEmptyCanvases();
  }
});

function renderTargetRows() {
  targetRows.innerHTML = "";
  targets.forEach((target, index) => {
    const row = document.createElement("div");
    row.className = "target-row";
    row.innerHTML = `
      <label>目标<input data-key="label" type="text" value="T${index + 1}" disabled /></label>
      <label>X 0～2000<input data-key="x" type="number" min="0" max="2000" step="1" value="${escapeHtml(target.x)}" /></label>
      <label>Y 0～2000<input data-key="y" type="number" min="0" max="2000" step="1" value="${escapeHtml(target.y)}" /></label>
      <label>深度 m（正值）<input data-key="depth" type="number" min="0" step="1" value="${escapeHtml(target.depth)}" /></label>
      <button type="button" class="danger" ${targets.length <= 1 ? "disabled" : ""}>删除</button>
    `;
    row.querySelectorAll("input:not([disabled])").forEach((input) => {
      input.addEventListener("input", () => {
        targets[index][input.dataset.key] = input.value;
      });
    });
    row.querySelector("button").addEventListener("click", () => {
      targets.splice(index, 1);
      renderTargetRows();
    });
    targetRows.appendChild(row);
  });
}

async function runClosedLoopSimulation() {
  const { positions: targetPositions, error: targetError } = readTargetPositions();
  const { value: discoveryRange, error: discoveryRangeError } = readDiscoveryRange();
  const command = dialogInput.value.trim();
  if (targetError) {
    setStatus(targetError, "待修正");
    return;
  }
  if (discoveryRangeError) {
    setStatus(discoveryRangeError, "待修正");
    discoveryRangeInput.focus();
    return;
  }
  if (!targetPositions.length) {
    setStatus("请输入真实饵物坐标", "待输入");
    return;
  }
  if (!command) {
    setStatus("请输入对话指令", "待输入");
    dialogInput.focus();
    return;
  }

  setBusy(true);
  setStatus("闭环仿真运行中", "运行中");
  agentThoughts.textContent = "Agent正在根据方位观测滚动决策。";
  pilotReport.textContent = "驾驶员待命，等待第一轮航向指令。";
  drawEmptyCanvases();

  try {
    const response = await postJson("/api/simulation/interactive", {
      target_positions: targetPositions,
      bearing_text: command,
      start_position: SIMULATION_DEFAULTS.startPosition,
      default_step: SIMULATION_DEFAULTS.stepDistance,
      approach_range: discoveryRange,
      bearing_noise_deg: SIMULATION_DEFAULTS.bearingNoise,
      orbit_turns: SIMULATION_DEFAULTS.orbitTurns,
      orbit_radius: SIMULATION_DEFAULTS.orbitRadius,
      max_iterations: SIMULATION_DEFAULTS.maxIterations,
    });
    lastResult = response.result;
    renderResult(lastResult, { animate: true });
    setStatus(lastResult.status === "success" ? "闭环仿真完成" : "闭环仿真未完成", "已完成");
  } catch (error) {
    setStatus("仿真失败", "失败");
    agentThoughts.textContent = error.message;
    pilotReport.textContent = "报告：仿真请求失败，未执行航行。";
  } finally {
    setBusy(false);
  }
}

function readTargetPositions() {
  const positions = [];
  for (let index = 0; index < targets.length; index += 1) {
    const target = targets[index];
    const hasAnyValue = target.x !== "" || target.y !== "" || target.depth !== "";
    if (!hasAnyValue) continue;

    const x = Number(target.x);
    const y = Number(target.y);
    if (!validWorldCoordinate(x) || !validWorldCoordinate(y)) {
      return {
        positions: [],
        error: `目标 ${index + 1} 的 X/Y 必须是 ${WORLD.min}～${WORLD.max} 范围内的整数坐标。`,
      };
    }
    const depthInput = target.depth === "" ? 50 : Number(target.depth);
    if (!Number.isFinite(depthInput) || depthInput < 0 || !Number.isInteger(depthInput)) {
      return { positions: [], error: `目标 ${index + 1} 的深度必须是非负整数。` };
    }
    positions.push([x, y, -Math.abs(depthInput)]);
  }
  return { positions, error: "" };
}

function validWorldCoordinate(value) {
  return Number.isInteger(value) && value >= WORLD.min && value <= WORLD.max;
}

function readDiscoveryRange() {
  const value = Number(discoveryRangeInput.value);
  if (!Number.isFinite(value) || value <= 0) {
    return { value: SIMULATION_DEFAULTS.approachRange, error: "发现距离必须是大于 0 的数字。" };
  }
  return { value, error: "" };
}

async function postJson(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.message || "请求失败");
  }
  return result;
}

function renderResult(result, options = {}) {
  drawBearingRecord(result);
  lastNarrativeKey = "";
  drawTrajectory(result, options.animate, (progress, update = {}) => {
    renderSynchronizedNarrative(result, progress, {
      force: Boolean(update.force),
      autoScroll: options.animate !== false,
      state: update.state,
    });
  });
}

function drawTrajectory(result, animate = true, onProgress = () => {}) {
  const segments = normalizedTrajectorySegments(result);
  const bounds = worldBounds();

  if (trajectoryAnimation) {
    cancelAnimationFrame(trajectoryAnimation);
    trajectoryAnimation = null;
  }

  const renderFrame = (progress) => {
    const { ctx, width, height } = prepareCanvas(trajectoryCanvas);
    const state = narrativeState(result, progress);
    drawPlotBackground(ctx, width, height, bounds);
    drawOriginMarker(ctx, width, height, bounds);
    drawTargets(ctx, width, height, bounds, result, state);
    drawSequentialSegments(ctx, width, height, bounds, segments, progress);
    drawUuvMarker(ctx, width, height, bounds, currentPointOnSegments(segments, progress));
    onProgress(progress, { state });
  };

  if (!animate) {
    renderFrame(1);
    onProgress(1, { force: true });
    return;
  }

  onProgress(0, { force: true });
  const startedAt = performance.now();
  const duration = 10000;
  const step = (now) => {
    const progress = Math.min(1, (now - startedAt) / duration);
    renderFrame(progress);
    if (progress < 1) {
      trajectoryAnimation = requestAnimationFrame(step);
    }
  };
  trajectoryAnimation = requestAnimationFrame(step);
}

function drawBearingRecord(result) {
  const observations = result.bearing_history || [];
  const { ctx, width, height } = prepareCanvas(bearingCanvas);
  ctx.clearRect(0, 0, width, height);
  drawChartFrame(ctx, width, height);

  if (!observations.length) {
    drawCenteredText(ctx, width, height, "等待角度记录");
    return;
  }

  const padding = chartPadding();
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xFor = (index) =>
    observations.length === 1
      ? padding.left + plotWidth / 2
      : padding.left + (index / (observations.length - 1)) * plotWidth;
  const yFor = (angle) => padding.top + plotHeight - (Number(angle) / 360) * plotHeight;

  drawAngleGrid(ctx, width, height);
  drawAngleLine(ctx, observations, xFor, yFor, "angle", "#087c89", false);

  ctx.fillStyle = "#172026";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillText("观测方位", padding.left, 20);
}

function drawEmptyCanvases() {
  const trajectory = prepareCanvas(trajectoryCanvas);
  trajectory.ctx.clearRect(0, 0, trajectory.width, trajectory.height);
  drawCenteredText(trajectory.ctx, trajectory.width, trajectory.height, "等待轨迹");

  const bearing = prepareCanvas(bearingCanvas);
  bearing.ctx.clearRect(0, 0, bearing.width, bearing.height);
  drawCenteredText(bearing.ctx, bearing.width, bearing.height, "等待角度记录");
}

function worldBounds() {
  return {
    minX: WORLD.min,
    maxX: WORLD.max,
    minY: WORLD.min,
    maxY: WORLD.max,
  };
}

function projectPoint(point, width, height, bounds) {
  const worldWidth = bounds.maxX - bounds.minX;
  const worldHeight = bounds.maxY - bounds.minY;
  return [
    ((Number(point[0]) - bounds.minX) / worldWidth) * width,
    height - ((Number(point[1]) - bounds.minY) / worldHeight) * height,
  ];
}

function drawPlotBackground(ctx, width, height, bounds) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e6ecef";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 5; index += 1) {
    const x = (width * index) / 5;
    const y = (height * index) / 5;
    const lineX = index === 5 ? width - 0.5 : x + 0.5;
    const lineY = index === 5 ? height - 0.5 : y + 0.5;
    ctx.beginPath();
    ctx.moveTo(lineX, 0);
    ctx.lineTo(lineX, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, lineY);
    ctx.lineTo(width, lineY);
    ctx.stroke();
  }
  ctx.fillStyle = "#667780";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillText(`X ${formatNumber(bounds.minX)} ~ ${formatNumber(bounds.maxX)} m`, Math.max(8, width - 142), height - 8);
  ctx.fillText(`Y ${formatNumber(bounds.minY)} ~ ${formatNumber(bounds.maxY)} m`, Math.max(8, width - 142), 16);
}

function drawOriginMarker(ctx, width, height, bounds) {
  const [x, y] = projectPoint([0, 0, 0], width, height, bounds);
  ctx.fillStyle = "#157f4f";
  ctx.strokeStyle = "#157f4f";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(x, y - 24);
  ctx.lineTo(x, y);
  ctx.lineTo(x + 24, y);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x + 7, y - 7, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#157f4f";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillText("起点 (0,0)", x + 8, y - 10);
}

function drawTargets(ctx, width, height, bounds, result, state = narrativeState(result, 1)) {
  const targetsInResult = result.target_positions || [];
  const route = Array.isArray(result.target_route) ? result.target_route : [result.active_target_index];
  const completedCount = Number(result.completed_target_count || 0);
  const discoveredTargets = new Set(state.completedTargetRuns.map((run) => Number(run.target_index)));
  targetsInResult.forEach((target, index) => {
    if (!discoveredTargets.has(index)) return;
    const [x, y] = projectPoint(target, width, height, bounds);
    const sequence = route.indexOf(index) + 1;
    const isPlanned = sequence > 0;
    const isCompleted = isPlanned && sequence <= completedCount;
    ctx.fillStyle = isCompleted ? "#157f4f" : isPlanned ? "#c04d31" : "#b2bdc2";
    ctx.strokeStyle = isCompleted ? "#157f4f" : isPlanned ? "#c04d31" : "#7f8d93";
    ctx.lineWidth = isPlanned ? 2 : 1;
    ctx.beginPath();
    ctx.arc(x, y, isPlanned ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
    if (isPlanned) {
      ctx.beginPath();
      ctx.moveTo(x - 9, y - 9);
      ctx.lineTo(x + 9, y + 9);
      ctx.moveTo(x + 9, y - 9);
      ctx.lineTo(x - 9, y + 9);
      ctx.stroke();
    }
    ctx.fillStyle = "#172026";
    ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
    ctx.fillText(sequence ? `T${index + 1} / ${sequence}` : `T${index + 1}`, x + 9, y - 9);
  });
}

function normalizedTrajectorySegments(result) {
  const sourceSegments = Array.isArray(result.trajectory_segments) ? result.trajectory_segments : [];
  if (sourceSegments.length) {
    return stitchSegments(sourceSegments
      .map((segment) => ({
        kind: segment.kind || "approach",
        targetIndex: Number(segment.target_index || 0),
        targetSequence: Number(segment.target_sequence || 0),
        distance: Number(segment.distance || 0),
        points: normalizePoints(segment.points || []),
      }))
      .filter((segment) => segment.points.length));
  }

  const approachPoints = normalizePoints((result.uuv_history || []).map((item) => item.position));
  const orbitPoints = normalizePoints((result.orbit_history || []).map((item) => item.position));
  const segments = [];
  if (approachPoints.length) {
    segments.push({
      kind: "approach",
      targetIndex: Number(result.active_target_index || 0),
      targetSequence: 1,
      distance: Number(result.approach_distance || 0),
      points: approachPoints,
    });
  }
  if (orbitPoints.length) {
    segments.push({
      kind: "orbit",
      targetIndex: Number(result.active_target_index || 0),
      targetSequence: 1,
      distance: Number(result.orbit_distance || 0),
      points: orbitPoints,
    });
  }
  return stitchSegments(segments);
}

function normalizePoints(points) {
  return points
    .filter((point) => Array.isArray(point) && point.length >= 2)
    .map((point) => [Number(point[0]), Number(point[1]), Number(point[2] || 0)])
    .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]));
}

function drawSequentialSegments(ctx, width, height, bounds, segments, progress) {
  const totalDistance = totalSegmentDistance(segments);
  let remainingDistance = Math.max(0, Math.min(1, progress)) * totalDistance;
  segments.forEach((segment) => {
    const distance = segmentDistance(segment);
    if (remainingDistance <= 0) return;
    const segmentProgress = Math.min(1, remainingDistance / distance);
    drawPath(ctx, width, height, bounds, segment.points, ROUTE_COLOR, segmentProgress);
    remainingDistance -= distance;
  });
}

function currentPointOnSegments(segments, progress) {
  if (!segments.length) return SIMULATION_DEFAULTS.startPosition;
  const totalDistance = totalSegmentDistance(segments);
  let remainingDistance = Math.max(0, Math.min(1, progress)) * totalDistance;
  let lastPoint = segments[0].points[0];

  for (const segment of segments) {
    if (!segment.points.length) continue;
    const distance = segmentDistance(segment);
    if (remainingDistance <= distance) {
      return pointAtPolyline(segment.points, remainingDistance / distance);
    }
    remainingDistance -= distance;
    lastPoint = segment.points[segment.points.length - 1];
  }
  return lastPoint;
}

function totalSegmentDistance(segments) {
  return segments.reduce((total, segment) => total + segmentDistance(segment), 0) || 1;
}

function segmentDistance(segment) {
  const explicitDistance = Number(segment.distance);
  if (Number.isFinite(explicitDistance) && explicitDistance > 0) return explicitDistance;
  const computedDistance = polylineWorldLength(segment.points || []);
  return computedDistance > 0 ? computedDistance : 1;
}

function polylineWorldLength(points) {
  let distance = 0;
  for (let index = 1; index < points.length; index += 1) {
    distance += Math.hypot(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1]);
  }
  return distance;
}

function stitchSegments(segments) {
  const stitched = [];
  let previousEnd = null;
  segments.forEach((segment) => {
    const points = [...segment.points];
    if (previousEnd && points.length) {
      const first = points[0];
      const gap = Math.hypot(first[0] - previousEnd[0], first[1] - previousEnd[1]);
      if (gap > 0.001) {
        points.unshift(previousEnd);
      }
    }
    if (points.length) {
      previousEnd = points[points.length - 1];
    }
    stitched.push({ ...segment, points });
  });
  return stitched;
}

function pointAtPolyline(points, progress) {
  if (!points.length) return SIMULATION_DEFAULTS.startPosition;
  const visible = partialPolyline(points, progress);
  return visible[visible.length - 1] || points[0];
}

function drawPath(ctx, width, height, bounds, points, color, progress) {
  if (!points.length || progress <= 0) return;
  const projected = points.map((point) => projectPoint(point, width, height, bounds));
  const visible = partialPolyline(projected, progress);
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  visible.forEach(([x, y], index) => {
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
}

function drawUuvMarker(ctx, width, height, bounds, point) {
  if (!point) return;
  const current = projectPoint(point, width, height, bounds);
  ctx.fillStyle = "#157f4f";
  ctx.beginPath();
  ctx.arc(current[0], current[1], 5, 0, Math.PI * 2);
  ctx.fill();
}

function partialPolyline(points, progress) {
  if (points.length <= 1) return points;
  const clamped = Math.max(0, Math.min(1, progress));
  const totalLength = polylineWorldLength(points);
  if (totalLength <= 0) return [points[0]];
  const targetLength = totalLength * clamped;
  const visible = [points[0]];
  let traveled = 0;
  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1];
    const end = points[index];
    const segmentLength = Math.hypot(end[0] - start[0], end[1] - start[1]);
    if (segmentLength <= 0) continue;
    if (traveled + segmentLength < targetLength) {
      visible.push(end);
      traveled += segmentLength;
      continue;
    }
    const fraction = (targetLength - traveled) / segmentLength;
    const interpolated = start.map((value, dimension) => {
      const endValue = end[dimension] === undefined ? value : end[dimension];
      return value + (endValue - value) * fraction;
    });
    visible.push(interpolated);
    return visible;
  }
  return visible;
}

function drawChartFrame(ctx, width, height) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const padding = chartPadding();
  ctx.strokeStyle = "#d8e0e3";
  ctx.lineWidth = 1;
  ctx.strokeRect(
    padding.left,
    padding.top,
    width - padding.left - padding.right,
    height - padding.top - padding.bottom
  );
}

function drawAngleGrid(ctx, width, height) {
  const padding = chartPadding();
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  [0, 90, 180, 270, 360].forEach((angle) => {
    const y = padding.top + plotHeight - (angle / 360) * plotHeight;
    ctx.strokeStyle = "#e6ecef";
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.fillStyle = "#667780";
    ctx.fillText(String(angle), 12, y + 4);
  });
  ctx.fillStyle = "#667780";
  ctx.fillText("观测轮次", width / 2 - 24, height - 10);
}

function drawAngleLine(ctx, observations, xFor, yFor, key, color, dashed) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = dashed ? 1.8 : 2.6;
  ctx.setLineDash(dashed ? [6, 5] : []);
  ctx.beginPath();
  observations.forEach((item, index) => {
    const x = xFor(index);
    const y = yFor(item[key]);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  ctx.setLineDash([]);
  observations.forEach((item, index) => {
    const x = xFor(index);
    const y = yFor(item[key]);
    ctx.beginPath();
    ctx.arc(x, y, dashed ? 3 : 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(320, rect.width);
  const height = Math.max(260, rect.height);
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * ratio);
  canvas.height = Math.floor(height * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width, height };
}

function chartPadding() {
  return { top: 32, right: 22, bottom: 34, left: 42 };
}

function drawCenteredText(ctx, width, height, text) {
  ctx.fillStyle = "#667780";
  ctx.font = "14px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
  ctx.textAlign = "start";
}

function renderSynchronizedNarrative(result, progress, options = {}) {
  const state = options.state || narrativeState(result, progress);
  const key = [
    state.thoughtDecisions.length,
    state.pilotDecisions.length,
    state.completedTargetRuns.length,
    state.visibleOrbitTurns.toFixed(1),
    state.isComplete ? "done" : "running",
  ].join(":");
  if (!options.force && key === lastNarrativeKey) return;

  agentThoughts.textContent = formatAgentThoughts(result, state);
  pilotReport.textContent = formatPilotReport(result, state);
  lastNarrativeKey = key;
  if (options.autoScroll) {
    agentThoughts.scrollTop = agentThoughts.scrollHeight;
    pilotReport.scrollTop = pilotReport.scrollHeight;
  }
}

function narrativeState(result, progress = 1) {
  const clampedProgress = Math.max(0, Math.min(1, Number(progress)));
  const segments = normalizedTrajectorySegments(result);
  const windows = segmentWindows(segments);
  const totalDistance = windows.totalDistance || 1;
  const currentDistance = clampedProgress * totalDistance;
  const decisions = result.decisions || [];
  const thoughtDecisionIndexes = new Set();
  const pilotDecisionIndexes = new Set();
  const localDistanceBySequence = new Map();

  decisions.forEach((decision, index) => {
    const sequence = Number(decision.target_sequence || 1);
    const approach = windows.approachBySequence.get(sequence);
    const approachStart = approach ? approach.start : 0;
    const approachDistance = approach ? Math.max(approach.distance, 1) : totalDistance;
    const localBefore = localDistanceBySequence.get(sequence) || 0;
    const moveDistance = Math.max(0, Number(decision.executed_distance || 0));
    const localAfter = Math.min(approachDistance, localBefore + moveDistance);
    const thoughtDistance = approachStart + localBefore;
    const pilotDistance = approachStart + localAfter;

    if (thoughtDistance <= currentDistance + 0.001) {
      thoughtDecisionIndexes.add(index);
    }
    if (pilotDistance <= currentDistance + 0.001) {
      pilotDecisionIndexes.add(index);
    }
    localDistanceBySequence.set(sequence, localAfter);
  });

  const completedTargetRuns = (result.target_runs || []).filter((run) => {
    const sequence = Number(run.target_sequence || 1);
    const orbit = windows.orbitBySequence.get(sequence);
    const approach = windows.approachBySequence.get(sequence);
    const completionDistance = orbit ? orbit.end : approach ? approach.end : totalDistance;
    return completionDistance <= currentDistance + 0.001;
  });

  return {
    progress: clampedProgress,
    currentDistance,
    totalDistance,
    isComplete: clampedProgress >= 0.999,
    thoughtDecisions: decisions.filter((_, index) => thoughtDecisionIndexes.has(index)),
    pilotDecisions: decisions.filter((_, index) => pilotDecisionIndexes.has(index)),
    completedTargetRuns,
    visibleOrbitTurns: visibleOrbitTurns(result, windows, currentDistance),
  };
}

function segmentWindows(segments) {
  const approachBySequence = new Map();
  const orbitBySequence = new Map();
  let cursor = 0;
  segments.forEach((segment) => {
    const distance = segmentDistance(segment);
    const window = {
      start: cursor,
      end: cursor + distance,
      distance,
      targetIndex: segment.targetIndex,
      targetSequence: segment.targetSequence,
      kind: segment.kind,
    };
    if (segment.kind === "orbit") {
      orbitBySequence.set(Number(segment.targetSequence || 1), window);
    } else {
      approachBySequence.set(Number(segment.targetSequence || 1), window);
    }
    cursor = window.end;
  });
  return {
    approachBySequence,
    orbitBySequence,
    totalDistance: cursor,
  };
}

function visibleOrbitTurns(result, windows, currentDistance) {
  let turns = 0;
  windows.orbitBySequence.forEach((window) => {
    const run = targetRunBySequence(result, window.targetSequence);
    const turnsForTarget = Number(run?.orbit_turns_completed || (result.constraints || {}).orbit_turns || 0);
    if (currentDistance >= window.end) {
      turns += turnsForTarget;
    } else if (currentDistance > window.start && window.distance > 0) {
      turns += ((currentDistance - window.start) / window.distance) * turnsForTarget;
    }
  });
  return Math.max(0, turns);
}

function targetRunBySequence(result, sequence) {
  return (result.target_runs || []).find((run) => Number(run.target_sequence || 0) === Number(sequence));
}

function formatAgentThoughts(result, state = narrativeState(result, 1)) {
  const measurement = result.bearing_measurement || {};
  const constraints = result.constraints || {};
  const coordinateSystem = result.coordinate_system || {};
  const xRange = coordinateSystem.x_range || [WORLD.min, WORLD.max];
  const yRange = coordinateSystem.y_range || [WORLD.min, WORLD.max];
  const start = coordinateSystem.start_position || SIMULATION_DEFAULTS.startPosition;
  const route = Array.isArray(result.target_route) ? result.target_route : [result.active_target_index || 0];
  const routeText = route.map((targetIndex) => `目标${Number(targetIndex) + 1}`).join(" -> ");
  const targetCount = (result.target_positions || []).length;
  const visibleCompletedCount = state.completedTargetRuns.length;
  const initialBearings = Array.isArray(measurement.initial_bearings) && measurement.initial_bearings.length
    ? measurement.initial_bearings
    : [measurement.initial_bearing];
  const lines = [
    state.isComplete ? result.summary || "闭环仿真结果" : "闭环仿真进行中：Agent按轨迹动画进度滚动输出判断。",
    "",
    `坐标系：X ${formatNumber(xRange[0])}～${formatNumber(xRange[1])}，Y ${formatNumber(
      yRange[0]
    )}～${formatNumber(yRange[1])}，分辨率 ${formatNumber(coordinateSystem.resolution || WORLD.resolution)} m`,
    `起点：(${formatNumber(start[0])}, ${formatNumber(start[1])})；方向角默认基于当前起点，0°为正北，90°为正东`,
    `状态：${state.isComplete ? statusLabel(result.status) : "执行中"}`,
    `侦察路线：${routeText || "--"}`,
    `当前进度：${formatNumber(state.progress * 100)}%`,
    `已完成目标：${visibleCompletedCount} / ${targetCount}`,
    `初始观测：${initialBearings.map((bearing) => `${formatNumber(bearing)}°`).join("，")}`,
    `已显示迭代：${state.thoughtDecisions.length} / ${result.iterations || 0}`,
    `当前绕行：${formatNumber(state.visibleOrbitTurns)} / ${result.orbit_turns_completed || 0} 圈`,
    "",
  ];

  const assessmentLines = formatBearingAssessmentLines(result);
  if (assessmentLines.length) {
    lines.push("方位可信度评估：");
    lines.push(...assessmentLines);
    lines.push("");
  }

  if (state.completedTargetRuns.length) {
    lines.push("已完成目标结果：");
    state.completedTargetRuns.forEach((run) => {
      lines.push(
        `目标${Number(run.target_index) + 1}：${statusLabel(run.status)}，确认坐标 ${formatVector(
          run.target_position
        )}，迭代 ${
          run.iterations || 0
        }轮，发现时距离 ${formatNumber(run.final_distance)} m，绕行 ${formatNumber(run.orbit_distance)} m`
      );
    });
    lines.push("");
  }

  state.thoughtDecisions.forEach((decision) => {
    const observation = decision.observation || {};
    const targetLabel =
      decision.target_sequence !== undefined
        ? `第${decision.target_sequence}站 / 目标${Number(decision.target_index) + 1}`
        : "当前目标";
    lines.push(
      [
        `第 ${decision.iteration} 轮（${targetLabel}）`,
        formatObservationForDecision(decision),
        `Agent思考：${formatReasoningForVisibility(decision)}`,
        `执行：${decision.executed_action}，航向 ${formatNumber(
          decision.executed_heading
        )}°，前进 ${formatNumber(decision.executed_distance)} m`,
        formatDecisionResult(decision),
        `反馈：${decision.feedback || "--"}`,
      ].join("\n")
    );
    lines.push("");
  });
  return lines.join("\n");
}

function formatPilotReport(result, state = narrativeState(result, 1)) {
  const completedCount = state.completedTargetRuns.length;
  const lines = ["驾驶员反馈："];

  state.pilotDecisions.forEach((decision) => {
    lines.push(formatPilotDecisionLine(decision, result));
  });

  if (!state.pilotDecisions.length) {
    lines.push("待命，等待方位指令。");
  }

  if (state.isComplete && result.status === "success") {
    lines.push(formatPilotConclusion(state.completedTargetRuns));
  } else if (state.isComplete) {
    lines.push(`发现${completedCount}个目标，任务未完成。${result.failure_reason || ""}`);
  } else {
    lines.push(`已发现${completedCount}个目标，继续执行。`);
  }
  return lines.join("\n");
}

function formatBearingAssessmentLines(result) {
  return (result.bearing_assessments || []).map((assessment) => {
    const status = assessment.status === "trusted" ? "待连续观测验证" : "需要复测确认";
    return `第${assessment.bearing_index}个方位：${formatNumber(
      assessment.bearing
    )}°，状态：${status}；发现前不使用真实坐标和距离。`;
  });
}

function formatObservationForDecision(decision) {
  const observation = decision.observation || {};
  if (decision.target_discovered) {
    return `观测：${formatNumber(observation.angle)}°；进入发现半径，目标坐标 ${formatVector(
      decision.discovered_position
    )}`;
  }
  return `观测：${formatNumber(observation.angle)}°；仅有方位信息，距离和目标坐标未知`;
}

function formatReasoningForVisibility(decision) {
  const reasoning = decision.reasoning || "--";
  if (decision.target_discovered) return reasoning;
  return reasoning
    .replace(/估算距离约[-+]?\d+(?:\.\d+)?m。/g, "距离未知，当前仅依据连续方位角滚动判断。")
    .replace(/距目标\s*[-+]?\d+(?:\.\d+)?m/g, "目标距离未知");
}

function formatDecisionResult(decision) {
  if (decision.target_discovered) {
    return `结果：进入发现半径，确认目标坐标 ${formatVector(
      decision.discovered_position
    )}；发现时距离 ${formatNumber(decision.distance_after)} m`;
  }
  return "结果：尚未进入发现半径，目标距离和坐标未知，继续复测方位";
}

function formatPilotDecisionLine(decision, result) {
  if (decision.target_discovered) {
    const run = targetRunBySequence(result, decision.target_sequence);
    const depth = run?.target_depth ?? targetDepth(decision.discovered_position);
    const deepNote =
      run?.is_deep_target || depth > 10
        ? `深度${formatNumber(depth)}m，重点复核，绕航${run?.orbit_turns_completed || "--"}圈。`
        : `深度${formatNumber(depth)}m，常规复核。`;
    return `第${decision.iteration}轮：发现目标${Number(decision.target_index) + 1}，位置${formatVector(
      decision.discovered_position
    )}，${deepNote}`;
  }
  if ((decision.warnings || []).some((warning) => String(warning).includes("疑似虚假信息"))) {
    return `第${decision.iteration}轮：复测方位${formatNumber(
      (decision.observation || {}).angle
    )}°，初始方位疑似虚假，改按复测方位执行。`;
  }
  return `第${decision.iteration}轮：航向${formatNumber(decision.executed_heading)}°，前进${formatNumber(
    decision.executed_distance
  )}m。`;
}

function formatPilotConclusion(targetRuns) {
  const positions = targetRuns.map((run) => formatVector(run.target_position));
  const positionText = positions.length ? `${positions.join("、")}等` : "无";
  return `发现${targetRuns.length}个目标，位置分别为${positionText}，已完成抵近侦察。`;
}

function statusLabel(status) {
  return status === "success" ? "成功发现目标" : "未完成";
}

function formatVector(position) {
  if (!Array.isArray(position)) return "(--, --, --)";
  return `(${formatNumber(position[0])}, ${formatNumber(position[1])}, ${formatNumber(targetDepth(position))})`;
}

function targetDepth(position) {
  if (!Array.isArray(position) || position.length < 3) return 0;
  return Math.abs(Number(position[2]));
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

function setStatus(text, badge) {
  statusText.textContent = text;
  simulationBadge.textContent = badge;
}

function setBusy(isBusy) {
  runSimulationBtn.disabled = isBusy;
  addTargetBtn.disabled = isBusy;
  discoveryRangeInput.disabled = isBusy;
  dialogInput.disabled = isBusy;
  targetRows.querySelectorAll("input, button").forEach((element) => {
    element.disabled = isBusy || (element.dataset.key === "label") || (element.classList.contains("danger") && targets.length <= 1);
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

renderTargetRows();
drawEmptyCanvases();
