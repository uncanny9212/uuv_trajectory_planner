const chatLog = document.getElementById("chatLog");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const scenarioBadge = document.getElementById("scenarioBadge");
const statusText = document.getElementById("statusText");
const metricStatus = document.getElementById("metricStatus");
const metricDistance = document.getElementById("metricDistance");
const metricConfidence = document.getElementById("metricConfidence");
const metricPoints = document.getElementById("metricPoints");
const reasoningText = document.getElementById("reasoningText");
const decisionOutput = document.getElementById("decisionOutput");
const trajectoryGif = document.getElementById("trajectoryGif");
const emptyState = document.getElementById("emptyState");
const obstacleList = document.getElementById("obstacleList");
const baitList = document.getElementById("baitList");
const safetyDistance = document.getElementById("safetyDistance");
const sendBtn = document.getElementById("sendBtn");
const detectionInput = document.getElementById("detectionInput");
const detectionRange = document.getElementById("detectionRange");
const detectionDepth = document.getElementById("detectionDepth");
const uuvX = document.getElementById("uuvX");
const uuvY = document.getElementById("uuvY");
const uuvZ = document.getElementById("uuvZ");
const parserBadge = document.getElementById("parserBadge");
const parseDetectionBtn = document.getElementById("parseDetectionBtn");
const planDetectionBtn = document.getElementById("planDetectionBtn");

const examples = {
  general: "从 (0,0) 出发到 (1000,800)，安全距离 50，航速 3，避开当前障碍物，并经过当前饵物。",
  area_coverage: "覆盖 500x500 米区域，扫测宽度 50，覆盖率 95%，航速 3。",
  detection:
    "检测到3个接触：\n1. 目标A：位置(150, 200)，静止，置信度0.8\n2. 目标B：北偏东45度方向，距离约600米，移动中\n3. 目标C：位置(400, 500)，半径约50米，疑似礁石",
};

let obstacles = defaultObstacles();
let baits = defaultBaits();

document.getElementById("generalBtn").addEventListener("click", () => fillExample("general"));
document.getElementById("coverageBtn").addEventListener("click", () => fillExample("area_coverage"));
document.getElementById("detectionBtn").addEventListener("click", fillDetectionExample);
document.getElementById("addObstacleBtn").addEventListener("click", addObstacle);
document.getElementById("addBaitBtn").addEventListener("click", addBait);
parseDetectionBtn.addEventListener("click", parseDetection);
planDetectionBtn.addEventListener("click", runDetectionPlan);
safetyDistance.addEventListener("input", () => {
  syncSafetyIntoPrompt();
  validateBaitsUI();
});
chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  runChatPlan();
});

function fillExample(kind) {
  messageInput.value = examples[kind];
  if (kind === "general") {
    obstacles = defaultObstacles();
    baits = defaultBaits();
    safetyDistance.value = "50";
  } else {
    obstacles = [];
    baits = [];
    safetyDistance.value = "50";
  }
  renderObstacles();
  renderBaits();
  messageInput.focus();
}

function fillDetectionExample() {
  detectionInput.value = examples.detection;
  uuvX.value = "0";
  uuvY.value = "0";
  uuvZ.value = "-50";
  detectionRange.value = "500";
  detectionDepth.value = "-50";
  safetyDistance.value = "50";
  detectionInput.focus();
}

async function runChatPlan() {
  const message = messageInput.value.trim();
  if (!message) return;

  const violations = baitViolations();
  if (violations.length > 0) {
    const warning = formatBaitViolation(violations[0]);
    validateBaitsUI();
    setStatus("饵物坐标不可用");
    metricStatus.textContent = "待修正";
    reasoningText.textContent = warning;
    addMessage("assistant", warning);
    return;
  }

  addMessage("user", message);
  messageInput.value = "";
  setStatus("规划中");
  metricStatus.textContent = "运行中";
  showGif(null);

  try {
    const response = await fetch("/api/chat-plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        min_obstacle_distance: numericValue(safetyDistance.value, 50),
        obstacles: obstacles.map((obstacle) => spatialPayload(obstacle, "static")),
        baits: baits.map((bait) => spatialPayload(bait)),
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.message || "规划失败");
    }

    const decision = result.decision;
    renderPlanningResult(result, "任务对话");
    decisionOutput.textContent = JSON.stringify(
      {
        decision_id: decision.decision_id,
        scenario: decision.scenario,
        total_distance: decision.total_distance,
        confidence: decision.confidence,
        constraints_satisfied: decision.constraints_satisfied,
        feedback: decision.feedback,
        obstacles: result.payload.environment.obstacles,
        baits: result.payload.environment.baits,
        min_obstacle_distance: result.payload.mission.constraints.min_obstacle_distance,
      },
      null,
      2
    );
    setStatus("规划完成");
  } catch (error) {
    metricStatus.textContent = "失败";
    reasoningText.textContent = error.message;
    addMessage("assistant", `规划失败：${error.message}`);
    setStatus("规划失败");
  }
}

async function parseDetection() {
  const body = detectionRequestBody();
  if (!body) return;

  setDetectionBusy(true);
  parserBadge.textContent = "解析中";
  setStatus("解析探测语义");

  try {
    const result = await postJson("/api/detection-parse", body);
    renderParsedPayload(result.payload);
    parserBadge.textContent = "已解析";
    setStatus("解析完成");
  } catch (error) {
    parserBadge.textContent = "失败";
    metricStatus.textContent = "失败";
    reasoningText.textContent = error.message;
    addMessage("assistant", `解析失败：${error.message}`);
    setStatus("解析失败");
  } finally {
    setDetectionBusy(false);
  }
}

async function runDetectionPlan() {
  const body = detectionRequestBody();
  if (!body) return;

  setDetectionBusy(true);
  parserBadge.textContent = "规划中";
  setStatus("探测规划中");
  metricStatus.textContent = "运行中";
  showGif(null);

  try {
    const result = await postJson("/api/detection-plan", body);
    renderPlanningResult(result, "探测语义");
    parserBadge.textContent = "已规划";
    setStatus("规划完成");
  } catch (error) {
    parserBadge.textContent = "失败";
    metricStatus.textContent = "失败";
    reasoningText.textContent = error.message;
    addMessage("assistant", `探测规划失败：${error.message}`);
    setStatus("规划失败");
  } finally {
    setDetectionBusy(false);
  }
}

function detectionRequestBody() {
  const detectionText = detectionInput.value.trim();
  if (!detectionText) {
    parserBadge.textContent = "待输入";
    reasoningText.textContent = "请输入探测语义。";
    detectionInput.focus();
    return null;
  }
  return {
    detection_text: detectionText,
    uuv_position: [
      numericValue(uuvX.value, 0),
      numericValue(uuvY.value, 0),
      numericValue(uuvZ.value, -50),
    ],
    default_detection_range: numericValue(detectionRange.value, 500),
    default_depth: numericValue(detectionDepth.value, -50),
    min_obstacle_distance: numericValue(safetyDistance.value, 50),
  };
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

function renderParsedPayload(payload) {
  syncSpatialControlsFromPayload(payload);
  const parsedSummary = formatParsedSummary(payload);
  scenarioBadge.textContent = payload.mission.scenario;
  metricStatus.textContent = "已解析";
  metricDistance.textContent = "--";
  metricConfidence.textContent = "--";
  metricPoints.textContent = "--";
  reasoningText.textContent = parsedSummary;
  decisionOutput.textContent = `${parsedSummary}\n\n原始结构化数据\n${JSON.stringify(payload, null, 2)}`;
  showGif(null);
  addMessage("assistant", `已解析 ${payload.mission.scenario} 输入。`);
}

function renderPlanningResult(result, sourceLabel) {
  const decision = result.decision;
  syncSpatialControlsFromPayload(result.payload);
  const parsedSummary = formatParsedSummary(result.payload);
  scenarioBadge.textContent = decision.scenario;
  metricStatus.textContent = decision.status;
  metricDistance.textContent = `${Number(decision.total_distance).toFixed(1)} m`;
  metricConfidence.textContent = Number(decision.confidence).toFixed(3);
  metricPoints.textContent = String(decision.trajectory.length);
  reasoningText.textContent = `${parsedSummary}\n\n${decision.reasoning_chain || ""}`;
  const planningDetail = JSON.stringify(
    {
      payload: result.payload,
      decision: {
        decision_id: decision.decision_id,
        scenario: decision.scenario,
        total_distance: decision.total_distance,
        confidence: decision.confidence,
        constraints_satisfied: decision.constraints_satisfied,
        feedback: decision.feedback,
      },
    },
    null,
    2
  );
  decisionOutput.textContent = `${parsedSummary}\n\n规划结果\n${planningDetail}`;
  showGif(result.animation_url);
  addMessage(
    "assistant",
    `${sourceLabel}已生成 ${decision.scenario === "area_coverage" ? "区域覆盖" : "点到点"} 轨迹，航程 ${Number(
      decision.total_distance
    ).toFixed(1)} m。`
  );
}

function syncSpatialControlsFromPayload(payload) {
  const environment = payload.environment || {};
  const constraints = (payload.mission || {}).constraints || {};
  obstacles = (environment.obstacles || []).map((obstacle, index) =>
    spatialStateFromPayload(obstacle, `O${String(index + 1).padStart(3, "0")}`, 50)
  );
  baits = (environment.baits || []).map((bait, index) =>
    spatialStateFromPayload(bait, `B${String(index + 1).padStart(3, "0")}`, 40)
  );
  if (constraints.min_obstacle_distance !== undefined) {
    safetyDistance.value = formatNumber(constraints.min_obstacle_distance);
  }
  renderObstacles();
  renderBaits();
}

function spatialStateFromPayload(item, fallbackId, defaultRadius) {
  const position = Array.isArray(item.position) ? item.position : [0, 0, -50];
  return {
    id: item.id || fallbackId,
    x: numericValue(position[0], 0),
    y: numericValue(position[1], 0),
    z: numericValue(position[2], -50),
    radius: numericValue(item.radius, defaultRadius),
    type: item.type || "static",
  };
}

function formatParsedSummary(payload) {
  const mission = payload.mission || {};
  const constraints = mission.constraints || {};
  const environment = payload.environment || {};
  const baits = environment.baits || [];
  const obstacles = environment.obstacles || [];
  const lines = ["语义解析结果"];

  lines.push(`任务类型：${mission.scenario || "--"}`);
  if (mission.target_position) {
    lines.push(`目标点：${formatVector(mission.target_position)}`);
  }
  if (Number(constraints.orbit_turns || 0) > 0) {
    const radiusText = constraints.orbit_radius ? `，半径 ${formatNumber(constraints.orbit_radius)} m` : "";
    lines.push(`动作：先接近目标，再环绕 ${constraints.orbit_turns} 圈${radiusText}`);
  }

  lines.push(`饵物：${baits.length ? `${baits.length} 个` : "无"}`);
  baits.forEach((bait) => {
    lines.push(`- ${bait.id || "B"} 坐标 ${formatVector(bait.position)}，逼近半径 ${formatNumber(bait.radius)} m`);
  });

  lines.push(`障碍物：${obstacles.length ? `${obstacles.length} 个` : "无"}`);
  obstacles.forEach((obstacle) => {
    const type = obstacle.type ? `，类型 ${obstacle.type}` : "";
    lines.push(`- ${obstacle.id || "O"} 坐标 ${formatVector(obstacle.position)}，半径 ${formatNumber(obstacle.radius)} m${type}`);
  });

  lines.push(`安全距离：${formatNumber(constraints.min_obstacle_distance || 0)} m`);
  return lines.join("\n");
}

function formatVector(position) {
  if (!Array.isArray(position)) return "(--, --, --)";
  const values = [position[0], position[1], position[2] ?? -50].map((value) => formatNumber(value));
  return `(${values.join(", ")})`;
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "--";
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

function setDetectionBusy(isBusy) {
  parseDetectionBtn.disabled = isBusy;
  planDetectionBtn.disabled = isBusy;
}

function defaultObstacles() {
  return [
    { id: "O001", x: 300, y: 200, z: -50, radius: 80 },
    { id: "O002", x: 600, y: 500, z: -50, radius: 60 },
    { id: "O003", x: 780, y: 650, z: -50, radius: 45 },
  ];
}

function defaultBaits() {
  return [{ id: "B001", x: 520, y: 350, z: -50, radius: 45 }];
}

function spatialPayload(item, type) {
  const payload = {
    id: item.id,
    position: [item.x, item.y, item.z],
    radius: item.radius,
  };
  if (type) payload.type = type;
  return payload;
}

function addObstacle() {
  const nextIndex = obstacles.length + 1;
  obstacles.push({
    id: `O${String(nextIndex).padStart(3, "0")}`,
    x: 420 + nextIndex * 35,
    y: 260 + nextIndex * 30,
    z: -50,
    radius: 50,
  });
  renderObstacles();
  validateBaitsUI();
}

function addBait() {
  const nextIndex = baits.length + 1;
  baits.push({
    id: `B${String(nextIndex).padStart(3, "0")}`,
    x: 860 + nextIndex * 20,
    y: 260 + nextIndex * 25,
    z: -50,
    radius: 40,
  });
  renderBaits();
}

function removeObstacle(index) {
  obstacles.splice(index, 1);
  renderObstacles();
  validateBaitsUI();
}

function removeBait(index) {
  baits.splice(index, 1);
  renderBaits();
}

function updateObstacle(index, key, value) {
  updateSpatialItem(obstacles, index, key, value, "O");
  validateBaitsUI();
}

function updateBait(index, key, value) {
  updateSpatialItem(baits, index, key, value, "B");
  validateBaitsUI();
}

function updateSpatialItem(collection, index, key, value, prefix) {
  if (key === "id") {
    collection[index][key] = value || `${prefix}${String(index + 1).padStart(3, "0")}`;
  } else {
    collection[index][key] = numericValue(value, key === "z" ? -50 : 0);
  }
}

function renderObstacles() {
  renderSpatialRows({
    container: obstacleList,
    items: obstacles,
    rowClass: "obstacle-row",
    emptyText: "无障碍物",
    remove: removeObstacle,
    update: updateObstacle,
    radiusLabel: "半径",
  });
}

function renderBaits() {
  renderSpatialRows({
    container: baitList,
    items: baits,
    rowClass: "bait-row",
    emptyText: "无饵物",
    remove: removeBait,
    update: updateBait,
    radiusLabel: "逼近半径",
  });
  validateBaitsUI();
}

function renderSpatialRows({ container, items, rowClass, emptyText, remove, update, radiusLabel }) {
  container.innerHTML = "";
  if (items.length === 0) {
    const empty = document.createElement("div");
    empty.className = "spatial-empty";
    empty.textContent = emptyText;
    container.appendChild(empty);
    return;
  }

  items.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = rowClass;
    row.dataset.index = String(index);
    row.innerHTML = `
      <label>ID<input data-key="id" type="text" value="${escapeHtml(item.id)}" /></label>
      <label>X<input data-key="x" type="number" step="10" value="${item.x}" /></label>
      <label>Y<input data-key="y" type="number" step="10" value="${item.y}" /></label>
      <label>Z<input data-key="z" type="number" step="5" value="${item.z}" /></label>
      <label>${radiusLabel}<input data-key="radius" type="number" min="1" step="5" value="${item.radius}" /></label>
      <button type="button" class="danger">删除</button>
    `;
    if (rowClass === "bait-row") {
      const validation = document.createElement("div");
      validation.className = "bait-validation";
      row.appendChild(validation);
    }
    row.querySelectorAll("input").forEach((input) => {
      input.addEventListener("input", () => update(index, input.dataset.key, input.value));
    });
    row.querySelector("button").addEventListener("click", () => remove(index));
    container.appendChild(row);
  });
}

function baitViolations() {
  const safety = numericValue(safetyDistance.value, 50);
  const violations = [];
  baits.forEach((bait, baitIndex) => {
    obstacles.forEach((obstacle) => {
      const distance = Math.hypot(bait.x - obstacle.x, bait.y - obstacle.y);
      const limit = numericValue(obstacle.radius, 0) + safety;
      if (distance < limit) {
        violations.push({ bait, baitIndex, obstacle, distance, limit });
      }
    });
  });
  return violations;
}

function validateBaitsUI() {
  const violations = baitViolations();
  const byBait = new Map();
  violations.forEach((violation) => {
    if (!byBait.has(violation.baitIndex)) {
      byBait.set(violation.baitIndex, violation);
    }
  });

  baitList.querySelectorAll(".bait-row").forEach((row) => {
    const index = Number(row.dataset.index);
    const violation = byBait.get(index);
    const validation = row.querySelector(".bait-validation");
    row.classList.toggle("invalid", Boolean(violation));
    row.querySelectorAll("input").forEach((input) => {
      input.setAttribute("aria-invalid", violation ? "true" : "false");
    });
    if (validation) {
      validation.textContent = violation ? formatBaitViolation(violation) : "";
      validation.style.display = violation ? "block" : "none";
    }
  });

  sendBtn.disabled = violations.length > 0;
  sendBtn.title = violations.length > 0 ? "存在不可用的饵物坐标" : "";
  return violations.length === 0;
}

function formatBaitViolation({ bait, obstacle, distance, limit }) {
  return `饵物 ${bait.id} 坐标不可用：位于障碍物 ${obstacle.id} 的安全距离内，至少需要 ${limit.toFixed(
    1
  )}m，当前距离 ${distance.toFixed(1)}m。`;
}

function syncSafetyIntoPrompt() {
  const value = numericValue(safetyDistance.value, 50);
  if (messageInput.value.includes("安全距离")) {
    messageInput.value = messageInput.value.replace(/安全距离\s*\d+(?:\.\d+)?/, `安全距离 ${value}`);
  }
}

function numericValue(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function addMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;
  bubble.textContent = text;
  chatLog.appendChild(bubble);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function showGif(url) {
  if (!url) {
    trajectoryGif.removeAttribute("src");
    trajectoryGif.style.display = "none";
    emptyState.style.display = "block";
    return;
  }
  trajectoryGif.src = url;
  trajectoryGif.style.display = "block";
  emptyState.style.display = "none";
}

function setStatus(text) {
  statusText.textContent = text;
}

renderObstacles();
renderBaits();
addMessage("assistant", "请直接描述任务。我会返回轨迹动图。");
fillExample("general");
