const statusText = document.getElementById("statusText");
const simulationBadge = document.getElementById("simulationBadge");
const targetRows = document.getElementById("targetRows");
const addTargetBtn = document.getElementById("addTargetBtn");
const discoveryRangeInput = document.getElementById("discoveryRangeInput");
const sonarTriggerRangeInput = document.getElementById("sonarTriggerRangeInput");
const apiBaseUrlInput = document.getElementById("apiBaseUrlInput");
const apiKeyInput = document.getElementById("apiKeyInput");
const agentChatMessages = document.getElementById("agentChatMessages");
const agentChatInput = document.getElementById("agentChatInput");
const sendAgentMessageBtn = document.getElementById("sendAgentMessageBtn");
const runSimulationBtn = document.getElementById("runSimulationBtn");
const trajectoryCanvas = document.getElementById("trajectoryCanvas");
const bearingCanvas = document.getElementById("bearingCanvas");
const sonarCanvas = document.getElementById("sonarCanvas");
const sonarStatusBadge = document.getElementById("sonarStatusBadge");
const sonarMetadata = document.getElementById("sonarMetadata");
const agentThoughts = document.getElementById("agentThoughts");
const pilotReport = document.getElementById("pilotReport");
const authorizationModal = document.getElementById("authorizationModal");
const authorizationMessage = document.getElementById("authorizationMessage");
const authorizationApproveBtn = document.getElementById("authorizationApproveBtn");
const authorizationHoldBtn = document.getElementById("authorizationHoldBtn");
const authorizationCloseBtn = document.getElementById("authorizationCloseBtn");

const SIMULATION_DEFAULTS = {
  startPosition: [0, 0, -50],
  stepDistance: 180,
  approachRange: 50,
  sonarTriggerRange: 15,
  bearingNoise: 1,
  orbitTurns: 5,
  orbitRadius: 10,
  maxIterations: 100,
  animationDurationMs: 20000,
};
const WORLD = {
  min: 0,
  max: 2000,
  resolution: 1,
};
const ROUTE_COLOR = "#087c89";
const TARGET_TYPE_OPTIONS = [
  ["submarine", "潜艇"],
  ["torpedo", "鱼雷"],
  ["ship", "水面舰"],
  ["reef", "礁石"],
  ["unknown", "不明"],
];

let targets = [{ x: "", y: "", depth: "", type: "submarine", iff: "blue" }];
let trajectoryAnimation = null;
let lastResult = null;
let lastNarrativeKey = "";
let lastDialogIntent = null;
let authorizationResume = null;
let authorizationHold = null;
let authorizationCloseAction = null;
let agentChatHistory = [
  {
    role: "assistant",
    content: "我是UUV闭环仿真Agent。你可以用对话配置目标坐标、方位、发现距离、反馈语气，也可以询问决策依据。",
  },
];

addTargetBtn.addEventListener("click", () => {
  targets.push({ x: "", y: "", depth: "", type: "submarine", iff: "blue" });
  renderTargetRows();
});

runSimulationBtn.addEventListener("click", runClosedLoopSimulation);
sendAgentMessageBtn.addEventListener("click", sendAgentMessage);
agentChatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    sendAgentMessage();
  }
});
authorizationApproveBtn.addEventListener("click", () => {
  showAuthorizationResult("approved");
});
authorizationHoldBtn.addEventListener("click", () => {
  showAuthorizationResult("held");
});
authorizationCloseBtn.addEventListener("click", () => {
  hideAuthorizationModal();
  if (authorizationCloseAction) {
    authorizationCloseAction();
  }
});
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
      <label>类型<select data-key="type">${targetTypeOptionsHtml(target.type)}</select></label>
      <label>IFF<select data-key="iff">
        <option value="blue" ${target.iff !== "red" ? "selected" : ""}>蓝方/敌方</option>
        <option value="red" ${target.iff === "red" ? "selected" : ""}>红方/中立</option>
      </select></label>
      <button type="button" class="danger" ${targets.length <= 1 ? "disabled" : ""}>删除</button>
    `;
    row.querySelectorAll("input:not([disabled]), select").forEach((input) => {
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

function targetTypeOptionsHtml(selectedType = "submarine") {
  return TARGET_TYPE_OPTIONS.map(([value, label]) => {
    const selected = value === selectedType ? "selected" : "";
    return `<option value="${value}" ${selected}>${label}</option>`;
  }).join("");
}

async function runClosedLoopSimulation() {
  const pendingMessage = agentChatInput.value.trim();
  if (pendingMessage) {
    appendAgentChatMessage("user", pendingMessage);
    agentChatInput.value = "";
  }
  const command = conversationTextForSimulation();
  if (!command) {
    setStatus("请先和Agent说明任务", "待输入");
    agentChatInput.focus();
    return;
  }
  const dialogIntent = parseDialogIntent(command);
  const { positions: targetPositions, error: targetError } = resolveSimulationTargets(dialogIntent);
  const { profiles: targetProfiles, error: profileError } = resolveTargetProfiles(dialogIntent, targetPositions.length);
  const { value: discoveryRange, error: discoveryRangeError } = resolveDiscoveryRange(dialogIntent);
  const { value: sonarTriggerRange, error: sonarRangeError } = resolveSonarTriggerRange(dialogIntent);
  if (targetError) {
    setStatus(targetError, "待修正");
    return;
  }
  if (profileError) {
    setStatus(profileError, "待修正");
    return;
  }
  if (discoveryRangeError) {
    setStatus(discoveryRangeError, "待修正");
    discoveryRangeInput.focus();
    return;
  }
  if (sonarRangeError) {
    setStatus(sonarRangeError, "待修正");
    sonarTriggerRangeInput.focus();
    return;
  }
  if (!targetPositions.length) {
    setStatus("请输入真实饵物坐标，或在对话里描述目标坐标", "待输入");
    return;
  }
  applyDialogIntentToControls(dialogIntent);
  lastDialogIntent = dialogIntent;

  setBusy(true);
  setStatus("闭环仿真运行中", "运行中");
  agentThoughts.textContent = formatIntentPreview(dialogIntent);
  pilotReport.textContent = "驾驶员待命，等待第一轮航向指令。";
  drawEmptyCanvases();

  try {
    const simulationOptions = simulationOptionsFromDialogIntent(dialogIntent, discoveryRange);
    const response = await postJson("/api/simulation/interactive", {
      target_positions: targetPositions,
      target_profiles: targetProfiles,
      bearing_text: command,
      start_position: SIMULATION_DEFAULTS.startPosition,
      default_step: simulationOptions.stepDistance,
      approach_range: simulationOptions.approachRange,
      bearing_noise_deg: simulationOptions.bearingNoise,
      orbit_turns: simulationOptions.orbitTurns,
      orbit_radius: simulationOptions.orbitRadius,
      max_iterations: simulationOptions.maxIterations,
      sonar_trigger_range: sonarTriggerRange,
    });
    lastResult = response.result;
    lastResult.dialog_intent = dialogIntent;
    lastResult.animation_duration_ms = simulationOptions.animationDurationMs;
    initializeAuthorizationState(lastResult);
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

async function sendAgentMessage() {
  const message = agentChatInput.value.trim();
  if (!message) {
    agentChatInput.focus();
    return;
  }
  appendAgentChatMessage("user", message);
  agentChatInput.value = "";
  const apiConfigIssue = apiKeyConfigIssueMessage();
  if (apiConfigIssue) {
    appendAgentChatMessage("assistant", `${apiConfigIssue}\n\n[本地模式：API配置未生效]`);
    return;
  }
  setChatBusy(true);
  try {
    const response = await postJson("/api/chat-plan", {
      messages: agentChatHistory,
      context: agentChatContext(),
    });
    const sourceLabel = agentChatSourceLabel(response);
    appendAgentChatMessage("assistant", `${response.reply || "收到。"}\n\n[${sourceLabel}]`);
  } catch (error) {
    appendAgentChatMessage("assistant", `对话请求失败：${error.message}`);
  } finally {
    setChatBusy(false);
  }
}

function apiKeyConfigIssueMessage() {
  const apiKey = apiKeyInput.value.trim();
  if (!apiKey) return "";
  if (looksLikeUrl(apiKey)) {
    return (
      "API_KEY 输入框里看起来是网页链接。这里需要粘贴 OpenAI 生成的实际密钥字符串；" +
      "API地址保持为 https://api.openai.com/v1 或填写兼容服务地址。"
    );
  }
  return "";
}

function looksLikeUrl(value) {
  const normalized = String(value || "").trim().toLowerCase();
  return normalized.startsWith("http://") || normalized.startsWith("https://");
}

function agentChatSourceLabel(response) {
  if (response.source === "llm") {
    return response.model ? `LLM：${response.model}` : "LLM";
  }
  const reason = response.fallback_reason || (response.llm_requested ? "LLM未连通，已回退" : "未输入API_KEY");
  return `本地模式：${reason}`;
}

function appendAgentChatMessage(role, content) {
  const item = { role, content: String(content || "").trim() };
  if (!item.content) return;
  agentChatHistory.push(item);
  renderAgentChat();
}

function renderAgentChat() {
  agentChatMessages.innerHTML = "";
  agentChatHistory.forEach((message) => {
    const bubble = document.createElement("div");
    bubble.className = `chat-message ${message.role === "user" ? "user" : "assistant"}`;
    bubble.textContent = message.content;
    agentChatMessages.appendChild(bubble);
  });
  agentChatMessages.scrollTop = agentChatMessages.scrollHeight;
}

function conversationTextForSimulation() {
  return agentChatHistory
    .filter((message) => message.role === "user")
    .map((message) => message.content)
    .join("\n")
    .trim();
}

function agentChatContext() {
  const apiKey = apiKeyInput.value.trim();
  const apiBaseUrl = apiBaseUrlInput.value.trim();
  return {
    target_count: targets.filter((target) => target.x !== "" || target.y !== "" || target.depth !== "").length,
    discovery_range: Number(discoveryRangeInput.value) || SIMULATION_DEFAULTS.approachRange,
    sonar_trigger_range: Number(sonarTriggerRangeInput.value) || SIMULATION_DEFAULTS.sonarTriggerRange,
    coordinate_system: "X/Y范围0～2000，起点(0,0)，方向角默认基于当前位置",
    post_decision_policy: "声呐成像后优先按目标价值、深度和IFF决策；红方/中立目标禁止打击，仅允许跟踪、复核、协同或返航；其他目标沿用深度策略。",
    api_key: apiKey,
    api_base_url: apiBaseUrl,
    llm_mode: apiKey ? "cloud" : "local",
  };
}

function initializeAuthorizationState(result) {
  const decision = result.post_mission_decision || {};
  result.authorization_status = decision.requires_authorization ? "pending" : "not_required";
  result.authorization_prompt_shown = false;
}

function resolveSimulationTargets(dialogIntent) {
  if (dialogIntent.target_parse_error) {
    return { positions: [], error: dialogIntent.target_parse_error };
  }
  if (dialogIntent.targets.length) {
    return {
      positions: dialogIntent.targets.map((target) => [target.x, target.y, -Math.abs(target.depth)]),
      error: "",
    };
  }
  return readTargetPositions();
}

function resolveDiscoveryRange(dialogIntent) {
  if (dialogIntent.parameters.approachRange !== null) {
    const value = dialogIntent.parameters.approachRange;
    if (!Number.isFinite(value) || value <= 0) {
      return { value: SIMULATION_DEFAULTS.approachRange, error: "发现距离必须是大于 0 的数字。" };
    }
    return { value, error: "" };
  }
  return readDiscoveryRange();
}

function resolveSonarTriggerRange(dialogIntent) {
  if (dialogIntent.parameters.sonarTriggerRange !== null) {
    const value = dialogIntent.parameters.sonarTriggerRange;
    if (!Number.isFinite(value) || value <= 0) {
      return { value: SIMULATION_DEFAULTS.sonarTriggerRange, error: "声呐触发距离必须是大于 0 的数字。" };
    }
    return { value, error: "" };
  }
  return readSonarTriggerRange();
}

function resolveTargetProfiles(dialogIntent, targetCount) {
  if (dialogIntent.targets.length) {
    const profiles = dialogIntent.targets.map((target, index) => ({
      target_type: target.type || targetProfileAt(index).target_type,
      target_heading_deg: target.heading ?? 0,
      is_blue_target: target.iff ? target.iff === "blue" : targetProfileAt(index).is_blue_target,
    }));
    return { profiles, error: "" };
  }
  const { profiles, error } = readTargetProfiles();
  return { profiles: profiles.slice(0, targetCount), error };
}

function simulationOptionsFromDialogIntent(dialogIntent, discoveryRange) {
  const parameters = dialogIntent.parameters;
  return {
    stepDistance: parameters.stepDistance ?? SIMULATION_DEFAULTS.stepDistance,
    approachRange: discoveryRange,
    bearingNoise: parameters.bearingNoise ?? SIMULATION_DEFAULTS.bearingNoise,
    orbitTurns: parameters.orbitTurns ?? SIMULATION_DEFAULTS.orbitTurns,
    orbitRadius: parameters.orbitRadius ?? SIMULATION_DEFAULTS.orbitRadius,
    maxIterations: parameters.maxIterations ?? SIMULATION_DEFAULTS.maxIterations,
    animationDurationMs: parameters.animationDurationMs ?? SIMULATION_DEFAULTS.animationDurationMs ?? 10000,
  };
}

function applyDialogIntentToControls(dialogIntent) {
  if (dialogIntent.targets.length) {
    targets = dialogIntent.targets.map((target) => ({
      x: String(target.x),
      y: String(target.y),
      depth: String(target.depth),
      type: target.type || "submarine",
      iff: target.iff || "blue",
    }));
    renderTargetRows();
  }
  if (dialogIntent.parameters.approachRange !== null) {
    discoveryRangeInput.value = String(dialogIntent.parameters.approachRange);
  }
  if (dialogIntent.parameters.sonarTriggerRange !== null) {
    sonarTriggerRangeInput.value = String(dialogIntent.parameters.sonarTriggerRange);
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

function readTargetProfiles() {
  const profiles = [];
  for (let index = 0; index < targets.length; index += 1) {
    const target = targets[index];
    const hasAnyValue = target.x !== "" || target.y !== "" || target.depth !== "";
    if (!hasAnyValue) continue;
    const type = TARGET_TYPE_OPTIONS.some(([value]) => value === target.type) ? target.type : "unknown";
    profiles.push({
      target_type: type,
      target_heading_deg: 0,
      is_blue_target: target.iff !== "red",
    });
  }
  return { profiles, error: "" };
}

function targetProfileAt(index) {
  const target = targets[index] || {};
  return {
    target_type: TARGET_TYPE_OPTIONS.some(([value]) => value === target.type) ? target.type : "submarine",
    target_heading_deg: 0,
    is_blue_target: target.iff !== "red",
  };
}

function parseDialogIntent(text) {
  const cleaned = String(text || "").trim();
  const targetExtraction = extractTargetsFromText(cleaned);
  const targetsFromText = targetExtraction.targets;
  const parameters = {
    approachRange: findNamedNumber(cleaned, ["发现距离", "发现半径", "确认距离", "抵近距离", "抵近阈值"]),
    stepDistance: findNamedNumber(cleaned, ["每次前进", "前进步长", "单步距离", "默认步长", "航段距离", "航段", "步长"]),
    orbitTurns: findOrbitTurns(cleaned),
    orbitRadius: findNamedNumber(cleaned, ["绕航半径", "绕行半径", "环绕半径", "盘旋半径", "绕圈半径"]),
    maxIterations: findNamedNumber(cleaned, ["最大迭代", "最多迭代", "迭代次数", "最大轮次", "最多轮次"]),
    bearingNoise: findNamedNumber(cleaned, ["角度噪声", "方位噪声", "测角误差", "角度偏差"]),
    sonarTriggerRange: findNamedNumber(cleaned, ["声呐触发距离", "声呐触发半径", "成像触发距离", "成像声呐距离"]),
    animationDurationMs: findAnimationDuration(cleaned),
  };
  if (parameters.orbitRadius === null && /(?:绕航|绕行|环绕|盘旋|绕圈)/.test(cleaned)) {
    parameters.orbitRadius = findNamedNumber(cleaned, ["半径"]);
  }

  const pilot = extractPilotPreferences(cleaned);
  const actions = [];
  if (targetsFromText.length) actions.push(`配置${targetsFromText.length}个真实饵物`);
  if (parameters.approachRange !== null) actions.push(`发现距离${formatNumber(parameters.approachRange)}m`);
  if (parameters.stepDistance !== null) actions.push(`单步前进${formatNumber(parameters.stepDistance)}m`);
  if (parameters.orbitTurns !== null) actions.push(`基础绕航${formatNumber(parameters.orbitTurns)}圈`);
  if (parameters.orbitRadius !== null) actions.push(`绕航半径${formatNumber(parameters.orbitRadius)}m`);
  if (parameters.maxIterations !== null) actions.push(`最多${formatNumber(parameters.maxIterations)}轮`);
  if (parameters.bearingNoise !== null) actions.push(`方位噪声${formatNumber(parameters.bearingNoise)}°`);
  if (parameters.sonarTriggerRange !== null) actions.push(`声呐触发距离${formatNumber(parameters.sonarTriggerRange)}m`);
  if (parameters.animationDurationMs !== null) actions.push(`轨迹动画${formatNumber(parameters.animationDurationMs / 1000)}秒`);
  if (pilot.style !== "concise" || pilot.mode !== "progressive") {
    actions.push(`驾驶员反馈${pilotStyleLabel(pilot.style)}${pilot.mode === "final_only" ? "，只报结论" : ""}`);
  }

  return {
    raw_text: cleaned,
    targets: targetsFromText,
    target_parse_error:
      targetExtraction.invalidCount > 0
        ? "对话中的饵物坐标未生效：X/Y 必须是 0～2000 的整数，深度必须是非负整数。"
        : "",
    parameters,
    pilot_style: pilot.style,
    pilot_report_mode: pilot.mode,
    interpreted_actions: actions,
  };
}

function extractTargetsFromText(text) {
  const found = [];
  let attemptedCount = 0;
  let invalidCount = 0;
  const addTarget = (x, y, depth, startIndex = -1) => {
    attemptedCount += 1;
    const target = normalizeDialogTarget(x, y, depth);
    if (!target) {
      invalidCount += 1;
      return;
    }
    if (startIndex >= 0) {
      const context = text.slice(Math.max(0, startIndex - 14), startIndex);
      if (/(?:起点|出发点|当前位置|UUV|uuv|无人艇|本艇)$/.test(context.trim())) return;
    }
    if (found.some((item) => item.x === target.x && item.y === target.y && item.depth === target.depth)) return;
    found.push({ ...target, ...extractTargetProfileNear(text, startIndex) });
  };

  const coordinatePattern = /(?:目标|饵物|诱饵|bait|target)?\s*[A-Za-z0-9一二三四五六七八九十]*\s*(?:坐标|位置)?\s*[\(（\[]\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)(?:\s*[,，]\s*(-?\d+(?:\.\d+)?))?\s*[\)）\]]/gi;
  for (const match of text.matchAll(coordinatePattern)) {
    addTarget(match[1], match[2], match[3], match.index || 0);
  }

  const xyPattern = /(?:目标|饵物|诱饵|bait|target)\s*[A-Za-z0-9一二三四五六七八九十]*[^。；;\n]*?(?:x|X|横坐标)\s*(?:为|=|:|：)?\s*(-?\d+(?:\.\d+)?)[^。；;\n]*?(?:y|Y|纵坐标)\s*(?:为|=|:|：)?\s*(-?\d+(?:\.\d+)?)(?:[^。；;\n]*?(?:深度|depth|z|Z)\s*(?:为|=|:|：)?\s*(-?\d+(?:\.\d+)?))?/gi;
  for (const match of text.matchAll(xyPattern)) {
    addTarget(match[1], match[2], match[3], match.index || 0);
  }

  const plainPattern = /(?:目标|饵物|诱饵|bait|target)\s*[A-Za-z0-9一二三四五六七八九十]*[^。；;\n]*?(?:坐标|位置)\s*(?:为|=|:|：)?\s*(-?\d+(?:\.\d+)?)\s*[,，]\s*(-?\d+(?:\.\d+)?)(?:\s*[,，]\s*(-?\d+(?:\.\d+)?))?(?:[^。；;\n]*?(?:深度|depth)\s*(?:为|=|:|：)?\s*(-?\d+(?:\.\d+)?))?/gi;
  for (const match of text.matchAll(plainPattern)) {
    addTarget(match[1], match[2], match[4] ?? match[3], match.index || 0);
  }
  return { targets: found, attemptedCount, invalidCount };
}

function extractTargetProfileNear(text, startIndex) {
  const windowText = text.slice(Math.max(0, startIndex - 30), Math.min(text.length, startIndex + 80));
  const typeMap = [
    ["submarine", /潜艇|submarine/i],
    ["torpedo", /鱼雷|torpedo/i],
    ["ship", /水面舰|舰船|ship/i],
    ["reef", /礁石|reef/i],
    ["unknown", /不明|未知|unknown/i],
  ];
  const matchedType = typeMap.find(([, pattern]) => pattern.test(windowText));
  const heading = findNamedNumber(windowText, ["目标航向", "航向"]);
  const iff = /红方|我方|中立/.test(windowText) ? "red" : /蓝方|敌方/.test(windowText) ? "blue" : "blue";
  return {
    type: matchedType ? matchedType[0] : "submarine",
    heading: heading ?? 0,
    iff,
  };
}

function normalizeDialogTarget(xValue, yValue, depthValue) {
  const x = Number(xValue);
  const y = Number(yValue);
  const depth = depthValue === undefined || depthValue === "" ? 50 : Math.abs(Number(depthValue));
  if (!validWorldCoordinate(x) || !validWorldCoordinate(y)) return null;
  if (!Number.isFinite(depth) || depth < 0 || !Number.isInteger(depth)) return null;
  return { x, y, depth };
}

function findNamedNumber(text, names) {
  for (const name of names) {
    const pattern = new RegExp(`${escapeRegExp(name)}\\s*(?:为|设为|设置为|调整为|改为|=|:|：)?\\s*(-?\\d+(?:\\.\\d+)?)`, "i");
    const match = text.match(pattern);
    if (match) return Number(match[1]);
  }
  return null;
}

function findOrbitTurns(text) {
  const direct = text.match(/(?:绕航|绕行|环绕|绕目标|绕圈|盘旋|绕)\s*(?:为|=|:|：)?\s*(\d+(?:\.\d+)?)\s*(?:圈|周|turns?)/i);
  if (direct) return Number(direct[1]);
  return findNamedNumber(text, ["绕航圈数", "绕行圈数", "环绕圈数", "盘旋圈数", "基础绕航"]);
}

function findAnimationDuration(text) {
  const explicit = text.match(/(?:动画时长|轨迹时长|播放时长)\s*(?:为|=|:|：)?\s*(\d+(?:\.\d+)?)\s*(秒|s|毫秒|ms)?/i);
  if (explicit) {
    const value = Number(explicit[1]);
    return /毫秒|ms/i.test(explicit[2] || "") ? value : value * 1000;
  }
  if (/(?:轨迹|动画|播放)[^。；;\n]*(?:慢一点|更慢|放慢)/.test(text)) return 24000;
  if (/(?:轨迹|动画|播放)[^。；;\n]*(?:快一点|更快|加快)/.test(text)) return 7000;
  return null;
}

function extractPilotPreferences(text) {
  let style = "concise";
  if (/(?:驾驶员|反馈|汇报|口吻|语气)[^。；;\n]*(?:正式|规范|报告式)/.test(text)) {
    style = "formal";
  } else if (/(?:驾驶员|反馈|汇报|口吻|语气)[^。；;\n]*(?:口语|自然|像人|驾驶员口吻)/.test(text)) {
    style = "plain";
  } else if (/(?:驾驶员|反馈|汇报|口吻|语气)[^。；;\n]*(?:详细|展开|多说)/.test(text)) {
    style = "detailed";
  } else if (/(?:驾驶员|反馈|汇报|口吻|语气)[^。；;\n]*(?:简练|简洁|短一点|精简|固定格式)/.test(text)) {
    style = "concise";
  }

  let mode = "progressive";
  if (/(?:驾驶员|反馈|汇报)[^。；;\n]*(?:只报|只要|仅报)[^。；;\n]*(?:结论|最终结果)/.test(text)) {
    mode = "final_only";
  } else if (/(?:不要|不用)[^。；;\n]*(?:逐轮|每轮|过程)/.test(text)) {
    mode = "final_only";
  } else if (/(?:逐轮|每轮|过程|持续)[^。；;\n]*(?:反馈|汇报|报告)/.test(text)) {
    mode = "progressive";
  }
  return { style, mode };
}

function pilotStyleLabel(style) {
  if (style === "formal") return "正式";
  if (style === "plain") return "口语化";
  if (style === "detailed") return "详细";
  return "简练";
}

function formatIntentPreview(dialogIntent) {
  if (!dialogIntent.interpreted_actions.length) {
    return "Agent正在根据方位观测滚动决策。\n对话输入未发现可直接改写的参数，本次沿用左侧参数。";
  }
  return [
    "Agent已理解对话输入，准备按以下配置执行：",
    ...dialogIntent.interpreted_actions.map((item) => `- ${item}`),
    "",
    "随后根据多次方位观测滚动决策。",
  ].join("\n");
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

function readSonarTriggerRange() {
  const value = Number(sonarTriggerRangeInput.value);
  if (!Number.isFinite(value) || value <= 0) {
    return { value: SIMULATION_DEFAULTS.sonarTriggerRange, error: "声呐触发距离必须是大于 0 的数字。" };
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
  const windows = segmentWindows(segments);

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
    drawBearingRecord(result, state);
    drawSonarRecord(state);
    onProgress(progress, { state });
  };

  if (!animate) {
    renderFrame(1);
    onProgress(1, { force: true });
    return;
  }

  onProgress(0, { force: true });
  let startedAt = performance.now();
  const duration = Math.max(3000, Number(result.animation_duration_ms || SIMULATION_DEFAULTS.animationDurationMs));
  const step = (now) => {
    let progress = Math.min(1, (now - startedAt) / duration);
    const authProgress = authorizationTriggerProgress(result, windows);
    if (authProgress !== null && progress >= authProgress) {
      progress = authProgress;
      renderFrame(progress);
      showAuthorizationModal(result, {
        onApprove: () => {
          result.authorization_status = "approved";
          startedAt = performance.now() - progress * duration;
          trajectoryAnimation = requestAnimationFrame(step);
        },
        onHold: () => {
          result.authorization_status = "held";
          renderFrame(progress);
          setStatus("等待授权", "待授权");
        },
      });
      return;
    }
    renderFrame(progress);
    if (progress < 1) {
      trajectoryAnimation = requestAnimationFrame(step);
    }
  };
  trajectoryAnimation = requestAnimationFrame(step);
}

function authorizationTriggerProgress(result, windows) {
  const decision = result.post_mission_decision || {};
  if (!decision.requires_authorization) return null;
  if (result.authorization_status !== "pending") return null;
  if (result.authorization_prompt_shown) return null;
  if (!windows.postMissionWindow || !windows.totalDistance) return null;
  return Math.max(0, Math.min(1, windows.postMissionWindow.start / windows.totalDistance));
}

function showAuthorizationModal(result, handlers) {
  const decision = result.post_mission_decision || {};
  result.authorization_prompt_shown = true;
  authorizationResume = handlers.onApprove;
  authorizationHold = handlers.onHold;
  authorizationCloseAction = null;
  authorizationMessage.textContent = [
    `${decision.decision || "后续行动"}。`,
    decision.reasoning ? `依据：${decision.reasoning}。` : "",
    "请确认是否授权进入模拟打击待机航段。",
  ].filter(Boolean).join("");
  authorizationApproveBtn.hidden = false;
  authorizationHoldBtn.hidden = false;
  authorizationCloseBtn.hidden = true;
  authorizationModal.hidden = false;
}

function showAuthorizationResult(status) {
  const result = lastResult || {};
  const decision = result.post_mission_decision || {};
  const approved = status === "approved";
  if (result.post_mission_decision) {
    result.authorization_status = approved ? "approved" : "held";
    renderSynchronizedNarrative(result, authorizationTriggerSnapshotProgress(result), { force: true });
  }
  authorizationMessage.textContent = approved
    ? `授权结果：授权通过。${decision.execution_summary || "UUV将继续进入后续行动航段。"}`
    : `授权结果：未获授权。UUV保持待机并持续标记目标。${decision.execution_summary || ""}`;
  authorizationApproveBtn.hidden = true;
  authorizationHoldBtn.hidden = true;
  authorizationCloseBtn.hidden = false;
  authorizationCloseAction = () => {
    if (approved && authorizationResume) {
      authorizationResume();
    } else if (!approved && authorizationHold) {
      authorizationHold();
    }
  };
}

function authorizationTriggerSnapshotProgress(result) {
  const segments = normalizedTrajectorySegments(result);
  const windows = segmentWindows(segments);
  if (!windows.postMissionWindow || !windows.totalDistance) return 1;
  return Math.max(0, Math.min(1, windows.postMissionWindow.start / windows.totalDistance));
}

function hideAuthorizationModal() {
  authorizationModal.hidden = true;
  authorizationResume = null;
  authorizationHold = null;
  authorizationCloseAction = null;
}

function drawBearingRecord(result, state = narrativeState(result, 1)) {
  const observations = result.bearing_history || [];
  const visibleObservations = visibleBearingObservations(result, state);
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
  if (visibleObservations.length) {
    drawAngleLine(ctx, visibleObservations, xFor, yFor, "angle", "#087c89", false);
  }

  ctx.fillStyle = "#172026";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillText("观测方位", padding.left, 20);
  ctx.fillStyle = "#667780";
  ctx.fillText(`${visibleObservations.length} / ${observations.length}`, width - padding.right - 54, 20);
  if (!visibleObservations.length) {
    drawCenteredText(ctx, width, height, "等待首次探测");
  }
}

function drawSonarRecord(state) {
  const event = state.activeSonarEvent || state.visibleSonarEvents[state.visibleSonarEvents.length - 1];
  const isActive = Boolean(state.activeSonarEvent);
  const rect = sonarCanvas.getBoundingClientRect();
  const width = Math.max(160, Math.floor(rect.width));
  const height = Math.max(120, Math.floor(rect.height));
  const ratio = window.devicePixelRatio || 1;
  sonarCanvas.width = Math.floor(width * ratio);
  sonarCanvas.height = Math.floor(height * ratio);
  const ctx = sonarCanvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#071319";
  ctx.fillRect(0, 0, width, height);

  if (!event) {
    sonarStatusBadge.textContent = "待机";
    sonarStatusBadge.classList.remove("active");
    sonarMetadata.textContent = "等待进入成像距离";
    drawSonarCenteredText(ctx, width, height, "成像声呐未开启");
    return;
  }

  const image = Array.isArray(event.image_rgb) ? event.image_rgb : [];
  const imageHeight = image.length;
  const imageWidth = imageHeight && Array.isArray(image[0]) ? image[0].length : 0;
  if (imageHeight && imageWidth) {
    const source = document.createElement("canvas");
    source.width = imageWidth;
    source.height = imageHeight;
    const sourceCtx = source.getContext("2d");
    const pixels = sourceCtx.createImageData(imageWidth, imageHeight);
    let offset = 0;
    image.forEach((row) => {
      row.forEach((pixel) => {
        pixels.data[offset] = Number(pixel[0] || 0);
        pixels.data[offset + 1] = Number(pixel[1] || 0);
        pixels.data[offset + 2] = Number(pixel[2] || 0);
        pixels.data[offset + 3] = 255;
        offset += 4;
      });
    });
    sourceCtx.putImageData(pixels, 0, 0);
    const scale = Math.min(width / imageWidth, height / imageHeight);
    const drawWidth = Math.max(1, Math.floor(imageWidth * scale));
    const drawHeight = Math.max(1, Math.floor(imageHeight * scale));
    ctx.drawImage(source, Math.floor((width - drawWidth) / 2), Math.floor((height - drawHeight) / 2), drawWidth, drawHeight);
  } else {
    drawSonarCenteredText(ctx, width, height, "未收到声呐图像");
  }

  sonarStatusBadge.textContent = isActive ? "成像中" : "成像完成";
  sonarStatusBadge.classList.toggle("active", isActive);
  const recognition = event.recognition || {};
  const resultText = Number(event.echo_strength || 0) > 0
    ? `${targetTypeLabel(recognition.target_type)} / 置信度${formatNumber((recognition.confidence || 0) * 100)}%`
    : "未形成有效回波";
  sonarMetadata.textContent = `目标${Number(event.target_index) + 1} · 距离${formatNumber(event.target_range_m)}m · ${resultText}`;
}

function drawSonarCenteredText(ctx, width, height, text) {
  ctx.fillStyle = "#8ea8b2";
  ctx.font = "13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
  ctx.textAlign = "start";
}

function visibleBearingObservations(result, state) {
  const observations = result.bearing_history || [];
  if (!observations.length) return [];
  if (!state || state.isComplete) {
    return observations.map((observation, index) => ({ ...observation, _plotIndex: index }));
  }

  const visibleKeys = new Set(
    (state.thoughtDecisions || [])
      .map((decision) => observationKey(decision.observation))
      .filter(Boolean)
  );
  const visible = [];
  observations.forEach((observation, index) => {
    if (visibleKeys.has(observationKey(observation))) {
      visible.push({ ...observation, _plotIndex: index });
    }
  });

  if (visible.length) return visible;
  const fallbackCount = Math.min(observations.length, Math.max(0, (state.thoughtDecisions || []).length));
  return observations.slice(0, fallbackCount).map((observation, index) => ({ ...observation, _plotIndex: index }));
}

function observationKey(observation) {
  if (!observation) return "";
  return [
    observation.timestamp || "",
    observation.target_index ?? "",
    observation.target_sequence ?? "",
    Number(observation.angle).toFixed(3),
  ].join("|");
}

function drawEmptyCanvases() {
  const trajectory = prepareCanvas(trajectoryCanvas);
  trajectory.ctx.clearRect(0, 0, trajectory.width, trajectory.height);
  drawCenteredText(trajectory.ctx, trajectory.width, trajectory.height, "等待轨迹");

  const bearing = prepareCanvas(bearingCanvas);
  bearing.ctx.clearRect(0, 0, bearing.width, bearing.height);
  drawCenteredText(bearing.ctx, bearing.width, bearing.height, "等待角度记录");

  drawSonarRecord({ visibleSonarEvents: [], activeSonarEvent: null });
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
  const area = trajectoryPlotArea(width, height);
  return [
    area.left + ((Number(point[0]) - bounds.minX) / worldWidth) * area.width,
    area.bottom - ((Number(point[1]) - bounds.minY) / worldHeight) * area.height,
  ];
}

function trajectoryPlotArea(width, height) {
  const padding = {
    left: Math.min(36, Math.max(24, width * 0.06)),
    right: Math.min(28, Math.max(18, width * 0.045)),
    top: 24,
    bottom: Math.max(32, Math.min(40, height * 0.12)),
  };
  return {
    left: padding.left,
    right: width - padding.right,
    top: padding.top,
    bottom: height - padding.bottom,
    width: Math.max(1, width - padding.left - padding.right),
    height: Math.max(1, height - padding.top - padding.bottom),
  };
}

function drawPlotBackground(ctx, width, height, bounds) {
  const area = trajectoryPlotArea(width, height);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e6ecef";
  ctx.lineWidth = 1;
  for (let index = 0; index <= 5; index += 1) {
    const x = area.left + (area.width * index) / 5;
    const y = area.top + (area.height * index) / 5;
    const lineX = index === 5 ? area.right - 0.5 : x + 0.5;
    const lineY = index === 5 ? area.bottom - 0.5 : y + 0.5;
    ctx.beginPath();
    ctx.moveTo(lineX, area.top);
    ctx.lineTo(lineX, area.bottom);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(area.left, lineY);
    ctx.lineTo(area.right, lineY);
    ctx.stroke();
  }
  ctx.strokeStyle = "#cbd6da";
  ctx.strokeRect(area.left + 0.5, area.top + 0.5, area.width - 1, area.height - 1);
  ctx.fillStyle = "#667780";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
  ctx.fillText(`X ${formatNumber(bounds.minX)} ~ ${formatNumber(bounds.maxX)} m`, area.left, height - 9);
  const yLabel = `Y ${formatNumber(bounds.minY)} ~ ${formatNumber(bounds.maxY)} m`;
  const yLabelWidth = ctx.measureText(yLabel).width;
  ctx.fillText(yLabel, Math.max(area.left, area.right - yLabelWidth), 16);
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
    const label = sequence ? `T${index + 1} / ${sequence}` : `T${index + 1}`;
    const labelWidth = ctx.measureText(label).width;
    const labelX = x + 9 + labelWidth <= width - 4 ? x + 9 : x - labelWidth - 9;
    const labelY = y - 9 >= 14 ? y - 9 : y + 18;
    ctx.fillText(label, labelX, labelY);
  });
}

function normalizedTrajectorySegments(result) {
  const sourceSegments = Array.isArray(result.trajectory_segments) ? result.trajectory_segments : [];
  if (sourceSegments.length) {
    return stitchSegments(sourceSegments
      .map((segment) => ({
        kind: segment.kind || "approach",
        postAction: segment.post_action || "",
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
    drawPath(ctx, width, height, bounds, segment.points, segmentColor(segment), segmentProgress);
    remainingDistance -= distance;
  });
}

function segmentColor(segment) {
  if (segment.kind === "post_mission") return "#8b5e1a";
  return ROUTE_COLOR;
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
    const plotIndex = Number.isFinite(item._plotIndex) ? item._plotIndex : index;
    const x = xFor(plotIndex);
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
    const plotIndex = Number.isFinite(item._plotIndex) ? item._plotIndex : index;
    const x = xFor(plotIndex);
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
    state.visibleSonarEvents.length,
    state.activeSonarEvent ? `sonar${state.activeSonarEvent._timelineIndex}` : "sonarOff",
    state.targetMissionComplete ? "targetDone" : "targetRun",
    state.visibleOrbitTurns.toFixed(1),
    state.postMissionVisible ? "post" : "target",
    state.postMissionProgress.toFixed(1),
    state.postMissionComplete ? "postDone" : "postRun",
    state.authorizationStatus,
    state.isComplete ? "done" : "running",
  ].join(":");
  if (!options.force && key === lastNarrativeKey) return;

  const thoughtsWereAtBottom = isScrolledNearBottom(agentThoughts);
  const pilotWasAtBottom = isScrolledNearBottom(pilotReport);
  agentThoughts.textContent = formatAgentThoughts(result, state);
  pilotReport.textContent = formatPilotReport(result, state);
  lastNarrativeKey = key;
  if (options.autoScroll) {
    if (thoughtsWereAtBottom) agentThoughts.scrollTop = agentThoughts.scrollHeight;
    if (pilotWasAtBottom) pilotReport.scrollTop = pilotReport.scrollHeight;
  }
}

function isScrolledNearBottom(element) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= 32;
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
  const decisionDistanceByKey = new Map();

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
    decisionDistanceByKey.set(sonarEventKey(decision), pilotDistance);
    localDistanceBySequence.set(sequence, localAfter);
  });

  const completedTargetRuns = (result.target_runs || []).filter((run) => {
    const sequence = Number(run.target_sequence || 1);
    const orbit = windows.orbitBySequence.get(sequence);
    const approach = windows.approachBySequence.get(sequence);
    const completionDistance = orbit ? orbit.end : approach ? approach.end : totalDistance;
    return completionDistance <= currentDistance + 0.001;
  });

  const sonarTimeline = (result.sonar_events || []).map((event, index) => {
    const eventDistance = distanceAlongSegmentsToPoint(segments, event.uuv_position);
    const decisionDistance = decisionDistanceByKey.get(sonarEventKey(event));
    const triggerDistance = Number.isFinite(eventDistance)
      ? eventDistance
      : Number.isFinite(decisionDistance)
        ? decisionDistance
        : totalDistance;
    return { ...event, _timelineIndex: index, _triggerDistance: triggerDistance };
  });
  const visibleSonarEvents = sonarTimeline.filter((event) => event._triggerDistance <= currentDistance + 0.001);
  const latestSonarEvent = visibleSonarEvents[visibleSonarEvents.length - 1] || null;
  const sonarActiveSpan = Math.max(6, totalDistance * 0.08);
  const activeSonarEvent = latestSonarEvent && clampedProgress < 0.999 &&
    currentDistance < latestSonarEvent._triggerDistance + sonarActiveSpan
      ? latestSonarEvent
      : null;

  return {
    progress: clampedProgress,
    currentDistance,
    totalDistance,
    isComplete: clampedProgress >= 0.999,
    thoughtDecisions: decisions.filter((_, index) => thoughtDecisionIndexes.has(index)),
    pilotDecisions: decisions.filter((_, index) => pilotDecisionIndexes.has(index)),
    completedTargetRuns,
    visibleSonarEvents,
    activeSonarEvent,
    targetMissionComplete: Boolean((result.target_runs || []).length) &&
      completedTargetRuns.length >= (result.target_runs || []).length,
    visibleOrbitTurns: visibleOrbitTurns(result, windows, currentDistance),
    postMissionVisible: Boolean(windows.postMissionWindow && currentDistance >= windows.postMissionWindow.start),
    postMissionProgress: postMissionProgress(windows.postMissionWindow, currentDistance),
    postMissionComplete: Boolean(windows.postMissionWindow) && currentDistance >= windows.postMissionWindow.end - 0.001,
    authorizationStatus: result.authorization_status || "not_required",
  };
}

function sonarEventKey(item) {
  return `${Number(item?.target_sequence || 1)}|${Number(item?.iteration || 0)}`;
}

function distanceAlongSegmentsToPoint(segments, point) {
  if (!Array.isArray(point) || point.length < 2 || !segments.length) return Number.NaN;
  const targetX = Number(point[0]);
  const targetY = Number(point[1]);
  if (!Number.isFinite(targetX) || !Number.isFinite(targetY)) return Number.NaN;

  let cursor = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  let bestTimelineDistance = Number.NaN;
  segments.forEach((segment) => {
    const points = segment.points || [];
    const worldLength = polylineWorldLength(points);
    const timelineLength = segmentDistance(segment);
    let traveled = 0;
    for (let index = 1; index < points.length; index += 1) {
      const start = points[index - 1];
      const end = points[index];
      const dx = end[0] - start[0];
      const dy = end[1] - start[1];
      const edgeLength = Math.hypot(dx, dy);
      if (edgeLength <= 0) continue;
      const projection = Math.max(0, Math.min(1, ((targetX - start[0]) * dx + (targetY - start[1]) * dy) / (edgeLength * edgeLength)));
      const projectedX = start[0] + dx * projection;
      const projectedY = start[1] + dy * projection;
      const pointDistance = Math.hypot(targetX - projectedX, targetY - projectedY);
      if (pointDistance < bestDistance) {
        bestDistance = pointDistance;
        const fraction = worldLength > 0 ? (traveled + edgeLength * projection) / worldLength : 0;
        bestTimelineDistance = cursor + fraction * timelineLength;
      }
      traveled += edgeLength;
    }
    cursor += timelineLength;
  });
  return bestTimelineDistance;
}

function segmentWindows(segments) {
  const approachBySequence = new Map();
  const orbitBySequence = new Map();
  let postMissionWindow = null;
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
    } else if (segment.kind === "post_mission") {
      postMissionWindow = window;
    } else {
      approachBySequence.set(Number(segment.targetSequence || 1), window);
    }
    cursor = window.end;
  });
  return {
    approachBySequence,
    orbitBySequence,
    postMissionWindow,
    totalDistance: cursor,
  };
}

function postMissionProgress(window, currentDistance) {
  if (!window) return 0;
  if (currentDistance <= window.start) return 0;
  if (currentDistance >= window.end) return 1;
  return Math.max(0, Math.min(1, (currentDistance - window.start) / Math.max(window.distance, 1)));
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

  const intentLines = formatDialogIntentLines(result.dialog_intent || lastDialogIntent);
  if (intentLines.length) {
    lines.push("对话意图解析：");
    lines.push(...intentLines);
    lines.push("");
  }

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
        }轮，发现时距离 ${formatNumber(run.final_distance)} m，绕行 ${formatNumber(run.orbit_distance)} m${formatRunSonarSuffix(run)}`
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
        formatDecisionSonarResult(decision),
        formatDecisionResult(decision),
        `反馈：${decision.feedback || "--"}`,
      ].join("\n")
    );
    lines.push("");
  });

  if (state.targetMissionComplete && result.post_mission_decision) {
    lines.push("任务后自主决策：");
    lines.push(formatPostMissionThought(result.post_mission_decision, state));
    lines.push("");
  }
  return lines.join("\n");
}

function formatPilotReport(result, state = narrativeState(result, 1)) {
  const completedCount = state.completedTargetRuns.length;
  const style = pilotStyleFromResult(result);
  const reportMode = pilotReportModeFromResult(result);
  const lines = [pilotReportHeader(style)];

  if (reportMode !== "final_only") {
    const reportedSonarEvents = new Set();
    state.pilotDecisions.forEach((decision) => {
      lines.push(formatPilotDecisionLine(decision, result, style));
      const matchingEvents = state.visibleSonarEvents.filter((event) => sonarEventKey(event) === sonarEventKey(decision));
      matchingEvents.forEach((event) => reportedSonarEvents.add(event._timelineIndex));
      lines.push(...formatPilotSonarLines(matchingEvents, state.activeSonarEvent, style));
    });
    const unmatchedEvents = state.visibleSonarEvents.filter((event) => !reportedSonarEvents.has(event._timelineIndex));
    lines.push(...formatPilotSonarLines(unmatchedEvents, state.activeSonarEvent, style));
  }

  if (reportMode !== "final_only" && !state.pilotDecisions.length) {
    lines.push(pilotStandbyLine(style));
  }

  if (result.status === "success" && state.targetMissionComplete) {
    lines.push(formatPilotConclusion(state.completedTargetRuns, style));
    if (result.post_mission_decision) {
      if (state.authorizationStatus === "held") {
        lines.push(formatPostMissionHoldLine(result.post_mission_decision, style));
      } else if (!state.postMissionVisible || state.postMissionProgress <= 0.001) {
        lines.push(formatPostMissionPendingLine(result.post_mission_decision, style, state));
      } else {
        lines.push(formatPostMissionPilotLine(result.post_mission_decision, style, state.postMissionComplete));
      }
    }
  } else if (state.isComplete) {
    lines.push(formatPilotFailure(completedCount, result, style));
  } else {
    lines.push(formatPilotProgress(completedCount, style));
  }
  return lines.join("\n");
}

function formatPilotSonarLines(events, activeEvent, style = "concise") {
  return events.flatMap((event) => {
    const target = `目标${Number(event.target_index) + 1}`;
    const prefix = `第${event.iteration}轮：`;
    const formatLine = (line) => style === "formal" ? `报告：${line}` : line;
    const openingLine = formatLine(`${prefix}已开启成像声呐，正在扫描${target}。`);
    if (activeEvent && activeEvent._timelineIndex === event._timelineIndex) {
      return [openingLine];
    }
    const closingLine = Number(event.echo_strength || 0) > 0
      ? `${prefix}成像完成，${formatSonarRecognition(event.recognition || {})}，已关闭成像声呐。`
      : `${prefix}未形成有效回波，已关闭成像声呐，继续抵近。`;
    return [openingLine, formatLine(closingLine)];
  });
}

function formatPostMissionThought(decision, state) {
  const selected =
    decision.selected_target_index !== null && decision.selected_target_index !== undefined
      ? `目标${Number(decision.selected_target_index) + 1}`
      : "当前目标";
  return [
    "侦察结论：目标坐标、深度和绕航复核结果已归档，进入任务后处置评估。",
    decision.sonar_recognition ? `声呐识别：${formatSonarRecognition(decision.sonar_recognition)}。` : "",
    `情报归纳：${decision.reasoning || "--"}`,
    `决策推理：主要依据已确认目标深度、类型和IFF，并结合目标数量与方位可信度，选择“${decision.decision || "--"}”。`,
    `关注对象：${selected}；置信度：${formatNumber((decision.confidence || 0) * 100)}%`,
    `执行阶段：${postMissionStageText(state)}`,
    `执行摘要：${decision.execution_summary || "--"}`,
    decision.requires_authorization ? "约束：打击类动作仅进入模拟待机与授权请求，不执行真实武器控制。" : "",
  ].filter(Boolean).join("\n");
}

function postMissionStageText(state) {
  if (!state.postMissionVisible || state.postMissionProgress <= 0.001) {
    if (state.authorizationStatus === "pending") {
      return "正在请求授权，后续行动尚未执行";
    }
    if (state.authorizationStatus === "approved") {
      return "授权已通过，准备进入后续行动航段";
    }
    if (state.authorizationStatus === "held") {
      return "授权未通过，保持待机并持续标记目标";
    }
    return "决策已形成，等待进入后续行动航段";
  }
  if (state.postMissionComplete) return "后续行动已完成";
  return `后续行动执行中，进度 ${formatNumber((state.postMissionProgress || 0) * 100)}%`;
}

function formatPostMissionPilotLine(decision, style = "concise", complete = false) {
  const prefix = complete ? "后续行动完成" : "正在执行后续行动";
  const base = `${prefix}：${decision.decision || "--"}。${decision.execution_summary || ""}`;
  if (style === "formal") return `报告：${base}`;
  if (style === "plain") return `${base}`;
  if (style === "detailed") {
    return `${base}决策依据：${decision.reasoning || "--"}。`;
  }
  return base;
}

function formatPostMissionPendingLine(decision, style = "concise", state = {}) {
  let suffix = "";
  if (decision.requires_authorization) {
    suffix = state.authorizationStatus === "approved" ? "授权已通过，准备进入后续行动航段。" : "正在请求授权。";
  }
  const base = `已形成后续决策：${decision.decision || "--"}。依据：${decision.reasoning || "--"}。${suffix}`;
  return style === "formal" ? `报告：${base}` : base;
}

function formatPostMissionHoldLine(decision, style = "concise") {
  const base = `授权未通过，保持待机并持续标记目标。${decision.execution_summary || ""}`;
  return style === "formal" ? `报告：${base}` : base;
}

function formatDialogIntentLines(dialogIntent) {
  if (!dialogIntent) return [];
  if (!dialogIntent.interpreted_actions.length) {
    return ["未识别到参数改写，沿用左侧参数；任务语义仍用于方位解析和滚动规划。"];
  }
  return dialogIntent.interpreted_actions.map((item) => `已应用：${item}`);
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
  if (decision.target_excluded) {
    const recognition = decision.sonar_recognition || {};
    return `结果：成像声呐识别为${targetTypeLabel(recognition.target_type)}，判定为假目标/非真实目标，跳过后续绕航。`;
  }
  if (decision.target_discovered) {
    return `结果：进入发现半径，确认目标坐标 ${formatVector(
      decision.discovered_position
    )}；发现时距离 ${formatNumber(decision.distance_after)} m`;
  }
  return "结果：尚未进入发现半径，目标距离和坐标未知，继续复测方位";
}

function formatPilotDecisionLine(decision, result, style = "concise") {
  if (decision.target_excluded) {
    const recognition = decision.sonar_recognition || {};
    const base = `第${decision.iteration}轮：成像声呐识别为${targetTypeLabel(
      recognition.target_type
    )}，置信度${formatNumber((recognition.confidence || 0) * 100)}%，判定为假目标，已排除。`;
    return style === "formal" ? `报告：${base}` : base;
  }
  if (decision.target_discovered) {
    const run = targetRunBySequence(result, decision.target_sequence);
    const depth = run?.target_depth ?? targetDepth(decision.discovered_position);
    const sonarNote = run?.sonar_recognition ? `声呐识别${formatSonarRecognition(run.sonar_recognition)}。` : "";
    const deepNote =
      run?.is_deep_target || depth > 10
        ? `深度${formatNumber(depth)}m，重点复核，绕航${run?.orbit_turns_completed || "--"}圈。`
        : `深度${formatNumber(depth)}m，常规复核。`;
    const base = `第${decision.iteration}轮：发现目标${Number(decision.target_index) + 1}，位置${formatVector(
      decision.discovered_position
    )}，${sonarNote}${deepNote}`;
    if (style === "formal") return `报告：${base}`;
    if (style === "plain") return `收到，第${decision.iteration}轮发现目标${Number(decision.target_index) + 1}，位置${formatVector(decision.discovered_position)}，${deepNote}`;
    if (style === "detailed") {
      return `${base}发现后已进入目标复核段，按任务要求完成绕航确认。`;
    }
    return base;
  }
  if ((decision.warnings || []).some((warning) => String(warning).includes("疑似虚假信息"))) {
    const base = `第${decision.iteration}轮：复测方位${formatNumber(
      (decision.observation || {}).angle
    )}°，初始方位疑似虚假，改按复测方位执行。`;
    return style === "formal" ? `报告：${base}` : base;
  }
  const base = `第${decision.iteration}轮：航向${formatNumber(decision.executed_heading)}°，前进${formatNumber(
    decision.executed_distance
  )}m。`;
  if (style === "formal") return `报告：${base}`;
  if (style === "plain") return `第${decision.iteration}轮，按${formatNumber(decision.executed_heading)}°走${formatNumber(decision.executed_distance)}m。`;
  if (style === "detailed") {
    return `${base}当前仍只掌握方位信息，未确认距离和坐标。`;
  }
  return base;
}

function formatPilotConclusion(targetRuns, style = "concise") {
  const realRuns = targetRuns.filter((run) => run.status === "success");
  const excludedRuns = targetRuns.filter((run) => run.status === "excluded");
  const positions = realRuns.map((run) => formatVector(run.target_position));
  const positionText = positions.length ? `${positions.join("、")}等` : "无";
  const exclusionText = excludedRuns.length ? `排除${excludedRuns.length}个假目标，` : "";
  const base = `发现${realRuns.length}个目标，${exclusionText}位置分别为${positionText}，已完成抵近侦察。`;
  if (style === "formal") return `最终报告：${base}`;
  if (style === "plain") return `任务完成，${base}`;
  if (style === "detailed") {
    const deepCount = targetRuns.filter((run) => run.is_deep_target).length;
    return `${base}其中深度超过10m的目标${deepCount}个，均已按重点复核要求增加绕航。`;
  }
  return base;
}

function pilotStyleFromResult(result) {
  return (result.dialog_intent || lastDialogIntent || {}).pilot_style || "concise";
}

function pilotReportModeFromResult(result) {
  return (result.dialog_intent || lastDialogIntent || {}).pilot_report_mode || "progressive";
}

function pilotReportHeader(style) {
  if (style === "formal") return "驾驶员正式报告：";
  if (style === "plain") return "驾驶员反馈：";
  if (style === "detailed") return "驾驶员详细反馈：";
  return "驾驶员反馈：";
}

function pilotStandbyLine(style) {
  if (style === "formal") return "报告：本艇待命，等待方位指令。";
  if (style === "plain") return "待命中，等第一轮方位。";
  return "待命，等待方位指令。";
}

function formatPilotProgress(completedCount, style) {
  if (style === "formal") return `报告：已发现${completedCount}个目标，继续执行。`;
  if (style === "plain") return `已发现${completedCount}个目标，继续找。`;
  return `已发现${completedCount}个目标，继续执行。`;
}

function formatPilotFailure(completedCount, result, style) {
  const base = `发现${completedCount}个目标，任务未完成。${result.failure_reason || ""}`;
  return style === "formal" ? `报告：${base}` : base;
}

function statusLabel(status) {
  if (status === "success") return "成功发现目标";
  if (status === "excluded") return "声呐排除";
  return "未完成";
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

function formatRunSonarSuffix(run) {
  if (!run?.sonar_recognition) return "";
  return `，声呐识别 ${formatSonarRecognition(run.sonar_recognition)}`;
}

function formatDecisionSonarResult(decision) {
  if (!decision.sonar_recognition) return "";
  return `声呐：${formatSonarRecognition(decision.sonar_recognition)}`;
}

function formatSonarRecognition(recognition) {
  const type = targetTypeLabel(recognition.target_type);
  const confidence = formatNumber((recognition.confidence || 0) * 100);
  const iff = recognition.is_blue_target ? "蓝方/敌方" : "红方/中立";
  const value = recognition.is_high_value_target ? "高价值" : recognition.is_real_target ? "真实目标" : "假目标";
  return `${type}，${value}，${iff}，置信度${confidence}%`;
}

function targetTypeLabel(type) {
  const match = TARGET_TYPE_OPTIONS.find(([value]) => value === type);
  return match ? match[1] : "不明";
}

function setStatus(text, badge) {
  statusText.textContent = text;
  simulationBadge.textContent = badge;
}

function setBusy(isBusy) {
  runSimulationBtn.disabled = isBusy;
  addTargetBtn.disabled = isBusy;
  discoveryRangeInput.disabled = isBusy;
  sonarTriggerRangeInput.disabled = isBusy;
  apiBaseUrlInput.disabled = isBusy;
  apiKeyInput.disabled = isBusy;
  agentChatInput.disabled = isBusy;
  sendAgentMessageBtn.disabled = isBusy;
  targetRows.querySelectorAll("input, select, button").forEach((element) => {
    element.disabled = isBusy || (element.dataset.key === "label") || (element.classList.contains("danger") && targets.length <= 1);
  });
}

function setChatBusy(isBusy) {
  agentChatInput.disabled = isBusy;
  sendAgentMessageBtn.disabled = isBusy;
  apiBaseUrlInput.disabled = isBusy;
  apiKeyInput.disabled = isBusy;
  sendAgentMessageBtn.textContent = isBusy ? "等待" : "发送";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

renderTargetRows();
renderAgentChat();
drawEmptyCanvases();
