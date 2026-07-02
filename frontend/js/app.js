// 매뉴얼 음성 도우미 — 프론트엔드
// 상태: idle → listening(파형+실시간 인식) → thinking(검색 중) → speaking(TTS, 탭하여 멈춤) → idle
// 제품 선택: 카테고리 칩(카테고리 스코핑) → 모델 칩(모델 스코핑). STT/TTS는 브라우저 Web Speech.

const micBtn = document.getElementById("mic-btn");
const statusEl = document.getElementById("status");
const conversationEl = document.getElementById("conversation");
const categoryChipsEl = document.getElementById("category-chips");
const modelChipsEl = document.getElementById("model-chips");
const suggestionsEl = document.getElementById("suggestions");
const listenPanel = document.getElementById("listen-panel");
const interimEl = document.getElementById("interim");
const textForm = document.getElementById("text-form");
const textInput = document.getElementById("text-input");

const IDLE_MSG = "버튼을 누르고 질문하세요";

// ── 제품 카테고리 ───────────────────────────────────────────
// 파일명이 washer지만 실제 내용은 식기세척기인 매뉴얼 보정
const CAT_OVERRIDES = {
  "lg-washer-d1220mf": "dishwasher",
  "lg-washer-mfl47377718": "dishwasher",
};
const CAT_LABELS = {
  washer: "🧺 세탁기",
  dishwasher: "🍽️ 식기세척기",
  microwave: "♨️ 전자레인지",
  purifier: "💧 정수기",
  vacuum: "🧹 청소기",
  fridge: "🧊 냉장고",
  dehumidifier: "💨 제습기",
  etc: "📦 기타",
};
const SUGGESTIONS = {
  all: ["예약 세탁은 최대 몇 시간까지 돼?", "전자레인지로 밥 데우는 법", "정수기 필터 교체 주기 알려줘"],
  washer: ["예약 세탁은 최대 몇 시간까지 돼?", "헹굼 횟수 바꾸는 법 알려줘", "통세척은 어떻게 해?"],
  dishwasher: ["차일드락 설정 어떻게 해?", "노즐 청소는 어떻게 해?"],
  microwave: ["쇠고기 500g 해동은 어떻게 해?", "밥 1인분 데우는 법 알려줘"],
  purifier: ["필터 교체 주기가 어떻게 돼?", "온수는 어떻게 사용해?"],
  vacuum: ["흡입이 잘 안 될 때 어떻게 해?", "먼지통 세척 방법 알려줘"],
  fridge: ["뜨거운 음식은 어떻게 보관해?", "탈취제 재생 방법 알려줘"],
  dehumidifier: ["리모컨 건전지 교환할 때 주의점은?", "실외기는 벽에서 얼마나 띄워야 해?"],
  etc: [],
};

function categoryOf(id) {
  if (CAT_OVERRIDES[id]) return CAT_OVERRIDES[id];
  if (id.includes("washer") || id.includes("sew")) return "washer";
  if (id.includes("microwave")) return "microwave";
  if (id.includes("waterpurifier")) return "purifier";
  if (id.includes("vacuum")) return "vacuum";
  if (id.includes("fridge")) return "fridge";
  if (id.includes("dehumidifier")) return "dehumidifier";
  return "etc";
}
const shortName = (id) => id.replace(/^lg-/, "");

let manualsByCat = {};   // { washer: [id, ...], ... }
let selectedCat = "all";
let selectedModel = null; // manual_id 또는 null

function currentScope() {
  if (selectedModel) return [selectedModel];
  if (selectedCat !== "all") return manualsByCat[selectedCat] || null;
  return null;
}

async function loadManuals() {
  try {
    const res = await fetch("/manuals");
    const ids = await res.json();
    manualsByCat = {};
    for (const id of ids) (manualsByCat[categoryOf(id)] ??= []).push(id);
    renderCategoryChips();
    renderSuggestions();
  } catch {
    setStatus("서버에 연결할 수 없습니다 — uvicorn이 실행 중인지 확인하세요");
  }
}

function renderCategoryChips() {
  categoryChipsEl.innerHTML = "";
  const cats = ["all", ...Object.keys(CAT_LABELS).filter((c) => manualsByCat[c]?.length)];
  for (const cat of cats) {
    const chip = document.createElement("button");
    chip.className = "chip" + (cat === selectedCat ? " selected" : "");
    chip.textContent = cat === "all" ? "전체" : CAT_LABELS[cat];
    chip.addEventListener("click", () => {
      selectedCat = cat;
      selectedModel = null;
      renderCategoryChips();
      renderModelChips();
      renderSuggestions();
    });
    categoryChipsEl.appendChild(chip);
  }
}

function renderModelChips() {
  const models = selectedCat === "all" ? [] : manualsByCat[selectedCat] || [];
  modelChipsEl.hidden = models.length < 2;
  modelChipsEl.innerHTML = "";
  if (models.length < 2) return;
  const allChip = document.createElement("button");
  allChip.className = "chip" + (selectedModel === null ? " selected" : "");
  allChip.textContent = "모델 전체";
  allChip.addEventListener("click", () => { selectedModel = null; renderModelChips(); });
  modelChipsEl.appendChild(allChip);
  for (const id of models) {
    const chip = document.createElement("button");
    chip.className = "chip" + (id === selectedModel ? " selected" : "");
    chip.textContent = shortName(id);
    chip.addEventListener("click", () => { selectedModel = id; renderModelChips(); });
    modelChipsEl.appendChild(chip);
  }
}

function renderSuggestions() {
  const qs = SUGGESTIONS[selectedCat] || [];
  suggestionsEl.innerHTML = "";
  suggestionsEl.hidden = qs.length === 0;
  for (const q of qs) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.textContent = q;
    chip.addEventListener("click", () => ask(q));
    suggestionsEl.appendChild(chip);
  }
}

// ── 상태 머신 ───────────────────────────────────────────────
let state = "idle"; // idle | listening | thinking | speaking

function setState(next) {
  state = next;
  micBtn.classList.toggle("listening", next === "listening");
  micBtn.classList.toggle("speaking", next === "speaking");
  micBtn.disabled = next === "thinking" || (!recognition && next === "idle" && micDisabled);
  listenPanel.hidden = next !== "listening";
  statusEl.classList.toggle("status--action", next === "speaking");
  if (next === "idle") {
    micBtn.textContent = "🎙️";
    setStatus(recognition ? IDLE_MSG : "이 브라우저는 음성 인식 미지원 — 텍스트로 질문하세요 (Chrome 권장)");
  } else if (next === "listening") {
    micBtn.textContent = "⏹";
    interimEl.innerHTML = "&nbsp;";
    setStatus("듣고 있어요 · 말을 마치면 자동 인식");
  } else if (next === "thinking") {
    micBtn.textContent = "🎙️";
    setStatus("매뉴얼 검색 중…");
  } else if (next === "speaking") {
    micBtn.textContent = "🔊";
    setStatus("답변 재생 중 · 탭하여 멈춤");
  }
}

// ── STT (Web Speech) ───────────────────────────────────────
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let micDisabled = false;

if (SpeechRecognition) {
  recognition = new SpeechRecognition();
  recognition.lang = "ko-KR";
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;

  recognition.onresult = (e) => {
    let interim = "", final = "";
    for (const r of e.results) (r.isFinal ? (final += r[0].transcript) : (interim += r[0].transcript));
    interimEl.textContent = (final || interim || " ").trim() || " ";
    if (final.trim()) {
      recognition.stop();
      ask(final.trim());
    }
  };
  recognition.onerror = (e) => {
    if (state !== "listening") return;
    setState("idle");
    setStatus(e.error === "not-allowed" ? "마이크 권한을 허용해주세요" : `음성 인식 오류: ${e.error}`);
  };
  recognition.onend = () => {
    if (state === "listening") setState("idle"); // 말 없이 종료/취소
  };
} else {
  micDisabled = true;
  micBtn.disabled = true;
}

micBtn.addEventListener("click", () => {
  if (state === "speaking") { speechSynthesis.cancel(); setState("idle"); return; }
  if (state === "listening") { recognition.stop(); return; }
  if (state === "idle" && recognition) { speechSynthesis.cancel(); setState("listening"); recognition.start(); }
});
statusEl.addEventListener("click", () => {
  if (state === "speaking") { speechSynthesis.cancel(); setState("idle"); }
});

// ── 텍스트 입력 ────────────────────────────────────────────
textForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = textInput.value.trim();
  if (!text || state === "thinking") return;
  textInput.value = "";
  ask(text);
});

// ── 질의 → 답변 ────────────────────────────────────────────
let thinkingBubble = null;

async function ask(text) {
  speechSynthesis.cancel();
  addBubble(text, "user");
  showThinking();
  setState("thinking");
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, manual_ids: currentScope() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `서버 오류 (${res.status})`);
    }
    const data = await res.json();
    hideThinking();
    addAnswer(data.answer, data.sources);
    speak(data.answer);
  } catch (err) {
    hideThinking();
    addBubble(`⚠️ ${err.message}`, "error");
    setState("idle");
  }
}

function showThinking() {
  thinkingBubble = document.createElement("div");
  thinkingBubble.className = "bubble bubble--bot bubble--thinking";
  thinkingBubble.innerHTML = "<span></span><span></span><span></span>";
  conversationEl.appendChild(thinkingBubble);
  thinkingBubble.scrollIntoView({ behavior: "smooth", block: "end" });
}
function hideThinking() {
  thinkingBubble?.remove();
  thinkingBubble = null;
}

// ── TTS ────────────────────────────────────────────────────
function speak(text) {
  speechSynthesis.cancel();
  const spoken = text.replace(/\(출처[^)]*\)/g, "").trim(); // 출처 표기는 칩으로 보이므로 낭독 생략
  const u = new SpeechSynthesisUtterance(spoken);
  u.lang = "ko-KR";
  u.rate = 1.05;
  u.onend = () => { if (state === "speaking") setState("idle"); };
  u.onerror = () => { if (state === "speaking") setState("idle"); };
  setState("speaking");
  speechSynthesis.speak(u);
}

// ── 렌더링 ─────────────────────────────────────────────────
function setStatus(msg) { statusEl.textContent = msg; }

function addBubble(text, who) {
  const div = document.createElement("div");
  div.className = `bubble bubble--${who === "error" ? "error" : who}`;
  div.textContent = text;
  conversationEl.appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
  return div;
}

function addAnswer(answer, sources = []) {
  const div = addBubble(answer, "bot");
  if (!sources.length) return;
  const wrap = document.createElement("div");
  wrap.className = "src-chips";
  const topManual = sources[0]?.manual_id;
  sources.slice(0, 3).forEach((s, i) => {
    const chip = document.createElement("span");
    chip.className = "src-chip" + (i === 0 ? " src-chip--top" : "");
    const label = i === 0 || s.manual_id !== topManual
      ? `${shortName(s.manual_id)} · ${s.page}쪽`
      : `${s.page}쪽`;
    chip.innerHTML = `<span class="src-chip__num">${i + 1}</span> ${label}`;
    wrap.appendChild(chip);
  });
  div.appendChild(wrap);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

// ── 시작 ───────────────────────────────────────────────────
loadManuals();
setState("idle");
