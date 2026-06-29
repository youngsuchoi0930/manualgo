// 매뉴얼 음성 도우미 — 프론트엔드 로직 (스켈레톤)
// 흐름: 마이크 녹음 -> /ask 전송(STT는 백엔드) -> 답변 표시 -> 음성 재생
//
// TODO: MediaRecorder로 녹음, fetch로 백엔드 /ask 호출, 출처/음성 렌더링.

const API_BASE = ""; // 예: "http://localhost:8000"

const micBtn = document.getElementById("mic-btn");
const statusEl = document.getElementById("status");
const conversationEl = document.getElementById("conversation");
const sourcesEl = document.getElementById("sources");
const sourcesListEl = document.getElementById("sources-list");

let mediaRecorder = null;
let chunks = [];

micBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  await startRecording();
});

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];

  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstart = () => {
    micBtn.classList.add("recording");
    statusEl.textContent = "듣고 있어요…";
  };
  mediaRecorder.onstop = async () => {
    micBtn.classList.remove("recording");
    statusEl.textContent = "검색 중…";
    const blob = new Blob(chunks, { type: "audio/webm" });
    await sendQuestion(blob);
    stream.getTracks().forEach((t) => t.stop());
  };
  mediaRecorder.start();
}

async function sendQuestion(audioBlob) {
  const form = new FormData();
  form.append("audio", audioBlob, "question.webm");

  // TODO: 백엔드 /ask 구현 후 연결
  const res = await fetch(`${API_BASE}/ask`, { method: "POST", body: form });
  const data = await res.json();
  renderTurn(data);
}

function renderTurn(data) {
  addBubble(data.question, "user");
  addBubble(data.answer, "bot");
  renderSources(data.sources);
  if (data.audio_url) new Audio(data.audio_url).play();
  statusEl.textContent = "버튼을 누르고 질문하세요";
}

function addBubble(text, who) {
  const div = document.createElement("div");
  div.className = `bubble bubble--${who}`;
  div.textContent = text;
  conversationEl.appendChild(div);
}

function renderSources(sources = []) {
  if (!sources.length) return;
  sourcesEl.hidden = false;
  sourcesListEl.innerHTML = "";
  for (const s of sources) {
    const li = document.createElement("li");
    li.textContent = `${s.manual_id} · ${s.page}p${s.section ? " · " + s.section : ""}`;
    sourcesListEl.appendChild(li);
  }
}
