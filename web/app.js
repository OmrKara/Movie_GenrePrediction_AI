const modelSelect = document.getElementById("model-select");
const form = document.getElementById("composer-form");
const textarea = document.getElementById("description");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("messages");
const welcome = document.getElementById("welcome");
const characterCount = document.getElementById("character-count");
const conversation = document.getElementById("conversation");

const labelNames = {
  Action: "Action",
  Comedy: "Comedy",
  Drama: "Drama",
  Horror: "Horror",
  Romance: "Romance",
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) throw new Error("Model listesi yüklenemedi.");
  const data = await response.json();
  modelSelect.replaceChildren();
  for (const model of data.models) {
    const option = document.createElement("option");
    option.value = model.id;
    const score = Number.isFinite(model.macro_f1) ? ` · Macro-F1 ${model.macro_f1.toFixed(3)}` : "";
    option.textContent = `${model.title}${score}`;
    modelSelect.append(option);
  }
  if (data.models.some((model) => model.id === "u3_logistic_regression")) {
    modelSelect.value = "u3_logistic_regression";
  }
}

function addUserMessage(description) {
  const wrapper = el("article", "message user");
  wrapper.append(el("div", "user-bubble", description));
  messages.append(wrapper);
  return wrapper;
}

function addAssistantShell(modelTitle) {
  const wrapper = el("article", "message assistant-message");
  const heading = el("div", "assistant-heading");
  heading.append(el("span", "assistant-mark", "L"));
  heading.append(el("span", "", modelTitle));
  const body = el("div", "assistant-body");
  body.append(el("div", "loading", "Tür tahminleri hesaplanıyor"));
  wrapper.append(heading, body);
  messages.append(wrapper);
  return body;
}

function renderResult(container, result) {
  container.replaceChildren();
  const card = el("section", "result-card");
  const top = el("div", "result-top");
  top.append(el("h2", "", "Tahmin edilen türler"));
  top.append(el("span", "threshold", `Karar eşiği: ${(result.threshold * 100).toFixed(0)}%`));

  const chips = el("div", "prediction-chips");
  if (result.predicted_labels.length) {
    for (const label of result.predicted_labels) chips.append(el("span", "chip", labelNames[label] || label));
  } else {
    chips.append(el("span", "chip empty", "Eşiği geçen tür yok"));
  }

  const topClass = el("p", "top-class");
  const topLabel = labelNames[result.top_label] || result.top_label;
  topClass.append(document.createTextNode("En yüksek olasılık: "));
  topClass.append(el("strong", "", topLabel));
  topClass.append(document.createTextNode(` · ${(result.probabilities[result.top_label] * 100).toFixed(1)}%`));

  const list = el("div", "probability-list");
  for (const [label, probability] of Object.entries(result.probabilities)) {
    const row = el("div", "probability-row");
    row.append(el("span", "probability-label", labelNames[label] || label));
    const track = el("div", "probability-track");
    const fill = el("div", "probability-fill");
    fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
    track.append(fill);
    row.append(track);
    row.append(el("span", "probability-value", `${(probability * 100).toFixed(1)}%`));
    list.append(row);
  }

  card.append(top, chips, topClass, list);
  card.append(el("p", "model-footnote", "Olasılıklar, seçilen modelin skorlarıdır. SVM skorları olasılık kalibrasyonu içermez."));
  container.append(card);
}

function scrollToLatest() {
  requestAnimationFrame(() => {
    conversation.scrollTop = conversation.scrollHeight;
  });
}

async function submitDescription(description) {
  const trimmed = description.trim();
  if (!trimmed || sendButton.disabled) return;
  welcome.hidden = true;
  const selected = modelSelect.selectedOptions[0];
  const modelName = modelSelect.value;
  addUserMessage(trimmed);
  const resultContainer = addAssistantShell(selected ? selected.textContent : modelName);
  sendButton.disabled = true;
  textarea.disabled = true;
  scrollToLatest();

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ description: trimmed, model: modelName }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Tahmin sırasında hata oluştu.");
    renderResult(resultContainer, result);
  } catch (error) {
    resultContainer.replaceChildren(el("p", "error-text", error.message));
  } finally {
    sendButton.disabled = false;
    textarea.disabled = false;
    textarea.value = "";
    updateComposer();
    textarea.focus();
    scrollToLatest();
  }
}

function updateComposer() {
  textarea.style.height = "auto";
  textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
  characterCount.textContent = `${textarea.value.length} / 5000`;
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitDescription(textarea.value);
});

textarea.addEventListener("input", updateComposer);
textarea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    textarea.value = button.textContent;
    updateComposer();
    textarea.focus();
  });
});

document.getElementById("new-chat").addEventListener("click", () => {
  messages.replaceChildren();
  welcome.hidden = false;
  textarea.value = "";
  updateComposer();
  textarea.focus();
});

loadModels().catch((error) => {
  const option = document.createElement("option");
  option.value = "u3_logistic_regression";
  option.textContent = "U3 · Lojistik regresyon";
  modelSelect.replaceChildren(option);
  const warning = el("p", "error-text", error.message);
  messages.append(warning);
});
