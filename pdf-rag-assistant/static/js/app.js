const form = document.querySelector("#chat-form");
const questionInput = document.querySelector("#question");
const messages = document.querySelector("#messages");
const welcomeCard = document.querySelector("#welcome-card");
const HISTORY_KEY = "pdf-chat-history";

function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch (error) {
    return [];
  }
}

function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-40)));
}

function hideWelcomeIfNeeded() {
  if (messages.querySelector(".message") && welcomeCard) {
    welcomeCard.style.display = "none";
  }
}

function addMessage(text, role, sources = []) {
  const bubble = document.createElement("div");
  bubble.className = `message ${role}`;

  const content = document.createElement("div");
  content.textContent = text;
  bubble.appendChild(content);

  if (sources.length) {
    const sourceBox = document.createElement("div");
    sourceBox.className = "sources";

    const title = document.createElement("strong");
    title.textContent = "Source:";
    sourceBox.appendChild(title);

    const list = document.createElement("ul");
    sources.forEach((source) => {
      const item = document.createElement("li");
      item.textContent = `${source.file} (Page ${source.page})`;
      list.appendChild(item);
    });
    sourceBox.appendChild(list);
    bubble.appendChild(sourceBox);
  }

  messages.appendChild(bubble);
  hideWelcomeIfNeeded();
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function addThinkingMessage() {
  const bubble = document.createElement("div");
  bubble.className = "message assistant thinking";
  bubble.innerHTML = '<span class="spinner"></span><span>Thinking...</span>';
  messages.appendChild(bubble);
  hideWelcomeIfNeeded();
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function renderHistory() {
  const history = loadHistory();
  history.forEach((message) => {
    addMessage(message.text, message.role, message.sources || []);
  });
}

if (form) {
  renderHistory();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    const history = loadHistory();
    addMessage(question, "user");
    history.push({ role: "user", text: question });
    saveHistory(history);
    questionInput.value = "";

    const thinking = addThinkingMessage();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      const data = await response.json();
      thinking.remove();

      const answer = data.answer || "No answer returned.";
      const sources = data.sources || [];
      addMessage(answer, "assistant", sources);

      const updatedHistory = loadHistory();
      updatedHistory.push({ role: "assistant", text: answer, sources });
      saveHistory(updatedHistory);
    } catch (error) {
      thinking.remove();
      addMessage("Something went wrong while asking the server.", "assistant");
    }
  });
}
