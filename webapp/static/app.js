const TOPICS = {
  all: {
    label: '무엇이든 물어봐요',
    greeting: '안녕하세요! 저는 말맥이에요. 한국 생활이나 문화에 대해 무엇이 궁금한가요?\nHi! I’m Malmaek. Ask me anything about life in Korea.',
    suggestions: ['“생각해 볼게요”는 어떤 뜻인가요?', 'What should I check before signing a lease?', '반찬은 셀프입니다가 무슨 뜻이야?']
  },
  culture: {
    label: '한국 문화',
    greeting: '한국 문화에 대해 무엇이 궁금한가요? 명절이나 전통, 일상의 작은 장면까지 함께 살펴봐요.\nWhat would you like to know about Korean culture?',
    suggestions: ['추석에는 보통 무엇을 하나요?', 'What does a tiger motif symbolize?', 'What is Seollal?']
  },
  etiquette: {
    label: '예절과 표현',
    greeting: '한국의 예절과 표현에 대해 무엇이 궁금한가요? 상대와 상황을 알려주면 더 잘 설명할 수 있어요.\nAsk me about greetings, polite speech, or tricky expressions.',
    suggestions: ['교수님께 처음 인사할 때 뭐라고 해요?', 'What does “생각해 볼게요” mean?', '친구에게 정중하게 거절하려면?']
  },
  food: {
    label: '음식과 식사',
    greeting: '한국 음식과 식사 문화에 대해 무엇이 궁금한가요? 메뉴나 식당 안내문도 물어보세요.\nCurious about a dish or a restaurant sign?',
    suggestions: ['“반찬은 셀프입니다”가 무슨 뜻인가요?', 'What is banchan?', '식당에서 물은 어떻게 주문해요?']
  },
  transit: {
    label: '교통과 이동',
    greeting: '한국 교통에 대해 무엇이 궁금한가요? 버스나 지하철 이용 방법을 함께 확인해 봐요.\nAsk me about getting around Korea.',
    suggestions: ['버스 교통카드 없이 타도 되나요?', 'How do I pay on a Korean bus?', '지하철 환승은 어떻게 하나요?']
  },
  life: {
    label: '유학생 생활',
    greeting: '한국 유학생 생활에 대해 무엇이 궁금한가요? 집, 등록, 보험처럼 중요한 내용은 공식 안내를 찾아볼게요.\nAsk about practical student life in Korea.',
    suggestions: ['외국인등록은 어디서 신청하나요?', 'What should I check before signing a housing lease?', '이사 후 전입신고는 어떻게 하나요?']
  }
};

const threads = Object.fromEntries(Object.keys(TOPICS).map(key => [key, []]));
let category = 'all';
let pending = false;

const $ = selector => document.querySelector(selector);
const messagesEl = $('#messages');
const suggestionsEl = $('#suggestions');
const input = $('#message-input');
const sendButton = $('#send-button');

function safeSourceUrl(value) {
  try {
    const url = new URL(value);
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null;
  } catch (_) { return null; }
}

function avatar(role) {
  const element = document.createElement('div');
  element.className = role === 'user' ? 'message-avatar user-avatar' : 'message-avatar';
  if (role === 'user') {
    element.textContent = '나';
  } else {
    const img = document.createElement('img');
    img.src = '/static/mascot.svg';
    img.alt = '';
    element.append(img);
  }
  return element;
}

function addMessage(role, content, extras = {}) {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;
  const body = document.createElement('div');
  body.className = 'message-content';
  const name = document.createElement('div');
  name.className = 'message-name';
  name.textContent = role === 'user' ? '나' : '말맥';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  body.append(name, bubble);
  if (extras.sources?.length) {
    const wrapper = document.createElement('div');
    wrapper.className = 'sources';
    extras.sources.forEach(source => {
      const href = safeSourceUrl(source.url);
      if (!href) return;
      const link = document.createElement('a');
      link.href = href;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = `↗ ${source.title}`;
      link.title = `${source.title}\n${source.license}${source.updated_at ? `\n자료 날짜: ${source.updated_at}` : ''}`;
      wrapper.append(link);
    });
    body.append(wrapper);
    const details = document.createElement('details');
    details.className = 'source-details';
    const summary = document.createElement('summary');
    summary.textContent = '자료 날짜와 이용 조건 / Source details';
    details.append(summary);
    extras.sources.forEach(source => {
      const line = document.createElement('p');
      line.textContent = `${source.title}${source.updated_at ? ` · ${source.updated_at}` : ''}\n${source.license}`;
      details.append(line);
    });
    body.append(details);
  }
  if (extras.source_type && extras.source_type !== 'none') {
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    const mode = extras.used_adapter ? 'KoRMo + 한국어 LoRA' : '기본 KoRMo';
    const source = {
      official: '공식 생활 안내', heritage: '전통문화 자료',
      encyclopedia: '백과 자료', general: '일반 안내'
    }[extras.source_type] || '일반 안내';
    meta.textContent = `${mode} · ${source}`;
    body.append(meta);
  }
  if (role === 'user') row.append(body, avatar(role));
  else row.append(avatar(role), body);
  messagesEl.append(row);
}

function render() {
  const current = threads[category];
  const showWelcome = category === 'all' && !current.some(turn => turn.role === 'user');
  $('#welcome').classList.toggle('hidden', !showWelcome);
  $('#chat-panel').classList.toggle('expanded', !showWelcome);
  $('#route-label').textContent = TOPICS[category].label;
  $('.chat-heading-kicker').textContent = category === 'all' ? 'YOUR KOREA COMPANION' : 'LET’S EXPLORE TOGETHER';
  $('#chat-title').textContent = category === 'all' ? '말맥과 대화하기' : TOPICS[category].label;
  document.querySelectorAll('[data-category]').forEach(button => {
    if (button.classList.contains('category-item')) {
      const active = button.dataset.category === category;
      button.classList.toggle('active', active);
      button.setAttribute('aria-current', active ? 'page' : 'false');
    }
  });
  messagesEl.replaceChildren();
  addMessage('assistant', TOPICS[category].greeting);
  current.forEach(turn => addMessage(turn.role, turn.content, turn));
  suggestionsEl.replaceChildren();
  if (!current.some(turn => turn.role === 'user')) {
    TOPICS[category].suggestions.forEach(question => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.textContent = question;
      chip.addEventListener('click', () => {
        input.value = question;
        submitMessage();
      });
      suggestionsEl.append(chip);
    });
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setCategory(next) {
  if (!TOPICS[next]) return;
  category = next;
  render();
  closeMenu();
  if (next !== 'all') $('#chat-panel').scrollIntoView({behavior: 'smooth', block: 'start'});
}

function setPending(value) {
  pending = value;
  sendButton.disabled = value;
  input.disabled = value;
}

function showTyping() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.id = 'typing-row';
  const body = document.createElement('div');
  body.className = 'message-content';
  const bubble = document.createElement('div');
  bubble.className = 'bubble typing';
  bubble.setAttribute('aria-label', '말맥이 답변을 작성 중입니다');
  for (let i = 0; i < 3; i++) bubble.append(document.createElement('i'));
  body.append(bubble);
  row.append(avatar('assistant'), body);
  messagesEl.append(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function submitMessage() {
  const message = input.value.trim();
  if (!message || pending) return;
  const activeCategory = category;
  const history = threads[activeCategory].slice(-8).map(turn => ({role: turn.role, content: turn.content}));
  threads[activeCategory].push({role: 'user', content: message});
  input.value = '';
  input.style.height = '';
  render();
  showTyping();
  setPending(true);
  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message, history, category: activeCategory, language: $('#language').value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '요청을 처리하지 못했어요.');
    threads[activeCategory].push({role: 'assistant', content: data.answer,
      sources: data.sources, source_type: data.source_type, used_adapter: data.used_adapter});
  } catch (error) {
    threads[activeCategory].push({role: 'assistant', content: `지금은 답변을 가져오지 못했어요. 잠시 후 다시 시도해 주세요.\n${error.message}`, source_type: 'none'});
  } finally {
    setPending(false);
    if (category === activeCategory) render();
    input.focus();
  }
}

function closeMenu() {
  $('#sidebar').classList.remove('open');
  $('#sidebar-backdrop').hidden = true;
  $('#menu-button').setAttribute('aria-expanded', 'false');
}

async function refreshStatus() {
  const status = $('#service-status');
  try {
    const response = await fetch('/api/health');
    const health = await response.json();
    status.classList.toggle('offline', !health.ready);
    status.lastChild.textContent = health.ready ? '말맥 연결됨' : '모델 설정 필요';
  } catch (_) {
    status.classList.add('offline');
    status.lastChild.textContent = '연결을 확인할 수 없어요';
  }
}

document.querySelectorAll('[data-category]').forEach(button => button.addEventListener('click', () => setCategory(button.dataset.category)));
$('#new-chat').addEventListener('click', () => { threads[category] = []; render(); closeMenu(); input.focus(); });
$('#hero-ask').addEventListener('click', () => { $('#chat-panel').scrollIntoView({behavior: 'smooth'}); input.focus(); });
$('#composer').addEventListener('submit', event => { event.preventDefault(); submitMessage(); });
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submitMessage();
  }
});
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = `${Math.min(input.scrollHeight, 115)}px`; });
$('#menu-button').addEventListener('click', () => {
  const opening = !$('#sidebar').classList.contains('open');
  $('#sidebar').classList.toggle('open', opening);
  $('#sidebar-backdrop').hidden = !opening;
  $('#menu-button').setAttribute('aria-expanded', String(opening));
});
$('#sidebar-backdrop').addEventListener('click', closeMenu);
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });
render();
refreshStatus();
