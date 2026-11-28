import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { FileText, Trash2, Send, CheckCircle2, Moon, Sun, Copy, Edit2, User, AlertTriangle, X, Info, Settings, Cloud, HardDrive, Minus, Square } from 'lucide-react';
import { api, apiJson, getBackend, isDesktop } from './api';
import Logo from './Logo';


// Must match the formats backend/parser.py can read (it rejects anything else).
const ACCEPTED_FILE_TYPES = [
  // documents, spreadsheets, presentations
  '.pdf', '.docx', '.txt', '.md', '.markdown', '.html', '.htm',
  '.xlsx', '.csv', '.tsv', '.pptx',
  // images (OCR)
  '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp',
  // code
  '.py', '.js', '.jsx', '.mjs', '.cjs', '.ts', '.tsx', '.java', '.kt', '.kts', '.scala', '.go', '.rs',
  '.swift', '.c', '.h', '.cpp', '.cc', '.cxx', '.hpp', '.cs', '.php', '.rb', '.lua', '.pl', '.r',
  '.ex', '.exs', '.hs', '.sol', '.proto', '.ps1', '.sh', '.bash', '.zsh', '.sql', '.css', '.scss',
  // config and logs
  '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.xml', '.properties', '.log',
].join(',');

// Upload limits. The backend sends its real values from GET /limits; these are the fallback.
// "enforced" is false on a local setup (limits apply only when deployed on Render).
const DEFAULT_LIMITS = { enforced: true, max_file_mb: 10, max_image_mb: 10, max_files: 5, max_total_mb: 25, max_ocr_pages: 20, max_chunks: 1500 };
const MB = 1024 * 1024;
const IMAGE_EXT_RE = /\.(jpe?g|png|bmp|tiff?|webp)$/i;
const fmtMB = (bytes) => `${(bytes / MB).toFixed(1)} MB`;

// Real indexing progress: each stage of a background job covers part of the ring.
const STAGE_RANGE = {
  queued: [0, 0], parsing: [0.02, 0.1], reading: [0.02, 0.15], ocr: [0.15, 0.6], pictures: [0.1, 0.3], describing: [0.3, 0.65],
  embedding: [0.6, 0.95], saving: [0.95, 0.98], reused: [0.9, 0.98], done: [1, 1], failed: [1, 1],
};
const jobFraction = (job) => {
  const [from, to] = STAGE_RANGE[job.stage] || [0, 0];
  const part = job.total ? Math.min(1, job.done / job.total) : 0;
  return from + (to - from) * part;
};
// Local AI memory estimate (GB) while answering: model weights (4-bit, as Ollama downloads them)
// + context-window cache per 1K tokens + ~0.5 GB working memory. Used to warn on small computers.
const OLLAMA_CHAT_MODELS = [
  { id: 'llama3.2:3b', label: 'Llama 3.2 3B', note: 'lighter, for 8 GB computers', weights: 2.0, cachePer1k: 0.11 },
  { id: 'llama3.1:8b', label: 'Llama 3.1 8B', note: 'better answers, for 16 GB or more', weights: 4.9, cachePer1k: 0.125 },
];
// Image model for photos without text. DocuMind loads one model at a time, so it never adds to the chat model.
const OLLAMA_VISION_MODELS = [
  { id: 'qwen3-vl:4b', label: 'Qwen3-VL 4B', note: 'for 8 GB computers', weights: 3.3, cachePer1k: 0.14 },
  { id: 'qwen3-vl:8b', label: 'Qwen3-VL 8B', note: 'more accurate, for 16 GB or more', weights: 6.1, cachePer1k: 0.14 },
];
// The image model's window: 4K on 8 GB computers (one photo per question), 8K otherwise.
const visionContext = (lowMemory) => (lowMemory ? 4096 : 8192);
const CONTEXT_OPTIONS = [
  { id: '16384', label: '16K tokens', note: 'reads all matching passages' },
  { id: '8192', label: '8K tokens', note: 'uses less memory' },
];
const estimateMemoryGB = (modelId, context) => {
  const m = [...OLLAMA_CHAT_MODELS, ...OLLAMA_VISION_MODELS].find(x => x.id === modelId);
  if (!m) return null;
  return m.weights + m.cachePer1k * (Number(context) / 1024) + 0.5;
};
const newSessionId = () => (window.crypto?.randomUUID ? window.crypto.randomUUID() : `s-${Date.now()}-${Math.random()}`);

// Custom Tooltip Component
function FileTooltip({ text, children }) {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0 });
  const triggerRef = useRef(null);

  const handleMouseEnter = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setCoords({
        top: rect.top + rect.height / 2,
        left: rect.right + 12
      });
      setIsVisible(true);
    }
  };

  const handleMouseLeave = () => {
    setIsVisible(false);
  };

  return (
    <>
      <div 
        ref={triggerRef}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        className="w-full"
      >
        {children}
      </div>

      {isVisible && createPortal(
        <div 
          className="fixed z-[9999] -translate-y-1/2 pointer-events-none"
          style={{ top: coords.top, left: coords.left }}
        >
          <div className="bg-gray-900/95 dark:bg-gray-700/95 backdrop-blur-sm text-white text-xs py-1.5 px-3 rounded-lg shadow-2xl whitespace-nowrap border border-gray-700 dark:border-gray-600">
            {text}
          </div>
        </div>,
        document.body
      )}
    </>
  );
}

// Custom Toast Notification Component
function Toast({ message, type, onClose }) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    requestAnimationFrame(() => setIsVisible(true));
    const timer = setTimeout(() => {
      setIsVisible(false);
      setTimeout(onClose, 300);
    }, 3500);
    return () => clearTimeout(timer);
  }, [onClose]);

  const isError = type === 'error';

  return createPortal(
    <div 
      className={`fixed top-6 left-1/2 -translate-x-1/2 z-[9999] transition-all duration-300 ease-in-out ${
        isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'
      }`}
    >
      <div className={`flex items-center gap-3 px-4 py-3 rounded-xl shadow-2xl border backdrop-blur-md ${
        isError 
          ? 'bg-red-50/95 dark:bg-red-900/90 border-red-200 dark:border-red-800 text-red-800 dark:text-red-200' 
          : 'bg-green-50/95 dark:bg-green-900/90 border-green-200 dark:border-green-800 text-green-800 dark:text-green-200'
      }`}>
        {isError ? <AlertTriangle className="w-5 h-5 shrink-0" /> : <CheckCircle2 className="w-5 h-5 shrink-0" />}
        <p className="text-sm font-medium">{message}</p>
        <button onClick={() => { setIsVisible(false); setTimeout(onClose, 300); }} className="ml-2 hover:opacity-70 transition-opacity">
          <X className="w-4 h-4" />
        </button>
      </div>
    </div>,
    document.body
  );
}

// Desktop window chrome. macOS keeps its own close/minimise/zoom buttons, drawn over our header
// (tauri.macos.conf.json: titleBarStyle "Overlay"); Windows and Linux have no system title bar
// (decorations: false), so DocuMind draws these buttons itself.
const isMacOS = typeof navigator !== 'undefined' && /Mac/i.test(navigator.platform || navigator.userAgent);
const drawsOwnWindowControls = isDesktop && !isMacOS;

function WindowControls() {
  const call = async (action) => {
    const { getCurrentWindow } = await import('@tauri-apps/api/window');
    const win = getCurrentWindow();
    if (action === 'minimize') await win.minimize();
    if (action === 'maximize') await win.toggleMaximize();
    if (action === 'close') await win.close();
  };
  const base = "w-11 h-8 flex items-center justify-center rounded-md text-gray-500 dark:text-gray-400 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500";
  return (
    <div className="flex items-center gap-0.5 pl-2 ml-1 border-l border-gray-200 dark:border-gray-800">
      <button aria-label="Minimize" title="Minimize" onClick={() => call('minimize')}
              className={`${base} hover:bg-gray-200 dark:hover:bg-gray-800`}><Minus className="w-4 h-4" /></button>
      <button aria-label="Maximize" title="Maximize" onClick={() => call('maximize')}
              className={`${base} hover:bg-gray-200 dark:hover:bg-gray-800`}><Square className="w-3.5 h-3.5" /></button>
      <button aria-label="Close" title="Close" onClick={() => call('close')}
              className={`${base} hover:bg-red-600 hover:text-white dark:hover:bg-red-600`}><X className="w-4 h-4" /></button>
    </div>
  );
}

function OllamaSettings({ form, setForm }) {
  const presets = OLLAMA_CHAT_MODELS.map(m => m.id);
  const [custom, setCustom] = useState(!presets.includes(form.ollama_chat_model));
  const set = (key, value) => setForm(f => ({ ...f, [key]: value }));
  const rec = form.recommended || {};
  const ram = form.ram_gb || 0;
  const need = estimateMemoryGB(form.ollama_chat_model, form.ollama_context);
  // macOS/Windows, DocuMind and a browser need ~3.5 GB; beyond the rest, Ollama spills onto the CPU.
  const tooHeavy = ram > 0 && need !== null && need > ram - 3.5;
  const model = OLLAMA_CHAT_MODELS.find(m => m.id === form.ollama_chat_model);
  const ctx = CONTEXT_OPTIONS.find(c => c.id === form.ollama_context);
  const vision = OLLAMA_VISION_MODELS.find(m => m.id === form.ollama_vision_model);
  const visionNeed = vision ? estimateMemoryGB(vision.id, visionContext(form.low_memory)) : null;
  const visionTooHeavy = ram > 0 && visionNeed !== null && visionNeed > ram - 3.5;
  const field = "mt-1 w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 text-sm focus:outline-none focus:border-blue-500";
  const recModel = OLLAMA_CHAT_MODELS.find(m => m.id === rec.ollama_chat_model);
  const recCtx = CONTEXT_OPTIONS.find(c => c.id === rec.ollama_context);

  return (
    <div className="space-y-3 mb-4">
      <label className="block text-sm text-gray-700 dark:text-gray-300">
        Chat model
        <select
          className={field}
          value={custom ? '__custom' : form.ollama_chat_model}
          onChange={e => {
            if (e.target.value === '__custom') { setCustom(true); return; }
            setCustom(false);
            set('ollama_chat_model', e.target.value);
          }}
        >
          {OLLAMA_CHAT_MODELS.map(m => (
            <option key={m.id} value={m.id}>
              {m.label}: {m.note}{m.id === rec.ollama_chat_model ? ' (recommended)' : ''}
            </option>
          ))}
          <option value="__custom">Another Ollama model…</option>
        </select>
      </label>
      {custom && (
        <label className="block text-sm text-gray-700 dark:text-gray-300">
          Model name
          <input className={field} value={form.ollama_chat_model} placeholder="e.g. qwen2.5:7b"
                 onChange={e => set('ollama_chat_model', e.target.value)} />
        </label>
      )}

      <label className="block text-sm text-gray-700 dark:text-gray-300">
        Context window
        <select className={field} value={form.ollama_context} onChange={e => set('ollama_context', e.target.value)}>
          {CONTEXT_OPTIONS.map(c => (
            <option key={c.id} value={c.id}>
              {c.label}: {c.note}{c.id === rec.ollama_context ? ' (recommended)' : ''}
            </option>
          ))}
        </select>
      </label>

      <label className="block text-sm text-gray-700 dark:text-gray-300">
        Image model, for photos without text
        <select className={field} value={form.ollama_vision_model || 'off'} onChange={e => set('ollama_vision_model', e.target.value)}>
          {OLLAMA_VISION_MODELS.map(m => (
            <option key={m.id} value={m.id}>
              {m.label}: {m.note}{m.id === rec.ollama_vision_model ? ' (recommended)' : ''}
            </option>
          ))}
          <option value="off">Off: only read text in images</option>
        </select>
        <span className="block mt-1 text-xs text-gray-500 dark:text-gray-400">
          Images with text are always read with OCR. The image model describes photos without text (products, charts, scenes)
          and answers questions about them. It runs on this computer only.
        </span>
      </label>

      {ram > 0 && (
        <p className="text-xs text-gray-500 dark:text-gray-400">
          This computer has {Math.round(ram)} GB of memory.
          {need !== null && ` ${model.label} with a ${ctx.label.replace(' tokens', '')} window needs about ${need.toFixed(1)} GB while answering.`}
          {vision && ` ${vision.label} needs about ${visionNeed.toFixed(1)} GB while it looks at photos.`}
          {vision && need !== null && ' DocuMind loads one model at a time, so they never need memory together.'}
        </p>
      )}

      {visionTooHeavy && !tooHeavy && (
        <div className="text-xs text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3" role="alert">
          <strong className="block mb-1">Photos will be analyzed slowly on this computer.</strong>
          {vision.label} needs about {visionNeed.toFixed(1)} GB, more than is free next to your system and other apps.
          Expect the CPU near full load and each photo taking minutes.
          {rec.ollama_vision_model && rec.ollama_vision_model !== vision.id && ` ${OLLAMA_VISION_MODELS.find(m => m.id === rec.ollama_vision_model)?.label} is recommended for this computer.`}
          <button type="button" onClick={() => set('ollama_vision_model', rec.ollama_vision_model)}
                  className="block mt-2 font-medium underline focus:outline-none">
            Use the recommended image model
          </button>
        </div>
      )}

      {tooHeavy && (
        <div className="text-xs text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-900/20 rounded-lg p-3" role="alert">
          <strong className="block mb-1">This will run slowly and use a lot of CPU on this computer.</strong>
          {model.label} with a {ctx.label.replace(' tokens', '')} window needs about {need.toFixed(1)} GB, more than is free next to
          your system and other apps. Ollama will run part of the model on the CPU instead: expect the CPU near full load,
          a hot, noisy computer, other apps slowing down, and answers taking minutes instead of seconds.
          {recModel && recCtx && ` ${recModel.label} with ${recCtx.label.replace(' tokens', '')} is recommended for this computer.`}
          <button
            type="button"
            onClick={() => { setCustom(false); setForm(f => ({ ...f, ollama_chat_model: rec.ollama_chat_model, ollama_context: rec.ollama_context, ollama_vision_model: rec.ollama_vision_model || f.ollama_vision_model })); }}
            className="block mt-2 font-medium underline focus:outline-none"
          >
            Use the recommended settings
          </button>
        </div>
      )}

      <details className="text-sm text-gray-700 dark:text-gray-300">
        <summary className="cursor-pointer text-xs text-gray-500">Ollama address and embedding model</summary>
        <label className="block mt-2">
          Ollama address
          <input className={field} value={form.ollama_url || ''} onChange={e => set('ollama_url', e.target.value)} />
        </label>
        <label className="block mt-2">
          Embedding model
          <input className={field} value={form.ollama_embed_model || ''} onChange={e => set('ollama_embed_model', e.target.value)} />
        </label>
      </details>

      <p className="text-xs text-gray-500">
        Install Ollama from ollama.com, then run: <code>ollama pull {form.ollama_embed_model}</code>, <code>ollama pull {form.ollama_chat_model}</code>
        {vision && <> and <code>ollama pull {vision.id}</code></>}
      </p>
    </div>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedDocs, setSelectedDocs] = useState(new Set());
  
  const [messages, setMessages] = useState([
    {
      id: 'msg-0',
      role: 'assistant',
      content: "Hello! I am DocuMind.\n\nI am your intelligent AI document assistant. To get started, simply upload your documents and ask me questions about their content. I will analyze the text and provide answers based solely on the information within your files."
    }
  ]);
  
  const [input, setInput] = useState('');
  
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadEta, setUploadEta] = useState('');
  
  const [isTyping, setIsTyping] = useState(false);
  
  const [editingMsgId, setEditingMsgId] = useState(null);
  const [editContent, setEditContent] = useState('');
  
  const [showClearModal, setShowClearModal] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState(null);
  const [showSupportedFiles, setShowSupportedFiles] = useState(false);
  
  const [toast, setToast] = useState(null);
  const [limits, setLimits] = useState(DEFAULT_LIMITS);
  const [appStatus, setAppStatus] = useState(null);
  const [showSettings, setShowSettings] = useState(false);
  const [settingsForm, setSettingsForm] = useState(null);
  const [settingsTest, setSettingsTest] = useState(null);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [backendError, setBackendError] = useState(null);
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false);
  const [engineReady, setEngineReady] = useState(!isDesktop);
  const [launchText, setLaunchText] = useState('Starting up…');
  const [showScrollFade, setShowScrollFade] = useState(false);

  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      const savedTheme = localStorage.getItem('theme');
      if (savedTheme) return savedTheme === 'dark';
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }
    return false;
  });

  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const docListRef = useRef(null);

  useEffect(() => {
    const root = window.document.documentElement;
    if (isDarkMode) {
      root.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      root.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [isDarkMode]);

  const toggleTheme = () => setIsDarkMode(prev => !prev);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 150)}px`;
    }
  }, [input]);

  const checkScroll = () => {
    if (docListRef.current) {
      const { scrollTop, scrollHeight, clientHeight } = docListRef.current;
      const isAtBottom = Math.ceil(scrollTop + clientHeight) >= scrollHeight;
      setShowScrollFade(!isAtBottom && documents.length > 3);
    }
  };

  useEffect(() => {
    checkScroll();
  }, [documents]);

  useEffect(() => {
    if (!isUploading) {
      setUploadProgress(0);
      setUploadEta('');
    }
  }, [isUploading]);

  // On start: read limits and status, and (desktop) restore the last documents and chat.
  useEffect(() => {
    (async () => {
      const shownAt = Date.now();
      try {
        await getBackend();          // desktop: waits until the local engine answers
        setLaunchText('Loading your documents and chats…');
        const lim = await apiJson('/limits');
        if (lim.ok) setLimits({ ...DEFAULT_LIMITS, ...lim.data });
        const st = await apiJson('/status');
        if (st.ok) {
          setAppStatus(st.data);
          if (st.data.desktop && st.data.provider === 'nvidia' && !st.data.nvidia_key_set) openSettings();
        }
        const ws = await apiJson('/workspace');
        if (ws.ok && ws.data.session_id) {
          setSessionId(ws.data.session_id);
          const docs = (ws.data.documents || []).map(d => ({ id: d.doc_id, name: d.filename, chunks: d.chunks_count }));
          setDocuments(docs);
          setSelectedDocs(new Set(docs.map(d => d.id)));
          const restored = (ws.data.messages || []).map(m => ({ id: `db-${m.id}`, role: m.role, content: m.content }));
          if (restored.length) setMessages(prev => [prev[0], ...restored]);
        }
        // Keep the logo up for a moment at least, so a fast start doesn't flash.
        const wait = 700 - (Date.now() - shownAt);
        if (wait > 0) await new Promise(r => setTimeout(r, wait));
        setEngineReady(true);
      } catch (err) {
        if (isDesktop) setBackendError(String(err?.message || err));
      }
    })();
  }, []);

  const openSettings = async () => {
    setSettingsTest(null);
    setConfirmDeleteAll(false);
    const { ok, data } = await apiJson('/settings');
    if (ok) setSettingsForm({ ...data, nvidia_api_key: '' });
    setShowSettings(true);
  };

  const settingsPayload = () => {
    const body = { ...settingsForm };
    ['nvidia_key_set', 'nvidia_key_hint', 'ram_gb', 'low_memory', 'recommended'].forEach(k => delete body[k]);
    if (!body.nvidia_api_key) delete body.nvidia_api_key; // empty field = keep the saved key
    return body;
  };

  const testSettings = async () => {
    setSettingsBusy(true);
    setSettingsTest(null);
    const { data } = await apiJson('/settings/test', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settingsPayload()),
    });
    setSettingsTest(data);
    setSettingsBusy(false);
  };

  const saveSettings = async () => {
    setSettingsBusy(true);
    const { ok, data } = await apiJson('/settings', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(settingsPayload()),
    });
    setSettingsBusy(false);
    if (ok) {
      setAppStatus(prev => ({ ...(prev || {}), provider: data.provider, nvidia_key_set: data.nvidia_key_set }));
      setShowSettings(false);
      showToast(data.provider === 'ollama' ? 'Using local AI (Ollama).' : 'Using NVIDIA cloud AI.', 'success');
    } else {
      showToast(`Could not save settings: ${data.detail || 'unknown error'}`, 'error');
    }
  };

  const deleteAllData = async () => {
    setSettingsBusy(true);
    const { ok, data } = await apiJson('/data', { method: 'DELETE' });
    setSettingsBusy(false);
    setConfirmDeleteAll(false);
    if (!ok) {
      showToast(data.detail || 'Could not delete your data.', 'error');
      return;
    }
    setSessionId(null);
    setDocuments([]);
    setSelectedDocs(new Set());
    setMessages(prev => [prev[0]]);
    setShowSettings(false);
    const st = await apiJson('/status');
    if (st.ok) setAppStatus(st.data);
    showToast('All DocuMind data was deleted from this computer.', 'success');
  };

  // Poll background indexing jobs until all are finished, updating the progress ring.
  const pollJobs = async (sid, jobIds) => {
    const params = new URLSearchParams({ session_id: sid });
    jobIds.forEach(id => params.append('job_ids', id));
    for (;;) {
      const { ok, data } = await apiJson(`/jobs?${params}`);
      if (ok) {
        const jobs = data.jobs || [];
        if (!jobs.length) return [];
        const pct = (jobs.reduce((sum, j) => sum + jobFraction(j), 0) / jobs.length) * 100;
        setUploadProgress(Math.min(99, pct));
        const active = jobs.find(j => j.status === 'running');
        setUploadEta(active ? `${active.label}${jobs.length > 1 ? ` · ${active.filename}` : ''}` : '');
        if (jobs.every(j => j.status !== 'running')) return jobs;
      }
      await new Promise(r => setTimeout(r, 500));
    }
  };

  const showToast = (message, type = 'success') => {
    setToast({ message, type });
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const existingNames = new Set(documents.map(d => d.name));
    const duplicates = files.filter(f => existingNames.has(f.name));
    const uniqueFiles = files.filter(f => !existingNames.has(f.name));

    if (duplicates.length > 0) {
      showToast("File(s) already indexed!", 'error');
    }

    if (uniqueFiles.length === 0) {
      if (fileInputRef.current) fileInputRef.current.value = null;
      return;
    }

    // Check size limits before uploading, so users get an instant, specific message.
    // Skipped entirely when the backend reports limits are off (local setup).
    const resetPicker = () => { if (fileInputRef.current) fileInputRef.current.value = null; };
    const maxBytesFor = (f) => (IMAGE_EXT_RE.test(f.name) ? limits.max_image_mb : limits.max_file_mb) * MB;
    const tooBig = limits.enforced ? uniqueFiles.filter(f => f.size > maxBytesFor(f)) : [];
    const filesToUpload = limits.enforced ? uniqueFiles.filter(f => f.size <= maxBytesFor(f)) : uniqueFiles;

    if (tooBig.length > 0) {
      const names = tooBig.map(f => `${f.name} (${fmtMB(f.size)})`).join(', ');
      const limitText = limits.max_file_mb === limits.max_image_mb
        ? `${limits.max_file_mb} MB per file`
        : `${limits.max_file_mb} MB per file, ${limits.max_image_mb} MB per image`;
      showToast(`Too large: ${names}. The limit is ${limitText}.`, 'error');
    }
    if (limits.enforced && filesToUpload.length > limits.max_files) {
      showToast(`You selected ${filesToUpload.length} files. You can upload up to ${limits.max_files} at a time.`, 'error');
      resetPicker();
      return;
    }
    const totalBytes = filesToUpload.reduce((sum, f) => sum + f.size, 0);
    if (limits.enforced && totalBytes > limits.max_total_mb * MB) {
      showToast(`These files total ${fmtMB(totalBytes)}. The limit is ${limits.max_total_mb} MB per upload.`, 'error');
      resetPicker();
      return;
    }
    if (filesToUpload.length === 0) {
      resetPicker();
      return;
    }

    setIsUploading(true);
    
    const formData = new FormData();
    filesToUpload.forEach(file => formData.append('files', file));
    if (sessionId) formData.append('session_id', sessionId);

    try {
      const { ok, data } = await apiJson('/upload', { method: 'POST', body: formData });
      if (!ok) {
        if (data.detail && data.detail.includes("No readable text found in image(s)")) {
          showToast("Upload failed: failed to process file(s). No readable text found in image(s)", 'error');
        } else {
          showToast(`Upload failed: ${data.detail}`, 'error');
        }
        setIsUploading(false);
        return;
      }
      if (!sessionId) setSessionId(data.session_id);

      const failed = [...(data.failed || [])];
      const finished = await pollJobs(data.session_id, (data.jobs || []).map(j => j.job_id));
      const newDocs = finished
        .filter(j => j.status === 'done' && j.doc)
        .map(j => ({ id: j.doc.doc_id, name: j.doc.filename, chunks: j.doc.chunks_count }));
      finished.filter(j => j.status === 'failed').forEach(j => failed.push({ filename: j.filename, error: j.error }));
      setUploadProgress(100);

      setDocuments(prev => [...prev, ...newDocs]);
      setSelectedDocs(prev => {
        const next = new Set(prev);
        newDocs.forEach(doc => next.add(doc.id));
        return next;
      });
      failed.forEach(f => {
        setMessages(prev => [...prev, {
          id: `msg-${Date.now()}-${Math.random()}`,
          role: 'assistant',
          content: `I couldn't analyze "${f.filename}". ${f.error}`
        }]);
      });
      if (newDocs.length > 0) {
        showToast(`Successfully indexed ${newDocs.length} document(s).`, 'success');
      } else if (failed.length > 0) {
        showToast(`Couldn't index ${failed.length === 1 ? `"${failed[0].filename}"` : `${failed.length} files`}.`, 'error');
      }
      setTimeout(() => setIsUploading(false), 400);
    } catch (err) {
      showToast("Error uploading files. Please try again.", 'error');
      setIsUploading(false);
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = null;
    }
  };

  const triggerFilePicker = () => fileInputRef.current?.click();

  const toggleDoc = (docId) => {
    setSelectedDocs(prev => {
      const next = new Set(prev);
      if (next.has(docId)) next.delete(docId);
      else next.add(docId);
      return next;
    });
  };

  const selectAll = () => setSelectedDocs(new Set(documents.map(d => d.id)));
  const selectNone = () => setSelectedDocs(new Set());

  const confirmDeleteDoc = async () => {
    if (!sessionId || !deleteConfirmId) return;
    try {
      await api(`/session/${sessionId}/document/${deleteConfirmId}`, { method: 'DELETE' });
      setDocuments(prev => prev.filter(d => d.id !== deleteConfirmId));
      setSelectedDocs(prev => {
        const next = new Set(prev);
        next.delete(deleteConfirmId);
        return next;
      });
    } catch (err) {
      showToast("Failed to delete document.", 'error');
    } finally {
      setDeleteConfirmId(null);
    }
  };

  const executeQuery = async (text, msgId) => {
    setIsTyping(true);

    // Every chat gets a session id, so its history can be saved and restored (desktop app).
    let sid = sessionId;
    if (!sid) {
      sid = newSessionId();
      setSessionId(sid);
    }
    const formData = new FormData();
    formData.append('session_id', sid);
    formData.append('question', text);
    selectedDocs.forEach(id => formData.append('doc_ids', id));

    try {
      const res = await api('/query', { method: 'POST', body: formData });
      const data = await res.json();

      if (res.ok) {
        setMessages(prev => [...prev, {
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: data.answer
        }]);
      } else {
        showToast(`Query failed: ${data.detail}`, 'error');
      }
    } catch (err) {
      showToast("Error getting answer.", 'error');
    } finally {
      setIsTyping(false);
    }
  };

  const handleSendMessage = (text) => {
    if (!text.trim()) return;

    const msgId = `msg-${Date.now()}`;
    const userMsg = { 
      id: msgId,
      role: 'user', 
      content: text
    };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    
    executeQuery(text, msgId);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(input);
    }
  };

  const handleCopy = (id, text) => {
    const plainText = text.replace(/<[^>]+>/g, '');
    navigator.clipboard.writeText(plainText);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const startEditing = (id, content) => {
    setEditingMsgId(id);
    setEditContent(content);
  };

  const saveEdit = (id) => {
    if (!editContent.trim()) return;
    setMessages(prev => 
      prev.map(msg => msg.id === id ? { ...msg, content: editContent } : msg)
    );
    setEditingMsgId(null);
    setEditContent('');
    executeQuery(editContent, id);
  };

  const cancelEdit = () => {
    setEditingMsgId(null);
    setEditContent('');
  };

  const confirmClearSession = () => {
    // Remove this session's documents and chat history from storage too, not just from the screen.
    if (sessionId) api(`/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    setSessionId(null);
    setDocuments([]);
    setSelectedDocs(new Set());
    setMessages([messages[0]]); 
    setShowClearModal(false);
  };

  const suggestedQuestions = [
    "What can you do?",
    "Show me example use cases",
    "What file types can you analyze?",
  ];

  const radius = 20;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (uploadProgress / 100) * circumference;

  return (
    <div className="flex flex-col h-screen w-full font-sans overflow-hidden bg-gray-50 dark:bg-gray-950 transition-colors duration-200">
      
      {toast && <Toast message={toast.message} type={toast.type} onClose={() => setToast(null)} />}

      {deleteConfirmId && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm px-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4 text-red-600 dark:text-red-400">
              <AlertTriangle className="w-6 h-6" />
              <h3 className="text-lg font-semibold">Delete Document?</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-6">
              Are you sure you want to remove this document from the index? This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button 
                onClick={() => setDeleteConfirmId(null)}
                className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors focus:outline-none"
              >
                Cancel
              </button>
              <button 
                onClick={confirmDeleteDoc}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 transition-colors focus:outline-none"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {showClearModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm px-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 max-w-sm w-full shadow-2xl border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4 text-red-600 dark:text-red-400">
              <AlertTriangle className="w-6 h-6" />
              <h3 className="text-lg font-semibold">Clear Chat?</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-6">
              Are you sure you want to clear the current session? This will delete all messages and remove your uploaded documents from the current workspace.
            </p>
            <div className="flex justify-end gap-3">
              <button 
                onClick={() => setShowClearModal(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors focus:outline-none"
              >
                Cancel
              </button>
              <button 
                onClick={confirmClearSession}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 transition-colors focus:outline-none"
              >
                Yes, Clear
              </button>
            </div>
          </div>
        </div>
      )}

      {!engineReady && !backendError && (
        <div data-tauri-drag-region className="fixed inset-0 z-[120] flex flex-col items-center justify-center gap-5 bg-gray-50 dark:bg-gray-950" role="status" aria-live="polite">
          <Logo className="w-20 h-20 drop-shadow-xl logo-breathe" />
          <div className="text-center">
            <p className="text-lg font-semibold text-gray-900 dark:text-white tracking-tight">DocuMind</p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{launchText}</p>
          </div>
        </div>
      )}

      {backendError && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 backdrop-blur-sm px-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 max-w-md w-full shadow-2xl border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4">
              <Logo className="w-10 h-10 shrink-0" />
              <h3 className="text-lg font-semibold text-red-600 dark:text-red-400">DocuMind couldn't start</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">{String(backendError).split('\n\nDetails:\n')[0]} Please restart DocuMind.</p>
            {String(backendError).includes('\n\nDetails:\n') && (
              <pre className="text-xs text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-900 rounded-lg p-3 max-h-48 overflow-auto whitespace-pre-wrap break-words select-text">
                {String(backendError).split('\n\nDetails:\n')[1]}
              </pre>
            )}
          </div>
        </div>
      )}

      {showSettings && settingsForm && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm px-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 max-w-lg w-full shadow-2xl border border-gray-200 dark:border-gray-700 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center gap-3 mb-4 text-blue-600 dark:text-blue-400">
              <Settings className="w-6 h-6" />
              <h3 className="text-lg font-semibold">AI settings</h3>
            </div>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Your documents, chats and settings are stored only on this computer{appStatus?.data_dir ? ` (${appStatus.data_dir})` : ''}.
            </p>

            <div className="grid grid-cols-2 gap-3 mb-5">
              {[
                { id: 'nvidia', icon: Cloud, title: 'NVIDIA cloud', text: 'Fast, best answers. Needs internet and your API key. Document text is sent to NVIDIA.' },
                { id: 'ollama', icon: HardDrive, title: 'Local AI (Ollama)', text: 'Private and offline. Needs the Ollama app and downloaded models.' },
              ].map(opt => (
                <button
                  key={opt.id}
                  onClick={() => { setSettingsForm(f => ({ ...f, provider: opt.id })); setSettingsTest(null); }}
                  className={`text-left p-3 rounded-xl border-2 transition-colors focus:outline-none ${settingsForm.provider === opt.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'}`}
                >
                  <div className="flex items-center gap-2 font-medium text-sm text-gray-900 dark:text-gray-100"><opt.icon className="w-4 h-4" />{opt.title}</div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{opt.text}</p>
                </button>
              ))}
            </div>

            {settingsForm.provider === 'nvidia' ? (
              <label className="block text-sm text-gray-700 dark:text-gray-300 mb-4">
                NVIDIA API key
                <input
                  type="password"
                  value={settingsForm.nvidia_api_key}
                  onChange={e => setSettingsForm(f => ({ ...f, nvidia_api_key: e.target.value }))}
                  placeholder={settingsForm.nvidia_key_set ? `Saved key ${settingsForm.nvidia_key_hint} (leave empty to keep)` : 'nvapi-...'}
                  className="mt-1 w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 text-sm focus:outline-none focus:border-blue-500"
                />
                <span className="block text-xs text-gray-500 mt-1">Get a free key at build.nvidia.com.</span>
              </label>
            ) : (
              <OllamaSettings form={settingsForm} setForm={setSettingsForm} />
            )}

            {appStatus && !appStatus.ocr && (
              <p className="text-xs text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 rounded-lg p-2 mb-4">
                Tesseract OCR isn't installed, so scanned PDFs and images can't be read. Mac: <code>brew install tesseract</code>
              </p>
            )}

            {settingsTest && (
              <p className={`text-xs rounded-lg p-2 mb-4 ${settingsTest.ok ? 'text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-900/20' : 'text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-900/20'}`}>
                {settingsTest.ok ? `Connected: ${settingsTest.embed_model} and ${settingsTest.chat_model} are ready.` : settingsTest.message}
                {settingsTest.ok && settingsTest.vision_model && settingsTest.vision_ready && ` Image model ${settingsTest.vision_model} is ready.`}
              </p>
            )}
            {settingsTest && settingsTest.ok && settingsTest.vision_model && settingsTest.vision_ready === false && (
              <p className="text-xs rounded-lg p-2 -mt-2 mb-4 text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-900/20">
                Image model not installed yet, so photos without text can't be analyzed. Run: <code>ollama pull {settingsTest.vision_model}</code>
              </p>
            )}

            <div className="flex justify-between gap-3">
              <button onClick={testSettings} disabled={settingsBusy} className="px-4 py-2 text-sm font-medium text-blue-600 dark:text-blue-400 bg-blue-50 dark:bg-blue-900/20 rounded-lg hover:bg-blue-100 disabled:opacity-50 focus:outline-none">
                {settingsBusy ? 'Checking…' : 'Test connection'}
              </button>
              <div className="flex gap-3">
                <button onClick={() => setShowSettings(false)} className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 focus:outline-none">
                  Cancel
                </button>
                <button onClick={saveSettings} disabled={settingsBusy} className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 disabled:opacity-50 focus:outline-none">
                  Save
                </button>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700">
              {!confirmDeleteAll ? (
                <button
                  onClick={() => setConfirmDeleteAll(true)}
                  className="text-sm font-medium text-red-600 dark:text-red-400 hover:underline focus:outline-none"
                >
                  Delete all my data
                </button>
              ) : (
                <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 p-3" role="alert">
                  <p className="text-sm text-red-800 dark:text-red-300 mb-3">
                    This permanently deletes every document, chat and setting DocuMind has stored on this computer,
                    including your NVIDIA API key. Your original files elsewhere on your computer are not touched.
                  </p>
                  <div className="flex justify-end gap-3">
                    <button onClick={() => setConfirmDeleteAll(false)} className="px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 rounded-lg border border-gray-200 dark:border-gray-600 focus:outline-none">
                      Cancel
                    </button>
                    <button onClick={deleteAllData} disabled={settingsBusy} className="px-3 py-1.5 text-sm font-medium text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50 focus:outline-none">
                      Delete everything
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {showSupportedFiles && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm px-4">
          <div className="bg-white dark:bg-gray-800 rounded-2xl p-6 max-w-md w-full shadow-2xl border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-3 mb-4 text-blue-600 dark:text-blue-400">
              <Info className="w-6 h-6" />
              <h3 className="text-lg font-semibold">Supported File Types</h3>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
              DocuMind can extract text from the following file formats:
            </p>
            <ul className="list-disc pl-5 text-sm text-gray-600 dark:text-gray-300 space-y-2 mb-6">
              <li>Documents: PDF, DOCX, TXT, MD, HTML</li>
              <li>Spreadsheets: XLSX, CSV, TSV</li>
              <li>Presentations: PPTX</li>
              <li>Code: PY, JS, TS, JSX, TSX, JAVA, GO, RS, C, CPP, CS, PHP, RB, KT, SWIFT, SQL, SH and more</li>
              <li>Config and logs: JSON, YAML, TOML, XML, INI, LOG</li>
              <li>
                Images: JPG, PNG, BMP, TIFF, WEBP 
                <span className="block text-xs text-gray-500 mt-1">Text in images is read with OCR. Photos without text (products, charts, scenes) need local AI with an image model, chosen in Settings.</span>
              </li>
            </ul>
            {limits.enforced && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-6 -mt-2">
              Limits: up to {limits.max_files} files and {limits.max_total_mb} MB per upload, {limits.max_file_mb} MB per file
              {limits.max_image_mb !== limits.max_file_mb ? ` (${limits.max_image_mb} MB per image)` : ''}.
              Scanned PDFs can have up to {limits.max_ocr_pages} pages that need OCR.
            </p>
            )}
            <div className="flex justify-end">
              <button 
                onClick={() => setShowSupportedFiles(false)}
                className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors focus:outline-none"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}

      <header
        data-tauri-drag-region
        className={`h-10 flex items-center justify-between bg-gray-50 dark:bg-gray-950 shrink-0 transition-colors duration-200 z-10 relative select-none ${
          isDesktop && isMacOS ? 'pl-20 pr-3' : drawsOwnWindowControls ? 'pl-4 pr-1' : 'px-4'
        }`}
      >
        <div data-tauri-drag-region className="flex-1 h-full" />
        <div data-tauri-drag-region className="flex items-center gap-2">
          {documents.length > 0 && (
            <div className="bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-400 px-2.5 py-0.5 rounded-full text-xs font-medium flex items-center gap-1.5 border border-green-200 dark:border-green-800 mr-1">
              <CheckCircle2 className="w-3.5 h-3.5" />
              {documents.length} docs indexed
            </div>
          )}
          
          {isDesktop && (
            <button
              onClick={openSettings}
              className="p-1.5 rounded-lg text-gray-500 dark:text-gray-400 hover:bg-gray-200 hover:text-gray-800 dark:hover:bg-gray-800 dark:hover:text-gray-100 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
              title="Settings"
              aria-label="Settings"
            >
              <Settings className="w-[18px] h-[18px]" />
            </button>
          )}

          <button 
            onClick={toggleTheme}
            className="p-1.5 rounded-lg text-gray-500 dark:text-gray-400 hover:bg-gray-200 hover:text-gray-800 dark:hover:bg-gray-800 dark:hover:text-gray-100 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
            title={isDarkMode ? 'Light mode' : 'Dark mode'}
            aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDarkMode ? <Sun className="w-[18px] h-[18px]" /> : <Moon className="w-[18px] h-[18px]" />}
          </button>

          {drawsOwnWindowControls && <WindowControls />}
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        
        <aside className="w-80 bg-gray-50 dark:bg-gray-950 flex flex-col px-6 pb-6 pt-2 shrink-0 transition-colors duration-200 relative z-20 outline-none">

          <div className="flex items-center gap-3 mb-7 shrink-0">
            <Logo className="w-10 h-10 drop-shadow-md" />
            <div>
              <h1 className="font-semibold text-gray-900 dark:text-white text-lg leading-tight tracking-tight">DocuMind</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-medium">Intelligent Document Search</p>
            </div>
          </div>
          
          <h2 className="text-xs font-semibold text-gray-500 dark:text-gray-400 tracking-wider mb-4 shrink-0">DOCUMENTS</h2>
          
          <input 
            type="file" 
            multiple
            accept={ACCEPTED_FILE_TYPES} 
            onChange={handleFileUpload} 
            ref={fileInputRef}
            className="hidden" 
            disabled={isUploading}
          />

          <div 
            onClick={isUploading ? undefined : triggerFilePicker}
            className={`relative border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-xl p-6 flex flex-col items-center justify-center text-center transition-colors group bg-white dark:bg-gray-900 shrink-0 ${isUploading ? 'cursor-wait border-blue-400 dark:border-blue-500' : 'cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-900'}`}
          >
            <div className="w-12 h-12 flex items-center justify-center mb-2 pointer-events-none">
              {isUploading ? (
                <div className="relative flex items-center justify-center">
                  <svg className="transform -rotate-90 w-12 h-12">
                    <circle
                      className="text-gray-200 dark:text-gray-700"
                      strokeWidth="4"
                      stroke="currentColor"
                      fill="transparent"
                      r={radius}
                      cx="24"
                      cy="24"
                    />
                    <circle
                      className="text-blue-500 transition-all duration-300 ease-out"
                      strokeWidth="4"
                      strokeDasharray={circumference}
                      strokeDashoffset={strokeDashoffset}
                      strokeLinecap="round"
                      stroke="currentColor"
                      fill="transparent"
                      r={radius}
                      cx="24"
                      cy="24"
                    />
                  </svg>
                  <span className="absolute text-[9px] font-bold text-blue-600 dark:text-blue-400">
                    {Math.round(uploadProgress)}%
                  </span>
                </div>
              ) : (
                <div className="w-10 h-10 bg-gray-50 dark:bg-gray-800 rounded-lg shadow-sm border border-gray-100 dark:border-gray-700 flex items-center justify-center group-hover:scale-105 transition-transform">
                  <FileText className="w-5 h-5 text-gray-400 dark:text-gray-500" />
                </div>
              )}
            </div>
            
            <p className={`text-sm font-medium pointer-events-none ${isUploading ? 'text-blue-600 dark:text-blue-400' : 'text-gray-700 dark:text-gray-300'}`}>
              {isUploading ? 'Indexing files...' : 'Drop files here'}
            </p>
            
            {isUploading ? (
              <p className="text-[11px] font-medium text-gray-500 dark:text-gray-400 mt-1 pointer-events-none">
                {uploadEta}
              </p>
            ) : (
              <>
                <p className="text-xs text-gray-400 dark:text-gray-500 mt-1 pointer-events-none">or click to browse</p>
                <p className="text-[10px] text-gray-400 dark:text-gray-600 mt-2 uppercase tracking-wide pointer-events-none">PDF · XLSX · DOCX · TXT · CSV</p>
              </>
            )}
          </div>

          {documents.length > 0 && (
            <div className="flex items-center justify-between mt-8 mb-3 shrink-0">
              <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 tracking-wider">INDEXED ({documents.length})</h3>
              <div className="flex gap-2 text-xs text-blue-600 dark:text-blue-400 font-medium">
                <button onClick={selectAll} className="hover:underline focus:outline-none">All</button>
                <span className="text-gray-300 dark:text-gray-700">|</span>
                <button onClick={selectNone} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:underline focus:outline-none">None</button>
              </div>
            </div>
          )}

          <div className="flex-1 relative min-h-0">
            {showScrollFade && (
              <div className="absolute bottom-0 left-0 right-0 h-16 bg-gradient-to-t from-gray-50 dark:from-gray-950 to-transparent z-20 pointer-events-none flex justify-center items-end pb-2 transition-opacity duration-300">
                <div className="w-8 h-1.5 bg-gray-400/50 dark:bg-gray-600/50 rounded-full"></div>
              </div>
            )}

            <div 
              ref={docListRef}
              onScroll={checkScroll}
              className="h-full overflow-y-auto pr-1 -mr-1 hide-scrollbar"
            >
              <div className="space-y-2 pb-6">
                {documents.map((doc) => (
                  <div key={doc.id} className="relative group/doc flex items-start gap-3 p-3 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm">
                    <input 
                      type="checkbox" 
                      checked={selectedDocs.has(doc.id)}
                      onChange={() => toggleDoc(doc.id)}
                      className="mt-1 rounded border-gray-300 dark:border-gray-600 text-blue-600 focus:ring-blue-500 cursor-pointer bg-white dark:bg-gray-800 shrink-0" 
                    />
                    <div className="flex-1 min-w-0 relative">
                      
                      <FileTooltip text={doc.name}>
                        <p className="text-sm font-medium text-gray-900 dark:text-gray-200 truncate cursor-default">
                          {doc.name}
                        </p>
                      </FileTooltip>

                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{doc.chunks} chunks</p>
                    </div>
                    
                    <button onClick={() => setDeleteConfirmId(doc.id)} className="opacity-0 group-hover/doc:opacity-100 transition-opacity shrink-0 focus:outline-none">
                      <Trash2 className="w-4 h-4 text-gray-400 hover:text-red-500" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="pt-4 mt-1 border-t border-gray-200 dark:border-gray-800 shrink-0">
            <h3 className="text-xs font-semibold text-gray-500 dark:text-gray-400 tracking-wider mb-4">HOW TO USE</h3>
            <ol className="space-y-4 text-sm text-gray-500 dark:text-gray-400">
              <li className="flex gap-3 items-start">
                <span className="w-5 h-5 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center text-[10px] font-bold text-gray-600 dark:text-gray-300 shrink-0">1</span>
                <span>
                  Upload one or more <button onClick={() => setShowSupportedFiles(true)} className="text-blue-600 dark:text-blue-400 font-medium hover:underline focus:outline-none">supported files</button>
                </span>
              </li>
              <li className="flex gap-3 items-start">
                <span className="w-5 h-5 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center text-[10px] font-bold text-gray-600 dark:text-gray-300 shrink-0">2</span>
                <span>Select docs to search using checkboxes</span>
              </li>
              <li className="flex gap-3 items-start">
                <span className="w-5 h-5 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center text-[10px] font-bold text-gray-600 dark:text-gray-300 shrink-0">3</span>
                <span>Ask anything about docs</span>
              </li>
              <li className="flex gap-3 items-start">
                <span className="w-5 h-5 bg-gray-200 dark:bg-gray-800 rounded-full flex items-center justify-center text-[10px] font-bold text-gray-600 dark:text-gray-300 shrink-0">4</span>
                <span>Get LLM-generated responses</span>
              </li>
            </ol>
          </div>
        </aside>

        <main className="flex-1 flex flex-col bg-white dark:bg-gray-900 min-w-0 transition-colors duration-200 z-10 relative overflow-hidden rounded-tl-2xl border-t border-l border-gray-200 dark:border-gray-800">
          
          <div className="h-16 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between px-6 shrink-0 bg-white dark:bg-gray-900">
            <div className="flex items-center gap-3">
              <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-100">Chat</h2>
              {selectedDocs.size > 0 ? (
                <div className="flex items-center gap-1.5 px-2.5 py-1 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 rounded-md text-xs font-medium border border-green-200 dark:border-green-800/50">
                  <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
                  {selectedDocs.size} Document{selectedDocs.size !== 1 ? 's' : ''} Active
                </div>
              ) : (
                <div className="flex items-center gap-1.5 px-2.5 py-1 bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 rounded-md text-xs font-medium border border-gray-200 dark:border-gray-700">
                  <div className="w-1.5 h-1.5 bg-gray-400 rounded-full"></div>
                  General Mode
                </div>
              )}
            </div>

            <div className="flex items-center gap-3">
              <button 
                onClick={() => setShowClearModal(true)} 
                className="flex items-center gap-2 text-sm font-medium text-gray-500 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 px-3 py-1.5 rounded-lg transition-colors focus:outline-none"
              >
                <Trash2 className="w-4 h-4" /> Clear
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-6 bg-gray-50/30 dark:bg-gray-900/50">
            <div className="max-w-4xl mx-auto w-full space-y-6">
              {messages.map((msg, idx) => (
                <div key={msg.id || idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                  
                  {msg.role === 'user' ? (
                    <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-blue-600 text-white">
                      <User className="w-4 h-4" />
                    </div>
                  ) : (
                    <Logo className="w-8 h-8 shrink-0" />
                  )}

                  <div className={`max-w-[80%] flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                    
                    {editingMsgId === msg.id ? (
                      <div className="flex flex-col gap-2 w-full min-w-[300px]">
                        <textarea
                          value={editContent}
                          onChange={(e) => setEditContent(e.target.value)}
                          className="w-full p-3 bg-white dark:bg-gray-800 border-2 border-blue-500 rounded-2xl text-sm focus:outline-none text-gray-900 dark:text-gray-100 resize-none"
                          rows={3}
                        />
                        <div className="flex gap-2 justify-end">
                          <button onClick={cancelEdit} className="px-3 py-1.5 text-xs font-medium text-gray-600 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 focus:outline-none">
                            Cancel
                          </button>
                          <button onClick={() => saveEdit(msg.id)} className="px-3 py-1.5 text-xs font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 focus:outline-none">
                            Save
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div 
                        className={`px-5 py-3.5 rounded-2xl text-sm leading-relaxed ${
                          msg.role === 'user' 
                            ? 'bg-blue-600 text-white rounded-tr-sm whitespace-pre-wrap' 
                            : 'chat-answer bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-800 dark:text-gray-200 rounded-tl-sm shadow-sm [&>h3]:text-xl [&>h3]:font-bold [&>h3]:mt-2 [&>h3]:mb-2 [&>h4]:text-lg [&>h4]:font-semibold [&>h4]:mt-3 [&>h4]:mb-1'
                        }`}
                        dangerouslySetInnerHTML={{ __html: msg.content.replace(/\n/g, '<br/>') }}
                      />
                    )}
                    
                    {editingMsgId !== msg.id && msg.role === 'user' && (
                      <div className="flex items-center gap-3 mt-2 px-1">
                        <div className="relative">
                          <button onClick={() => handleCopy(msg.id, msg.content)} className="text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors focus:outline-none" title="Copy">
                            <Copy className="w-3.5 h-3.5" />
                          </button>
                          {copiedId === msg.id && (
                            <span className="absolute top-full mt-1 left-1/2 -translate-x-1/2 bg-gray-800 text-white text-[10px] px-2 py-0.5 rounded shadow whitespace-nowrap z-10">
                              Copied
                            </span>
                          )}
                        </div>
                        <button onClick={() => startEditing(msg.id, msg.content)} className="text-gray-400 hover:text-blue-500 dark:hover:text-blue-400 transition-colors focus:outline-none" title="Edit">
                          <Edit2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    )}

                  </div>
                </div>
              ))}
              
              {isTyping && (
                <div className="flex gap-4">
                  <Logo className="w-8 h-8 shrink-0" />
                  <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 px-5 py-4 rounded-2xl rounded-tl-sm shadow-sm flex gap-1 items-center">
                    <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce [animation-delay:0.2s]"></div>
                    <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-bounce [animation-delay:0.4s]"></div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="p-4 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shrink-0">
            <div className="max-w-4xl mx-auto w-full">
              <div className="flex items-center gap-2 mb-3 overflow-x-auto pb-1">
                <span className="text-xs text-gray-400 dark:text-gray-500 shrink-0">Try asking:</span>
                {suggestedQuestions.map((q, i) => (
                  <button 
                    key={i} 
                    onClick={() => handleSendMessage(q)}
                    className="whitespace-nowrap px-3 py-1.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-full text-xs text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors focus:outline-none"
                  >
                    {q}
                  </button>
                ))}
              </div>

              <div className="relative flex items-end">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Ask a question about your documents or how to use DocuMind..."
                  disabled={isTyping}
                  rows={1}
                  className="w-full pl-5 pr-14 py-4 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 rounded-2xl text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 disabled:opacity-50 transition-all resize-none overflow-y-auto max-h-[150px] hide-scrollbar"
                />
                <button 
                  onClick={() => handleSendMessage(input)}
                  disabled={!input.trim() || isTyping}
                  className="absolute right-2 bottom-2 p-2.5 bg-white dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-xl hover:bg-gray-50 dark:hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus:outline-none"
                >
                  <Send className="w-4 h-4 text-gray-600 dark:text-gray-300" />
                </button>
              </div>
              <p className="text-center text-[10px] text-gray-400 dark:text-gray-500 mt-2">
                Enter to send · Shift+Enter for new line
              </p>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}