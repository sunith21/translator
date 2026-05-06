import React, { useState, useEffect, useRef, useCallback } from 'react';
import axios from 'axios';
import { LayoutDashboard, UserPlus, Search, Mic, Square, FileText, LogOut, Activity, Clock, User, Send, Keyboard } from 'lucide-react';
import './App.css';

const API_BASE = 'http://localhost:8000';

// Convert any browser audio blob (webm/ogg/opus) → 16-bit PCM WAV at 16 kHz mono
// This avoids the ffmpeg dependency on the backend entirely.
async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
  await audioCtx.close();

  // Mix down to mono
  const numChannels = audioBuffer.numberOfChannels;
  const numSamples  = audioBuffer.length;
  const pcm = new Float32Array(numSamples);
  for (let c = 0; c < numChannels; c++) {
    const channelData = audioBuffer.getChannelData(c);
    for (let i = 0; i < numSamples; i++) pcm[i] += channelData[i] / numChannels;
  }

  // Encode as 16-bit PCM WAV
  const wavBuffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(wavBuffer);
  const writeStr = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };

  writeStr(0, 'RIFF');
  view.setUint32(4,  36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16,       true);  // chunk size
  view.setUint16(20, 1,        true);  // PCM
  view.setUint16(22, 1,        true);  // mono
  view.setUint32(24, 16000,    true);  // sample rate
  view.setUint32(28, 16000*2,  true);  // byte rate
  view.setUint16(32, 2,        true);  // block align
  view.setUint16(34, 16,       true);  // bits per sample
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    view.setInt16(44 + i * 2, s * 32767, true);
  }

  return new Blob([wavBuffer], { type: 'audio/wav' });
}

// Encode raw Float32Array (16kHz mono) → WAV Blob — used by live VAD pipeline
function encodeWAVFromFloat32(samples, sampleRate = 16000) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true);
  w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s * 32767, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}

function App() {
  const [view, setView] = useState('dashboard');
  const [currentPatient, setCurrentPatient] = useState(null);

  const navigateTo = (newView, patient = null) => {
    setView(newView);
    if (patient) setCurrentPatient(patient);
  };

  return (
    <div className="app-container">
      <nav className="sidebar">
        <div className="sidebar-logo">
          <span className="logo-icon">🏥</span>
          <span className="logo-text">CLINIC AI</span>
        </div>
        <div className="nav-section">
          <p className="nav-label">NAVIGATION</p>
          <button className={`nav-btn ${view === 'dashboard' ? 'active' : ''}`} onClick={() => navigateTo('dashboard')}>
            <LayoutDashboard size={18} /> Dashboard
          </button>
          <button className={`nav-btn ${view === 'new-patient' ? 'active' : ''}`} onClick={() => navigateTo('new-patient')}>
            <UserPlus size={18} /> New Consultation
          </button>
          <button className={`nav-btn ${view === 'search' ? 'active' : ''}`} onClick={() => navigateTo('search')}>
            <Search size={18} /> Search Patient
          </button>
        </div>
        <div className="sidebar-footer">
          <Activity size={14} /> System Online
        </div>
      </nav>

      <main className="main-content">
        {view === 'dashboard' && <Dashboard onOpenPatient={(p) => navigateTo('consultation', p)} />}
        {view === 'new-patient' && <NewPatient onStart={(p) => navigateTo('consultation', p)} />}
        {view === 'search' && <SearchPatient onOpenPatient={(p) => navigateTo('consultation', p)} />}
        {view === 'consultation' && <Consultation key={currentPatient.id} patient={currentPatient} onEnd={() => navigateTo('dashboard')} />}
      </main>
    </div>
  );
}

function Dashboard({ onOpenPatient }) {
  const [recent, setRecent] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get(`${API_BASE}/patients/recent`)
      .then(res => setRecent(res.data))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="page animate-in">
      <div className="page-header">
        <h1>Dashboard</h1>
        <p className="page-subtitle">Welcome back to Clinic AI</p>
      </div>

      <div className="card">
        <div className="card-header">
          <Clock size={18} className="accent-icon" />
          <h3>Recent Patients</h3>
        </div>
        {loading ? (
          <div className="loading-row">Loading...</div>
        ) : recent.length === 0 ? (
          <p className="empty-state">No recent records found. Start a new consultation.</p>
        ) : (
          <div className="patient-list">
            {recent.map(p => (
              <div key={p.id} className="patient-row">
                <div className="patient-info">
                  <User size={16} />
                  <span className="patient-name">{p.name}</span>
                  <span className="patient-id">ID: {p.id}</span>
                </div>
                <button className="btn-primary btn-sm" onClick={() => onOpenPatient(p)}>Open →</button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function NewPatient({ onStart }) {
  const [formData, setFormData] = useState({ name: '', id: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.name || !formData.id) { setError('Both fields are required.'); return; }
    setLoading(true);
    setError('');
    try {
      await axios.post(`${API_BASE}/patients`, formData);
      onStart(formData);
    } catch (err) {
      setError('Failed to create patient. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page animate-in">
      <div className="page-header">
        <h1>New Consultation</h1>
        <p className="page-subtitle">Register a patient to begin</p>
      </div>
      <div className="card form-card">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Patient Name</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. Ramesh Kumar"
              value={formData.name}
              onChange={e => setFormData({...formData, name: e.target.value})}
            />
          </div>
          <div className="form-group">
            <label>Patient ID</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. P001"
              value={formData.id}
              onChange={e => setFormData({...formData, id: e.target.value})}
            />
          </div>
          {error && <p className="error-msg">{error}</p>}
          <button type="submit" className="btn-primary btn-full" disabled={loading}>
            {loading ? 'Starting…' : 'Start Session →'}
          </button>
        </form>
      </div>
    </div>
  );
}

function SearchPatient({ onOpenPatient }) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);

  useEffect(() => {
    if (query.trim()) {
      axios.get(`${API_BASE}/patients/search?q=${query}`).then(res => setResults(res.data));
    } else {
      setResults([]);
    }
  }, [query]);

  return (
    <div className="page animate-in">
      <div className="page-header">
        <h1>Search Patients</h1>
        <p className="page-subtitle">Find existing patients by name or ID</p>
      </div>
      <div className="search-bar-wrap">
        <Search size={18} className="search-icon" />
        <input
          type="text"
          className="search-input"
          placeholder="Enter name or patient ID…"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
      </div>
      {results.length > 0 && (
        <div className="card">
          <div className="patient-list">
            {results.map(r => (
              <div key={r.id} className="patient-row">
                <div className="patient-info">
                  <User size={16} />
                  <span className="patient-name">{r.name}</span>
                  <span className="patient-id">ID: {r.id}</span>
                </div>
                <button className="btn-primary btn-sm" onClick={() => onOpenPatient(r)}>Open →</button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function Consultation({ patient, onEnd }) {
  const [messages, setMessages]           = useState([]);
  const [isRecording, setIsRecording]     = useState(false);
  const [isProcessing, setIsProcessing]   = useState(false);
  const [currentRole, setCurrentRole]     = useState(null);
  const [langKey, setLangKey]             = useState('Hindi (हिन्दी)');
  const [status, setStatus]               = useState('Ready');
  const [patientText, setPatientText]     = useState('');
  const [isTranslatingText, setIsTranslatingText] = useState(false);

  // Live / VAD state
  const [isLive, setIsLive]       = useState(false);
  const [liveRole, setLiveRole]   = useState('doctor');
  const [vadState, setVadState]   = useState('idle'); // idle | listening | speaking | processing
  const [audioLevel, setAudioLevel] = useState(0);
  const [queueLen, setQueueLen]   = useState(0);

  // Refs
  const mediaRecorder   = useRef(null);
  const audioChunks     = useRef([]);
  const langKeyRef      = useRef(langKey);
  const chatEndRef      = useRef(null);
  const liveRoleRef     = useRef(liveRole);
  const liveCtxRef      = useRef(null);
  const liveProcessorRef= useRef(null);
  const liveStreamRef   = useRef(null);
  const vadQueue        = useRef([]);
  const vadProcessing   = useRef(false);

  useEffect(() => { langKeyRef.current = langKey; }, [langKey]);
  useEffect(() => { liveRoleRef.current = liveRole; }, [liveRole]);

  useEffect(() => {
    axios.get(`${API_BASE}/patients/${patient.id}/transcripts`)
      .then(res => setMessages(res.data))
      .catch(() => setStatus('Could not load transcripts'));
  }, [patient.id]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Cleanup live session on unmount
  useEffect(() => () => stopLive(), []);

  // ── Process queued VAD segments one-by-one ──
  const drainQueue = async (pid) => {
    if (vadProcessing.current || vadQueue.current.length === 0) return;
    vadProcessing.current = true;
    const { samples, role, lk } = vadQueue.current.shift();
    setQueueLen(vadQueue.current.length);
    setVadState('processing');
    setStatus(`Processing ${role} speech…`);
    try {
      const wav = encodeWAVFromFloat32(samples);
      const fd  = new FormData();
      fd.append('file', wav, 'recording.wav');
      fd.append('role', role);
      fd.append('lang_key', lk);
      fd.append('patient_id', pid);
      const res   = await axios.post(`${API_BASE}/stt`, fd);
      const entry = { role, original: res.data.original, translated: res.data.translated };
      await axios.post(`${API_BASE}/patients/${pid}/append_transcript`, entry);
      setMessages(prev => [...prev, { ...entry, timestamp: new Date().toLocaleTimeString() }]);
      setStatus('Done ✓');
    } catch (err) {
      setStatus(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      vadProcessing.current = false;
      if (vadQueue.current.length > 0) drainQueue(pid);
      else setVadState('listening');
    }
  };

  // ── Start live VAD session ──
  const startLive = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      liveStreamRef.current = stream;
      const RATE = 16000;
      const ctx  = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RATE });
      liveCtxRef.current = ctx;
      const src  = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      liveProcessorRef.current = proc;

      const THRESH     = 0.012;   // RMS speech threshold
      const SILENCE_MS = 1500;    // silence duration before segment commit
      const MIN_MS     = 350;     // minimum speech duration

      let speaking     = false;
      let silenceStart = null;
      let speechBuf    = [];
      let speechStartMs= 0;
      let levelThrottle= 0;

      proc.onaudioprocess = (e) => {
        const data = e.inputBuffer.getChannelData(0);
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
        const rms = Math.sqrt(sum / data.length);
        const now = Date.now();

        // Throttle level updates to ~10 fps
        if (now - levelThrottle > 100) {
          setAudioLevel(Math.min(1, rms / THRESH));
          levelThrottle = now;
        }

        if (rms > THRESH) {
          if (!speaking) {
            speaking = true; silenceStart = null;
            speechBuf = []; speechStartMs = now;
            setVadState('speaking');
          }
          speechBuf.push(Float32Array.from(data));
          silenceStart = null;
        } else if (speaking) {
          speechBuf.push(Float32Array.from(data));
          if (!silenceStart) { silenceStart = now; }
          else if (now - silenceStart > SILENCE_MS) {
            speaking = false;
            if (now - speechStartMs > MIN_MS) {
              const total = speechBuf.reduce((a, b) => a + b.length, 0);
              const combined = new Float32Array(total);
              let off = 0;
              for (const c of speechBuf) { combined.set(c, off); off += c.length; }
              vadQueue.current.push({ samples: combined, role: liveRoleRef.current, lk: langKeyRef.current });
              setQueueLen(vadQueue.current.length);
              drainQueue(patient.id);
            }
            speechBuf = []; silenceStart = null;
            setVadState('listening');
          }
        }
      };

      src.connect(proc);
      proc.connect(ctx.destination);
      setIsLive(true);
      setVadState('listening');
      setStatus('Live session active — speak now');
    } catch (err) {
      setStatus(`Live mode error: ${err.message}`);
    }
  };

  const stopLive = () => {
    liveProcessorRef.current?.disconnect();
    liveCtxRef.current?.close();
    liveStreamRef.current?.getTracks().forEach(t => t.stop());
    liveCtxRef.current = liveProcessorRef.current = liveStreamRef.current = null;
    vadQueue.current = []; vadProcessing.current = false;
    setIsLive(false); setVadState('idle'); setAudioLevel(0); setQueueLen(0);
    setStatus('Live session ended');
  };

  // ── Manual recording ──
  const startRecording = async (role) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];
      mediaRecorder.current.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.current.push(e.data); };
      mediaRecorder.current.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        setIsRecording(false); setIsProcessing(true); setStatus('Encoding WAV…');
        try {
          const wav = await blobToWav(new Blob(audioChunks.current));
          const fd  = new FormData();
          fd.append('file', wav, 'recording.wav');
          fd.append('role', role);
          fd.append('lang_key', langKeyRef.current);
          fd.append('patient_id', patient.id);
          setStatus('Transcribing & translating…');
          const res   = await axios.post(`${API_BASE}/stt`, fd);
          const entry = { role, original: res.data.original, translated: res.data.translated };
          await axios.post(`${API_BASE}/patients/${patient.id}/append_transcript`, entry);
          setMessages(prev => [...prev, { ...entry, timestamp: new Date().toLocaleTimeString() }]);
          setStatus('Done ✓');
        } catch (err) {
          setStatus(`Error: ${err.response?.data?.detail || err.message}`);
        } finally { setIsProcessing(false); setCurrentRole(null); }
      };
      mediaRecorder.current.start();
      setIsRecording(true); setCurrentRole(role);
      setStatus(`Recording ${role === 'doctor' ? 'Doctor' : 'Patient'}… click Stop when done`);
    } catch { setStatus('Microphone access denied.'); }
  };

  const stopRecording = () => {
    if (mediaRecorder.current?.state !== 'inactive') mediaRecorder.current.stop();
  };

  const submitPatientText = async () => {
    if (!patientText.trim()) return;
    setIsTranslatingText(true); setStatus('Translating patient text…');
    try {
      const res   = await axios.post(`${API_BASE}/translate`, { text: patientText, direction: 'native_to_en', lang_key: langKey });
      const entry = { role: 'patient', original: patientText, translated: res.data.translated };
      await axios.post(`${API_BASE}/patients/${patient.id}/append_transcript`, entry);
      setMessages(prev => [...prev, { ...entry, timestamp: new Date().toLocaleTimeString() }]);
      setPatientText(''); setStatus('Done ✓');
    } catch (err) {
      setStatus(`Text error: ${err.response?.data?.detail || err.message}`);
    } finally { setIsTranslatingText(false); }
  };

  const downloadPDF = () => window.open(`${API_BASE}/patients/${patient.id}/pdf`, '_blank');
  
  const clearChat = async () => {
    if (!window.confirm("Are you sure you want to clear this chat? It will be archived securely as a PDF.")) return;
    setIsProcessing(true);
    setStatus('Archiving chat…');
    try {
      await axios.post(`${API_BASE}/patients/${patient.id}/clear_chat`);
      setMessages([]);
      setStatus('Chat cleared and archived ✓');
    } catch (err) {
      setStatus(`Error clearing chat: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const busy = isRecording || isProcessing;
  const anyBusy = busy || isTranslatingText;

  const vadLabel = { idle: '', listening: '👂 Listening…', speaking: '🔴 Speech detected', processing: '⚙️ Processing…' }[vadState];

  return (
    <div className="consultation-page animate-in">
      {/* Header */}
      <div className="consult-header card">
        <div>
          <h2 style={{ margin: 0 }}>{patient.name}</h2>
          <span className="patient-id">Patient ID: {patient.id}</span>
        </div>
        <div className="lang-selector-wrap">
          <label>Language</label>
          <select className="lang-select" value={langKey} onChange={e => setLangKey(e.target.value)} disabled={anyBusy || isLive}>
            <option>Hindi (हिन्दी)</option>
            <option>Kannada (ಕನ್ನಡ)</option>
            <option>Marathi (मराठी)</option>
            <option>Bengali (বাংলা)</option>
            <option>Malayalam (മലയാളം)</option>
            <option>Tamil (தமிழ்)</option>
            <option>Konkani (कोंकणी)</option>
            <option>English</option>
          </select>
        </div>
      </div>

      <div className="consultation-layout">
        {/* Chat Panel */}
        <div className="card chat-panel">
          <div className="chat-box">
            {messages.length === 0 && <p className="empty-state">No messages yet. Use Live Session or the buttons below.</p>}
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.role}`}>
                <div className="message-header">
                  <span className={`role-badge ${m.role}`}>{m.role.toUpperCase()}</span>
                  <span className="msg-time">{m.timestamp}</span>
                </div>
                <p className="msg-original"><span className="msg-label">Original:</span> {m.original}</p>
                <p className="msg-translated"><span className="msg-label">Translated:</span> {m.translated}</p>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Status bar */}
          <div className={`status-bar ${(isProcessing || isTranslatingText || vadState === 'processing') ? 'processing' : ''}`}>
            {(isProcessing || isTranslatingText || vadState === 'processing') && <span className="spinner" />}
            {status}
            {queueLen > 0 && <span className="queue-badge">{queueLen} queued</span>}
          </div>

          {/* ── LIVE SESSION PANEL ── */}
          <div className={`live-panel ${isLive ? 'live-active' : ''}`}>
            <div className="live-panel-header">
              <div className="live-title-row">
                <span className={`live-dot ${isLive ? vadState : ''}`} />
                <span className="live-title-text">Live Session</span>
                {isLive && <span className="vad-label-text">{vadLabel}</span>}
              </div>
              <button
                className={`btn-live ${isLive ? 'stop' : 'start'}`}
                onClick={isLive ? stopLive : startLive}
                disabled={anyBusy}
              >
                {isLive ? '⏹ Stop Live' : '🎙️ Start Live Session'}
              </button>
            </div>

            {isLive && (
              <div className="live-controls">
                <div className="live-role-row">
                  <span className="live-role-label">Who is speaking?</span>
                  <button className={`btn-role-live ${liveRole === 'doctor' ? 'active-doc' : ''}`} onClick={() => setLiveRole('doctor')}>Doctor</button>
                  <button className={`btn-role-live ${liveRole === 'patient' ? 'active-pat' : ''}`} onClick={() => setLiveRole('patient')}>Patient</button>
                </div>
                <div className="audio-meter-wrap">
                  <div className="audio-meter-bar">
                    <div
                      className={`audio-meter-fill ${vadState === 'speaking' ? 'speaking' : ''}`}
                      style={{ width: `${Math.min(100, audioLevel * 100)}%` }}
                    />
                  </div>
                  <span className="audio-meter-label">mic level</span>
                </div>
              </div>
            )}
          </div>

          {/* Patient text input */}
          <div className="patient-text-panel">
            <div className="patient-text-header">
              <Keyboard size={15} />
              <span>Patient — Type Message</span>
              <span className="lang-badge">{langKey.split(' ')[0]}</span>
            </div>
            <div className="patient-text-row">
              <textarea
                className="patient-textarea"
                placeholder={`Type patient's message in ${langKey.split('(')[0].trim()}…`}
                value={patientText}
                onChange={e => setPatientText(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitPatientText(); } }}
                disabled={anyBusy}
                rows={2}
              />
              <button className="btn-send" onClick={submitPatientText} disabled={!patientText.trim() || anyBusy}>
                <Send size={16} />
              </button>
            </div>
          </div>

          {/* Manual recording controls */}
          <div className="controls">
            <button
              className={`btn-record ${currentRole === 'doctor' && isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : () => startRecording('doctor')}
              disabled={isLive || (anyBusy && currentRole !== 'doctor')}
              title={isLive ? 'Stop Live Session to use manual recording' : ''}
            >
              {currentRole === 'doctor' && isRecording ? <Square size={18} /> : <Mic size={18} />}
              {currentRole === 'doctor' && isRecording ? '■ Stop' : '🎤 Doctor Speak'}
            </button>
            <button
              className={`btn-record ${currentRole === 'patient' && isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : () => startRecording('patient')}
              disabled={isLive || (anyBusy && currentRole !== 'patient')}
              title={isLive ? 'Stop Live Session to use manual recording' : ''}
            >
              {currentRole === 'patient' && isRecording ? <Square size={18} /> : <Mic size={18} />}
              {currentRole === 'patient' && isRecording ? '■ Stop' : '🎤 Patient Speak'}
            </button>
          </div>
        </div>

        {/* Sidebar */}
        <div className="session-sidebar">
          <div className="card sidebar-card">
            <h4>Session Tools</h4>
            <button className="btn-outline" onClick={downloadPDF}>
              <FileText size={16} /> Download PDF Report
            </button>
            <button className="btn-outline" onClick={clearChat} disabled={anyBusy} style={{ borderColor: '#f59e0b', color: '#f59e0b' }}>
              <Square size={16} /> Clear Chat & Archive
            </button>
            <button className="btn-danger" onClick={onEnd} disabled={anyBusy}>
              <LogOut size={16} /> End Session
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
