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
        {view === 'consultation' && <Consultation patient={currentPatient} onEnd={() => navigateTo('dashboard')} />}
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
  const [messages, setMessages] = useState([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentRole, setCurrentRole] = useState(null);
  const [langKey, setLangKey] = useState('Hindi (हिन्दी)');
  const [status, setStatus] = useState('Ready');
  const [patientText, setPatientText] = useState('');
  const [isTranslatingText, setIsTranslatingText] = useState(false);
  const mediaRecorder = useRef(null);
  const audioChunks = useRef([]);
  const langKeyRef = useRef(langKey);
  const chatEndRef = useRef(null);

  // Keep ref in sync with state so closures always see latest value
  useEffect(() => { langKeyRef.current = langKey; }, [langKey]);

  useEffect(() => {
    axios.get(`${API_BASE}/patients/${patient.id}/transcripts`)
      .then(res => setMessages(res.data))
      .catch(() => setStatus('Could not load transcripts'));
  }, [patient.id]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const startRecording = async (role) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder.current = new MediaRecorder(stream);
      audioChunks.current = [];

      mediaRecorder.current.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunks.current.push(e.data);
      };

      mediaRecorder.current.onstop = async () => {
        // Stop all tracks to release mic
        stream.getTracks().forEach(t => t.stop());

        setIsRecording(false);
        setIsProcessing(true);
        setStatus('Converting audio…');

        try {
          const rawBlob = new Blob(audioChunks.current);  // browser-native format (webm/ogg)
          
          // Convert to 16kHz mono WAV so backend can read without ffmpeg
          setStatus('Encoding WAV…');
          const wavBlob = await blobToWav(rawBlob);

          const formData = new FormData();
          formData.append('file', wavBlob, 'recording.wav');
          formData.append('role', role);
          formData.append('lang_key', langKeyRef.current);  // use ref, not stale closure

          setStatus('Transcribing & translating…');
          const res = await axios.post(`${API_BASE}/stt`, formData);
          const { original, translated } = res.data;

          const entry = { role, original, translated };
          await axios.post(`${API_BASE}/patients/${patient.id}/append_transcript`, entry);

          setMessages(prev => [...prev, { ...entry, timestamp: new Date().toLocaleTimeString() }]);
          setStatus('Done ✓');
        } catch (err) {
          const msg = err.response?.data?.detail || err.message;
          setStatus(`Error: ${msg}`);
        } finally {
          setIsProcessing(false);
          setCurrentRole(null);
        }
      };

      mediaRecorder.current.start();
      setIsRecording(true);
      setCurrentRole(role);
      setStatus(`Recording ${role === 'doctor' ? 'Doctor' : 'Patient'}… click Stop when done`);
    } catch (err) {
      setStatus('Microphone access denied. Please allow mic permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop();
    }
  };

  const downloadPDF = () => {
    window.open(`${API_BASE}/patients/${patient.id}/pdf`, '_blank');
  };

  const submitPatientText = async () => {
    if (!patientText.trim()) return;
    setIsTranslatingText(true);
    setStatus('Translating patient text…');
    try {
      const res = await axios.post(`${API_BASE}/translate`, {
        text: patientText,
        direction: 'native_to_en',
        lang_key: langKey,
      });
      const entry = { role: 'patient', original: patientText, translated: res.data.translated };
      await axios.post(`${API_BASE}/patients/${patient.id}/append_transcript`, entry);
      setMessages(prev => [...prev, { ...entry, timestamp: new Date().toLocaleTimeString() }]);
      setPatientText('');
      setStatus('Done ✓');
    } catch (err) {
      setStatus(`Text translate error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsTranslatingText(false);
    }
  };

  const busy = isRecording || isProcessing;

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
          <select
            className="lang-select"
            value={langKey}
            onChange={e => setLangKey(e.target.value)}
            disabled={busy}
          >
            <option>Hindi (हिन्दी)</option>
            <option>Kannada (ಕನ್ನಡ)</option>
            <option>Marathi (मराठी)</option>
            <option>Bengali (বাংলা)</option>
          </select>
        </div>
      </div>

      <div className="consultation-layout">
        {/* Chat Panel */}
        <div className="card chat-panel">
          <div className="chat-box">
            {messages.length === 0 && (
              <p className="empty-state">No messages yet. Use the buttons below to start speaking.</p>
            )}
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
          <div className={`status-bar ${(isProcessing || isTranslatingText) ? 'processing' : ''}`}>
            {(isProcessing || isTranslatingText) && <span className="spinner" />}
            {status}
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
                disabled={busy || isTranslatingText}
                rows={2}
              />
              <button
                className="btn-send"
                onClick={submitPatientText}
                disabled={!patientText.trim() || busy || isTranslatingText}
              >
                <Send size={16} />
              </button>
            </div>
          </div>

          {/* Recording controls */}
          <div className="controls">
            <button
              className={`btn-record ${currentRole === 'doctor' && isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : () => startRecording('doctor')}
              disabled={(busy || isTranslatingText) && currentRole !== 'doctor'}
            >
              {currentRole === 'doctor' && isRecording ? <Square size={18} /> : <Mic size={18} />}
              {currentRole === 'doctor' && isRecording ? '■ Stop' : '🎤 Doctor Speak'}
            </button>
            <button
              className={`btn-record ${currentRole === 'patient' && isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : () => startRecording('patient')}
              disabled={(busy || isTranslatingText) && currentRole !== 'patient'}
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
            <button className="btn-danger" onClick={onEnd} disabled={busy}>
              <LogOut size={16} /> End Session
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
