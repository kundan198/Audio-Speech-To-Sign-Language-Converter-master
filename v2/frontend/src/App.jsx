import { useState, useRef, useEffect } from 'react';
import { 
  Brain, 
  MessageSquare, 
  Settings, 
  Play, 
  Zap, 
  Terminal as TerminalIcon,
  ShieldCheck,
  ChevronRight,
  RefreshCw,
  Mic,
  Camera,
  Activity
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import Webcam from 'react-webcam';

const API_BASE = 'http://127.0.0.1:8001';

const SignVideo = ({ signs }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    setCurrentIndex(0);
    if (signs.length > 0) setIsPlaying(true);
  }, [signs]);

  const handleEnded = () => {
    if (currentIndex < signs.length - 1) {
      setCurrentIndex(prev => prev + 1);
    } else {
      setIsPlaying(false);
    }
  };

  if (!signs || signs.length === 0) return null;

  return (
    <div className="sign-video-player">
      <video 
        key={signs[currentIndex]}
        src={`${API_BASE}/assets/${signs[currentIndex]}.mp4`}
        autoPlay
        onEnded={handleEnded}
        className="main-video"
      />
      <div className="video-overlay">
        <span className="current-word">{signs[currentIndex]}</span>
        <span className="progress">{currentIndex + 1} / {signs.length}</span>
      </div>
    </div>
  );
};

function App() {
  const [activeTab, setActiveTab] = useState('nanollm');
  const [prompt, setPrompt] = useState('');
  const [output, setOutput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainLoss, setTrainLoss] = useState([]);
  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_key') || 'AIzaSyBaOuMgoe3mudTuiFPneVNKyUqz-PnTt8s');
  const [slInput, setSlInput] = useState('');
  const [slOutput, setSlOutput] = useState([]);
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [isRecognizing, setIsRecognizing] = useState(false);
  const [terminalLogs, setTerminalLogs] = useState(['System ready. Ready for input.']);
  const webcamRef = useRef(null);

  const addLog = (msg) => setTerminalLogs(prev => [...prev.slice(-10), `> ${msg}`]);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const res = await fetch(`${API_BASE}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, max_tokens: 100 })
      });
      const data = await res.json();
      setOutput(data.output);
      addLog('Text generated successfully.');
    } catch {
      addLog('Error generating text.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    addLog('Starting training session...');
    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ iterations: 100 })
      });
      const data = await res.json();
      setTrainLoss(data.losses);
      addLog(`Training complete. Final Loss: ${data.final_loss.val.toFixed(4)}`);
    } catch {
      addLog('Training failed.');
    } finally {
      setIsTraining(false);
    }
  };

  const handleSimplify = async (textToUse) => {
    setIsGenerating(true);
    const text = textToUse || slInput;
    try {
      const res = await fetch(`${API_BASE}/simplify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, gemini_key: apiKey })
      });
      const data = await res.json();
      setSlOutput(data.output);
      if (data.note) addLog(data.note);
      addLog(`Translated: "${text}" into ${data.output.length} signs.`);
    } catch {
      addLog('Simplification failed.');
    } finally {
      setIsGenerating(false);
    }
  };

  const startListening = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      addLog('Speech recognition not supported in this browser.');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => {
      setIsListening(true);
      addLog('Listening...');
    };

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setSlInput(transcript);
      handleSimplify(transcript);
    };

    recognition.onerror = () => {
      setIsListening(false);
      addLog('Speech recognition error.');
    };

    recognition.onend = () => {
      setIsListening(false);
    };

    recognition.start();
  };

  const handleRecognize = async () => {
    if (!webcamRef.current) return;
    const screenshot = webcamRef.current.getScreenshot();
    if (!screenshot) return;

    setIsRecognizing(true);
    try {
      const res = await fetch(`${API_BASE}/recognize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          frames: [screenshot], 
          gemini_key: apiKey,
          context: transcript
        })
      });
      const data = await res.json();
      if (data.text && data.text !== '[unclear]') {
        setTranscript(prev => prev ? prev + ' ' + data.text : data.text);
        addLog(`Recognized sign: ${data.text}`);
      } else {
        addLog('Sign unclear, try again.');
      }
    } catch {
      addLog('Recognition error.');
    } finally {
      setIsRecognizing(false);
    }
  };

  const saveKey = (e) => {
    const val = e.target.value;
    setApiKey(val);
    localStorage.setItem('gemini_key', val);
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '2rem' }}>
          <Brain color="var(--accent-primary)" size={32} />
          <span style={{ fontWeight: 800, fontSize: '1.2rem' }}>ANTIGRAVITY</span>
        </div>
        
        <div 
          className={`nav-item ${activeTab === 'nanollm' ? 'active' : ''}`}
          onClick={() => setActiveTab('nanollm')}
        >
          <Zap size={20} />
          <span>NanoLLM Engine</span>
        </div>
        
        <div 
          className={`nav-item ${activeTab === 'sign' ? 'active' : ''}`}
          onClick={() => setActiveTab('sign')}
        >
          <MessageSquare size={20} />
          <span>Speech to Sign</span>
        </div>

        <div 
          className={`nav-item ${activeTab === 'recognize' ? 'active' : ''}`}
          onClick={() => setActiveTab('recognize')}
        >
          <Camera size={20} />
          <span>Sign to Text</span>
        </div>

        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--glass-border)', paddingTop: '1rem' }}>
          <div className="nav-item">
            <Settings size={20} />
            <span>Settings</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="main-content">
        <AnimatePresence mode="wait">
          {activeTab === 'nanollm' ? (
            <motion.div 
              key="nanollm"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <h1>Technical LLM Dashboard</h1>
              
              <div className="grid-2">
                <div className="glass-card">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                    <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Play size={18} color="var(--accent-primary)" />
                      Training Progress
                    </h3>
                    <button 
                      className="btn btn-primary" 
                      onClick={handleTrain}
                      disabled={isTraining}
                    >
                      {isTraining ? <RefreshCw className="animate-spin" size={18} /> : 'Start Training'}
                    </button>
                  </div>
                  
                  <div style={{ height: '200px', width: '100%' }}>
                    <ResponsiveContainer>
                      <LineChart data={trainLoss}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                        <XAxis dataKey="step" stroke="#666" />
                        <YAxis stroke="#666" />
                        <Tooltip 
                          contentStyle={{ background: '#111', border: '1px solid #333' }}
                          itemStyle={{ color: 'var(--accent-primary)' }}
                        />
                        <Line type="monotone" dataKey="loss" stroke="var(--accent-primary)" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="glass-card">
                  <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.5rem' }}>
                    <TerminalIcon size={18} color="#10b981" />
                    System Logs
                  </h3>
                  <div className="terminal">
                    {terminalLogs.map((log, i) => (
                      <div key={i} style={{ marginBottom: '4px' }}>{log}</div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="glass-card">
                <h3>Prompt NanoLLM</h3>
                <div style={{ position: 'relative', marginTop: '1rem' }}>
                  <textarea 
                    placeholder="Enter context for the model..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    rows={3}
                  />
                  <button 
                    className="btn btn-primary" 
                    style={{ position: 'absolute', bottom: '1rem', right: '1rem' }}
                    onClick={handleGenerate}
                    disabled={isGenerating}
                  >
                    Generate <ChevronRight size={18} />
                  </button>
                </div>
                
                {output && (
                  <div style={{ marginTop: '2rem', padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', borderLeft: '4px solid var(--accent-primary)' }}>
                    <p style={{ fontStyle: 'italic', color: '#ccc' }}>{output}</p>
                  </div>
                )}
              </div>
            </motion.div>
          ) : activeTab === 'sign' ? (
            <motion.div 
              key="sign"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <h1>Speech to Sign Translator</h1>
              
              <div className="glass-card" style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                <ShieldCheck size={24} color="#10b981" />
                <div style={{ flex: 1 }}>
                  <label style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>Gemini API Key (Optional)</label>
                  <input 
                    type="password" 
                    placeholder="AIza..." 
                    value={apiKey} 
                    onChange={saveKey}
                    style={{ marginTop: '0.25rem' }}
                  />
                </div>
              </div>

              <div className="glass-card">
                <h3>Sentence to Translate</h3>
                <textarea 
                  placeholder="e.g., I am going to the store because I need to buy some milk."
                  value={slInput}
                  onChange={(e) => setSlInput(e.target.value)}
                  rows={3}
                  style={{ marginTop: '1rem' }}
                />
                <div style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
                  <button 
                    className="btn btn-primary" 
                    onClick={() => handleSimplify()}
                    disabled={isGenerating}
                  >
                    Translate <Zap size={18} />
                  </button>
                  <button 
                    className={`btn ${isListening ? 'btn-pulse' : 'btn-secondary'}`}
                    onClick={startListening}
                    disabled={isGenerating}
                  >
                    <Mic size={18} /> {isListening ? 'Listening...' : 'Voice Input'}
                  </button>
                </div>
              </div>

              <div className="grid-2">
                {slOutput.length > 0 && (
                  <div className="glass-card">
                    <h3>Sign Playback</h3>
                    <div style={{ marginTop: '1.5rem' }}>
                      <SignVideo signs={slOutput} />
                    </div>
                  </div>
                )}

                {slOutput.length > 0 && (
                  <div className="glass-card">
                    <h3>Sign Sequence</h3>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '1rem' }}>
                      {slOutput.map((word, i) => (
                        <motion.div 
                          key={i} 
                          className="word-pill"
                          initial={{ scale: 0 }}
                          animate={{ scale: 1 }}
                          transition={{ delay: i * 0.05 }}
                        >
                          {word}
                        </motion.div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="recognize"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
            >
              <h1>Sign to Text Translator</h1>
              
              <div className="grid-2">
                <div className="glass-card">
                  <h3>Webcam Feed</h3>
                  <div className="webcam-container" style={{ marginTop: '1.5rem', position: 'relative' }}>
                    <Webcam 
                      ref={webcamRef}
                      screenshotFormat="image/jpeg"
                      style={{ width: '100%', borderRadius: '16px', border: '2px solid var(--glass-border)' }}
                    />
                    <div className="webcam-overlay">
                      <Activity className={isRecognizing ? 'animate-pulse' : ''} color="var(--accent-primary)" />
                    </div>
                  </div>
                  <button 
                    className="btn btn-primary" 
                    style={{ marginTop: '1.5rem', width: '100%', justifyContent: 'center' }}
                    onClick={handleRecognize}
                    disabled={isRecognizing}
                  >
                    {isRecognizing ? <RefreshCw className="animate-spin" /> : <Camera />} Capture & Recognize
                  </button>
                </div>

                <div className="glass-card">
                  <h3>Live Transcript</h3>
                  <div className="transcript-box" style={{ marginTop: '1.5rem' }}>
                    {transcript || "Signs will appear here as you perform them..."}
                  </div>
                  <button 
                    className="btn btn-secondary" 
                    style={{ marginTop: '1rem' }}
                    onClick={() => setTranscript('')}
                  >
                    Clear Transcript
                  </button>
                </div>
              </div>

              <div className="glass-card">
                <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Activity size={18} color="var(--accent-primary)" />
                  How it works
                </h3>
                <p style={{ marginTop: '1rem', color: 'var(--text-dim)', lineHeight: '1.6' }}>
                  Position yourself in front of the camera and perform an ASL sign. 
                  Click <b>Capture & Recognize</b> to send the current frame to Gemini Vision. 
                  The AI will analyze your hand position and add the translated text to your transcript.
                </p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
      
      <style>{`
        .animate-spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;
