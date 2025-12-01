import React, { useState } from 'react';
import './App.css';
import HomePage from './components/HomePage';
import PDFUpload from './components/PDFUpload';
import JobStatus from './components/JobStatus';
import SummaryPrompt from './components/SummaryPrompt';
import SummaryVideoPrompt from './components/SummaryVideoPrompt';
import SummaryGeneration from './components/SummaryGeneration';
import SummaryReview from './components/SummaryReview';
import ReelsShorts from './components/ReelsShorts';
import AudioUpload from './components/AudioUpload';
function App() {
  const [currentView, setCurrentView] = useState('home'); // 'home', 'generate-video', 'summary-video', 'summary-generation', 'summary-review', 'job-status'
  const [currentJobId, setCurrentJobId] = useState(null);
  const [summaryJobId, setSummaryJobId] = useState(null);
  const [summaryText, setSummaryText] = useState(null);
  const [showSummaryPrompt, setShowSummaryPrompt] = useState(false);
  const [showSummaryVideoPrompt, setShowSummaryVideoPrompt] = useState(false);
  const [jobStatus, setJobStatus] = useState(null);
  const [summaryPromptShown, setSummaryPromptShown] = useState(false);
  const [summaryVideoPromptShown, setSummaryVideoPromptShown] = useState(false);
  const [stopPolling, setStopPolling] = useState(false);

  const handleSelectOption = (option) => {
    if (option === 'generate-video') {
      setCurrentView('generate-video');
    } else if (option === 'summary-video') {
      setCurrentView('summary-generation');
    } else if (option === 'reels-shorts') {
      setCurrentView('reels-shorts');
    } else if (option === 'audio-video') { // <--- Added this
      setCurrentView('audio-video');
    }
  };

  const handleUploadSuccess = (jobId) => {
    setCurrentJobId(jobId);
    setCurrentView('job-status');
    setShowSummaryPrompt(false);
    setShowSummaryVideoPrompt(false);
    setSummaryPromptShown(false);
    setSummaryVideoPromptShown(false);
    setJobStatus(null);
  };

  const handleSummaryGenerated = (status) => {
    setSummaryJobId(status.job_id);
    setSummaryText(status.metadata?.summary_text || '');
    setCurrentView('summary-review');
  };

  const handleVideoGenerated = (response) => {
    setCurrentJobId(response.job_id);
    setCurrentView('job-status');
  };

  const handleMainVideoComplete = (status) => {
    setJobStatus(status);
    // Show summary prompt when main video is completed and has video path
    if (status?.status?.toLowerCase() === 'completed' && status?.metadata?.final_video_path) {
      // Only show if we haven't shown it yet and summary hasn't been generated
      if (!summaryPromptShown && !status?.metadata?.summary_path) {
        setShowSummaryPrompt(true);
        setSummaryPromptShown(true);
      }
    }
  };

  const handleSummaryGeneratedFromPrompt = (status) => {
    setJobStatus(status);
    // Show summary video prompt when summary is completed
    if (status?.metadata?.summary_path) {
      // Only show if we haven't shown it yet and summary video hasn't been generated
      if (!summaryVideoPromptShown && !status?.metadata?.summary_video_path) {
        setShowSummaryVideoPrompt(true);
        setSummaryVideoPromptShown(true);
      }
    }
  };

  const handleSummaryVideoComplete = (status) => {
    setJobStatus(status);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>📚 PDF to Video Generator</h1>
        <p>Convert your PDF books into narrated videos</p>
      </header>

      <main className="App-main">
        {currentView === 'home' && (
          <HomePage onSelectOption={handleSelectOption} />
        )}

        {currentView === 'generate-video' && (
          <PDFUpload
            onUploadSuccess={handleUploadSuccess}
            onBack={() => setCurrentView('home')}
          />
        )}

        {currentView === 'summary-generation' && (
          <SummaryGeneration
            onSummaryGenerated={handleSummaryGenerated}
            onBack={() => setCurrentView('home')}
          />
        )}

        {currentView === 'summary-review' && summaryText && (
          <SummaryReview
            summaryJobId={summaryJobId}
            summaryText={summaryText}
            onVideoGenerated={handleVideoGenerated}
            onBack={() => setCurrentView('summary-generation')}
          />
        )}

        {currentView === 'reels-shorts' && (
          <ReelsShorts
            onVideoGenerated={handleVideoGenerated}
            onBack={() => setCurrentView('home')}
          />
        )}
        {currentView === 'audio-video' && (
          <AudioUpload
            onUploadSuccess={handleUploadSuccess}
            onBack={() => setCurrentView('home')}
          />
        )}

        {currentView === 'job-status' && currentJobId && (
          <>
            <JobStatus
              jobId={currentJobId}
              onStatusUpdate={handleMainVideoComplete}
              stopPolling={stopPolling}
            />

            {showSummaryPrompt && (
              <SummaryPrompt
                jobId={currentJobId}
                onSummaryGenerated={handleSummaryGeneratedFromPrompt}
                onDismiss={() => setShowSummaryPrompt(false)}
              />
            )}

            {showSummaryVideoPrompt && jobStatus?.metadata?.summary_path && (
              <SummaryVideoPrompt
                jobId={currentJobId}
                onVideoGenerated={handleSummaryVideoComplete}
                onDismiss={() => setShowSummaryVideoPrompt(false)}
              />
            )}

            <button
              className="btn-secondary"
              onClick={() => {
                setCurrentView('home');
                setCurrentJobId(null);
                setSummaryJobId(null);
                setSummaryText(null);
                setShowSummaryPrompt(false);
                setShowSummaryVideoPrompt(false);
                setJobStatus(null);
                setSummaryPromptShown(false);
                setSummaryVideoPromptShown(false);
              }}
            >
              Start New Job
            </button>
          </>
        )}
      </main>
    </div>
  );
}

export default App;

