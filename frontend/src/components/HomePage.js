import React from 'react';
import './HomePage.css';

const HomePage = ({ onSelectOption }) => {
  return (
    <div className="home-page">
      <div className="options-container">
        <div className="option-card" onClick={() => onSelectOption('generate-video')}>
          <div className="option-icon">🎬</div>
          <h2>Generate Video from PDF</h2>
          <p>Upload a PDF and generate a video from specific pages</p>
        </div>

        <div className="option-card" onClick={() => onSelectOption('summary-video')}>
          <div className="option-icon">📝</div>
          <h2>Create Summary Video</h2>
          <p>Upload a PDF, generate an extensive summary, and create a video</p>
        </div>

        <div className="option-card" onClick={() => onSelectOption('reels-shorts')}>
          <div className="option-icon">📱</div>
          <h2>Create Reels/Shorts</h2>
          <p>Enter text directly to create short social media videos</p>
        </div>

        {/* --- NEW CARD ADDED HERE --- */}
        <div className="option-card" onClick={() => onSelectOption('audio-video')}>
          <div className="option-icon">🎵</div>
          <h2>Audio to Video</h2>
          <p>Upload an audio file (MP3/WAV) and convert it into a video visualization</p>
        </div>
      </div>
    </div>
  );
};

export default HomePage;