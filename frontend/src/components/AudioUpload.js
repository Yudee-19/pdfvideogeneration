import React, { useState, useRef } from 'react';
import { uploadAudio } from '../services/api';
import './PDFUpload.css'; // Reuse existing CSS for consistent look

const AudioUpload = ({ onUploadSuccess, onBack }) => {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (selectedFile) => {
    // Check if file type starts with 'audio/'
    if (selectedFile && selectedFile.type.startsWith('audio/')) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid Audio file (MP3, WAV, etc.)');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    handleFileSelect(droppedFile);
  };

  const handleFileInputChange = (e) => {
    const selectedFile = e.target.files[0];
    handleFileSelect(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Please select an audio file');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      // Call the new API endpoint
      const response = await uploadAudio(file);

      // Pass the job_id back to App.js to trigger the JobStatus view
      onUploadSuccess(response.job_id);
    } catch (err) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to upload audio.';
      setError(errorMessage);
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="card">
      <div className="card-header">
        <h2>Upload Audio</h2>
        <p className="subtitle">Convert audio files directly into video</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Select Audio File</label>
          <div
            className={`file-input-wrapper ${isDragging ? 'dragover' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*" // Accepts mp3, wav, etc.
              onChange={handleFileInputChange}
              disabled={isUploading}
            />
            <div className="file-input-label">
              {file ? (
                <div>
                  <p>🎵 {file.name}</p>
                  <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <div>
                  <p>🎧 Drag and drop your Audio file here</p>
                  <p>or click to browse</p>
                </div>
              )}
            </div>
          </div>
        </div>

        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}

        <div className="button-group">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={onBack}
            disabled={isUploading}
          >
            Back
          </button>
          <button
            type="submit"
            className="btn"
            disabled={isUploading || !file}
          >
            {isUploading ? 'Uploading...' : 'Start Visualization'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default AudioUpload;