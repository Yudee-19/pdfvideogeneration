import React, { useState, useEffect } from 'react';
import { getJobStatus, downloadSummary, generateSummary } from '../services/api';
import './SummaryPrompt.css';

const SummaryPrompt = ({ jobId, onSummaryGenerated, onDismiss }) => {
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [summaryStatus, setSummaryStatus] = useState(null);

  useEffect(() => {
    // Check if summary already exists
    const checkSummary = async () => {
      try {
        const response = await getJobStatus(jobId);
        if (response.metadata?.summary_path) {
          setSummaryStatus('completed');
          if (onSummaryGenerated) {
            onSummaryGenerated(response);
          }
        }
      } catch (err) {
        console.error('Error checking summary:', err);
      }
    };

    checkSummary();
  }, [jobId, onSummaryGenerated]);

  const handleGenerateSummary = async () => {
    setGenerating(true);
    setError(null);

    try {
      // Call API to generate summary
      await generateSummary(jobId);

      // Poll for summary completion
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await getJobStatus(jobId);
          if (statusResponse.metadata?.summary_path) {
            clearInterval(pollInterval);
            setSummaryStatus('completed');
            setGenerating(false);
            if (onSummaryGenerated) {
              onSummaryGenerated(statusResponse);
            }
          }
        } catch (err) {
          console.error('Error polling summary status:', err);
        }
      }, 3000);

      // Stop polling after 5 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        if (summaryStatus !== 'completed') {
          setError('Summary generation is taking longer than expected. Please check back later.');
          setGenerating(false);
        }
      }, 300000);
    } catch (err) {
      setError(err.message || 'Failed to generate summary');
      setGenerating(false);
    }
  };

  const handleDownloadSummary = async () => {
    try {
      const blob = await downloadSummary(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `summary_${jobId}.txt`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to download summary');
    }
  };

  if (summaryStatus === 'completed') {
    return (
      <div className="card summary-prompt">
        <h3>✓ Summary Generated!</h3>
        <p>The book summary has been generated successfully.</p>
        <div className="button-group">
          <button className="btn btn-success" onClick={handleDownloadSummary}>
            Download Summary
          </button>
          <button className="btn-secondary" onClick={onDismiss}>
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className="card summary-prompt"
      style={{ display: "none" }}
    >
      <h3>Generate Book Summary?</h3>
      <p>
        Would you like to generate a comprehensive summary of the entire book?
        This will create a detailed summary that can be used to generate a summary video.
      </p>

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {generating && (
        <div className="generating-section">
          <div className="loading-spinner"></div>
          <p>Generating summary... This may take a few minutes.</p>
        </div>
      )}

      <div className="button-group">
        <button
          className="btn btn-success"
          onClick={handleGenerateSummary}
          disabled={generating}
        >
          {generating ? 'Generating...' : 'Yes, Generate Summary'}
        </button>
        <button
          className="btn-secondary"
          onClick={onDismiss}
          disabled={generating}
        >
          No, Skip
        </button>
      </div>
    </div>
  );
};

export default SummaryPrompt;

