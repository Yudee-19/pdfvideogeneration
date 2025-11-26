import React, { useState, useEffect } from 'react';
import { getJobStatus, downloadVideo } from '../services/api';
import './JobStatus.css';

const JobStatus = ({ jobId, onStatusUpdate }) => {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    let isMounted = true;
    let intervalId = null;

    const fetchStatus = async () => {
      try {
        const response = await getJobStatus(jobId);

        if (!isMounted) return;

        // Debug logging
        console.log('Job status response:', {
          status: response.status,
          progress: response.progress,
          message: response.message,
          metadata: response.metadata
        });

        setStatus(response);
        setLoading(false);

        // Notify parent component of status update
        if (onStatusUpdate) {
          onStatusUpdate(response);
        }

        // Stop polling if job is complete or failed
        const isTerminalState = ['completed', 'failed', 'cancelled'].includes(response.status);
        if (isTerminalState && intervalId) {
          console.log('Job reached terminal state, stopping polling');
          clearInterval(intervalId);
          intervalId = null;
        }
      } catch (err) {
        if (!isMounted) return;

        const isTimeout = err.code === 'ECONNABORTED' || err.message.includes('timeout');
        const errorMessage = err.response?.data?.detail || err.message || 'Failed to fetch job status';

        setStatus(currentStatus => {
          if (!currentStatus) {
            if (isTimeout) {
              setError('Request timeout - the server may be slow or unreachable. Please check if the backend is running.');
            } else {
              setError(errorMessage);
            }
            setLoading(false);
          } else {
            if (isTimeout) {
              console.warn('Status fetch timeout (continuing with cached status):', err.message);
            } else {
              console.warn('Status fetch error (continuing with cached status):', err.message);
            }
          }
          return currentStatus;
        });

        console.error('Status fetch error:', err);
      }
    };

    // Initial fetch
    fetchStatus();

    // Poll every 3 seconds
    intervalId = setInterval(fetchStatus, 3000);

    return () => {
      isMounted = false;
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [jobId, onStatusUpdate]);

  const handleDownload = async () => {
    if (!status?.metadata?.final_video_path) {
      setError('Video not available yet');
      return;
    }

    setDownloading(true);
    try {
      const blob = await downloadVideo(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `video_${jobId}.mp4`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to download video');
      console.error('Download error:', err);
    } finally {
      setDownloading(false);
    }
  };

  if (loading) {
    return (
      <div className="card">
        <div className="loading-spinner"></div>
        <p style={{ textAlign: 'center' }}>Loading job status...</p>
      </div>
    );
  }

  if (error && !status) {
    return (
      <div className="card">
        <div className="alert alert-error">
          <strong>Error:</strong> {error}
          <br />
          <small>
            Make sure the backend API is running on {process.env.REACT_APP_API_URL || 'http://54.198.232.153:8000/api'}
            <br />
            Job ID: {jobId}
          </small>
        </div>
        <button className="btn" onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }

  const getStatusBadgeClass = (status) => {
    switch (status?.toLowerCase()) {
      case 'completed':
        return 'status-completed';
      case 'processing':
      case 'running':
        return 'status-running';
      case 'failed':
        return 'status-failed';
      default:
        return 'status-pending';
    }
  };

  const mainProgress = status?.progress ?? status?.metadata?.progress ?? 0;
  const isProcessing = status?.status === 'processing' || status?.status === 'pending' || status?.status === 'running';
  const showMainProgress = isProcessing && mainProgress < 100;
  const isCompleted = status?.status === 'completed';
  const hasVideo = status?.metadata?.final_video_path;
  const summaryProgress = status?.metadata?.summary_progress;
  const summaryStatus = status?.metadata?.summary_status;
  const summaryVideoProgress = status?.metadata?.summary_video_progress;
  const summaryVideoStatus = status?.metadata?.summary_video_status;
  const showSummaryProgress = summaryStatus === 'processing';
  const showSummaryVideoProgress = summaryVideoStatus === 'processing';

  return (
    <div className="card">
      <h2>Job Status</h2>
      <div className="job-info">
        <div className="job-id">
          <strong>Job ID:</strong> {jobId}
        </div>
        <div className="status-section">
          <span className={`status-badge ${getStatusBadgeClass(status?.status)}`}>
            {status?.status || 'Unknown'}
          </span>
        </div>
      </div>

      {showMainProgress && (
        <div className="progress-section">
          <div className="progress-title">
            <span className="progress-icon">🎬</span>
            Main Video Generation
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${mainProgress}%`,
                transition: 'width 0.3s ease'
              }}
            ></div>
          </div>
          <div className="progress-info">
            <p className="progress-text">
              {status?.message || 'Processing video... This may take several minutes.'}
            </p>
            <p className="progress-percentage">
              {mainProgress}%
            </p>
          </div>
          {mainProgress < 30 && (
            <div className="progress-stage">
              <span className="stage-indicator">📄</span>
              <span>Extracting and processing PDF content...</span>
            </div>
          )}
          {mainProgress >= 30 && mainProgress < 60 && (
            <div className="progress-stage">
              <span className="stage-indicator">🎤</span>
              <span>Generating audio narration and timestamps...</span>
            </div>
          )}
          {mainProgress >= 60 && mainProgress < 90 && (
            <div className="progress-stage">
              <span className="stage-indicator">🎨</span>
              <span>Rendering video frames with synchronized text...</span>
            </div>
          )}
          {mainProgress >= 90 && mainProgress < 100 && (
            <div className="progress-stage">
              <span className="stage-indicator">⚙️</span>
              <span>Encoding final video...</span>
            </div>
          )}
        </div>
      )}

      {showSummaryProgress && (
        <div className="progress-section">
          <div className="progress-title">
            <span className="progress-icon">📝</span>
            Summary Generation
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${summaryProgress || 0}%`,
                transition: 'width 0.3s ease'
              }}
            ></div>
          </div>
          <div className="progress-info">
            <p className="progress-text">
              Generating book summary... This may take several minutes.
            </p>
            <p className="progress-percentage">
              {summaryProgress || 0}%
            </p>
          </div>
        </div>
      )}

      {summaryStatus === 'completed' && summaryProgress === 100 && (
        <div className="alert alert-success">
          ✓ Summary generated successfully. Ready to create summary video.
        </div>
      )}

      {showSummaryVideoProgress && (
        <div className="progress-section">
          <div className="progress-title">
            <span className="progress-icon">🎥</span>
            Summary Video Generation
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${summaryVideoProgress || 0}%`,
                transition: 'width 0.3s ease'
              }}
            ></div>
          </div>
          <div className="progress-info">
            <p className="progress-text">
              Generating summary video... This may take a few minutes.
            </p>
            <p className="progress-percentage">
              {summaryVideoProgress || 0}%
            </p>
          </div>
        </div>
      )}

      {summaryVideoStatus === 'completed' && summaryVideoProgress === 100 && (
        <div className="alert alert-success">
          ✓ Summary video generated successfully. You can download it from the summary section.
        </div>
      )}

      {status?.message && (
        <div className="alert alert-info">
          {status.message}
        </div>
      )}

      {error && (
        <div className="alert alert-error">
          {error}
        </div>
      )}

      {isCompleted && hasVideo && (
        <div className="download-section">
          <div className="alert alert-success">
            <span className="success-icon">✓</span>
            <div>
              <strong>Main video generated successfully!</strong>
              <p style={{ margin: '8px 0 0 0', fontSize: '0.9em', opacity: 0.9 }}>
                Your video includes synchronized text highlighting with all words and punctuation preserved.
              </p>
            </div>
          </div>
          <button
            className="btn btn-success"
            onClick={handleDownload}
            disabled={downloading}
          >
            {downloading ? (
              <>
                <span className="spinner-small"></span>
                Downloading...
              </>
            ) : (
              <>
                <span>⬇️</span>
                Download Main Video
              </>
            )}
          </button>
        </div>
      )}

      {status?.metadata && (
        <div className="metadata-section">
          <h3>Job Details</h3>
          <div className="metadata-grid">
            {status.metadata.book_title && (
              <div className="metadata-item">
                <span className="metadata-icon">📚</span>
                <div>
                  <strong>Book:</strong> {status.metadata.book_title}
                </div>
              </div>
            )}
            {status.metadata.genre && (
              <div className="metadata-item">
                <span className="metadata-icon">🏷️</span>
                <div>
                  <strong>Genre:</strong> {status.metadata.genre}
                </div>
              </div>
            )}
            {status.metadata.start_page && (
              <div className="metadata-item">
                <span className="metadata-icon">📄</span>
                <div>
                  <strong>Pages:</strong> {status.metadata.start_page} - {status.metadata.end_page}
                </div>
              </div>
            )}
            {status.metadata.voice_provider && (
              <div className="metadata-item">
                <span className="metadata-icon">🎙️</span>
                <div>
                  <strong>Voice Provider:</strong> {status.metadata.voice_provider.charAt(0).toUpperCase() + status.metadata.voice_provider.slice(1)}
                </div>
              </div>
            )}
            {status.metadata.cartesia_voice_id && (
              <div className="metadata-item">
                <span className="metadata-icon">🔊</span>
                <div>
                  <strong>Cartesia Voice:</strong> {status.metadata.cartesia_voice_id.substring(0, 8)}...
                </div>
              </div>
            )}
            {status.metadata.cartesia_model_id && (
              <div className="metadata-item">
                <span className="metadata-icon">🤖</span>
                <div>
                  <strong>Cartesia Model:</strong> {status.metadata.cartesia_model_id}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default JobStatus;

