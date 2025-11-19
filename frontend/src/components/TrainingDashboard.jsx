import React, { useState, useEffect } from 'react';
import './TrainingDashboard.css';

const TrainingDashboard = () => {
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [architecture, setArchitecture] = useState(null);
  const [recordedGames, setRecordedGames] = useState([]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState(null);
  const [weightComparison, setWeightComparison] = useState(null);
  const [activeTab, setActiveTab] = useState('status');
  const [trainingConfig, setTrainingConfig] = useState({
    episodes_per_phase: 10000,
    checkpoint_interval: 1000,
    games_to_record: 3,
    num_simulations: 100
  });

  const API_BASE = 'http://localhost:8000/api/training';

  useEffect(() => {
    fetchTrainingStatus();
    fetchArchitecture();
    const interval = setInterval(fetchTrainingStatus, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchTrainingStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      const data = await response.json();
      setTrainingStatus(data);
    } catch (error) {
      console.error('Error fetching status:', error);
    }
  };

  const fetchCheckpoints = async () => {
    try {
      const response = await fetch(`${API_BASE}/checkpoints`);
      const data = await response.json();
      setCheckpoints(data.checkpoints || []);
    } catch (error) {
      console.error('Error fetching checkpoints:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE}/metrics?limit=1000`);
      const data = await response.json();
      setMetrics(data.metrics || []);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const fetchArchitecture = async () => {
    try {
      const response = await fetch(`${API_BASE}/architecture`);
      const data = await response.json();
      setArchitecture(data);
    } catch (error) {
      console.error('Error fetching architecture:', error);
    }
  };

  const fetchRecordedGames = async () => {
    try {
      const response = await fetch(`${API_BASE}/games`);
      const data = await response.json();
      setRecordedGames(data.game_files || []);
    } catch (error) {
      console.error('Error fetching games:', error);
    }
  };

  const initializeTraining = async () => {
    try {
      const response = await fetch(`${API_BASE}/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingConfig)
      });
      const data = await response.json();
      alert(data.message);
      fetchTrainingStatus();
      fetchCheckpoints();
    } catch (error) {
      console.error('Error initializing training:', error);
      alert('Failed to initialize training');
    }
  };

  const startPhase = async (phaseNumber, numEpisodes) => {
    try {
      const response = await fetch(`${API_BASE}/start-phase`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          phase_number: phaseNumber,
          num_episodes: numEpisodes,
          num_simulations: trainingConfig.num_simulations
        })
      });
      const data = await response.json();
      alert(data.message);
      fetchTrainingStatus();
    } catch (error) {
      console.error('Error starting phase:', error);
      alert('Failed to start training phase');
    }
  };

  const pauseTraining = async () => {
    try {
      const response = await fetch(`${API_BASE}/pause`, { method: 'POST' });
      const data = await response.json();
      alert('Training paused at episode ' + data.episode);
      fetchTrainingStatus();
    } catch (error) {
      console.error('Error pausing training:', error);
    }
  };

  const resumeTraining = async () => {
    try {
      const response = await fetch(`${API_BASE}/resume`, { method: 'POST' });
      const data = await response.json();
      alert('Training resumed from episode ' + data.episode);
      fetchTrainingStatus();
    } catch (error) {
      console.error('Error resuming training:', error);
    }
  };

  const compareCheckpoints = async (cp1, cp2) => {
    try {
      const response = await fetch(`${API_BASE}/weight-comparison?checkpoint1=${cp1}&checkpoint2=${cp2}`);
      const data = await response.json();
      setWeightComparison(data);
    } catch (error) {
      console.error('Error comparing checkpoints:', error);
    }
  };

  const renderStatus = () => (
    <div className="tab-content">
      <h2>Training Status</h2>
      {!trainingStatus?.initialized ? (
        <div className="initialization-section">
          <h3>Initialize Training</h3>
          <div className="config-form">
            <label>
              Episodes per phase:
              <input
                type="number"
                value={trainingConfig.episodes_per_phase}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, episodes_per_phase: parseInt(e.target.value) })}
              />
            </label>
            <label>
              Checkpoint interval:
              <input
                type="number"
                value={trainingConfig.checkpoint_interval}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, checkpoint_interval: parseInt(e.target.value) })}
              />
            </label>
            <label>
              Games to record:
              <input
                type="number"
                value={trainingConfig.games_to_record}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, games_to_record: parseInt(e.target.value) })}
              />
            </label>
            <label>
              MCTS simulations:
              <input
                type="number"
                value={trainingConfig.num_simulations}
                onChange={(e) => setTrainingConfig({ ...trainingConfig, num_simulations: parseInt(e.target.value) })}
              />
            </label>
          </div>
          <button onClick={initializeTraining} className="btn btn-primary">
            Initialize Training
          </button>
        </div>
      ) : (
        <div className="status-info">
          <div className="status-card">
            <h3>Current Status</h3>
            <p><strong>Training Active:</strong> {trainingStatus.is_training ? 'Yes' : 'No'}</p>
            <p><strong>Task Running:</strong> {trainingStatus.task_active ? 'Yes' : 'No'}</p>
            <p><strong>Current Episode:</strong> {trainingStatus.current_episode}</p>
            <p><strong>Current Phase:</strong> {trainingStatus.current_phase}</p>
            <p><strong>Total Checkpoints:</strong> {trainingStatus.total_checkpoints}</p>
            <p><strong>Recorded Games:</strong> {trainingStatus.total_recorded_games}</p>
          </div>

          <div className="control-buttons">
            <button onClick={() => startPhase(1, 10000)} className="btn btn-success" disabled={trainingStatus.task_active}>
              Start Phase 1 (10,000 episodes)
            </button>
            <button onClick={pauseTraining} className="btn btn-secondary" disabled={!trainingStatus.is_training}>
              Pause Training
            </button>
            <button onClick={resumeTraining} className="btn btn-info" disabled={trainingStatus.is_training}>
              Resume Training
            </button>
          </div>

          {trainingStatus.recent_metrics && trainingStatus.recent_metrics.length > 0 && (
            <div className="recent-metrics">
              <h3>Recent Metrics</h3>
              <table>
                <thead>
                  <tr>
                    <th>Episode</th>
                    <th>Phase</th>
                    <th>Total Loss</th>
                    <th>Policy Loss</th>
                    <th>Value Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingStatus.recent_metrics.map((m, idx) => (
                    <tr key={idx}>
                      <td>{m.episode}</td>
                      <td>{m.phase}</td>
                      <td>{m.total_loss?.toFixed(4)}</td>
                      <td>{m.policy_loss?.toFixed(4)}</td>
                      <td>{m.value_loss?.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );

  const renderArchitecture = () => (
    <div className="tab-content">
      <h2>Neural Network Architecture</h2>
      {architecture ? (
        <div className="architecture-info">
          <div className="arch-summary">
            <h3>Model Summary</h3>
            <p><strong>Board Size:</strong> {architecture.board_size}x{architecture.board_size}</p>
            <p><strong>Input Shape:</strong> {architecture.input_shape.join(' × ')}</p>
            <p><strong>Total Parameters:</strong> {architecture.total_parameters?.toLocaleString()}</p>
            <p><strong>Trainable Parameters:</strong> {architecture.trainable_parameters?.toLocaleString()}</p>
          </div>

          <div className="layers-list">
            <h3>Network Layers</h3>
            <button onClick={fetchArchitecture} className="btn btn-secondary">Refresh</button>
            <div className="layers-table">
              <table>
                <thead>
                  <tr>
                    <th>Layer Name</th>
                    <th>Type</th>
                    <th>Weight Shape</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  {architecture.layers.map((layer, idx) => (
                    <tr key={idx}>
                      <td className="layer-name">{layer.name || 'N/A'}</td>
                      <td>{layer.type}</td>
                      <td>{layer.weight_shape ? layer.weight_shape.join(' × ') : '-'}</td>
                      <td>{layer.num_parameters?.toLocaleString() || '-'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : (
        <p>Loading architecture...</p>
      )}
    </div>
  );

  const renderCheckpoints = () => (
    <div className="tab-content">
      <h2>Weight Checkpoints</h2>
      <button onClick={fetchCheckpoints} className="btn btn-primary">Load Checkpoints</button>

      {checkpoints.length > 0 ? (
        <div className="checkpoints-list">
          <table>
            <thead>
              <tr>
                <th>Checkpoint</th>
                <th>Episode</th>
                <th>Phase</th>
                <th>Timestamp</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {checkpoints.map((cp, idx) => (
                <tr key={idx} className={selectedCheckpoint === cp ? 'selected' : ''}>
                  <td>{cp.checkpoint_file?.split('/').pop()?.replace('.pt', '')}</td>
                  <td>{cp.episode}</td>
                  <td>{cp.phase}</td>
                  <td>{new Date(cp.timestamp).toLocaleString()}</td>
                  <td>
                    <button onClick={() => setSelectedCheckpoint(cp)} className="btn-small">
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {selectedCheckpoint && (
            <div className="checkpoint-details">
              <h3>Checkpoint Details</h3>
              <p><strong>Episode:</strong> {selectedCheckpoint.episode}</p>
              <p><strong>Phase:</strong> {selectedCheckpoint.phase}</p>

              <h4>Weight Statistics</h4>
              <div className="weight-stats-table">
                <table>
                  <thead>
                    <tr>
                      <th>Layer</th>
                      <th>Shape</th>
                      <th>Mean</th>
                      <th>Std Dev</th>
                      <th>Min</th>
                      <th>Max</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(selectedCheckpoint.weight_stats || {}).map(([name, stats]) => (
                      <tr key={name}>
                        <td className="layer-name">{name}</td>
                        <td>{stats.shape.join(' × ')}</td>
                        <td>{stats.mean.toFixed(6)}</td>
                        <td>{stats.std.toFixed(6)}</td>
                        <td>{stats.min.toFixed(6)}</td>
                        <td>{stats.max.toFixed(6)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      ) : (
        <p>No checkpoints available. Initialize training first.</p>
      )}
    </div>
  );

  const renderMetrics = () => (
    <div className="tab-content">
      <h2>Training Metrics</h2>
      <button onClick={fetchMetrics} className="btn btn-primary">Load Metrics</button>

      {metrics.length > 0 ? (
        <div className="metrics-visualization">
          <div className="metrics-summary">
            <h3>Metrics Summary</h3>
            <p><strong>Total Episodes:</strong> {metrics.length}</p>
            <p><strong>Latest Episode:</strong> {metrics[metrics.length - 1]?.episode}</p>
            <p><strong>Latest Total Loss:</strong> {metrics[metrics.length - 1]?.total_loss?.toFixed(4)}</p>
          </div>

          <div className="metrics-table">
            <table>
              <thead>
                <tr>
                  <th>Episode</th>
                  <th>Phase</th>
                  <th>Total Loss</th>
                  <th>Policy Loss</th>
                  <th>Value Loss</th>
                </tr>
              </thead>
              <tbody>
                {metrics.slice(-50).reverse().map((m, idx) => (
                  <tr key={idx}>
                    <td>{m.episode}</td>
                    <td>{m.phase}</td>
                    <td>{m.total_loss?.toFixed(4)}</td>
                    <td>{m.policy_loss?.toFixed(4)}</td>
                    <td>{m.value_loss?.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <p>No metrics available yet. Start training to generate metrics.</p>
      )}
    </div>
  );

  const renderGames = () => (
    <div className="tab-content">
      <h2>Recorded Self-Play Games</h2>
      <button onClick={fetchRecordedGames} className="btn btn-primary">Load Recorded Games</button>

      {recordedGames.length > 0 ? (
        <div className="games-list">
          <table>
            <thead>
              <tr>
                <th>File</th>
                <th>Checkpoint Episode</th>
                <th>Number of Games</th>
                <th>Timestamp</th>
              </tr>
            </thead>
            <tbody>
              {recordedGames.map((gameFile, idx) => (
                <tr key={idx}>
                  <td>{gameFile.filename}</td>
                  <td>{gameFile.checkpoint_episode}</td>
                  <td>{gameFile.num_games}</td>
                  <td>{gameFile.timestamp ? new Date(gameFile.timestamp).toLocaleString() : 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p>No recorded games available. Start training to record games.</p>
      )}
    </div>
  );

  return (
    <div className="training-dashboard">
      <h1>AlphaGo Zero Training Dashboard</h1>

      <div className="tabs">
        <button
          className={activeTab === 'status' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('status')}
        >
          Training Status
        </button>
        <button
          className={activeTab === 'architecture' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('architecture')}
        >
          Architecture
        </button>
        <button
          className={activeTab === 'checkpoints' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('checkpoints')}
        >
          Checkpoints
        </button>
        <button
          className={activeTab === 'metrics' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('metrics')}
        >
          Metrics
        </button>
        <button
          className={activeTab === 'games' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('games')}
        >
          Recorded Games
        </button>
      </div>

      <div className="tab-container">
        {activeTab === 'status' && renderStatus()}
        {activeTab === 'architecture' && renderArchitecture()}
        {activeTab === 'checkpoints' && renderCheckpoints()}
        {activeTab === 'metrics' && renderMetrics()}
        {activeTab === 'games' && renderGames()}
      </div>
    </div>
  );
};

export default TrainingDashboard;
