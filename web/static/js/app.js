// Rhythm Tatsujin web client.
// Keep this file deliberately boring: every visible control maps to a real route or local UI action.

class RhythmTatsujinApp {
  constructor() {
    this.apiBase = '/api';
    this.socket = null;
    this.currentUpload = null;
    this.currentEvaluationChart = null;
    this.skippedEvaluationIds = new Set();
    this.charts = [];
    this.filters = { search: '', difficulty: '', sort: 'recent' };
  }

  async init() {
    this.setupNavigation();
    this.setupGeneration();
    this.setupLibrary();
    this.setupTraining();
    this.setupEvaluation();
    this.setupDashboard();
    this.setupResearch();
    this.setupSocket();

    await Promise.all([
      this.checkBackend(),
      this.loadDashboard(),
      this.loadCharts(),
    ]);
  }

  authToken() {
    return (
      localStorage.getItem('rhythm_tatsujin_token') ||
      localStorage.getItem('taikonation_token') ||
      ''
    );
  }

  async request(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const token = this.authToken();
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    return fetch(`${this.apiBase}${path}`, { ...options, headers });
  }

  async responseError(response, fallback) {
    const payload = await response.json().catch(() => ({}));
    return payload.detail || payload.error || payload.message || fallback || `HTTP ${response.status}`;
  }

  setupNavigation() {
    document.querySelectorAll('.nav-item[data-page]').forEach((button) => {
      button.addEventListener('click', () => this.showPage(button.dataset.page));
    });

    document.querySelectorAll('[data-go]').forEach((button) => {
      button.addEventListener('click', () => this.showPage(button.dataset.go));
    });
  }

  showPage(page) {
    document.querySelectorAll('.page-content').forEach((section) => {
      section.classList.toggle('hidden', section.id !== `${page}-page`);
    });

    document.querySelectorAll('.nav-item[data-page]').forEach((button) => {
      const active = button.dataset.page === page;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page');
      else button.removeAttribute('aria-current');
    });

    const titles = {
      dashboard: 'Dashboard',
      generation: 'Generate',
      library: 'Library',
      evaluation: 'Evaluate',
      training: 'Research Training',
      research: 'Research',
    };
    document.getElementById('pageTitle').textContent = titles[page] || 'Rhythm Tatsujin';

    if (page === 'library') this.loadCharts();
    if (page === 'research') this.loadResearch();
  }

  async checkBackend() {
    try {
      const response = await this.request('/status');
      if (!response.ok) throw new Error(await this.responseError(response, 'Backend unavailable'));
      this.setConnectionStatus('connected', 'Connected');
    } catch (error) {
      console.error(error);
      this.setConnectionStatus('error', 'Backend unavailable');
    }
  }

  setConnectionStatus(state, label) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    if (dot) dot.className = `status-dot ${state}`;
    if (text) text.textContent = label;
  }

  setupSocket() {
    if (typeof io === 'undefined') {
      this.setConnectionStatus('error', 'Realtime unavailable');
      return;
    }

    this.socket = io({
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionAttempts: 5,
    });

    this.socket.on('connect', () => this.setConnectionStatus('connected', 'Connected'));
    this.socket.on('disconnect', () => this.setConnectionStatus('disconnected', 'Reconnecting…'));

    this.socket.on('system_log', (log) => {
      this.addSystemLog(log.level || 'info', log.message || '', log.timestamp);
    });

    this.socket.on('generation_progress', (data) => {
      this.updateGenerationProgress(data.progress || 0, data.message || '');
      if (data.cancelled) {
        this.showToast('Generation cancelled', 'warning');
        this.showStep('step-configure');
      }
    });

    this.socket.on('generation_failed', (data) => {
      this.showToast(`Generation failed: ${data.error || 'unknown error'}`, 'error');
      this.showStep('step-configure');
    });

    this.socket.on('chart_generated', (data) => {
      this.onChartGenerated(data.chart);
    });

    this.socket.on('training_progress', (data) => {
      if (data.cancelled) {
        this.setTrainingVisible(false);
        this.showToast('Training cancelled', 'warning');
        return;
      }
      this.updateTrainingProgress(data.progress || 0, data.metrics || {});
      if ((data.progress || 0) >= 100) {
        this.setTrainingVisible(false);
        this.showToast('Training completed', 'success');
      }
    });

    this.socket.on('training_failed', (data) => {
      this.setTrainingVisible(false);
      this.showToast(`Training failed: ${data.error || 'unknown error'}`, 'error');
    });
  }

  setupDashboard() {
    document.getElementById('refresh-status')?.addEventListener('click', async () => {
      await Promise.all([this.checkBackend(), this.loadDashboard()]);
    });
  }

  async loadDashboard() {
    try {
      const response = await this.request('/dashboard');
      if (!response.ok) throw new Error(await this.responseError(response, 'Dashboard unavailable'));
      const data = await response.json();
      const metrics = data.metrics || {};

      this.setText('generatedCharts', metrics.generated_charts ?? 0);
      this.setText('ratedCharts', metrics.rated_charts ?? 0);
      this.setText(
        'avgRating',
        metrics.avg_rating == null ? '—' : Number(metrics.avg_rating).toFixed(1),
      );
      this.renderSystemLogs(data.recent_logs || []);
    } catch (error) {
      console.error('Dashboard load failed:', error);
    }
  }

  renderSystemLogs(logs) {
    const container = document.getElementById('systemLogs');
    if (!container) return;
    container.innerHTML = '';
    if (!logs.length) {
      this.addSystemLog('info', 'No runtime events yet');
      return;
    }
    [...logs].reverse().forEach((log) => {
      this.addSystemLog(log.level || 'info', log.message || '', log.timestamp, false);
    });
  }

  addSystemLog(level, message, timestamp = null, prepend = true) {
    const container = document.getElementById('systemLogs');
    if (!container) return;

    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const time = document.createElement('span');
    time.className = 'log-timestamp';
    const parsed = timestamp ? new Date(timestamp) : new Date();
    time.textContent = Number.isNaN(parsed.getTime())
      ? '[unknown time]'
      : `[${parsed.toLocaleTimeString()}]`;

    const body = document.createElement('span');
    body.className = `log-level-${level}`;
    body.textContent = message;

    entry.append(time, body);
    if (prepend) container.prepend(entry);
    else container.append(entry);

    while (container.children.length > 50) container.lastElementChild?.remove();
  }

  setupGeneration() {
    const zone = document.getElementById('upload-zone');
    const input = document.getElementById('audio-input');

    zone?.addEventListener('dragover', (event) => {
      event.preventDefault();
      zone.classList.add('drag-over');
    });
    zone?.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone?.addEventListener('drop', (event) => {
      event.preventDefault();
      zone.classList.remove('drag-over');
      const file = event.dataTransfer?.files?.[0];
      if (file) this.uploadAudio(file);
    });
    zone?.addEventListener('click', (event) => {
      if (event.target !== input) input?.click();
    });
    input?.addEventListener('change', () => {
      const file = input.files?.[0];
      if (file) this.uploadAudio(file);
    });

    document.querySelectorAll('.difficulty-btn').forEach((button) => {
      button.addEventListener('click', () => {
        document.querySelectorAll('.difficulty-btn').forEach((item) => item.classList.remove('selected'));
        button.classList.add('selected');
      });
    });

    document.getElementById('btn-generate')?.addEventListener('click', () => this.generateChart());
    document.getElementById('btn-change-audio')?.addEventListener('click', () => this.resetGeneration());
  }

  validAudioFile(file) {
    const allowed = ['.mp3', '.wav', '.ogg', '.flac', '.m4a', '.aac'];
    const lower = file.name.toLowerCase();
    return allowed.some((extension) => lower.endsWith(extension));
  }

  async uploadAudio(file) {
    if (!this.validAudioFile(file)) {
      this.showToast('Unsupported audio file type', 'error');
      return;
    }
    if (file.size > 100 * 1024 * 1024) {
      this.showToast('Audio file is larger than 100 MB', 'error');
      return;
    }

    this.toggleUploadProgress(true);
    try {
      const form = new FormData();
      form.append('audio', file);
      const response = await this.request('/upload-audio', { method: 'POST', body: form });
      if (!response.ok) throw new Error(await this.responseError(response, 'Upload failed'));

      const result = await response.json();
      this.currentUpload = result;
      this.populateUpload(result);
      this.showStep('step-configure');
    } catch (error) {
      this.showToast(error.message || 'Upload failed', 'error');
    } finally {
      this.toggleUploadProgress(false);
    }
  }

  toggleUploadProgress(show) {
    const progress = document.getElementById('upload-progress');
    const content = document.querySelector('#upload-zone .upload-content');
    if (progress) progress.style.display = show ? 'block' : 'none';
    if (content) content.style.display = show ? 'none' : '';
  }

  populateUpload(data) {
    this.setText('detected-title', data.title || 'Unknown');
    this.setText('detected-duration', data.duration ? `${Math.round(data.duration)}s` : 'Unknown');
    this.setText('detected-bpm', data.detected_bpm ? Number(data.detected_bpm).toFixed(1) : 'Unknown');
    this.setText('features-status', data.ready_for_generation ? 'Ready' : 'Unavailable');

    const title = document.getElementById('chart-title');
    const bpm = document.getElementById('chart-bpm');
    if (title) title.value = data.title || '';
    if (bpm) {
      bpm.value = '';
      bpm.placeholder = data.detected_bpm ? `Auto: ${Math.round(data.detected_bpm)}` : 'Auto';
    }
  }

  showStep(stepId) {
    document.querySelectorAll('.upload-step').forEach((step) => {
      step.classList.toggle('active', step.id === stepId);
    });
  }

  resetGeneration() {
    this.currentUpload = null;
    const input = document.getElementById('audio-input');
    if (input) input.value = '';
    this.showStep('step-upload');
    this.updateGenerationProgress(0, '');
  }

  async generateChart() {
    if (!this.currentUpload) {
      this.showToast('Upload an audio file first', 'warning');
      this.showStep('step-upload');
      return;
    }

    const bpmRaw = document.getElementById('chart-bpm')?.value.trim() || '';
    const payload = {
      audio_filename: this.currentUpload.filename,
      title: document.getElementById('chart-title')?.value.trim() || 'Untitled',
      artist: document.getElementById('chart-artist')?.value.trim() || 'Unknown',
      difficulty: document.querySelector('.difficulty-btn.selected')?.dataset.difficulty || 'oni',
      bpm: bpmRaw ? Number(bpmRaw) : null,
    };

    this.showStep('step-generate');
    this.updateGenerationProgress(1, 'Starting generation…');

    try {
      const response = await this.request('/generate-chart', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await this.responseError(response, 'Generation failed'));
    } catch (error) {
      this.showToast(error.message || 'Generation failed', 'error');
      this.showStep('step-configure');
    }
  }

  updateGenerationProgress(progress, message) {
    const value = Math.max(0, Math.min(100, Number(progress) || 0));
    const bar = document.getElementById('generation-progress');
    const text = document.getElementById('generation-text');
    if (bar) bar.style.width = `${value}%`;
    if (text && message) text.textContent = message;
  }

  async onChartGenerated(chart) {
    if (chart) {
      const index = this.charts.findIndex((item) => item.id === chart.id);
      if (index >= 0) this.charts[index] = chart;
      else this.charts.unshift(chart);
    }
    this.renderCharts();
    this.updateLibraryStats();
    await this.loadDashboard();
    this.showToast(`Generated “${chart?.title || 'chart'}”`, 'success', {
      action: chart?.id ? { text: 'Download', callback: () => this.downloadChart(chart.id) } : null,
    });
    this.resetGeneration();
    this.showPage('library');
  }

  setupLibrary() {
    document.getElementById('chart-search')?.addEventListener('input', (event) => {
      this.filters.search = event.target.value || '';
      this.renderCharts();
    });
    document.getElementById('chart-difficulty-filter')?.addEventListener('change', (event) => {
      this.filters.difficulty = event.target.value || '';
      this.renderCharts();
    });
    document.getElementById('chart-sort')?.addEventListener('change', (event) => {
      this.filters.sort = event.target.value || 'recent';
      this.renderCharts();
    });
  }

  async loadCharts() {
    try {
      const response = await this.request('/charts');
      if (!response.ok) throw new Error(await this.responseError(response, 'Could not load charts'));
      const data = await response.json();
      this.charts = data.charts || [];
      this.renderCharts();
      this.updateLibraryStats();
    } catch (error) {
      console.error('Chart library load failed:', error);
    }
  }

  updateLibraryStats() {
    this.setText('total-charts', this.charts.length);
    const rated = this.charts.filter((chart) => Number(chart.rating) > 0);
    const average = rated.length
      ? rated.reduce((sum, chart) => sum + Number(chart.rating || 0), 0) / rated.length
      : null;
    this.setText('avg-rating', average == null ? '—' : average.toFixed(1));
  }

  filteredCharts() {
    const aliases = { easy: 'kantan', normal: 'futsuu', hard: 'muzukashii' };
    const search = this.filters.search.toLowerCase();

    const result = this.charts.filter((chart) => {
      const title = String(chart.title || '').toLowerCase();
      const artist = String(chart.artist || '').toLowerCase();
      const difficulty = aliases[chart.difficulty] || chart.difficulty || '';
      const matchesSearch = !search || title.includes(search) || artist.includes(search);
      const matchesDifficulty = !this.filters.difficulty || difficulty === this.filters.difficulty;
      return matchesSearch && matchesDifficulty;
    });

    result.sort((a, b) => {
      if (this.filters.sort === 'rating') return Number(b.rating || 0) - Number(a.rating || 0);
      if (this.filters.sort === 'title') return String(a.title || '').localeCompare(String(b.title || ''));
      if (this.filters.sort === 'difficulty') {
        const order = ['easy', 'kantan', 'normal', 'futsuu', 'hard', 'muzukashii', 'oni', 'ura'];
        return order.indexOf(a.difficulty) - order.indexOf(b.difficulty);
      }
      return new Date(b.created_at || 0) - new Date(a.created_at || 0);
    });

    return result;
  }

  renderCharts() {
    const grid = document.getElementById('charts-grid');
    const empty = document.getElementById('charts-empty');
    const template = document.getElementById('chart-card-template');
    if (!grid || !empty || !template) return;

    const charts = this.filteredCharts();
    grid.innerHTML = '';
    grid.style.display = charts.length ? 'grid' : 'none';
    empty.style.display = charts.length ? 'none' : 'block';

    charts.forEach((chart) => {
      const fragment = template.content.cloneNode(true);
      const card = fragment.querySelector('.chart-card');
      card.dataset.chartId = chart.id;

      fragment.querySelector('.chart-title').textContent = chart.title || 'Untitled';
      fragment.querySelector('.chart-artist').textContent = chart.artist || 'Unknown artist';
      fragment.querySelector('.chart-bpm').textContent = chart.bpm ? `${Number(chart.bpm).toFixed(1)} BPM` : 'BPM auto';
      fragment.querySelector('.chart-created').textContent = this.formatDate(chart.created_at);

      const diffClass = { easy: 'kantan', normal: 'futsuu', hard: 'muzukashii' }[chart.difficulty] || chart.difficulty || 'oni';
      const diffColor = fragment.querySelector('.difficulty-color');
      diffColor.className = `difficulty-color ${diffClass}`;
      fragment.querySelector('.difficulty-text').textContent = this.difficultyName(chart.difficulty);

      const rating = Number(chart.rating || 0);
      fragment.querySelector('.chart-rating').textContent = rating > 0 ? rating.toFixed(1) : 'Unrated';

      this.renderDensityPreview(fragment.querySelector('.chart-waveform'), chart);

      fragment.querySelector('.preview-btn').addEventListener('click', () => {
        window.open(`/editor?id=${chart.id}`, '_blank');
      });
      fragment.querySelector('.download-btn').addEventListener('click', () => this.downloadChart(chart.id, 'osz'));
      fragment.querySelector('.download-osu-btn').addEventListener('click', () => this.downloadChart(chart.id, 'osu'));
      fragment.querySelector('.rate-btn').addEventListener('click', () => this.quickRate(chart));

      grid.appendChild(fragment);
    });
  }

  async renderDensityPreview(container, chart) {
    if (!container || !chart?.id) return;
    container.textContent = '································';
    try {
      const response = await this.request(`/chart-data?id=${encodeURIComponent(chart.id)}`);
      if (!response.ok) throw new Error('preview unavailable');
      const data = await response.json();
      const notes = data.notes || [];
      if (!notes.length) {
        container.textContent = '— no hit notes —';
        return;
      }

      const bins = new Array(32).fill(0);
      const end = Math.max(...notes.map((note) => Number(note.time) || 0), 1);
      notes.forEach((note) => {
        const index = Math.min(31, Math.floor(((Number(note.time) || 0) / end) * 32));
        bins[index] += 1;
      });
      const max = Math.max(...bins, 1);
      const blocks = '▁▂▃▄▅▆▇█';
      container.textContent = bins.map((count) => {
        const level = count ? Math.min(7, Math.ceil((count / max) * 7)) : 0;
        return blocks[level];
      }).join('');
      container.title = `${notes.length} playable hit notes`;
    } catch {
      container.textContent = '— preview unavailable —';
    }
  }

  async downloadChart(chartId, format = 'osz') {
    const url = new URL('/api/download-chart', window.location.origin);
    url.searchParams.set('id', chartId);
    url.searchParams.set('format', format);
    const token = this.authToken();
    if (token) url.searchParams.set('api_token', token);

    const link = document.createElement('a');
    link.href = url.toString();
    link.download = '';
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  quickRate(chart) {
    const raw = prompt(`Rate “${chart.title || 'chart'}” from 1–5:`, '4');
    if (raw == null) return;
    const rating = Number.parseInt(raw, 10);
    if (!Number.isInteger(rating) || rating < 1 || rating > 5) {
      this.showToast('Rating must be a whole number from 1 to 5', 'warning');
      return;
    }

    this.request('/submit-rating', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ chart_id: chart.id, rating }),
    }).then(async (response) => {
      if (!response.ok) throw new Error(await this.responseError(response, 'Rating failed'));
      chart.rating = rating;
      this.renderCharts();
      this.updateLibraryStats();
      this.loadDashboard();
    }).catch((error) => this.showToast(error.message, 'error'));
  }

  setupEvaluation() {
    document.getElementById('start-evaluation')?.addEventListener('click', () => this.loadEvaluationChart());
    document.getElementById('submit-evaluation')?.addEventListener('click', () => this.submitEvaluation());
    document.getElementById('skip-evaluation')?.addEventListener('click', () => this.skipEvaluation());
    document.getElementById('edit-evaluation-chart')?.addEventListener('click', () => {
      if (this.currentEvaluationChart?.id) {
        window.open(`/editor?id=${this.currentEvaluationChart.id}`, '_blank');
      }
    });
  }

  async loadEvaluationChart() {
    try {
      const response = await this.request('/charts?limit=500');
      if (!response.ok) throw new Error(await this.responseError(response, 'Could not load charts'));
      const data = await response.json();
      const chart = (data.charts || []).find(
        (item) => Number(item.rating || 0) <= 0 && !this.skippedEvaluationIds.has(item.id),
      );
      if (!chart) {
        document.getElementById('evaluationForm').style.display = 'none';
        this.showToast('No unrated charts remain in this session', 'info');
        return;
      }

      this.currentEvaluationChart = chart;
      const form = document.getElementById('evaluationForm');
      form.style.display = 'block';
      form.dataset.chartId = chart.id;
      document.getElementById('evaluationChartTitle').textContent =
        `${chart.title || 'Untitled'} — ${this.difficultyName(chart.difficulty)}`;
    } catch (error) {
      this.showToast(error.message || 'Could not load an evaluation chart', 'error');
    }
  }

  async submitEvaluation() {
    const form = document.getElementById('evaluationForm');
    if (!form || !this.currentEvaluationChart) return;
    const values = new FormData(form);

    const payload = {
      chart_id: Number(this.currentEvaluationChart.id),
      fun: Number(values.get('fun')),
      musicality: Number(values.get('musicality')),
      playability: Number(values.get('playability')),
      coherence: Number(values.get('coherence')),
      comments: String(values.get('comments') || ''),
    };

    try {
      const response = await this.request('/submit-evaluation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await this.responseError(response, 'Evaluation failed'));

      this.showToast('Evaluation saved', 'success');
      this.currentEvaluationChart = null;
      await Promise.all([this.loadCharts(), this.loadDashboard()]);
      await this.loadEvaluationChart();
    } catch (error) {
      this.showToast(error.message || 'Evaluation failed', 'error');
    }
  }

  async skipEvaluation() {
    if (this.currentEvaluationChart?.id) {
      this.skippedEvaluationIds.add(this.currentEvaluationChart.id);
    }
    this.currentEvaluationChart = null;
    await this.loadEvaluationChart();
  }

  setupTraining() {
    document.getElementById('start-training')?.addEventListener('click', () => this.startTraining());
    document.getElementById('stop-training')?.addEventListener('click', () => this.stopTraining());
    document.getElementById('load-training-config')?.addEventListener('click', () => this.loadTrainingConfig(true));
    document.getElementById('save-training-config')?.addEventListener('click', () => this.saveTrainingConfig());
  }

  trainingValues() {
    const form = document.getElementById('trainingForm');
    const values = new FormData(form);
    return {
      d_model: Number(values.get('d_model')),
      nhead: Number(values.get('nhead')),
      num_encoder_layers: Number(values.get('num_encoder_layers')),
      num_decoder_layers: Number(values.get('num_decoder_layers')),
      learning_rate: Number(values.get('learning_rate')),
      batch_size: Number(values.get('batch_size')),
    };
  }

  async startTraining() {
    const payload = this.trainingValues();
    try {
      const response = await this.request('/start-training', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await this.responseError(response, 'Training failed to start'));
      this.setTrainingVisible(true);
      this.updateTrainingProgress(0, {});
      this.showToast('Training queued', 'info');
    } catch (error) {
      this.showToast(error.message || 'Training failed to start', 'error');
    }
  }

  async stopTraining() {
    try {
      const response = await this.request('/stop-training', { method: 'POST' });
      if (!response.ok) throw new Error(await this.responseError(response, 'Could not stop training'));
      this.showToast('Training stop requested', 'info');
    } catch (error) {
      this.showToast(error.message || 'Could not stop training', 'error');
    }
  }

  setTrainingVisible(show) {
    const section = document.getElementById('trainingProgress');
    if (section) section.style.display = show ? 'block' : 'none';
  }

  updateTrainingProgress(progress, metrics) {
    this.setTrainingVisible(true);
    const value = Math.max(0, Math.min(100, Number(progress) || 0));
    const bar = document.getElementById('trainingProgressBar');
    const text = document.getElementById('trainingProgressText');
    if (bar) bar.style.width = `${value}%`;
    if (text) text.textContent = `${value}% complete`;

    const container = document.getElementById('trainingMetrics');
    if (!container) return;
    container.innerHTML = '';
    Object.entries(metrics || {}).forEach(([key, metric]) => {
      if (metric == null) return;
      const item = document.createElement('div');
      item.className = 'training-metric';
      const valueNode = document.createElement('div');
      valueNode.className = 'training-metric-value';
      valueNode.textContent = typeof metric === 'number' ? metric.toFixed(4) : String(metric);
      const label = document.createElement('div');
      label.className = 'training-metric-label';
      label.textContent = key.replaceAll('_', ' ');
      item.append(valueNode, label);
      container.appendChild(item);
    });
  }

  async loadTrainingConfig(notify = false) {
    try {
      const response = await this.request('/config');
      if (!response.ok) throw new Error(await this.responseError(response, 'Could not load config'));
      const config = await response.json();
      const form = document.getElementById('trainingForm');
      const fields = {
        d_model: config.model?.d_model,
        nhead: config.model?.nhead,
        num_encoder_layers: config.model?.num_encoder_layers,
        num_decoder_layers: config.model?.num_decoder_layers,
        learning_rate: config.training?.learning_rate,
        batch_size: config.training?.batch_size,
      };
      Object.entries(fields).forEach(([name, value]) => {
        if (value == null) return;
        const input = form?.elements?.namedItem(name);
        if (input) input.value = value;
      });
      if (notify) this.showToast('Saved training config reloaded', 'success');
    } catch (error) {
      if (notify) this.showToast(error.message || 'Could not load config', 'error');
    }
  }

  async saveTrainingConfig() {
    const values = this.trainingValues();
    const payload = {
      config: {
        model: {
          d_model: values.d_model,
          nhead: values.nhead,
          num_encoder_layers: values.num_encoder_layers,
          num_decoder_layers: values.num_decoder_layers,
        },
        training: {
          learning_rate: values.learning_rate,
          batch_size: values.batch_size,
        },
      },
    };

    try {
      const response = await this.request('/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await this.responseError(response, 'Could not save config'));
      this.showToast('Training settings saved', 'success');
    } catch (error) {
      this.showToast(error.message || 'Could not save config', 'error');
    }
  }

  setupResearch() {
    document.getElementById('export-research')?.addEventListener('click', () => {
      window.open('/api/research/export-dataset', '_blank');
    });
  }

  async loadResearch() {
    try {
      const response = await this.request('/research/experiments');
      if (!response.ok) throw new Error(await this.responseError(response, 'Research data unavailable'));
      const data = await response.json();
      this.renderResearch(data);
    } catch (error) {
      this.showToast(error.message || 'Research data unavailable', 'error');
    }
  }

  renderResearch(data) {
    const experiments = data.experiments || [];
    const runs = data.training_runs || [];
    const history = data.chart_history || {};
    const feedback = data.human_feedback_stats || {};

    this.renderList(
      'experimentHistory',
      experiments.map((experiment) => {
        const name = experiment.experiment_name || experiment.experiment_id || 'Experiment';
        const status = experiment.status || 'unknown';
        return `${name} — ${status}`;
      }),
      'No persisted experiment metadata yet.',
    );

    this.renderList(
      'trainingRuns',
      runs.map((run) => {
        const loss = run.val_loss == null ? 'no val loss' : `val loss ${Number(run.val_loss).toFixed(4)}`;
        return `${run.name || run.experiment_id || 'Run'} — ${loss}`;
      }),
      'No completed training runs recorded yet.',
    );

    const difficulties = (history.difficulties || []).reduce((counts, difficulty) => {
      const key = this.difficultyName(difficulty);
      counts[key] = (counts[key] || 0) + 1;
      return counts;
    }, {});
    const chartLines = [
      `${(history.timestamps || []).length} persisted generated charts`,
      ...Object.entries(difficulties).map(([difficulty, count]) => `${difficulty}: ${count}`),
    ];
    this.renderList('chartHistory', chartLines, 'No generated chart history yet.');

    const feedbackLines = [
      `${Number(feedback.total_evaluations || 0)} detailed evaluations`,
      `${Number(feedback.rated_charts || 0)} rated charts`,
      `Average chart rating: ${feedback.average_rating == null ? '—' : Number(feedback.average_rating).toFixed(2)}`,
    ];
    this.renderList('humanFeedbackStats', feedbackLines, 'No human feedback yet.');
  }

  renderList(id, lines, emptyText) {
    const container = document.getElementById(id);
    if (!container) return;
    container.innerHTML = '';
    const items = lines.length ? lines : [emptyText];
    const list = document.createElement('ul');
    list.className = 'research-list';
    items.forEach((line) => {
      const item = document.createElement('li');
      item.textContent = line;
      list.appendChild(item);
    });
    container.appendChild(list);
  }

  setText(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = String(value);
  }

  formatDate(value) {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  difficultyName(value) {
    return {
      easy: 'Kantan',
      kantan: 'Kantan',
      normal: 'Futsuu',
      futsuu: 'Futsuu',
      hard: 'Muzukashii',
      muzukashii: 'Muzukashii',
      oni: 'Oni',
      ura: 'Ura Oni',
    }[value] || value || 'Unknown';
  }

  showToast(message, type = 'info', options = {}) {
    let container = document.getElementById('toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'toast-container';
      container.className = 'toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const content = document.createElement('div');
    content.className = 'toast-content';

    const icon = document.createElement('div');
    icon.className = 'toast-icon';
    icon.innerHTML = this.statusIcon(type);

    const body = document.createElement('div');
    body.className = 'toast-message';
    body.textContent = message;

    content.append(icon, body);

    if (options.action) {
      const action = document.createElement('button');
      action.className = 'toast-action';
      action.type = 'button';
      action.textContent = options.action.text;
      action.addEventListener('click', () => {
        options.action.callback();
        toast.remove();
      });
      content.appendChild(action);
    }

    const close = document.createElement('button');
    close.className = 'toast-close';
    close.type = 'button';
    close.setAttribute('aria-label', 'Dismiss');
    close.textContent = '×';
    close.addEventListener('click', () => toast.remove());
    content.appendChild(close);

    toast.appendChild(content);
    container.appendChild(toast);
    setTimeout(() => toast.remove(), options.duration || 5000);
  }

  statusIcon(type) {
    const paths = {
      success: '<circle cx="12" cy="12" r="9"></circle><path d="m8.5 12 2.2 2.2 4.8-5"></path>',
      error: '<circle cx="12" cy="12" r="9"></circle><path d="m9 9 6 6M15 9l-6 6"></path>',
      warning: '<path d="M12 3 2.8 19h18.4z"></path><path d="M12 9v4M12 16h.01"></path>',
      info: '<circle cx="12" cy="12" r="9"></circle><path d="M12 11v5M12 8h.01"></path>',
    };
    return `<svg class="toast-status-icon" viewBox="0 0 24 24" aria-hidden="true">${paths[type] || paths.info}</svg>`;
  }
}

document.addEventListener('DOMContentLoaded', async () => {
  window.rhythmTatsujin = new RhythmTatsujinApp();
  await window.rhythmTatsujin.init();
});
