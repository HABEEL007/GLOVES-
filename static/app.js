// GlovesLabeler AI Studio Client Engine
class LabelerStudio {
  constructor() {
    this.images = [];
    this.currentIndex = -1;
    this.currentImage = null;
    this.currentBoxes = [];
    this.selectedBoxIndex = -1;
    this.hoverBoxIndex = -1;
    this.activeHandle = null;

    // Active Class state (Default to 1: Hand as requested!)
    this.activeClassId = 1;
    this.activeClassName = 'Hand';

    // Interaction states: 'idle', 'drawing', 'moving', 'resizing'
    this.mode = 'idle';
    this.dragStart = { x: 0, y: 0 };
    this.activeBoxInitial = null;

    // Canvas & Render
    this.canvas = document.getElementById('annotationCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.canvasWrapper = document.getElementById('canvasWrapper');
    this.scale = 1.0;
    this.zoom = 1.0;

    // Filter
    this.currentFilter = 'all';
    this.searchQuery = '';

    this.initElements();
    this.bindEvents();
    this.loadFolders();
  }

  initElements() {
    this.folderSelect = document.getElementById('folderSelect');
    this.thumbnailList = document.getElementById('thumbnailList');
    this.progressBar = document.getElementById('progressBar');
    this.progressText = document.getElementById('progressText');
    this.glovesBadge = document.getElementById('glovesBadge');
    this.negBadge = document.getElementById('negBadge');
    
    this.currentImageName = document.getElementById('currentImageName');
    this.currentImageDim = document.getElementById('currentImageDim');
    this.currentStatusBadge = document.getElementById('currentStatusBadge');
    this.zoomLevelText = document.getElementById('zoomLevelText');
    
    this.classHandBtn = document.getElementById('classHandBtn');
    this.classGlovesBtn = document.getElementById('classGlovesBtn');

    this.confSlider = document.getElementById('confSlider');
    this.confValue = document.getElementById('confValue');
    this.boxList = document.getElementById('boxList');
    this.boxCount = document.getElementById('boxCount');
    this.selectedBoxProps = document.getElementById('selectedBoxProps');
    this.deleteSelectedBtn = document.getElementById('deleteSelectedBtn');

    this.prevBtn = document.getElementById('prevBtn');
    this.nextBtn = document.getElementById('nextBtn');
    this.approveBtn = document.getElementById('approveBtn');
    this.markNegativeBtn = document.getElementById('markNegativeBtn');
    this.reDetectBtn = document.getElementById('reDetectBtn');
    this.clearAllBtn = document.getElementById('clearAllBtn');
    this.batchAiBtn = document.getElementById('batchAiBtn');
    this.trainModelBtn = document.getElementById('trainModelBtn');
    this.trainStatusBadge = document.getElementById('trainStatusBadge');
    this.toastEl = document.getElementById('toastNotification');
    this.wasTraining = false;

    // CCTV Stream Grabber Elements
    this.cctvGrabberBtn = document.getElementById('cctvGrabberBtn');
    this.cctvLiveBadge = document.getElementById('cctvLiveBadge');
    this.cctvModal = document.getElementById('cctvModal');
    this.cctvCloseBtn = document.getElementById('cctvCloseBtn');
    this.cctvModalCancelBtn = document.getElementById('cctvModalCancelBtn');
    this.cctvRtspUrl = document.getElementById('cctvRtspUrl');
    this.cctvRateSelect = document.getElementById('cctvRateSelect');
    this.cctvCategorySelect = document.getElementById('cctvCategorySelect');
    this.cctvStartBtn = document.getElementById('cctvStartBtn');
    this.cctvStopBtn = document.getElementById('cctvStopBtn');
    this.cctvStatusDot = document.getElementById('cctvStatusDot');
    this.cctvStatusText = document.getElementById('cctvStatusText');
    this.cctvElapsedTime = document.getElementById('cctvElapsedTime');
    this.cctvSavedCount = document.getElementById('cctvSavedCount');
    this.cctvIntervalVal = document.getElementById('cctvIntervalVal');
    this.cctvLastTime = document.getElementById('cctvLastTime');
    this.cctvErrorBanner = document.getElementById('cctvErrorBanner');

    // Image Upload & AI Detection Elements
    this.detectUploadBtn = document.getElementById('detectUploadBtn');
    this.detectModal = document.getElementById('detectModal');
    this.detectCloseBtn = document.getElementById('detectCloseBtn');
    this.detectModalCloseBtn = document.getElementById('detectModalCloseBtn');
    this.detectConfSlider = document.getElementById('detectConfSlider');
    this.detectConfVal = document.getElementById('detectConfVal');
    this.detectDropzone = document.getElementById('detectDropzone');
    this.detectFileInput = document.getElementById('detectFileInput');
    this.detectLoadingIndicator = document.getElementById('detectLoadingIndicator');
    this.detectResultContainer = document.getElementById('detectResultContainer');
    this.detectStatusBanner = document.getElementById('detectStatusBanner');
    this.detectStatusText = document.getElementById('detectStatusText');
    this.detectLatencyBadge = document.getElementById('detectLatencyBadge');
    this.detectResultImg = document.getElementById('detectResultImg');
    this.detectGlovesCount = document.getElementById('detectGlovesCount');
    this.detectHandCount = document.getElementById('detectHandCount');
    this.detectDetectionsList = document.getElementById('detectDetectionsList');
    this.detectSaveGlovesBtn = document.getElementById('detectSaveGlovesBtn');
    this.detectSaveNegBtn = document.getElementById('detectSaveNegBtn');
    this.detectUploadNewBtn = document.getElementById('detectUploadNewBtn');

    this.currentDetectionResult = null;
    this.currentUploadedFile = null;
  }

  bindEvents() {
    // Training Button
    if (this.trainModelBtn) {
      this.trainModelBtn.addEventListener('click', () => this.triggerTraining());
    }

    // Class Selectors
    if (this.classHandBtn) {
      this.classHandBtn.addEventListener('click', () => this.setActiveClass(1, 'Hand'));
    }
    if (this.classGlovesBtn) {
      this.classGlovesBtn.addEventListener('click', () => this.setActiveClass(0, 'Gloves'));
    }

    // Folder Switch
    this.folderSelect.addEventListener('change', (e) => {
      this.switchFolder(e.target.value);
    });

    // Search & Filter tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        this.currentFilter = e.target.dataset.filter;
        this.renderThumbnails();
      });
    });

    document.getElementById('searchBox').addEventListener('input', (e) => {
      this.searchQuery = e.target.value.toLowerCase();
      this.renderThumbnails();
    });

    // Confidence Slider
    this.confSlider.addEventListener('input', (e) => {
      this.confValue.textContent = Number(e.target.value).toFixed(2);
    });

    // Action Buttons
    this.prevBtn.addEventListener('click', () => this.navigate(-1));
    this.nextBtn.addEventListener('click', () => this.navigate(1));
    this.approveBtn.addEventListener('click', () => this.saveCurrent(false));
    this.markNegativeBtn.addEventListener('click', () => this.saveCurrent(true));
    this.reDetectBtn.addEventListener('click', () => this.reDetectAI());
    this.clearAllBtn.addEventListener('click', () => this.clearAll());
    this.deleteSelectedBtn.addEventListener('click', () => this.deleteSelected());
    this.batchAiBtn.addEventListener('click', () => this.batchAiAutoLabel());

    // Zoom Buttons
    document.getElementById('zoomInBtn').addEventListener('click', () => this.setZoom(this.zoom + 0.15));
    document.getElementById('zoomOutBtn').addEventListener('click', () => this.setZoom(this.zoom - 0.15));
    document.getElementById('zoomFitBtn').addEventListener('click', () => this.setZoom(1.0));

    // Canvas Mouse Events
    this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
    this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
    this.canvas.addEventListener('mouseleave', (e) => this.onMouseLeave(e));

    // Global Keyboard Shortcuts
    window.addEventListener('keydown', (e) => this.onKeyDown(e));
    window.addEventListener('resize', () => this.fitCanvasToScreen());
    const refreshQueueBtn = document.getElementById('refreshQueueBtn');
    if (refreshQueueBtn) {
      refreshQueueBtn.addEventListener('click', () => this.loadImages(true));
    }

    // Auto-poll every 5 seconds for new captured frames
    setInterval(() => {
      this.loadImages(true);
    }, 5000);

    // CCTV Stream Grabber Modal controls
    if (this.cctvGrabberBtn) {
      this.cctvGrabberBtn.addEventListener('click', () => this.openCctvModal());
    }
    if (this.cctvCloseBtn) {
      this.cctvCloseBtn.addEventListener('click', () => this.closeCctvModal());
    }
    if (this.cctvModalCancelBtn) {
      this.cctvModalCancelBtn.addEventListener('click', () => this.closeCctvModal());
    }
    if (this.cctvModal) {
      this.cctvModal.addEventListener('click', (e) => {
        if (e.target === this.cctvModal) this.closeCctvModal();
      });
    }
    if (this.cctvStartBtn) {
      this.cctvStartBtn.addEventListener('click', () => this.startCctvCapture());
    }
    if (this.cctvStopBtn) {
      this.cctvStopBtn.addEventListener('click', () => this.stopCctvCapture());
    }

    // Image Upload & AI Detection Modal controls
    if (this.detectUploadBtn) {
      this.detectUploadBtn.addEventListener('click', () => this.openDetectModal());
    }
    if (this.detectCloseBtn) {
      this.detectCloseBtn.addEventListener('click', () => this.closeDetectModal());
    }
    if (this.detectModalCloseBtn) {
      this.detectModalCloseBtn.addEventListener('click', () => this.closeDetectModal());
    }
    if (this.detectModal) {
      this.detectModal.addEventListener('click', (e) => {
        if (e.target === this.detectModal) this.closeDetectModal();
      });
    }
    if (this.detectUploadNewBtn) {
      this.detectUploadNewBtn.addEventListener('click', () => this.resetDetectView());
    }

    // Confidence Slider in Detection Modal
    if (this.detectConfSlider) {
      this.detectConfSlider.addEventListener('input', (e) => {
        this.detectConfVal.textContent = Number(e.target.value).toFixed(2);
        if (this.currentUploadedFile) {
          this.runUploadDetection(this.currentUploadedFile);
        }
      });
    }

    // Dropzone Events
    if (this.detectDropzone) {
      this.detectDropzone.addEventListener('click', () => {
        if (this.detectFileInput) this.detectFileInput.click();
      });

      this.detectFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.handleFileSelected(e.target.files[0]);
        }
      });

      ['dragenter', 'dragover'].forEach(eventName => {
        this.detectDropzone.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.detectDropzone.classList.add('dragover');
        });
      });

      ['dragleave', 'drop'].forEach(eventName => {
        this.detectDropzone.addEventListener(eventName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.detectDropzone.classList.remove('dragover');
        });
      });

      this.detectDropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
          this.handleFileSelected(dt.files[0]);
        }
      });
    }

    // Paste support (Ctrl+V) when modal is open
    window.addEventListener('paste', (e) => {
      if (this.detectModal && this.detectModal.style.display !== 'none') {
        const items = (e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData))?.items;
        if (items) {
          for (let i = 0; i < items.length; i++) {
            if (items[i].type.indexOf('image') !== -1) {
              const blob = items[i].getAsFile();
              this.handleFileSelected(blob);
              break;
            }
          }
        }
      }
    });

    // Save to Dataset
    if (this.detectSaveGlovesBtn) {
      this.detectSaveGlovesBtn.addEventListener('click', () => this.saveDetectedToDataset('gloves'));
    }
    if (this.detectSaveNegBtn) {
      this.detectSaveNegBtn.addEventListener('click', () => this.saveDetectedToDataset('negatives'));
    }

    // Auto-poll training status
    this.checkTrainingStatus();
    setInterval(() => {
      this.checkTrainingStatus();
    }, 3500);

    // Auto-poll CCTV Stream status
    this.checkCctvStatus();
    setInterval(() => {
      this.checkCctvStatus();
    }, 2500);
  }

  setActiveClass(id, name) {
    this.activeClassId = id;
    this.activeClassName = name;

    if (this.classHandBtn && this.classGlovesBtn) {
      this.classHandBtn.classList.toggle('active', id === 1);
      this.classGlovesBtn.classList.toggle('active', id === 0);
    }

    // If a box is selected, switch its class immediately!
    if (this.selectedBoxIndex >= 0 && this.selectedBoxIndex < this.currentBoxes.length) {
      this.currentBoxes[this.selectedBoxIndex].class_id = id;
      this.currentBoxes[this.selectedBoxIndex].class_name = name;
      this.renderInspector();
      this.draw();
    }
  }

  // =========================================================================
  // API Calls
  // =========================================================================
  async loadFolders() {
    try {
      const res = await fetch('/api/folders');
      const data = await res.json();
      this.folderSelect.innerHTML = '';
      data.folders.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.path;
        opt.textContent = `${f.path} (${f.image_count} imgs)`;
        if (f.path === data.current_images) {
          opt.selected = true;
        }
        this.folderSelect.appendChild(opt);
      });
      this.loadImages();
    } catch (err) {
      console.error('Failed to load folders:', err);
    }
  }

  async switchFolder(path) {
    try {
      const res = await fetch('/api/set_folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path })
      });
      if (res.ok) {
        this.loadImages(false);
      }
    } catch (err) {
      console.error('Failed to switch folder:', err);
    }
  }

  async loadImages(preserveSelection = false) {
    try {
      const prevSelectedName = (this.currentIndex >= 0 && this.currentIndex < this.images.length) 
        ? this.images[this.currentIndex].name 
        : null;

      const res = await fetch('/api/images');
      const data = await res.json();
      this.images = data.images || [];
      this.updateStats();
      this.renderThumbnails();

      if (this.images.length > 0) {
        if (preserveSelection && prevSelectedName) {
          const newIdx = this.images.findIndex(img => img.name === prevSelectedName);
          if (newIdx >= 0) {
            this.currentIndex = newIdx;
            document.querySelectorAll('.thumb-item').forEach(el => {
              el.classList.toggle('active', el.dataset.name === prevSelectedName);
            });
            return;
          }
        }
        if (!preserveSelection || this.currentIndex < 0) {
          this.selectImage(0);
        }
      } else {
        this.clearCanvas();
      }
    } catch (err) {
      console.error('Failed to load images:', err);
    }
  }

  async updateStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      this.progressText.textContent = `${data.reviewed} / ${data.total} Reviewed (${data.percent}%)`;
      this.progressBar.style.width = `${data.percent}%`;
      this.glovesBadge.textContent = `${data.total_gloves} Objects`;
      this.negBadge.textContent = `${data.negatives} Negatives`;
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  }

  async selectImage(index) {
    if (index < 0 || index >= this.images.length) return;
    this.currentIndex = index;
    const item = this.images[index];

    this.currentImageName.textContent = item.name;
    this.currentStatusBadge.textContent = item.status.toUpperCase();
    this.currentStatusBadge.className = `status-indicator ${item.status}`;

    // Highlight thumbnail
    document.querySelectorAll('.thumb-item').forEach((el, idx) => {
      el.classList.toggle('active', el.dataset.name === item.name);
    });

    // Load Image Object
    const img = new Image();
    img.src = `/api/image/${encodeURIComponent(item.name)}`;
    img.onload = async () => {
      this.currentImage = img;
      this.currentImageDim.textContent = `${img.naturalWidth} × ${img.naturalHeight}px`;
      this.fitCanvasToScreen();
      await this.loadBoxesForImage(item.name);
    };
  }

  async loadBoxesForImage(filename, forceAi = false) {
    try {
      const conf = this.confSlider.value;
      const res = await fetch(`/api/boxes/${encodeURIComponent(filename)}?conf=${conf}&force_ai=${forceAi}&class_id=${this.activeClassId}`);
      const data = await res.json();
      this.currentBoxes = (data.boxes || []).map((b, idx) => ({
        ...b,
        id: idx + 1,
        class_id: b.class_id !== undefined ? b.class_id : this.activeClassId,
        class_name: b.class_name || (b.class_id === 0 ? 'Gloves' : 'Hand')
      }));
      this.selectedBoxIndex = -1;
      this.renderInspector();
      this.draw();
    } catch (err) {
      console.error('Error loading boxes:', err);
    }
  }

  async saveCurrent(isNegative = false) {
    if (this.currentIndex < 0 || !this.currentImage) return;
    const item = this.images[this.currentIndex];

    const boxesToSave = isNegative ? [] : this.currentBoxes;

    try {
      const res = await fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: item.name,
          boxes: boxesToSave,
          is_negative: isNegative
        })
      });

      if (res.ok) {
        item.is_reviewed = true;
        item.status = 'reviewed';
        item.box_count = boxesToSave.length;
        this.updateStats();
        this.renderThumbnails();

        // Advance to next image smoothly
        this.navigate(1);
      }
    } catch (err) {
      console.error('Error saving boxes:', err);
    }
  }

  async reDetectAI() {
    if (this.currentIndex < 0) return;
    const item = this.images[this.currentIndex];
    const origHTML = this.reDetectBtn.innerHTML;
    try {
      this.reDetectBtn.disabled = true;
      this.reDetectBtn.innerHTML = `⏳ Detecting ${this.activeClassName}s...`;
      await this.loadBoxesForImage(item.name, true);
    } catch (err) {
      console.error('Error in reDetectAI:', err);
      alert('AI Detection error: ' + err.message);
    } finally {
      this.reDetectBtn.disabled = false;
      this.reDetectBtn.innerHTML = origHTML;
    }
  }

  showToast(msg, isSuccess = true, duration = 4500) {
    if (!this.toastEl) return;
    this.toastEl.textContent = msg;
    this.toastEl.style.display = 'block';
    this.toastEl.style.borderColor = isSuccess ? '#10b981' : '#ef4444';
    setTimeout(() => {
      this.toastEl.style.display = 'none';
    }, duration);
  }

  async triggerTraining() {
    if (!confirm("Start background retraining on all reviewed images (5 epochs on GPU)? You can continue labeling without interruption!")) return;
    try {
      if (this.trainModelBtn) {
        this.trainModelBtn.disabled = true;
        this.trainModelBtn.innerHTML = '⏳ Initializing...';
      }
      const res = await fetch('/api/trigger_train', { method: 'POST' });
      const data = await res.json();
      this.showToast(data.message || "Training initiated!");
      this.checkTrainingStatus();
    } catch (err) {
      console.error("Trigger training error:", err);
      this.showToast("Failed to start training: " + err.message, false);
    } finally {
      if (this.trainModelBtn) {
        this.trainModelBtn.disabled = false;
        this.trainModelBtn.innerHTML = '🚀 Train Model Now';
      }
    }
  }

  async checkTrainingStatus() {
    try {
      const res = await fetch('/api/training_status');
      const data = await res.json();
      const isTraining = data.is_training;

      if (this.trainStatusBadge) {
        if (isTraining) {
          const ep = data.current_epoch || 0;
          const tot = data.total_epochs || 5;
          this.trainStatusBadge.style.display = 'inline-block';
          this.trainStatusBadge.textContent = `⚡ Training: Epoch ${ep}/${tot}`;
          if (this.trainModelBtn) {
            this.trainModelBtn.disabled = true;
            this.trainModelBtn.style.opacity = '0.6';
          }
        } else {
          this.trainStatusBadge.style.display = 'none';
          if (this.trainModelBtn) {
            this.trainModelBtn.disabled = false;
            this.trainModelBtn.style.opacity = '1.0';
          }
        }
      }

      if (this.wasTraining && !isTraining) {
        // Training just finished!
        this.showToast('🎉 Model retrained & updated! AI predictions are now smarter.');
      }
      this.wasTraining = isTraining;
    } catch (err) {
      // Silent catch for background poll
    }
  }

  async batchAiAutoLabel() {
    if (!confirm(`Run AI Auto-Annotation for all images as [${this.activeClassName}]?`)) return;
    try {
      const conf = this.confSlider.value;
      const res = await fetch('/api/batch_auto_label', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conf, class_id: this.activeClassId })
      });
      const data = await res.json();
      alert(data.message || 'Batch auto-labeling started!');
      setTimeout(() => this.loadImages(), 2000);
    } catch (err) {
      console.error('Batch auto label error:', err);
    }
  }

  // =========================================================================
  // CCTV Live RTSP Stream Grabber
  // =========================================================================
  openCctvModal() {
    if (this.cctvModal) {
      this.cctvModal.style.display = 'flex';
      this.checkCctvStatus();
    }
  }

  closeCctvModal() {
    if (this.cctvModal) {
      this.cctvModal.style.display = 'none';
    }
  }

  async startCctvCapture() {
    const rtspUrl = this.cctvRtspUrl ? this.cctvRtspUrl.value.trim() : '';
    if (!rtspUrl) {
      alert("Please provide an RTSP stream URL.");
      return;
    }

    const fpm = this.cctvRateSelect ? (parseInt(this.cctvRateSelect.value) || 15) : 15;
    const category = this.cctvCategorySelect ? this.cctvCategorySelect.value : 'negatives';

    if (this.cctvStartBtn) {
      this.cctvStartBtn.disabled = true;
      this.cctvStartBtn.innerHTML = "⏳ Connecting to RTSP...";
    }

    try {
      const res = await fetch('/api/cctv/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rtsp_url: rtspUrl,
          frames_per_minute: fpm,
          category: category
        })
      });
      const data = await res.json();
      if (data.success) {
        this.showToast(`📹 RTSP Frame Capture started! Rate: ${fpm} frames/min (every ${(60.0/fpm).toFixed(1)}s)`);
      } else {
        this.showToast(`Capture failed: ${data.message}`, false);
      }
      this.checkCctvStatus();
    } catch (err) {
      console.error("Start CCTV capture error:", err);
      this.showToast("Failed to start RTSP capture: " + err.message, false);
    } finally {
      if (this.cctvStartBtn) {
        this.cctvStartBtn.disabled = false;
        this.cctvStartBtn.innerHTML = "▶ Start Capturing Frames";
      }
    }
  }

  async stopCctvCapture() {
    if (!confirm("Stop RTSP frame capturing?")) return;
    if (this.cctvStopBtn) {
      this.cctvStopBtn.disabled = true;
      this.cctvStopBtn.innerHTML = "⏳ Stopping...";
    }

    try {
      const res = await fetch('/api/cctv/stop', { method: 'POST' });
      const data = await res.json();
      this.showToast(data.message || "RTSP Capture stopped.");
      this.checkCctvStatus();
      this.loadImages(true);
    } catch (err) {
      console.error("Stop CCTV capture error:", err);
      this.showToast("Error stopping capture: " + err.message, false);
    } finally {
      if (this.cctvStopBtn) {
        this.cctvStopBtn.disabled = false;
        this.cctvStopBtn.innerHTML = "⏹ Stop Capture";
      }
    }
  }

  async checkCctvStatus() {
    try {
      const res = await fetch('/api/cctv/status');
      if (!res.ok) return;
      const st = await res.json();

      const isCapturing = st.is_capturing;
      const count = st.captured_count || 0;
      const fpm = st.frames_per_minute || 15;
      const interval = st.interval_seconds || (60.0 / fpm).toFixed(1);

      // Header live badge
      if (this.cctvLiveBadge) {
        if (isCapturing) {
          this.cctvLiveBadge.style.display = 'inline-block';
          this.cctvLiveBadge.textContent = `🔴 CCTV: ${count} frames (${fpm}/min)`;
        } else {
          this.cctvLiveBadge.style.display = 'none';
        }
      }

      // Modal UI elements
      if (this.cctvModal && this.cctvModal.style.display !== 'none') {
        if (this.cctvSavedCount) this.cctvSavedCount.textContent = count;
        if (this.cctvIntervalVal) this.cctvIntervalVal.textContent = `${interval}s`;
        if (this.cctvLastTime) this.cctvLastTime.textContent = st.last_capture_time || '--:--:--';

        if (this.cctvElapsedTime) {
          const el = st.elapsed_seconds || 0;
          const mins = Math.floor(el / 60).toString().padStart(2, '0');
          const secs = (el % 60).toString().padStart(2, '0');
          this.cctvElapsedTime.textContent = `Elapsed: ${mins}:${secs}`;
        }

        if (isCapturing) {
          if (this.cctvStatusText) {
            this.cctvStatusText.textContent = `Status: Capturing (${fpm} frames/min)`;
            this.cctvStatusText.style.color = '#38bdf8';
          }
          if (this.cctvStatusDot) {
            this.cctvStatusDot.className = 'status-dot reviewed';
            this.cctvStatusDot.style.background = '#38bdf8';
            this.cctvStatusDot.style.boxShadow = '0 0 8px #38bdf8';
          }
          if (this.cctvStartBtn) this.cctvStartBtn.style.display = 'none';
          if (this.cctvStopBtn) {
            this.cctvStopBtn.style.display = 'inline-flex';
            this.cctvStopBtn.disabled = false;
          }
          if (this.cctvRtspUrl) this.cctvRtspUrl.disabled = true;
          if (this.cctvRateSelect) this.cctvRateSelect.disabled = true;
          if (this.cctvCategorySelect) this.cctvCategorySelect.disabled = true;
        } else {
          if (this.cctvStatusText) {
            this.cctvStatusText.textContent = 'Status: Idle';
            this.cctvStatusText.style.color = 'var(--text-secondary)';
          }
          if (this.cctvStatusDot) {
            this.cctvStatusDot.className = 'status-dot pending';
            this.cctvStatusDot.style.background = '#6e7681';
            this.cctvStatusDot.style.boxShadow = 'none';
          }
          if (this.cctvStartBtn) {
            this.cctvStartBtn.style.display = 'inline-flex';
            this.cctvStartBtn.disabled = false;
          }
          if (this.cctvStopBtn) this.cctvStopBtn.style.display = 'none';
          if (this.cctvRtspUrl) this.cctvRtspUrl.disabled = false;
          if (this.cctvRateSelect) this.cctvRateSelect.disabled = false;
          if (this.cctvCategorySelect) this.cctvCategorySelect.disabled = false;
        }

        if (this.cctvErrorBanner) {
          if (st.error) {
            this.cctvErrorBanner.style.display = 'block';
            this.cctvErrorBanner.textContent = `⚠️ ${st.error}`;
          } else {
            this.cctvErrorBanner.style.display = 'none';
          }
        }
      }
    } catch (e) {
      // Silent catch
    }
  }

  // =========================================================================
  // Canvas Rendering & Geometry
  // =========================================================================
  fitCanvasToScreen() {
    if (!this.currentImage) return;
    const wrapper = this.canvasWrapper.getBoundingClientRect();
    const padding = 30;
    const maxW = (wrapper.width - padding) * this.zoom;
    const maxH = (wrapper.height - padding) * this.zoom;

    const imgW = this.currentImage.naturalWidth;
    const imgH = this.currentImage.naturalHeight;

    const ratio = Math.min(maxW / imgW, maxH / imgH);
    this.scale = ratio;

    this.canvas.width = Math.round(imgW * ratio);
    this.canvas.height = Math.round(imgH * ratio);

    this.draw();
  }

  setZoom(val) {
    this.zoom = Math.max(0.5, Math.min(2.5, val));
    this.zoomLevelText.textContent = `${Math.round(this.zoom * 100)}%`;
    this.fitCanvasToScreen();
  }

  draw() {
    if (!this.currentImage) return;
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;

    ctx.clearRect(0, 0, w, h);

    // 1. Draw Image
    ctx.drawImage(this.currentImage, 0, 0, w, h);

    // 2. Draw Bounding Boxes
    this.currentBoxes.forEach((box, idx) => {
      const isSelected = idx === this.selectedBoxIndex;
      const isHovered = idx === this.hoverBoxIndex;

      // Convert normalized xywh to canvas pixels
      const bx = (box.xc - box.w / 2) * w;
      const by = (box.yc - box.h / 2) * h;
      const bw = box.w * w;
      const bh = box.h * h;

      // Class Color: Hand = Warm Amber (#ff9f1c), Gloves = Neon Cyan (#00d2ff)
      const isHand = (box.class_id === 1) || (box.class_name === 'Hand');
      let strokeColor = isHand ? '#ff9f1c' : '#00d2ff';
      let fillColor = isHand ? 'rgba(255, 159, 28, 0.18)' : 'rgba(0, 210, 255, 0.15)';

      if (isSelected) {
        strokeColor = '#3fb950'; // Neon Green for selected
        fillColor = 'rgba(63, 185, 80, 0.28)';
      } else if (isHovered) {
        strokeColor = isHand ? '#ffa938' : '#58a6ff';
        fillColor = isHand ? 'rgba(255, 159, 28, 0.28)' : 'rgba(88, 166, 255, 0.25)';
      }

      ctx.fillStyle = fillColor;
      ctx.fillRect(bx, by, bw, bh);

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeRect(bx, by, bw, bh);

      // Draw Badge / Label on Top
      const className = isHand ? 'Hand' : 'Gloves';
      const confText = box.conf < 1.0 ? ` (${Math.round(box.conf * 100)}%)` : '';
      const labelText = `${className} #${idx + 1}${confText}`;

      ctx.font = 'bold 11px Outfit, sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const badgeH = 18;
      const badgeY = Math.max(0, by - badgeH);

      ctx.fillStyle = strokeColor;
      ctx.fillRect(bx, badgeY, textWidth + 12, badgeH);

      ctx.fillStyle = '#000000';
      ctx.fillText(labelText, bx + 6, badgeY + 13);

      // Draw Resize Handles if Selected
      if (isSelected) {
        this.drawHandles(ctx, bx, by, bw, bh);
      }
    });

    // 3. Draw Active User Box if drawing
    if (this.mode === 'drawing') {
      const { x1, y1, x2, y2 } = this.getBoxCoords(this.dragStart.x, this.dragStart.y, this.currentMouseX, this.currentMouseY);
      const isHand = this.activeClassId === 1;
      const previewColor = isHand ? '#ff9f1c' : '#00d2ff';

      ctx.strokeStyle = previewColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = isHand ? 'rgba(255, 159, 28, 0.2)' : 'rgba(0, 210, 255, 0.2)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }
  }

  drawHandles(ctx, bx, by, bw, bh) {
    const handleSize = 8;
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 1.5;

    const handles = [
      { name: 'nw', x: bx, y: by },
      { name: 'ne', x: bx + bw, y: by },
      { name: 'se', x: bx + bw, y: by + bh },
      { name: 'sw', x: bx, y: by + bh },
      { name: 'n', x: bx + bw / 2, y: by },
      { name: 's', x: bx + bw / 2, y: by + bh },
      { name: 'w', x: bx, y: by + bh / 2 },
      { name: 'e', x: bx + bw, y: by + bh / 2 }
    ];

    handles.forEach(h => {
      ctx.fillRect(h.x - handleSize / 2, h.y - handleSize / 2, handleSize, handleSize);
      ctx.strokeRect(h.x - handleSize / 2, h.y - handleSize / 2, handleSize, handleSize);
    });
  }

  // =========================================================================
  // Canvas Mouse Interactions
  // =========================================================================
  getCanvasMousePos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }

  onMouseDown(e) {
    if (!this.currentImage) return;
    const pos = this.getCanvasMousePos(e);
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Check handle collision on selected box first
    if (this.selectedBoxIndex >= 0) {
      const handle = this.getHandleAt(pos.x, pos.y);
      if (handle) {
        this.mode = 'resizing';
        this.activeHandle = handle;
        this.dragStart = pos;
        this.activeBoxInitial = { ...this.currentBoxes[this.selectedBoxIndex] };
        return;
      }
    }

    // Check box collision
    const boxIdx = this.getBoxAt(pos.x, pos.y);
    if (boxIdx >= 0) {
      this.selectedBoxIndex = boxIdx;
      this.mode = 'moving';
      this.dragStart = pos;
      this.activeBoxInitial = { ...this.currentBoxes[boxIdx] };
      this.renderInspector();
      this.draw();
      return;
    }

    // Clicked empty space -> start drawing new box
    this.selectedBoxIndex = -1;
    this.mode = 'drawing';
    this.dragStart = pos;
    this.currentMouseX = pos.x;
    this.currentMouseY = pos.y;
    this.renderInspector();
    this.draw();
  }

  onMouseMove(e) {
    if (!this.currentImage) return;
    const pos = this.getCanvasMousePos(e);
    this.currentMouseX = pos.x;
    this.currentMouseY = pos.y;
    const w = this.canvas.width;
    const h = this.canvas.height;

    if (this.mode === 'drawing') {
      this.draw();
    } else if (this.mode === 'moving') {
      const dx = (pos.x - this.dragStart.x) / w;
      const dy = (pos.y - this.dragStart.y) / h;
      const init = this.activeBoxInitial;

      let newXc = Math.max(init.w / 2, Math.min(1 - init.w / 2, init.xc + dx));
      let newYc = Math.max(init.h / 2, Math.min(1 - init.h / 2, init.yc + dy));

      this.currentBoxes[this.selectedBoxIndex].xc = newXc;
      this.currentBoxes[this.selectedBoxIndex].yc = newYc;
      this.draw();
      this.renderSelectedProps();
    } else if (this.mode === 'resizing') {
      this.handleResize(pos.x, pos.y);
      this.draw();
      this.renderSelectedProps();
    } else {
      // Update cursor and hover state
      if (this.selectedBoxIndex >= 0) {
        const handle = this.getHandleAt(pos.x, pos.y);
        if (handle) {
          this.canvas.style.cursor = `${handle}-resize`;
          return;
        }
      }

      const boxIdx = this.getBoxAt(pos.x, pos.y);
      if (boxIdx >= 0) {
        this.hoverBoxIndex = boxIdx;
        this.canvas.style.cursor = 'move';
      } else {
        this.hoverBoxIndex = -1;
        this.canvas.style.cursor = 'crosshair';
      }
      this.draw();
    }
  }

  onMouseUp(e) {
    if (this.mode === 'drawing') {
      const pos = this.getCanvasMousePos(e);
      const w = this.canvas.width;
      const h = this.canvas.height;

      const { x1, y1, x2, y2 } = this.getBoxCoords(this.dragStart.x, this.dragStart.y, pos.x, pos.y);
      const pixelW = Math.abs(x2 - x1);
      const pixelH = Math.abs(y2 - y1);

      // Only accept box if greater than 10x10 pixels
      if (pixelW > 10 && pixelH > 10) {
        const xc = (x1 + pixelW / 2) / w;
        const yc = (y1 + pixelH / 2) / h;
        const normW = pixelW / w;
        const normH = pixelH / h;

        this.currentBoxes.push({
          id: this.currentBoxes.length + 1,
          class_id: this.activeClassId,
          class_name: this.activeClassName,
          xc: Math.max(0, Math.min(1, xc)),
          yc: Math.max(0, Math.min(1, yc)),
          w: Math.max(0.001, Math.min(1, normW)),
          h: Math.max(0.001, Math.min(1, normH)),
          conf: 1.0,
          is_ai: false
        });

        this.selectedBoxIndex = this.currentBoxes.length - 1;
        this.renderInspector();
      }
    }

    this.mode = 'idle';
    this.activeHandle = null;
    this.activeBoxInitial = null;
    this.draw();
  }

  onMouseLeave() {
    if (this.mode === 'drawing') {
      this.mode = 'idle';
      this.draw();
    }
  }

  getBoxCoords(x1, y1, x2, y2) {
    return {
      x1: Math.min(x1, x2),
      y1: Math.min(y1, y2),
      x2: Math.max(x1, x2),
      y2: Math.max(y1, y2)
    };
  }

  getBoxAt(x, y) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    for (let i = this.currentBoxes.length - 1; i >= 0; i--) {
      const b = this.currentBoxes[i];
      const bx = (b.xc - b.w / 2) * w;
      const by = (b.yc - b.h / 2) * h;
      const bw = b.w * w;
      const bh = b.h * h;
      if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
        return i;
      }
    }
    return -1;
  }

  getHandleAt(x, y) {
    if (this.selectedBoxIndex < 0) return null;
    const b = this.currentBoxes[this.selectedBoxIndex];
    const w = this.canvas.width;
    const h = this.canvas.height;
    const bx = (b.xc - b.w / 2) * w;
    const by = (b.yc - b.h / 2) * h;
    const bw = b.w * w;
    const bh = b.h * h;
    const radius = 10;

    const handles = [
      { name: 'nw', x: bx, y: by },
      { name: 'ne', x: bx + bw, y: by },
      { name: 'se', x: bx + bw, y: by + bh },
      { name: 'sw', x: bx, y: by + bh },
      { name: 'n', x: bx + bw / 2, y: by },
      { name: 's', x: bx + bw / 2, y: by + bh },
      { name: 'w', x: bx, y: by + bh / 2 },
      { name: 'e', x: bx + bw, y: by + bh / 2 }
    ];

    for (let h of handles) {
      if (Math.abs(x - h.x) <= radius && Math.abs(y - h.y) <= radius) {
        return h.name;
      }
    }
    return null;
  }

  handleResize(mouseX, mouseY) {
    const w = this.canvas.width;
    const h = this.canvas.height;
    const init = this.activeBoxInitial;

    let x1 = (init.xc - init.w / 2) * w;
    let y1 = (init.yc - init.h / 2) * h;
    let x2 = (init.xc + init.w / 2) * w;
    let y2 = (init.yc + init.h / 2) * h;

    const handle = this.activeHandle;
    if (handle.includes('w')) x1 = Math.min(mouseX, x2 - 10);
    if (handle.includes('e')) x2 = Math.max(mouseX, x1 + 10);
    if (handle.includes('n')) y1 = Math.min(mouseY, y2 - 10);
    if (handle.includes('s')) y2 = Math.max(mouseY, y1 + 10);

    const newW = (x2 - x1) / w;
    const newH = (y2 - y1) / h;
    const newXc = (x1 + (x2 - x1) / 2) / w;
    const newYc = (y1 + (y2 - y1) / 2) / h;

    this.currentBoxes[this.selectedBoxIndex].xc = Math.max(0, Math.min(1, newXc));
    this.currentBoxes[this.selectedBoxIndex].yc = Math.max(0, Math.min(1, newYc));
    this.currentBoxes[this.selectedBoxIndex].w = Math.max(0.005, Math.min(1, newW));
    this.currentBoxes[this.selectedBoxIndex].h = Math.max(0.005, Math.min(1, newH));
  }

  // =========================================================================
  // Inspector & Sidebar
  // =========================================================================
  renderInspector() {
    this.boxCount.textContent = this.currentBoxes.length;
    this.boxList.innerHTML = '';

    if (this.currentBoxes.length === 0) {
      this.boxList.innerHTML = '<p class="empty-hint">No boxes. Click & drag on image to add a Hand box, or mark negative.</p>';
      this.deleteSelectedBtn.disabled = true;
      this.selectedBoxProps.innerHTML = '<p class="empty-hint">No box selected.</p>';
      return;
    }

    this.currentBoxes.forEach((box, idx) => {
      const item = document.createElement('div');
      item.className = `box-item ${idx === this.selectedBoxIndex ? 'selected' : ''}`;
      const isHand = (box.class_id === 1) || (box.class_name === 'Hand');
      const className = isHand ? 'Hand' : 'Gloves';
      const confText = box.conf < 1.0 ? ` (AI: ${Math.round(box.conf * 100)}%)` : ' (Manual)';
      
      item.innerHTML = `
        <span class="box-item-name" style="color: ${isHand ? '#ff9f1c' : '#00d2ff'}">#${idx + 1} ${className}</span>
        <span class="box-item-conf">${confText}</span>
      `;
      item.addEventListener('click', () => {
        this.selectedBoxIndex = idx;
        this.renderInspector();
        this.draw();
      });
      this.boxList.appendChild(item);
    });

    this.renderSelectedProps();
  }

  renderSelectedProps() {
    if (this.selectedBoxIndex >= 0 && this.selectedBoxIndex < this.currentBoxes.length) {
      const b = this.currentBoxes[this.selectedBoxIndex];
      const isHand = (b.class_id === 1) || (b.class_name === 'Hand');
      this.deleteSelectedBtn.disabled = false;
      this.selectedBoxProps.innerHTML = `
        <div style="margin-bottom: 6px;">
          <strong>Class:</strong> 
          <span style="color: ${isHand ? '#ff9f1c' : '#00d2ff'}; font-weight: bold;">
            ${isHand ? '✋ Hand' : '🧤 Gloves'}
          </span>
          <button id="switchClassBtn" class="btn btn-outline" style="padding: 2px 8px; font-size: 11px; margin-left: 8px;">
            Switch to ${isHand ? 'Gloves' : 'Hand'}
          </button>
        </div>
        <div>X Center: ${(b.xc * 100).toFixed(1)}%</div>
        <div>Y Center: ${(b.yc * 100).toFixed(1)}%</div>
        <div>Width:    ${(b.w * 100).toFixed(1)}%</div>
        <div>Height:   ${(b.h * 100).toFixed(1)}%</div>
      `;

      const switchBtn = document.getElementById('switchClassBtn');
      if (switchBtn) {
        switchBtn.addEventListener('click', () => {
          b.class_id = isHand ? 0 : 1;
          b.class_name = isHand ? 'Gloves' : 'Hand';
          this.renderInspector();
          this.draw();
        });
      }
    } else {
      this.deleteSelectedBtn.disabled = true;
      this.selectedBoxProps.innerHTML = '<p class="empty-hint">Click a box to inspect or resize.</p>';
    }
  }

  renderThumbnails() {
    this.thumbnailList.innerHTML = '';
    const filtered = this.images.filter(img => {
      if (this.searchQuery && !img.name.toLowerCase().includes(this.searchQuery)) {
        return false;
      }
      if (this.currentFilter === 'pending') return img.status !== 'reviewed';
      if (this.currentFilter === 'reviewed') return img.status === 'reviewed';
      return true;
    });

    filtered.forEach((img, idx) => {
      const div = document.createElement('div');
      div.className = `thumb-item ${img.name === this.currentImageName.textContent ? 'active' : ''}`;
      div.dataset.name = img.name;

      const subtext = img.status === 'reviewed' ? (img.box_count > 0 ? `${img.box_count} objects` : 'negative') : 'unreviewed';

      div.innerHTML = `
        <img class="thumb-preview" src="/api/image/${encodeURIComponent(img.name)}" loading="lazy">
        <div class="thumb-info">
          <div class="thumb-name">${img.name}</div>
          <div class="thumb-meta">
            <span class="status-dot ${img.status}"></span>
            <span>${subtext}</span>
          </div>
        </div>
      `;
      div.addEventListener('click', () => {
        const realIdx = this.images.findIndex(i => i.name === img.name);
        this.selectImage(realIdx);
      });
      this.thumbnailList.appendChild(div);
    });
  }

  navigate(dir) {
    const nextIdx = this.currentIndex + dir;
    if (nextIdx >= 0 && nextIdx < this.images.length) {
      this.selectImage(nextIdx);
    }
  }

  deleteSelected() {
    if (this.selectedBoxIndex >= 0) {
      this.currentBoxes.splice(this.selectedBoxIndex, 1);
      this.selectedBoxIndex = -1;
      this.renderInspector();
      this.draw();
    }
  }

  clearAll() {
    if (confirm('Clear all bounding boxes from this image?')) {
      this.currentBoxes = [];
      this.selectedBoxIndex = -1;
      this.renderInspector();
      this.draw();
    }
  }

  clearCanvas() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.currentImageName.textContent = 'No images in folder';
    this.currentImageDim.textContent = '';
  }

  onKeyDown(e) {
    // Prevent shortcut conflicts if focused on inputs
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

    if (e.code === 'Space' || e.code === 'Enter') {
      e.preventDefault();
      this.saveCurrent(false);
    } else if (e.key === 'n' || e.key === 'N') {
      e.preventDefault();
      this.saveCurrent(true);
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      this.deleteSelected();
    } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
      this.navigate(1);
    } else if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
      this.navigate(-1);
    } else if (e.key === '1') {
      this.setActiveClass(1, 'Hand');
    } else if (e.key === '2') {
      this.setActiveClass(0, 'Gloves');
    } else if (e.key === 'r' || e.key === 'R') {
      this.reDetectAI();
    } else if (e.key === 'c' || e.key === 'C') {
      this.clearAll();
    }
  }

  // =========================================================================
  // Image Upload & AI Detection Modal Logic
  // =========================================================================
  openDetectModal() {
    if (this.detectModal) {
      this.detectModal.style.display = 'flex';
      if (!this.currentUploadedFile) {
        this.resetDetectView();
      }
    }
  }

  closeDetectModal() {
    if (this.detectModal) {
      this.detectModal.style.display = 'none';
    }
  }

  resetDetectView() {
    this.currentUploadedFile = null;
    this.currentDetectionResult = null;
    if (this.detectFileInput) this.detectFileInput.value = '';
    if (this.detectDropzone) this.detectDropzone.style.display = 'block';
    if (this.detectLoadingIndicator) this.detectLoadingIndicator.style.display = 'none';
    if (this.detectResultContainer) this.detectResultContainer.style.display = 'none';
    if (this.detectUploadNewBtn) this.detectUploadNewBtn.style.display = 'none';
  }

  handleFileSelected(file) {
    if (!file || !file.type.startsWith('image/')) {
      alert('Please select a valid image file (JPG, PNG, WEBP, BMP).');
      return;
    }
    this.currentUploadedFile = file;
    this.runUploadDetection(file);
  }

  async runUploadDetection(file) {
    if (this.detectDropzone) this.detectDropzone.style.display = 'none';
    if (this.detectResultContainer) this.detectResultContainer.style.display = 'none';
    if (this.detectLoadingIndicator) this.detectLoadingIndicator.style.display = 'block';
    if (this.detectUploadNewBtn) this.detectUploadNewBtn.style.display = 'inline-block';

    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf', this.detectConfSlider ? this.detectConfSlider.value : 0.25);

    try {
      const res = await fetch('/api/detect_upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (this.detectLoadingIndicator) this.detectLoadingIndicator.style.display = 'none';

      if (!res.ok || !data.success) {
        alert('Detection error: ' + (data.error || 'Server failed to analyze image'));
        this.resetDetectView();
        return;
      }

      this.currentDetectionResult = data;
      this.renderDetectionResult(data);

    } catch (err) {
      if (this.detectLoadingIndicator) this.detectLoadingIndicator.style.display = 'none';
      alert('Network error while running AI detection: ' + err.message);
      this.resetDetectView();
    }
  }

  renderDetectionResult(data) {
    if (this.detectResultContainer) this.detectResultContainer.style.display = 'block';

    // Status Banner
    if (this.detectStatusText) this.detectStatusText.textContent = data.status_text;
    if (this.detectLatencyBadge) this.detectLatencyBadge.textContent = `${data.latency_ms} ms (${data.model_name})`;
    if (this.detectStatusBanner) this.detectStatusBanner.className = `status-banner-${data.badge_type}`;

    // Annotated Result Image Preview
    if (this.detectResultImg) this.detectResultImg.src = data.annotated_image;

    // Counts
    if (this.detectGlovesCount) this.detectGlovesCount.textContent = data.gloves_count;
    if (this.detectHandCount) this.detectHandCount.textContent = data.hand_count;

    // Detailed List of detections
    if (this.detectDetectionsList) {
      this.detectDetectionsList.innerHTML = '';
      if (!data.detections || data.detections.length === 0) {
        this.detectDetectionsList.innerHTML = '<p class="empty-hint" style="font-size:12px; margin:0; padding:4px 0;">No gloves or bare hands detected at this confidence threshold.</p>';
      } else {
        data.detections.forEach((d, idx) => {
          const isGlove = d.class_id === 0;
          const chip = document.createElement('div');
          chip.className = `detection-chip ${isGlove ? 'glove-chip' : 'hand-chip'}`;
          chip.innerHTML = `
            <span>${isGlove ? '🧤' : '✋'} #${idx+1} ${d.class_name}</span>
            <strong style="font-weight:700;">${Math.round(d.conf * 100)}%</strong>
          `;
          this.detectDetectionsList.appendChild(chip);
        });
      }
    }

    this.showToast(`Detection: ${data.gloves_count} Gloves, ${data.hand_count} Hands in ${data.latency_ms}ms`);
  }

  async saveDetectedToDataset(category) {
    if (!this.currentDetectionResult) return;

    const payload = {
      category: category,
      image_b64: this.currentDetectionResult.annotated_image,
      boxes: this.currentDetectionResult.detections,
      filename: this.currentDetectionResult.filename || 'uploaded_sample.jpg'
    };

    try {
      const res = await fetch('/api/save_detected_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok && data.success) {
        this.showToast(`Saved to ${category.toUpperCase()} dataset as ${data.filename}!`);
        // Refresh folders & queue in background
        this.loadFolders();
        this.loadImages();
      } else {
        alert('Could not save to dataset: ' + (data.error || 'Server error'));
      }
    } catch (err) {
      alert('Network error saving to dataset: ' + err.message);
    }
  }
}

// Start on document load
document.addEventListener('DOMContentLoaded', () => {
  window.studio = new LabelerStudio();
});
