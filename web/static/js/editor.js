class ChartEditor {
    constructor() {
        this.canvas = document.getElementById('timelineCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.chartId = null;
        this.chartData = { notes: [], metadata: {} };
        this.noteColors = {
            'don': '#FF4D4D',
            'ka': '#4DA6FF',
            'big_don': '#FF0000',
            'big_ka': '#0077FF'
        };
        this.noteSize = 20;
        this.timeScale = 0.1; // pixels per millisecond
        this.currentTool = 'don';

        this.init();
    }

    async init() {
        this.resizeCanvas();
        this.setupEventListeners();
        await this.loadChartData();
        this.render();
    }

    resizeCanvas() {
        const lastTime = Math.max(0, ...this.chartData.notes.map(note => Number(note.time) || 0));
        this.canvas.width = Math.max(this.canvas.parentElement.clientWidth, lastTime * this.timeScale + 200);
        this.canvas.height = this.canvas.parentElement.clientHeight;
        this.render();
    }

    setupEventListeners() {
        window.addEventListener('resize', this.resizeCanvas.bind(this));
        document.getElementById('saveButton').addEventListener('click', this.saveChart.bind(this));
        this.canvas.addEventListener('click', this.handleCanvasClick.bind(this));

        const toolButtons = document.querySelectorAll('.tool-btn');
        toolButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                toolButtons.forEach(b => {
                    b.classList.remove('active');
                    b.setAttribute('aria-pressed', 'false');
                });
                btn.classList.add('active');
                btn.setAttribute('aria-pressed', 'true');
                this.currentTool = btn.dataset.type;
            });
        });
    }

    async loadChartData() {
        const urlParams = new URLSearchParams(window.location.search);
        this.chartId = parseInt(urlParams.get('id'));

        if (!this.chartId) {
            console.error('No chart ID provided in URL');
            return;
        }

        try {
            const response = await fetch(`/api/chart-data?id=${this.chartId}`);
            if (!response.ok) {
                throw new Error('Failed to load chart data');
            }
            this.chartData = await response.json();
            document.getElementById('chartTitle').textContent = `Editing: ${this.chartData.metadata.title}`;

            // Adjust canvas width to fit the entire song
            const lastNote = this.chartData.notes[this.chartData.notes.length - 1];
            if (lastNote) {
                this.canvas.width = (lastNote.time * this.timeScale) + 200;
            }

        } catch (error) {
            console.error('Error loading chart data:', error);
        }
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw timeline ruler
        this.drawRuler();

        // Draw notes
        this.chartData.notes.forEach(note => {
            const x = note.time * this.timeScale;
            const y = this.canvas.height / 2;
            const isBig = note.type.startsWith('big_');

            this.ctx.fillStyle = this.noteColors[note.type];
            this.ctx.beginPath();
            this.ctx.arc(x, y, isBig ? this.noteSize * 1.5 : this.noteSize, 0, Math.PI * 2);
            this.ctx.fill();
        });
    }

    drawRuler() {
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.font = '12px sans-serif';

        for (let t = 0; t * this.timeScale < this.canvas.width; t += 1000) {
            const x = t * this.timeScale;
            this.ctx.fillRect(x, 20, 1, this.canvas.height - 40);
            this.ctx.fillText(`${t / 1000}s`, x + 5, 35);
        }
    }

    handleCanvasClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        const time = Math.round(x / this.timeScale);

        // Check if clicking on an existing note to delete it
        const noteIndex = this.findNoteAt(x, y);
        if (noteIndex !== -1) {
            this.chartData.notes.splice(noteIndex, 1);
        } else {
            // Add a new note
            this.chartData.notes.push({
                time: time,
                type: this.currentTool
            });
        }

        // Sort notes by time
        this.chartData.notes.sort((a, b) => a.time - b.time);

        this.render();
    }

    findNoteAt(x, y) {
        return this.chartData.notes.findIndex(note => {
            const noteX = note.time * this.timeScale;
            const noteY = this.canvas.height / 2;
            const isBig = note.type.startsWith('big_');
            const size = isBig ? this.noteSize * 1.5 : this.noteSize;
            const distance = Math.sqrt((x - noteX)**2 + (y - noteY)**2);
            return distance <= size;
        });
    }

    async saveChart() {
        try {
            const response = await fetch('/api/save-chart', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    id: this.chartId,
                    notes: this.chartData.notes
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to save chart');
            }

            const result = await response.json();
            alert('Chart saved successfully!');
            console.log(result);
        } catch (error) {
            console.error('Error saving chart:', error);
            alert('Error saving chart. See console for details.');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new ChartEditor();
});
