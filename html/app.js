// State variables
let videos = [];
let currentIndex = 0;
let annotations = [];
let annotationMap = {};
let currentData = {};
let csvFileHandle = null;
let currentFilter = { field: '', value: '' };

// Initialize event listeners when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
});

function initializeEventListeners() {
    // CSV input handler
    document.getElementById('csvInput').addEventListener('change', handleCSVLoad);

    // Video input handler
    document.getElementById('videoInput').addEventListener('change', handleVideoLoad);

    // Loop checkbox
    document.getElementById('loopCheckbox').addEventListener('change', (e) => {
        document.getElementById('videoPlayer').loop = e.target.checked;
    });

    // Filter handlers
    document.getElementById('filterField').addEventListener('change', handleFilterFieldChange);
    document.getElementById('filterValue').addEventListener('change', handleFilterValueChange);

    // Button groups
    document.querySelectorAll('.btn-group').forEach(group => {
        const field = group.dataset.field;
        group.querySelectorAll('.btn-option').forEach(btn => {
            btn.addEventListener('click', () => handleButtonClick(group, btn, field));
        });
    });

    // Grid cells
    document.querySelectorAll('.grid-cell').forEach(cell => {
        cell.addEventListener('click', () => handleGridCellClick(cell));
    });
}

function handleFilterFieldChange(e) {
    const field = e.target.value;
    const valueSelect = document.getElementById('filterValue');

    if (!field) {
        valueSelect.style.display = 'none';
        currentFilter = { field: '', value: '' };
        updateVideoList();
        return;
    }

    // Get unique values for this field
    const values = new Set();
    videos.forEach(video => {
        const annotation = annotationMap[video.name];
        if (annotation && annotation[field]) {
            values.add(annotation[field]);
        }
    });

    valueSelect.innerHTML = '<option value="">Select value...</option>';
    Array.from(values).sort().forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        valueSelect.appendChild(option);
    });

    valueSelect.style.display = 'block';
    currentFilter.field = field;
    currentFilter.value = '';
    updateVideoList();
}

function handleFilterValueChange(e) {
    currentFilter.value = e.target.value;
    updateVideoList();
}

async function handleCSVLoad(e) {
    const file = e.target.files[0];
    if (!file) return;

    csvFileHandle = file;

    const text = await file.text();
    const lines = text.split('\n');

    annotations = [];
    annotationMap = {};

    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;

        const parts = line.split(',');
        if (parts.length >= 10) {
            const annotation = {
                video_file: parts[0],
                foot: parts[1],
                side: parts[2],
                height: parts[3],
                in_out: parts[4],
                saved: parts[5],
                camera_angle: parts[6],
                player_visibility: parts[7],
                run_speed: parts[8],
                fake: parts[9]
            };
            annotations.push(annotation);
            annotationMap[parts[0]] = annotation;
        }
    }

    alert(`Loaded CSV with ${annotations.length} existing annotations`);
    updateVideoList();
}

function handleVideoLoad(e) {
    const files = Array.from(e.target.files);
    videos = files.filter(f => f.type.startsWith('video/'));
    currentIndex = 0;
    updateVideoList();
    if (videos.length > 0) {
        loadVideo(0);
    }
}

function handleButtonClick(group, btn, field) {
    group.querySelectorAll('.btn-option').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentData[field] = btn.dataset.value;

    if (field === 'in_out') {
        document.getElementById('savedGroup').style.display =
            btn.dataset.value === 'in' ? 'block' : 'none';
        if (btn.dataset.value === 'out') {
            currentData.saved = 'n/a';
        }
    }

    updateSummary();
}

function handleGridCellClick(cell) {
    document.querySelectorAll('.grid-cell').forEach(c => c.classList.remove('active'));
    cell.classList.add('active');
    currentData.side = cell.dataset.side;
    currentData.height = cell.dataset.height;
    updateSummary();
}

function updateVideoList() {
    const container = document.getElementById('videoListContainer');
    container.innerHTML = '';

    // Filter videos
    let filteredVideos = videos;
    if (currentFilter.field && currentFilter.value) {
        filteredVideos = videos.filter(video => {
            const annotation = annotationMap[video.name];
            return annotation && annotation[currentFilter.field] === currentFilter.value;
        });
    }

    filteredVideos.forEach((video) => {
        const index = videos.indexOf(video);
        const item = document.createElement('div');
        item.className = 'video-item';

        const isAnnotated = annotationMap[video.name];
        if (isAnnotated) {
            item.classList.add('annotated');
        }
        if (index === currentIndex) {
            item.classList.add('active');
        }

        item.innerHTML = `
            ${isAnnotated ? '<span class="check">✓</span>' : '<span style="width:12px;display:inline-block;"></span>'}
            <span class="name" title="${video.name}">${video.name}</span>
        `;

        item.onclick = () => loadVideo(index);
        container.appendChild(item);

        // Scroll to active item
        if (index === currentIndex) {
            setTimeout(() => {
                item.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }, 100);
        }
    });

    const annotated = videos.filter(v => annotationMap[v.name]).length;
    const filterInfo = currentFilter.field && currentFilter.value
        ? ` | Filtered: ${filteredVideos.length}`
        : '';
    document.getElementById('progress').textContent =
        `Video: ${currentIndex + 1}/${videos.length} | Annotated: ${annotated}/${videos.length}${filterInfo}`;
}

function loadVideo(index) {
    currentIndex = index;
    const video = videos[currentIndex];
    const url = URL.createObjectURL(video);
    const videoPlayer = document.getElementById('videoPlayer');
    videoPlayer.src = url;

    videoPlayer.onloadeddata = () => {
        videoPlayer.play();
    };

    updateVideoList();
    loadAnnotationForCurrentVideo();
}

function loadAnnotationForCurrentVideo() {
    const videoName = videos[currentIndex].name;
    const existingAnnotation = annotationMap[videoName];

    resetForm();
    currentData.video_file = videoName;

    if (existingAnnotation) {
        currentData = { ...existingAnnotation };

        Object.keys(existingAnnotation).forEach(field => {
            if (field === 'video_file') return;

            const value = existingAnnotation[field];

            const group = document.querySelector(`[data-field="${field}"]`);
            if (group) {
                group.querySelectorAll('.btn-option').forEach(btn => {
                    if (btn.dataset.value === value) {
                        btn.classList.add('active');
                    }
                });
            }

            if (field === 'side' || field === 'height') {
                const side = existingAnnotation.side;
                const height = existingAnnotation.height;
                document.querySelectorAll('.grid-cell').forEach(cell => {
                    if (cell.dataset.side === side && cell.dataset.height === height) {
                        cell.classList.add('active');
                    }
                });
            }
        });

        if (existingAnnotation.in_out === 'in') {
            document.getElementById('savedGroup').style.display = 'block';
        }

        updateSummary();
    }
}

function resetForm() {
    currentData = { video_file: videos[currentIndex].name };
    document.querySelectorAll('.btn-option').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.grid-cell').forEach(c => c.classList.remove('active'));
    document.getElementById('savedGroup').style.display = 'none';
    document.getElementById('summary').style.display = 'none';
}

function updateSummary() {
    const required = ['camera_angle', 'player_visibility', 'foot', 'run_speed', 'fake', 'in_out', 'side', 'height'];
    if (currentData.in_out === 'in') required.push('saved');

    const allFilled = required.every(f => currentData[f]);

    if (allFilled) {
        const summary = document.getElementById('summary');
        summary.style.display = 'block';
        summary.innerHTML = `
            <div class="summary-row"><span>Camera:</span><span>${currentData.camera_angle}</span></div>
            <div class="summary-row"><span>Visibility:</span><span>${currentData.player_visibility}</span></div>
            <div class="summary-row"><span>Foot:</span><span>${currentData.foot}</span></div>
            <div class="summary-row"><span>Speed:</span><span>${currentData.run_speed}</span></div>
            <div class="summary-row"><span>Fake:</span><span>${currentData.fake}</span></div>
            <div class="summary-row"><span>Position:</span><span>${currentData.side}-${currentData.height}</span></div>
            <div class="summary-row"><span>Result:</span><span>${currentData.in_out}</span></div>
            <div class="summary-row"><span>Saved:</span><span>${currentData.saved}</span></div>
        `;
    }
}

function saveAnnotation() {
    const required = ['camera_angle', 'player_visibility', 'foot', 'run_speed', 'fake', 'in_out', 'side', 'height'];
    if (currentData.in_out === 'in') required.push('saved');

    if (!required.every(f => currentData[f])) {
        alert('Please complete all fields!');
        return;
    }

    const videoName = videos[currentIndex].name;

    annotations = annotations.filter(a => a.video_file !== videoName);

    annotations.push({ ...currentData });
    annotationMap[videoName] = { ...currentData };

    updateVideoList();

    goToNextUnannotated();
}

function exportCSV() {
    const headers = ['video_file', 'foot', 'side', 'height', 'in_out', 'saved',
        'camera_angle', 'player_visibility', 'run_speed', 'fake'];

    let csv = headers.join(',') + '\n';
    annotations.forEach(a => {
        csv += headers.map(h => a[h] || '').join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;

    if (csvFileHandle) {
        a.download = csvFileHandle.name;
    } else {
        a.download = 'annotations.csv';
    }

    a.click();
    URL.revokeObjectURL(url);

    alert(`✅ CSV exported with ${annotations.length} annotations!`);
}

function skipVideo() {
    goToNextUnannotated();
}

function goToNextUnannotated() {
    for (let i = currentIndex + 1; i < videos.length; i++) {
        if (!annotationMap[videos[i].name]) {
            loadVideo(i);
            return;
        }
    }

    for (let i = 0; i < currentIndex; i++) {
        if (!annotationMap[videos[i].name]) {
            loadVideo(i);
            return;
        }
    }

    alert('All videos have been annotated!');
}

function playVideo() {
    document.getElementById('videoPlayer').play();
}

function replayVideo() {
    const video = document.getElementById('videoPlayer');
    video.currentTime = 0;
    video.play();
}
