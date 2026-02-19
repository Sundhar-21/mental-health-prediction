let stream = null;
let mediaRecorder = null;
let audioChunks = [];
let audioBlob = null;
let faceBlob = null;
let recordingInterval = null;

// DOM Elements
const webcamVideo = document.getElementById('webcam');
const faceCanvas = document.getElementById('face-canvas');
const cameraStatus = document.getElementById('camera-status');
const step1Content = document.getElementById('step1-content');
const step2Content = document.getElementById('step2-content');
const step3Content = document.getElementById('step3-content');
const progressBar1 = document.getElementById('progress-bar-1');
const progressBar2 = document.getElementById('progress-bar-2');

// Initialize Camera
async function initCamera() {
    try {
        // Request VIDEO ONLY for the first step to avoid hogging the mic
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        webcamVideo.srcObject = stream;
        webcamVideo.muted = true; // Fix Echo: Mute the video feedback
        cameraStatus.style.display = 'none';

        // Start face detection once camera is ready
        startAutoCapture();

    } catch (err) {
        console.error("Camera Error:", err);
        cameraStatus.textContent = "Error: Camera access denied. " + err.message;
        cameraStatus.className = "absolute bottom-4 left-1/2 -translate-x-1/2 text-white bg-red-600 px-3 py-1 rounded-full text-sm";
    }
}

// Auto-Capture Face Detection Logic
let detectionInterval = null;

function startAutoCapture() {
    if (!stream) return;

    const status = document.getElementById('camera-status');
    status.style.display = 'block';
    status.textContent = "Looking for face...";
    status.className = "absolute bottom-4 left-1/2 -translate-x-1/2 text-white bg-blue-600 px-3 py-1 rounded-full text-sm font-semibold shadow-lg";

    // Polling backend for face detection
    detectionInterval = setInterval(async () => {
        if (!stream) return;

        // Capture frame to canvas
        const canvas = document.createElement('canvas');
        canvas.width = webcamVideo.videoWidth;
        canvas.height = webcamVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(webcamVideo, 0, 0);

        // Convert to base64
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

        // Send to backend
        const formData = new FormData();
        formData.append('face_image_base64', dataUrl);

        try {
            const res = await fetch('/detect_face', { method: 'POST', body: formData });

            if (!res.ok) {
                // Ignore 400 errors (bad image)
                return;
            }

            const data = await res.json();

            if (data.face_detected) {
                if (data.eyes_open) {
                    status.textContent = "Perfect! Capturing...";
                    status.className = "absolute bottom-4 left-1/2 -translate-x-1/2 text-white bg-green-600 px-3 py-1 rounded-full text-sm font-semibold animate-pulse";
                    clearInterval(detectionInterval);
                    // Slight delay for user to settle
                    setTimeout(captureFace, 500);
                } else {
                    status.textContent = "Face detected. Please open eyes / remove glasses.";
                    status.className = "absolute bottom-4 left-1/2 -translate-x-1/2 text-white bg-yellow-600 px-3 py-1 rounded-full text-sm font-semibold";
                }
            } else {
                status.textContent = "Looking for face...";
                status.className = "absolute bottom-4 left-1/2 -translate-x-1/2 text-white bg-blue-600 px-3 py-1 rounded-full text-sm font-semibold";
            }
        } catch (err) {
            console.error("Detection error:", err);
        }

    }, 1000); // Check every 1 second to avoid overload
}

// Capture Face
function captureFace() {
    if (!stream) return;

    // Draw video frame to canvas
    faceCanvas.width = webcamVideo.videoWidth;
    faceCanvas.height = webcamVideo.videoHeight;
    const ctx = faceCanvas.getContext('2d');

    // Flip context horizontally to match mirrored video
    ctx.translate(faceCanvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(webcamVideo, 0, 0, faceCanvas.width, faceCanvas.height);

    // Stop Camera
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    // Convert to Blob
    faceCanvas.toBlob(blob => {
        faceBlob = blob;
        goToStep2();
    }, 'image/jpeg', 0.95);
}

// Step Navigation
function goToStep2() {
    // Hide Step 1, Show Step 2
    step1Content.classList.add('hidden');
    step2Content.classList.remove('hidden');

    // Update Indicators
    document.getElementById('step2-indicator').classList.remove('opacity-50');
    progressBar1.style.width = '100%';
}

function goToStep3() {
    const nameInput = document.getElementById('patient-name');
    if (!nameInput.value.trim()) {
        alert("Please enter the patient's name.");
        nameInput.focus();
        return;
    }

    step2Content.classList.add('hidden');
    step3Content.classList.remove('hidden');

    document.getElementById('step3-indicator').classList.remove('opacity-50');
    progressBar2.style.width = '100%';

    submitData();
}

// Language Switch
const texts = {
    'English': "Life is a journey filled with a spectrum of emotions, each teaching us something valuable about ourselves. Today, I might feel a mix of excitement for what lies ahead and perhaps a touch of anxiety about the unknown. It is important to acknowledge these feelings, as they are a natural part of the human experience. By expressing them, I allow myself to process and understand my inner world better.",
    'Hindi': "जीवन भावनाओं का एक सफर है, जो हमें खुद के बारे में कुछ न कुछ सिखाता है। आज मैं भविष्य को लेकर उत्साह और अनजाने डर का मिला-जुला अनुभव कर रहा हूँ। इन भावनाओं को स्वीकार करना महत्वपूर्ण है, क्योंकि ये हमारे मानवीय अनुभव का स्वाभाविक हिस्सा हैं। इन्हें व्यक्त करके, मैं अपनी आंतरिक दुनिया को बेहतर ढंग से समझ पाता हूँ।",
    'Tamil': "வாழ்க்கை என்பது பலவிதமான உணர்ச்சிகள் நிறைந்த ஒரு பயணம், ஒவ்வொன்றும் நம்மைப் பற்றி மதிப்புமிக்க ஒன்றை நமக்குக் கற்றுக்கொடுக்கிறது. இன்று, எதிர்காலத்தைப் பற்றிய உற்சாகமும், தெரியாததைப் பற்றிய சற்றே கவலையும் கலந்த உணர்வை நான் கொண்டிருக்கலாம். இந்த உணர்வுகளை அங்கீகரிப்பது முக்கியம், ஏனென்றால் அவை மனித அனுபவத்தின் இயல்பான பகுதியாகும். அவற்றை வெளிப்படுத்துவதன் மூலம், என் உள் உலகத்தை நான் நன்றாகப் புரிந்துகொள்ள முடிகிறது."
};

function changeLang(lang) {
    document.getElementById('reading-text').textContent = texts[lang];
}

// Audio Recording with WAV encoding
let audioContext = null;
let audioStream = null;
let recorder = null;
let isRecording = false;
let startTime = 0;

function toggleRecording() {
    const btn = document.getElementById('record-btn');
    const ping = document.getElementById('record-ping');
    const status = document.getElementById('recording-status');

    if (!isRecording) {
        // Start Recording
        startRecording();
        btn.classList.add('bg-gray-800');
        ping.classList.remove('hidden');
        status.textContent = "Recording...";
        status.classList.add('text-red-500');
        isRecording = true;
    } else {
        // Stop Recording

        // CHECK: Enforce minimum 10 seconds
        const elapsed = Date.now() - startTime;
        if (elapsed < 10000) {
            alert("Please record for at least 10 seconds to ensure accurate analysis.");
            return;
        }

        stopRecording();
        btn.classList.remove('bg-gray-800');
        ping.classList.add('hidden');
        status.textContent = "Recording saved. Ready to analyze.";
        status.classList.remove('text-red-500');
        status.classList.add('text-green-500');
        isRecording = false;

        // Enable Next Button
        const nextBtn = document.getElementById('next-btn-2');
        nextBtn.removeAttribute('disabled');
        nextBtn.classList.remove('bg-gray-300', 'pointer-events-none');
        nextBtn.classList.add('bg-brand-600', 'hover:bg-brand-700', 'shadow-lg');
    }
}

async function startRecording() {
    try {
        // 1. Get a fresh Audio Stream
        console.log("Requesting audio stream...");
        audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // 2. Initialize Audio Context
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } else if (audioContext.state === 'closed') {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        // 3. Resume context
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }

        console.log("Audio Context State:", audioContext.state);

        // 4. Create Source & Recorder
        const input = audioContext.createMediaStreamSource(audioStream);
        recorder = new Recorder(input, { numChannels: 1 });
        recorder.record();

        startTime = Date.now();
        recordingInterval = setInterval(updateTimer, 100);

    } catch (err) {
        console.error("Error starting recording:", err);
        alert("Could not start recording: " + err.message + "\nPlease ensure microphone is allowed.");
    }
}

function stopRecording() {
    if (recorder) {
        recorder.stop();
        recorder.exportWAV(createDownloadLink);
        clearInterval(recordingInterval);
    }

    // Stop the audio stream tracks to release mic
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }
}

function createDownloadLink(blob) {
    audioBlob = blob;
    // Optional: Create a playback element
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.controls = true;

    // Check if we already added an audio player
    const existingAudio = document.getElementById('audio-preview');
    if (existingAudio) {
        existingAudio.src = url;
    } else {
        const container = document.getElementById('step2-content');
        const player = document.createElement('audio');
        player.id = 'audio-preview';
        player.controls = true;
        player.src = url;
        player.className = "mt-4 w-full max-w-md";

        // Insert before the Next button
        const nextBtn = document.getElementById('next-btn-2');
        container.insertBefore(player, nextBtn);
    }
}

function updateTimer() {
    const elapsed = Date.now() - startTime;
    const seconds = Math.floor(elapsed / 1000);
    const ms = Math.floor((elapsed % 1000) / 10);
    document.getElementById('timer').textContent = `${seconds.toString().padStart(2, '0')}:${ms.toString().padStart(2, '0')}`;
}

// Simple Recorder.js implementation inline for simplicity
// Based on Matt Diamond's Recorderjs
class Recorder {
    constructor(source, cfg) {
        this.config = cfg || {};
        this.bufferLen = this.config.bufferLen || 4096;
        this.context = source.context;
        this.node = (this.context.createScriptProcessor ||
            this.context.createJavaScriptNode).call(this.context,
                this.bufferLen, 2, 2);

        this.worker = new InlineWorker(function () {
            let recLength = 0,
                recBuffers = [];

            this.onmessage = function (e) {
                switch (e.data.command) {
                    case 'init':
                        init(e.data.config);
                        break;
                    case 'record':
                        record(e.data.buffer);
                        break;
                    case 'exportWAV':
                        exportWAV(e.data.type);
                        break;
                    case 'clear':
                        clear();
                        break;
                }
            };

            function init(config) {
                // sampleRate = config.sampleRate;
            }

            function record(inputBuffer) {
                recBuffers.push(inputBuffer[0]);
                recLength += inputBuffer[0].length;
            }

            function exportWAV(type) {
                let buffer = mergeBuffers(recBuffers, recLength);
                let dataview = encodeWAV(buffer);
                let audioBlob = new Blob([dataview], { type: type });
                this.postMessage(audioBlob);
            }

            function mergeBuffers(recBuffers, recLength) {
                let result = new Float32Array(recLength);
                let offset = 0;
                for (let i = 0; i < recBuffers.length; i++) {
                    result.set(recBuffers[i], offset);
                    offset += recBuffers[i].length;
                }
                return result;
            }

            function encodeWAV(samples) {
                let buffer = new ArrayBuffer(44 + samples.length * 2);
                let view = new DataView(buffer);

                /* RIFF identifier */
                writeString(view, 0, 'RIFF');
                /* RIFF chunk length */
                view.setUint32(4, 36 + samples.length * 2, true);
                /* RIFF type */
                writeString(view, 8, 'WAVE');
                /* fmt sub-chunk */
                writeString(view, 12, 'fmt ');
                /* fmt chunk length */
                view.setUint32(16, 16, true);
                /* sample format (raw) */
                view.setUint16(20, 1, true);
                /* channel count */
                view.setUint16(22, 1, true); // Mono
                /* sample rate */
                view.setUint32(24, 44100, true);
                /* byte rate (sample rate * block align) */
                view.setUint32(28, 44100 * 2, true);
                /* block align (channel count * bytes per sample) */
                view.setUint16(32, 2, true);
                /* bits per sample */
                view.setUint16(34, 16, true);
                /* data chunk identifier */
                writeString(view, 36, 'data');
                /* data chunk length */
                view.setUint32(40, samples.length * 2, true);

                floatTo16BitPCM(view, 44, samples);

                return view;
            }

            function floatTo16BitPCM(output, offset, input) {
                for (let i = 0; i < input.length; i++, offset += 2) {
                    let s = Math.max(-1, Math.min(1, input[i]));
                    output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                }
            }

            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            function clear() {
                recLength = 0;
                recBuffers = [];
            }
        });

        this.worker.postMessage({
            command: 'init',
            config: {
                sampleRate: this.context.sampleRate
            }
        });

        this.worker.onmessage = function (e) {
            let cb = _this.currCallback;
            _this.currCallback = null;
            if (cb) cb(e.data);
        };

        this.node.onaudioprocess = function (e) {
            if (!_this.recording) return;
            _this.worker.postMessage({
                command: 'record',
                buffer: [
                    e.inputBuffer.getChannelData(0)
                ]
            });
        };

        source.connect(this.node);
        this.node.connect(this.context.destination); // Required for proper scheduling

        let _this = this;
    }

    record() {
        this.recording = true;
    }

    stop() {
        this.recording = false;
    }

    clear() {
        this.worker.postMessage({ command: 'clear' });
    }

    exportWAV(cb, type) {
        this.currCallback = cb || this.config.callback;
        type = type || this.config.type || 'audio/wav';
        if (!this.currCallback) throw new Error('Callback not set');
        this.worker.postMessage({
            command: 'exportWAV',
            type: type
        });
    }
}

// Inline Worker Helper
class InlineWorker {
    constructor(func) {
        let functionBody = func.toString().trim().match(/^function\s*\w*\s*\([\w\s,]*\)\s*{([\w\W]*?)}$/)[1];
        let url = window.URL.createObjectURL(new Blob([functionBody], { type: "text/javascript" }));
        return new Worker(url);
    }
}



// Submit Data
async function submitData() {
    const formData = new FormData();
    if (faceBlob) formData.append('face_image', faceBlob, 'capture.jpg');
    if (audioBlob) formData.append('voice_audio', audioBlob, 'voice.wav');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            // Add patient name to result
            result.patient_name = document.getElementById('patient-name').value.trim();

            // Save to localStorage to pass to report page
            localStorage.setItem('visomind_report', JSON.stringify(result));
            window.location.href = '/report';
        } else {
            alert('Analysis failed: ' + result.error);
            location.reload();
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred during analysis.');
    }
}

// Start camera on load
window.addEventListener('load', () => {
    initCamera();
});
