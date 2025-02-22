// 🚀 Real-Time AI Speech Pipeline: Synthia with Local Speech Processing in JavaScript

const vosk = require('vosk');
const fs = require('fs');
const axios = require('axios');
const { spawn } = require('child_process');
const VAD = require('node-vad');
const path = require('path');
const { Transform } = require('stream');

// Configuration
const config = {
    API_URL: "http://127.0.0.1:5001/synthiaquery",
    API_KEY: "lucidia",
    MEMORY_FILE: "synthia_memory.json",
    MODEL_PATH: "./vosk-model-en-us-0.42-gigaspeech",
    DEFAULT_VOICE: "en-US-AvaNeural",
    SAMPLE_RATE: 48000,
    VAD_MODE: VAD.Mode.VERY_AGGRESSIVE,
    FRAME_LENGTH_MS: 20,
    MIN_SPEECH_MS: 300,
    MAX_SPEECH_MS: 10000,
    SILENCE_THRESHOLD_MS: 500,
    CHUNK_SIZE: 960,  // 20ms at 48kHz
    SPEECH_THRESHOLD: 0.8,
    SILENCE_THRESHOLD: 0.1,
};

class AudioProcessor {
    constructor() {
        this.levelBuffer = [];
        this.maxLevel = 0.1;
    }

    processFormat(data) {
        try {
            // Convert to 16-bit PCM and handle stereo
            const samples = data.length / 4; // 2 bytes per sample, 2 channels
            const monoData = new Int16Array(samples);
            
            for (let i = 0; i < samples; i++) {
                const left = data.readInt16LE(i * 4);
                const right = data.readInt16LE(i * 4 + 2);
                monoData[i] = Math.round((left + right) / 2);
            }
            
            // Calculate RMS level with smoothing
            const rms = Math.sqrt(monoData.reduce((sum, val) => sum + val * val, 0) / samples) / 32768;
            this.levelBuffer.push(rms);
            if (this.levelBuffer.length > 10) {
                this.levelBuffer.shift();
            }
            
            // Update max level with decay
            this.maxLevel = Math.max(
                this.levelBuffer.reduce((a, b) => a + b) / this.levelBuffer.length,
                this.maxLevel * 0.95
            );
            
            // Normalize level
            const normalizedLevel = rms / (this.maxLevel || 1);
            
            // Visual feedback
            const barLength = Math.min(Math.floor(normalizedLevel * 30), 30);
            const color = barLength <= 10 ? "\x1b[32m" : // Green
                         barLength <= 20 ? "\x1b[33m" : // Yellow
                         "\x1b[31m"; // Red
            
            const bar = "█".repeat(barLength) + " ".repeat(30 - barLength);
            process.stdout.write(`\r🎙️ Level: ${color}${bar}\x1b[0m`);
            
            // Apply noise gate
            const noiseGate = 0.001;
            const gatedData = Buffer.alloc(samples * 2);
            for (let i = 0; i < samples; i++) {
                const sample = monoData[i];
                gatedData.writeInt16LE(Math.abs(sample) < noiseGate * 32768 ? 0 : sample, i * 2);
            }
            
            return { audio: gatedData, level: normalizedLevel };
        } catch (error) {
            console.error("❌ Audio Processing Error:", error);
            return null;
        }
    }
}

class VadProcessor {
    constructor() {
        this.vad = new VAD(config.VAD_MODE);
        this.state = {
            isSpeech: false,
            speechStartTime: null,
            speechProbability: 0,
            consecutiveSpeech: 0,
            consecutiveSilence: 0
        };
        this.lastDebugTime = Date.now();
    }

    async processFrame(audioData) {
        try {
            // Resample to 16kHz for VAD
            const samples = audioData.length / 2;
            const resampled = Buffer.alloc(samples / 3 * 2);
            for (let i = 0; i < samples / 3; i++) {
                resampled.writeInt16LE(audioData.readInt16LE(i * 6), i * 2);
            }
            
            const result = await this.vad.processAudio(resampled, 16000);
            const isSpeech = result === VAD.Event.VOICE;
            
            // Update state with hysteresis
            if (isSpeech) {
                this.state.consecutiveSpeech++;
                this.state.consecutiveSilence = 0;
                this.state.speechProbability = Math.min(1, this.state.speechProbability + 0.3);
            } else {
                this.state.consecutiveSpeech = 0;
                this.state.consecutiveSilence++;
                this.state.speechProbability = Math.max(0, this.state.speechProbability - 0.05);
            }
            
            // State transitions
            const now = Date.now();
            if (!this.state.isSpeech && this.state.speechProbability > config.SPEECH_THRESHOLD) {
                this.state.isSpeech = true;
                this.state.speechStartTime = now;
                console.log("\n🗣️ Speech started");
            } else if (this.state.isSpeech && this.state.speechProbability < config.SILENCE_THRESHOLD) {
                this.state.isSpeech = false;
                if (this.state.speechStartTime) {
                    const duration = (now - this.state.speechStartTime) / 1000;
                    console.log(`\n🤫 Speech ended (${duration.toFixed(1)}s)`);
                }
                this.state.speechStartTime = null;
            }
            
            // Debug output
            if (now - this.lastDebugTime >= 5000) {
                if (this.state.isSpeech) {
                    const duration = (now - (this.state.speechStartTime || now)) / 1000;
                    console.log(`\n🔍 VAD: Speech ongoing (${duration.toFixed(1)}s), prob: ${this.state.speechProbability.toFixed(2)}`);
                } else {
                    console.log(`\n🔍 VAD: Monitoring, prob: ${this.state.speechProbability.toFixed(2)}`);
                }
                this.lastDebugTime = now;
            }
            
            return {
                isSpeech: this.state.isSpeech,
                probability: this.state.speechProbability
            };
        } catch (error) {
            console.error("❌ VAD Error:", error);
            return { isSpeech: false, probability: 0 };
        }
    }
}

class SpeechRecognizer {
    constructor() {
        if (!fs.existsSync(config.MODEL_PATH)) {
            throw new Error(`Model not found: ${config.MODEL_PATH}`);
        }
        
        this.model = new vosk.Model(config.MODEL_PATH);
        this.recognizer = new vosk.Recognizer({
            model: this.model,
            sampleRate: config.SAMPLE_RATE
        });
        
        this.audioProcessor = new AudioProcessor();
        this.vadProcessor = new VadProcessor();
        this.speechBuffer = [];
        this.lastProcessedText = '';
        this.isListening = true;
        this.currentTTS = null;
        
        // Create output directory
        if (!fs.existsSync('./output')) {
            fs.mkdirSync('./output', { recursive: true });
        }
    }

    async start() {
        console.log("🎤 Starting speech recognition...");
        
        const recordProcess = spawn('rec', [
            '-q',
            '-t', 'raw',
            '-r', String(config.SAMPLE_RATE),
            '-b', '16',
            '-c', '2',
            '-e', 'signed-integer',
            '-'
        ]);
        
        // Create streaming transform
        const processChunk = new Transform({
            transform: async (chunk, encoding, callback) => {
                try {
                    const processed = this.audioProcessor.processFormat(chunk);
                    if (processed) {
                        const vadResult = await this.vadProcessor.processFrame(processed.audio);
                        
                        if (vadResult.isSpeech || vadResult.probability > config.SILENCE_THRESHOLD) {
                            this.speechBuffer.push(processed.audio);
                            
                            // Process if we have enough audio
                            if (Buffer.concat(this.speechBuffer).length >= config.CHUNK_SIZE * 10) {
                                await this.processSpeech();
                            }
                        } else if (this.speechBuffer.length > 0) {
                            // Process remaining buffer on silence
                            await this.processSpeech();
                            this.speechBuffer = [];
                        }
                    }
                    callback();
                } catch (error) {
                    console.error("Processing error:", error);
                    callback();
                }
            }
        });
        
        // Pipe audio through processor
        recordProcess.stdout.pipe(processChunk);
        
        // Handle process events
        recordProcess.on('error', error => {
            console.error("Recording error:", error);
            this.stop();
        });
        
        process.on('SIGINT', () => {
            console.log("\n🛑 Stopping...");
            this.stop();
        });
    }

    async processSpeech() {
        if (this.speechBuffer.length === 0) return;
        
        const audioData = Buffer.concat(this.speechBuffer);
        
        // Get partial results
        const partial = JSON.parse(this.recognizer.partialResult());
        if (partial.partial) {
            const newText = partial.partial.replace(this.lastProcessedText, '').trim();
            if (newText && newText.split(' ').length >= 3) {
                console.log(`\n✏️ Partial: ${newText}`);
                await this.processResponse(newText);
            }
        }
        
        // Process complete utterance
        if (this.recognizer.acceptWaveform(audioData)) {
            const result = JSON.parse(this.recognizer.result());
            if (result.text) {
                const newText = result.text.replace(this.lastProcessedText, '').trim();
                if (newText) {
                    console.log(`\n📝 Transcribed: ${newText}`);
                    this.lastProcessedText = result.text;
                    await this.processResponse(newText);
                }
            }
        }
    }

    async processResponse(text) {
        if (!text) return;
        
        try {
            const response = await axios.post(config.API_URL, 
                { user_message: text },
                { 
                    headers: {
                        'Content-Type': 'application/json',
                        'X-API-KEY': config.API_KEY
                    },
                    timeout: 10000
                }
            );
            
            if (response.data?.synthia_response) {
                const sentences = response.data.synthia_response.match(/[^.!?]+[.!?]+/g) || 
                                [response.data.synthia_response];
                
                for (const sentence of sentences) {
                    if (sentence.trim()) {
                        console.log(`\n🤖 Synthia: ${sentence.trim()}`);
                        await this.synthesizeSpeech(sentence.trim());
                    }
                }
            }
        } catch (error) {
            console.error("❌ API Error:", error.message);
        }
    }

    async synthesizeSpeech(text) {
        if (!text) return;
        
        try {
            // Stop any current TTS
            if (this.currentTTS) {
                this.currentTTS.kill();
                this.currentTTS = null;
            }
            
            const outputFile = path.join('output', `speech_${Date.now()}.mp3`);
            console.log(`\n🔊 Speaking: ${text}`);
            
            // Generate speech
            await new Promise((resolve, reject) => {
                const ttsProcess = spawn(
                    path.join(__dirname, '.venv/bin/edge-tts'),
                    [
                        '--voice', process.env.SYNTHIA_VOICE || config.DEFAULT_VOICE,
                        '--rate', '-10%',
                        '--text', text,
                        '--write-media', outputFile
                    ]
                );
                
                ttsProcess.on('error', reject);
                ttsProcess.on('exit', code => {
                    if (code === 0) resolve();
                    else reject(new Error(`TTS failed with code ${code}`));
                });
            });
            
            // Play audio
            await new Promise((resolve, reject) => {
                this.currentTTS = spawn('afplay', ['-q', '1', outputFile]);
                this.currentTTS.on('error', reject);
                this.currentTTS.on('exit', code => {
                    this.currentTTS = null;
                    if (code === 0) resolve();
                    else reject(new Error(`Playback failed with code ${code}`));
                });
            });
            
            // Cleanup
            fs.unlinkSync(outputFile);
            
        } catch (error) {
            console.error("❌ TTS Error:", error);
            this.currentTTS = null;
            // Fallback to say command
            try {
                await new Promise(resolve => {
                    const say = spawn('say', [text]);
                    say.on('exit', resolve);
                });
            } catch (e) {
                console.error("❌ Fallback TTS failed:", e);
            }
        }
    }

    stop() {
        this.isListening = false;
        if (this.currentTTS) {
            this.currentTTS.kill();
        }
        if (this.recognizer) {
            this.recognizer.free();
        }
        if (this.model) {
            this.model.free();
        }
        process.exit(0);
    }
}

// Main execution
(async () => {
    try {
        const recognizer = new SpeechRecognizer();
        await recognizer.start();
    } catch (error) {
        console.error("❌ Fatal error:", error);
        process.exit(1);
    }
})();