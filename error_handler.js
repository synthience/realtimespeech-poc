const ErrorHandler = {
    checkSystemRequirements() {
        // Check for required commands
        const requiredCommands = ['sox', 'rec', 'edge-tts', 'afplay'];
        const missing = [];
        
        for (const cmd of requiredCommands) {
            try {
                require('child_process').execSync(`which ${cmd}`);
            } catch (error) {
                missing.push(cmd);
            }
        }
        
        if (missing.length > 0) {
            console.error('\n❌ Missing required commands:');
            console.error(missing.join(', '));
            console.error('\nPlease install the missing dependencies:');
            console.error('1. For sox and rec: brew install sox');
            console.error('2. For edge-tts: pip install edge-tts');
            console.error('3. afplay should be available on macOS by default\n');
            process.exit(1);
        }
        
        // Check for Vosk model
        const fs = require('fs');
        const MODEL_PATH = './vosk-model-en-us-0.42-gigaspeech';
        if (!fs.existsSync(MODEL_PATH)) {
            console.error('\n❌ Vosk model not found!');
            console.error('Please run: ./run_node.sh to download and set up the model\n');
            process.exit(1);
        }
        
        // Check for output directory
        if (!fs.existsSync('./output')) {
            try {
                fs.mkdirSync('./output');
            } catch (error) {
                console.error('\n❌ Could not create output directory:', error);
                process.exit(1);
            }
        }
        
        return true;
    },
    
    handleProcessError(error, process) {
        console.error(`\n❌ Process error in ${process}:`, error);
        return false;
    },
    
    handleAudioError(error) {
        if (error.message.includes('No such file or directory')) {
            console.error('\n❌ Audio device not found. Please check your microphone connection.');
        } else if (error.message.includes('Permission denied')) {
            console.error('\n❌ Permission denied accessing the microphone.');
            console.error('Please grant microphone access and try again.');
        } else {
            console.error('\n❌ Audio error:', error);
        }
        return false;
    },
    
    handleVADError(error) {
        console.error('\n❌ Voice Activity Detection error:', error);
        console.error('Attempting to reinitialize VAD...');
        return false;
    },
    
    handleAPIError(error) {
        if (error.code === 'ECONNREFUSED') {
            console.error('\n❌ Could not connect to API server.');
            console.error('Please ensure the API server is running at:', error.address);
        } else if (error.response) {
            console.error('\n❌ API error:', error.response.status, error.response.statusText);
            console.error('Message:', error.response.data);
        } else {
            console.error('\n❌ API error:', error.message);
        }
        return false;
    },
    
    handleTTSError(error) {
        if (error.message.includes('edge-tts')) {
            console.error('\n❌ Text-to-Speech error: edge-tts failed');
            console.error('Falling back to system TTS...');
            return true; // Allow fallback
        } else if (error.message.includes('afplay')) {
            console.error('\n❌ Audio playback error');
            console.error('Please check your audio output settings.');
        } else {
            console.error('\n❌ TTS error:', error);
        }
        return false;
    }
};

module.exports = ErrorHandler;