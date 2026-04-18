/**
 * AudioWorklet Processor for real-time microphone PCM capture.
 * Runs at native sample rate (usually 48kHz) and downsamples to 16kHz
 * for CX STT Engine compatibility.
 */
class STTProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = [];
        this._bufferSize = 2048; // Flush every 2048 samples at 16kHz (~128ms chunks) - fast enough for single short words
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || input.length === 0) return true;
        
        const channelData = input[0]; // Mono channel
        if (!channelData) return true;

        // Downsample from native rate (48000) to 16000 Hz
        // Ratio = 48000 / 16000 = 3, so we pick every 3rd sample
        const ratio = Math.round(sampleRate / 16000);
        
        for (let i = 0; i < channelData.length; i += ratio) {
            // Convert Float32 [-1.0, 1.0] to Int16 [-32768, 32767]
            let sample = channelData[i];
            sample = Math.max(-1, Math.min(1, sample)); // Clamp
            this._buffer.push(sample < 0 ? sample * 0x8000 : sample * 0x7FFF);
        }

        // When buffer is full, flush it to the main thread
        if (this._buffer.length >= this._bufferSize) {
            const int16Array = new Int16Array(this._buffer.splice(0, this._bufferSize));
            this.port.postMessage({
                type: 'audio',
                samples: int16Array.buffer
            }, [int16Array.buffer]); // Transfer ownership for zero-copy
        }

        return true; // Keep processor alive
    }
}

registerProcessor('stt-processor', STTProcessor);
