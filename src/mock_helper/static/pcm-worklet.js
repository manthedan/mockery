class PCM16DownsamplerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetSampleRate = opts.targetSampleRate || 16000;
    this.chunkSamples = opts.chunkSamples || 1600; // 100ms @ 16kHz
    this.inputRate = sampleRate;
    this.ratio = this.inputRate / this.targetSampleRate;

    this.outBuffer = new Int16Array(this.chunkSamples);
    this.outIndex = 0;
    this.carry = new Float32Array(0);
    this.readPos = 0;
    this.sumSq = 0;
    this.meterCounter = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) {
      return true;
    }

    const ch0 = input[0];
    const ch1 = input.length > 1 ? input[1] : null;
    const inLen = ch0.length;
    const carryLen = this.carry.length;
    const totalLen = carryLen + inLen;

    // Combine residual samples from prior quantum with current mono mix
    const mono = new Float32Array(totalLen);
    if (carryLen > 0) {
      mono.set(this.carry, 0);
    }
    if (ch1) {
      for (let i = 0; i < inLen; i++) {
        mono[carryLen + i] = 0.5 * (ch0[i] + ch1[i]);
      }
    } else {
      mono.set(ch0, carryLen);
    }

    // Only emit an output sample when the full averaging window [readPos, readPos + ratio) is available
    while (this.readPos + this.ratio <= totalLen) {
      const start = this.readPos;
      const end = start + this.ratio;
      const i0 = Math.floor(start);
      const i1 = Math.min(totalLen, Math.ceil(end));

      // Fractional boxcar filter across [start, end)
      let acc = 0;
      let weight = 0;
      for (let j = i0; j < i1; j++) {
        const w = Math.min(j + 1, end) - Math.max(j, start);
        if (w > 0) {
          acc += mono[j] * w;
          weight += w;
        }
      }
      const sample = weight > 0 ? acc / weight : mono[i0];
      this.sumSq += sample * sample;
      this.meterCounter++;

      const clamped = Math.max(-1, Math.min(1, sample));
      this.outBuffer[this.outIndex++] =
        clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);

      if (this.outIndex >= this.chunkSamples) {
        const rms = Math.sqrt(this.sumSq / Math.max(1, this.meterCounter));
        this.sumSq = 0;
        this.meterCounter = 0;

        const copy = new Int16Array(this.outBuffer);
        this.port.postMessage(
          { pcmBuffer: copy.buffer, rms },
          [copy.buffer]
        );
        this.outIndex = 0;
      }

      this.readPos = end;
    }

    const dropCount = Math.floor(this.readPos);
    this.carry = dropCount < totalLen ? mono.slice(dropCount) : new Float32Array(0);
    this.readPos -= dropCount;
    return true;
  }
}

registerProcessor("pcm16-downsampler", PCM16DownsamplerProcessor);
