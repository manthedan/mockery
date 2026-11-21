class PCM16DownsamplerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetSampleRate = opts.targetSampleRate || 16000;
    this.chunkSamples = opts.chunkSamples || 1600; // 100ms @ 16kHz
    this.inputRate = sampleRate;
    this.ratio = this.inputRate / this.targetSampleRate;
    // 99% of macOS audio hardware runs at 48 kHz -> exact 3:1 integer decimation
    this.isRatio3 = this.inputRate === this.targetSampleRate * 3;

    this.outBuffer = new Int16Array(this.chunkSamples);
    this.outIndex = 0;
    this._monoBuf = new Float32Array(256);
    this._carryLen = 0;
    this.readPos = 0;
    this.sumSq = 0;
    this.meterCounter = 0;
  }

  _emitChunk() {
    const rms = Math.sqrt(this.sumSq / Math.max(1, this.meterCounter));
    this.sumSq = 0;
    this.meterCounter = 0;

    // Transfer the completed buffer directly with zero copy; allocate next 100ms slab (10x/sec)
    const finished = this.outBuffer;
    this.outBuffer = new Int16Array(this.chunkSamples);
    this.outIndex = 0;
    this.port.postMessage({ pcmBuffer: finished.buffer, rms }, [finished.buffer]);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) {
      return true;
    }

    const ch0 = input[0];
    const ch1 = input.length > 1 ? input[1] : null;
    const inLen = ch0.length;
    const carryLen = this._carryLen;
    const totalLen = carryLen + inLen;

    // Grow scratch buffer only if quantum size ever exceeds initial capacity (never at 128 samples)
    if (totalLen > this._monoBuf.length) {
      const next = new Float32Array(totalLen + 64);
      if (carryLen > 0) {
        next.set(this._monoBuf.subarray(0, carryLen), 0);
      }
      this._monoBuf = next;
    }

    const mono = this._monoBuf;
    if (ch1) {
      for (let i = 0; i < inLen; i++) {
        mono[carryLen + i] = 0.5 * (ch0[i] + ch1[i]);
      }
    } else {
      mono.set(ch0, carryLen);
    }

    if (this.isRatio3) {
      const INV3 = 1 / 3;
      let pos = 0;
      const limit = totalLen - 2;
      while (pos < limit) {
        const sample = (mono[pos] + mono[pos + 1] + mono[pos + 2]) * INV3;
        pos += 3;
        this.sumSq += sample * sample;
        this.meterCounter++;

        const clamped = sample < -1 ? -1 : sample > 1 ? 1 : sample;
        this.outBuffer[this.outIndex++] =
          clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);

        if (this.outIndex >= this.chunkSamples) {
          this._emitChunk();
        }
      }

      const rem = totalLen - pos;
      for (let i = 0; i < rem; i++) {
        mono[i] = mono[pos + i];
      }
      this._carryLen = rem;
      return true;
    }

    // General fractional boxcar filter across [start, end) for non-48kHz rates (e.g. 44.1kHz, 96kHz)
    while (this.readPos + this.ratio <= totalLen) {
      const start = this.readPos;
      const end = start + this.ratio;
      const i0 = Math.floor(start);
      const i1 = Math.min(totalLen, Math.ceil(end));

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

      const clamped = sample < -1 ? -1 : sample > 1 ? 1 : sample;
      this.outBuffer[this.outIndex++] =
        clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff);

      if (this.outIndex >= this.chunkSamples) {
        this._emitChunk();
      }

      this.readPos = end;
    }

    const dropCount = Math.floor(this.readPos);
    const rem = totalLen - dropCount;
    if (rem > 0 && dropCount > 0) {
      mono.copyWithin(0, dropCount, totalLen);
    }
    this._carryLen = rem > 0 ? rem : 0;
    this.readPos -= dropCount;
    return true;
  }
}

registerProcessor("pcm16-downsampler", PCM16DownsamplerProcessor);
