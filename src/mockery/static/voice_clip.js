/* Spoken-take helpers shared by the Practice mic: a long answer is transcribed in pieces
 * the local transcriber accepts, cut at the quietest moment near each limit so no word is split.
 */

export const ASR_SAMPLE_RATE = 16000;
/* One upload stays under the transcriber's 4 MB limit (16-bit mono at 16 kHz). */
export const ASR_PIECE_SECONDS = 110;
/* How far back from a limit to look for a pause to cut at. */
export const ASR_PAUSE_SEARCH_SECONDS = 8;

const FRAME_SAMPLES = ASR_SAMPLE_RATE / 10;

export function splitClipForAsr(
  samples,
  {
    maxSamples = ASR_PIECE_SECONDS * ASR_SAMPLE_RATE,
    searchSamples = ASR_PAUSE_SEARCH_SECONDS * ASR_SAMPLE_RATE,
    frameSamples = FRAME_SAMPLES,
  } = {}
) {
  const pieces = [];
  let start = 0;
  while (samples.length - start > maxSamples) {
    const limit = start + maxSamples;
    const searchFrom = Math.max(start + frameSamples, limit - searchSamples);
    let cut = limit;
    let quietest = Infinity;
    for (let frame = searchFrom; frame + frameSamples <= limit; frame += frameSamples) {
      let energy = 0;
      for (let i = frame; i < frame + frameSamples; i++) energy += Math.abs(samples[i]);
      if (energy < quietest) {
        quietest = energy;
        cut = frame + (frameSamples >> 1);
      }
    }
    pieces.push(samples.subarray(start, cut));
    start = cut;
  }
  // A closing sliver shorter than one frame cannot hold a word.
  if (samples.length - start >= frameSamples || pieces.length === 0) {
    pieces.push(samples.subarray(start));
  }
  return pieces;
}
