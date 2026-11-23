/* Practice voice helpers. A long spoken answer is transcribed in pieces the local transcriber
 * accepts, cut at the quietest moment near each limit so no word is split. A long message for
 * the interviewer's voice is read in pieces too, cut between sentences, so none of it is dropped.
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

/* Split a message into pieces of at most `maxChars`, breaking between sentences where possible
   and between words otherwise. Fenced code blocks are never spoken, so they are dropped here: split
   across two pieces they would no longer look like blocks to the voice. An inline code span is kept
   whole for the same reason. */
export function splitForSpeech(text, maxChars) {
  const pieces = [];
  let current = "";
  const sentences = String(text || "")
    .replace(/```[\s\S]*?```/g, " ")
    .split(/(?<=[.!?])\s+|\n+/)
    .map((sentence) => sentence.trim())
    .filter(Boolean);
  const units = [];
  for (const sentence of sentences) {
    const open = units.length > 0 && (units[units.length - 1].split("`").length - 1) % 2 === 1;
    if (open) units[units.length - 1] += ` ${sentence}`;
    else units.push(sentence);
  }
  for (const unit of units) {
    const joined = current ? `${current} ${unit}` : unit;
    if (joined.length <= maxChars) {
      current = joined;
      continue;
    }
    if (current) pieces.push(current);
    let rest = unit;
    while (rest.length > maxChars) {
      const breakAt = rest.lastIndexOf(" ", maxChars);
      const cut = breakAt > 0 ? breakAt : maxChars;
      pieces.push(rest.slice(0, cut).trim());
      rest = rest.slice(cut).trim();
    }
    current = rest;
  }
  if (current) pieces.push(current);
  return pieces;
}
