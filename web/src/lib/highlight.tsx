/** Mirrors `stageground.evaluation.normalize._normalize_for_match` (Python):
 * lowercase, turn `.`/`,` into whitespace, collapse runs of whitespace to a
 * single space. The backend's `evidence_span_found` check is OCR-noise-
 * tolerant via this exact normalization -- a naive client-side substring
 * search would report "not found" for spans the backend correctly found
 * (report text has PDF line-wrap artifacts like "metastases of a poorly.
 * differentiated carcinoma" that the evidence quote doesn't reproduce
 * verbatim). `map[i]` is the original-string index the normalized
 * character at `i` came from, so a match in normalized space can be
 * projected back onto the original text for highlighting. */
function normalizeWithMap(s: string): { normalized: string; map: number[] } {
  let normalized = "";
  const map: number[] = [];
  let inWhitespaceRun = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    const isWhitespaceEquivalent = ch === "." || ch === "," || /\s/.test(ch);
    if (isWhitespaceEquivalent) {
      if (!inWhitespaceRun) {
        normalized += " ";
        map.push(i);
        inWhitespaceRun = true;
      }
    } else {
      normalized += ch.toLowerCase();
      map.push(i);
      inWhitespaceRun = false;
    }
  }
  return { normalized, map };
}

/** Locate `needle` inside `text` using the same OCR-noise-tolerant
 * normalization as the backend's grounding check, then project the match
 * back onto the original (unnormalized) text so it can be highlighted.
 * Returns the three parts to render (before/match/after) rather than JSX
 * directly, so the caller can attach a ref to the match element itself --
 * passing a ref through a plain helper function trips the
 * react-hooks/refs lint rule. */
export function findSpan(
  text: string,
  needle: string | null
): { before: string; match: string; after: string } | null {
  if (!needle) return null;
  const { normalized: normalizedNeedle } = normalizeWithMap(needle);
  const trimmedNeedle = normalizedNeedle.trim();
  if (!trimmedNeedle) return null;

  const { normalized: normalizedText, map } = normalizeWithMap(text);
  const idx = normalizedText.indexOf(trimmedNeedle);
  if (idx === -1) return null;

  const start = map[idx];
  const end = map[idx + trimmedNeedle.length - 1] + 1;
  return {
    before: text.slice(0, start),
    match: text.slice(start, end),
    after: text.slice(end),
  };
}
