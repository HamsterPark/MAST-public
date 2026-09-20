// `**强调**` — the only markup the backend's operator-facing copy uses.
//
// The instrument-init catalog (mast/core/instrument_init.py) writes its warnings
// with markdown emphasis: 「**运行时有兜底**：退针分级进行…」. The page rendered
// that text raw, so the operator read literal asterisks in the middle of the one
// sentence that stops a tip being driven into the sample. called the
// page「不专业」and this is a large part of why.
//
// A full markdown renderer is the wrong tool: react-markdown emits block
// elements with their own spacing, and these strings sit inline inside a label.
// So: split, and let the caller decide the tag.
//
// Deliberately NOT supporting `*single*`. Scientific copy is full of bare
// asterisks (footnote marks, `a*` states, glob patterns), and swallowing one
// into emphasis would silently delete a character from a physics label.

export interface BoldSpan {
  text: string;
  bold: boolean;
}

export function splitBold(source: string | null | undefined): BoldSpan[] {
  const s = source ?? "";
  if (!s) return [];
  const out: BoldSpan[] = [];
  let i = 0;
  while (i < s.length) {
    const open = s.indexOf("**", i);
    if (open < 0) break;
    const close = s.indexOf("**", open + 2);
    // An unclosed `**` is literal text, not the start of emphasis running to the
    // end of the string — half-typed markup must not reflow the whole sentence.
    if (close < 0) break;
    if (open > i) out.push({ text: s.slice(i, open), bold: false });
    if (close === open + 2) {
      // `****` — no content. Keep it as text rather than emitting an empty span.
      out.push({ text: "****", bold: false });
      i = close + 2;
      continue;
    }
    out.push({ text: s.slice(open + 2, close), bold: true });
    i = close + 2;
  }
  if (i < s.length) out.push({ text: s.slice(i), bold: false });
  return out.filter((sp) => sp.text.length > 0);
}

/** The text with markup removed — for `title=`, `aria-label=`, and searching. */
export function stripBold(source: string | null | undefined): string {
  return splitBold(source)
    .map((s) => s.text)
    .join("");
}
