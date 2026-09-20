/**
 * Engineering-notation SI formatting for live instrument readouts.
 *
 * Why this exists
 * ------------------------------------------------------
 * The instrument panel printed the tunnelling current as ``9.63e-11 A``.
 * Operators do not read STM currents in scientific notation — they read them
 * in pA and nA, and the panel is the thing they glance at while the tip is
 * down. Being asked to decode ``e-11`` at that moment is not a cosmetic
 * complaint; it is the readout failing at its one job.
 *
 * The rule here is ENGINEERING notation: pick the SI prefix that puts the
 * mantissa in [1, 1000), so 9.63e-11 A → ``96.3 pA`` and 1e-9 A → ``1.00 nA``.
 * Not "always pA": a 1 nA setpoint rendered as ``1000.0 pA`` (which the panel
 * also did) is the same failure with extra digits.
 *
 * Deliberate non-goals:
 *   * No rounding to a "nicer" number — a readout must not invent precision it
 *     does not have, nor hide a value drifting off a setpoint.
 *   * Exponential is still the fallback below femto / above tera. Outside the
 *     prefix table, ``0.0001 fA`` would be a worse lie than ``1e-19 A``.
 */

/** SI prefixes, largest first. Index 4 (``""``) is unity. */
const PREFIXES: ReadonlyArray<readonly [number, string]> = [
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
  [1, ""],
  [1e-3, "m"],
  [1e-6, "µ"],
  [1e-9, "n"],
  [1e-12, "p"],
  [1e-15, "f"],
];

export interface SIOptions {
  /** Significant digits in the mantissa (default 3 → ``96.3 pA``). */
  digits?: number;
  /** Rendered when the value is null/undefined/NaN. */
  placeholder?: string;
}

/**
 * Format *value* (in base SI units) with an engineering prefix.
 *
 * @example fmtSI(9.63e-11, "A")  // "96.3 pA"
 * @example fmtSI(1e-9, "A")      // "1.00 nA"
 * @example fmtSI(0, "A")         // "0 A"
 */
export function fmtSI(
  value: number | null | undefined,
  unit: string,
  opts: SIOptions = {},
): string {
  const { digits = 3, placeholder = "---" } = opts;
  if (value == null || !Number.isFinite(value)) return placeholder;
  if (value === 0) return `0 ${unit}`;

  const abs = Math.abs(value);
  const entry = PREFIXES.find(([scale]) => abs >= scale);
  if (!entry) {
    // Below femto — no prefix is honest here; exponential is.
    return `${value.toExponential(2)} ${unit}`;
  }
  const [scale, prefix] = entry;
  const mantissa = value / scale;

  // toPrecision keeps SIGNIFICANT digits, so 96.3 and 1.00 both read naturally
  // and a value never gains fake precision. Guard the >=1000 T case, where
  // toPrecision would flip to its own exponential form.
  if (Math.abs(mantissa) >= 1000) return `${value.toExponential(2)} ${unit}`;
  return `${mantissa.toPrecision(digits)} ${prefix}${unit}`;
}

/**
 * Same value, but with the number and the (prefixed) unit as separate strings.
 *
 * The top bar renders the two in different type styles, so it needs the split
 * — and it needs the PREFIXED unit ("pA", not "A"), which is exactly what the
 * old hard-coded ``<span>pA</span>`` got wrong the moment the current left the
 * picoamp decade.
 */
export function splitSI(
  value: number | null | undefined,
  unit: string,
  opts: SIOptions = {},
): { num: string; unit: string } {
  const text = fmtSI(value, unit, { ...opts, placeholder: opts.placeholder ?? "—" });
  const cut = text.lastIndexOf(" ");
  if (cut < 0) return { num: text, unit: "" };
  return { num: text.slice(0, cut), unit: text.slice(cut + 1) };
}

/**
 * Tunnelling current. Same as :func:`fmtSI` with the STM-sane default: pA/nA
 * are what the panel, the setpoint field and every operator conversation use.
 */
export function fmtCurrent(a: number | null | undefined, opts?: SIOptions): string {
  return fmtSI(a, "A", opts);
}

/** Z position / scan extents — nm and µm, never metres-with-an-exponent. */
export function fmtLength(m: number | null | undefined, opts?: SIOptions): string {
  return fmtSI(m, "m", opts);
}

/**
 * Bias. NOT engineering-formatted: bias is set and read in volts across the
 * whole useful range (mV to ±10 V), and an operator watching a ramp wants a
 * stable column of digits, not a prefix that flips at 1 mV. Sub-mV falls back
 * to the prefix form so a near-zero bias does not read as a flat ``0.0000``.
 */
export function fmtBias(v: number | null | undefined, opts: SIOptions = {}): string {
  const { placeholder = "---" } = opts;
  if (v == null || !Number.isFinite(v)) return placeholder;
  if (v !== 0 && Math.abs(v) < 1e-3) return fmtSI(v, "V", opts);
  return `${v.toFixed(4)} V`;
}
