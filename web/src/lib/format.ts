export function pct(x: number, digits = 1): string {
  return `${(x * 100).toFixed(digits)}%`;
}

export function pp(x: number, digits = 2): string {
  const v = x * 100;
  const sign = v > 0 ? "+" : v < 0 ? "−" : "±";
  return `${sign}${Math.abs(v).toFixed(digits)} pp`;
}

export function ciLabel(ciLow: number, ciHigh: number, digits = 1): string {
  return `[${(ciLow * 100).toFixed(digits)}, ${(ciHigh * 100).toFixed(digits)}] pp`;
}

export function pLabel(pValue: number, nBoot: number): string {
  if (pValue <= 1 / nBoot) return `p < 1/${nBoot}`;
  return `p = ${pValue.toFixed(4)}`;
}
