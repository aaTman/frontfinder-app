// Pure, DOM-free logic shared by index.html and lib.test.js.
//
// Colors: colorblind-safe categorical order, validated 2026-08-22 with the
// dataviz skill's validate_palette.js against this app's dark legend/panel
// surface (~#0f121a): `node scripts/validate_palette.js
// "#3987e5,#d95926,#199e70,#9085e9" --mode dark --surface "#0f121a"` passes
// every check on the default adjacent-pair gate (worst adjacent CVD ΔE 9.4,
// normal-vision ΔE 24.6). These are dark-column slots 1, 2, 3, and 7 (blue/
// orange/aqua/violet) of the skill's reference categorical palette --
// occluded moved from slot 4 (yellow, #c98500) to slot 7 (violet) to read
// as purple while keeping every check passing. The all-pairs gate (relevant
// for undifferentiated point clouds) is not the applicable gate here per
// the skill's own palette.md ("past three, fold to Other or facet") --
// accepted because every class also carries a legend swatch + text label +
// independent checkbox toggle, exactly the "secondary encoding" the skill
// requires to accept the 6-8 CVD band.
export const CLASS_STYLE = {
  cold: { color: "#3987e5", label: "Cold" },
  warm: { color: "#d95926", label: "Warm" },
  stationary: { color: "#199e70", label: "Stationary" },
  occluded: { color: "#9085e9", label: "Occluded" },
};

// Raw-probability breakpoints for the filled-contour look. See index.html's
// customClassFrag for how these drive per-pixel alpha.
export const FRONT_DISCARD_BELOW = 0.01; // below this = background noise, not a front
export const FRONT_BIN_WIDTH = 0.1;
export const FRONT_N_BINS = 10; // 0.1-0.2, 0.2-0.3, ... 0.9-1.0, plus the 0.01-0.1 bin

export function hexToRgb01(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

// Opacity assigned to a given bin index by customClassFrag's
// `bandOpacity = (bin + 1) / FRONT_N_BINS`.
export function bandOpacityForBin(bin) {
  return (bin + 1) / FRONT_N_BINS;
}

// Inverts bandOpacityForBin to recover which bin produced a given alpha.
export function binForBandOpacity(bandOpacity) {
  const bin = Math.round(bandOpacity * FRONT_N_BINS) - 1;
  return Math.max(0, Math.min(FRONT_N_BINS - 1, bin));
}

export function binLabel(bin) {
  const lo = bin === 0 ? FRONT_DISCARD_BELOW : bin * FRONT_BIN_WIDTH;
  const hi = (bin + 1) * FRONT_BIN_WIDTH;
  return `${lo.toFixed(2)}–${hi.toFixed(1)}`;
}

// The colorbar's tick values: the discard floor, then every bin's upper
// edge up to 1.0 -- e.g. [0.01, 0.1, 0.2, ..., 1.0].
export function frontBinBoundaries() {
  const boundaries = [FRONT_DISCARD_BELOW];
  for (let i = 1; i <= FRONT_N_BINS; i++) {
    boundaries.push(Number((i * FRONT_BIN_WIDTH).toFixed(10)));
  }
  return boundaries;
}

// Which of frontBinBoundaries()'s 11 boundaries get a numeric label under
// the colorbar. All 11 crowd and overlap at the legend card's width (~200px
// at 9px type -- confirmed by screenshotting the rendered legend), so only
// every other boundary is labeled: [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]. Returns
// {value, index} pairs -- `index` is the position in frontBinBoundaries()'s
// array, which the caller needs to place the label at the matching
// `i / FRONT_N_BINS` fraction of the colorbar's width.
export function labeledTickIndices() {
  const boundaries = frontBinBoundaries();
  return boundaries
    .map((value, index) => ({ value, index }))
    .filter(({ index }) => index % 2 === 0);
}

// FRONT_DISCARD_BELOW (0.01) needs 2 decimal places to be legible; every
// other boundary is a clean multiple of FRONT_BIN_WIDTH and only needs 1.
export function formatTickLabel(value) {
  return value < FRONT_BIN_WIDTH ? value.toFixed(2) : value.toFixed(1);
}

// Undoes customClassFrag's alpha math for one class, given the rendered
// pixel's RGB (0-1) and that class's pure hue (0-1): assumes the pixel is
// `white*(1-a) + classRgb*a` and solves for `a` via least-squares
// projection onto the (white - classRgb) direction.
export function estimateClassAlpha(pixelRgb01, classRgb01) {
  let num = 0;
  let den = 0;
  for (let i = 0; i < 3; i++) {
    const dir = 1 - classRgb01[i];
    const diff = 1 - pixelRgb01[i];
    num += diff * dir;
    den += dir * dir;
  }
  if (den === 0) return 0;
  return Math.max(0, Math.min(1, num / den));
}
