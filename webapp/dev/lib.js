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
//
// 2026-10-08: trough (#992288) and tropical_trough (#bb1111) added for
// nbs_square_dilation_trof_tt_no_inst. The reference palette's remaining
// slots (magenta/green/red) all collapse against aqua or orange under CVD
// simulation, so these were searched for directly: every pair across all
// seven classes stays >= 9.5 OKLab dE under normal, protan, deutan and
// tritan simulation (trough alone >= 15.6), with OKLCH L ~0.48-0.50.
export const CLASS_STYLE = {
  cold: { color: "#3987e5", label: "Cold" },
  warm: { color: "#d95926", label: "Warm" },
  stationary: { color: "#199e70", label: "Stationary" },
  occluded: { color: "#9085e9", label: "Occluded" },
  dryline: { color: "#c98500", label: "Dryline" },
  trough: { color: "#992288", label: "Trough" },
  tropical_trough: { color: "#bb1111", label: "Tropical trough" },
};

// Raw-probability breakpoints for the filled-contour look. See index.html's
// customClassFrag for how these drive per-pixel alpha.
export const FRONT_DISCARD_BELOW = 0.05; // below this = background noise, not a front
export const FRONT_BIN_WIDTH = 0.1;
export const FRONT_N_BINS = 10; // 0.1-0.2, 0.2-0.3, ... 0.9-1.0, plus the 0.05-0.1 bin

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
// edge up to 1.0 -- e.g. [0.05, 0.1, 0.2, ..., 1.0].
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
// every other boundary is labeled: [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]. Returns
// {value, index} pairs -- `index` is the position in frontBinBoundaries()'s
// array, which the caller needs to place the label at the matching
// `i / FRONT_N_BINS` fraction of the colorbar's width.
export function labeledTickIndices() {
  const boundaries = frontBinBoundaries();
  return boundaries
    .map((value, index) => ({ value, index }))
    .filter(({ index }) => index % 2 === 0);
}

// FRONT_DISCARD_BELOW (0.05) needs 2 decimal places to be legible; every
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

// ---------------------------------------------------------------------------
// Front-lines overlay (the optional "Front lines" checkbox).
//
// The scheduler derives smooth front lines from each step's probability grids
// (frontfinder/postprocess/front_lines.py) and writes one GeoJSON per step,
// listed as `fronts` on that step's latest.json entry. Each Feature is a
// MultiLineString named for its type (`cold_front`, `trough`, ...). Pure
// helpers only here -- no DOM/map access -- so lib.test.js can cover them.
// ---------------------------------------------------------------------------

// GeoJSON feature `name` -> CLASS_STYLE key.
export const FRONT_LINE_CLASS_BY_NAME = {
  cold_front: "cold",
  warm_front: "warm",
  stationary_front: "stationary",
  occluded_front: "occluded",
  dryline: "dryline",
  trough: "trough",
  tropical_trough: "tropical_trough",
};

// Troughs and drylines are not fronts in the strict sense; drawing them dashed
// keeps the two readable apart where both are on at once (and when color alone
// is not enough). MapLibre's `line-dasharray` cannot be data-driven, so these
// get their own layer.
export const DASHED_FRONT_LINE_CLASSES = ["dryline", "trough", "tropical_trough"];

// Where a step's front-lines file lives, or null if this step has none
// (cycles published before the feature, or a step whose lines failed).
export function frontLinesUrl(zarrRoot, modelKey, step) {
  if (!step || typeof step.fronts !== "string" || step.fronts === "") return null;
  return `${zarrRoot}/${modelKey}/${step.fronts}`;
}

// Turns a step's front-lines file into what the map style reads: one
// LineString per line part (MapLibre places symbols along single lines, and
// each part has its own symbol side), tagged with `cls` (a CLASS_STYLE key)
// and `side` ("left" or "right": which side of the line its symbols go on,
// walking from its first point to its last; see postprocess/splines.py).
// Drops anything unrecognized, and the per-feature `summary`. Returns a new
// FeatureCollection; the input is not modified. `oriented` (whether the sides
// were chosen from real temperature/wind fields, see front_lines.py) is copied
// to the collection and every feature: symbols are only drawn when it is true,
// because sides from motion alone can be wrong.
export function withLineClasses(collection) {
  const oriented = collection?.oriented === true;
  const features = [];
  for (const feature of collection?.features ?? []) {
    const cls = FRONT_LINE_CLASS_BY_NAME[feature?.properties?.name];
    if (!cls || feature?.geometry?.type !== "MultiLineString") continue;
    const sides = feature.properties.summary?.symbol_sides ?? [];
    feature.geometry.coordinates.forEach((coordinates, i) => {
      if (!Array.isArray(coordinates) || coordinates.length < 2) return;
      features.push({
        type: "Feature",
        geometry: { type: "LineString", coordinates },
        properties: { event_id: feature.properties.event_id, name: feature.properties.name, cls, side: sides[i] === "right" ? "right" : "left", oriented },
      });
    });
  }
  return { type: "FeatureCollection", oriented, features };
}

// MapLibre `line-color` expression: the same per-class colors as the
// probability layers and the legend.
export function lineColorExpression() {
  const expr = ["match", ["get", "cls"]];
  // Stationary fronts are never green on the lines: the green is only the
  // raster/legend color. Unoriented ones (no symbols) use the cold blue.
  for (const [cls, style] of Object.entries(CLASS_STYLE)) expr.push(cls, cls === "stationary" ? CLASS_STYLE.cold.color : style.color);
  expr.push("#888888");
  return expr;
}

// MapLibre filter for the solid or dashed layer, limited to the classes whose
// checkbox is currently checked (`visibleClasses`: iterable of class keys).
export function lineFilter(dashed, visibleClasses) {
  const shown = [...visibleClasses].filter((cls) => DASHED_FRONT_LINE_CLASSES.includes(cls) === dashed);
  return ["in", ["get", "cls"], ["literal", shown]];
}

// ---------------------------------------------------------------------------
// Front symbols, drawn along the lines (WPC surface-analysis convention).
//
// `side` (see withLineClasses) is the side the pointing symbols go on, which is
// the way the front is moving:
//   cold        solid triangles, toward the warm air the front pushes into
//   warm        semicircles, toward the cold air the front pushes into
//   occluded    triangles and semicircles alternating, both on that side
//   stationary  triangles toward the warm air on `side`, and semicircles on the
//               opposite side (the cold air) -- the cold/warm front mix
// Troughs and drylines get no symbols (they are dashed lines).
//
// Each pattern is one repeating tile: an image `period` px wide with the line
// through its middle, and `symbols` placed along it. MapLibre repeats the tile
// along the line (`symbol-spacing` = the tile width), and "same"/"opposite" is
// relative to `side`. Colors are the viewer's class colors: cold blue, warm
// orange, occluded violet.
// ---------------------------------------------------------------------------
export const FRONT_SYMBOL_PATTERNS = {
  cold: { period: 26, symbols: [{ shape: "triangle", color: CLASS_STYLE.cold.color, at: 0.5, side: "same" }] },
  warm: { period: 26, symbols: [{ shape: "semicircle", color: CLASS_STYLE.warm.color, at: 0.5, side: "same" }] },
  occluded: {
    period: 52,
    symbols: [
      { shape: "triangle", color: CLASS_STYLE.occluded.color, at: 0.25, side: "same" },
      { shape: "semicircle", color: CLASS_STYLE.occluded.color, at: 0.75, side: "same" },
    ],
  },
  stationary: {
    period: 52,
    // The line itself is drawn in the tile too (blue under the triangle, red
    // under the semicircle, as on WPC charts), so no single-color line is
    // drawn under it.
    line: [
      { from: 0, to: 0.5, color: CLASS_STYLE.cold.color },
      { from: 0.5, to: 1, color: CLASS_STYLE.warm.color },
    ],
    symbols: [
      { shape: "triangle", color: CLASS_STYLE.cold.color, at: 0.25, side: "same" },
      { shape: "semicircle", color: CLASS_STYLE.warm.color, at: 0.75, side: "opposite" },
    ],
  },
};

// Filter for the single-color solid line layer: `lineFilter(false, ...)` minus
// the lines whose pattern tile already draws them (oriented stationary fronts).
export function solidLineFilter(visibleClasses) {
  const tiled = Object.entries(FRONT_SYMBOL_PATTERNS).filter(([, p]) => p.line).map(([cls]) => cls);
  return ["all", lineFilter(false, visibleClasses),
    ["!", ["all", ["in", ["get", "cls"], ["literal", tiled]], ["==", ["get", "oriented"], true]]]];
}

// Name of the map image holding `cls`'s tile for `side`, or null for a class
// that has no symbols.
export function symbolImageName(cls, side) {
  if (!(cls in FRONT_SYMBOL_PATTERNS)) return null;
  return `front-sym-${cls}-${side === "right" ? "right" : "left"}`;
}

// The classes whose tile is `period` px wide. A symbol layer has one
// `symbol-spacing`, so each distinct period needs its own layer.
export function symbolClassesWithPeriod(period) {
  return Object.entries(FRONT_SYMBOL_PATTERNS)
    .filter(([, pattern]) => pattern.period === period)
    .map(([cls]) => cls);
}

// Filter for the symbol layer of `period`: checked classes only, and only lines
// whose sides were chosen from real fields (`oriented`).
export function symbolFilter(period, visibleClasses) {
  const visible = new Set(visibleClasses);
  const classes = symbolClassesWithPeriod(period).filter((cls) => visible.has(cls));
  return ["all", ["in", ["get", "cls"], ["literal", classes]], ["==", ["get", "oriented"], true]];
}

// The distinct tile widths, ascending.
export function symbolPeriods() {
  return [...new Set(Object.values(FRONT_SYMBOL_PATTERNS).map((pattern) => pattern.period))].sort((a, b) => a - b);
}
