import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CLASS_STYLE,
  FRONT_DISCARD_BELOW,
  FRONT_BIN_WIDTH,
  FRONT_N_BINS,
  hexToRgb01,
  bandOpacityForBin,
  binForBandOpacity,
  binLabel,
  frontBinBoundaries,
  labeledTickIndices,
  formatTickLabel,
  estimateClassAlpha,
  FRONT_LINE_CLASS_BY_NAME,
  frontLinesUrl,
  withLineClasses,
  lineColorExpression,
  lineFilter,
  solidLineFilter,
  FRONT_SYMBOL_PATTERNS,
  symbolImageName,
  symbolClassesWithPeriod,
  symbolFilter,
  symbolPeriods,
  PRESSURE_COLORS,
  PRESSURE_FONT,
  PRESSURE_FONT_BOLD,
  PRESSURE_KINDS,
  pressureUrl,
  isobarFilter,
  centerFilter,
  isobarLabelFilter,
  centerColorExpression,
  withPressureKinds,
  pressureStatus,
  pressureInitialState,
  layerGridOptions,
  initializationLabel,
} from "./lib.js";

test("layerGridOptions keeps global defaults without a grid block", () => {
  const bounds = [-180.125, -90.125, 179.875, 90.125];
  assert.deepEqual(layerGridOptions(undefined, bounds, false), { bounds, latIsAscending: false });
});

test("layerGridOptions places a regional store in its own CRS", () => {
  const grid = {
    proj4: "+proj=lcc +lat_0=38.5",
    bounds: [-2699020, -1588806, 2676979, 1579193],
    lat_ascending: false,
    dimensions: { lat: "y", lon: "x" },
  };
  assert.deepEqual(layerGridOptions(grid, [0, 0, 1, 1], true), {
    bounds: grid.bounds,
    latIsAscending: false,
    proj4: grid.proj4,
    spatialDimensions: { lat: "y", lon: "x" },
  });
});

test("initializationLabel names the source and defaults to IFS", () => {
  assert.equal(initializationLabel("HRRR", "2026 Nov 20 12Z"), "HRRR Initialization 2026 Nov 20 12Z");
  assert.equal(initializationLabel(undefined, "2026 Nov 20 12Z"), "IFS Initialization 2026 Nov 20 12Z");
});

test("hexToRgb01 converts pure primaries and grays", () => {
  assert.deepEqual(hexToRgb01("#ff0000"), [1, 0, 0]);
  assert.deepEqual(hexToRgb01("#00ff00"), [0, 1, 0]);
  assert.deepEqual(hexToRgb01("#0000ff"), [0, 0, 1]);
  assert.deepEqual(hexToRgb01("#000000"), [0, 0, 0]);
  assert.deepEqual(hexToRgb01("#ffffff"), [1, 1, 1]);
});

test("CLASS_STYLE has one distinct color per class, colorblind-safe order", () => {
  const classes = Object.keys(CLASS_STYLE);
  assert.deepEqual(classes, ["cold", "warm", "stationary", "occluded", "dryline", "trough", "tropical_trough"]);
  const colors = classes.map((cls) => CLASS_STYLE[cls].color);
  // trough and tropical_trough share a color; line style (dashed vs solid) separates them.
  assert.equal(new Set(colors).size, colors.length - 1, "every class but the trough pair needs a distinct color");
  for (const color of colors) {
    assert.match(color, /^#[0-9a-f]{6}$/, `${color} must be a lowercase 6-digit hex string`);
  }
  // Locks in the specific colorblind-safe palette (dark-mode slots 1, 2, 3,
  // 7, 4 of the dataviz skill's reference categorical order, plus the two
  // CVD-searched trough colors -- see lib.js) so a future edit that silently
  // reverts to a non-validated set fails loudly.
  assert.deepEqual(colors, ["#3987e5", "#d62728", "#199e70", "#9085e9", "#c98500", "#8c8c8c", "#8c8c8c"]);
});

test("bandOpacityForBin and binForBandOpacity round-trip for every bin", () => {
  for (let bin = 0; bin < FRONT_N_BINS; bin++) {
    const opacity = bandOpacityForBin(bin);
    assert.equal(binForBandOpacity(opacity), bin);
  }
});

test("binForBandOpacity clamps out-of-range input", () => {
  assert.equal(binForBandOpacity(-1), 0);
  assert.equal(binForBandOpacity(0), 0);
  assert.equal(binForBandOpacity(2), FRONT_N_BINS - 1);
});

test("binLabel formats the first bin with the discard floor, others with clean multiples", () => {
  assert.equal(binLabel(0), "0.05–0.1");
  assert.equal(binLabel(1), "0.10–0.2");
  assert.equal(binLabel(FRONT_N_BINS - 1), "0.90–1.0");
});

test("frontBinBoundaries returns the discard floor plus every bin edge up to 1.0", () => {
  const boundaries = frontBinBoundaries();
  assert.equal(boundaries.length, FRONT_N_BINS + 1);
  assert.equal(boundaries[0], FRONT_DISCARD_BELOW);
  assert.equal(boundaries[1], FRONT_BIN_WIDTH);
  assert.equal(boundaries[boundaries.length - 1], 1);
  // Monotonically increasing -- ticks must read left-to-right.
  for (let i = 1; i < boundaries.length; i++) {
    assert.ok(boundaries[i] > boundaries[i - 1]);
  }
});

test("labeledTickIndices thins frontBinBoundaries to every other index, endpoints included", () => {
  const labeled = labeledTickIndices();
  assert.deepEqual(
    labeled.map((t) => t.value),
    [0.05, 0.2, 0.4, 0.6, 0.8, 1]
  );
  assert.deepEqual(
    labeled.map((t) => t.index),
    [0, 2, 4, 6, 8, 10]
  );
  // The last boundary (1.0, "full probability") must always be labeled --
  // an odd-length boundary list could otherwise drop it silently.
  assert.equal(labeled[labeled.length - 1].value, 1);
});

test("formatTickLabel gives the discard floor two decimals, others one", () => {
  assert.equal(formatTickLabel(0.05), "0.05");
  assert.equal(formatTickLabel(0.1), "0.1");
  assert.equal(formatTickLabel(0.5), "0.5");
  assert.equal(formatTickLabel(1), "1.0");
});

test("every boundary from frontBinBoundaries formats to a short, distinct label", () => {
  const labels = frontBinBoundaries().map(formatTickLabel);
  assert.deepEqual(labels, ["0.05", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]);
});

test("estimateClassAlpha recovers 0 for a pixel identical to white (no front)", () => {
  const classRgb = hexToRgb01("#3987e5");
  assert.equal(estimateClassAlpha([1, 1, 1], classRgb), 0);
});

test("estimateClassAlpha recovers 1 for a pixel identical to the pure class color", () => {
  const classRgb = hexToRgb01("#3987e5");
  assert.equal(estimateClassAlpha(classRgb, classRgb), 1);
});

test("estimateClassAlpha recovers ~0.5 for a pixel halfway between white and the class color", () => {
  const classRgb = hexToRgb01("#3987e5");
  const half = classRgb.map((c) => (1 + c) / 2);
  assert.ok(Math.abs(estimateClassAlpha(half, classRgb) - 0.5) < 1e-9);
});

test("estimateClassAlpha clamps to [0, 1] for pixels outside the white<->color line", () => {
  const classRgb = hexToRgb01("#3987e5");
  // A pixel "past" the pure class color along the same direction should
  // still clamp to 1, not overshoot.
  const overshoot = classRgb.map((c) => c - 0.2 * (1 - c));
  assert.equal(estimateClassAlpha(overshoot, classRgb), 1);
});

// --- front-lines overlay ---

const sampleLines = {
  type: "FeatureCollection",
  features: [
    { type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[0, 0], [1, 1]]] }, properties: { name: "cold_front", event_id: "front_0000" } },
    { type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[2, 2], [3, 3]]] }, properties: { name: "trough" } },
    { type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[4, 4], [5, 5]]] }, properties: { name: "overall_geometry" } },
    { type: "Feature", geometry: { type: "Point", coordinates: [0, 0] }, properties: { name: "warm_front" } },
  ],
};

test("every front-line feature name maps to a CLASS_STYLE class", () => {
  for (const cls of Object.values(FRONT_LINE_CLASS_BY_NAME)) assert.ok(cls in CLASS_STYLE, cls);
  assert.deepEqual(new Set(Object.values(FRONT_LINE_CLASS_BY_NAME)), new Set(Object.keys(CLASS_STYLE)));
});

test("frontLinesUrl builds a path under the model dir, and is null without a fronts file", () => {
  assert.equal(frontLinesUrl("/data", "m", { fronts: "a.fronts.geojson" }), "/data/m/a.fronts.geojson");
  assert.equal(frontLinesUrl("/data", "m", { store: "a.zarr" }), null);
  assert.equal(frontLinesUrl("/data", "m", { fronts: "" }), null);
  assert.equal(frontLinesUrl("/data", "m", undefined), null);
});

test("withLineClasses makes one tagged LineString per part, with its own side, and drops the rest", () => {
  const multi = {
    type: "FeatureCollection",
    features: [
      { type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[0, 0], [1, 1]], [[5, 5], [6, 6], [7, 7]]] },
        properties: { name: "cold_front", event_id: "front_0007", summary: { symbol_sides: ["right", "left"] } } },
      { type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[2, 2], [3, 3]]] }, properties: { name: "trough", summary: {} } },
      { type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[4, 4], [5, 5]]] }, properties: { name: "overall_geometry" } },
      { type: "Feature", geometry: { type: "Point", coordinates: [0, 0] }, properties: { name: "warm_front" } },
      { type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[9, 9]]] }, properties: { name: "warm_front" } },
    ],
  };
  const before = JSON.stringify(multi);
  const out = withLineClasses(multi);
  assert.equal(JSON.stringify(multi), before); // input untouched
  assert.deepEqual(out.features.map((f) => [f.properties.cls, f.properties.side, f.geometry.type]), [
    ["cold", "right", "LineString"],
    ["cold", "left", "LineString"],
    ["trough", "left", "LineString"], // no side given -> left
  ]);
  assert.deepEqual(out.features[1].geometry.coordinates, [[5, 5], [6, 6], [7, 7]]);
  assert.equal(out.features[0].properties.event_id, "front_0007");
  assert.equal("summary" in out.features[0].properties, false);
  assert.deepEqual(withLineClasses(undefined), { type: "FeatureCollection", oriented: false, features: [] });
});

test("withLineClasses carries `oriented` to the collection and every feature, and only true counts", () => {
  const one = (extra) => ({
    type: "FeatureCollection", ...extra,
    features: [{ type: "Feature", geometry: { type: "MultiLineString", coordinates: [[[0, 0], [1, 1]]] }, properties: { name: "cold_front", summary: { symbol_sides: ["left"] } } }],
  });
  const yes = withLineClasses(one({ oriented: true }));
  assert.equal(yes.oriented, true);
  assert.equal(yes.features[0].properties.oriented, true);
  for (const bad of [{}, { oriented: false }, { oriented: "true" }, { oriented: 1 }]) {
    const out = withLineClasses(one(bad));
    assert.equal(out.oriented, false);
    assert.equal(out.features[0].properties.oriented, false);
  }
});

test("lineColorExpression matches CLASS_STYLE colors and ends with a fallback", () => {
  const expr = lineColorExpression();
  assert.deepEqual(expr.slice(0, 2), ["match", ["get", "cls"]]);
  assert.equal(expr[expr.indexOf("cold") + 1], CLASS_STYLE.cold.color);
  assert.equal(expr[expr.indexOf("tropical_trough") + 1], CLASS_STYLE.tropical_trough.color);
  assert.equal(typeof expr.at(-1), "string");
  assert.equal(expr.length, 2 + 2 * Object.keys(CLASS_STYLE).length + 1);
});

test("stationary lines are never green, and oriented ones are drawn by the tile", () => {
  const expr = lineColorExpression();
  assert.notEqual(expr[expr.indexOf("stationary") + 1], CLASS_STYLE.stationary.color);
  assert.deepEqual(FRONT_SYMBOL_PATTERNS.stationary.line.map((l) => l.color), [CLASS_STYLE.cold.color, CLASS_STYLE.warm.color]);
  const f = solidLineFilter(["cold", "stationary"]);
  assert.equal(f[0], "all");
  assert.deepEqual(f[1][2][1], ["cold", "stationary"]);
  assert.deepEqual(f[2][1][1][2][1], ["stationary"]); // hidden only when oriented
});

test("lineFilter splits solid from dashed classes and honors the class checkboxes", () => {
  const all = Object.keys(CLASS_STYLE);
  assert.deepEqual(lineFilter(false, all)[2][1], ["cold", "warm", "stationary", "occluded", "tropical_trough"]);
  assert.deepEqual(lineFilter(true, all)[2][1], ["dryline", "trough"]);
  assert.deepEqual(lineFilter(false, ["warm", "trough"])[2][1], ["warm"]);
  assert.deepEqual(lineFilter(true, new Set(["warm", "trough"]))[2][1], ["trough"]);
  assert.deepEqual(lineFilter(false, [])[2][1], []);
});

// --- front symbols ---

test("only cold, warm, occluded and stationary fronts get symbols; troughs and drylines do not", () => {
  assert.deepEqual(Object.keys(FRONT_SYMBOL_PATTERNS).sort(), ["cold", "occluded", "stationary", "warm"]);
  for (const cls of ["dryline", "trough", "tropical_trough"]) assert.equal(symbolImageName(cls, "left"), null);
  assert.equal(symbolImageName("cold", "left"), "front-sym-cold-left");
  assert.equal(symbolImageName("cold", "right"), "front-sym-cold-right");
  assert.equal(symbolImageName("cold", "anything"), "front-sym-cold-left");
});

test("each pattern follows the WPC convention", () => {
  const shapes = (cls) => FRONT_SYMBOL_PATTERNS[cls].symbols.map((s) => s.shape);
  assert.deepEqual(shapes("cold"), ["triangle"]);
  assert.deepEqual(shapes("warm"), ["semicircle"]);
  assert.deepEqual(shapes("occluded"), ["triangle", "semicircle"]); // alternating
  assert.deepEqual(shapes("stationary"), ["triangle", "semicircle"]); // alternating cold/warm
  const sides = (cls) => FRONT_SYMBOL_PATTERNS[cls].symbols.map((s) => s.side);
  assert.deepEqual(sides("occluded"), ["same", "same"]); // both on the same side
  assert.deepEqual(sides("stationary"), ["same", "opposite"]); // on opposite sides
  // stationary: blue triangles, warm-colored semicircles; occluded: one color
  const [tri, semi] = FRONT_SYMBOL_PATTERNS.stationary.symbols;
  assert.equal(tri.color, CLASS_STYLE.cold.color);
  assert.equal(semi.color, CLASS_STYLE.warm.color);
  assert.equal(new Set(FRONT_SYMBOL_PATTERNS.occluded.symbols.map((s) => s.color)).size, 1);
});

test("symbols sit inside their tile, alternating ones do not overlap, and tiles are even widths", () => {
  for (const { period, symbols } of Object.values(FRONT_SYMBOL_PATTERNS)) {
    assert.ok(period % 2 === 0);
    for (const s of symbols) assert.ok(s.at > 0 && s.at < 1);
    if (symbols.length === 2) assert.ok(Math.abs(symbols[1].at - symbols[0].at) * period >= 12); // 12px = symbol width
  }
});

test("symbol layers: one per tile width, filtered to the checked classes", () => {
  assert.deepEqual(symbolPeriods(), [26, 52]);
  assert.deepEqual(symbolClassesWithPeriod(26), ["cold", "warm"]);
  assert.deepEqual(symbolClassesWithPeriod(52), ["occluded", "stationary"]);
  const classesIn = (filter) => filter[1][2][1];
  assert.deepEqual(classesIn(symbolFilter(26, Object.keys(CLASS_STYLE))), ["cold", "warm"]);
  assert.deepEqual(classesIn(symbolFilter(52, ["cold", "stationary", "trough"])), ["stationary"]);
  assert.deepEqual(classesIn(symbolFilter(26, [])), []);
});

test("symbols are only drawn for oriented lines", () => {
  const filter = symbolFilter(26, ["cold"]);
  assert.equal(filter[0], "all");
  assert.deepEqual(filter[2], ["==", ["get", "oriented"], true]);
});

// --- pressure overlays (isobars + highs/lows) ---

const samplePressure = {
  type: "FeatureCollection",
  valid_time: "2026-11-19T12:00:00",
  features: [
    { type: "Feature", geometry: { type: "LineString", coordinates: [[0, 0], [1, 1]] }, properties: { kind: "isobar", hpa: 1008, major: false } },
    { type: "Feature", geometry: { type: "LineString", coordinates: [[175, 60], [185, 61]] }, properties: { kind: "isobar", hpa: 1020, major: true } },
    { type: "Feature", geometry: { type: "Point", coordinates: [-100, 45] }, properties: { kind: "L", hpa: 1003 } },
    { type: "Feature", geometry: { type: "Point", coordinates: [-80, 30] }, properties: { kind: "H", hpa: 1027 } },
  ],
};

test("pressureUrl builds a path under the model dir, and is null without a pressure file", () => {
  assert.equal(pressureUrl("/data", "m", { pressure: "a.pressure.geojson" }), "/data/m/a.pressure.geojson");
  assert.equal(pressureUrl("/data", "m", { store: "a.zarr", fronts: "a.fronts.geojson" }), null);
  assert.equal(pressureUrl("/data", "m", { pressure: "" }), null);
  assert.equal(pressureUrl("/data", "m", { pressure: 7 }), null);
  assert.equal(pressureUrl("/data", "m", { pressure: ["a.geojson"] }), null);
  assert.equal(pressureUrl("/data", "m", undefined), null);
  assert.equal(pressureUrl("/data", "m", null), null);
});

test("PRESSURE_COLORS are distinct lowercase hex colors, H blue and L red", () => {
  const colors = Object.values(PRESSURE_COLORS);
  for (const color of colors) assert.match(color, /^#[0-9a-f]{6}$/, `${color} must be a lowercase 6-digit hex string`);
  assert.equal(new Set(colors).size, colors.length, "every pressure color must be distinct");
  // H is the blue end, L the red end -- and neither may collide with the
  // cold/warm front colors they share the map with.
  const [hr, hg, hb] = hexToRgb01(PRESSURE_COLORS.high);
  const [lr, lg, lb] = hexToRgb01(PRESSURE_COLORS.low);
  assert.ok(hb > hr && hb > hg, `${PRESSURE_COLORS.high} must read as blue`);
  assert.ok(lr > lg && lr > lb, `${PRESSURE_COLORS.low} must read as red`);
  assert.notEqual(PRESSURE_COLORS.high, CLASS_STYLE.cold.color);
  assert.notEqual(PRESSURE_COLORS.low, CLASS_STYLE.warm.color);
  // The isobar pair is a muted slate: no channel dominates, and the major
  // stroke is the darker of the two.
  for (const color of [PRESSURE_COLORS.isobar, PRESSURE_COLORS.isobarMajor]) {
    const [r, g, b] = hexToRgb01(color);
    assert.ok(Math.abs(r - b) < 0.25 && Math.abs(r - g) < 0.25, `${color} must read as neutral slate`);
  }
  const lum = (color) => hexToRgb01(color).reduce((sum, c) => sum + c, 0);
  assert.ok(lum(PRESSURE_COLORS.isobarMajor) < lum(PRESSURE_COLORS.isobar));
});

test("pressure label fonts are stacks the basemap's glyph server serves", () => {
  // The positron style's own label layers only ever ask for these three
  // (see lib.js); anything else 404s and no text is drawn at all.
  const served = ["Noto Sans Regular", "Noto Sans Italic", "Noto Sans Bold"];
  for (const stack of [PRESSURE_FONT, PRESSURE_FONT_BOLD]) {
    assert.ok(Array.isArray(stack) && stack.length > 0);
    for (const name of stack) assert.ok(served.includes(name), `${name} is not served by the basemap's glyph endpoint`);
  }
  assert.deepEqual(PRESSURE_FONT, ["Noto Sans Regular"]);
  assert.deepEqual(PRESSURE_FONT_BOLD, ["Noto Sans Bold"]);
});

test("the isobar and center filters split the file by kind", () => {
  assert.deepEqual(isobarFilter(), ["==", ["get", "kind"], "isobar"]);
  assert.deepEqual(centerFilter(), ["in", ["get", "kind"], ["literal", ["L", "H"]]]);
  assert.deepEqual(PRESSURE_KINDS, ["isobar", "H", "L"]);
});

test("isobarLabelFilter keeps only the major lines below the zoom cutoff", () => {
  const filter = isobarLabelFilter(4);
  assert.deepEqual(filter[0], "all");
  assert.deepEqual(filter[1], isobarFilter());
  assert.deepEqual(filter[2], ["any", ["==", ["get", "major"], true], [">=", ["zoom"], 4]]);
});

test("centerColorExpression colors H blue and L red with a fallback", () => {
  const expr = centerColorExpression();
  assert.deepEqual(expr.slice(0, 2), ["match", ["get", "kind"]]);
  assert.equal(expr[expr.indexOf("H") + 1], PRESSURE_COLORS.high);
  assert.equal(expr[expr.indexOf("L") + 1], PRESSURE_COLORS.low);
  assert.equal(typeof expr.at(-1), "string");
  assert.equal(expr.length, 2 + 2 * 2 + 1);
});

test("withPressureKinds keeps the recognizable features and rewrites their properties", () => {
  const before = JSON.stringify(samplePressure);
  const out = withPressureKinds(samplePressure);
  assert.equal(JSON.stringify(samplePressure), before); // input untouched
  assert.equal(out.type, "FeatureCollection");
  assert.deepEqual(out.features.map((f) => f.properties), [
    { kind: "isobar", hpa: 1008, major: false },
    { kind: "isobar", hpa: 1020, major: true },
    { kind: "L", hpa: 1003, major: false },
    { kind: "H", hpa: 1027, major: false },
  ]);
  // Coordinates survive untouched, including the un-cut antimeridian ones.
  assert.deepEqual(out.features[1].geometry.coordinates, [[175, 60], [185, 61]]);
  // The collection-level `valid_time` is not something the style reads.
  assert.equal("valid_time" in out, false);
});

test("withPressureKinds drops anything a layer expression could choke on", () => {
  const feature = (geometry, properties) => ({ type: "Feature", geometry, properties });
  const out = withPressureKinds({
    type: "FeatureCollection",
    features: [
      feature({ type: "LineString", coordinates: [[0, 0], [1, 1]] }, { kind: "isobar", hpa: "1008" }), // numeric string
      feature({ type: "LineString", coordinates: [[0, 0], [1, 1]] }, { kind: "isobar", hpa: 1000 }),
      feature({ type: "MultiLineString", coordinates: [[[0, 0], [1, 1]]] }, { kind: "isobar", hpa: 1004 }), // wrong geometry
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "L", hpa: 1002 }),
      feature({ type: "MultiPoint", coordinates: [[0, 0]] }, { kind: "H", hpa: 1030 }), // wrong geometry
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "l", hpa: 999 }), // wrong case
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "front", hpa: 1 }), // unknown kind
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "L" }), // no hpa
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "L", hpa: null }), // null hpa
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "L", hpa: "hPa" }), // not a number
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "L", hpa: "  " }), // blank string
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "L", hpa: true }), // not a number
      feature({ type: "Point", coordinates: [0, 0] }, { kind: "L", hpa: Number.NaN }), // not a number
      feature({ type: "Point", coordinates: [0, 0] }, undefined), // no properties at all
      undefined, // no feature at all
    ],
  });
  assert.deepEqual(out.features.map((f) => [f.properties.kind, f.properties.hpa]), [
    ["isobar", 1008],
    ["isobar", 1000],
    ["L", 1002],
  ]);
  // `major` is always a real boolean, never the string "true".
  assert.equal(out.features[0].properties.major, false);
  assert.deepEqual(withPressureKinds(undefined), { type: "FeatureCollection", features: [] });
  assert.deepEqual(withPressureKinds({ type: "FeatureCollection" }), { type: "FeatureCollection", features: [] });
});

test("pressureStatus names what the fetched file is missing, and is quiet when it is whole", () => {
  const isobars = { type: "FeatureCollection", features: [{ properties: { kind: "isobar" } }] };
  const centers = { type: "FeatureCollection", features: [{ properties: { kind: "L" } }] };
  const empty = { type: "FeatureCollection", features: [] };
  assert.equal(pressureStatus(samplePressure, true, true), "");
  assert.equal(pressureStatus(centers, true, false), ""); // isobars not asked for
  assert.equal(pressureStatus(isobars, false, true), ""); // centres not asked for
  assert.equal(pressureStatus(centers, false, true), "(isobars for this step)");
  assert.equal(pressureStatus(isobars, true, false), "(centres for this step)");
  assert.equal(pressureStatus(empty, true, true), "(isobars or centres for this step)");
  assert.equal(pressureStatus(undefined, true, true), "(isobars or centres for this step)");
});

test("pressureInitialState: param wins (off included), then stored, then default on", () => {
  // The URL param beats everything, "0" included, so a shared link
  // reproduces exactly what the sender saw.
  assert.equal(pressureInitialState("1", "0"), true);
  assert.equal(pressureInitialState("0", "1"), false);
  assert.equal(pressureInitialState("0", null), false);
  assert.equal(pressureInitialState("1", null), true);
  // No usable param: the remembered choice decides.
  assert.equal(pressureInitialState(null, "1"), true);
  assert.equal(pressureInitialState(null, "0"), false);
  assert.equal(pressureInitialState(undefined, "0"), false);
  // Neither param nor stored value (fresh page): on.
  assert.equal(pressureInitialState(null, null), true);
  assert.equal(pressureInitialState(undefined, undefined), true);
  // Garbage at one level falls through to the next, not to the default.
  assert.equal(pressureInitialState("yes", "0"), false);
  assert.equal(pressureInitialState("", "0"), false);
  assert.equal(pressureInitialState("2", "on"), true); // garbage param AND garbage stored -> default
  assert.equal(pressureInitialState("true", undefined), true);
  // An explicit default is honoured when nothing else applies.
  assert.equal(pressureInitialState(null, null, false), false);
  assert.equal(pressureInitialState(null, "1", false), true);
});
