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
} from "./lib.js";

test("hexToRgb01 converts pure primaries and grays", () => {
  assert.deepEqual(hexToRgb01("#ff0000"), [1, 0, 0]);
  assert.deepEqual(hexToRgb01("#00ff00"), [0, 1, 0]);
  assert.deepEqual(hexToRgb01("#0000ff"), [0, 0, 1]);
  assert.deepEqual(hexToRgb01("#000000"), [0, 0, 0]);
  assert.deepEqual(hexToRgb01("#ffffff"), [1, 1, 1]);
});

test("CLASS_STYLE has one distinct color per class, colorblind-safe order", () => {
  const classes = Object.keys(CLASS_STYLE);
  assert.deepEqual(classes, ["cold", "warm", "stationary", "occluded"]);
  const colors = classes.map((cls) => CLASS_STYLE[cls].color);
  assert.equal(new Set(colors).size, colors.length, "every class needs a distinct color");
  for (const color of colors) {
    assert.match(color, /^#[0-9a-f]{6}$/, `${color} must be a lowercase 6-digit hex string`);
  }
  // Locks in the specific colorblind-safe palette (dark-mode slots 1-4 of
  // the dataviz skill's reference categorical order) so a future edit that
  // silently reverts to the old blue/red/green/purple set fails loudly.
  assert.deepEqual(colors, ["#3987e5", "#d95926", "#199e70", "#c98500"]);
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
  assert.equal(binLabel(0), "0.01–0.1");
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
    [0.01, 0.2, 0.4, 0.6, 0.8, 1]
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
  assert.equal(formatTickLabel(0.01), "0.01");
  assert.equal(formatTickLabel(0.1), "0.1");
  assert.equal(formatTickLabel(0.5), "0.5");
  assert.equal(formatTickLabel(1), "1.0");
});

test("every boundary from frontBinBoundaries formats to a short, distinct label", () => {
  const labels = frontBinBoundaries().map(formatTickLabel);
  assert.deepEqual(labels, ["0.01", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]);
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
