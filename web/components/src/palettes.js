/**
 * Color palettes for bionym components. These ship separately so consumers
 * can:
 *   - use ours by default (components fall back to these automatically)
 *   - drop in their own:  BionymNetwork(el, g, { palette: MY_PALETTE })
 *   - extend ours:        { palette: { ...TYPE_PALETTE, Gene: "#0066cc" } }
 *
 * `palette` options are merged over TYPE_PALETTE, so partial palettes are
 * fine — unspecified types keep our defaults.
 *
 * Sources (all chosen for color-vision-deficiency safety):
 *   - Paul Tol's "muted" qualitative scheme (9 hues + gray) — the anchor for
 *     TYPE_PALETTE. https://personal.sron.nl/~pault/ (section "Muted")
 *   - Fabio Crameri's "batlow" continuous map — the 3 extra type colors are
 *     hues sampled from it, since batlow traverses many distinct hues and
 *     is perceptually uniform + CVD-tested. https://www.fabiocrameri.ch/colourmaps/
 *   - Okabe & Ito's palette for PREDICATE_PALETTE — the canonical 8-color
 *     CVD-safe qualitative set. http://jfly.uni-koeln.de/color/
 */

/**
 * Node fill per node type (keys match TYPE_ORDER in data.js).
 * First 9 = Tol muted; IdType = neutral gray (a Tol-sanctioned "bad-data"
 * color, apt for an input that was merely classified); Dataset/Condition/
 * Publication are batlow hues spaced to stay distinct under CVD simulation.
 */
export const TYPE_PALETTE = {
  IdType: "#BBBBBB",        // gray — Tol's muted 'bad data' slot
  Gene: "#332288",          // Tol muted: indigo   — anchor, saturated
  Organism: "#117733",      // Tol muted: green
  Assembly: "#999933",      // Tol muted: olive
  OrthologGroup: "#882255", // Tol muted: wine
  Transcript: "#44AA99",    // Tol muted: teal
  Protein: "#88CCEE",       // Tol muted: cyan
  Domain: "#CC6677",        // Tol muted: rose
  GOTerm: "#AA4499",        // Tol muted: purple
  Pathway: "#DDCC77",       // Tol muted: sand
  Dataset: "#011959",       // batlow[0]   — navy
  Condition: "#FCA993",     // batlow[208] — salmon
  Publication: "#4D734D",   // batlow[96]  — dark olive-green
};

export const FALLBACK_COLOR = "#999";

/**
 * Overview-chart predicate stacking: Okabe-Ito's 8-color CVD-safe set
 * (black dropped — too heavy for thin bar segments). Predicates past the
 * palette length collapse to PREDICATE_OTHER.
 */
export const PREDICATE_PALETTE = [
  "#E69F00", // orange
  "#56B4E9", // sky blue
  "#009E73", // bluish green
  "#F0E442", // yellow
  "#0072B2", // blue
  "#D55E00", // vermillion
  "#CC79A7", // reddish purple
  "#999999", // neutral gray
];
export const PREDICATE_OTHER = "#BBBBBB";

/** Overview flat-bar color — Okabe-Ito blue. */
export const BAR_COLOR = "#0072B2";

/** Merge a consumer palette over the defaults. */
export const mergePalette = (base, overrides) => ({ ...base, ...overrides });
