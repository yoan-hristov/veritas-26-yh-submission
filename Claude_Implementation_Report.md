# Implementation report: website upgrade

Code was written only; nothing was run or tested (per CLAUDE.md). Files changed: `public/index.html`, `server.js`, `website_parameters.json`. New: `models/rules.json`. Nothing else was touched (backend, images and the empty `public/assets/index.html` are unchanged).

## What was done

### Header
- Row of small buttons, left icon then text, placed in the header's right column so the first button starts above the upload box (same 420px width as the upload button, centered).
- **Rules** is the one that stands out (solid blue, opens the Rules sheet, not a link). Then GitHub, Colab, YouTube, Writeup as quiet, low-contrast blue buttons that open in a new tab.
- Below 1100px the row drops under the title; the connection indicator is no longer absolutely positioned there.

### Model names, descriptions, glow
- `NameM#` / `DescM#` in `website_parameters.json` override what the backend returns. An empty value falls back to the backend text. (`DescM#` is currently `"a"` as in your file, so the descriptions show "a" until you edit them.)
- Model 5 has a slight, slow RGB glow. `GlowM5` is the relative size (`1` = default, `2` = twice as large, `false` = off). Other models can glow too by adding `GlowM#`. Reduced-motion users get a static glow.
- The model list's inner padding was enlarged (and offset with a negative margin) so the glow is not clipped. Layout is otherwise the same.

### Benchmarks ("View benchmarks")
- Opens a table built from `models/benchmarks_simple.csv`. Model subtitles (e.g. "Consensus Cascade") are read from the header of `benchmarks_full.csv`.
- Per row: best value gets a green fill; middle values get an amber outline only; weakest get a red outline only; `-` and blanks stay neutral. Rows where all values are within 2% (e.g. Recall) are all green.
- Lower-is-better rows are listed in `LowerIsBetter` (default `ECE`, `follows_frame`).
- The best model (`BestModel`, default `M5`) is circled in blue across the whole column and has a filled blue title card; the other title cards are unfilled.
- Close with Esc, the x, OK, or a click on the backdrop. **View extended benchmarking** swaps to the plain `benchmarks_full.csv` table (sections, sticky first column and header, scrolls). There the button becomes **Simple benchmarking**.

### Rules
- Rules button opens a sheet with the English translation of your Bulgarian table: five terms and the Tier table (U, X, A, B, C, D, in the order they are evaluated, top to bottom). Tier badges are colored: U slate, X red, A green, B amber, C pink, D indigo.
- Same close options as the benchmarks sheet. All text lives in `models/rules.json`.

### Editing without touching code
| What | Where |
| --- | --- |
| Names, descriptions, glow, link URLs and order, best model, lower-is-better rows | `website_parameters.json` |
| Benchmark tables | `models/benchmarks_simple.csv`, `models/benchmarks_full.csv` |
| Rules text | `models/rules.json` |

Files are fetched on every open and served with no-store, so a refresh shows edits. `server.js` now exposes `/website_parameters.json` and `/models/*.csv|json` (other files in `models/` return 404).

## Decisions made on ambiguities
- `website_parameters.json` contained no links or order. I added a `Links` array (order = array order) with placeholder URLs: GitHub, Colab and YouTube point to their home pages, Writeup is `#`. **Replace these.** An entry with an empty `url` is hidden.
- The reference image was too small to read in detail, so the cell coloring is a rule (above), not a copy.
- `models/classification_rules.md` and `Claude_Implementation_Report.md` were empty, so rules were taken from `Instructions_Upgrade.md`.
- The tier colors in the existing Decision output (`TIER_COLOR` in `index.html`) were left as they were and differ from the new Rules badges (e.g. D is amber there, indigo in Rules). You did not ask for that change.

## Notes for testing
- The server must be restarted (`server.js` changed). Also commit/deploy `models/` and `website_parameters.json`; they were untracked in git.
- Header alignment at 1100-1300px width, glow clipping at large `GlowM5` values, and the 5-button row width are the things most worth a visual check.

## Follow-up Enhancements (Considerations Upgrade)

1. **Benchmark Table Outlines & Aesthetics:**
   - Unified all cell `.v` containers to a consistent height (`46px`), uniform padding, and consistent border radius (`8px`) with 1.5px borders.
   - Vertically and horizontally centered single-line values (e.g. `1.000`, `100.0%`) and two-line values (`0.980` with `± 0.010`), eliminating the previous jagged alternating row heights.
   - NA and unranked cells (e.g. M4 `follows_frame: –`) now have a matching neutral border instead of an empty missing gap.
   - Legend indicators updated to matching rounded chips (`width: 12px; height: 12px; border-radius: 4px;`), eliminating the previous inconsistency between circles and squares.

2. **Extended Benchmarking Window Enlargement:**
   - Added dynamic `.sheet-wide` class when switching to Extended Benchmarks (`View extended benchmarking`).
   - Expanded the dialog dimensions up to `96vw` (max 1500px) and `94dvh`, giving all 8 columns and 46 rows comfortable space to display clearly.
   - Smoothly restores to compact modal when toggled back to Simple Benchmarks or closed.

3. **Rules Decision Table Readability:**
   - Restructured the table in `models/rules.json` to feature bold, crystal-clear titles for each condition (`Not a chest X-ray or quality fails gate`, `Gemini unavailable`, `Both readers agree and both exceed thresholds`, `Both readers agree, but one or both are weak`, `COVID override`, `Readers disagree or etiologic suspicion is mixed`).
   - Displayed detailed technical rules underneath the titles in clean muted text.
   - Introduced styled semantic chips for `Agreement` (`AGREE`, `DISAGREE`, `N/A`) and `Risk` (`LOW`, `HIGH`, `REVIEW`, `NONE`).
   - Tiers ordered consistently: `X`, `U`, `A`, `B`, `C`, `D`.

4. **Clipboard Image Pasting (`Ctrl+V`):**
   - Added a global `paste` event listener allowing users to paste images directly from their clipboard (`Ctrl+V`) at any time.
   - Closes open sheets, wraps the pasted image, requests consent if not yet given, and immediately triggers analysis.

5. **Decision Box Tier Letter Color:**
   - Synchronized `TIER_COLOR` in the decision output with the exact semantic colors from the Rules badges:
     - `U`: `#94a3b8` (Slate)
     - `X`: `#f43f5e` (Crimson)
     - `A`: `#10b981` (Emerald)
     - `B`: `#f59e0b` (Amber)
     - `C`: `#ec4899` (Pink)
     - `D`: `#818cf8` (Indigo)

