# Experiment notes

User supplied three YouTube videos and requested a trial. Scope: offline exploratory prototype, not production implementation of the draft plan.

## Setup
- Workspace initially contained only implementation-plan.html.
- Downloaded video-only formats 135/136 using available yt-dlp. Its existing user configuration automatically loaded Firefox cookies; no cookie contents are retained in this report.
- Initially installed opencv-python-headless 5.0.0.93; execution failed because that build exposed no CascadeClassifier. Pinned opencv-python-headless 4.13.0.92 instead; full experiment then passed. numpy 2.5.3, project-local .venv.

## Deviations
- Instead of implementing the full segmentation/panel pipeline, test 48 sparse frames per supplied video first, inspect candidates visually, and measure face-position recurrence. This conservatively tests assumptions before investing in panel extraction.
- YuNet weights are reachable in this environment; previous discussion's LFS restriction does not apply here. Compare Haar and YuNet on identical frames.
- Recurrence is not panel confidence. No face bbox will be represented as a confirmed webcam panel.
- Global sparse samples cannot evaluate short appearances, transition timing or live latency.

## Findings
- Largest spatial clusters, Haar/YuNet respectively: clip_1 40/48 vs 48/48; clip_2 25/48 vs 47/48; clip_3 24/48 vs 34/48.
- Visually reviewed 12-frame overviews of all videos, all 48 clip_3 samples, and last 12 clip_2 samples. This is not manual annotation of every detection across all videos.
- clip_3: 5 additional YuNet detections in a second positional cluster are the same host in the same lower-left panel (manual association); 39/48 target-region face observations total. Samples 6/7 have a fullscreen insert, 47 is the outro; 22–25/27/35 keep the panel but have face occlusion/pose/detection failures.
- clip_2's last sample keeps the panel with turned/partly occluded face. Game character forms another face cluster (4 samples).
- 3/12/24/48 deterministic subsampling completed from cache. Recurrence is not monotonic with sample count; greedy spatial clustering is itself sensitive to sampling and head motion.
- No automatic panel borders, segment boundaries or calibrated confidence produced. report.html is self-contained with embedded contact sheets.

## Validation
- Full probe.py run: 144 frames decoded and both detectors executed (~23.9 seconds total excluding download/setup).
- summarize.py completed and generated report.html, sampling-comparison.json, preview.jpg.
- py_compile passed for both scripts.
- Assertions passed: 3 videos × 48 frames; strictly increasing actual timestamps; seek error <0.1 seconds; no duplicate frame assignment per cluster. HTML parser accepted report.
- This is smoke validation, not an independently labelled accuracy benchmark. No merge performed.

# Session 2 — panel boundaries and faceless samples

User approved the proposed next experiment. Scope: reuse three local videos, recover full panel boundaries, validate panel presence independently of face visibility, and produce a visual development report. Existing face experiment is kept intact.

## Decisions before implementation
- Start with corner-attached, axis-aligned rectangles, matching the observed examples. This is explicitly narrower than general overlays or arbitrary masks.
- Face recurrence seeds candidate search; panel borders and appearance must be revalidated on each sampled frame. No unlimited blind carry-forward.
- Independent annotation delegated before prediction implementation; reference data is evaluation-only and must not be read by inference code.
- Preserve absence, uncertainty, and changed geometry rather than forcing a bbox. Results are sampled-frame decisions, not exact continuous segments.

## Deviations / discoveries
- Independent reference inspection found that clip_3 sample 6 still contains a dark smaller panel; the prior report's assertion that both 6 and 7 lack a panel was incorrect. Recheck and explicitly correct this in the new report.
- The three supplied videos remain development examples. Independent annotation prevents circular box measurement but does not turn them into a held-out benchmark.
- A closer look at clip_2 #7 found imagery from a displayed clip/thumbnail in the same corner. The former 47/48 face-position count included this wrong-context observation. Independent reference inspection of surrounding frames supports excluding it from the current-session target; this annotation has medium confidence.
- Instead of Canny/contours, used directional colour differences, long-side support and temporal median for corner rectangles. The two outer sides are an explicit scope assumption, not detected boundaries. No per-video coordinates are hardcoded into inference.
- Added a small RGB appearance bank because border-only geometry also accepted unrelated rectangles. Bank uses face-supported samples across the entire video; base samples exclude themselves from appearance matching, but geometry estimation is not leave-one-out. Appearance threshold and border thresholds are development heuristics, not probabilities.
- Strong appearance change conservatively abstains: the smaller dark clip_3 panel remains a known miss. Do not relax the appearance filter just to hide this error.
- Validation is at sampled timestamps, not continuous segments. Added at most 24 local samples per video (±1/±2 seconds around selected suspects), 36 in this run. They do not update the offline model.
- The test-writing delegation produced experiment/test_panel_probe.py but its final response failed due to an oversized tool payload. Inspected the resulting tests and ran them locally. One meaningful failure found a rounded timestamp could reach/exceed duration; fixed validation to check rounded seek targets. All 37 tests then passed.

## Results
- All base samples: clip_1 48 detected / 48; clip_2 46 detected + 2 ambiguous; clip_3 45 detected + 2 ambiguous + 1 not_detected.
- Seven base samples have detected panels without a YuNet face in the candidate: clip_2 #47 and clip_3 #22–25/#27/#35. Current border and appearance evidence validates each; no blind carry-forward.
- Independent manual reference: 50 samples, 47 target panels present and 3 absent. 45 localized at IoU>=0.90, 2 present-panel abstentions (clip_2 #32 weak edge, clip_3 #6 changed dark appearance), 0 returned boxes on the 3 absent samples. Absence semantics include uncertain content identification, and only three negatives cannot establish a general false-positive rate.
- Mean IoU among returned reference boxes: clip_1 0.9904, clip_2 0.9903, clip_3 0.9938. Maximum side error versus annotations: 2/1/2 original pixels respectively, within the manual reference's approximate 3–6 px tolerance. IoU including misses as zero is separately reported.
- Visually inspected all 36 dense-sample outputs and the six-case comparison preview. No independent quantitative labels for dense samples.
- Historical first-report text corrected in summarize.py; regenerate report.html to avoid repeating old false interpretations.

## New artifacts and validation
- panel_probe.py: inference independent of panel-reference.json; panel-results.json and contact sheets.
- panel_report.py: evaluator/report generator, panel-evaluation.json, panel-preview.jpg, self-contained panel-report.html (~5 MB).
- panel-reference.json: separate annotations; README.md documents scope, JSON and reproduction.
- Synthetic unit tests cover all four corners, border discovery, appearance exclusion, disappearance/reappearance, ambiguity, normalization, decode failures and time budgets. 37 tests passed; py_compile passed.
- Final full run reproduced the same metrics; both reports regenerated. All 37 unit tests and py_compile passed. Additional assertions passed for all 180 samples: timestamp order/range/seek tolerance, normalized coordinates, in-frame pixel boxes, null-on-abstention contract, model SHA256, and HTML parsing. The first ad-hoc assertion command had a bracket typo; corrected command passed (no source-code defect).
- No commit or merge. Remaining W8 questions are in panel-report.html for a future merge review.
