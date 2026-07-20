# Experiment Change Log

This file records implementation changes that were kept because experiments
showed a positive effect. Add a new entry only after a change is validated by an
experiment report. Keep entries in chronological order.

## 2026-06-21: Thermal-Aware Runtime Logging And Plotting

Change kept:

- Added thermal-aware runtime adjustment fields to the experiment log.
- Added post-experiment plotting for temperature, FPS, latency, workload knobs,
  and device state.
- Changed plot x-axis to elapsed minutes for long Raspberry Pi runs.

Experiment outcome:

- The plots made it clear that the early thermal strategy reacted only after
  the CPU reached about 85 C and the Pi had already downclocked to about
  800 MHz.
- This result motivated the later fan control and interval-first thermal
  strategy work.

## 2026-06-30: IMX219 Camera Integrated As A Background Peripheral

Change kept:

- Added IMX219 raw capture support and a gray runtime mode.
- Kept `data/sample.mp4` as the controlled runtime input for experiments while
  the IMX219 camera continues running in the background.
- Added `--background-camera-trigger during-tracking`, which requests one
  asynchronous camera capture when LK tracking begins and skips the request if a
  previous capture is still active.

Experiment outcome:

- With `data/sample.mp4` as runtime input, frame-source cost returned to a few
  milliseconds instead of hundreds of milliseconds from serial camera capture.
- Background camera work became separately visible as:
  `Background camera frames captured`, `Background camera skipped triggers`,
  and last camera timing fields.
- Camera capture remained available for the dashboard/peripheral requirement
  without making every runtime frame wait for IMX219 capture and ISP.

## 2026-06-30: Motion-Triggered 320 ROI Refresh

Change kept:

- Enabled motion-triggered ROI refresh for unexplained motion outside tracked
  boxes.
- Kept ROI resolution at 320.
- Disabled LK-quality-degraded ROI by default because it produced unstable
  long-tail latency when failed tracks covered an unsafe region.

Representative reports:

- Good ROI reports showed motion-triggered ROI median around 0.77-0.96 s,
  while full-frame forced refresh was usually about 2.6-3.3 s.
- Later stable reports after power/device-monitor fixes showed ROI median
  around 0.83-0.87 s.

Experiment outcome:

- 320 ROI is useful when the ROI is truly local.
- The kept policy is: use ROI for unexplained outside motion; use full-frame
  RT-DETR for broad LK-quality degradation or safety refresh.

## 2026-07-01: ONNX Warmup And ROI Slow-Fuse Adjustment

Change kept:

- Warmed up the active 640 full-frame session and the 320 ROI session before
  timed logging.
- Avoided warming unused model sizes to reduce memory pressure.
- Made ROI slow-fuse permissive: a slow ROI row does not immediately disable
  ROI; repeated extreme slow rows are required.

Representative reports:

- The first logged inference stopped dominating the run with cold-start ONNX
  latency.
- ROI latency recovered to the expected sub-second class on good runs.

Experiment outcome:

- The remaining ONNX long tail was less likely to come from first-use session
  initialization.
- ROI stayed available during normal operation instead of being disabled by one
  bad row.

## 2026-07-01: Background Device Monitor For Firmware State

Change kept:

- Moved `vcgencmd measure_clock arm` and `vcgencmd get_throttled` polling to a
  background firmware monitor.
- The main loop reads cached firmware state and still reads fresh sysfs
  temperature each frame.
- Added log fields for stale firmware data, firmware poll cost, and historical
  throttling flags.

Representative reports after reboot:

- Run 01: ROI median 867.617 ms; device monitor mean 1.986 ms, median
  1.775 ms, max 23.824 ms.
- Run 02: ROI median 839.590 ms; device monitor mean 1.959 ms, median
  1.771 ms, max 10.170 ms.
- Run 03: ROI median 827.174 ms; device monitor mean 1.927 ms, median
  1.763 ms, max 17.055 ms.
- `uv`, `uvH`, `throt`, and `thrH` stayed clear after reboot; ARM clock stayed
  around 1800 MHz.

Experiment outcome:

- Device monitor stopped creating large per-frame blocking spikes.
- Temperature-based control remained per-frame, while slow firmware calls no
  longer blocked tracking or inference.

## 2026-07-17: Batched Sparse LK Tracking

Change kept:

- Batch all active LK tracks into one forward/backward optical-flow call.
- Reuse tracked feature points across frames and redetect points only every
  `lk_redetect_interval` frames or when the point count drops too low.
- Reduce default `lk_max_corners` from 40 to 24.
- Expose LK tuning parameters in strategy configs:
  `lk_win_size`, `lk_max_level`, `lk_max_iterations`,
  `lk_redetect_interval`, and `lk_redetect_min_points`.

Representative repeated-run report:

- Run 01: `LK tracking` mean 94.498 ms, median 29.010 ms, p95 467.126 ms;
  loop FPS median 5.535; ROI median 871.229 ms.
- Run 02: `LK tracking` mean 93.826 ms, median 29.255 ms, p95 459.937 ms;
  loop FPS median 5.655; ROI median 842.091 ms.
- Run 03: `LK tracking` mean 94.965 ms, median 29.367 ms, p95 459.082 ms;
  loop FPS median 5.322; ROI median 837.235 ms.

Experiment outcome:

- Tracking median improved from the previous hundreds-of-milliseconds range to
  about 29 ms.
- End-to-end loop FPS increased to about 5.1-5.3 mean / 5.3-5.7 median FPS.
- Motion-triggered 320 ROI remained stable at the expected 0.84-0.87 s median.
- Full-frame forced refresh still costs about 2.8-3.0 s, so later optimization
  should focus on reducing forced full-frame refresh frequency or cost.

## 2026-07-18: Repeat-Run Fan Reuse And Quiet Camera Progress

Change kept:

- `run_live_dashboard.py` now keeps one shared fan controller across all
  repeated runs and closes it only after the full repeated experiment finishes.
- Background IMX219 command output is suppressed, and compact camera progress is
  printed only every 20 completed captures.

Representative report:

- User confirmed the fan works normally after the change, including later runs
  in a repeated experiment.
- User confirmed the camera text output is acceptable.
- Three repeated runs kept stable tracking performance:
  `LK tracking` median 27.815 ms, 28.605 ms, and 28.445 ms.
- Loop FPS median stayed high at 5.526, 5.712, and 5.653 FPS.
- ROI median stayed near the expected sub-second level: 881.552 ms,
  870.331 ms, and 870.428 ms.

Experiment outcome:

- The repeated-run GPIO/PWM lifecycle problem is resolved.
- Terminal output remains readable during camera-enabled repeated experiments.
- The next bottleneck is no longer LK tracking; it is full-frame forced
  refresh, especially `forced_refresh_lk_tracking_quality_degraded` and
  long-interval safety refresh.

## Pending Validation

### LK-Quality Refresh Diagnostics Preservation

Change under consideration:

- Preserve the original LK-quality degradation diagnostics on rows that end in
  `forced_refresh_lk_tracking_quality_degraded`. Current reports show forced
  rows with `failure_ratio=0.000`, `quality=1.000`, and `failed_boxes=0`
  because the row is logged after detector reset. This is misleading for
  analysis even though the runtime decision is still correct.

Validation target:

- Summary output should show the real failure ratio, quality, failed box count,
  and confirmation count that caused a full-frame LK-quality refresh.

## 2026-07-18: LK-Quality Refresh Confirmation

Change kept:

- When LK tracking reports a soft `lk_tracking_quality_degraded` event, defer
  the first full-frame refresh for one frame if at least one track survived and
  the failure ratio is not catastrophic.
- If the next frame is still degraded, allow the normal full-frame RT-DETR
  refresh.
- Log confirmation diagnostics:
  `tracking_track_count_before`, `tracking_track_count_after`,
  `tracking_failed_box_count`, `lk_quality_confirm_count`,
  `lk_quality_confirm_deferred`, and
  `lk_quality_confirm_total_deferred`.
- `scripts/summarize_stage_latency.py` prints an
  `LK Quality Refresh Confirmation` section so the effect can be checked from
  pasted reports.

Representative repeated-run report:

- Run 01: inferences 52; `forced_refresh_lk_tracking_quality_degraded` 15;
  `LK tracking` median 28.152 ms; ROI median 860.457 ms.
- Run 02: inferences 52; `forced_refresh_lk_tracking_quality_degraded` 15;
  `LK tracking` median 27.900 ms; ROI median 860.591 ms.
- Run 03: inferences 52; `forced_refresh_lk_tracking_quality_degraded` 15;
  `LK tracking` median 28.101 ms; ROI median 856.517 ms.
- Each run deferred 24 soft LK-quality refresh rows.

Experiment outcome:

- Compared with the previous stable repeated runs, total inference count fell
  from 58 to 52 per 10-minute run.
- Full-frame LK-quality refresh fell from 18 to 15 per run.
- Tracking speed stayed stable at about 28 ms median.
- ROI stayed stable around 0.86 s median with no ONNX tail above about 1.3 s.
- The improvement is real but modest; the next bottleneck is still the
  remaining 15 LK-quality full-frame refreshes plus 3 safety full-frame
  refreshes per run.
