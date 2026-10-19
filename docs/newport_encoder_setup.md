# Newport ESP301 encoder setup

This guide records the successful 2026-10-16 commissioning of the tomography
rotation axis and explains how to repeat the setup. It applies to the **ESP301
with the installed legacy ESP300 EPICS driver**. Check the driver and controller
manual before applying it to other controller models or firmware versions.

## Working configuration

| Item | Commissioned axis |
| --- | --- |
| Controller | Newport ESP301, firmware 1.1.1 |
| Controller channel | 3 |
| EPICS motor | `4dh4:m3` |
| Bluesky name | `stage1.theta` |
| Description | `sample tomo rotation` |
| Matching reference stage confirmed by the operator | `4dh4:m13`, `stage2.theta` on ACS |
| Encoder label, likely corrected spelling | EPC `15T-01SF-1000NV1RHV-F00` |

The encoder model is inferred from the supplied label and EPC ordering guide;
the complete marking was not independently inspected. `15T` is the likely
correction to `1st`. The `V1` voltage option is valid.

| Setting | Before commissioning | After successful tests |
| --- | --- | --- |
| Newport `3ZB?` | `0H` | `100H`: encoder positioning on, automatic correction off |
| Newport `3SU?` | `0.00125` | Same value, **explicitly reapplied after enabling feedback** |
| Newport `3FP?` | `4` | `4` |
| Newport `3SN?` | `7` (degrees) | Unchanged |
| Newport `3FR?` / `3QS?` | `0.025` / `200` | Unchanged |
| Newport `ZZ?` | `113H` | `113H`; protection restored after setup |
| Newport `3VA?` / `3VB?` | `5` / `0.5` degrees/s | Restored after testing |
| Newport `3AC?` / `3AG?` | `4.5` degrees/s² | Restored after testing |
| EPICS `MRES` / `ERES` | `0.00125` degrees/count | Unchanged |
| EPICS `UEIP` / `URIP` | `0` / `0` | Unchanged |
| EPICS `RDBL` | Empty | Unchanged |
| EPICS `RTRY` | `10` | `0`: retries disabled for the trial |
| EPICS `SPMG` | `Go` | `Go`; temporarily `Stop` during setup |
| EPICS `VELO` / `ACCL` | `5` degrees/s / `1` second | Restored after testing |

Motor power stayed enabled. Controller automatic correction and EPICS retries
were left disabled. No motor calibration, homing, or controller zero-definition
command was needed. Other Newport motor records were unchanged.

**These are recorded commissioning settings, not a live status report.** After
the motion tests, the operator requested persistence. On 2026-10-16, controller
`SM` saved the verified configuration to **nonvolatile flash**. It completed
without reported errors; all queried controller settings and EPICS motor fields
were unchanged by the save. No reset or power-cycle test was performed. EPICS
autosave is separate and may retain the temporary `RTRY=0` value.

## What fixed the initial 10× error

With encoder positioning first enabled, a native 10° demand produced visible
approximately 10° rotation in camera 2, but Newport `TP` reported `0.9996°`.
Although `SU?` already returned `0.00125`, the active conversion did not agree.

During a stationary diagnostic, temporarily writing `SU=0.0125` produced
`TP=99.96°`. Writing the original **`SU=0.00125`** back then produced
`TP=9.996°`, without moving the stage. The successful recipe therefore explicitly
applies `SU` **after** enabling encoder feedback. A permanent 10× scaling change
was not the solution. The internal firmware cause was not established.

Writing `SU` automatically changed display precision to `FP=3`. Restoring `FP=4`
then correctly reported `9.9963°`; display precision itself did not reproduce
the original error once `SU` had been reapplied.

## Encoder resolution and wiring

For this likely encoder model, `1000` means 1,000 cycles per encoder revolution.
With the ESP301's ×4 quadrature decoding, that gives 4,000 counts/revolution.
For an encoder turning once per motor revolution, with 5° stage travel per motor
revolution, the expected resolution is:

```text
SU = stage travel per encoder revolution / decoded counts per revolution
   = 5 degrees / (1000 cycles × 4)
   = 0.00125 degrees/count
```

For a different rotary setup, let `G` be encoder revolutions per stage revolution
and `C` be encoder cycles per revolution: `SU = 360 / (G × 4 × C)`. An encoder
directly on the output platform has `G=1`; the same 1,000-cycle encoder would then
give `0.09°/count`. For translation, use travel per encoder revolution instead of
degrees. Do not multiply by four again if a specification already gives decoded
counts. Encoder mounting and gearing must be established before selecting `SU`.

The EPC code `R` specifies A/B quadrature plus index; `HV` specifies differential
line-driver outputs. Its line-driver specification meets RS-422 with a **5 V
supply**. The wider `V1` supply rating does not establish compatibility of
higher-voltage signals with Newport's inputs.

The following mapping is for the EPC original cable and the ESP301 **axis motor
connector, 25-pin D-sub**, not its GPIO connector. Check any adapter's continuity.

| Signal | EPC wire | Newport pin |
| --- | --- | --- |
| A+ | Brown | 19 |
| A− | Yellow | 23 |
| B+ | Red | 20 |
| B− | Green | 24 |
| Index+ | Orange | 15 |
| Index− | Blue | 25 |
| +5 V | White | 21 |
| Encoder ground | Black | 22 |
| Shield ground | Bare | 14 |

Index is a homing reference; A/B provide continuous position feedback. A
motor-shaft index repeats at every motor revolution, so it does not uniquely
identify a stage revolution through reduction gearing. A motor-shaft encoder
also does not directly measure backlash between the motor and the output stage.

## Controller setup procedure

1. **Capture the baseline.** Save controller identity, `ZZ`, and the axis's `ZB`,
   `SU`, `FP`, `SN`, `FR`, `QS`, `TP`, `DP`, motion profile, correction settings,
   limits and motor-power state. Save the motor record's calibration, demand,
   readback, retry, Stop/Go and readback-selection fields. Record the physical
   position as well as its software coordinates.
2. **Pause automatic action.** With the axes stationary, put the selected EPICS
   motor in `SPMG=Stop` and set `RTRY=0`. Disable controller stepper correction
   while checking feedback. Use the existing IOC's serial connection for
   diagnostics, so a second serial reader cannot consume the driver's replies.
3. **Preserve configuration bits.** Read `ZZ?` and the axis's `ZB?`. If required,
   temporarily clear only `ZZ` bit 8, the critical-settings protection bit.
   Set `ZB` bit 8 to enable encoder positioning and clear bit 9 to keep automatic
   stepper correction off. Preserve unrelated settings and restore the original
   `ZZ` afterward, including if a later step fails.
4. **Apply encoder resolution after feedback is enabled.** Write the calculated
   `SU` even if its queried value already looks correct. Restore the desired
   `FP` afterward, since writing `SU` changed it on this controller.
5. **Verify stationary state.** Read back `ZB?`, `SU?`, `FP?`, `TP?`, `DP?`, motor
   power and errors. Refresh EPICS status. Do not redefine zero to hide a scale
   or polarity error.
6. **Clear pending demands before Go.** A target entered while stopped can remain
   queued. Confirm both `VAL`/`RBV` and `DVAL`/`DRBV` agree, with no queued jog or
   home. This motor-record version defers `SYNC` while in Stop; do not assume
   writing `SYNC` in Stop has cleared a pending target. During commissioning,
   both stopped demand fields were aligned to their readbacks before Go.
7. **Test signed displacement.** Choose travel with suitable clearance. Compare
   demand, actual feedback and independent observation in both directions,
   including return to the original position. Encoder polarity is `ZB` bit 5;
   if direction is wrong, stop and verify the wiring/sign convention before
   changing it. Do not change motor calibration to compensate for a sign error.
8. **Restore the motion profile.** Return speed and acceleration to their saved
   values. Decide separately whether to enable correction, restore retries, or
   save controller settings permanently.

The exact ASCII write sequence used for **our channel 3 baseline** is:

```text
ZZ013H
3ZB0100H
3SU0.00125
3FP4
ZZ0113H
```

These are individual controller configuration writes, with queries and error
checks between steps; they are not IOC shell commands. Configuration writes
do not return position replies. Through an asyn diagnostic record, use Write
transactions for setters and Write/Read for queries, and restore its previous
settings afterward. No `QD`, reset, homing, or `DH` command was used.

The literal masks above apply because our saved `ZZ=113H` and axis `ZB=0H`.
For another axis/controller, derive the masks from its own baseline. Clearing
all other feedback/error bits by copying `100H` could discard existing settings.

The installed driver caches its conversion scale during IOC initialization.
For this feedback mode it uses `SU`; `MRES=ERES=SU` preserves the degree mapping.
Reapplying the same value needed no IOC restart. A numerical scale change needs
a coordinated review of driver conversion, motor-record calibration and saved
settings before further motion or an IOC restart. Recheck this behavior after
a driver upgrade; newer upstream code differs from the installed version.

## Why EPICS readback switches remain off

**Position readback is operating.** The switches select where the motor record
gets position; neither switch directly enables Newport's encoder electronics.

| Control | Selects | Appropriate use |
| --- | --- | --- |
| Newport `ZB` bit 8 | Controller uses its encoder for stepper positioning | Enabled for this axis |
| EPICS `UEIP` — Use Encoder If Present | Driver's raw encoder position `REP`, scaled by `ERES` | Driver exposes a useful encoder reading distinct from its normal position |
| EPICS `URIP` — Use Readback If Present | External input link `RDBL`, scaled by `RRES` | An independent position sensor publishes a suitable EPICS PV |

For the installed ESP300/301 driver, the normal data path is:

```text
Newport TP -> cached driver conversion -> RMP -> MRES -> DRBV -> DIR/OFF -> RBV
                                        REP -> ERES -> DRBV when UEIP=1
```

The driver copies the **same `TP`-derived raw position** into both `RMP` and
`REP`. Once Newport feedback is enabled and its conversion is correct, `TP`
already follows the physical encoder. With `MRES=ERES`, switching `UEIP` on
would select the same position data. Leaving it off avoids an unnecessary
selection change; it does not substitute the demand for measured position.

This conclusion is specific to the installed driver. On a controller/driver
that supplies commanded step position in `RMP` and a separate measured encoder
position in `REP`, `UEIP=1` is useful: EPICS then reports measured movement,
including missed steps, using the calibrated `ERES`. Verify that the driver
reports the intended counts and that the encoder-present indication is reliable.
Our legacy driver sets encoder-present unconditionally, so that status bit alone
does not verify wiring or a working encoder.

Use `URIP=1` when an external sensor should define position: for example, a
linear scale on the carriage, an output-shaft encoder exposed by another IOC,
or an interferometer. Configure a valid, updating `RDBL` and appropriate `RRES`,
units, sign and datum first. Our `RDBL` is empty, so `URIP` provides no useful
alternative. Select one source; this motor record makes `UEIP` and `URIP`
mutually exclusive.

Readback selection can also affect motion behavior. The installed motor record
can choose relative moves when `UEIP` or `URIP` is enabled with retries, and
changing `UEIP` can call its position-loading path (`LOAD_POS`, implemented as
Newport `DH`). Treat a selection change as configuration to commission while
stationary, rather than a cosmetic way to make the screen say On.

## Reading position versus correcting position

The successful tests enabled measured position **without automatic correction**:

- Newport `ZB=100H` enables encoder positioning; bit 9 remains clear.
- Newport bit 9 enables stepper correction after the trajectory finishes.
  `ZB=300H` would enable both bits for this baseline, but was not tested.
- EPICS `RTRY` allows additional motor-record attempts to reach the target;
  it is separate from Newport's correction mechanism.

Before commissioning correction, establish the sign, scale, repeatable settling
behavior and useful tolerance. Choose Newport `DB` (encoder counts) and `CL`
(milliseconds), and EPICS retry/deadband settings, accordingly. Existing `DB=0`
and `CL=1` ms were left unchanged with controller correction off; they are not
validated correction settings. Commission one correction mechanism at a time.

### Correction trial, 2026-10-16

At the operator's request, a bounded trial used `ZB=300H`, `DB=2` encoder
counts (0.0025°), and `CL=100` ms, with EPICS retries still disabled. At the
initial 0° demand, feedback held steadily at −0.0025°. A subsequent 10° move
at 1°/s settled at 9.9963° (about three counts short). It was stable over the
final ten holding samples but did not meet the two-count acceptance threshold.
This trial did not establish an improvement over feedback-only operation.

The test restored the working `ZB=100H`, `DB=0`, `CL=1` ms configuration and
returned the stage to its original 0° demand. Native feedback on return was
−0.0037°; EPICS readback was −0.0025° at its status refresh. Normal 5°/s speed,
acceleration, `SPMG=Go`, and alarm-free stationary status were verified. Encoder
feedback remains enabled; automatic correction and EPICS retries remain off.
No `SM` was sent for this trial, so the previously saved working configuration
was unchanged. Further correction tuning was deferred to keep the test bounded.
The host-local trace and restoration verification are in
`artifacts/newport/encoder-commissioning-20261016/correction-trial.json`.

An operator-authorized second trial used `DB=1` count and `CL=100` ms. At
a 120° demand, native feedback stayed at 119.9962° throughout 40 holding
samples (approximately 7 seconds), with no reported controller errors. The
0.0038° residual exceeded the one-count acceptance threshold, so no 10° move
was issued. The trial again restored `ZB=100H`, `DB=0`, `CL=1`, protection,
normal speed/acceleration, and alarm-free `SPMG=Go`. No `SM` was sent.

During cleanup, EPICS demand was synchronized to its 119.99625° readback to
clear pending actions before Go; native `DP` remained 120°. A zero EPICS
demand/readback difference after that synchronization is **not evidence that
correction removed the original error**. No controller zero was redefined.
The second trace is `correction-trial-one-count.json` in the same host-local
directory. Neither trial established a reliable improvement from controller
correction; the saved feedback-only configuration remains the working baseline.

## Verification, persistence and rollback

At 1°/s, the corrected tests produced:

| Test | Native demand | Native encoder position |
| --- | --- | --- |
| First corrected return | 0° | −0.0025° |
| Repeated outward move | 10° | 9.9963° |
| Repeated return | 0° | −0.0025° |

Camera 2 showed real rotation and return of the breadboard/post assembly and
approximately 10° travel on the engraved dial. These results compare encoder
position with demand; they are not independent metrology of millidegree accuracy.
All corrected tests ended without controller errors or EPICS alarms.

Saving controller settings is a separate step: `SM` writes the current system
and axis configuration to nonvolatile memory. It saves configuration for all
axes, so review their state first. It does not replace EPICS autosave or make an
incremental encoder an absolute position reference. Initial motion testing ran
without `SM`; the operator subsequently requested a save, which completed on
2026-10-16 with no reported errors and unchanged readbacks. Power-cycle recovery
was not tested. After a future restart, verify readback and position restoration
before normal operation; saving configuration does not establish the absolute
position of an incremental encoder after power loss.

To undo the trial, stop the motor record, disable retries, and restore the saved
feedback configuration with protection restored afterward. For our baseline:

```text
ZZ013H
3ZB00H
ZZ0113H
```

`SU`, `FP`, mechanical calibration and the normal motion profile already match
their original values. Refresh status and check demand/readback and queued
actions before restoring `RTRY=10` and `SPMG=Go`. This rollback does not command
a return move or redefine zero.

**The successful configuration is now saved in flash.** The commands above
restore the baseline in RAM only. To make a rollback survive a controller
restart, verify the restored configuration and the other axes, then issue `SM`
again to save the restored settings. The host-local rollback helper does not
issue `SM` automatically.

Host-local evidence is under the Git-ignored directory
`artifacts/newport/encoder-commissioning-20261016/`: `baseline.json`,
`settings-and-rollback.txt`, `journal.jsonl`, and `motion-test-10deg/` containing
the final snapshot, move traces, stationary experiments and camera images.
The `persistent-save/` subdirectory contains the all-axis snapshots before and
after `SM`, the verified preflight, and the save-intent record.
Those files are not required to understand or repeat the procedure in this guide.
The host-local `artifacts/newport/encoder_session.py rollback` helper performs
the guarded rollback and restores EPICS retry/Go state only if its checks pass.
It was prepared but not executed after the successful trial.

## References and implementation details

- [Newport ESP301 manual](https://www.newport.com/medias/sys_master/images/images/hda/h3e/9117547069470/ESP301-User-s-Manual.pdf):
  `SU`, `FP`, `ZB`, `ZZ`, `SM`; §4.3 stepper correction; §5.5 quadrature decoding;
  appendix C axis-connector assignments.
- [EPC Model 15T/H datasheet](https://www.encoder.com/hubfs/products/Model_15/15T-H/datasheet_model-15th.pdf):
  model decoding and electrical specifications.
- [EPC installation and wiring guide](https://www.encoder.com/hubfs/Installation_Wiring_Guide_Rev2_20241017.pdf?hsLang=en):
  printed page 10, Model 15T/H HV cable colors.
- [EPICS motor-record documentation](https://epics-modules.github.io/motor/motorRecord.html):
  `UEIP`, `URIP`, `ERES`, `RRES`, `RDBL`, readback and retry behavior.
- Installed sources inspected for this guide: motor R7-3-1
  `motorApp/MotorSrc/motorRecord.cc` (`do_work`, `load_pos`, `special`) and
  motorNewport `newportApp/src/drvESP300.cc` (`set_status`, initialization) /
  `devESP300.cc` (`SET_ENC_RATIO`, `LOAD_POS`). These installed implementations
  establish the shared `TP` data path and selection side effects described above.
