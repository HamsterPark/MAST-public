# MAST Skill Catalog

All 427 skills registered by `SkillRegistry.discover()` in the catalog-generation run are listed below. (Public snapshot: skills whose modules are not shipped here are left out — 34 entries.)

Grouped by declared `composition_level` (L0..L5). The table captures the catalog-generation run; it is not a live count of enabled skills or the tools bound to a particular agent.

The skill layer connects named operations to parameter metadata, execution policy
and results. Start with [the registry](../../MASTv2/mast/core/registry.py),
[the agent adapter](../../MASTv2/mast/agents/_shared/skill_adapter.py) and
[ExecutionContext](../../MASTv2/mast/core/execution_context.py) to see how those
declarations enter executable checks.

**Safety** records the skill's declared `auto`, `confirm` or `dangerous` level.
It does not by itself grant permission or guarantee a human-approval dialog.
Effective handling also depends on operating mode, parameters, approval provenance
and interlocks; inspect [the safety rules](../../MASTv2/mast/core/safety.py).
**Category** and level labels are declared classifications, not guarantees about
the exact number of controller calls, absence of branching or filesystem effects.

Descriptions below are abbreviated metadata. Consult each implementation for
complete parameters, preconditions, units, cancellation and result semantics.
Instrument-specific configuration and calibration are deployment inputs;
registration is not a claim that a skill has been validated on every instrument.

## Keeping the generated inventory current

The level counts, detailed rows and alphabetical index are generated together.
When skill registrations change, regenerate and reconcile all three against the
same source revision and public exclusions. Hand-editing a count alone does not
update the inventory. The generator owns this preface as well as the inventory;
keep its text in sync when updating catalog guidance.

## Counts

| Level | Count | Meaning |
|---|---|---|
| L0 | 347 | Atomic Nanonis TCP wrapper — 1:1 call, no branching. |
| L1 | 45 | Short bounded sequence — a few TCP calls or a poll loop. |
| L2 | 16 | Pure data analysis — no Nanonis writes, may have zero TCP calls. |
| L3 | 17 | Multi-step hardware workflow — orchestrates other skills via step()/step_or_fail(). |
| L4 | 2 | Long autonomous / RL / paper-replication procedure. |

## L0 — Atomic Nanonis TCP wrapper — 1:1 call, no branching.

| Name | Folder | Safety | Category | Params | Tags | Description |
|---|---|---|---|---|---|---|
| `GetOsciTimebases` | builtins/acquire_osci_trace | auto | read | - | oscilloscope, timebase, samplerate, read | List the available Oscilloscope-1-Channel (Osci1T) timebases. Each timebase is the per-sample interval dt (s); the sample rate is fs = 1/dt. The set of timebases depends on the RT frequency and RT oversampling. Requir… |
| `SetOsciTimebase` | builtins/acquire_osci_trace | auto | write | timebase_index | oscilloscope, timebase, samplerate, configure | Set the Oscilloscope-1-Channel (Osci1T) timebase by index. Use GetOsciTimebases first to obtain the index→sample-rate mapping. Configuration-only — does not move the tip or change any setpoint. |
| `LoadMultiPassConfig` | builtins/advanced_ops | dangerous | write | file_path | scan, multipass, file, advanced, dangerous | Load a multi-pass scan configuration from a file on the Nanonis machine.  The file decides what EACH PASS does — its bias, its Z offset, whether the feedback is on. MAST cannot read the file and cannot tell you what i… |
| `QuitNanonis` | builtins/advanced_ops | dangerous | write | save_settings, settings_name, layout_name | system, quit, advanced, dangerous | Quit the Nanonis software.  **Stops the scan and RETRACTS THE TIP first, always.** Quitting with the tip engaged leaves it in the surface with no software watching it — the Z feedback dies with the process. If the ret… |
| `SaveMultiPassConfig` | builtins/advanced_ops | confirm | write | file_path | scan, multipass, file, advanced | Save the current multi-pass scan configuration to a file on the Nanonis machine. Changes nothing on the instrument; it will overwrite an existing file at the given path. |
| `SetMultiPass` | builtins/advanced_ops | confirm | write | on | scan, multipass, write | Switch multi-pass scanning on or off.  Multi-pass scans the same line several times with different settings on each pass — the standard way to separate topography from an electrostatic or magnetic signal (pass 1 recor… |
| `WaitForScanEndBlocking` | builtins/advanced_ops | confirm | read | timeout_s | scan, wait, blocking, advanced | Block until the current scan finishes, using Nanonis' own wait.  **You almost certainly want WaitScanComplete instead** — the polling version. It does not hold the connection and it reports progress. This one is for w… |
| `GetAutoApproachStatus` | builtins/approach | auto | read | - | approach, status, read | Get whether the auto-approach procedure is running. |
| `StopAutoApproach` | builtins/approach | auto | write | - | approach, stop, safety, emergency | Immediately stop the auto-approach procedure (AutoApproach_OnOffSet 0). Emergency halt for a running coarse approach — always safe, never gated. |
| `WithdrawTip` | builtins/approach | confirm | write | - | tip, withdraw, safety | Withdraw the tip fully from the surface. |
| `AtomTrackDriftComp` | builtins/atom_track | confirm | write | - | atomtrack, drift, compensation, write | Apply the Atom Tracking drift measurement to the drift compensation. |
| `AtomTrackQuickCompStart` | builtins/atom_track | confirm | write | compensation_type | atomtrack, compensation, write | Start tilt or drift compensation via Atom Tracking. |
| `AtomTrackStatusGet` | builtins/atom_track | auto | read | control | atomtrack, status, read | Get the on/off status of an Atom Tracking control (modulation, controller, or drift). |
| `GetBias` | builtins/bias | auto | read | - | bias, read | Read the current bias voltage. |
| `GetBiasCalibration` | builtins/bias | auto | read | - | bias, calibration, read | Read the bias calibration factor and offset. |
| `GetCurrent` | builtins/bias | auto | read | - | current, read | Read the tunneling current. |
| `SetBias` | builtins/bias | confirm | write | bias_v, slew_rate_v_per_s | bias, write | Set the bias voltage. Use slew_rate_v_per_s for gradual ramping during large voltage changes. |
| `SetBiasCalibration` | builtins/bias | confirm | write | calibration, offset | bias, calibration, write, dangerous | Set the bias calibration factor and offset. Affects all bias measurements. |
| `SetBiasRange` | builtins/bias | confirm | write | range_index | bias, range, write | Select the bias range by index. |
| `BiasPulse` | builtins/bias_pulse | auto | write | width_s, bias_v, z_hold, absolute | bias, pulse, write | Generate a single bias pulse with hardware timing. |
| `GetSignalCalibration` | builtins/bias_sweep | auto | read | signal_index | signals, calibration, read | Read a signal's calibration (gain + offset) — what one unit of the raw value means in physical units. MAST could list the signals and read their values, and could not say what the numbers MEANT. Use before interpretin… |
| `RunBiasSweep` | builtins/bias_sweep | confirm | write | lower_limit_v, upper_limit_v, steps, period_ms, z_controller_off, sweep_direction, autosave | bias, sweep, spectroscopy, write | Run the Nanonis BIAS SWEEPER: ramp the bias between two limits and record the acquisition channels. This is NOT bias spectroscopy (BiasSpectr / RunSTS) — the sweeper just ramps and records, without the spectroscopy mo… |
| `SetAcquisitionPeriod` | builtins/bias_sweep | confirm | write | period_s | util, acquisition, write | Set the Nanonis acquisition period (the controller's sampling interval). MAST could READ this (GetAcqPeriod) and not set it. Lowering it samples faster and costs bandwidth; it affects EVERY measurement the controller … |
| `SetAdditionalRealtimeSignals` | builtins/bias_sweep | auto | write | signal_1, signal_2 | signals, write | Select the two ADDITIONAL real-time signals Nanonis computes and streams (on top of the fixed ones). Configuration only — it changes what is measured, never what the instrument does. |
| `ReadCalibrations` | builtins/calibration_readout | auto | read | which | read, calibration, diagnostic, instrument_profile | 读取仪器档案里的标定值:①倾斜响应矩阵 G(AutoTilt 依赖它)②接触点 dI/dV ③qPlus 实测共振 f₀/Q。每一项都带**年龄**与**适用边界**。**纯读,不动任何硬件。**  ⚠️ **被「未标定」挡住时先调本工具**——它会告诉你到底是「从未标定过」还是「读不到档案」,这两件事的处置完全不同。  ⚠️ **条件数只管形状不管方向**:一个好看的条件数**不等于**标定可用。 |
| `GetChamberPressure` | builtins/chamber | auto | read | - | vacuum, pressure, read, safety, coarse | 读取腔体压强(Pa)并给出**粗动互锁裁决**:现在能不能动粗动马达,以及理由。 在中间真空区(约 0.1–1000 Pa = 1e-3–10 mbar,Paschen 极小值附近)给粗动压电加几百伏会打火击穿叠堆 —— 抽气和放气途中正好穿过这个区间。 **allow=false 时不要重试粗动**:这是硬闸门,不是建议。读不到真空计也会是 false(读不到 ≠ 真空好);那种情况需要操作员在界面上签署一次「当前气压安全」,你… |
| `FindCleanSpot` | builtins/clean_spot | auto | read | purpose, exclude_spots, from_x_m, from_y_m, max_distance_m, count | map, position, tip, read | Nearest spot to the tip that the scan map shows has NOT been damaged, for the next pulse or plunge. Reads the recorded markers — never guesses from a picture. Returns map_known=false when there is no experiment record… |
| `ExtractClusters` | builtins/cluster_extract | auto | analysis | scan_path, channel, polarity, threshold_mad, level, tilt_warn_ratio, min_area_px, max_clusters | scan, analysis, cluster, extract, read | Segment ALL clusters in a saved .sxm and return them as a LIST with geometry + real xy coordinates + peak height. Works on a PARTIALLY scanned frame (uses only the fully-scanned rows). Returns everything it finds, inc… |
| `CoarseMotionSelfCheck` | builtins/coarse_selfcheck | auto | read | probe_step_counter | coarse, motor, vacuum, selfcheck, read, commissioning | 粗动子系统**只读**自检:真空计(型号/量程/当前读数/互锁裁决)、粗动驱动电压(声明值 + 实际读回)、步进计数器是否支持、qPlus 振幅通道、lock-in 索引、退针方向配置、粗动大地图状态、以及本机的温度。 **不发任何移动命令、不改任何设定值** —— 进针状态下、扫描过程中都可以跑。 返回的 todo 列表就是「还差哪些只有真机能回答的数」。 |
| `GetCurrentBEEM` | builtins/current | auto | read | - | current, beem, read | Get the BEEM current value from the Current module. |
| `SetCurrentCalibration` | builtins/current | confirm | write | gain_index, calibration, offset | current, calibration, write | Set the calibration and offset for a selected gain in the Current module. |
| `SetCurrentGain` | builtins/current | confirm | write | gain_index, filter_index | current, gain, write | Set the current amplifier gain index and filter. |
| `GetDataLogStatus` | builtins/datalog | auto | read | - | datalog, read | Read the data logger's status (running / stopped), its configured channels, and its properties. Check this before starting a new log — starting one over a running log loses the first. |
| `GetTcpLogStatus` | builtins/datalog | auto | read | - | tcplog, read | Read the TCP logger's status (streaming / stopped / error). |
| `StartDataLog` | builtins/datalog | auto | write | channels, duration_s, basename, averaging, comment | datalog, record, monitor | Record one or more signal channels to a file, for a fixed duration or until stopped. Use this to WATCH something over time instead of polling it: thermal drift settling, a slow tip degradation, the current during a lo… |
| `StartTcpLog` | builtins/datalog | auto | write | channels, oversampling | tcplog, record, stream | Start the TCP logger: stream selected channels over TCP rather than recording them to a file on the Nanonis machine. Use when the data should come HERE. Reading only — touches no hardware. |
| `StopDataLog` | builtins/datalog | auto | write | - | datalog, record, stop | Stop the Nanonis data logger and close the file. |
| `StopTcpLog` | builtins/datalog | auto | write | - | tcplog, record, stop | Stop the Nanonis TCP logger stream. |
| `GetPointShootOnOff` | builtins/folme | auto | read | - | folme, point_shoot, read | Read whether Point & Shoot is enabled or disabled in Follow-Me mode. |
| `GetPointShootProps` | builtins/folme | auto | read | - | folme, point_shoot, config, read | Read Point & Shoot configuration: auto-resume, basename, external VI path, pre-measure delay. |
| `GetTipSpeed` | builtins/folme | auto | read | - | folme, speed, read | Read the tip surface speed and custom-speed flag in Follow-Me mode. |
| `SetFolMeOversampling` | builtins/folme | confirm | write | oversampling | folme, oversampling, write | Set the oversampling of acquired data when moving in Follow-Me mode. |
| `SetPointShootExperiment` | builtins/folme | confirm | write | experiment_index | folme, point_shoot, experiment, write | Select which experiment to run for Point & Shoot in Follow-Me mode. |
| `SetPointShootOnOff` | builtins/folme | confirm | write | enable | folme, point_shoot, write | Enable or disable Point & Shoot in Follow-Me mode. |
| `SetTipSpeed` | builtins/folme | confirm | write | speed_m_s, custom_speed | folme, speed, write | Set the tip movement speed for Follow-Me (XY positioning) mode. |
| `StopFolMe` | builtins/folme | auto | write | - | folme, stop, safety | Stop the tip movement in Follow-Me mode. |
| `ConfigureWaveform` | builtins/function_generator | confirm | write | generator, amplitude, frequency_hz, channel, shape, polarity, direction | output, function_generator, waveform, write | Configure a Nanonis function generator: amplitude, frequency, and (2-channel generator only) the waveform shape. This drives EXTERNAL hardware MAST cannot see — a modulation, a lock-in reference, a chopper, a gate ram… |
| `GetWaveformStatus` | builtins/function_generator | auto | read | generator, channel | output, function_generator, read | Read a function generator's status and its current settings (amplitude / frequency / shape / idle value). Check before starting one you did not configure yourself. |
| `SetWaveformChannelOnOff` | builtins/function_generator | confirm | write | channel, on | output, function_generator, write | Enable or disable one channel of the 2-channel function generator without stopping the other. |
| `SetWaveformIdleValue` | builtins/function_generator | confirm | write | generator, idle_value, device | output, function_generator, write | Set the IDLE value a function generator holds when stopped. This is the resting state of an output line, so it matters more than it sounds: it is what the external hardware sees between bursts, and after a StopWavefor… |
| `StartWaveform` | builtins/function_generator | confirm | write | generator, periods, wait_until_finished | output, function_generator, waveform, write | Start a function generator for a number of periods (0 = run until stopped). Configure it first with ConfigureWaveform. This puts a live signal on an output line — MAST cannot see what is on the other end. |
| `StopWaveform` | builtins/function_generator | auto | write | generator | output, function_generator, stop | Stop a running function generator. A STOP is always allowed — it is the one thing you always want to be able to do to an output. |
| `ReadHardwareEvents` | builtins/hardware_events | auto | read | limit, min_severity | read, events, monitoring, hitl, diagnostic | 读取硬件事件缓冲区:①当前有没有审批闸门拦着你、拦的是哪条事件;②最近 N 条事件(种类/严重度/来源/一句话摘要/**触发时的实测指标**/建议动作)。**纯读,不动任何硬件,也不解除任何拦截。**  ⚠️ **被 buffer_hitl 拦住写入类工具时,先调用本工具看证据再决定**——拦截文案只给事件种类,指标在这里。不要反复重试被拒的工具。  也用于日常自主判断:监控在报什么、要不要主动修针。注意 `read_latest… |
| `GetScanXYPosition` | builtins/imaging | auto | read | wait_for_newest | scan, position, read | Read the current scan X and Y position. |
| `ScanBackgroundDelete` | builtins/imaging | confirm | write | wait_until_deleted, timeout_ms, delete_all | scan, background, delete, write | Delete the latest or all pasted scan backgrounds. |
| `ScanBackgroundPaste` | builtins/imaging | confirm | write | wait_until_pasted, timeout_ms | scan, background, paste, write | Paste the current scan databuffer into the background. |
| `SetScanSpeed` | builtins/imaging | confirm | write | fwd_speed, bwd_speed, fwd_line_time, bwd_line_time, keep_const, speed_ratio | scan, speed, write | Set scan speed: forward/backward speed or line time. |
| `StopScan` | builtins/imaging | auto | write | - | scan, imaging, write | Stop the current scan. |
| `HomeZController` | builtins/instrument_limits | confirm | write | - | z, home, write | Move Z to its configured HOME position.  Home is a Z position the operator set as the safe/neutral parking spot — read it with GetZControllerState.home before calling this, because MAST does not choose it and a home c… |
| `SetActiveZController` | builtins/instrument_limits | confirm | write | controller_index | z, controller, write | Select which Z controller is ACTIVE. Only meaningful on a rig with more than one — call GetZControllerState / ListZControllers first.  Switching the active controller changes which loop every other Z skill talks to. G… |
| `SetPiezoLimits` | builtins/instrument_limits | confirm | write | x_low_v, x_high_v, y_low_v, y_high_v, z_low_v, z_high_v, enable | piezo, limits, safety, write | Set the piezo's X/Y/Z VOLTAGE limits (and enable them).  These bound how far the scanner can be driven, in volts, before the range calibration turns them into metres. They protect the piezo itself (over-voltage depole… |
| `SetSafeTipProps` | builtins/instrument_limits | confirm | write | threshold, auto_recovery, auto_pause_scan | safetip, safety, write | Configure SafeTip — the automatic tip-protection system that watches a signal and retracts when it crosses a threshold.  MAST could READ this configuration and not set it, which meant the agent could see that the thre… |
| `SetWithdrawRate` | builtins/instrument_limits | confirm | write | rate_m_per_s | z, withdraw, safety, write | Set the Z withdraw slew rate, in metres per second — how fast the tip retracts when Withdraw (or an emergency retract, or SafeTip) fires.  This is a safety parameter in both directions. Too SLOW and an emergency retra… |
| `SetZLimits` | builtins/instrument_limits | confirm | write | z_high_limit_m, z_low_limit_m, enable | z, limits, safety, write | Set the Z-controller's high and low position limits, in METRES.  These are the bounds beyond which the Z piezo will not extend or retract — the last software barrier between the piezo and the tip. Widening them is leg… |
| `GetDemodHPFilter` | builtins/lockin | auto | read | demodulator | lockin, demodulator, hp, filter, read | Read high-pass filter order and cutoff for a lock-in demodulator. |
| `GetDemodHarmonic` | builtins/lockin | auto | read | demodulator | lockin, demodulator, harmonic, read | Read the harmonic overtone of a lock-in demodulator. |
| `GetDemodLPFilter` | builtins/lockin | auto | read | demodulator | lockin, demodulator, lp, filter, read | Read low-pass filter order and cutoff for a lock-in demodulator. |
| `GetDemodPhasReg` | builtins/lockin | auto | read | demodulator | lockin, demodulator, phase, register, read | Read the phase register index (1-8) of a lock-in demodulator. |
| `GetDemodPhase` | builtins/lockin | auto | read | demodulator | lockin, demodulator, phase, read | Read the reference phase of a lock-in demodulator. |
| `GetDemodSignal` | builtins/lockin | auto | read | demodulator | lockin, demodulator, signal, read | Read the demodulated signal index (0-127) for a lock-in demodulator. |
| `SetDemodRTSignals` | builtins/lockin | confirm | write | demodulator, rt_signals | lockin, demodulator, rt, signals, write | Set RT signals (X/Y or R/phi) for a lock-in demodulator. |
| `SetDemodSyncFilter` | builtins/lockin | confirm | write | demodulator, sync_filter_on | lockin, demodulator, sync, filter, write | Switch the sync filter on/off for a lock-in demodulator. |
| `SetModHarmonic` | builtins/lockin | confirm | write | modulator, harmonic | lockin, modulator, harmonic, write | Set the harmonic overtone of a lock-in modulator. |
| `SetModPhasReg` | builtins/lockin | confirm | write | modulator, phase_register_index | lockin, modulator, phase, register, write | Assign a lock-in modulator to a phase register (1-8). |
| `SetModSignal` | builtins/lockin | confirm | write | modulator, signal_index | lockin, modulator, signal, write | Select the modulated signal (by index 0-127) for a lock-in modulator. |
| `ListLockInPresets` | builtins/lockin_presets_skills | auto | read | - | lockin, preset, read | 列出 lock-in 常用参数组:每个值是多少、来自哪个档案键、哪些键操作员还没填(没填的**不会下发**)。调制侧相位永远不在组里 —— 本机固件不接受写它。 |
| `DrawScanMarker` | builtins/marks | auto | write | kind, x_m, y_m, x2_m, y2_m, text, color | marks, annotation, scan | Draw a POINT or a LINE marker onto the Nanonis scan frame. Use it to record WHERE you did something — mark each STS position, the flat region you chose, a defect you found, the line you took a profile along. Coordinat… |
| `EraseScanMarkers` | builtins/marks | auto | write | kind, index, hide_only | marks, annotation, scan | Erase a marker (or hide it without deleting). index=-1 erases ALL markers of that kind. Erasing a marker touches only the display. |
| `ListScanMarkers` | builtins/marks | auto | read | - | marks, annotation, scan, read | List the point and line markers on the Nanonis scan frame, with their coordinates. Useful to recall where you already measured before choosing the next spot. |
| `ConfigureScopeTrigger` | builtins/misc_setters | auto | write | trigger_mode, trigger_slope, trigger_level, trigger_hysteresis | oscilloscope, trigger, write | Set the 1-channel oscilloscope's trigger: mode, slope, level and hysteresis. Reads only — a scope digitises, it drives nothing.  The level is in the triggered channel's own physical units. Hysteresis is what stops a n… |
| `SetLockInDemodPhaseRegister` | builtins/misc_setters | auto | write | demodulator, phase_register | lockin, demod, write | Set which PHASE REGISTER a lock-in demodulator references.  The demodulator's phase must be referenced to the modulation that produced the signal. Pointing it at the wrong register does not fail — it rotates X into Y,… |
| `SetLockInFrequencySweepSignal` | builtins/misc_setters | confirm | write | signal_index | lockin, sweep, write | Choose which signal the lock-in FREQUENCY SWEEP sweeps.  The sweep drives whatever you name across the frequency range — this is how you find a resonance. Naming the wrong signal means driving the wrong thing. Read th… |
| `SetPatternExperiment` | builtins/misc_setters | confirm | write | experiment, basename, pre_measure_delay_s, save_scan_channels, external_vi_path | pattern, grid, write | Choose WHICH EXPERIMENT a pattern (grid / line / cloud) runs at each point, plus the file basename and the pre-measure delay.  This is the parameter that turns a set of coordinates into a measurement. A grid pointed a… |
| `SetPllDemodHarmonic` | builtins/misc_setters | auto | write | demodulator, harmonic | pll, demod, write | Set which HARMONIC a PLL demodulator locks to (1 = the fundamental).  Higher harmonics carry different information about the tip-sample interaction, and a demodulator locked to a harmonic that is not there reports noi… |
| `SetPllExcitationAdd` | builtins/misc_setters | confirm | write | modulator, add | pll, excitation, write | Add (or stop adding) the PLL's excitation signal to its output.  With Add on, the modulator's excitation reaches the cantilever/tuning fork — the probe is being DRIVEN. With it off, the loop still tracks but drives no… |
| `SetPointShootProps` | builtins/misc_setters | confirm | write | auto_resume, basename, use_own_basename, pre_measure_delay_s, external_vi_path | folme, point-and-shoot, write | Configure Follow-Me point-and-shoot: whether the scan auto-resumes afterwards, the file basename, and the pre-measure delay.  `auto_resume` is the one worth thinking about. With it on, the scan picks up again after ea… |
| `SetWaveformSignal` | builtins/misc_setters | confirm | write | channel, signal_index | fungen, waveform, write | Choose WHICH SIGNAL a function-generator channel drives.  This is the parameter that decides what the waveform actually does. The same 1 V sine is a harmless test signal on a spare output and a 1 V bias modulation on … |
| `GetMotorFreqAmp` | builtins/motor | auto | read | axis | motor, coarse, read, safety | 读回粗动马达当前的驱动频率与幅度(电压),并与本机声明的耐压上限比对。**只读** —— 设置驱动电压是操作员的权限,agent 不能改。粗动移动前会自动做这个核对;读不到就拒绝粗动(读不到 ≠ 没问题)。 |
| `GetMotorStepCounter` | builtins/motor | auto | read | reset_x, reset_y, reset_z | motor, step_counter, read | Read step counter values for X, Y, Z axes. Optionally reset after reading. Attocube ANC150 only. |
| `MotorGetPos` | builtins/motor | auto | read | - | motor, read | Read coarse motor position. |
| `MotorMove` | builtins/motor | confirm | write | direction, steps | motor, coarse, dangerous | Move coarse motor (pan-type stepper). x±/y± lateral; 'z-approach' steps TOWARD the sample (DANGEROUS — human approval required); 'z-retract' steps AWAY (safe). |
| `MotorMoveClosedLoop` | builtins/motor | confirm | write | absolute, target_x_m, target_y_m, target_z_m, wait, group | motor, coarse, closed_loop, dangerous | Move coarse motor in closed loop to target XYZ position. DANGEROUS: can collide with sample. Not all controllers support this. |
| `SetMotorFreqAmp` | builtins/motor | confirm | write | frequency_hz, amplitude_v, axis | motor, frequency, amplitude, write | 设置粗动马达的驱动频率与幅度(电压)。**这是操作员的参数,不是 agent 的。**写入必须落在【高级】页声明的本机耐压上限之内;未声明则一律拒绝,超上限**直接拒绝而不是降到上限**(悄悄降下来会让调用方以为自己设的是另一个值)。 |
| `StopMotor` | builtins/motor | auto | write | - | motor, stop, safety | Emergency stop all motor movement. |
| `DeployNanonisScript` | builtins/nanonis_script | confirm | write | slot | script, realtime, write | Deploy a vetted script slot onto the real-time controller (compile + push). Deploying does not run it — call RunNanonisScript after. Only vetted slots may be deployed. |
| `DeployScriptLUT` | builtins/nanonis_script | confirm | write | lut_index, wait_until_finished, timeout_ms | script, lut, write | Deploy a LUT onto the real-time controller so the running script can step through it. Load the values first (LoadScriptLUT). |
| `GetScriptChannels` | builtins/nanonis_script | auto | read | buffer | script, read | Read the channel list of a script's Acquire Buffer. |
| `GetScriptData` | builtins/nanonis_script | auto | read | buffer, sweep | script, data, read | Read the data a Nanonis script recorded into an Acquire Buffer. The script writes channels into buffer 1 or 2 at real-time speed; each 'sweep' is one pass as defined in the script (sweeps start at 0). Returns a 2-D ar… |
| `ListNanonisScripts` | builtins/nanonis_script | auto | read | - | script, read, safety | List the Nanonis script slots the OPERATOR has vetted for autonomous use, with what each does and what LUT range it accepts. Call this BEFORE trying to run anything: an unvetted slot is refused, and the list is usuall… |
| `LoadScriptLUT` | builtins/nanonis_script | confirm | write | slot, lut_index, values | script, lut, write, safety | Load values into a script's Look-Up Table — the array the script steps through at real-time speed. This is how a delay scan works: load the delay values, the script walks them.  The LUT is the ONE thing you can really… |
| `RunNanonisScript` | builtins/nanonis_script | confirm | write | slot, wait_until_finished | script, realtime, write, safety | Run a Nanonis script on the REAL-TIME CONTROLLER. Only slots the operator has vetted (config/nanonis_scripts.json) may be run — call ListNanonisScripts first.  READ THIS BEFORE USING IT. A running script executes on t… |
| `SetScriptAutosave` | builtins/nanonis_script | auto | write | buffer, sweep, all_sweeps_same_file, folder_path, basename | script, write | Auto-save a script's Acquire Buffer data to file after the run. Recommended for any long sequence — the buffer is finite, and data you did not save is data you did not take. |
| `SetScriptChannels` | builtins/nanonis_script | auto | write | buffer, channels | script, write | Set which signal channels a script's Acquire Buffer records. Configuration only — it changes what is measured, never what the instrument does. |
| `StopNanonisScript` | builtins/nanonis_script | auto | write | - | script, stop, safety | Stop the script running on the real-time controller. This is the ONLY thing that stops a running script — the abort gate cannot, because the script is not making TCP calls. It is therefore never gated and never refuse… |
| `UndeployNanonisScript` | builtins/nanonis_script | auto | write | slot | script, realtime, write | Undeploy a script slot from the real-time controller. Removing a script is never the dangerous direction, so this is never gated. |
| `LoadNanonisScript` | builtins/nanonis_script_files | dangerous | write | slot, file_path, load_session | script, file, advanced, dangerous | Load a Nanonis script file into a script SLOT.  **It will REFUSE to load into a slot that is on the vetted allow-list.** The allow-list means 'a human read the script in this slot and approved it'. Putting a different… |
| `SaveNanonisScript` | builtins/nanonis_script_files | confirm | write | slot, file_path, save_session | script, file, advanced | Save the script currently in a slot out to a file on the NANONIS machine.  This reads the slot and WRITES A FILE. It cannot change what the instrument does — but it can overwrite an existing file at the path you give … |
| `SaveNanonisScriptLut` | builtins/nanonis_script_files | confirm | write | slot, file_path | script, lut, file, advanced | Save a script slot's LOOKUP TABLE (LUT) out to a file on the Nanonis machine.  The LUT is how a script receives parameters — it is the only thing about a vetted script the agent can change (within the bounds the allow… |
| `MoveToXY` | builtins/navigation | confirm | write | x_m, y_m, wait | navigation, move, write | Move the tip to a specified XY position using Follow Me. |
| `PumpProbeScan` | builtins/optics_pump_probe | confirm | write | delay_start_ps, delay_stop_ps, points, samples_per_point, sample_interval_s, settle_extra_s, read_current, read_lockin, lockin_signal_index, lockin_demod, return_to_start, move_timeout_s, tag, extra_signal_indices, trigger_enable, trigger_port, trigger_line, trigger_width_s, secondary_device_id, secondary_axis, secondary_position, secondary_scan, secondary_start, secondary_stop, secondary_points, dry_run | optics, pump_probe, delay_line, acquisition, write | Pump-probe delay scan: sweep the optical delay line from delay_start_ps to delay_stop_ps in `points` steps; at each point wait for the stage to settle, then average the tunnel current (Current_Get) and/or a lock-in de… |
| `AcquireSignalPoint` | builtins/optics_scan | auto | read | read_current, signal_indices, samples, sample_interval_s | optics, acquire, signal, read | Read and software-average Nanonis signals at the CURRENT optical position: the tunnel current and/or any signal slots (0-127), each sampled `samples` times. The atomic 'measure here' — pair it with OpticalStageMove / … |
| `OpticalStageScan` | builtins/optics_scan | confirm | write | device1, axis1, start1, stop1, points1, device2, axis2, start2, stop2, points2, read_current, signal_indices, samples_per_point, sample_interval_s, settle_extra_s, move_timeout_s, trigger_enable, trigger_port, trigger_line, trigger_width_s, serpentine, return_to_start, dry_run, tag | optics, stage, scan, map, acquisition, write | Raster-scan an optical stage and map a Nanonis signal over it. Steps the fast axis (device1/axis1) from start1→stop1 in points1 steps; give a second (slow) axis for a 2D map. At each point it settles, optionally pulse… |
| `DelayLineGetDelay` | builtins/optics_stage | auto | read | - | optics, delay_line, pump_probe, read | Read the current pump-probe optical delay in picoseconds (and the reachable delay range from the stage travel). |
| `DelayLineMoveTo` | builtins/optics_stage | confirm | write | delay_ps, wait, timeout_s | optics, delay_line, pump_probe, move, write | Move the pump-probe delay line to a target optical delay in picoseconds (converted to stage position via the calibrated zero-offset; stage soft limits enforced in the driver). |
| `HomeOpticalStage` | builtins/optics_stage | confirm | write | device_id, axis, timeout_s | optics, stage, home, write | Reference (find-zero) an optical stage axis. The stage sweeps toward its reference mark, losing the current position — confirm no measurement depends on it. Piezo axes with absolute sensors (PI E-816) report no-homing. |
| `ListOpticalDevices` | builtins/optics_stage | auto | read | - | optics, inventory, read | List optical-bench motion devices from config/optical_instruments.json: id, type, axes with travel limits and roles, connection state, and the pump-probe delay-line binding if configured. Start here to learn which dev… |
| `OpticalStageGetPos` | builtins/optics_stage | auto | read | device_id, axis | optics, stage, read | Read the current position of an optical-bench stage axis (native unit, see ListOpticalDevices for units/limits). |
| `OpticalStageMove` | builtins/optics_stage | confirm | write | device_id, axis, position, relative, wait, timeout_s | optics, stage, move, write | Move an optical-bench stage axis to a position (native unit). Soft travel limits are enforced in the driver and reject out-of-range targets before any motion. Use relative=true for a delta move. |
| `OpticalStageWiggle` | builtins/optics_stage | confirm | write | device_id, axis, delta, tolerance, return_to_start, timeout_s | optics, stage, diagnostic, selftest, write | Diagnostic wiggle of an optical stage axis: move by a small delta and back, then report whether it actually moved, in which direction, and the observed/commanded ratio (scale check). Use during bring-up to confirm an … |
| `StopOpticalStage` | builtins/optics_stage | auto | write | device_id | optics, stage, stop, safety | Immediately stop optical stage motion. Omit device_id to panic-stop every connected optical device. |
| `AutoZeroBeamDeflection` | builtins/optional_afm | confirm | write | deflection_signal | beam_deflection, afm, optional-hardware | Auto-zero the beam-deflection signal: measure its present value and subtract it as an offset, so 'zero deflection' means the cantilever's current resting position.  Do this with the tip WITHDRAWN and the cantilever fr… |
| `ConfigureBeamDeflection` | builtins/optional_afm | confirm | write | axis, name, units, calibration, offset | beam_deflection, afm, calibration, optional-hardware | Set the calibration of ONE beam-deflection axis: vertical (normal force), horizontal (lateral / friction), or the intensity sum.  The calibration converts photodiode volts into physical units (N/m of deflection, nN of… |
| `ConfigureInterferometer` | builtins/optional_afm | confirm | write | integral, proportional, sign, w_piezo, null_deflection | interferometer, afm, optional-hardware | Configure the interferometric deflection detector: its PI loop gains and sign, and the working-point piezo voltage.  null_deflection=true additionally runs the null-deflection routine, which moves the interferometer p… |
| `ConfigureKelvinController` | builtins/optional_afm | confirm | write | bias_high_limit_v, bias_low_limit_v, setpoint, p_gain, time_constant_s, slope, control_signal_index, modulation_frequency_hz, modulation_amplitude, modulation_phase_deg | kpfm, kelvin, optional-hardware | Configure the Kelvin (KPFM) controller: the demodulated signal it servos on, its P gain and time constant, the setpoint, the AC modulation, and the BIAS LIMITS.  The bias limits are the important parameter. The Kelvin… |
| `GetBeamDeflection` | builtins/optional_afm | auto | read | - | beam_deflection, afm, read, optional-hardware | Read the beam-deflection detector's configuration for all three axes (vertical, horizontal, sum): name, units, calibration and offset. Check the calibration before trusting any force number. |
| `GetCpdCompensation` | builtins/optional_afm | auto | read | - | kpfm, cpd, read, optional-hardware | Read the CPD compensation module's measured contact potential difference and its current parameters. |
| `GetInterferometer` | builtins/optional_afm | auto | read | - | interferometer, afm, read, optional-hardware | Read the interferometer: the current deflection value, whether its loop is closed, its gains, and the piezo working point. |
| `GetKelvinController` | builtins/optional_afm | auto | read | - | kpfm, kelvin, read, optional-hardware | Read the Kelvin controller: whether the loop is closed, its setpoint, gains, bias limits, modulation parameters, and the demodulated amplitude. Read the amplitude to judge whether the loop has anything to servo on — a… |
| `GetLaser` | builtins/optional_afm | auto | read | - | laser, optical, read, optional-hardware | Read the laser: whether it is ON, its measured power, and its setpoint. Check this before assuming the laser is off. |
| `RunCpdCompensation` | builtins/optional_afm | confirm | write | range_v, speed_hz, averaging | kpfm, cpd, optional-hardware | Open the CPD compensation module and run it: it SWEEPS THE BIAS over the given range to find the contact-potential parabola, rather than servoing to it like the Kelvin loop does.  CAUTION: this drives the bias across … |
| `SetInterferometerOnOff` | builtins/optional_afm | confirm | write | on, reset | interferometer, afm, optional-hardware | Switch the interferometer's PI control loop on or off. reset=true resets the loop's integrator first — do that if the loop has wound up and is sitting at a rail. |
| `SetKelvinControllerOnOff` | builtins/optional_afm | confirm | write | on, modulation_on, ac_mode | kpfm, kelvin, optional-hardware | Switch the Kelvin (KPFM) feedback loop on or off, together with its AC modulation.  CAUTION when switching ON: from that moment the loop DRIVES THE BIAS on its own, continuously, with the tip in range. A mis-tuned loo… |
| `SetLaserOnOff` | builtins/optional_afm | dangerous | write | on | laser, optical, dangerous, optional-hardware | Switch the laser on or off.  DANGEROUS when switching ON. Laser light is an eye hazard and a heat load on the junction; MAST cannot see whether a shutter is closed, whether anyone is at the microscope, or where the be… |
| `SetLaserPower` | builtins/optional_afm | confirm | write | setpoint | laser, optical, optional-hardware | Set the laser's power setpoint. Does NOT switch the laser on — SetLaserOnOff does that. Setting the power while the laser is off is the safe way to stage a value: set it, read it back with GetLaser, then turn it on. |
| `ConfigureOcSync` | builtins/optional_controllers | confirm | write | ch1_on_deg, ch1_off_deg, ch2_on_deg, ch2_off_deg, link_channels | oc_sync, phase, optional-hardware | Set the OC Sync module's on/off phase angles, in DEGREES. These gate the oscillation-control outputs to fire only within a phase window of the oscillation — the basis of phase-resolved (pump-probe-style) measurement o… |
| `ConfigurePiController` | builtins/optional_controllers | confirm | write | controller_index, input_index, control_signal_index, setpoint, output_lower_limit, output_upper_limit, p_gain, i_gain, slope | pi_controller, feedback, optional-hardware | Configure one generic PI controller (V5e): which signal it WATCHES (input), which it DRIVES (control signal), the setpoint, the gains, and the output limits.  The output limits are the important parameter. This loop d… |
| `ConfigurePllSignalAnalyzer` | builtins/optional_controllers | auto | write | channel_index, timebase, update_rate, fft_window, averaging_mode, weighting_mode, count | pll, analyzer, optional-hardware | Configure the PLL signal analyser: which channel, the timebase, and the FFT window/averaging. It is an oscilloscope + FFT on the PLL's own signals — reads only, drives nothing.  Use GetPllSignalAnalyzerData to collect… |
| `ConfigurePreamp` | builtins/optional_controllers | confirm | write | preamp, channel, gain, coupling, input_mode | preamp, mcva5, gain, optional-hardware | Set an MCVA5 preamplifier channel's gain, coupling (AC/DC) and input mode.  Changing the preamp gain changes what a given physical current READS AS. With the Z-controller closed, the feedback loop sees that as a sudde… |
| `ConfigureTipRecorder` | builtins/optional_controllers | auto | write | buffer_size, clear | tip_recorder, optional-hardware | Set the tip-move recorder's buffer size, and optionally clear it.  The recorder logs every tip movement into a ring buffer — it is how you reconstruct where the tip has BEEN, which is exactly what you want after an un… |
| `GetGenericPiController` | builtins/optional_controllers | auto | read | - | pi_controller, read, optional-hardware | Read the V5 generic PI controller: loop state, setpoint and gains, the analogue output's current value, and — the one you need before writing anything — the output's NAME, UNITS AND LIMITS. |
| `GetOcSync` | builtins/optional_controllers | auto | read | - | oc_sync, read, optional-hardware | Read the OC Sync module's phase angles and channel links. |
| `GetPiController` | builtins/optional_controllers | auto | read | controller_index | pi_controller, feedback, read, optional-hardware | Read one generic PI controller (V5e): whether it is closed, which signal it watches, WHICH SIGNAL IT DRIVES, its setpoint, gains and output limits.  Read this before closing any loop you did not configure yourself. Th… |
| `GetPllSignalAnalyzerData` | builtins/optional_controllers | auto | read | rearm | pll, analyzer, read, optional-hardware | Read the PLL signal analyser: the oscilloscope time trace AND the FFT spectrum, plus the trigger state.  rearm=true re-arms the trigger before reading — do that when the trigger is not free-running, or you get the pre… |
| `GetPllZoomFftData` | builtins/optional_controllers | auto | read | - | pll, fft, read, optional-hardware | Read the PLL Zoom-FFT spectrum and its current settings. |
| `GetPreamp` | builtins/optional_controllers | auto | read | preamp, channel | preamp, mcva5, read, optional-hardware | Read an MCVA5 preamplifier channel: its gain, coupling and input mode. Read the gain before interpreting any current — the number the ADC reports means nothing without it. |
| `GetTipRecorderData` | builtins/optional_controllers | auto | read | - | tip_recorder, read, optional-hardware | Read the recorded tip-movement history from the tip recorder's buffer. Use this to reconstruct where the tip went — the first thing to look at when an unattended run ends somewhere unexpected. |
| `RunPllPhaseSweep` | builtins/optional_controllers | confirm | write | modulator_index, get_data | pll, phase, sweep, optional-hardware | Sweep the PLL phase on one modulator and (optionally) return the resulting curve.  This is how you find the phase at which the PLL actually locks. It DRIVES THE EXCITATION while sweeping — a cantilever or tuning fork … |
| `RunPllZoomFft` | builtins/optional_controllers | auto | write | channel_index, fft_window, averaging_mode, weighting_mode, count, restart_averaging | pll, fft, read, optional-hardware | Open the PLL Zoom-FFT and start it on a channel, with a window and averaging. Reads only — an FFT drives nothing.  Use it to see the noise floor and the spurs around the resonance: the zoom FFT resolves structure the … |
| `SetGenericPiOutput` | builtins/optional_controllers | confirm | write | value | pi_controller, output, optional-hardware | Set the V5 generic PI controller's analogue output directly, in its own PHYSICAL UNITS (not volts — the module applies its own calibration).  CAUTION: this writes straight to a physical output whose wiring MAST cannot… |
| `SetPiControllerOnOff` | builtins/optional_controllers | dangerous | write | controller_index, on | pi_controller, feedback, dangerous, optional-hardware | Close or open one generic PI control loop (V5e).  DANGEROUS when closing. From that moment the loop DRIVES ITS OUTPUT on its own — and the output can be a piezo, the bias, a heater, a laser. Nothing in the module know… |
| `StopPllPhaseSweep` | builtins/optional_controllers | auto | write | modulator_index | pll, phase, stop, optional-hardware | Stop a running PLL phase sweep. Always allowed, including after an abort. |
| `ConfigureProbeCurrentGain` | builtins/optional_multiprobe | confirm | write | probe, gain_index, filter_index | multiprobe, current, gain, optional-hardware | Set ONE probe's current-preamp gain and filter, by INDEX into that probe's own gain list (read it with GetProbeCurrent).  Changing the gain changes what a given current READS AS. Do it with the Z-controller off or the… |
| `ConfigureProbeScanner` | builtins/optional_multiprobe | confirm | write | probe, speed, factor_x, factor_y, factor_z | multiprobe, scanner, calibration, optional-hardware | Set ONE probe's scanner calibration (X/Y/Z factors) and movement speed.  The calibration converts commanded volts into metres. It is not a cosmetic setting: a wrong X factor means MoveProbeXY travels a different dista… |
| `GetProbeBias` | builtins/optional_multiprobe | auto | read | probe | multiprobe, bias, read, optional-hardware | Read ONE probe's bias voltage, its range setting and its calibration. |
| `GetProbeCurrent` | builtins/optional_multiprobe | auto | read | probe | multiprobe, current, read, optional-hardware | Read ONE probe's tunnelling current (in amperes) and its available preamp gains. Use this per-probe reading to judge whether that specific probe is in tunnelling range — the main Current channel reports one probe only. |
| `GetProbeZController` | builtins/optional_multiprobe | auto | read | probe | multiprobe, zcontroller, read, optional-hardware | Read ONE probe's Z-controller: whether it is on, its setpoint, gains, Z position and Z limits. Read this before moving that probe. |
| `MoveProbeXY` | builtins/optional_multiprobe | dangerous | write | probe, x_m, y_m | multiprobe, scanner, motion, dangerous, optional-hardware | Move ONE probe to an absolute (X, Y) position, in METRES.  DANGEROUS. This physically drives a tip across the sample. On a multi-probe rig the probes share one surface and can be micrometres apart — a move computed ag… |
| `PulseProbeBias` | builtins/optional_multiprobe | confirm | write | probe, value_v, width_s, hold_z, relative, wait | multiprobe, bias, pulse, optional-hardware | Apply a bias PULSE on ONE probe: jump to a value for a set width, then return.  CAUTION. A bias pulse is how you deliberately modify the tip or the surface — it is the multi-probe version of TipPulse. It will blunt a … |
| `SetProbeBias` | builtins/optional_multiprobe | confirm | write | probe, bias_v | multiprobe, bias, optional-hardware | Set ONE probe's bias voltage, in VOLTS.  Each probe has its own bias. Setting probe 1's bias does not change probe 0's — and the plain SetBias skill acts on the main channel, which is one particular probe, not the one… |
| `SetProbeZController` | builtins/optional_multiprobe | confirm | write | probe, on, setpoint, p_gain, i_gain | multiprobe, zcontroller, optional-hardware | Set ONE probe's Z-controller: switch it on/off, and set its setpoint and gains.  CAUTION. Switching a probe's Z loop ON with a setpoint it cannot reach drives that probe INTO THE SAMPLE. Switching it OFF parks the pro… |
| `StopProbeScanner` | builtins/optional_multiprobe | auto | write | probe | multiprobe, scanner, stop, optional-hardware | Stop ONE probe's scanner motion immediately. Always allowed, including after an abort. Stopping halts the move — it does not retract the probe; use WithdrawProbe for that. |
| `WithdrawProbe` | builtins/optional_multiprobe | auto | write | probe | multiprobe, withdraw, safe, optional-hardware | Retract ONE probe: switch its Z-controller off and drive its Z to the safe (fully withdrawn) end.  This is the SAFE state for a probe, and it is always allowed — including after an abort. If you are unsure about a pro… |
| `ConfigureDualScope` | builtins/optional_scopes | auto | write | channel_a, channel_b, timebase_index, trigger_mode, trigger_channel, trigger_slope, trigger_level, trigger_hysteresis, trigger_position | oscilloscope, osci_2t, optional-hardware | Configure the 2-channel oscilloscope: the two signals, the timebase, and the trigger. Use this when you need to see two signals against each other on ONE timebase (a pump-probe pair, current vs. bias, a lock-in X/Y). … |
| `ConfigureHighResScope` | builtins/optional_scopes | auto | write | signal_index, samples, oversampling_index, trigger_mode, trigger_channel, trigger_level, trigger_slope, trigger_hysteresis, osci_index | oscilloscope, osci_hr, optional-hardware | Configure the high-resolution oscilloscope (OsciHR): which signal, how many samples, oversampling, and the trigger.  Configuring a scope does not touch the instrument — it only decides what gets DIGITISED. Nothing mov… |
| `ConfigureSignalChart` | builtins/optional_scopes | auto | write | channel_a, channel_b | signal_chart, display, optional-hardware | Open the Signal Chart and set which two signals it displays. This is a DISPLAY module on the Nanonis machine — it changes what the operator sees there, and nothing about the measurement. |
| `GetDualScopeData` | builtins/optional_scopes | auto | read | run_first, data_to_get | oscilloscope, osci_2t, read, optional-hardware | Run the 2-channel oscilloscope and read both traces back. Call ConfigureDualScope first to pick the signals and the timebase.  Returns channel A and channel B on a common time axis. |
| `GetHighResScopeData` | builtins/optional_scopes | auto | read | wait_for_trigger, timeout_s, include_psd, osci_index | oscilloscope, osci_hr, read, optional-hardware | Read the high-resolution oscilloscope's captured trace, and optionally its power spectral density.  wait_for_trigger=true BLOCKS on the scope until the next trigger fires or timeout_s elapses — use it after RunHighRes… |
| `GetHighResScopeStatus` | builtins/optional_scopes | auto | read | osci_index | oscilloscope, osci_hr, read, optional-hardware | Read back the high-resolution oscilloscope's current configuration: channel, sample count, oversampling and trigger mode. Use before trusting a capture that someone else (or the Nanonis GUI) set up. |
| `RunHighResScope` | builtins/optional_scopes | auto | write | rearm | oscilloscope, osci_hr, optional-hardware | Start the high-resolution oscilloscope (and re-arm its trigger). Call GetHighResScopeData afterwards to collect the trace.  Read-only with respect to the instrument: running a scope digitises a signal, it does not dri… |
| `ConfigureHighSpeedSweep` | builtins/optional_sweepers | confirm | write | sweep_signal_index, start, stop, relative_limits, points, acquire_channels, settling_time_s, integration_time_s, initial_settling_time_s, max_slew_time_s, backward_sweep, num_sweeps, z_controller_off, z_offset_m | sweep, hs_sweeper, optional-hardware | Configure the High-Speed Sweeper: which signal to sweep, over what range, how many points, the timing, and which channels to record.  **Read this before using it.** The sweep signal is chosen by INDEX from the sweepab… |
| `ConfigureRfGenerator` | builtins/optional_sweepers | confirm | write | frequency_hz, power_dbm | rf, aprf_gen, optional-hardware | Set the RF generator's frequency and power WITHOUT switching the output on. Use this to stage a setting, check it, and only then call StartRfGenerator.  power_dbm is in dBm, a LOG scale: +10 dBm is ten times the power… |
| `GetHighSpeedSweepStatus` | builtins/optional_sweepers | auto | read | - | sweep, hs_sweeper, read, optional-hardware | Read the High-Speed Sweeper's state: whether a sweep is running, the current sweep signal and limits, and the LIST OF SWEEPABLE SIGNALS.  Call this BEFORE ConfigureHighSpeedSweep — the sweep signal is chosen by index … |
| `GetRfGeneratorStatus` | builtins/optional_sweepers | auto | read | - | rf, aprf_gen, read, optional-hardware | Read the RF generator's frequency, power, and — the one that matters — whether the OUTPUT IS ON. Check this before assuming the RF is off. |
| `RunHighSpeedSweep` | builtins/optional_sweepers | confirm | write | wait, timeout_s | sweep, hs_sweeper, optional-hardware | Execute the sweep configured by ConfigureHighSpeedSweep. This DRIVES the signal you named — bias, a piezo, an output — from start to stop, fast.  CAUTION: the sweeper is signal-agnostic. It will happily drive a piezo … |
| `RunRfFrequencySweep` | builtins/optional_sweepers | dangerous | write | lower_hz, upper_hz, points, dwell_s, repetitions, direction, auto_off | rf, aprf_gen, sweep, dangerous, optional-hardware | Sweep the RF frequency from one limit to the other at the current power, dwelling at each point.  DANGEROUS for the same reason as StartRfGenerator — this puts RF out for the whole sweep. Set the power with ConfigureR… |
| `StartRfGenerator` | builtins/optional_sweepers | dangerous | write | - | rf, aprf_gen, dangerous, optional-hardware | Switch the RF output ON at the currently configured frequency and power.  DANGEROUS: dBm into a tunnel junction is ENERGY. RF power couples into the tip and the sample, and enough of it modifies or destroys both. Call… |
| `StopHighSpeedSweep` | builtins/optional_sweepers | auto | write | - | sweep, hs_sweeper, stop, optional-hardware | Stop a running high-speed sweep immediately. Always allowed — including after an abort, which is the whole point of a stop. |
| `StopRfGenerator` | builtins/optional_sweepers | auto | write | - | rf, aprf_gen, stop, optional-hardware | Stop any running RF sweep AND switch the RF output off. Always allowed, including after an abort. Both, in that order — stopping the sweep without cutting the output leaves RF on at whatever value the sweep had reached. |
| `GetPatternCloud` | builtins/pattern | auto | read | - | pattern, cloud, read | Read the cloud of XY points configured for cloud-pattern spectroscopy. |
| `GetPatternProps` | builtins/pattern | auto | read | - | pattern, config, read | Read grid experiment configuration: available experiments, selected experiment, external VI, pre-measure delay, save channels. |
| `OpenPatternExperiment` | builtins/pattern | confirm | write | - | pattern, experiment, open, write | Open the selected grid experiment. Required before configuring or starting the experiment. |
| `PausePatternExperiment` | builtins/pattern | confirm | write | pause | pattern, experiment, pause, write | Pause or resume the currently running grid experiment. |
| `SetPatternCloud` | builtins/pattern | confirm | write | set_active, x_coords, y_coords | pattern, cloud, write | Set a cloud of XY points for cloud-pattern spectroscopy. |
| `SetPatternLine` | builtins/pattern | confirm | write | set_active, num_points, use_scan_frame, p1_x_m, p1_y_m, p2_x_m, p2_y_m | pattern, line, write | Set line pattern parameters: number of points and two endpoints. |
| `GetDriftCompensation` | builtins/piezo | auto | read | - | piezo, drift, read | Read piezo drift compensation status and velocities. |
| `GetPiezoHVAInfo` | builtins/piezo | auto | read | - | piezo, hva, gain, read | Read HVA gain readout information for AUX, X, Y, Z axes and their enabled status. |
| `GetPiezoHVAStatusLED` | builtins/piezo | auto | read | - | piezo, hva, status, led, read | Read HVA LED status: overheated, HV supply, high temperature, output connector. |
| `GetPiezoSensitivity` | builtins/piezo | auto | read | - | piezo, sensitivity, calibration, read | Read piezo sensitivity (m/V) for all 3 axes. |
| `GetPiezoTilt` | builtins/piezo | auto | read | - | piezo, tilt, read | Read tilt correction angles for piezo X and Y axes. |
| `GetPiezoXYZLimits` | builtins/piezo | auto | read | - | piezo, limits, voltage, read | Read XYZ voltage limits and enabled status from Piezo Calibration. |
| `LoadPiezoHysteresisFile` | builtins/piezo | confirm | write | file_path | piezo, hysteresis, file, write | Load and apply hysteresis compensation values for both axes from a .csv file in the Piezo Configuration module. |
| `SetDriftCompensation` | builtins/piezo | confirm | write | enable, vx, vy, vz | piezo, drift, compensation | Enable or disable piezo drift compensation. |
| `SetPiezoHysteresisOnOff` | builtins/piezo | confirm | write | enable | piezo, hysteresis, write | Enable or disable hysteresis compensation in Piezo Configuration. |
| `SetPiezoHysteresisValues` | builtins/piezo | confirm | write | fast_x, fast_y, slow_x, slow_y | piezo, hysteresis, calibration, write | Set and apply hysteresis compensation points for fast and slow axes in the Piezo Calibration module. |
| `SetPiezoRange` | builtins/piezo | confirm | write | range_x_m, range_y_m, range_z_m | piezo, range, calibration, write | Set piezo range (m) for all 3 axes. Changing range also changes sensitivity (HV gain unchanged). |
| `SetPiezoSensitivity` | builtins/piezo | confirm | write | sens_x, sens_y, sens_z | piezo, sensitivity, calibration, write | Set piezo sensitivity (m/V) for all 3 axes. Changing sensitivity also changes range (HV gain unchanged). |
| `SetPiezoTilt` | builtins/piezo | confirm | write | tilt_x_deg, tilt_y_deg | piezo, tilt, write | Set tilt correction angles for piezo X and Y axes. |
| `GetPLLAddOnOff` | builtins/pll | auto | read | modulator_index | pll, add, read | Return whether the Add external signal to output is on or off. |
| `GetPLLAmpCtrlOnOff` | builtins/pll | auto | read | modulator_index | pll, amplitude, onoff, read | Return whether the amplitude controller is on or off. |
| `GetPLLDemodFilter` | builtins/pll | auto | read | demodulator_index | pll, demod, filter, read | Return the filter order of the low-pass filter after the PLL lock-in. |
| `GetPLLDemodHarmonic` | builtins/pll | auto | read | demodulator_index | pll, demod, harmonic, read | Return the harmonic selected in the PLL lock-in demodulator. |
| `GetPLLDemodInput` | builtins/pll | auto | read | demodulator_index | pll, demod, input, read | Return the input and frequency generator of the selected demodulator. |
| `GetPLLExcRange` | builtins/pll | auto | read | modulator_index | pll, excitation, range, read | Return the excitation output range index (0=10V, 1=1V, 2=0.1V, 3=0.01V, 4=0.001V). |
| `GetPLLFreqRange` | builtins/pll | auto | read | modulator_index | pll, frequency, range, read | Return the frequency range of the oscillation control module. |
| `GetPLLFreqSwpParams` | builtins/pll | auto | read | modulator_index | pll, sweep, params, read | Return frequency sweep parameters: number of points, period, settling time. |
| `GetPLLInpCalibr` | builtins/pll | auto | read | modulator_index | pll, input, calibration, read | Return the input calibration (m/V) of the oscillation control module. |
| `GetPLLInpProps` | builtins/pll | auto | read | modulator_index | pll, input, properties, read | Return the input properties (differential input, 1/10 divider) of the PLL. |
| `GetPLLPhasCtrlOnOff` | builtins/pll | auto | read | modulator_index | pll, phase, onoff, read | Return whether the phase controller is on or off. |
| `GetPLLSignalAnlzrCh` | builtins/pll | auto | read | - | pll, analyzer, channel, read | Return the current channel index of the PLL Signal Analyzer. |
| `GetPLLSignalAnlzrFFTProps` | builtins/pll | auto | read | - | pll, analyzer, fft, read | Return the FFT configuration: window function, averaging mode, weighting mode, and count. |
| `GetPLLSignalAnlzrTimebase` | builtins/pll | auto | read | - | pll, analyzer, timebase, read | Return the time base index and update rate of the PLL Signal Analyzer. |
| `PLLFreqShiftAutoCenter` | builtins/pll | confirm | write | modulator_index | pll, frequency, autocenter, write | Auto-center frequency shift: adds the current frequency shift to the center frequency and resets the frequency shift to zero. |
| `PLLPerfectPLLUpdtZTC` | builtins/pll | confirm | write | modulator_index | pll, perfectpll, ztc, write | Update the Z-Controller time constant using the PerfectPLL algorithm. |
| `PLLSignalAnlzrTrigAuto` | builtins/pll | confirm | write | - | pll, analyzer, trigger, write | Set the PLL Signal Analyzer trigger parameters to pre-defined values. |
| `SetPLLAmpCtrlBandwidth` | builtins/pll | confirm | write | modulator_index, bandwidth_hz | pll, amplitude, bandwidth, write | Set the amplitude controller bandwidth. Uses the current Q factor and amplitude-to-excitation ratio (from a previous frequency sweep). |
| `SetPLLAmpCtrlSetpnt` | builtins/pll | confirm | write | modulator_index, setpoint_m | pll, amplitude, setpoint, write | Set the amplitude controller setpoint in meters. |
| `SetPLLDemodFilter` | builtins/pll | confirm | write | demodulator_index, filter_order | pll, demod, filter, write | Set the filter order of the low-pass filter after the PLL lock-in. |
| `SetPLLDemodInput` | builtins/pll | confirm | write | demodulator_index, input, frequency_generator | pll, demod, input, write | Set the input and frequency generator of the selected demodulator. |
| `SetPLLDemodPhasRef` | builtins/pll | confirm | write | demodulator_index, phase_reference_deg | pll, demod, phase, write | Set the phase reference of the selected demodulator. |
| `SetPLLFreqExcOverwrite` | builtins/pll | confirm | write | modulator_index, excitation_overwrite_index, frequency_overwrite_index | pll, overwrite, write | Set signals to overwrite the Frequency Shift and/or Excitation. Works when the corresponding controller is not active. Use -2 for no change. |
| `SetPLLFreqRange` | builtins/pll | confirm | write | modulator_index, frequency_range_hz | pll, frequency, range, write | Set the frequency range of the oscillation control module. |
| `SetPLLInpCalibr` | builtins/pll | confirm | write | modulator_index, calibration_m_per_v | pll, input, calibration, write | Set the input calibration (m/V) of the oscillation control module. |
| `SetPLLInpProps` | builtins/pll | confirm | write | modulator_index, differential_input, divider_1_10 | pll, input, properties, write | Set the input properties (differential input, 1/10 divider) of the PLL. |
| `SetPLLInpRange` | builtins/pll | confirm | write | modulator_index, input_range_m | pll, input, range, write | Set the input range (m) of the oscillation control module. |
| `SetPLLPhasCtrlBandwidth` | builtins/pll | confirm | write | modulator_index, bandwidth_hz | pll, phase, bandwidth, write | Set the phase controller bandwidth. Uses the current Q factor (from a previous frequency sweep). |
| `SetPLLSignalAnlzrTrig` | builtins/pll | confirm | write | trigger_mode, trigger_source, trigger_slope, trigger_level, trigger_position_s, arming_mode | pll, analyzer, trigger, write | Set the trigger configuration in the PLL Signal Analyzer. |
| `StopPLLFreqSwp` | builtins/pll | confirm | write | modulator_index | pll, sweep, stop, write | Stop the sweep in the PLL Frequency Sweep module. |
| `CheckTipCrashByAmplitude` | builtins/qplus_amplitude | auto | read | signal_index | qplus, afm, crash, tip, safety | 用 qPlus 振幅判断针尖是否已接触表面（撞针）。判据：振幅低于自由振荡基线的 10%。**这是对电流类判据的补充，不替代它们。**status ∈ ok\|crash\|unavailable\|no_baseline —— 后两者是「判断不了」，不是「没撞」。 |
| `ReadTipOscillationAmplitude` | builtins/qplus_amplitude | auto | read | signal_index, set_baseline | qplus, afm, read, tip | 读取 qPlus 振荡振幅（OCD1 Amplitude 等通道）。可作为**独立于电流**的针尖状态判据：针尖接触表面后振幅会被阻尼到接近 0，退针后才恢复。没有 qPlus 的机器上返回 status='unavailable'（这不是故障）。 |
| `GetMiscInstrumentConfig` | builtins/readback | auto | read | - | read, readback, verify, calibration | Read back the settings that do not belong to a bigger subsystem: the BIAS RANGE, the current preamp's calibration and available gains, the atom-tracker's parameters, the coarse motor's frequency/amplitude, the bias- a… |
| `GetPiezoConfig` | builtins/readback | auto | read | - | piezo, read, readback, verify | Read the piezo back: its RANGE, calibration, sensitivity, the hysteresis correction (on/off and its coefficients), the drift compensation, the tilt, and the XYZ VOLTAGE LIMITS.  The range and the limits are what turn … |
| `GetPllConfig` | builtins/readback | auto | read | modulator, demodulator | pll, read, readback, verify | Read the PLL back: **whether the excitation output is ON**, the amplitude-controller setpoint and bandwidth, the phase-controller bandwidth, the demodulator's phase reference and harmonic, the input range, and the fre… |
| `GetScanPatternConfig` | builtins/readback | auto | read | - | pattern, grid, read, readback, verify | Read the pattern module back: the grid definition, the line definition, and the pattern experiment's properties (which experiment, basename, pre-measure delay).  A grid experiment that runs on the wrong grid is hours … |
| `GetSpectroscopyConfig` | builtins/readback | auto | read | which | spectroscopy, sts, read, readback, verify | Read back the spectroscopy configuration — bias spectroscopy, Z spectroscopy, or both: the sweep properties, the advanced properties (Z-controller hold, final Z, record-final-Z), the multi-line-segment (MLS) mode and … |
| `GetTipShaperConfig` | builtins/readback | auto | read | - | tip, tipshaper, read, readback, verify | Read the tip-shaper configuration back: the switch-off delay, whether the bias is changed, the bias and lift used, the lift/return speeds, the wait times, and whether the feedback is restored afterwards.  **Use `props… |
| `GetZControllerState` | builtins/readback | auto | read | - | z, controller, read, readback, verify | Read the ENTIRE Z-controller state in one call: whether the loop is closed, the setpoint, gains, Z position, the Z limits and whether they are enabled, the tip lift, the withdraw rate, the switch-off delay and the hom… |
| `EnableSafeTip` | builtins/safety_hw | auto | write | enable | safety, hardware | Enable or disable Nanonis hardware SafeTip protection. |
| `GetSafeTipProps` | builtins/safety_hw | auto | read | - | safety, hardware, props, read | Get SafeTip configuration: auto recovery, auto pause scan, threshold. |
| `GetSafeTipSignal` | builtins/safety_hw | auto | read | - | safety, hardware, signal, read | Get the current SafeTip signal value. |
| `GetSafeTipStatus` | builtins/safety_hw | auto | read | - | safety, hardware, read | Read current SafeTip protection status. |
| `GetScanBuffer` | builtins/scan_extra | auto | read | - | scan, buffer, read | Read the current scan buffer (channels, pixels, lines). |
| `GetScanSpeed` | builtins/scan_extra | auto | read | - | scan, speed, read | Read the current scan speed parameters. |
| `CheckScanForCrash` | builtins/scan_frame | auto | analysis | channels, direction | scan, crash, safety, analysis, g1 | Detect a tip crash from the just-acquired scan: grab the probe channels and flag a crash if any has near-zero variance or NaN. Returns crash_indicator + status (ok/crash/skipped) + per-channel verdicts. Channels = com… |
| `ComputeDriftVector` | builtins/scan_frame | auto | analysis | ref_path, scan_width_m, channel_index, direction | scan, drift, analysis, g1 | Grab the current scan frame and cross-correlate it against a reference .npy image to estimate sample drift, returned in METERS (drift_x_m, drift_y_m). Used by drift-tracking workflows. |
| `GrabScanFrameData` | builtins/scan_frame | auto | read | channel_index, direction, save_path | scan, frame, read, g1 | Grab one scan channel's frame samples (Scan_FrameDataGrab) and save them to a .npy file; returns the file PATH (not the array). direction=1 for the forward scan, 0 for backward. Channel 0 is the first acquired channel… |
| `LoadScanFrameFromFile` | builtins/scan_frame | auto | analysis | scan_path, channel, save_dir | scan, frame, read, offline, sxm | Extract one channel's FORWARD and BACKWARD 2-D frames from a SAVED .sxm file and write each to a .npy; returns both PATHS (never the arrays). This is the offline counterpart of GrabScanFrameData, which can only read t… |
| `ParseRegions` | builtins/scan_frame | auto | analysis | regions | scan, regions, analysis, g1 | Parse + validate a JSON array of scan regions (each {center_x_m, center_y_m, width_m, height_m, [angle_deg], [label]}) into a normalised list a composite can foreach over. Rejects bad JSON, >64 regions, out-of-range c… |
| `GetScanFrame` | builtins/scan_utils | auto | read | - | scan, frame, read | Read the current scan frame (center, size, angle). |
| `GetSignalRange` | builtins/signals | auto | read | signal_index | signals, range, read | Read the maximum and minimum range limits of a signal (0-127). |
| `GetSignalValues` | builtins/signals | auto | read | signal_indexes, wait_for_newest | signals, values, read | Read current values of selected signals (oversampled). Provide a list of signal indexes (0-127). |
| `GetSignalsAddRT` | builtins/signals | auto | read | - | signals, rt, read | Read the list of available additional RT signals and the names currently assigned to Internal 23 and Internal 24. |
| `ListSignalChannels` | builtins/signals | auto | read | - | signals, enumerate, channels, read | List the 128 available Nanonis signals (physical inputs / outputs / internal channels) with their 0-127 index. Flags which indices are current channels so a caller can pick the tunnelling-current signal for high-rate … |
| `ConfigureSTSChannels` | builtins/spectroscopy | confirm | write | channel_indexes | spectroscopy, sts, channels, write | Set which channels are recorded during Bias Spectroscopy. |
| `ConfigureSTSTiming` | builtins/spectroscopy | confirm | write | z_avg_time_s, z_offset_m, init_settling_s, max_slew_rate_v_s, settling_s, integration_s, end_settling_s, z_ctrl_time_s | spectroscopy, sts, timing, write | Configure Bias Spectroscopy timing parameters. |
| `ConfigureZSpectrTiming` | builtins/spectroscopy | confirm | write | z_avg_time_s, init_settling_s, max_slew_rate_v_s, settling_s, integration_s, end_settling_s, z_ctrl_time_s | spectroscopy, z, timing, write | Configure Z Spectroscopy timing parameters. |
| `GetSTSAltZCtrl` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, zctrl, read | Get Bias Spectroscopy alternative Z controller settings. |
| `GetSTSChannels` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, channels, read | Get the list of recorded channels for Bias Spectroscopy. |
| `GetSTSDigSync` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, digsync, read | Get the digital sync mode (Off/TTL/PulseSeq) for Bias Spectroscopy. |
| `GetSTSLimits` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, limits, read | Get the bias voltage range for Bias Spectroscopy. |
| `GetSTSMLSLockinPerSeg` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, mls, lockin, read | Get the Lock-In per Segment flag for MLS mode. |
| `GetSTSPulseSeqSync` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, pulseseq, read | Get pulse sequence sync configuration for Bias Spectroscopy. |
| `GetSTSSafeCond1` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, safecond, read | [DEPRECATED — Bias Spectroscopy has no safe-condition API in Nanonis; this always fails. Use Z-spectroscopy's GetZSpectrRetract instead.] Get the 1st safe condition for Bias Spectroscopy. |
| `GetSTSTTLSync` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, ttlsync, read | Get TTL sync configuration for Bias Spectroscopy. |
| `GetSTSTiming` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, timing, read | Get Bias Spectroscopy timing parameters. |
| `GetSTSZOffRevert` | builtins/spectroscopy | auto | read | - | spectroscopy, sts, zoffrevert, read | Get the Z Offset Revert flag for Bias Spectroscopy. |
| `GetZSpectrChannels` | builtins/spectroscopy | auto | read | - | spectroscopy, z, channels, read | Get the list of recorded channels for Z Spectroscopy. |
| `GetZSpectrDigSync` | builtins/spectroscopy | auto | read | - | spectroscopy, z, digsync, read | Get the digital sync mode (Off/TTL/PulseSeq) for Z Spectroscopy. |
| `GetZSpectrPulseSeqSync` | builtins/spectroscopy | auto | read | - | spectroscopy, z, pulseseq, read | Get pulse sequence sync configuration for Z Spectroscopy. |
| `GetZSpectrRange` | builtins/spectroscopy | auto | read | - | spectroscopy, z, range, read | Get Z Spectroscopy range settings. |
| `GetZSpectrRetract` | builtins/spectroscopy | auto | read | - | spectroscopy, z, retract, read | Get Z Spectroscopy auto-retract configuration. |
| `GetZSpectrRetract2nd` | builtins/spectroscopy | auto | read | - | spectroscopy, z, retract, read | Get the 2nd auto-retract condition for Z Spectroscopy. |
| `GetZSpectrTTLSync` | builtins/spectroscopy | auto | read | - | spectroscopy, z, ttlsync, read | Get TTL sync configuration for Z Spectroscopy. |
| `GetZSpectrTiming` | builtins/spectroscopy | auto | read | - | spectroscopy, z, timing, read | Get the timing parameters for Z Spectroscopy. |
| `SetSTSAdvancedProps` | builtins/spectroscopy | confirm | write | reset_bias, z_controller_hold, record_final_z, lockin_run | spectroscopy, sts, advanced, write | Set Bias Spectroscopy advanced properties. |
| `SetSTSChannels` | builtins/spectroscopy | confirm | write | channel_indexes | spectroscopy, sts, channels, write | Set the recorded channels for Bias Spectroscopy. |
| `SetSTSMLSMode` | builtins/spectroscopy | confirm | write | mode | spectroscopy, sts, mls, mode, write | Set Bias Spectroscopy sweep mode: Linear or MLS. |
| `SetSTSMLSVals` | builtins/spectroscopy | confirm | write | bias_start_v, bias_end_v, initial_settling_s, settling_s, integration_s, steps, lockin_run | spectroscopy, sts, mls, write | Set multi-line-segment configuration for Bias Spectroscopy MLS mode. |
| `SetSTSSafeCond1` | builtins/spectroscopy | confirm | write | condition, threshold, signal_index, comparison | spectroscopy, sts, safecond, write | [DEPRECATED — Bias Spectroscopy has no safe-condition API in Nanonis; this always fails. Use Z-spectroscopy's SetZSpectrRetract instead.] Set 1st safe condition for Bias Spectroscopy. |
| `SetSTSSafeCond2` | builtins/spectroscopy | confirm | write | condition, threshold, signal_index, comparison | spectroscopy, sts, safecond, write | [DEPRECATED — Bias Spectroscopy has no safe-condition API in Nanonis; this always fails. Use Z-spectroscopy's 2nd retract (GetZSpectrRetract2nd) instead.] Set 2nd safe condition for Bias Spectroscopy. |
| `SetZSpectrAdvProps` | builtins/spectroscopy | confirm | write | time_between_sweeps_s, record_final_z, lockin_run, reset_z | spectroscopy, z, advanced, write | Set advanced properties for Z Spectroscopy. |
| `SetZSpectrChannels` | builtins/spectroscopy | confirm | write | channel_indexes | spectroscopy, z, channels, write | Set the recorded channels for Z Spectroscopy. |
| `SetZSpectrRange` | builtins/spectroscopy | confirm | write | z_offset_m, z_sweep_distance_m | spectroscopy, z, range, write | Set Z Spectroscopy range. |
| `SetZSpectrRetract` | builtins/spectroscopy | confirm | write | enabled, threshold, signal_index, comparison | spectroscopy, z, retract, write | Set Z Spectroscopy auto-retract conditions. |
| `SetZSpectrRetractDelay` | builtins/spectroscopy | confirm | write | retract_delay_s | spectroscopy, z, retract, delay, write | Set the retract delay (s) between forward and backward sweep in Z Spectroscopy. |
| `StopSTS` | builtins/spectroscopy | auto | write | - | spectroscopy, sts, stop | Stop the current Bias Spectroscopy measurement. |
| `StopZSpectr` | builtins/spectroscopy | auto | write | - | spectroscopy, z, stop | Stop the current Z Spectroscopy measurement. |
| `GetSpectroscopyStatus` | builtins/spectroscopy_sync | auto | read | which | spectroscopy, status, read, poll | Ask whether a bias- or Z-spectroscopy is currently RUNNING.  Until now MAST could only run spectroscopy blocking — start it and wait. With this you can start it, then poll: watch the current, check the abort flag, rep… |
| `SetMlsLockinPerSegment` | builtins/spectroscopy_sync | auto | write | enable | spectroscopy, mls, lockin, write | Turn the lock-in on or off PER SEGMENT of a multi-line-segment (MLS) bias spectroscopy, instead of for the whole sweep.  MLS lets you sweep different bias ranges at different speeds in one curve. This switch lets the … |
| `SetSpectroscopyPulseSync` | builtins/spectroscopy_sync | confirm | write | which, digital_sync, pulse_sequence_nr, pulse_periods | spectroscopy, sync, pulse, pump-probe, write | Set spectroscopy's digital-line gating and/or its pulse-sequence synchronisation.  `digital_sync` gates the sweep on a digital line. `pulse_sequence_nr` runs one of the pulse generator's programmed sequences for `puls… |
| `SetSpectroscopyTtlSync` | builtins/spectroscopy_sync | confirm | write | which, line, polarity, time_to_on_s, on_duration_s | spectroscopy, sync, ttl, pump-probe, write | Make spectroscopy pulse a TTL line at each point — the hook a pump laser, a shutter or a spectrometer exposure hangs off.  TIMES ARE IN SECONDS. `time_to_on_s` is the delay from the point starting to the line going ac… |
| `SetSpectroscopyZControl` | builtins/spectroscopy_sync | confirm | write | use_alternate_setpoint, setpoint, settling_time_s, revert_z_offset | spectroscopy, sts, zcontroller, write | Configure how spectroscopy treats the Z-controller: the ALTERNATE SETPOINT it moves to before the sweep, and whether the Z offset is reverted afterwards.  The alternate setpoint is how you take a spectrum at a differe… |
| `SetZSpectroscopySecondRetract` | builtins/spectroscopy_sync | confirm | write | enable, signal_index, threshold, comparison | spectroscopy, zspectr, retract, safety, write | Add a SECOND retract condition to Z spectroscopy: abort the sweep and pull back when a chosen signal crosses a threshold.  This is a safety feature and it was read-only until now. A Z sweep drives the tip toward the s… |
| `ConfigureSpectrumAnalyzer` | builtins/spectrum_analyzer | auto | write | instance, fft_window, averaging_count, averaging_mode, weighting_mode, ac_coupling | spectrum, noise, fft, diagnostics | Configure the spectrum analyser: the FFT window, the averaging, and the AC coupling. Reads only — an analyser digitises a signal, it drives nothing.  These three decide whether the spectrum is worth looking at: • **av… |
| `GetSpectrumAnalyzerData` | builtins/spectrum_analyzer | auto | read | instance | spectrum, noise, read, diagnostics | Read the spectrum analyser: the spectrum itself, the BAND RMS (the noise in the band set by SetSpectrumAnalyzerBand — the one number worth quoting), the DC value, and the current settings so you can tell whether the s… |
| `SetSpectrumAnalyzerBand` | builtins/spectrum_analyzer | auto | write | f_low_hz, f_high_hz, instance | spectrum, noise, diagnostics | Set the frequency band (the cursor pair) over which the spectrum analyser reports its band RMS, in HERTZ.  The band RMS is the single number that answers 'how much noise is there, really' — e.g. 1 Hz to 1 kHz on the c… |
| `GenSwpAcqChsGet` | builtins/sweep | auto | read | - | sweep, generic, channels, read | Get the list of recorded acquisition channels for the Generic Sweeper. |
| `GenSwpPropsGet` | builtins/sweep | auto | read | - | sweep, generic, props, read | Get the configuration properties of the Generic Sweeper. |
| `GenSwpStop` | builtins/sweep | confirm | write | - | sweep, generic, stop, write | Stop the currently running Generic Sweeper sweep. |
| `GenSwpSwpSignalGet` | builtins/sweep | auto | read | - | sweep, generic, signal, read | Get the sweep signal name for the Generic Sweeper. |
| `GetLockInSweepLimits` | builtins/sweep | auto | read | - | sweep, lockin, frequency, limits, read | Read the lower and upper frequency limits of the lock-in sweep. |
| `GetLockInSweepProps` | builtins/sweep | auto | read | - | sweep, lockin, frequency, props, read | Read lock-in frequency sweep properties (steps, integration, settling). |
| `GetLockInSweepSignal` | builtins/sweep | auto | read | - | sweep, lockin, frequency, signal, read | Read the sweep signal index for the lock-in frequency sweep. |
| `SafeRetract` | builtins/tip | auto | write | - | tip, safety, retract | Safely retract the tip by withdrawing. |
| `ConfigureCalculatedOutput` | builtins/user_output | confirm | write | output_index, signal_1, operation, signal_2, name | output, user_output, calc_signal, write | Make a user output carry the RESULT of an arithmetic operation on two signals (Calc.Signal mode): e.g. output (Signal_A − Signal_B) as a live analog line. Useful for a difference channel, a normalised signal, or a der… |
| `ConfigureDigitalLine` | builtins/user_output | confirm | write | line, port, direction, polarity | output, digital, ttl, write | Configure a digital line's direction (input/output) and polarity (active high/low). Flipping a line from input to output starts DRIVING whatever is on the other end — check the wiring first. |
| `GetCalculatedOutputConfig` | builtins/user_output | auto | read | output_index | output, user_output, calc_signal, read | Read which two signals a user output combines, with what operation, and under what name (Calc.Signal mode). |
| `GetDigitalLineTTL` | builtins/user_output | auto | read | port | output, digital, ttl, read | Read the TTL values of all 8 lines on a digital port. |
| `GetUserOutputLimits` | builtins/user_output | auto | read | output_index, raw | output, user_output, read, safety | Read the physical upper/lower limits of a user output channel. These limits are the SAFETY ENVELOPE for SetUserOutput — they are configured by the operator in Nanonis and cannot be changed from here. Read them before … |
| `GetUserOutputMode` | builtins/user_output | auto | read | output_index | output, user_output, read | Read a user output's mode: 0=User Output (you drive it), 1=Monitor (it mirrors a signal), 2=Calc.Signal. SetUserOutput only does anything in mode 0 — in Monitor mode the channel is driven by the instrument, not by you. |
| `GetUserOutputMonitorChannel` | builtins/user_output | auto | read | output_index | output, user_output, read | Read the monitor channel index of a user output (Monitor mode). |
| `PulseDigitalLine` | builtins/user_output | confirm | write | port, lines, pulse_width_s, pulse_pause_s, n_pulses, wait_until_finished | output, digital, ttl, trigger, write | Fire a TTL pulse train on one or more digital output lines. This is how Nanonis triggers EXTERNAL hardware — a camera exposure, a pulse generator, a chopper, a shutter. MAST does not know what is wired to the line: ch… |
| `SetDigitalLineStatus` | builtins/user_output | confirm | write | port, line, status | output, digital, ttl, write | Set one digital output line HIGH or LOW and hold it there (unlike PulseDigitalLine, which returns it). Use for a latching enable — a shutter held open, an amplifier held on. MAST does not know what is wired to the lin… |
| `SetUserOutput` | builtins/user_output | confirm | write | output_index, value | output, user_output, write | Set a user output channel to a value, in that channel's CALIBRATED PHYSICAL UNITS (NOT necessarily volts — a channel may be µm, mW, or anything else, depending on its Nanonis calibration).  This drives EXTERNAL hardwa… |
| `SetUserOutputCalibration` | builtins/user_output | confirm | write | output_index, calibration_per_volt, offset | output, user_output, write, calibration | Set the calibration (units per volt + offset) of a user output or monitor channel. This redefines what one physical unit MEANS on that channel: after a calibration change, the same numeric value drives a DIFFERENT vol… |
| `SetUserOutputLimits` | builtins/user_output | confirm | write | output_index, upper_limit, lower_limit, raw | output, user_output, write, safety | Set the physical upper/lower limits of a user output channel. These limits are the envelope SetUserOutput checks against, so widening them widens what the agent may drive — it is a guardrail change, it needs operator … |
| `SetUserOutputMode` | builtins/user_output | confirm | write | output_index, mode | output, user_output, write | Set a user output's mode: 0=User Output (you drive it), 1=Monitor (it mirrors an instrument signal), 2=Calc.Signal. Switching a channel INTO mode 0 hands control of external hardware to the agent; switching OUT of it … |
| `SetUserOutputMonitorChannel` | builtins/user_output | confirm | write | output_index, monitor_channel_index | output, user_output, write | Set which signal a user output mirrors (only meaningful in Monitor mode). Use ListSignalNames / the signals catalogue to find the channel index. |
| `GetAcqPeriod` | builtins/util | auto | read | - | util, acquisition, read | Get the acquisition period (s) in the TCP Receiver. |
| `GetRTFreq` | builtins/util | auto | read | - | util, rt, frequency, read | Get the Real Time controller frequency in Hz. |
| `GetRTOversample` | builtins/util | auto | read | - | util, rt, oversampling, read | Get the Real-time oversampling value in the TCP Receiver. |
| `GetSessionPath` | builtins/util | auto | read | - | util, session, path, read | Get the current Nanonis session folder path. |
| `LoadLayout` | builtins/util | confirm | write | file_path, use_session | util, layout, write | Load a Nanonis layout from an .ini file. |
| `LockNanonisUI` | builtins/util | dangerous | write | - | util, lock, dangerous, write | Lock the Nanonis software — this puts a MODAL WINDOW over it and PREVENTS THE OPERATOR FROM INTERACTING WITH THE INSTRUMENT until it is unlocked.  DANGEROUS, and not because of anything it does to the hardware. Every … |
| `SaveLayout` | builtins/util | confirm | write | file_path, use_session | util, layout, write | Save the current Nanonis layout to an .ini file. |
| `SaveSettings` | builtins/util | confirm | write | action, file_path, use_session | util, settings, write | Save or load Nanonis settings to/from an .ini file. |
| `SetRTFreq` | builtins/util | confirm | write | frequency_hz | util, rt, frequency, write | Set the Real Time controller frequency in Hz. |
| `SetRTOversample` | builtins/util | confirm | write | oversampling | util, rt, oversampling, write | Set the Real-time oversampling value in the TCP Receiver. |
| `SetSessionPath` | builtins/util | confirm | write | session_path, save_settings_to_previous | util, session, path, write | Set the Nanonis session folder path. |
| `UnlockNanonisUI` | builtins/util | auto | write | - | util, unlock, write | Unlock the Nanonis UI (close the Lock modal window). |
| `GetHomeProps` | builtins/zcontrol | auto | read | - | z, home, read | Get the Z controller Home position mode and value. |
| `GetSetpoint` | builtins/zcontrol | auto | read | - | z, setpoint, read | Read the current tunneling-current setpoint (Z controller). |
| `GetTipLift` | builtins/zcontrol | auto | read | - | z, tip_lift, read | Read the current tip lift amount. |
| `GetWithdrawRate` | builtins/zcontrol | auto | read | - | z, withdraw, rate, read | Get the Z controller withdraw slew rate in m/s. |
| `GetZCtrlList` | builtins/zcontrol | auto | read | - | z, controller, list, read | Get the list of Z controllers and the active controller index. |
| `GetZLimitsEnabled` | builtins/zcontrol | auto | read | - | z, limits, safety, read | Read whether Z position safety limits are enabled. |
| `GetZPosition` | builtins/zcontrol | auto | read | - | z, position, read | Read the current Z piezo position. |
| `SetHomeProps` | builtins/zcontrol | confirm | write | rel_or_abs, home_position_m | z, home, write | Set the Z controller Home position mode and value. |
| `SetSetpoint` | builtins/zcontrol | confirm | write | setpoint_a | z, setpoint, write, readback | Set the tunneling current setpoint for the Z controller. |
| `SetSwitchOffDelay` | builtins/zcontrol | confirm | write | delay_s | z, switchoff, delay, write | Set the Z controller switch-off delay in seconds. |
| `SetTipLift` | builtins/zcontrol | confirm | write | tip_lift_m | z, tip_lift, write | Set the amount the tip retracts when Z controller is turned off. |
| `SetZLimitsEnabled` | builtins/zcontrol | confirm | write | enabled | z, limits, safety, write | Enable or disable Z position safety limits. |
| `SetZPosition` | builtins/zcontrol | confirm | write | z_pos_m | z, position, write | Set the Z piezo position directly. Z controller must be OFF. |
| `ZControllerOnOff` | builtins/zcontrol | confirm | write | enable | z, controller, write | Enable or disable the Z controller (feedback loop). |
| `GetZCtrlGain` | builtins/zctrl_gain | auto | read | - | z, gain, read | Read the current Z controller P/I gains and time constant. P 的单位是米(m),I 的单位是米每秒(m/s),T 是秒。 |
| `SetZCtrlGain` | builtins/zctrl_gain | confirm | write | p_gain, time_constant_s, i_gain | z, gain, write, readback | Set the Z controller P/I gains and time constant.  **优先使用 ApplyZCtrlPreset**(按参数组名应用,数值由代码从操作员维护的存储里取)。只有当操作员在本次对话里逐字念出了具体数值时,才直接调用本技能传裸数值。  三个参数都是有量纲的物理量,且都是极小的数 —— **写成带 SI 前缀的字符串**(如 '3p'、'16.667u'、'180n')。⚠️ 前缀不可省… |
| `CreateZCtrlPreset` | builtins/zctrl_presets_skills | dangerous | write | name, p_gain, i_gain, setpoint_a, note, overwrite | z, gain, preset, config | 新建(或覆盖)一个自定义 Z 参数组,之后可以用 ApplyZCtrlPreset 按名应用。  **数值必须写成带 SI 前缀的字符串**,例如 p_gain='3p'、i_gain='180n'、setpoint_a='150p' —— 与 Nanonis 面板上的写法一致。**前缀不可省略**:裸数字(如 '3')会被直接拒绝。原因是量级一旦丢失,裸数字仍然是一个合法的数,错一万亿倍也没人发现;而前缀掉了就解析失败,你会立刻… |
| `ListZCtrlPresets` | builtins/zctrl_presets_skills | auto | read | - | z, gain, preset, read | 列出当前可用的全部 Z 参数组名及其数值与来源。在调用 ApplyZCtrlPreset 之前不确定有哪些组时使用。 |

## L1 — Short bounded sequence — a few TCP calls or a poll loop.

| Name | Folder | Safety | Category | Params | Tags | Description |
|---|---|---|---|---|---|---|
| `AcquireOsciTrace` | builtins/acquire_osci_trace | auto | read | data_to_get, signal_index | oscilloscope, trace, hardware, read | Acquire one buffered time-series from the Nanonis 1-channel Oscilloscope (Osci1T). Samples at the hardware RT rate (~20 kHz on V5e). Returns (t0, dt, y_array). Requires the Osci1T module loaded in Nanonis — not availa… |
| `AcquirePSD` | builtins/acquire_psd | auto | read | instance, signal_index, freq_range_index, freq_range_indices, freq_resolution_index | spectrum, psd, fft, hardware, read, composite | Read Power Spectral Density(s) from the Nanonis-side Spectrum Analyzer (hardware FFT). Returns (f0, df, psd_array) per requested frequency-range. Latency: ~0.5 ms RTT per range. Use when you need a refresh rate beyond… |
| `AutoApproach` | builtins/approach | auto | write | wait_timeout_s | approach, tip | Start the auto approach procedure. Moves tip toward the surface. |
| `ConfigureAtomTrack` | builtins/atom_track | confirm | write | integral_gain, frequency_hz, amplitude_m, phase_deg, switch_off_delay_s, enable_modulation, enable_controller | atomtrack, tracking, write | Configure Atom Tracking parameters and enable/disable controls. |
| `SetBiasRamp` | builtins/bias | confirm | write | bias_v_end, bias_v_start, slew_rate_v_per_s, step_interval_s | bias, ramp, write, composite | Ramp the bias voltage from a starting value to a target value in small steps (slew rate limited). Use this for large bias changes to protect the sample / tip. |
| `BiasPulseWithReadback` | builtins/bias_pulse_readback | auto | write | bias_v, width_s, z_hold, absolute, poll_hz, pre_roll_s, post_roll_s, max_capture_s, jump_k, step_tol_nm, step_tol_k | bias, pulse, readback, z, tip, write | Fire one bias pulse while streaming Z and current, then report how far and which way Z stepped between its settled value before the pulse and after it (mid-pulse transients ignored). Use this instead of BiasPulse when… |
| `BiasWiggle` | builtins/bias_wiggle | auto | write | base_bias_v, wiggle_lower_v, wiggle_upper_v, dwell_min_s, dwell_max_s, slew_rate_v_per_s, burst_s, abort_current_a, seed, allow_feedback_off | bias, wiggle, tip, write | Randomly hop the bias inside a small bipolar window (default ±4..20 mV) for one short burst, to nudge the tip apex into an atomically sharp configuration — the software version of dragging the bias slider back and for… |
| `CaptureSignalBuffer` | builtins/capture_signal_buffer | auto | read | channel, duration_s, poll_hz, include_samples | signal, capture, stream, read | Capture a time-series of one Nanonis signal at high rate. Use channel='current'/'z'/'bias' for sub-ms RTT, or a signal index string ('0'..'127') for the generic Signals_ValGet route (limited to ~50 Hz by TCP Tap). |
| `SelectPokedCluster` | builtins/cluster_select | auto | analysis | scan_path, min_aspect, min_area_px, min_peak_height_m, anchor_tolerance_m, near_x_m, near_y_m, channel, polarity, level | scan, analysis, cluster, select, read | Pick THE cluster we just poked out of a frame that may contain many. Runs ExtractClusters, applies a conjunction (round AND large AND tall), then takes the one nearest the poke coordinate. ABSTAINS (selected=null) whe… |
| `MonitorCurrent` | builtins/current_monitor | auto | read | duration_s, poll_hz, contact_threshold_a, min_contact_samples | current, monitor, contact, read | Sample tunneling current at high rate for a fixed window. Returns min/max/mean/std plus a `contact_detected` flag set when \|I\| stays above contact_threshold_a for at least min_contact_samples consecutive polls. |
| `AssessHerringbone` | builtins/herringbone_assess | auto | analysis | scan_path, channel, substrate, period_prior_nm, snr_min, concentration_min, corrugation_min_pm, allow_reduced_scale | tip, herringbone, reconstruction, au111, fft, analysis, read | Decide whether a saved .sxm frame shows the Au(111) 22x-sqrt3 HERRINGBONE reconstruction, and report the tip-quality numbers that go with it (stripe corrugation in pm, FFT sharpness, forward/backward instability, doub… |
| `ConfigureScan` | builtins/imaging | confirm | write | center_x_m, center_y_m, width_m, height_m, angle_deg, channels, set_scan_speed, line_time_s | scan, imaging, write | Configure scan frame: center, size, angle, and acquisition channels. By default also sets the scan speed derived from line_time_s (linear speed = width_m / line_time_s); pass set_scan_speed=False to leave the current … |
| `StartScan` | builtins/imaging | confirm | write | - | scan, imaging, write | Start a scan with current parameters. |
| `ConfigureLockIn` | builtins/lockin | confirm | write | mod_on, amplitude_v, frequency_hz, phase_deg | lockin, modulation, write | Configure lock-in amplifier modulation on/off and parameters. |
| `ConfigureLockInDemod` | builtins/lockin | confirm | write | demodulator, signal_index, harmonic, lp_order, lp_cutoff_hz, hp_order, hp_cutoff_hz, phase_deg | lockin, demodulator, write | Configure lock-in demodulator: signal, harmonic, filters, phase. |
| `GetLockInConfig` | builtins/lockin | auto | read | modulator, demodulator | lockin, config, read, readback, verify | Read the lock-in back in full: on/off, amplitude, frequency, phase, and — the part that used to be write-only — WHICH SIGNAL it modulates, the harmonic, the modulator's and demodulator's phase registers, the demodulat… |
| `ApplyLockInPreset` | builtins/lockin_presets_skills | confirm | write | preset, mod_on | lockin, preset, write, readback | 按**参数组名**设置 lock-in 调制(频率 / 幅度),数值由代码从操作员维护的仪器档案里取出,你不需要也不应该自己写任何数字。  组名:`didv`(dI/dV 常用组)。先用 ListLockInPresets 看里面有什么、哪些键还没配。  **不会下发调制侧相位** —— 本机 Modulate 区没有该字段,固件恒拒写(写同样的值也拒)。要调相位请用 AutoPhase 或 ConfigureLockInDemo… |
| `AutoPhase` | builtins/lockin_presets_skills | confirm | write | mode, window_s, demodulator, x_signal_index, y_signal_index | lockin, phase, auto, didv, write, readback | 自动对齐 lock-in **解调相位**(GUI 上的 Auto 按钮做的事,但那个按钮没有 TCP 命令,所以这里是读 X/Y → 算角 → 写相位)。  两种模式: - `signal_to_x`(默认,**隧穿态**用):把 dI/dV 信号转到 X 轴; - `crosstalk_to_y`(**退针态**用):把电容串扰转到 Y 轴,信号轴自然对齐 X —— 不用进针就能定相位轴。  取样窗口内平均后再算角;**X/Y… |
| `AcquirePLLFreqSweep` | builtins/pll | confirm | write | modulator_index, num_points, period_s, settling_time_s, sweep_up | pll, sweep, write | Run a PLL frequency sweep to find the resonance peak. |
| `ConfigurePLL` | builtins/pll | confirm | write | modulator_index, center_freq_hz, freq_shift_hz, amp_p_gain, amp_time_constant_s, phas_p_gain, phas_time_constant_s | pll, configure, write | Configure PLL center frequency, frequency shift, and controller gains. |
| `ConfigurePLLExcitation` | builtins/pll | confirm | write | modulator_index, excitation_v, output_range | pll, excitation, write | Set PLL excitation amplitude and output range. |
| `GetPLLStatus` | builtins/pll | auto | read | modulator_index | pll, status, read | Read current PLL status: frequency, gains, excitation. |
| `PLLOnOff` | builtins/pll | confirm | write | modulator_index, output_on, phase_ctrl_on, amp_ctrl_on | pll, onoff, write | Turn PLL output, phase controller, and amplitude controller on or off. |
| `PLLSignalAnalyzer` | builtins/pll | auto | read | channel_index, get_fft | pll, analyzer, read | Open the PLL signal analyzer and acquire oscilloscope/FFT data. |
| `SetScanBuffer` | builtins/scan_buffer | confirm | write | pixels, lines | scan, buffer, resolution, write | Set the scan resolution: pixels per line and number of lines. Preserves the currently selected acquisition channels (they are read back and re-sent unchanged). Prefer ScanAt, which picks the resolution for the request… |
| `SaveScan` | builtins/scan_extra | auto | write | timeout_ms | scan, save, write | Save the current scan data buffer to file. |
| `WaitScanComplete` | builtins/scan_utils | auto | read | timeout_ms | scan, wait, read | Wait for the end of the current scan or timeout. |
| `AcquireSTS` | builtins/spectroscopy | confirm | write | save_basename | spectroscopy, sts, write | Acquire a single STS spectrum at the current tip position. Uses current lock-in and bias sweep settings. Pass save_basename to control the saved .dat filename (traceability — e.g. per grid point). |
| `AcquireZSpectr` | builtins/spectroscopy | confirm | write | - | spectroscopy, z, write | Acquire a Z spectrum at the current tip position. |
| `ConfigureSTS` | builtins/spectroscopy | confirm | write | start_v, end_v, num_points, z_offset_m | spectroscopy, sts, configure, write | Configure STS bias sweep parameters. |
| `ConfigureZSpectr` | builtins/spectroscopy | confirm | write | z_offset_m, z_sweep_distance_m, num_points, backward_sweep | spectroscopy, z, configure, write | Configure Z spectroscopy: Z offset, sweep distance, and points. |
| `AcquireBiasSweep` | builtins/sweep | confirm | write | - | sweep, bias, write | Acquire a bias sweep at the current tip position. |
| `AcquireLockInSweep` | builtins/sweep | confirm | write | - | sweep, lockin, frequency, write | Acquire a lock-in frequency sweep. |
| `ConfigureBiasSweep` | builtins/sweep | confirm | write | lower_v, upper_v, num_steps, period_ms | sweep, bias, configure, write | Configure bias sweep limits, steps, and recording channels. |
| `ConfigureLockInSweep` | builtins/sweep | confirm | write | lower_hz, upper_hz, num_steps, integration_periods, settling_periods | sweep, lockin, frequency, configure, write | Configure lock-in frequency sweep limits and parameters. |
| `EmergencyRetract` | builtins/tip | auto | write | - | tip, safety, emergency, retract | Emergency tip retraction via dedicated emergency port. |
| `TipConditioningSelfCheck` | builtins/tip_conditioning_selfcheck | auto | read | - | tip, conditioning, selfcheck, read | Read-only pre-flight for the tip-conditioning workflow: are all the skills it needs actually registered in this build, is the Tip Shaper module running, is the tip registered (an unregistered tip caps pulses at the co… |
| `TipForgeSelfCheck` | builtins/tip_forge_selfcheck | auto | read | substrate | tip, forge, selfcheck, read | Pre-flight check for the special-tip recipes (MakeSpectroscopyTip / MakeAtomicResolutionTip): are the skills present in this build, can the substrate be resolved into a Shockley onset, will the evaluation frame actual… |
| `TipShape` | builtins/tip_shaper | auto | write | switch_off_delay_s, change_bias, bias_v, tip_lift_m, lift_time_1_s, bias_lift_v, bias_settling_s, lift_height_m, lift_time_2_s, end_wait_s, restore_feedback, allow_on_qplus, timeout_ms | tip, shaper, write | Run the hardware tip shaper procedure (controlled tip conditioning). |
| `TipShapeWithReadback` | builtins/tip_shaper_readback | auto | write | switch_off_delay_s, change_bias, bias_v, tip_lift_m, lift_time_1_s, bias_lift_v, bias_settling_s, lift_height_m, lift_time_2_s, end_wait_s, restore_feedback, timeout_ms, poll_hz, pre_roll_s, post_roll_s, max_capture_s, jump_k, indent_tol_nm, indent_tol_k | tip, shaper, readback, current, z, write | Run the hardware tip shaper AND stream current+Z during the procedure to capture how both channels jump at the apex-change (uses TipShaper_Start wait=0 so polling overlaps the action). |
| `AssessTipSharpness` | builtins/tip_sharpness | auto | analysis | scan_path, channel, sharp_edge_nm | tip, sharpness, step, analysis, read | How sharp the tip is, measured from a saved .sxm: the 10-90 rise width of the sharpest step edge (a sharp tip resolves a step in a few px; a blunt or double tip smears it), plus forward/backward instability and FFT sh… |
| `AssessAtomicPhase` | builtins/tip_spectro_assess | auto | analysis | scan_path, channel, expected_a_nm, substrate, snr_min, concentration_min, sharpness_min, allow_reduced_scale | tip, atomic, lattice, fft, analysis, read | Decide whether a saved .sxm frame shows ATOMIC resolution. Three independent criteria must agree: a strong peak in the atomic period band, discrete Bragg spots rather than a diffuse ring (this is what separates a real… |
| `AssessShockleyOnset` | builtins/tip_spectro_assess | auto | analysis | dat_path, expected_onset_v, substrate, tol_v, lockin_mod_vrms, temperature_k, width_max_v | tip, sts, spectroscopy, shockley, analysis, read | Fit the Shockley surface-state STEP in a saved dI/dV spectrum (.dat) and check its onset energy against the value for this substrate — the operator's test for a METALLIC tip. Reads only. Leave expected_onset_v empty t… |
| `TryEngageController` | builtins/zcontrol | confirm | write | settle_s, poll_hz, engage_fraction | z, controller, approach, engage, tip | Try to engage tunneling by turning the Z-controller ON (no coarse auto-approach). If the tunneling current reaches ~setpoint the tip is engaged and feedback is left ON; otherwise the controller is turned back OFF and … |
| `ApplyZCtrlPreset` | builtins/zctrl_presets_skills | confirm | write | preset | z, gain, setpoint, preset, write, readback | 按**参数组名**把 Z 控制器参数(P/I 增益 + 设定点)写进硬件。  **这是设置 Z 参数的首选方式**,优先于 SetZCtrlGain / SetSetpoint:具体数值由代码从操作员维护的存储里取出并写入,你只需要说用哪一组,不需要(也不应该)自己写出任何数字。  常用组名: - `approach` —— 进针参数(来自仪器档案,操作员填写) - `scan` —— 扫图参数(按当前扫描帧尺寸自动选档,与 Sc… |

## L2 — Pure data analysis — no Nanonis writes, may have zero TCP calls.

| Name | Folder | Safety | Category | Params | Tags | Description |
|---|---|---|---|---|---|---|
| `ApproachTip` | builtins/approach | auto | write | settle_s | approach, engage, tip, 进针, smart | Establish tunnelling the SAFE way for a plain '进针': first try engaging the Z-controller (feedback ON, no motor); ONLY if that can't reach tunnelling, fall back to the current-feedback AutoApproach (Nanonis stops at th… |
| `AssessClusterRoundness` | builtins/cluster_roundness | auto | read | scan_path, threshold_sigma, polarity, channel, select, min_axis_ratio, min_aspect | scan, analysis, cluster, roundness, read | Segment the cluster crater in a small .sxm and say how round it is. Main output is equivalent_axis_ratio ∈ (0,1]: 'as irregular as an ellipse with this short/long axis ratio' — 1.0 = perfect disc, and it means the sam… |
| `FindFlatRegion` | builtins/flat_region | auto | read | scan_path, window_fraction, stride_fraction, channel, exclude_used_spots, min_separation_m, min_window_m, same_terrace | scan, analysis, flat, read | Slide a window over a topography channel of a .sxm scan and return the window with the lowest plane-subtracted RMS roughness. Returns centre coords in metres (instrument frame). |
| `AnalyzeFrameTilt` | builtins/frame_tilt | auto | analysis | scan_path, channel, check_steps | scan, tilt, analysis, read | Measure the sample tilt, whether steps dominate the height spread, and the surface's own roughness — all from a saved .sxm file. Reads only; touches no hardware. Use it to obtain surface_rms_m for AutoTilt (the 'is th… |
| `MonitorCurrentFFT` | builtins/monitor_current_fft | auto | read | duration_s, poll_hz, window, detrend, output | current, fft, psd, software, read | Software FFT of tunneling current via TCP polling. Collects samples for `duration_s` at `poll_hz` Hz, then returns the magnitude (or power) spectrum. Latency = duration_s + 22 µs FFT. Nyquist = poll_hz / 2. Use when N… |
| `RunGridExperiment` | builtins/pattern | confirm | composite | nx, ny, center_x_m, center_y_m, width_m, height_m, angle_deg, wait_timeout_s | pattern, grid, spectroscopy, composite | Run grid spectroscopy using Nanonis Pattern module. Handles drift compensation and auto-save. |
| `GetLatestScanFile` | builtins/scan_extra | auto | read | max_age_s | scan, file, read, sxm | Locate the most recently written .sxm scan file. Looks first at Nanonis's reported session path, then at the data dir's working-sessions/, then at legacy dev locations. Returns {path: str, age_s: float} or {path: null… |
| `ScanIntelSelfCheck` | builtins/scan_intel_selfcheck | auto | read | probe_buffer_semantics | scan, diagnostics, commissioning, read | Read-only self-check of the scripted scan layer: which of its skills actually registered in THIS build, whether the per-scale policy table is the operator's or the factory one, whether the rig constants and the piezo-… |
| `AnalyzeScanImage` | builtins/scan_prep | auto | analysis | scan_path, threshold_profile, channel, flatten, save_png, output_dir | scan, analysis, flatten, preprocessing, read | Measure ONE saved .sxm frame and decide how it should be processed: which flattening (plane / 2nd-order surface / line-by-line / line-fitted-on-the-dominant-terrace) and how tight the colour scale should be — with the… |
| `AutoProcessScanBatch` | builtins/scan_prep | auto | analysis | folder, threshold_profile, channel, flatten, render, write_report, output_dir, max_files | scan, analysis, flatten, preprocessing, batch, read | Measure EVERY .sxm in a folder, pick the flattening and colour scale per frame, then HARMONISE frames that share a scan size and bias so a contrast difference between two of them can only come from the sample, never f… |
| `TiltProbeCircle` | builtins/tilt_probe | confirm | write | radius_m, n_points, settle_s, center_x_m, center_y_m, noise_floor_m | tilt, measure, smartilt, write | Measure the sample tilt by walking the tip around a small circle with the Z feedback ON (constant current) and fitting Z(theta). This is the self-hosted equivalent of the Nanonis SmarTilt button (the TCP protocol does… |
| `AssessImageQuality` | composite/assess_quality | auto | analysis | scan_path | analysis, quality, fft, composite | Assess scan image quality: FFT score, RMS roughness, and noise estimate. |
| `BiasSettleChange` | composite/bias_settle | confirm | write | bias_v, settle_s, allow_stop_in_deadband | bias, safety, settle, composite | Change the sample bias through the safe path: ramps across zero without lingering in the low-bias dead band (where the constant-current feedback would drive the tip into the surface), ramps large changes instead of st… |
| `PokeConditionTip` | composite/prepare_noble_tip | confirm | composite | poke_depth_nm, poke_dwell_s, cluster_scan_nm, min_axis_ratio, critical_start_pm, critical_step_pm, critical_repeat_n, poke_budget, allow_on_qplus | tip, shaper, conditioning, cluster, composite | Plunge-based tip refinement on a flat noble-metal area: a deep plunge first (shape), then step the depth up from 100 pm until Z just barely jumps and repeat at that threshold (size). Scans the cluster after each bite … |
| `PulseConditionTip` | composite/prepare_noble_tip | confirm | composite | pulse_v, pulse_width_s, pulse_success_dz_nm, pulses_per_polarity, pulse_budget, junction_bias_v, junction_setpoint_a | tip, pulse, conditioning, composite | Bias-pulse tip conditioning on a noble-metal surface: fire, watch Z step between its settled value before and after, move to fresh surface, repeat until Z jumps up by tens of nm. Flips polarity after a run of ineffect… |
| `RetractForSampleChange` | composite/retract_for_sample_change | confirm | write | total_steps | tip, safety, retract, sample-change, coarse, dangerous | 换样品/关机前的大退针：先收压电，再用粗动马达沿配置的退针方向分级(1→10→100→剩余)后退约数千步；每级退完开反馈回读 Z 压电走向自检(只有确认在远离才继续，检测到逼近立即停+撤针)。方向/步数/阈值来自 instrument_profile。 |

## L3 — Multi-step hardware workflow — orchestrates other skills via step()/step_or_fail().

| Name | Folder | Safety | Category | Params | Tags | Description |
|---|---|---|---|---|---|---|
| `AutoTilt` | composite/auto_tilt | confirm | composite | next_frame_m, radius_m, n_points, max_iterations, surface_rms_m, force | tilt, levelling, composite | Measure the sample tilt on a flat spot and compensate it with the piezo tilt, then re-measure to verify. Decides internally whether compensation is needed (from how much Z range the slope eats over the frame you are a… |
| `TiltCalibrate` | composite/auto_tilt | confirm | composite | step_deg, radius_m, n_points | tilt, calibration, composite | One-off calibration of how Piezo_TiltSet's two axes map onto the measured surface slope: applies a small probe step on each axis and solves the 2x2 response matrix (sign + axis swap + gain in one). Stores it in the in… |
| `BatchRegionsScan` | composite/batch_regions_scan | confirm | composite | regions, channels, line_time_s, wait_timeout_s, save_each, assess_quality | scan, batch, regions, survey, overview, composite | Scan a list of operator-specified regions one after another, producing one .sxm per region plus a summary. Use when the operator names several areas to image (e.g. before leaving the instrument overnight) and wants an… |
| `ConditionTip` | composite/condition_tip | confirm | composite | pulse_v, max_attempts, target_quality, center_x_m, center_y_m, scan_width_m | tip, conditioning, composite, quality | Automated tip conditioning loop: pulse, scan, FFT quality check, repeat until target quality is reached. |
| `DemoScanAndSTS` | composite/demo_scan_and_sts | auto | composite | center_x_m, center_y_m, scan_size_m, line_time_s, sts_count, sts_start_v, sts_end_v, sts_num_points, scan_timeout_s | demo, scan, sts, composite | DEMO ONLY - one-shot scan + STS at preset parameters. Skips ALL tip quality checks (no AssessImageQuality, ConditionTip, TipShape, PreScanCheck). Use for live demos where the operator already knows tip is good. Return… |
| `TrackDrift_ReferenceScan` | composite/drift_track | confirm | composite | ref_x_m, ref_y_m, ref_width_m, bias_v, ref_image_path | composite, drift, tracking, reference | Track sample drift by comparing to reference scan. Compensates scan frame position for measured drift. |
| `FullScan` | composite/full_scan | confirm | composite | center_x_m, center_y_m, width_m, height_m, line_time_s, channels, wait_timeout_s | scan, imaging, composite | One-step scan: configure area, set speed, start, and wait for completion. Combines ConfigureScan + SetScanSpeed + StartScan. |
| `GridSTS` | composite/grid_sts | confirm | composite | center_x_m, center_y_m, nx, ny, spacing_m, start_v, end_v, num_points | spectroscopy, grid, sts, composite | Acquire STS spectra on an NxN grid. Moves to each point and acquires a spectrum. |
| `MakeAtomicResolutionTip` | composite/make_special_tip | confirm | composite | eval_frame_nm, eval_pixels, eval_bias_v, eval_setpoint_a, eval_line_time_s, wiggle_upper_v, wiggle_burst_s, max_cycles, cycles_per_fallback, fallback_budget, expected_a_nm, substrate, allow_on_qplus | tip, atomic, lattice, wiggle, composite | Coax a tip into ATOMIC resolution: find a step-free patch, scan it fast at low bias (20 mV / 500 pA) while randomly hopping the bias inside ±20 mV, then take a clean frame and check the FFT for a real lattice. The wig… |
| `MakeSpectroscopyTip` | composite/make_special_tip | confirm | composite | substrate, expected_onset_v, onset_tol_v, poke_depth_nm, critical_repeat_n, mod_amp_v, mod_freq_hz, sts_points, stab_bias_v, stab_setpoint_a, temperature_k, max_rounds, allow_on_qplus | tip, sts, spectroscopy, shockley, composite | Turn a working tip into a METALLIC one fit for STS: plunge SHALLOW to grow a small round cluster, then switch the lock-in on and take a dI/dV spectrum, and check that the noble-metal (111) Shockley surface state appea… |
| `PrepareNobleTip` | composite/prepare_noble_tip | confirm | composite | pulse_v, pulse_width_s, pulse_success_dz_nm, pulses_per_polarity, pulse_budget, junction_bias_v, junction_setpoint_a, verify_bias_v, verify_setpoint_a, verify_scan_nm, fwdbwd_threshold, max_verify_rounds, poke_depth_nm, poke_dwell_s, cluster_scan_nm, min_axis_ratio, critical_start_pm, critical_step_pm, critical_repeat_n, poke_budget, step_scan_nm, flat_region_nm, allow_on_qplus, skip_refine | tip, conditioning, pulse, shaper, composite | Full tip-conditioning run on Au/Ag/Cu (single crystal or film): bias-pulse until Z steps up by tens of nm, verify forward/backward scan lines agree, find steps and level a flat patch, then plunge from deep to threshol… |
| `PreScanCheck` | composite/prescan_check | confirm | composite | center_x_m, center_y_m, width_m, quality_threshold, line_time_s, wait_timeout_s | composite, quality, prescan, tip | Quick single-line pre-scan to check tip quality. Returns tip_ready status and similarity score. |
| `RelocateCoarseXY` | composite/relocate_coarse_xy | confirm | write | axis, direction, steps, reapproach, prewithdraw_steps, dry_run | motor, coarse, relocate, safety, dangerous | **换区专用**:把样品台横向粗动到一片新表面。这是唯一应该自主使用的换区方式,不要直接调 MotorMove 做横向移动。 顺序:前置检查(真空互锁/驱动电压读回核对/扫描已停/落点不与去过的站点重叠) → 清障(收压电 + 粗动 Z 退针,逐级自检方向,并确认电流归零、qPlus 振幅恢复) → 分块横移(每块之后看电流/振幅/真空,异常立即停并再退针) → 步进计数器对账 → 可选重新进针。 成功后扫描地图自动进入**新的坐… |
| `ScanAt` | composite/scan_at | confirm | composite | center_x_m, center_y_m, size_m, purpose, bias_v, setpoint_a, line_time_s, pixels, angle_deg, channels, wait_timeout_s | scan, imaging, composite, policy | Acquire one scan at a location. You give WHERE (centre) and HOW BIG (size) — that is the intent. Scan speed, resolution, feedback gains and the default setpoint are chosen DETERMINISTICALLY from the operator's per-sca… |
| `ShapeTipOnSurface` | composite/shape_tip_on_surface | confirm | composite | wide_scan_path, wide_scan_width_m, cluster_window_m, shallow_depth_m, deep_depth_m, n_depth_steps, contact_threshold_a, monitor_after_s, min_axis_ratio, max_attempts, allow_on_qplus | tip, shaper, cluster, composite, autonomous | End-to-end tip shaping on the sample: locate a flat region on a wide scan, progressively plunge the tip via TipShape while monitoring current for contact, then scan the crater and assess roundness. Retries on a fresh … |
| `SurveySurface_TileScan` | composite/survey_surface | confirm | composite | center_x_m, center_y_m, total_size_m, tile_size_m, line_time_s, channels, wait_timeout_s, assess_quality | scan, survey, overview, composite | Survey a square surface region with NxN scan tiles to get a global overview. Each tile produces one .sxm file. Use BEFORE zooming into specific features. Run only after the tip is approached and stable. Optionally rat… |
| `TipPulse` | composite/tip_pulse | confirm | composite | pulse_v, duration_s, count | tip, pulse, conditioning, composite | Apply bias voltage pulse(s) for tip conditioning. Snapshots bias, pulses, restores. |

## L4 — Long autonomous / RL / paper-replication procedure.

| Name | Folder | Safety | Category | Params | Tags | Description |
|---|---|---|---|---|---|---|
| `ExecuteScanPlan` | composite/execute_scan_plan | confirm | composite | plan_json, stop_on_bad_quality, save_each | scan, batch, plan, composite | Execute a scan plan produced by plan_scan_batch: each frame is acquired with the parameters the planner resolved for it, with guards between frames (tip event -> abort the whole batch, poor quality -> one rescan, high… |
| `ForgeAuTip` | composite/forge_au_tip | confirm | composite | max_sites, max_rounds_per_site, forge_scan_nm, forge_step_scan_nm, forge_scan_timeout_s, forge_pixels, forge_line_time_s, relocate_steps, allow_on_qplus, pulse_v, pulse_budget, poke_budget, poke_depth_nm, poke_dwell_s, fwdbwd_threshold | tip, conditioning, forge, au111, coarse, composite | **Au(111) 全流程修针,中途不问人**:在当前站点用电脉冲大修 ⇄ 扫图验证正反扫描线,通过后找台阶调平、先深后浅扎针精修,再用台阶边缘锐度验收。这一站修不出来(表面用完 / 本站轮数耗尽)就**果断粗动换位置**并重新进针,在新站点从头再来。 有硬顶:站点数 × 每站轮数,加上粗动里程表自己的行程预算。预算耗尽会**如实报告未达标**并给出每站战绩 —— 不会谎报成功。收尾会**实测回读**偏压/电流设定/Z 反馈并如实… |

## Alphabetical index

- `AcquireBiasSweep` (L1, confirm)
- `AcquireLockInSweep` (L1, confirm)
- `AcquireOsciTrace` (L1, auto)
- `AcquirePLLFreqSweep` (L1, confirm)
- `AcquirePSD` (L1, auto)
- `AcquireSignalPoint` (L0, auto)
- `AcquireSTS` (L1, confirm)
- `AcquireZSpectr` (L1, confirm)
- `AnalyzeFrameTilt` (L2, auto)
- `AnalyzeScanImage` (L2, auto)
- `ApplyLockInPreset` (L1, confirm)
- `ApplyZCtrlPreset` (L1, confirm)
- `ApproachTip` (L2, auto)
- `AssessAtomicPhase` (L1, auto)
- `AssessClusterRoundness` (L2, auto)
- `AssessHerringbone` (L1, auto)
- `AssessImageQuality` (L2, auto)
- `AssessShockleyOnset` (L1, auto)
- `AssessTipSharpness` (L1, auto)
- `AtomTrackDriftComp` (L0, confirm)
- `AtomTrackQuickCompStart` (L0, confirm)
- `AtomTrackStatusGet` (L0, auto)
- `AutoApproach` (L1, auto)
- `AutoPhase` (L1, confirm)
- `AutoProcessScanBatch` (L2, auto)
- `AutoTilt` (L3, confirm)
- `AutoZeroBeamDeflection` (L0, confirm)
- `BatchRegionsScan` (L3, confirm)
- `BiasPulse` (L0, auto)
- `BiasPulseWithReadback` (L1, auto)
- `BiasSettleChange` (L2, confirm)
- `BiasWiggle` (L1, auto)
- `CaptureSignalBuffer` (L1, auto)
- `CheckScanForCrash` (L0, auto)
- `CheckTipCrashByAmplitude` (L0, auto)
- `CoarseMotionSelfCheck` (L0, auto)
- `ComputeDriftVector` (L0, auto)
- `ConditionTip` (L3, confirm)
- `ConfigureAtomTrack` (L1, confirm)
- `ConfigureBeamDeflection` (L0, confirm)
- `ConfigureBiasSweep` (L1, confirm)
- `ConfigureCalculatedOutput` (L0, confirm)
- `ConfigureDigitalLine` (L0, confirm)
- `ConfigureDualScope` (L0, auto)
- `ConfigureHighResScope` (L0, auto)
- `ConfigureHighSpeedSweep` (L0, confirm)
- `ConfigureInterferometer` (L0, confirm)
- `ConfigureKelvinController` (L0, confirm)
- `ConfigureLockIn` (L1, confirm)
- `ConfigureLockInDemod` (L1, confirm)
- `ConfigureLockInSweep` (L1, confirm)
- `ConfigureOcSync` (L0, confirm)
- `ConfigurePiController` (L0, confirm)
- `ConfigurePLL` (L1, confirm)
- `ConfigurePLLExcitation` (L1, confirm)
- `ConfigurePllSignalAnalyzer` (L0, auto)
- `ConfigurePreamp` (L0, confirm)
- `ConfigureProbeCurrentGain` (L0, confirm)
- `ConfigureProbeScanner` (L0, confirm)
- `ConfigureRfGenerator` (L0, confirm)
- `ConfigureScan` (L1, confirm)
- `ConfigureScopeTrigger` (L0, auto)
- `ConfigureSignalChart` (L0, auto)
- `ConfigureSpectrumAnalyzer` (L0, auto)
- `ConfigureSTS` (L1, confirm)
- `ConfigureSTSChannels` (L0, confirm)
- `ConfigureSTSTiming` (L0, confirm)
- `ConfigureTipRecorder` (L0, auto)
- `ConfigureWaveform` (L0, confirm)
- `ConfigureZSpectr` (L1, confirm)
- `ConfigureZSpectrTiming` (L0, confirm)
- `CreateZCtrlPreset` (L0, dangerous)
- `DelayLineGetDelay` (L0, auto)
- `DelayLineMoveTo` (L0, confirm)
- `DemoScanAndSTS` (L3, auto)
- `DeployNanonisScript` (L0, confirm)
- `DeployScriptLUT` (L0, confirm)
- `DrawScanMarker` (L0, auto)
- `EmergencyRetract` (L1, auto)
- `EnableSafeTip` (L0, auto)
- `EraseScanMarkers` (L0, auto)
- `ExecuteScanPlan` (L4, confirm)
- `ExtractClusters` (L0, auto)
- `FindCleanSpot` (L0, auto)
- `FindFlatRegion` (L2, auto)
- `ForgeAuTip` (L4, confirm)
- `FullScan` (L3, confirm)
- `GenSwpAcqChsGet` (L0, auto)
- `GenSwpPropsGet` (L0, auto)
- `GenSwpStop` (L0, confirm)
- `GenSwpSwpSignalGet` (L0, auto)
- `GetAcqPeriod` (L0, auto)
- `GetAutoApproachStatus` (L0, auto)
- `GetBeamDeflection` (L0, auto)
- `GetBias` (L0, auto)
- `GetBiasCalibration` (L0, auto)
- `GetCalculatedOutputConfig` (L0, auto)
- `GetChamberPressure` (L0, auto)
- `GetCpdCompensation` (L0, auto)
- `GetCurrent` (L0, auto)
- `GetCurrentBEEM` (L0, auto)
- `GetDataLogStatus` (L0, auto)
- `GetDemodHarmonic` (L0, auto)
- `GetDemodHPFilter` (L0, auto)
- `GetDemodLPFilter` (L0, auto)
- `GetDemodPhase` (L0, auto)
- `GetDemodPhasReg` (L0, auto)
- `GetDemodSignal` (L0, auto)
- `GetDigitalLineTTL` (L0, auto)
- `GetDriftCompensation` (L0, auto)
- `GetDualScopeData` (L0, auto)
- `GetGenericPiController` (L0, auto)
- `GetHighResScopeData` (L0, auto)
- `GetHighResScopeStatus` (L0, auto)
- `GetHighSpeedSweepStatus` (L0, auto)
- `GetHomeProps` (L0, auto)
- `GetInterferometer` (L0, auto)
- `GetKelvinController` (L0, auto)
- `GetLaser` (L0, auto)
- `GetLatestScanFile` (L2, auto)
- `GetLockInConfig` (L1, auto)
- `GetLockInSweepLimits` (L0, auto)
- `GetLockInSweepProps` (L0, auto)
- `GetLockInSweepSignal` (L0, auto)
- `GetMiscInstrumentConfig` (L0, auto)
- `GetMotorFreqAmp` (L0, auto)
- `GetMotorStepCounter` (L0, auto)
- `GetOcSync` (L0, auto)
- `GetOsciTimebases` (L0, auto)
- `GetPatternCloud` (L0, auto)
- `GetPatternProps` (L0, auto)
- `GetPiController` (L0, auto)
- `GetPiezoConfig` (L0, auto)
- `GetPiezoHVAInfo` (L0, auto)
- `GetPiezoHVAStatusLED` (L0, auto)
- `GetPiezoSensitivity` (L0, auto)
- `GetPiezoTilt` (L0, auto)
- `GetPiezoXYZLimits` (L0, auto)
- `GetPLLAddOnOff` (L0, auto)
- `GetPLLAmpCtrlOnOff` (L0, auto)
- `GetPllConfig` (L0, auto)
- `GetPLLDemodFilter` (L0, auto)
- `GetPLLDemodHarmonic` (L0, auto)
- `GetPLLDemodInput` (L0, auto)
- `GetPLLExcRange` (L0, auto)
- `GetPLLFreqRange` (L0, auto)
- `GetPLLFreqSwpParams` (L0, auto)
- `GetPLLInpCalibr` (L0, auto)
- `GetPLLInpProps` (L0, auto)
- `GetPLLPhasCtrlOnOff` (L0, auto)
- `GetPllSignalAnalyzerData` (L0, auto)
- `GetPLLSignalAnlzrCh` (L0, auto)
- `GetPLLSignalAnlzrFFTProps` (L0, auto)
- `GetPLLSignalAnlzrTimebase` (L0, auto)
- `GetPLLStatus` (L1, auto)
- `GetPllZoomFftData` (L0, auto)
- `GetPointShootOnOff` (L0, auto)
- `GetPointShootProps` (L0, auto)
- `GetPreamp` (L0, auto)
- `GetProbeBias` (L0, auto)
- `GetProbeCurrent` (L0, auto)
- `GetProbeZController` (L0, auto)
- `GetRfGeneratorStatus` (L0, auto)
- `GetRTFreq` (L0, auto)
- `GetRTOversample` (L0, auto)
- `GetSafeTipProps` (L0, auto)
- `GetSafeTipSignal` (L0, auto)
- `GetSafeTipStatus` (L0, auto)
- `GetScanBuffer` (L0, auto)
- `GetScanFrame` (L0, auto)
- `GetScanPatternConfig` (L0, auto)
- `GetScanSpeed` (L0, auto)
- `GetScanXYPosition` (L0, auto)
- `GetScriptChannels` (L0, auto)
- `GetScriptData` (L0, auto)
- `GetSessionPath` (L0, auto)
- `GetSetpoint` (L0, auto)
- `GetSignalCalibration` (L0, auto)
- `GetSignalRange` (L0, auto)
- `GetSignalsAddRT` (L0, auto)
- `GetSignalValues` (L0, auto)
- `GetSpectroscopyConfig` (L0, auto)
- `GetSpectroscopyStatus` (L0, auto)
- `GetSpectrumAnalyzerData` (L0, auto)
- `GetSTSAltZCtrl` (L0, auto)
- `GetSTSChannels` (L0, auto)
- `GetSTSDigSync` (L0, auto)
- `GetSTSLimits` (L0, auto)
- `GetSTSMLSLockinPerSeg` (L0, auto)
- `GetSTSPulseSeqSync` (L0, auto)
- `GetSTSSafeCond1` (L0, auto)
- `GetSTSTiming` (L0, auto)
- `GetSTSTTLSync` (L0, auto)
- `GetSTSZOffRevert` (L0, auto)
- `GetTcpLogStatus` (L0, auto)
- `GetTipLift` (L0, auto)
- `GetTipRecorderData` (L0, auto)
- `GetTipShaperConfig` (L0, auto)
- `GetTipSpeed` (L0, auto)
- `GetUserOutputLimits` (L0, auto)
- `GetUserOutputMode` (L0, auto)
- `GetUserOutputMonitorChannel` (L0, auto)
- `GetWaveformStatus` (L0, auto)
- `GetWithdrawRate` (L0, auto)
- `GetZControllerState` (L0, auto)
- `GetZCtrlGain` (L0, auto)
- `GetZCtrlList` (L0, auto)
- `GetZLimitsEnabled` (L0, auto)
- `GetZPosition` (L0, auto)
- `GetZSpectrChannels` (L0, auto)
- `GetZSpectrDigSync` (L0, auto)
- `GetZSpectrPulseSeqSync` (L0, auto)
- `GetZSpectrRange` (L0, auto)
- `GetZSpectrRetract` (L0, auto)
- `GetZSpectrRetract2nd` (L0, auto)
- `GetZSpectrTiming` (L0, auto)
- `GetZSpectrTTLSync` (L0, auto)
- `GrabScanFrameData` (L0, auto)
- `GridSTS` (L3, confirm)
- `HomeOpticalStage` (L0, confirm)
- `HomeZController` (L0, confirm)
- `ListLockInPresets` (L0, auto)
- `ListNanonisScripts` (L0, auto)
- `ListOpticalDevices` (L0, auto)
- `ListScanMarkers` (L0, auto)
- `ListSignalChannels` (L0, auto)
- `ListZCtrlPresets` (L0, auto)
- `LoadLayout` (L0, confirm)
- `LoadMultiPassConfig` (L0, dangerous)
- `LoadNanonisScript` (L0, dangerous)
- `LoadPiezoHysteresisFile` (L0, confirm)
- `LoadScanFrameFromFile` (L0, auto)
- `LoadScriptLUT` (L0, confirm)
- `LockNanonisUI` (L0, dangerous)
- `MakeAtomicResolutionTip` (L3, confirm)
- `MakeSpectroscopyTip` (L3, confirm)
- `MonitorCurrent` (L1, auto)
- `MonitorCurrentFFT` (L2, auto)
- `MotorGetPos` (L0, auto)
- `MotorMove` (L0, confirm)
- `MotorMoveClosedLoop` (L0, confirm)
- `MoveProbeXY` (L0, dangerous)
- `MoveToXY` (L0, confirm)
- `OpenPatternExperiment` (L0, confirm)
- `OpticalStageGetPos` (L0, auto)
- `OpticalStageMove` (L0, confirm)
- `OpticalStageScan` (L0, confirm)
- `OpticalStageWiggle` (L0, confirm)
- `ParseRegions` (L0, auto)
- `PausePatternExperiment` (L0, confirm)
- `PLLFreqShiftAutoCenter` (L0, confirm)
- `PLLOnOff` (L1, confirm)
- `PLLPerfectPLLUpdtZTC` (L0, confirm)
- `PLLSignalAnalyzer` (L1, auto)
- `PLLSignalAnlzrTrigAuto` (L0, confirm)
- `PokeConditionTip` (L2, confirm)
- `PrepareNobleTip` (L3, confirm)
- `PreScanCheck` (L3, confirm)
- `PulseConditionTip` (L2, confirm)
- `PulseDigitalLine` (L0, confirm)
- `PulseProbeBias` (L0, confirm)
- `PumpProbeScan` (L0, confirm)
- `QuitNanonis` (L0, dangerous)
- `ReadCalibrations` (L0, auto)
- `ReadHardwareEvents` (L0, auto)
- `ReadTipOscillationAmplitude` (L0, auto)
- `RelocateCoarseXY` (L3, confirm)
- `RetractForSampleChange` (L2, confirm)
- `RunBiasSweep` (L0, confirm)
- `RunCpdCompensation` (L0, confirm)
- `RunGridExperiment` (L2, confirm)
- `RunHighResScope` (L0, auto)
- `RunHighSpeedSweep` (L0, confirm)
- `RunNanonisScript` (L0, confirm)
- `RunPllPhaseSweep` (L0, confirm)
- `RunPllZoomFft` (L0, auto)
- `RunRfFrequencySweep` (L0, dangerous)
- `SafeRetract` (L0, auto)
- `SaveLayout` (L0, confirm)
- `SaveMultiPassConfig` (L0, confirm)
- `SaveNanonisScript` (L0, confirm)
- `SaveNanonisScriptLut` (L0, confirm)
- `SaveScan` (L1, auto)
- `SaveSettings` (L0, confirm)
- `ScanAt` (L3, confirm)
- `ScanBackgroundDelete` (L0, confirm)
- `ScanBackgroundPaste` (L0, confirm)
- `ScanIntelSelfCheck` (L2, auto)
- `SelectPokedCluster` (L1, auto)
- `SetAcquisitionPeriod` (L0, confirm)
- `SetActiveZController` (L0, confirm)
- `SetAdditionalRealtimeSignals` (L0, auto)
- `SetBias` (L0, confirm)
- `SetBiasCalibration` (L0, confirm)
- `SetBiasRamp` (L1, confirm)
- `SetBiasRange` (L0, confirm)
- `SetCurrentCalibration` (L0, confirm)
- `SetCurrentGain` (L0, confirm)
- `SetDemodRTSignals` (L0, confirm)
- `SetDemodSyncFilter` (L0, confirm)
- `SetDigitalLineStatus` (L0, confirm)
- `SetDriftCompensation` (L0, confirm)
- `SetFolMeOversampling` (L0, confirm)
- `SetGenericPiOutput` (L0, confirm)
- `SetHomeProps` (L0, confirm)
- `SetInterferometerOnOff` (L0, confirm)
- `SetKelvinControllerOnOff` (L0, confirm)
- `SetLaserOnOff` (L0, dangerous)
- `SetLaserPower` (L0, confirm)
- `SetLockInDemodPhaseRegister` (L0, auto)
- `SetLockInFrequencySweepSignal` (L0, confirm)
- `SetMlsLockinPerSegment` (L0, auto)
- `SetModHarmonic` (L0, confirm)
- `SetModPhasReg` (L0, confirm)
- `SetModSignal` (L0, confirm)
- `SetMotorFreqAmp` (L0, confirm)
- `SetMultiPass` (L0, confirm)
- `SetOsciTimebase` (L0, auto)
- `SetPatternCloud` (L0, confirm)
- `SetPatternExperiment` (L0, confirm)
- `SetPatternLine` (L0, confirm)
- `SetPiControllerOnOff` (L0, dangerous)
- `SetPiezoHysteresisOnOff` (L0, confirm)
- `SetPiezoHysteresisValues` (L0, confirm)
- `SetPiezoLimits` (L0, confirm)
- `SetPiezoRange` (L0, confirm)
- `SetPiezoSensitivity` (L0, confirm)
- `SetPiezoTilt` (L0, confirm)
- `SetPLLAmpCtrlBandwidth` (L0, confirm)
- `SetPLLAmpCtrlSetpnt` (L0, confirm)
- `SetPLLDemodFilter` (L0, confirm)
- `SetPllDemodHarmonic` (L0, auto)
- `SetPLLDemodInput` (L0, confirm)
- `SetPLLDemodPhasRef` (L0, confirm)
- `SetPllExcitationAdd` (L0, confirm)
- `SetPLLFreqExcOverwrite` (L0, confirm)
- `SetPLLFreqRange` (L0, confirm)
- `SetPLLInpCalibr` (L0, confirm)
- `SetPLLInpProps` (L0, confirm)
- `SetPLLInpRange` (L0, confirm)
- `SetPLLPhasCtrlBandwidth` (L0, confirm)
- `SetPLLSignalAnlzrTrig` (L0, confirm)
- `SetPointShootExperiment` (L0, confirm)
- `SetPointShootOnOff` (L0, confirm)
- `SetPointShootProps` (L0, confirm)
- `SetProbeBias` (L0, confirm)
- `SetProbeZController` (L0, confirm)
- `SetRTFreq` (L0, confirm)
- `SetRTOversample` (L0, confirm)
- `SetSafeTipProps` (L0, confirm)
- `SetScanBuffer` (L1, confirm)
- `SetScanSpeed` (L0, confirm)
- `SetScriptAutosave` (L0, auto)
- `SetScriptChannels` (L0, auto)
- `SetSessionPath` (L0, confirm)
- `SetSetpoint` (L0, confirm)
- `SetSpectroscopyPulseSync` (L0, confirm)
- `SetSpectroscopyTtlSync` (L0, confirm)
- `SetSpectroscopyZControl` (L0, confirm)
- `SetSpectrumAnalyzerBand` (L0, auto)
- `SetSTSAdvancedProps` (L0, confirm)
- `SetSTSChannels` (L0, confirm)
- `SetSTSMLSMode` (L0, confirm)
- `SetSTSMLSVals` (L0, confirm)
- `SetSTSSafeCond1` (L0, confirm)
- `SetSTSSafeCond2` (L0, confirm)
- `SetSwitchOffDelay` (L0, confirm)
- `SetTipLift` (L0, confirm)
- `SetTipSpeed` (L0, confirm)
- `SetUserOutput` (L0, confirm)
- `SetUserOutputCalibration` (L0, confirm)
- `SetUserOutputLimits` (L0, confirm)
- `SetUserOutputMode` (L0, confirm)
- `SetUserOutputMonitorChannel` (L0, confirm)
- `SetWaveformChannelOnOff` (L0, confirm)
- `SetWaveformIdleValue` (L0, confirm)
- `SetWaveformSignal` (L0, confirm)
- `SetWithdrawRate` (L0, confirm)
- `SetZCtrlGain` (L0, confirm)
- `SetZLimits` (L0, confirm)
- `SetZLimitsEnabled` (L0, confirm)
- `SetZPosition` (L0, confirm)
- `SetZSpectrAdvProps` (L0, confirm)
- `SetZSpectrChannels` (L0, confirm)
- `SetZSpectroscopySecondRetract` (L0, confirm)
- `SetZSpectrRange` (L0, confirm)
- `SetZSpectrRetract` (L0, confirm)
- `SetZSpectrRetractDelay` (L0, confirm)
- `ShapeTipOnSurface` (L3, confirm)
- `StartDataLog` (L0, auto)
- `StartRfGenerator` (L0, dangerous)
- `StartScan` (L1, confirm)
- `StartTcpLog` (L0, auto)
- `StartWaveform` (L0, confirm)
- `StopAutoApproach` (L0, auto)
- `StopDataLog` (L0, auto)
- `StopFolMe` (L0, auto)
- `StopHighSpeedSweep` (L0, auto)
- `StopMotor` (L0, auto)
- `StopNanonisScript` (L0, auto)
- `StopOpticalStage` (L0, auto)
- `StopPLLFreqSwp` (L0, confirm)
- `StopPllPhaseSweep` (L0, auto)
- `StopProbeScanner` (L0, auto)
- `StopRfGenerator` (L0, auto)
- `StopScan` (L0, auto)
- `StopSTS` (L0, auto)
- `StopTcpLog` (L0, auto)
- `StopWaveform` (L0, auto)
- `StopZSpectr` (L0, auto)
- `SurveySurface_TileScan` (L3, confirm)
- `TiltCalibrate` (L3, confirm)
- `TiltProbeCircle` (L2, confirm)
- `TipConditioningSelfCheck` (L1, auto)
- `TipForgeSelfCheck` (L1, auto)
- `TipPulse` (L3, confirm)
- `TipShape` (L1, auto)
- `TipShapeWithReadback` (L1, auto)
- `TrackDrift_ReferenceScan` (L3, confirm)
- `TryEngageController` (L1, confirm)
- `UndeployNanonisScript` (L0, auto)
- `UnlockNanonisUI` (L0, auto)
- `WaitForScanEndBlocking` (L0, confirm)
- `WaitScanComplete` (L1, auto)
- `WithdrawProbe` (L0, auto)
- `WithdrawTip` (L0, confirm)
- `ZControllerOnOff` (L0, confirm)
