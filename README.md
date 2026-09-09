# CageOdor — BME688 heater-sweep cage logger

Firmware for the Bosch BME688 Development Kit (Adafruit HUZZAH32 ESP32
Feather + 8×BME688 shield) that sweeps the gas-heater profile across all 8
sensors and logs raw temperature / humidity / pressure / gas-resistance data
to microSD, to see whether the sensor can distinguish a soiled cage from a
clean one.

This deliberately does **not** use Bosch's BME AI Studio / BSEC2 — no
`.bmeconfig` blobs, no proprietary IAQ index. It uses the plain
[`BME68x Sensor library`](https://github.com/boschsensortec/Bosch-BME68x-Library),
which exposes raw gas resistance in Ohms and full manual control of the
heater profile, and the raw sensor data is what's logged.

The firmware lives in [bme688_cage_odor/bme688_cage_odor.ino](bme688_cage_odor/bme688_cage_odor.ino).
That file's header comment is the authoritative reference for install,
wiring, and how to run an experiment — this README is the short version.

## Install

1. **Arduino IDE**, Boards Manager → install *esp32 by Espressif Systems*.
   Board setting: **Adafruit ESP32 Feather**.
2. **Library Manager** → install:
   - `BME68x Sensor library` (Bosch Sensortec)
   - `RTClib` (Adafruit)

   `SD`, `SPI`, and `Wire` ship with the ESP32 core — nothing to install.
3. This repo already includes [commMux.h](bme688_cage_odor/commMux.h) and
   [commMux.cpp](bme688_cage_odor/commMux.cpp) next to the `.ino`, copied
   verbatim (BSD-3-Clause, Bosch Sensortec) from the BME68x library's
   `examples/bme688_dev_kit/` folder. They drive the TCA6408 I²C GPIO
   expander that owns the 8 sensors' SPI chip-select lines — the dev-kit
   shield does **not** wire chip-select to ESP32 GPIOs directly, so a plain
   `bme.begin(csPin, SPI)` will not work on this board. Do not edit these
   two files; if you ever need to re-copy them, they live at
   `<Arduino>/libraries/BME68x_Sensor_library/examples/bme688_dev_kit/`.

## Wiring

None needed. The dev-kit shield carries all 8 BME688 sensors, the microSD
slot, both buttons, and a PCF8523 RTC with backup coin cell. Seat the shield
on the HUZZAH32 and power over USB.

## First run

Open the Serial Monitor at 115200 baud after flashing. If the RTC has never
been set (or its coin cell died), the sketch prompts for the time — send the
current Unix epoch in seconds as plain digits + newline:

```sh
python -c "import time; print(int(time.time()))"
```

The RTC keeps time on its coin cell afterward, so this is normally a one-off.

## Running the soiled-vs-clean experiment

1. Power the board from a USB battery pack. Wait for the LED to stop
   blinking (SD + sensors OK) — logging starts automatically.
2. Place the board in the soiled cage. Press **Button A** (GPIO 14) until
   serial output reads `SOILED`. Leave it for ~1 hour.
3. Move the board to the clean cage. Press **Button A** until it reads
   `CLEAN`. Leave it for ~1 hour.
4. Press **Button B** (GPIO 32) whenever something worth marking happens
   (cage opened, bedding added, etc.) — it drops a timestamped event row.
5. Power down, pull the microSD card, then run the analysis script (see
   [Analyze](#analyze)).

## Configuration

All tunables — heater profile temperatures/durations, which sensors are
enabled, the profile-rotation period, filenames, pins — sit in one block at
the top of the `.ino`, clearly separated from the logic below it. Edit that
block; the rest of the sketch shouldn't need to change.

Four heater profiles ship by default, each a full 10-step sweep: Bosch's
stock profile, a monotonic 200→350 °C ramp, a short-dwell variant, and a
long-dwell variant. The firmware rotates between them every 2 minutes
(configurable), so a run sweeps both heater *temperature* and *dwell time*
over time. All 8 sensors always run the same profile at the same time, so
their readings are direct replicates for measuring sensor-to-sensor
variance.

## CSV format

One header row, preceded by `#`-commented metadata (firmware version,
session start time, detected sensor unique IDs, and the full profile table
actually used). Data rows and marker rows (`label_change`, `event_marker`,
`profile_change`, `session_start`) share one schema, so `pandas.read_csv`
loads the whole file in one call:

```python
import pandas as pd
df = pd.read_csv("bme688_log_...csv", comment="#")
data = df[df.marker.isna()]     # readings only
markers = df[df.marker.notna()] # label changes / events
```

**Discard the first cycle after each profile change** — the hot plate
hasn't settled yet. The `profile_cycle` column makes this a one-line filter.
Then compare `gas_resistance_ohm` between labels, grouped by
`(profile_id, heater_step)`, to see which heater conditions separate
"soiled" from "clean" best relative to within-label spread.

`UNKNOWN` is ambient air (board out of the cage), not discarded data. Keep it
when you look at how room air sits next to both cages.

## Analyze

The script in [`analysis/analyze_cage.py`](analysis/analyze_cage.py) loads one
labeled CSV, drops the minutes after each cage/label swap, plots all three
labels (UNKNOWN = air, CLEAN, SOILED), then trains a **CLEAN vs SOILED**
classifier with a time-based holdout.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis/analyze_cage.py path/to/bme688_log_YYYYMMDD_HHMMSS.csv
```

Useful flags:

- `--drop-after-label-min 10` — minutes of mixed air to drop after each `label_change` (default 10). The rest of each UNKNOWN block is kept as air.
- `--test-frac 0.3` — last fraction of each CLEAN and SOILED block used as the test set (not a shuffled split).
- `--out analysis/out` — where plots and CSVs are written.
- `--ref-sensor 0` — `logical_id` for the time-series plots.

Writes `heater_step_scores.csv`, `fingerprints.csv`, `metrics.csv`, and three PNGs under `--out`. UNKNOWN is scored after training as mean P(SOILED) so you can see whether air looks like a cage class; it is not part of accuracy.

## Data rate

8 sensors × 10 heater steps every ~1.4 s ≈ 57 rows/s ≈ 7 kB/s. A 1-hour run
is roughly 200k rows / ~25 MB — use a decent microSD card. To thin the data,
raise `measDurMs` in the profile table.
