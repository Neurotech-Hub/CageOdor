/******************************************************************************
 * bme688_cage_odor.ino
 *
 * Multi-sensor BME688 heater-sweep data logger for cage-odor discrimination.
 *
 * Target hardware : Bosch BME688 Development Kit
 *                   (Adafruit HUZZAH32 ESP32 Feather + 8x BME688 shield)
 *
 * WHAT THIS DOES
 *   Runs all 8 BME688 sensors in BME68X_PARALLEL_MODE, where the sensor
 *   hardware itself walks a 10-step heater profile and tags each reading with
 *   the step index it came from. The firmware rotates through a table of
 *   several heater profiles over time (see PROFILES[] below), so the dataset
 *   sweeps both heater temperature and heater dwell time. Every reading is
 *   written to microSD as one CSV row that fully describes itself: which
 *   sensor, which profile, which heater step, what temperature and dwell that
 *   step used, and what experiment label was active.
 *
 *   Two on-board buttons label the data while an experiment runs:
 *     Button A (GPIO 14) : cycle label UNKNOWN -> CLEAN -> SOILED -> UNKNOWN
 *     Button B (GPIO 32) : drop a generic event marker row
 *
 *   This sketch deliberately does NOT use BSEC2 / BME AI Studio. There are no
 *   .bmeconfig blobs and no IAQ index (IAQ only exists inside BSEC2). What you
 *   get instead is raw gas resistance in Ohms plus the raw heater registers,
 *   which is the better substrate for offline analysis.
 *
 * ---------------------------------------------------------------------------
 * INSTALL
 *   1. Arduino IDE -> Boards Manager -> install "esp32 by Espressif Systems".
 *      Select board: "Adafruit ESP32 Feather".
 *   2. Library Manager -> install:
 *        "BME68x Sensor library" by Bosch Sensortec
 *        "RTClib" by Adafruit
 *      (SD, SPI and Wire ship with the ESP32 core - nothing to install.)
 *   3. Copy these two files from the Bosch library's examples folder into
 *      THIS sketch folder, next to this .ino:
 *        commMux.h
 *        commMux.cpp
 *      They live at:
 *        <Arduino>/libraries/BME68x_Sensor_library/examples/bme688_dev_kit/
 *      They are Bosch BSD-3-Clause sources that drive the TCA6408 I2C GPIO
 *      expander which owns the 8 sensor chip-select lines. Do not rewrite them.
 *
 * WIRING
 *   None. The dev-kit shield already carries the 8 sensors, the microSD slot,
 *   both buttons and the RTC. Just seat the shield on the HUZZAH32 and power
 *   the board over USB.
 *
 * FIRST RUN
 *   Open Serial Monitor at 115200 baud. If the RTC has never been set (or its
 *   coin cell died) the sketch asks for the time. Send the current Unix epoch
 *   seconds as plain digits + newline, e.g.  1757328000
 *   Get it from: python -c "import time; print(int(time.time()))"
 *   The RTC keeps time on its coin cell afterwards, so this is a one-off.
 *
 * RUNNING THE SOILED-vs-CLEAN EXPERIMENT
 *   1. Power the board from a USB battery pack. Wait for the LED to stop
 *      blinking (SD + sensors OK) and for logging to start.
 *   2. Place the board in the soiled cage. Press Button A until the serial
 *      output reads SOILED. Leave it for ~1 hour.
 *   3. Move the board to the clean cage. Press Button A until it reads CLEAN.
 *      Leave it for ~1 hour.
 *   4. Press Button B whenever you do something worth marking (opened the
 *      cage, added bedding, walked past). Those rows are recoverable later.
 *   5. Power down, pull the card, analyse bme688_log_YYYYMMDD_HHMMSS.csv.
 *
 * ANALYSIS HINT
 *   The '#' metadata lines at the top of the CSV are comments:
 *     df = pd.read_csv(path, comment='#')
 *   Marker rows share the data schema; split them out with df['marker'].
 *   DISCARD THE FIRST CYCLE AFTER EACH PROFILE CHANGE - the hot plate has not
 *   settled yet. That is what the profile_cycle column is for:
 *     data = df[df.marker.isna() & (df.profile_cycle > df.groupby(
 *                 ['profile_id']).profile_cycle.transform('min'))]
 *   Then compare gas_resistance_ohm between labels, grouped by
 *   (profile_id, heater_step). The heater conditions showing the largest
 *   separation relative to within-label spread are the ones worth keeping.
 *
 * DATA RATE
 *   8 sensors x 10 steps per ~1.4 s ~= 57 rows/s ~= 7 kB/s.
 *   A 1-hour run is roughly 200k rows / ~25 MB. Use a decent microSD card.
 *   To thin the data out, raise measDurMs in the profile table below.
 ******************************************************************************/

#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <RTClib.h>

#include "bme68xLibrary.h"
#include "commMux.h"

/* ===========================================================================
 * CONFIGURATION - this block is the part you are meant to edit.
 * Everything below it is logic and should not need touching.
 * ===========================================================================
 */

#define FW_VERSION "cage-odor-1.0.0"

/* --- Pins. Fixed by the dev-kit shield; change only if you rewire. -------- */
static const uint8_t PIN_SD_CS = 33;           /* microSD chip select        */
static const uint8_t PIN_BTN_A = 14;           /* label button, active low   */
static const uint8_t PIN_BTN_B = 32;           /* marker button, active low  */
static const uint8_t PIN_LED   = LED_BUILTIN;  /* GPIO 13 on the HUZZAH32    */

/* --- Sensors -------------------------------------------------------------
 * Set an entry to false to skip a sensor that is flaky or missing. Logical
 * IDs are what you refer to in analysis; they stay stable even if you disable
 * a unit, so keep them fixed once you have started collecting data.
 */
#define N_KIT_SENS 8
static const bool    SENSOR_ENABLED[N_KIT_SENS]    = { true, true, true, true,
                                                       true, true, true, true };
static const uint8_t SENSOR_LOGICAL_ID[N_KIT_SENS] = { 0, 1, 2, 3, 4, 5, 6, 7 };

/* --- Heater profiles -----------------------------------------------------
 * Each profile is 10 steps. In parallel mode the sensor cycles all 10 steps
 * back to back and tags each reading with its step index.
 *
 *   temp[i]      heater target for step i, degrees C. The driver clamps to
 *                150-350, so do not exceed 350.
 *   mul[i]       dwell multiplier for step i. Actual dwell is
 *                mul[i] * sharedHeatrDur, where sharedHeatrDur is derived
 *                from measDurMs at profile-apply time. Keep mul in 1..63.
 *   measDurMs    total budget for one step, in ms. This is the knob that
 *                sweeps DURATION: a bigger budget means a longer dwell for
 *                the same multipliers, and a slower (thinner) data rate.
 *                Do not go below ~120 ms or the shared duration underflows.
 *
 * The firmware holds each profile for PROFILE_ROTATE_MS, then moves to the
 * next and starts over at the end of the table. Every row records which
 * profile it came from, so you can compare them offline.
 */
struct HeaterProfile {
  const char *name;
  uint16_t    temp[10];
  uint16_t    mul[10];
  uint16_t    measDurMs;
};

static const HeaterProfile PROFILES[] = {
  /* Bosch's stock dev-kit profile. Good general-purpose baseline and a
   * sensible reference point to compare the others against.             */
  { "bosch_default",
    { 320, 100, 100, 100, 200, 200, 200, 320, 320, 320 },
    {   5,   2,  10,  30,   5,   5,   5,   5,   5,   5 },
    140 },

  /* Monotonic temperature ramp at constant dwell. Isolates the effect of
   * hot-plate temperature alone - the cleanest thing to read off a plot. */
  { "ramp_200_350",
    { 200, 217, 233, 250, 267, 283, 300, 317, 333, 350 },
    {  10,  10,  10,  10,  10,  10,  10,  10,  10,  10 },
    140 },

  /* Same ramp, short dwell. Favours fast-responding surface reactions.   */
  { "ramp_short_dwell",
    { 200, 217, 233, 250, 267, 283, 300, 317, 333, 350 },
    {   3,   3,   3,   3,   3,   3,   3,   3,   3,   3 },
    140 },

  /* Same ramp, long dwell and a bigger budget. Favours slow equilibration;
   * VOCs that need time on the plate show up here and nowhere else.      */
  { "ramp_long_dwell",
    { 200, 217, 233, 250, 267, 283, 300, 317, 333, 350 },
    {  20,  20,  20,  20,  20,  20,  20,  20,  20,  20 },
    300 },
};
static const uint8_t N_PROFILES = sizeof(PROFILES) / sizeof(PROFILES[0]);

/* How long to hold each profile before rotating to the next one. */
static const uint32_t PROFILE_ROTATE_MS = 120000UL;   /* 2 minutes */

/* --- Logging ------------------------------------------------------------- */
static const char *LOG_DIR       = "/";               /* "/" = card root     */
static const char *LOG_BASE_NAME = "bme688_log";      /* filename prefix     */

/* --- Serial monitoring --------------------------------------------------- */
static const uint32_t SERIAL_SUMMARY_MS = 5000UL;  /* summary line interval  */
static const uint8_t  REFERENCE_SENSOR  = 0;       /* which sensor to print  */

/* --- Tuning (rarely changed) --------------------------------------------- */
static const uint16_t BTN_DEBOUNCE_MS  = 50;
static const size_t   SD_BUF_FLUSH_AT  = 2048;     /* flush at this many B   */
static const uint32_t SD_FLUSH_MS      = 2000UL;   /* ...or this often       */

/* ===========================================================================
 * END OF CONFIGURATION
 * ===========================================================================
 */

/* Experiment labels. Numeric values land in the CSV, so keep them stable. */
enum ExperimentLabel : uint8_t {
  LABEL_UNKNOWN = 0,
  LABEL_CLEAN   = 1,
  LABEL_SOILED  = 2,
  LABEL_COUNT   = 3
};
static const char *LABEL_NAMES[LABEL_COUNT] = { "UNKNOWN", "CLEAN", "SOILED" };

/* --- Globals ------------------------------------------------------------- */
static Bme68x   bme[N_KIT_SENS];
static commMux  commSetup[N_KIT_SENS];
static bool     sensorOk[N_KIT_SENS];
static uint32_t sensorUid[N_KIT_SENS];
static uint8_t  lastMeasIndex[N_KIT_SENS];
static bool     haveLastMeasIndex[N_KIT_SENS];
static uint8_t  nSensorsOk = 0;

static RTC_PCF8523 rtc;
static bool        rtcOk       = false;
static uint32_t    sessionUnix = 0;   /* epoch at session start, 0 if no RTC */

static File     logFile;
static bool     sdOk      = false;
static char     logPath[64] = { 0 };
static String   sdBuf;
static uint32_t lastSdFlushMs = 0;

static uint8_t  currentLabel   = LABEL_UNKNOWN;
static uint8_t  currentProfile = 0;
static uint32_t profileCycle   = 0;   /* increments on every profile apply  */
static uint32_t profileStartMs = 0;
static uint32_t lastSampleMs   = 0;
static uint32_t lastSummaryMs  = 0;

/* Dwell time actually programmed for each step of the current profile, ms.
 * Computed at apply time and written into every row so the CSV is
 * self-describing and analysis never has to look at this source file. */
static uint16_t currentStepDurMs[10];

/* Button edge-detect state. */
static bool     btnAPrev = true, btnBPrev = true;
static uint32_t btnALastMs = 0, btnBLastMs = 0;

/* LED error blink state. */
static uint32_t ledLastMs   = 0;
static bool     ledState    = false;
static uint32_t ledFlashUntil = 0;

/* ===========================================================================
 * Time helpers
 * ===========================================================================
 */

/* Current epoch seconds, or 0 if we have no RTC. */
static uint32_t nowUnix()
{
  if (!rtcOk) return 0;
  return rtc.now().unixtime();
}

/* ISO-8601 timestamp into buf. Empty string if we have no RTC, so the column
 * is blank rather than misleading - millis is always there as a fallback. */
static void nowIso(char *buf, size_t len)
{
  if (!rtcOk) { buf[0] = '\0'; return; }
  DateTime t = rtc.now();
  snprintf(buf, len, "%04d-%02d-%02dT%02d:%02d:%02d",
           t.year(), t.month(), t.day(), t.hour(), t.minute(), t.second());
}

/* Wait up to timeoutMs for a Unix epoch typed on the serial port. Returns 0
 * if nothing usable arrived. Non-blocking-ish: we only ever call this in
 * setup(), and only when the RTC actually needs setting. */
static uint32_t readEpochFromSerial(uint32_t timeoutMs)
{
  uint32_t start = millis();
  String   line;

  while (millis() - start < timeoutMs) {
    while (Serial.available()) {
      char c = (char)Serial.read();
      if (c == '\n' || c == '\r') {
        line.trim();
        if (line.length() >= 10) {          /* a plausible epoch */
          uint32_t v = (uint32_t)strtoul(line.c_str(), NULL, 10);
          if (v > 1600000000UL) return v;   /* sanity: after Sept 2020 */
        }
        line = "";
      } else if (isdigit((int)c)) {
        line += c;
      }
    }
    delay(10);
  }
  return 0;
}

/* ===========================================================================
 * LED
 * ===========================================================================
 */

static void ledFlash(uint16_t ms)
{
  ledFlashUntil = millis() + ms;
}

/* Steady blink when SD is dead, brief flash on button press, else off. */
static void serviceLed()
{
  uint32_t now = millis();

  if (!sdOk) {                       /* error pattern: fast, obvious */
    if (now - ledLastMs >= 150) {
      ledLastMs = now;
      ledState  = !ledState;
      digitalWrite(PIN_LED, ledState);
    }
    return;
  }

  if (now < ledFlashUntil) {
    digitalWrite(PIN_LED, HIGH);
  } else {
    digitalWrite(PIN_LED, LOW);
  }
}

/* ===========================================================================
 * SD logging
 * ===========================================================================
 */

static const char *CSV_HEADER =
  "timestamp_iso,millis,unix_time,label,label_name,marker,sensor_idx,logical_id,"
  "profile_id,profile_name,profile_cycle,heater_step,heater_temp_c,heater_dur_ms,"
  "gas_resistance_ohm,temperature_c,humidity_pct,pressure_pa,idac,res_heat,"
  "gas_wait_reg,status,gas_valid,heat_stable,meas_index,skipped";

/* Write straight to the card, bypassing the buffer. Used for the header and
 * metadata block. Returns false on failure. */
static bool sdWriteDirect(const char *s)
{
  if (!sdOk) return false;
  if (logFile.print(s) == 0) return false;
  return true;
}

/* Push the accumulated buffer to the card. One retry, then a close/reopen
 * retry, then we give up and fall back to serial-only so the bench session
 * survives a card that has come loose. */
static void sdFlush(bool force)
{
  if (!sdOk || sdBuf.length() == 0) return;

  uint32_t now = millis();
  if (!force && sdBuf.length() < SD_BUF_FLUSH_AT &&
      (now - lastSdFlushMs) < SD_FLUSH_MS) {
    return;
  }

  size_t want = sdBuf.length();
  size_t got  = logFile.print(sdBuf);

  if (got != want) {
    /* Retry once in place. */
    got = logFile.print(sdBuf);
  }
  if (got != want) {
    /* Close, reopen, retry. */
    Serial.println(F("[SD] write failed - reopening log file"));
    logFile.close();
    logFile = SD.open(logPath, FILE_APPEND);
    if (logFile) {
      got = logFile.print(sdBuf);
    }
  }
  if (got != want) {
    Serial.println(F("[SD] PERSISTENT WRITE FAILURE - logging disabled, "
                     "sensors still live on serial"));
    sdOk = false;
    logFile.close();
    sdBuf = "";
    return;
  }

  logFile.flush();          /* a power cut now costs at most one buffer */
  sdBuf = "";
  lastSdFlushMs = now;
}

static void sdAppend(const String &row)
{
  if (!sdOk) return;
  sdBuf += row;
  sdFlush(false);
}

/* Build the per-session filename. Uses the RTC when we have one; otherwise
 * scans for a free session number so runs never overwrite each other. */
static void buildLogPath()
{
  if (rtcOk) {
    DateTime t = rtc.now();
    snprintf(logPath, sizeof(logPath), "%s%s_%04d%02d%02d_%02d%02d%02d.csv",
             LOG_DIR, LOG_BASE_NAME,
             t.year(), t.month(), t.day(), t.hour(), t.minute(), t.second());
  } else {
    for (uint16_t n = 0; n < 10000; n++) {
      snprintf(logPath, sizeof(logPath), "%s%s_%04u.csv",
               LOG_DIR, LOG_BASE_NAME, n);
      if (!SD.exists(logPath)) return;
    }
    snprintf(logPath, sizeof(logPath), "%s%s_overflow.csv",
             LOG_DIR, LOG_BASE_NAME);
  }
}

/* The '#' comment block at the top of every file. Records everything needed
 * to interpret the run without going back to the firmware source. */
static void writeMetadataBlock()
{
  char line[192];
  char iso[24];
  nowIso(iso, sizeof(iso));

  sdWriteDirect("# BME688 cage-odor heater-sweep log\n");
  snprintf(line, sizeof(line), "# firmware_version,%s\n", FW_VERSION);
  sdWriteDirect(line);
  snprintf(line, sizeof(line), "# session_start_iso,%s\n",
           iso[0] ? iso : "(no RTC)");
  sdWriteDirect(line);
  snprintf(line, sizeof(line), "# session_start_unix,%lu\n",
           (unsigned long)sessionUnix);
  sdWriteDirect(line);
  snprintf(line, sizeof(line), "# rtc_present,%d\n", rtcOk ? 1 : 0);
  sdWriteDirect(line);
  snprintf(line, sizeof(line), "# sensors_detected,%u of %u\n",
           (unsigned)nSensorsOk, (unsigned)N_KIT_SENS);
  sdWriteDirect(line);

  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    snprintf(line, sizeof(line),
             "# sensor,%u,logical_id,%u,enabled,%d,ok,%d,unique_id,0x%08lX\n",
             (unsigned)i, (unsigned)SENSOR_LOGICAL_ID[i],
             SENSOR_ENABLED[i] ? 1 : 0, sensorOk[i] ? 1 : 0,
             (unsigned long)sensorUid[i]);
    sdWriteDirect(line);
  }

  snprintf(line, sizeof(line), "# profile_rotate_ms,%lu\n",
           (unsigned long)PROFILE_ROTATE_MS);
  sdWriteDirect(line);
  snprintf(line, sizeof(line), "# n_profiles,%u\n", (unsigned)N_PROFILES);
  sdWriteDirect(line);

  for (uint8_t p = 0; p < N_PROFILES; p++) {
    const HeaterProfile &hp = PROFILES[p];
    String t, m;
    for (uint8_t s = 0; s < 10; s++) {
      if (s) { t += ' '; m += ' '; }
      t += hp.temp[s];
      m += hp.mul[s];
    }
    snprintf(line, sizeof(line),
             "# profile,%u,%s,meas_dur_ms,%u,temps,%s,muls,%s\n",
             (unsigned)p, hp.name, (unsigned)hp.measDurMs,
             t.c_str(), m.c_str());
    sdWriteDirect(line);
  }

  sdWriteDirect("#\n");
  sdWriteDirect(CSV_HEADER);
  sdWriteDirect("\n");
  logFile.flush();
}

/* Marker rows share the data schema, with the sensor fields left empty. One
 * schema means pd.read_csv just works and markers filter out in one line. */
static void logMarker(const char *marker)
{
  char iso[24];
  nowIso(iso, sizeof(iso));

  String row;
  row.reserve(160);
  row += iso;                     row += ',';
  row += millis();                row += ',';
  row += nowUnix();               row += ',';
  row += currentLabel;            row += ',';
  row += LABEL_NAMES[currentLabel]; row += ',';
  row += marker;                  row += ',';
  row += ",,";                                     /* sensor_idx, logical_id */
  row += currentProfile;          row += ',';
  row += PROFILES[currentProfile].name; row += ',';
  row += profileCycle;            row += ',';
  row += ",,,,,,,,,,,,,";                          /* step..skipped, empty   */
  row += '\n';

  sdAppend(row);
  sdFlush(true);   /* markers matter - get them onto the card immediately */
}

static void logReading(uint8_t sensorIdx, const bme68xData &d, uint8_t skipped)
{
  char iso[24];
  nowIso(iso, sizeof(iso));

  uint8_t step = d.gas_index;
  if (step > 9) step = 9;                 /* defensive: index the tables safely */

  String row;
  row.reserve(200);
  row += iso;                                  row += ',';
  row += millis();                             row += ',';
  row += nowUnix();                            row += ',';
  row += currentLabel;                         row += ',';
  row += LABEL_NAMES[currentLabel];            row += ',';
  row += ',';                                  /* marker: empty for data rows */
  row += sensorIdx;                            row += ',';
  row += SENSOR_LOGICAL_ID[sensorIdx];         row += ',';
  row += currentProfile;                       row += ',';
  row += PROFILES[currentProfile].name;        row += ',';
  row += profileCycle;                         row += ',';
  row += d.gas_index;                          row += ',';
  row += PROFILES[currentProfile].temp[step];  row += ',';
  row += currentStepDurMs[step];               row += ',';
  row += String(d.gas_resistance, 2);          row += ',';
  row += String(d.temperature, 3);             row += ',';
  row += String(d.humidity, 3);                row += ',';
  row += String(d.pressure, 2);                row += ',';
  row += d.idac;                               row += ',';
  row += d.res_heat;                           row += ',';
  row += d.gas_wait;                           row += ',';
  row += d.status;                             row += ',';
  row += (d.status & BME68X_GASM_VALID_MSK) ? 1 : 0; row += ',';
  row += (d.status & BME68X_HEAT_STAB_MSK)  ? 1 : 0; row += ',';
  row += d.meas_index;                         row += ',';
  row += skipped;
  row += '\n';

  sdAppend(row);
}

/* ===========================================================================
 * Heater profile management
 * ===========================================================================
 */

/* Push profile idx to every working sensor.
 *
 * Order matters: the sensor must be in sleep mode before the heater profile
 * is rewritten. Reconfiguring mid-measurement is the classic way to get a
 * cycle of garbage readings. */
static void applyProfile(uint8_t idx)
{
  const HeaterProfile &p = PROFILES[idx];
  currentProfile = idx;
  profileCycle++;

  uint16_t shared = 0;

  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    if (!sensorOk[i]) continue;

    bme[i].setOpMode(BME68X_SLEEP_MODE);
    delay(2);
    bme[i].setTPH();     /* 2x temperature, 16x pressure, 1x humidity */

    /* Shared duration is what is left of the per-step budget once the TPH
     * conversion has been paid for. Bosch's own dev-kit example computes it
     * exactly this way. */
    uint32_t measDurUs = bme[i].getMeasDur(BME68X_PARALLEL_MODE);
    uint32_t measDurMs = measDurUs / 1000UL;
    if (p.measDurMs > measDurMs) {
      shared = (uint16_t)(p.measDurMs - measDurMs);
    } else {
      shared = 1;      /* budget too small; clamp rather than underflow */
    }

    bme[i].setHeaterProf((uint16_t *)p.temp, (uint16_t *)p.mul, shared, 10);
    bme[i].setOpMode(BME68X_PARALLEL_MODE);

    haveLastMeasIndex[i] = false;   /* meas_index restarts with the profile */
  }

  /* Record the dwell each step actually got, for the CSV. */
  for (uint8_t s = 0; s < 10; s++) {
    currentStepDurMs[s] = (uint16_t)(p.mul[s] * shared);
  }

  /* Print what the hardware was actually given. Cheapest possible check that
   * the arithmetic above produced the dwell times we intended. */
  Serial.println();
  Serial.printf("[PROFILE] -> %u '%s'  cycle=%lu  meas_dur_budget=%u ms  "
                "shared=%u ms\n",
                (unsigned)idx, p.name, (unsigned long)profileCycle,
                (unsigned)p.measDurMs, (unsigned)shared);
  Serial.print(F("[PROFILE] step:  "));
  for (uint8_t s = 0; s < 10; s++) Serial.printf("%5u ", (unsigned)s);
  Serial.print(F("\n[PROFILE] temp:  "));
  for (uint8_t s = 0; s < 10; s++) Serial.printf("%5u ", (unsigned)p.temp[s]);
  Serial.print(F("\n[PROFILE] dwell: "));
  for (uint8_t s = 0; s < 10; s++)
    Serial.printf("%5u ", (unsigned)currentStepDurMs[s]);
  Serial.println(F("  (ms)"));

  profileStartMs = millis();
  logMarker("profile_change");
}

/* ===========================================================================
 * Buttons
 * ===========================================================================
 */

static void serviceButtons()
{
  uint32_t now = millis();

  bool a = digitalRead(PIN_BTN_A);          /* active low */
  if (btnAPrev && !a && (now - btnALastMs) > BTN_DEBOUNCE_MS) {
    btnALastMs   = now;
    currentLabel = (uint8_t)((currentLabel + 1) % LABEL_COUNT);
    Serial.printf("\n[LABEL] now %s\n", LABEL_NAMES[currentLabel]);
    logMarker("label_change");
    ledFlash(200);
  }
  btnAPrev = a;

  bool b = digitalRead(PIN_BTN_B);
  if (btnBPrev && !b && (now - btnBLastMs) > BTN_DEBOUNCE_MS) {
    btnBLastMs = now;
    Serial.println(F("\n[EVENT] marker"));
    logMarker("event_marker");
    ledFlash(200);
  }
  btnBPrev = b;
}

/* ===========================================================================
 * Setup
 * ===========================================================================
 */

static void setupSerialBanner()
{
  Serial.println();
  Serial.println(F("========================================================"));
  Serial.println(F(" BME688 cage-odor heater-sweep logger"));
  Serial.printf ("  firmware      : %s\n", FW_VERSION);
  Serial.printf ("  sensors       : %u on the kit\n", (unsigned)N_KIT_SENS);
  Serial.printf ("  profiles      : %u, rotating every %lu s\n",
                 (unsigned)N_PROFILES,
                 (unsigned long)(PROFILE_ROTATE_MS / 1000UL));
  Serial.println(F("--------------------------------------------------------"));

  for (uint8_t p = 0; p < N_PROFILES; p++) {
    const HeaterProfile &hp = PROFILES[p];
    Serial.printf("  profile %u '%s' (budget %u ms)\n",
                  (unsigned)p, hp.name, (unsigned)hp.measDurMs);
    Serial.print(F("    temps (C) :"));
    for (uint8_t s = 0; s < 10; s++) Serial.printf(" %u", (unsigned)hp.temp[s]);
    Serial.print(F("\n    dwell mult:"));
    for (uint8_t s = 0; s < 10; s++) Serial.printf(" %u", (unsigned)hp.mul[s]);
    Serial.println();
  }
  Serial.println(F("--------------------------------------------------------"));
  Serial.println(F("  Button A (GPIO 14): cycle label UNKNOWN/CLEAN/SOILED"));
  Serial.println(F("  Button B (GPIO 32): drop an event marker"));
  Serial.println(F("========================================================"));
}

static void setupRtc()
{
  if (!rtc.begin()) {
    Serial.println(F("[RTC] PCF8523 NOT FOUND - falling back to millis only"));
    rtcOk = false;
    return;
  }
  rtcOk = true;

  if (!rtc.initialized() || rtc.lostPower()) {
    Serial.println(F("[RTC] not set. Send current Unix epoch seconds now"));
    Serial.println(F("[RTC]   python -c \"import time; print(int(time.time()))\""));
    Serial.println(F("[RTC] waiting 15 s..."));
    uint32_t epoch = readEpochFromSerial(15000);
    if (epoch) {
      rtc.adjust(DateTime(epoch));
      rtc.start();
      Serial.printf("[RTC] set to epoch %lu\n", (unsigned long)epoch);
    } else {
      Serial.println(F("[RTC] no time received - timestamps will be blank, "
                       "use the millis column"));
      rtcOk = false;
      return;
    }
  } else {
    rtc.start();
  }

  char iso[24];
  nowIso(iso, sizeof(iso));
  Serial.printf("[RTC] now %s\n", iso);
}

static void setupSd()
{
  if (!SD.begin(PIN_SD_CS)) {
    Serial.println(F("[SD] INIT FAILED - is a card inserted? "
                     "Continuing without logging."));
    sdOk = false;
    return;
  }
  sdOk = true;

  buildLogPath();
  logFile = SD.open(logPath, FILE_WRITE);
  if (!logFile) {
    Serial.printf("[SD] could not open %s - continuing without logging\n",
                  logPath);
    sdOk = false;
    return;
  }
  Serial.printf("[SD] logging to %s\n", logPath);
}

static void setupSensors()
{
  Wire.begin();
  commMuxBegin(Wire, SPI);

  Serial.println(F("[BME] probing sensors..."));
  Serial.println(F("  idx  logical  status  unique_id"));

  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    sensorOk[i]  = false;
    sensorUid[i] = 0;
    haveLastMeasIndex[i] = false;

    if (!SENSOR_ENABLED[i]) {
      Serial.printf("  %-4u %-8u %-7s -\n",
                    (unsigned)i, (unsigned)SENSOR_LOGICAL_ID[i], "SKIP");
      continue;
    }

    /* The 8 chip-selects live behind a TCA6408 I2C expander, not on ESP32
     * GPIOs, so we hand the driver Bosch's commMux transport rather than
     * calling begin(csPin, SPI). */
    commSetup[i] = commMuxSetConfig(Wire, SPI, i, commSetup[i]);
    bme[i].begin(BME68X_SPI_INTF, commMuxRead, commMuxWrite, commMuxDelay,
                 &commSetup[i]);

    if (bme[i].checkStatus() == BME68X_ERROR) {
      Serial.printf("  %-4u %-8u %-7s %s\n",
                    (unsigned)i, (unsigned)SENSOR_LOGICAL_ID[i], "FAIL",
                    bme[i].statusString().c_str());
      continue;
    }

    sensorUid[i] = bme[i].getUniqueId();
    sensorOk[i]  = true;
    nSensorsOk++;
    Serial.printf("  %-4u %-8u %-7s 0x%08lX\n",
                  (unsigned)i, (unsigned)SENSOR_LOGICAL_ID[i], "OK",
                  (unsigned long)sensorUid[i]);
  }

  Serial.printf("[BME] %u of %u sensors online\n",
                (unsigned)nSensorsOk, (unsigned)N_KIT_SENS);
  if (nSensorsOk == 0) {
    Serial.println(F("[BME] NO SENSORS - check the shield is fully seated"));
  }
}

void setup()
{
  Serial.begin(115200);
  delay(1500);                       /* let USB CDC come up before printing */

  pinMode(PIN_LED, OUTPUT);
  digitalWrite(PIN_LED, LOW);
  pinMode(PIN_BTN_A, INPUT_PULLUP);
  pinMode(PIN_BTN_B, INPUT_PULLUP);

  setupSerialBanner();
  setupRtc();
  setupSensors();
  setupSd();

  sessionUnix = nowUnix();
  sdBuf.reserve(SD_BUF_FLUSH_AT + 512);

  if (sdOk) {
    writeMetadataBlock();
  }

  logMarker("session_start");
  applyProfile(0);

  lastSampleMs  = millis();
  lastSummaryMs = millis();
  lastSdFlushMs = millis();

  Serial.println(F("[RUN] logging started"));
}

/* ===========================================================================
 * Loop
 * ===========================================================================
 */

static void sampleAllSensors()
{
  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    if (!sensorOk[i]) continue;

    uint8_t nFields = bme[i].fetchData();
    if (nFields == 0) continue;

    bme68xData d;
    while (bme[i].getData(d)) {
      if (!(d.status & BME68X_NEW_DATA_MSK)) continue;

      /* Cycles the MCU missed show up as a jump in meas_index. Log the gap
       * rather than swallowing it, so dropouts are visible in analysis
       * instead of masquerading as real signal. */
      uint8_t skipped = 0;
      if (haveLastMeasIndex[i]) {
        skipped = (uint8_t)(d.meas_index - lastMeasIndex[i] - 1);
      }
      lastMeasIndex[i]     = d.meas_index;
      haveLastMeasIndex[i] = true;

      logReading(i, d, skipped);

      if (i == REFERENCE_SENSOR) {
        /* stash nothing - the summary printer re-reads on its own cadence */
      }
    }
  }
}

static void printSummary()
{
  /* Take one fresh reading from the reference sensor purely for the bench
   * display. It comes from the same parallel-mode stream, so it costs
   * nothing extra on the wire. */
  if (!sensorOk[REFERENCE_SENSOR]) {
    Serial.println(F("[SUM] reference sensor offline"));
    return;
  }

  char iso[24];
  nowIso(iso, sizeof(iso));

  Serial.printf("[SUM] %s ms=%lu label=%s prof=%s cyc=%lu sd=%s\n",
                iso[0] ? iso : "(no-rtc)",
                (unsigned long)millis(),
                LABEL_NAMES[currentLabel],
                PROFILES[currentProfile].name,
                (unsigned long)profileCycle,
                sdOk ? "ok" : "FAIL");

  bme68xData d;
  uint8_t n = bme[REFERENCE_SENSOR].fetchData();
  if (n == 0) return;
  while (bme[REFERENCE_SENSOR].getData(d)) {
    if (!(d.status & BME68X_NEW_DATA_MSK)) continue;

    uint8_t step = d.gas_index;
    if (step > 9) step = 9;

    /* This reading is real data - do not throw it away just because we
     * pulled it for the display. */
    uint8_t skipped = 0;
    if (haveLastMeasIndex[REFERENCE_SENSOR]) {
      skipped = (uint8_t)(d.meas_index -
                          lastMeasIndex[REFERENCE_SENSOR] - 1);
    }
    lastMeasIndex[REFERENCE_SENSOR]     = d.meas_index;
    haveLastMeasIndex[REFERENCE_SENSOR] = true;
    logReading(REFERENCE_SENSOR, d, skipped);

    Serial.printf("      s%u step=%u T=%uC dwell=%ums  gas=%.0f ohm  "
                  "%.2f C  %.2f %%RH  %.0f Pa  %s%s\n",
                  (unsigned)REFERENCE_SENSOR, (unsigned)d.gas_index,
                  (unsigned)PROFILES[currentProfile].temp[step],
                  (unsigned)currentStepDurMs[step],
                  d.gas_resistance, d.temperature, d.humidity, d.pressure,
                  (d.status & BME68X_GASM_VALID_MSK) ? "" : "[gas invalid] ",
                  (d.status & BME68X_HEAT_STAB_MSK)  ? "" : "[unstable]");
  }
}

void loop()
{
  uint32_t now = millis();

  serviceButtons();
  serviceLed();

  /* Poll the sensors on the current profile's cadence. Nothing here blocks:
   * the heater work happens inside the sensor, not on the MCU. */
  if (now - lastSampleMs >= PROFILES[currentProfile].measDurMs) {
    lastSampleMs = now;
    sampleAllSensors();
  }

  if (now - profileStartMs >= PROFILE_ROTATE_MS && N_PROFILES > 1) {
    sdFlush(true);
    applyProfile((uint8_t)((currentProfile + 1) % N_PROFILES));
    lastSampleMs = millis();
  }

  if (now - lastSummaryMs >= SERIAL_SUMMARY_MS) {
    lastSummaryMs = now;
    printSummary();
  }

  sdFlush(false);
}
