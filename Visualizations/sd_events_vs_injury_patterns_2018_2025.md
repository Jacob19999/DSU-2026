## South Dakota events vs. injury patterns (2018–2025) — links from our visuals

**Scope**: This write-up is based on the heatmaps in:
- `Visualizations/injury_year_month_faceted*.png` (monthly counts by year for many injury types)
- `Visualizations/injury_types_seasonal_patterns.png` (season totals for top 12 injury types)
- `Visualizations/admission_rate_patterns.png` (admission rate by hour/day/month)

**Important constraint**: We only have **month-level** resolution and (from the plots) **injury-type** counts, not location/cause/mechanism. So connections to events are **correlational** (flagged where speculative).

---

## Cross-cutting signals (these show up everywhere)

- **Summer dominates total volume** (Jun–Aug):
  - `injury_types_seasonal_patterns.png` shows **Summer is the peak season** for nearly every top injury type (laceration, MVC, trauma, finger/ankle/arm injuries, assault victim).
  - The monthly heatmaps show repeated **Jun–Aug peaks** across trauma, lacerations, extremity/hand/arm injuries, insect/animal bites, and heat exposure.

- **Winter shows “environmental” injury signatures**:
  - **Cold exposure** and **frostbite** concentrate in **Jan–Feb** (and sometimes Dec).
  - Falls remain high (winter is often #2 behind summer), consistent with ice/slip risk.

- **Severity vs. volume mismatch** (strong signal):
  - `admission_rate_patterns.png` shows admissions are **highest in spring (peak ~April)** and lowest in late summer/early fall.
  - This matches “spring has fewer total injuries but more severe admissions” (storms, farm/industrial ramps, shoulder seasons), while summer has lots of “treat-and-release” lacerations/sprains/bites.

---

## Highest-signal month/event alignments (by injury type)

### Sturgis Motorcycle Rally (early August, annually) → **August spikes in trauma/MVC/laceration/assault**
**What we see**
- Multiple categories show repeated **July–August peaks**, and several years show **August** as one of the top months for:
  - **Trauma**, **Motor vehicle crash**, **Laceration**, **Assault victim**, **Extremity laceration**.

**Why this is a plausible SD-specific link**
- The Sturgis Motorcycle Rally is a uniquely large, annual, early-August SD event that increases:
  - road congestion + long-distance driving
  - motorcycle crashes
  - alcohol-related incidents
  - nighttime activity / assaults

**How to validate (if we have geography/cause fields)**
- Filter to **Black Hills counties** (Meade / Pennington) + **motorcycle mechanism** + **Aug 1–15** windows.

> **Confidence**: Medium–High as a recurring *timing* explanation; attribution to Sturgis specifically is **probable but not provable** from month-only statewide counts.

### COVID behavior shift (Mar–Apr 2020) → **large drop in motor vehicle crash**
**What we see**
- `Motor vehicle crash` has a very obvious **Mar–Apr 2020 dip** (Mar ~77, Apr ~53) compared to surrounding years/months (typically ~100–180).

**Why this links to SD events**
- Spring 2020 is when travel/activity patterns changed sharply due to COVID (reduced mobility, fewer events, fewer commutes), even without uniform “lockdown” policies.

> **Confidence**: High (pattern is classic and strong in the plot).

### Heat waves / hottest months → **Jun–Jul heat exposure spikes**
**What we see**
- `Heat exposure` is almost entirely a **Jun–Jul** phenomenon every year (with standout summers like 2018, 2024, 2025).

**Why this links to SD**
- SD heat waves and prolonged high-heat periods increase:
  - heat illness encounters
  - dehydration/syncope leading to falls
  - risk-taking outdoors (water recreation, alcohol)

> **Confidence**: High (mechanism is direct).

### Cold snaps / blizzards → **Jan–Feb cold exposure + frostbite spikes**
**What we see**
- `Cold exposure` and `Frostbite` spike hard in:
  - **Jan–Feb 2019**
  - **Jan–Feb 2022**
  - plus some **Dec** spikes (e.g., Dec 2022 frostbite is notably high in the plot)

**Why this links to SD**
- SD routinely experiences severe cold outbreaks / blizzards that drive:
  - exposure injuries (stranding, outdoor work, homelessness risk)
  - secondary crash/fall injuries (ice + visibility)

> **Confidence**: High (timing + injury type are tightly coupled).

### Mosquito / tick season & outdoor exposure → **Jun–Aug insect bites + animal bites**
**What we see**
- `Insect bite` shows huge **summer spikes** (notably Jul–Aug).
- `Animal bite` (and sometimes `Dog bite`) rises in summer, with particularly strong summers in later years.

**Why this links to SD**
- Upper Plains mosquito season peaks in summer; outdoor recreation + yard work + camping increase exposures.

> **Confidence**: High.

### Hunting season (Sep–Nov) → **Oct–Nov gunshot wound bumps**
**What we see**
- `Gun shot wound` shows several **Oct–Nov bumps** (e.g., 2020 Oct–Nov, 2021 Sep–Oct, 2024 Oct).

**Why this links to SD**
- Fall hunting season increases firearm handling, rural travel, and outdoor activity.

> **Confidence**: Medium (fits timing; still not provable without mechanism/context fields).

---

## Year-by-year narrative (2018–2025)

### 2018
- **Summer surge** across trauma/lacerations/extremity injuries; **July heat exposure** stands out.
- **August** is high for several “activity + crowding” categories (fits Sturgis/summer travel).

### 2019
- **Jan–Feb cold/frostbite** spike signature (classic severe-cold year).
- Summer stays high as usual; some categories show strong mid-summer peaks.

### 2020
- **Mar–Apr MVC dip** (COVID-era behavior shift).
- **Jul–Aug trauma stays high** (summer rebound + SD events; Sturgis 2020 is a plausible amplifier for August).

### 2021
- Summer remains dominant; insect bites and outdoor-injury categories are strong in mid-summer.

### 2022
- **Jan–Feb cold/frostbite** spike signature again.
- **Dec frostbite** also spikes (suggesting a severe late-year cold outbreak).

### 2023
- Strong **Aug** spikes in some “crowding/violence/outdoor” categories (assault victim is especially high in August).
- Animal/insect bite summer peaks remain strong (outdoor exposure).

### 2024
- Summer is strong across the board.
- `Heat exposure` is high in Jun–Jul; `Motor vehicle crash` shows a very large Aug cell (fits major travel/event window).

### 2025 (partial year in plots)
- `Heat exposure` spikes again in Jun–Jul.
- Many plots stop after Aug (so treat 2025 comparisons as incomplete).

---

## What to do next (to turn “plausible” into “defensible”)

- **Add an “event overlay” dataset** (date, county, event type, source link) and re-plot:
  - Sturgis window (Aug 1–15) by county
  - NWS storm days (hail/tornado/wind) by county
  - major winter storm days by county
- **Split injuries by mechanism** (if available): motorcycle vs car, slip/fall, firearm, assault, etc.
- **Do anomaly scoring**: for each injury type, compute a seasonal baseline and mark the top 1–2 outlier months per year; then match to events.

---

## Sources (starting points for event timelines)

- NOAA/NCEI billion-dollar disasters (South Dakota state summary): `https://www.ncei.noaa.gov/access/billions/state-summary/SD`
- NWS Sioux Falls event summaries (use for SD storm/flood/winter event timelines): `https://www.weather.gov/fsd/events`
- Sturgis Motorcycle Rally (timing/background): `https://en.wikipedia.org/wiki/Sturgis_Motorcycle_Rally`
- COVID-19 timeline in South Dakota (timing/background): `https://en.wikipedia.org/wiki/COVID-19_pandemic_in_South_Dakota`
