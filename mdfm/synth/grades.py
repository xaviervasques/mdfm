"""Grade maps of the synthetic patient: from a clinical grade to operator parameters in physical units, as
docs/proposals/SYNTH_PATIENT_DRAFT.md (sections 3 to 6) proposes them, with decisions D3 and D4 as proposed
(docs/OWNER_ACTIONS.md, section 2): TWSTRS bands in degrees for the neck and DIS factors for trunk and limbs,
the SP1 joint-limit table as the range-of-motion denominator, tremor amplitude as peak-to-peak wrist
displacement.

Every map here is a proposal of the draft or a reading of this module, marked as such in SOURCES and in
reports/synth/operators.md; none binds until written into a rule file whose SHA-256 is asserted before an
outcome is read (draft, conventions). The MDS-UPDRS anchors are paraphrased in the draft and never copied
here. Levels in physical units are computed (levels()) and printed by scripts/sp3_decoder_prior_probe.py
--stimuli-only, never typed into a document.
"""
import math
from functools import lru_cache

import numpy as np

SOURCES = {
    "mds_updrs": "Goetz et al. 2008, doi 10.1002/mds.22340; items 3.6 to 3.8, 3.11, 3.15 to 3.18 as paraphrased in "
                 "docs/proposals/SYNTH_PATIENT_DRAFT.md sections 3 to 5",
    "elble2006": "Elble et al. 2006, doi 10.1093/brain/awl190 (abstract): ratings logarithmic in amplitude; draft "
                 "section 3 grade map",
    "charles1999": "Charles et al. 1999, Am Fam Physician 59(6): 1565-1572: rest tremor 4 to 6 Hz, essential 4 to "
                   "11 Hz, "
                   "physiological 8 to 12 Hz; draft section 3",
    "jankovic1999": "Jankovic et al. 1999, doi 10.1136/jnnp.67.5.646 (abstract): re-emergent tremor latency, mean "
                    "9.37 s; draft section 3",
    "bhatia2018": "Bhatia KP et al. 2018, Mov Disord 33(1): 75-87, doi 10.1002/mds.27121 (consensus statement on the "
                  "classification of tremors, full text as read in docs/proposals/SYMPTOM_CATALOGUE_DRAFT.md, "
                  "finding 1 and P27, and in mdfm/synth/tremors.py SOURCES): classic parkinsonian rest tremor "
                  "typically 4 to 7 Hz; dystonic tremor relieved at a null point (T18)",
    "video_floor": "docs/proposals/VIDEO_SYNTH_ROUNDTRIP_DRAFT.md section 3.4: 95th percentile in-band amplitude of "
                   "noise-only controls at the tremor framing, 4.69 mm (rest) and 4.89 mm (action), peak amplitude",
    "forearm": "docs/proposals/VIDEO_SYNTH_ROUNDTRIP_DRAFT.md section 3.2: neutral forearm 0.2494 m on the SMPL+H body "
               "(recomputed here from the selected body model's rest joints, the open body by default)",
    "schaafsma2003": "Schaafsma et al. 2003, doi 10.1046/j.1468-1331.2003.00611.x (abstract): triggers turn 63, start "
                     "23, narrow space 12, destination 9 percent; most episodes under 10 s; draft section 4",
    "wang2020": "Wang et al. 2020, arXiv 2009.12660: 551 episodes, about 1 s to 2 min, median 3 s; draft section 4",
    "moore2008": "Moore et al. 2008: freeze band 3 to 8 Hz against the locomotor band 0.5 to 3 Hz (DR-10.1 (b), "
                 "DR-10.13 of docs/rules/DESCRIPTOR_RULES_DRAFT.md); draft section 4",
    "twstrs": "TWSTRS severity scale (Consky and Lang 1994) as reproduced by Dystonia Canada; draft section 6",
    "dis": "Dyskinesia Impairment Scale (Monbaliu et al. 2012) as reproduced in the IAACD toolkit 2018; draft "
           "section 6",
    "table": "SP1 joint-limit table (body.joint_limits): since the open-body port's wave 1 the anatomical table of "
             "reports/openbody/joint_limits.json (scripts/openbody_limits.py, jarm2022 for every joint); decision D3",
    "limits_ext": "the SH2 dystonia extension of the SP1 joint-limit table (mdfm/synth/limits_ext.py, owner's go of "
                  "2026-11-02), read by the dystonia operators only: ankle inversion and toe flexion widened from "
                  "jarm2022 (corroborated by menadue2006 and ocran2023), each shoulder's adduction bound widened by "
                  "its collar's rise over the base; the sources and the numbers read in limits_ext.SOURCES",
    "module": "a reading or default of this module, not taken from a source (stated where used)",
    # task SP5 (2026-10-05): the exam motions the first exam set lacked; read on Europe PMC or PMC full text
    "maremmani2022": "Maremmani C et al. 2022, Acta Neurol Scand 146: 304-317, doi 10.1111/ane.13667 (full text, "
                     "Table 3): 50 healthy controls aged 65.5 (SD 2.7), wearable sensors, 10 s per task; movements "
                     "in 10 s, mean (SD), right and left: thumb-forefinger tapping 45.4 (7.5) and 43.8 (6.9), hand "
                     "opening-closing 35.7 (6.9) and 33.6 (6.5), forearm pronation-supination 22.6 (5.9) and 22.0 "
                     "(6.0), pronation-supination excursion 156.2 (27.9) and 154.0 (33.5) deg",
    "matejicka2024": "Matejicka P et al. 2024, Front Hum Neurosci 18: 1395827, doi 10.3389/fnhum.2024.1395827 (full "
                     "text, Table 3): 12 healthy controls aged 60 (SD 7.37), Leap Motion, finger tapping 10 s; "
                     "amplitude, the index-to-thumb fingertip distance from its global minimum to its peaks, 68.09 "
                     "(SE 3.24) mm; frequency 3.67 (SE 0.22) Hz; mean speed 0.44 (SE 0.02) m/s",
    "paparella2023": "Paparella G et al. 2023, J Parkinsons Dis 13: 1047-1060, doi 10.3233/JPD-230119 (full text, "
                     "Table 2): 77 healthy elderly, finger tapping 15 s, median (IQR): 50.91 (19.6) movements, "
                     "amplitude 49.24 (12.98) deg; a lower published rate, reported, not used",
    "inkster2004": "Inkster LM, Eng JJ 2004, Exp Brain Res 154: 33-38, doi 10.1007/s00221-003-1629-8 (author "
                   "manuscript PMC3478322, Tables 1 and 2): 10 controls aged 65.5 (SD 12.4) rising at a comfortable "
                   "pace without the arms; sit-to-stand 1.89 (SD 0.37) s, lift-off at 44.4 percent, hip flexion "
                   "(pelvis to thigh) 17.0 deg before extension, peak at 49.9 percent",
    "li2021": "Li J et al. 2021, Medicine 100(22): e26208, doi 10.1097/MD.0000000000026208 (full text): sit-to-stand "
              "from a 43-cm seat with the arms folded on the chest, 30 healthy adults",
    "bk2024": "B.K. MI et al. 2024, Indian J Plast Surg 57(4): 248-255, doi 10.1055/s-0044-1788593 (abstract): active "
              "range of motion of the finger joints, 390 hands of healthy adults",
    "boike2025": "Boike S et al. 2025, J Hand Surg Glob Online 7: 100792, doi 10.1016/j.jhsg.2025.100792 (full text, "
                 "Table 2): active thumb MCP and IP flexion, 162 adults",
    # review of task SP5 (2026-10-05): the balance of the arising (mdfm/synth/motions.py)
    "hof2005": "Hof AL, Gazendam MGJ, Sinke WE 2005, J Biomech 38: 1-8: the extrapolated centre of mass, position plus "
               "velocity over sqrt(g / l), inside the base of support as the condition for dynamic stability",
    "deleva1996": "de Leva P 1996, J Biomech 29: 1223-1230: segment masses and centre-of-mass positions, the table of "
                  "mdfm/kernels/force.py (DE_LEVA) used for the centre of mass",
}

# ---------------------------------------------------------------- tremor (draft section 3)
TREMOR_BANDS_CM = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, math.inf))   # MDS-UPDRS 3.15 to 3.17, grades 1 to 4
CONSTANCY_BANDS_PCT = ((0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0))   # 3.18, grades 1 to 4
FLOOR_PEAK_MM = {"rest": 4.69, "action": 4.89}                                   # SOURCES['video_floor']
# published frequency bands (Hz) of tremor.py's in_published_band flag: parkinsonian rest SOURCES['bhatia2018'] (4 to 7;
# 4 to 6 in SOURCES['charles1999'] until 2026-11-02, shared wave of the clinical review), the others charles1999
TREMOR_TYPES = {"parkinsonian_rest": (4.0, 7.0), "essential": (4.0, 11.0), "physiological": (8.0, 12.0)}
TREMOR_DEFAULT_HZ = 5.0           # centre of charles1999's rest band, inside bhatia2018's (the SP3 stimuli keep it)
REEMERGENCE_MEAN_S = 9.37         # SOURCES['jankovic1999']


def tremor_level_cm(grade, grade1="above_floor"):
    """Peak-to-peak displacement (cm) of a 3.15 to 3.17 grade at the hand (D4 read it at the wrist; since wave SH1d,
    decision of 2026-11-03, every hand tremor is levelled at the hand, tremor.point_readings), log-placed (draft
    section 3): grades 2 and 3 at the geometric centre of their bands; grade 4 at the ratio above 10 cm that grade
    2's centre has above 1 cm; grade 1 at two levels, 'above_floor' (geometric mean of the higher estimator floor, in
    peak-to-peak terms, and 1 cm) and 'below_floor' (the ratio below 1 cm that grade 2's centre has above it): module
    readings of 'one above, one below the estimator floor'. Grade 0: 0."""
    g = int(grade)
    if g == 0:
        return 0.0
    c2 = math.sqrt(TREMOR_BANDS_CM[1][0] * TREMOR_BANDS_CM[1][1])
    if g == 1:
        if grade1 == "above_floor":
            floor_pp = 2 * max(FLOOR_PEAK_MM.values()) / 10.0
            return math.sqrt(floor_pp * TREMOR_BANDS_CM[0][1])
        if grade1 == "below_floor":
            return TREMOR_BANDS_CM[0][1] / c2
        raise ValueError(f"grade1 {grade1!r}, expected 'above_floor' or 'below_floor'")
    if g in (2, 3):
        lo, hi = TREMOR_BANDS_CM[g - 1]
        return math.sqrt(lo * hi)
    if g == 4:
        return TREMOR_BANDS_CM[3][0] * c2
    raise ValueError(f"tremor grade {grade}, expected 0 to 4")


def tremor_grade_of_cm(cm):
    """The 3.15 to 3.17 grade whose band holds a peak-to-peak amplitude (cm)."""
    if cm <= 0:
        return 0
    for g, (lo, hi) in enumerate(TREMOR_BANDS_CM, 1):
        if lo < cm < hi or (g > 1 and cm == lo):
            return g
    return 4


def constancy_share(grade):
    """Share of the examination with rest tremor (0 to 1) of a 3.18 grade, at the centre of its band."""
    g = int(grade)
    if g == 0:
        return 0.0
    lo, hi = CONSTANCY_BANDS_PCT[g - 1]
    return (lo + hi) / 200.0


def constancy_grade_of(share):
    pct = 100.0 * share
    if pct <= 0:
        return 0
    for g, (lo, hi) in enumerate(CONSTANCY_BANDS_PCT, 1):
        if pct <= hi:
            return g
    return 4


def elbow_angle_deg(cm_pp, radius_m):
    """Peak angle (deg) at the driving hinge that moves a point at radius_m by cm_pp peak to peak: the chord
    between the two extremes of a symmetric oscillation, cm_pp / 100 = 2 r sin(theta)."""
    x = cm_pp / 100.0 / (2.0 * radius_m)
    if x > 1:
        raise ValueError(f"{cm_pp} cm peak to peak needs more than 90 degrees at a radius of {radius_m} m")
    return math.degrees(math.asin(x))


# ---------------------------------------------------------------- freezing of gait (draft section 4)
FOG_GRADES = {0: {"trigger": 0, "straight": 0}, 1: {"trigger": 1, "straight": 0}, 2: {"trigger": 2, "straight": 0},
              3: {"trigger": 2, "straight": 1}, 4: {"trigger": 2, "straight": 2}}   # module reading for grade 4
TRIGGER_SHARES = {"turn": 63.0, "start": 23.0, "doorway": 12.0, "destination": 9.0}  # SOURCES['schaafsma2003']
# Trigger events of the 3.11 grade-1 and grade-2 anchors (draft section 4). Reaching the destination is a published
# episode context (TRIGGER_SHARES) but not a trigger of the grade map, so the grade map never draws it.
GRADE_TRIGGERS = ("start", "turn", "doorway")
FREEZE_BAND_HZ = (3.0, 8.0)          # SOURCES['moore2008']
LOCOMOTOR_BAND_HZ = (0.5, 3.0)
TREMBLE_DEFAULT_HZ = 5.5             # centre of the freeze band (module default)
DURATION_MEDIAN_S = 3.0              # SOURCES['wang2020']
DURATION_P90_S = 10.0                # 'most under 10 s' read as the 90th percentile (module reading)
DURATION_RANGE_S = (1.0, 120.0)      # SOURCES['wang2020']
MANIFESTATIONS = ("trembling", "akinesia")    # shuffling: not in version 1 (reports/synth/operators.md)


def _truncated_cdf(x, mu, sigma, lo, hi):
    from scipy.stats import norm
    a, b = norm.cdf((math.log(lo) - mu) / sigma), norm.cdf((math.log(hi) - mu) / sigma)
    return (norm.cdf((math.log(x) - mu) / sigma) - a) / (b - a)


@lru_cache(maxsize=1)
def fog_lognormal():
    """(mu, sigma) of the lognormal whose truncation to DURATION_RANGE_S has its median at DURATION_MEDIAN_S and
    its 90th percentile at DURATION_P90_S (solved numerically; the published figures describe the episodes as
    observed, hence the truncated distribution)."""
    from scipy.optimize import fsolve
    lo, hi = DURATION_RANGE_S

    def eq(v):
        return [_truncated_cdf(DURATION_MEDIAN_S, v[0], v[1], lo, hi) - 0.5,
                _truncated_cdf(DURATION_P90_S, v[0], v[1], lo, hi) - 0.9]
    mu, sigma = fsolve(eq, [math.log(DURATION_MEDIAN_S), math.log(DURATION_P90_S / DURATION_MEDIAN_S) / 1.2816])
    if max(abs(x) for x in eq((mu, sigma))) > 1e-9:
        raise RuntimeError("the truncated lognormal of the episode durations did not solve")
    return float(mu), float(sigma)


def fog_durations(rng, n):
    """n episode durations (s) from the truncated lognormal of fog_lognormal (inverse transform of one uniform
    draw per episode), rounded to 0.01 s."""
    from scipy.stats import norm
    mu, sigma = fog_lognormal()
    lo, hi = DURATION_RANGE_S
    a, b = norm.cdf((math.log(lo) - mu) / sigma), norm.cdf((math.log(hi) - mu) / sigma)
    u = rng.uniform(a, b, size=int(n))
    d = np.exp(mu + sigma * norm.ppf(u))
    return [float(x) for x in np.clip(np.round(d, 2), lo, hi)]


# ---------------------------------------------------------------- bradykinesia (draft section 5)
N_CYCLES = 10
BRADY_INTERRUPTIONS = {0: 0, 1: 2, 2: 4, 3: 7}       # ladder (a); grade 3 also as one arrest
BRADY_DECREMENT_ONSET = {0: None, 1: 9, 2: 5, 3: 2}  # ladder (b), 1-based cycle of the first reduced amplitude
BRADY_DECREMENT_SLOPE = 0.10                          # share of A0 lost per cycle after onset (module default)
BRADY_TASKS = {"pronation_supination": "3.6", "toe_tapping": "3.7", "leg_agility": "3.8"}


def brady_parameters(grade, criterion, slope=BRADY_DECREMENT_SLOPE):
    """{n_hesitations, n_arrests, decrement_onset, decrement_slope} of a single-criterion ladder grade (draft
    section 5): 'interruptions' (0, 2, 4 and 7 hesitations), 'arrest' (grade 3 only: one arrest), 'decrement'
    (onset cycles 9, 5 and 2 at a fixed slope). 'slowing' has no physical anchor until the neurologist sets the
    speed gains after SP7 and raises; grade 4 (the task can barely be performed) has no ladder level and
    raises."""
    g = int(grade)
    if g == 4:
        raise ValueError("bradykinesia grade 4 has no ladder level in the draft (the task can barely be performed)")
    if criterion == "slowing":
        raise ValueError("slowing gains are set by the neurologist after SP7 (draft section 5): pass speed_gain")
    p = {"n_hesitations": 0, "n_arrests": 0, "decrement_onset": None, "decrement_slope": 0.0}
    if criterion == "interruptions":
        p["n_hesitations"] = BRADY_INTERRUPTIONS[g]
    elif criterion == "arrest":
        if g != 3:
            raise ValueError("the arrest level is a grade-3 level")
        p["n_arrests"] = 1
    elif criterion == "decrement":
        p["decrement_onset"] = BRADY_DECREMENT_ONSET[g]
        p["decrement_slope"] = slope if g else 0.0
    else:
        raise ValueError(f"criterion {criterion!r}, expected 'interruptions', 'arrest', 'decrement' or 'slowing'")
    return p


# Normal anchors of the repeated hand and forearm tasks (task SP5): the cycle rate of healthy controls, mean and
# SD, and the amplitude. Finger tapping (review of 2026-10-05): rate and amplitude of the same 12 controls,
# SOURCES['matejicka2024'] (frequency mean and SE, the SD taken as SE times the square root of n; one value for
# both hands), so that the level 0's rate, amplitude and speed are those of one cohort; the 50 controls of
# SOURCES['maremmani2022'] (movements in 10 s per side) are listed, not applied: their mean rate with the
# Matejicka amplitude made a level 0 faster than the amplitude source's own controls. Hand movements: movements
# per second of SOURCES['maremmani2022'] per side (movements in 10 s divided by 10); the amplitude is the full
# fist-to-open range of the cited finger joint ranges (mdfm/synth/hands.py). Pronation-supination is listed for
# reference: its version-2 level 0 (bradykinesia TASKS) is kept, so that the SP3 probe stimuli do not change.
BRADY_NORMAL = {
    "finger_tapping": {"item": "3.4", "rate_hz": {"mean": 3.67, "se": 0.22, "n": 12},
                       "rate_source": "matejicka2024", "amplitude": 68.09, "amplitude_unit": "mm",
                       "amplitude_source": "matejicka2024", "speed_m_s": {"mean": 0.44, "se": 0.02},
                       "not_applied": {"maremmani2022": {"movements": {"right": (45.4, 7.5), "left": (43.8, 6.9)},
                                                         "window_s": 10.0}}},
    "hand_movements": {"item": "3.5", "movements": {"right": (35.7, 6.9), "left": (33.6, 6.5)}, "window_s": 10.0,
                       "rate_source": "maremmani2022", "amplitude": 1.0,
                       "amplitude_unit": "share of the fist-to-open range", "amplitude_source": "bk2024"},
    "pronation_supination": {"item": "3.6", "movements": {"right": (22.6, 5.9), "left": (22.0, 6.0)},
                             "window_s": 10.0, "rate_source": "maremmani2022",
                             "amplitude": {"right": 156.2, "left": 154.0}, "amplitude_unit": "deg",
                             "amplitude_source": "maremmani2022", "applied": False},
}
SLOWING_SDS = 2.0               # grade 1 at the normal mean minus this many SD (module reading)


def normal_rate_hz(task, side="left"):
    """(mean, SD) Hz of the cited normal cycle rate of a task of BRADY_NORMAL: movements per window / window, or
    a cited rate whose SD is its SE times the square root of the cohort size (the same for both sides)."""
    a = BRADY_NORMAL[task]
    if "rate_hz" in a:
        r = a["rate_hz"]
        return r["mean"], r["se"] * math.sqrt(r["n"])
    m, sd = a["movements"][side]
    return m / a["window_s"], sd / a["window_s"]


def slowing_ratio(task, side="left"):
    """The step of the slowing ladder (module reading, a proposal for the neurologist, draft section 5): the ratio
    of the normal mean rate minus SLOWING_SDS SD to the mean (BRADY_NORMAL), so that grade 1 starts at the lower
    edge of the normal range; each grade slows by the same ratio again (rates on a log scale, as Elble et al. 2006
    found for tremor ratings; for slowing this is a reading of this module)."""
    m, sd = normal_rate_hz(task, side)
    return (m - SLOWING_SDS * sd) / m


def slowing_rate_hz(task, grade, side="left"):
    """Cycle rate (Hz) of a slowing grade 0 to 3 of a task of BRADY_NORMAL: the normal mean rate times
    slowing_ratio to the power of the grade (grade 0: the cited normal rate). Grade 4 (the task can barely be
    performed) has no level and raises."""
    g = int(grade)
    if not 0 <= g <= 3:
        raise ValueError("slowing grades run 0 to 3 (grade 4 has no ladder level)")
    return normal_rate_hz(task, side)[0] * slowing_ratio(task, side) ** g


def brady_grade_of(n_hesitations, n_arrests, decrement_onset):
    """Grade by the any-criterion rule of draft section 5 (interruptions 1-2, 3-5, more than 5 or an arrest;
    decrement onset near the end (9 or 10), midway (4 to 8), after the first (2 or 3); module reading of the
    cycle cut points)."""
    gi = 0 if n_hesitations == 0 else 1 if n_hesitations <= 2 else 2 if n_hesitations <= 5 else 3
    if n_arrests:
        gi = 3
    gd = 0 if decrement_onset is None else 1 if decrement_onset >= 9 else 2 if decrement_onset >= 4 else 3
    return max(gi, gd)


# ---------------------------------------------------------------- dystonic posture (draft section 6)
TWSTRS_BANDS_DEG = {"rotation": ((1, 22), (23, 45), (46, 67), (68, 90)),
                    "laterocollis": ((1, 15), (16, 35), (36, None))}
TWSTRS_THIRDS = ("anterocollis", "retrocollis", "shoulder_elevation")   # qualitative or by thirds of the range
TWSTRS_DURATION_PCT = ((0.0, 25.0), (25.0, 50.0), (50.0, 75.0), (75.0, 100.0))
DIS_BANDS_PCT = ((0.0, 10.0), (10.0, 50.0), (50.0, 90.0), (90.0, 100.0))   # amplitude and duration factors
SUBMAXIMAL = 0.5                   # share of the maximal level during 'mostly submaximal' time (module default)


def twstrs_level_deg(kind, grade, possible_deg):
    """TWSTRS severity level (deg) of a neck or shoulder item: rotation and laterocollis at the centre of their
    degree bands, the open laterocollis band closed by possible_deg (the SP1 table's range in that direction,
    None when unreachable); anterocollis, retrocollis and shoulder elevation at the centre of a third of the
    possible range (the scale's 'about half the possible range' at grade 2; thirds are a module reading for
    antero- and retrocollis). Grade 0: 0."""
    g = int(grade)
    if g == 0:
        return 0.0
    if kind in TWSTRS_BANDS_DEG:
        if not 1 <= g <= len(TWSTRS_BANDS_DEG[kind]):
            raise ValueError(f"TWSTRS {kind} grades run 0 to {len(TWSTRS_BANDS_DEG[kind])}")
        lo, hi = TWSTRS_BANDS_DEG[kind][g - 1]
        if hi is None:
            if possible_deg is None or possible_deg <= lo:
                return None
            hi = possible_deg
        return (lo + hi) / 2.0
    if kind in TWSTRS_THIRDS:
        if g > 3:
            raise ValueError(f"TWSTRS {kind} grades run 0 to 3")
        return (2 * g - 1) / 6.0 * possible_deg
    raise ValueError(f"TWSTRS kind {kind!r}")


def twstrs_duration(grade, excursion="maximal"):
    """(share of time 0 to 1, excursion) of a TWSTRS duration score 0 to 5: score = share band (1 to 4) + 1 when
    the excursion is often maximal; 'maximal' takes band score - 1 with maximal excursion, 'submaximal' band
    score with mostly submaximal excursion (score 1 is submaximal only, 5 maximal only)."""
    g = int(grade)
    if g == 0:
        return 0.0, None
    if g == 1 or (excursion == "submaximal" and g <= 4):
        band, exc = g, "submaximal"
    else:
        band, exc = g - 1, "maximal"
    lo, hi = TWSTRS_DURATION_PCT[band - 1]
    return (lo + hi) / 200.0, exc


def dis_share(grade):
    """Share (0 to 1) at the centre of a DIS amplitude or duration band; grade 0: 0."""
    g = int(grade)
    if g == 0:
        return 0.0
    lo, hi = DIS_BANDS_PCT[g - 1]
    return (lo + hi) / 200.0


def dis_grade_of(share):
    pct = 100.0 * share
    if pct <= 0:
        return 0
    for g, (lo, hi) in enumerate(DIS_BANDS_PCT, 1):
        if pct < hi:
            return g
    return 4
