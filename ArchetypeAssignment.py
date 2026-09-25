import pandas as pd
import numpy as np
import os

# 
# CONFIG
# 
DATA_DIR = "NBAPlayerStatsJoined"
OUT_DIR = "NBAPlayerArchetypesAssigned"
os.makedirs(OUT_DIR, exist_ok=True)


# HELPERS

def pct(df, col):
    """Return percentile ranks for a column (0–1)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[col].rank(pct=True)

def add_per36(df):
    """Add per-36 stats for points and assists."""
    mp = pd.to_numeric(df["MP"], errors="coerce")
    df["PTS_per36"] = np.where(mp > 0, df["PTS"] * (36 / mp), np.nan)
    df["AST_per36"] = np.where(mp > 0, df["AST"] * (36 / mp), np.nan)
    return df



# HARD-CODED ARCHETYPE LOGIC

def assign_archetype(row):
    """
    Assign archetype using:
      - position filter
      - hard conditions (must be true)
      - soft conditions (score = fraction satisfied)
      - guard fallback for high-usage scoring engines
    """
    pos_raw = str(row.get("Pos", "")).upper()

    def has_pos(allowed_positions):
        """Check if player's position matches any token in allowed_positions."""
        return any(p in pos_raw for p in allowed_positions)

    def is_guard():
        return has_pos(["PG", "SG"])

    def g(col, default=0.5):
        """Safe getter for percentile columns, defaulting to median if missing."""
        val = row.get(col)
        if pd.isna(val):
            return default
        try:
            return float(val)
        except Exception:
            return default

    # Convenience helpers for thresholds
    def high(col, thr=0.7):   return g(col) >= thr
    def mid(col, thr=0.5):    return g(col) >= thr
    def low(col, thr=0.4):    return g(col) <= thr

    
    # Archetype definitions (hard + soft)
    
    archetypes = [
        #  BIGS 
        {
            "name": "Stretch Big",
            "positions": ["PF", "C"],
            "hard": [
                lambda: high("3PAr_pct", 0.60),
                lambda: high("3PA_pct", 0.60),
            ],
            "soft": [
                lambda: high("3P_pct", 0.55),
                lambda: low("2PA_pct", 0.45),
                lambda: low("ORB_pct", 0.45),
                lambda: low("FTr_pct", 0.50),
            ],
        },
        {
            "name": "Rim running shotblocker",
            "positions": ["PF", "C"],
            "hard": [
                lambda: high("2P_pct", 0.65),
                lambda: high("BLK_pct", 0.65),
            ],
            "soft": [
                lambda: high("FG_pct", 0.60),
                lambda: high("TS_pct", 0.60),
                lambda: high("ORB_pct", 0.60),
                lambda: high("DRB_pct", 0.60),
                lambda: low("USG_pct", 0.55),
                lambda: low("3PAr_pct", 0.40),
                lambda: low("AST_pct", 0.45),
            ],
        },
        {
            # ENGINE BIG — Embiid/Jokic type
            "name": "Interior Creator",
            "positions": ["PF", "C"],
            "hard": [
                lambda: high("USG_pct", 0.60),
                lambda: high("PTS_pct", 0.60),
                lambda: high("FTr_pct", 0.65),
                lambda: high("AST_pct", 0.55),
            ],
            "soft": [
                lambda: high("TRB_pct", 0.55),
                lambda: high("OBPM_pct", 0.65),
                lambda: high("VORP_pct", 0.70),
                lambda: high("WS48_pct", 0.65),
            ],
        },

        #  GUARDS / CREATORS 
        {
            # Engine guard (Luka, Shai, Trae, etc.)
            "name": "Lead Playmaker",
            "positions": ["PG", "SG"],
            "hard": [
                lambda: high("AST_pct", 0.75),
                lambda: high("USG_pct", 0.55),
                lambda: high("PTS_pct", 0.55),
            ],
            "soft": [
                lambda: high("AST_TOV_pct", 0.80),
                lambda: high("OBPM_pct", 0.60),
                lambda: high("VORP_pct", 0.65),
                lambda: high("MP_pct", 0.60),
            ],
        },
        {
            # Scoring guard, not quite full-on floor general
            "name": "Combo Guard",
            "positions": ["PG", "SG"],
            "hard": [
                lambda: g("PTS_pct") >= 0.65,
                lambda: g("USG_pct") >= 0.65,
                lambda: g("FGA_pct") >= 0.60,
                lambda: g("3PA_pct") >= 0.50,
            ],
            "soft": [
                # Some playmaking, but not required to be elite
                lambda: 0.30 <= g("AST_pct") <= 0.75,
                lambda: g("3PAr_pct") >= 0.50,
                lambda: g("TOV_pct") >= 0.45,
                lambda: g("DBPM_pct") <= 0.60,
                lambda: g("PTS_pct") >= 0.75,
            ],
        },
        {
            "name": "Secondary Playmaker",
            "positions": ["PG", "SG", "SF"],
            "hard": [
                lambda: high("AST_pct", 0.60),
                lambda: low("USG_pct", 0.50),
            ],
            "soft": [
                lambda: high("AST_TOV_pct", 0.65),
                lambda: high("TS_pct", 0.50),
            ],
        },
        {
            "name": "Pitbull",
            "positions": ["PG", "SG"],
            "hard": [
                lambda: high("STL_pct", 0.70),
                lambda: low("USG_pct", 0.55),
            ],
            "soft": [
                lambda: low("3P_pct", 0.55),
                lambda: high("DBPM_pct", 0.60),
                lambda: high("DWS_pct", 0.50),
            ],
        },

        #  WINGS 
        {
            "name": "3&D Specialist",
            "positions": ["SG", "SF"],
            "hard": [
                lambda: high("3PA_pct", 0.65),
                lambda: high("3P_pct", 0.60),
            ],
            "soft": [
                lambda: high("3PAr_pct", 0.60),
                lambda: high("DBPM_pct", 0.55),
                lambda: low("TOV_pct", 0.50),
                lambda: low("AST_pct", 0.50),
                lambda: low("USG_pct", 0.55),
            ],
        },
        {
            "name": "2 Way Player",
            "positions": ["SG", "SF"],
            "hard": [
                lambda: high("OBPM_pct", 0.55),
                lambda: high("DBPM_pct", 0.55),
            ],
            "soft": [
                lambda: high("PTS_pct", 0.55),
                lambda: mid("USG_pct", 0.50),
            ],
        },
        {
            "name": "Slasher",
            "positions": ["SG", "SF"],
            "hard": [
                lambda: high("FTr_pct", 0.60),
                lambda: high("FTA_pct", 0.60),
            ],
            "soft": [
                lambda: mid("3P_pct", 0.45),      # average-ish 3PT
                lambda: high("2P_pct", 0.55),
                lambda: mid("USG_pct", 0.50),
                lambda: low("3PAr_pct", 0.55),
            ],
        },
        {
            "name": "Point Forward",
            "positions": ["SF", "PF"],
            "hard": [
                lambda: high("AST_pct", 0.65),
                lambda: high("USG_pct", 0.55),
            ],
            "soft": [
                lambda: high("TRB_pct", 0.55),
                lambda: high("OBPM_pct", 0.55),
                lambda: high("VORP_pct", 0.55),
            ],
        },
        {
            "name": "Shooter",
            "positions": ["SG", "SF"],
            "hard": [
                lambda: high("3PA_pct", 0.70),
                lambda: high("3PAr_pct", 0.70),
            ],
            "soft": [
                lambda: high("3P_pct", 0.60),
                lambda: high("eFG_pct", 0.60),
                lambda: low("USG_pct", 0.55),
                lambda: low("TOV_pct", 0.50),
                lambda: low("FTA_pct", 0.50),
                lambda: low("FTr_pct", 0.50),
                lambda: low("AST_pct", 0.50),
                lambda: low("ORB_pct", 0.50),
            ],
        },
        {
            "name": "Switchable defender",
            "positions": ["SG", "SF", "PF"],
            "hard": [
                lambda: high("DBPM_pct", 0.65),
            ],
            "soft": [
                lambda: high("DWS_pct", 0.60),
                lambda: high("STL_pct", 0.60),
                lambda: high("BLK_pct", 0.55),
                lambda: low("USG_pct", 0.55),
                lambda: low("AST_pct", 0.50),
            ],
        },
        {
            "name": "Playmaking forward",
            "positions": ["SF", "PF"],
            "hard": [
                lambda: high("AST_pct", 0.60),
            ],
            "soft": [
                lambda: low("3PA_pct", 0.55),
                lambda: high("2P_pct", 0.50),
                lambda: mid("USG_pct", 0.45),
            ],
        },

        #  BENCH / ROLE ARCHETYPES 
        {
            # Cam Thomas / Jordan Clarkson archetype
            "name": "Microwave",
            "positions": ["PG", "SG", "SF"],
            "hard": [
                lambda: high("USG_pct", 0.70),
                lambda: high("PTS36_pct", 0.70),
                lambda: low("VORP_pct", 0.65),
                lambda: low("WS48_pct", 0.65),
                lambda: low("DBPM_pct", 0.55),
                lambda: low("DWS_pct", 0.55),
            ],
            "soft": [
                lambda: 0.30 <= g("MP_pct") <= 0.70,
                lambda: g("AST_pct") <= 0.55,
                lambda: g("TOV_pct") >= 0.45,
                lambda: g("OBPM_pct") >= 0.50,
            ],
        },
        {
            "name": "Glue guy",
            "positions": ["SG", "SF", "PF"],
            "hard": [
                lambda: mid("AST_pct", 0.45),
                lambda: low("USG_pct", 0.55),
            ],
            "soft": [
                lambda: low("TOV_pct", 0.50),
                lambda: high("TS_pct", 0.55),
                lambda: high("WS48_pct", 0.55),
            ],
        },
        {
            "name": "Athletic finisher",
            "positions": ["SG", "SF", "PF"],
            "hard": [
                lambda: high("2P_pct", 0.60),
                lambda: high("FG_pct", 0.60),
            ],
            "soft": [
                lambda: low("3PAr_pct", 0.50),
                lambda: mid("USG_pct", 0.50),
                lambda: high("ORB_pct", 0.55),
            ],
        },
    ]

    best_name = "Unclassified"
    best_score = -1.0
    MIN_SOFT_SCORE = 0.25  

    
    # Primary archetype scoring
    
    for arch in archetypes:
        if not has_pos(arch["positions"]):
            continue

        # Hard conditions must all be true
        if not all(cond() for cond in arch["hard"]):
            continue

        soft_tests = arch["soft"]
        soft_score = 0.0
        if soft_tests:
            soft_true = sum(1 for cond in soft_tests if cond())
            soft_score = soft_true / len(soft_tests)

        if soft_score > best_score:
            best_score = soft_score
            best_name = arch["name"]

    
    # Guard fallback logic
    
    if best_name == "Unclassified" or best_score < MIN_SOFT_SCORE:
        if is_guard():
            usg = g("USG_pct")
            pts = g("PTS_pct")
            ast = g("AST_pct")

            # Engine scoring guard → Lead Playmaker
            if usg >= 0.60 and pts >= 0.65 and ast >= 0.55:
                return "Lead Playmaker"

            # Heavy scoring guard → Combo Guard
            if usg >= 0.60 and pts >= 0.70:
                return "Combo Guard"

        return "Unclassified"

    return best_name



# PROCESS ALL SEASONS

all_seasons = []

for file in os.listdir(DATA_DIR):
    if not file.endswith(".csv"):
        continue

    season = file.split("(")[1].split(")")[0]
    print(f"Processing {season}...")

    df = pd.read_csv(os.path.join(DATA_DIR, file))

    # Per-36 stats
    df = add_per36(df)

    # Percentiles needed for archetypes
    stats_needed = {
        "3PA": "3PA_pct",
        "3P%": "3P_pct",
        "3PAr": "3PAr_pct",
        "2P%": "2P_pct",
        "FG%": "FG_pct",
        "USG%": "USG_pct",
        "PTS": "PTS_pct",
        "AST%": "AST_pct",
        "STL%": "STL_pct",
        "STL": "STL_raw_pct",
        "BLK%": "BLK_pct",
        "FTr": "FTr_pct",
        "TOV%": "TOV_pct",
        "TS%": "TS_pct",
        "ORB%": "ORB_pct",
        "DRB%": "DRB_pct",
        "TRB%": "TRB_pct",
        "OBPM": "OBPM_pct",
        "DBPM": "DBPM_pct",
        "DWS": "DWS_pct",
        "VORP": "VORP_pct",
        "eFG%": "eFG_pct",
        "PTS_per36": "PTS36_pct",
        "AST_per36": "AST36_pct",
        "WS/48": "WS48_pct",
        "FGA": "FGA_pct",
        "FTA": "FTA_pct",
        "MP": "MP_pct",
    }

    for col, pctcol in stats_needed.items():
        df[pctcol] = pct(df, col)

    # AST/TOV ratio and percentile
    df["AST_TOV"] = df["AST"] / df["TOV"].replace(0, np.nan)
    df["AST_TOV_pct"] = pct(df, "AST_TOV")

    # Assign archetypes
    df["Archetype"] = df.apply(assign_archetype, axis=1)

    # Output only the columns you want
    slim = df[["Player", "Team", "Pos", "Archetype"]]

    out_path = os.path.join(OUT_DIR, f"PlayerArchetypes({season}).csv")
    slim.to_csv(out_path, index=False)


    all_seasons.append(df)

combined = pd.concat(all_seasons, ignore_index=True)
combined.to_csv(os.path.join(OUT_DIR, "PlayerArchetypes_AllSeasons.csv"), index=False)

print("Done. All archetypes assigned.")