# BioTDMS CLI — Integration Guide for Notebook Team

## Overview

`cli.py` processes raw physiological sensor data into the format the BioTDMS
Explorer app expects. Call it from notebooks via `os.system()` or `subprocess`
to stand up the app's data layer without launching the Streamlit UI.

## Prerequisites

- Python 3.12+
- Dependencies: `pandas`, `pyarrow` (for parquet), `openpyxl` (for subtask Excel)
- The `cli.py` script and `core/data_ingest.py` must be in the BioTDMS project directory

## File Naming Conventions

| File type             | Naming pattern                                          | Example                                               |
|-----------------------|---------------------------------------------------------|-------------------------------------------------------|
| Entropy/AMI CSV       | `team_entropy_ami_{DCE}.csv`                            | `team_entropy_ami_DCE3.csv`                           |
| Subtask lookup        | `SubTask_LookupTable_{DCE}.xlsx`                        | `SubTask_LookupTable_DCE3.xlsx`                       |
| Merged sensor CSV (legacy)  | `merged_{Day}_{Session}_{Role}_{Subject}_1hz.csv` | `merged_Day3_Session2_FOA_Subj023_1hz.csv`            |
| Merged sensor CSV (integration team format) | `merged_CRA_DCE{N}_Team{N}_{Day}_{Session}_{Role}_{Subject}_v{VERSION}_1hz.csv` | `merged_CRA_DCE1_Team3_Day1_Session3_FOA_Subj001_v1.0.0_1hz.csv` |
| Communication (zoom)  | `zoom_timeseries_Team{N}Day{N}Session{N}{Role}.csv`     | `zoom_timeseries_Team1Day2Session2FOA.csv`            |

**Merged sensor CSV — accepted variants.** The parser handles both the legacy filename format and the integration team's 2026-05 format. As long as the filename contains the tokens `Day{N}`, `Session{N}`, `{Role}`, `Subj{NNN}`, and ends with `_{N}hz.csv`, any extra prefix or version segments between them are absorbed. The file must still live in a folder literally named `merged/`, and the path above that folder must include both a DCE-bearing segment (e.g., `DCE2` or `CRA_DCE1`) and a `Team{N}` segment.

**Important**: Entropy and subtask files must have unique names per DCE so
multiple files can coexist in the app's data directory. Communication files
are scoped per-(team, day, session, role) — one file per role per session.

## Raw Data Directory Structure

`--raw-dir` accepts either of two source layouts (the parser handles both):

**Layout A — legacy:**
```
{raw_root}/
  DCE{N}/
    Team{N}/
      Session{N}/
        {Role}_Subj{NNN}/
          merged/
            merged_Day{N}_Session{N}_{Role}_Subj{NNN}_1hz.csv
```

**Layout B — integration team format:**
```
{raw_root}/
  CRA_DCE{N}/
    Team{N}/
      Day{N}/
        Session{N}/
          {Role}_Subj{NNN}/
            merged/
              merged_CRA_DCE{N}_Team{N}_Day{N}_Session{N}_{Role}_Subj{NNN}_v{VERSION}_1hz.csv
```

Point `--raw-dir` at the parent of the `DCE{N}` or `CRA_DCE{N}` folders. Either way the output parquets land at `{output-dir}/DCE{N}/Team{N}_Day{N}_Session{N}.parquet` — the `CRA_` prefix is stripped during ingestion to keep parquet paths consistent across deliveries.

### Communication file internal structure

Each `zoom_timeseries_*.csv` is a dense binary speaking trace for one role:

| Column    | Type   | Description                                                              |
|-----------|--------|--------------------------------------------------------------------------|
| `time`    | float  | Seconds since this role's recording start (~100 Hz, i.e. 10 ms steps)    |
| `Talking` | 0 / 1  | Binary speaker activity at that sample                                   |

**Alignment policy**: All comm files in a session share the same wall-clock
*end* (session stop), but recording *start* times vary per role. The app
end-anchors each file to the entropy length and pads the start with NaN if
the comm file is shorter than the entropy axis (early dropout). No upstream
trimming is required.

## Basic Usage

```python
import os

# From a notebook, process DCE3 data including the communication directory:
os.system(
    "python /path/to/biotdms/cli.py "
    "--raw-dir /path/to/DCE3_data "
    "--output-dir /path/to/biotdms/data/processed_sessions "
    "--data-dir /path/to/biotdms/data "
    "--entropy /path/to/team_entropy_ami_DCE3.csv "
    "--subtask /path/to/SubTask_LookupTable_DCE3.xlsx "
    "--com-dir /path/to/DCE3_zoom_timeseries/"
)
```

## Full Argument Reference

| Argument              | Required | Description |
|-----------------------|----------|-------------|
| `--raw-dir DIR`       | Yes      | Path to raw data (DCE{N}/Team{N}/... structure) |
| `--output-dir DIR`    | Yes      | Where processed session parquets are written |
| `--data-dir DIR`      | No       | Where entropy CSVs and subtask tables go (default: `data/`) |
| `--entropy FILE`      | No       | Entropy/AMI CSV to install (repeatable for multiple DCEs) |
| `--subtask FILE`      | No       | Subtask Excel to install (repeatable for multiple DCEs) |
| `--com-dir DIR`       | No       | Directory of `zoom_timeseries_*.csv` files to install into `<data-dir>/com_timeseries/` (repeatable). Non-matching CSVs are skipped and logged. |
| `--force`             | No       | Reprocess sessions even if parquets already exist |
| `--quiet`             | No       | Print summary only, suppress per-session output |
| `--rebuild-index-only`| No       | Just rebuild the session index from existing parquets |

## Common Patterns

### Process new data incrementally (skip existing sessions)
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions
```

### Process everything from scratch
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions --force
```

### Multiple entropy files in one call
```bash
python cli.py --raw-dir /data/raw \
    --output-dir data/processed_sessions \
    --entropy /data/team_entropy_ami_DCE2.csv \
    --entropy /data/team_entropy_ami_DCE3.csv
```

### Install communication (zoom_timeseries) data alongside other files
```bash
python cli.py --raw-dir /data/DCE3 \
    --output-dir data/processed_sessions \
    --entropy /data/team_entropy_ami_DCE3.csv \
    --subtask /data/SubTask_LookupTable_DCE3.xlsx \
    --com-dir /data/DCE3_zoom_timeseries/
```

The `--com-dir` flag walks the source directory, copies every
`zoom_timeseries_Team{N}Day{N}Session{N}{Role}.csv` it finds into
`<data-dir>/com_timeseries/`, and skips/logs anything else.

### Quiet mode for automated pipelines
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions --quiet
```

### Just rebuild the session index (no reprocessing)
```bash
python cli.py --raw-dir /data/DCE3 --output-dir data/processed_sessions --rebuild-index-only
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0    | Success — all sessions processed without errors |
| 1    | Partial success — some sessions failed (check output for details) |
| 2    | Fatal error — bad arguments, missing directories, invalid files |

## What Gets Created

```
data/
├── processed_sessions/
│   ├── DCE2/
│   │   ├── Team1_Day1_Session1.parquet
│   │   └── ...
│   ├── DCE3/
│   │   ├── Team1_Day1_Session1.parquet
│   │   └── ...
│   └── sessions_index.parquet                # auto-rebuilt after each run
├── com_timeseries/                            # installed via --com-dir
│   ├── zoom_timeseries_Team1Day2Session2FOA.csv
│   ├── zoom_timeseries_Team1Day2Session2Lead.csv
│   └── ...
├── team_entropy_ami_DCE2.csv                 # installed via --entropy
├── team_entropy_ami_DCE3.csv
└── SubTask_LookupTable_DCE3.xlsx             # installed via --subtask
```

Each session parquet contains:
- Metadata columns: `dce`, `team`, `day`, `session`
- `time_stamp` (unix epoch, 1hz)
- `trial_running` (boolean)
- Role-prefixed signals: `{Role}_{measure}` (e.g., `FOA_IBI`, `Lead_alpha_Fp1`, `JTAC_hbo`)
