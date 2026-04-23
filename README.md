# Egypt Credit Transportability

This repository contains code, data, and regenerated results for a reproducible study of cross-wave transportability of firm-level credit outcomes in Egypt using World Bank Enterprise Survey (WBES) data.

## Project overview

The project evaluates whether firm-level credit outcomes can be predicted across WBES waves in Egypt and whether objective credit outcomes are more transportable over time than subjective financing perceptions.

The analysis focuses on three outcomes:

- `target_k30`: severe financing obstacle
- `target_formal_credit`: formal credit access
- `target_applied_credit`: credit application

The workflow uses pooled Egypt survey data and a strict 2025 holdout evaluation design.

## Repository structure

- `code/` — Python scripts for data preparation, model reruns, and figure generation
- `data/` — raw and processed input data
- `results/` — regenerated output tables, summaries, and manuscript figure folders
- `.gitignore` — Git ignore rules

## Main scripts

### `build_combined_targets_egypt.py`
Builds the final modeling file used by the project:

- reads `egypt_wbes_pooled_clean_improved.csv`
- reads `egypt_alt_targets_pooled_clean.csv`
- verifies that the rows are aligned
- creates `egypt_combined_targets_pooled_clean.csv`

This script is required for full reproducibility because the final modeling file is created from two processed source files.

### `innovation_credit_pipeline_egypt_full_auto_proxyfallback_fixed.py`
Runs the main analysis pipeline. The script:

- loads the final combined processed data
- builds innovation variables
- applies proxy innovation fallback when raw innovation variables cannot be merged by ID
- fits the target-specific models
- writes transportability outputs, robustness summaries, coefficient tables, and innovation summaries
- creates manuscript-oriented result folders

### `make_manuscript_figures_egypt_color_v2.py`
Optional figure-only rerun script. Use this when you want to recreate figures from the final `egypt_final_*` output files without rerunning the full modeling pipeline.

## Data files needed

The full reproducible workflow depends on these processed files being present in `data/processed/`:

- `egypt_wbes_pooled_clean_improved.csv`
- `egypt_alt_targets_pooled_clean.csv`

The builder script creates:

- `egypt_combined_targets_pooled_clean.csv`

The project may also use raw WBES wave files stored under `data/raw/`.

## Reproducible workflow

Run the project from the repository root.

### 1. Build the final combined processed data

```powershell
python .\code\build_combined_targets_egypt.py --base-dir ".\data\processed"
```

### 2. Run the main modeling pipeline

```powershell
python .\code\innovation_credit_pipeline_egypt_full_auto_proxyfallback_fixed.py --input ".\data\processed\egypt_combined_targets_pooled_clean.csv" --outdir ".\results\final_results"
```

### 3. Optional: rerun the figure-only script

```powershell
Set-Location ".\results\final_results"
python "..\..\code\make_manuscript_figures_egypt_color_v2.py"
Set-Location "..\.."
```

## Expected outputs

After a successful rerun, the main results folder should contain files such as:

- `egypt_final_all_target_summary.csv`
- `egypt_final_target_k30_transport_results.csv`
- `egypt_final_target_formal_credit_transport_results.csv`
- `egypt_final_target_applied_credit_transport_results.csv`
- `egypt_final_target_*_recency_ablation.csv`
- `egypt_final_target_*_repeated_seed_summary.csv`
- `egypt_final_target_*_innovation_effects.csv`
- `run_note_innovation.json`

You should also see subfolders such as:

- `manuscript_figures_innovation/`
- `manuscript_tables_innovation/`

## Notes

- In the successful clean rerun for this project, the pipeline used proxy innovation measures because raw innovation variables could not be merged by ID from the pooled modeling file.
- The project has been tested by deleting previous results and regenerating outputs from code.
- If you later add a `manuscript/` folder to the repository, you can include the final paper file there without changing the code workflow.

## Suggested citation note

If you use this repository in a manuscript, describe it as the reproducible companion repository for the Egypt credit transportability study, including code, processed inputs, and regenerated results.
