
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
BASE_DIR = Path(".")
FIG_DIR = BASE_DIR / "manuscript_figures_color_v2"
TAB_DIR = BASE_DIR / "manuscript_tables_color_v2"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

OUTPUT_FORMAT = "png"   # use one format only
USE_TITLES = False      # keep titles off for manuscript figures

plt.rcParams["figure.dpi"] = 180
plt.rcParams["savefig.dpi"] = 320
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# ============================================================
# COLOR PALETTE
# ============================================================
# Journal-style palette with strong contrast
COLORS = {
    "navy":   "#1f4e79",
    "teal":   "#1b9e77",
    "orange": "#d95f02",
    "gold":   "#e6ab02",
    "purple": "#7570b3",
    "red":    "#c44e52",
    "gray1":  "#d9d9d9",
    "gray2":  "#bdbdbd",
    "gray3":  "#969696",
    "gray4":  "#636363",
    "sky":    "#4c9ed9",
    "mint":   "#66c2a5",
}

FILES = {
    "target_summary": "egypt_final_all_target_summary.csv",

    "transport_k30": "egypt_final_target_k30_transport_results.csv",
    "transport_formal": "egypt_final_target_formal_credit_transport_results.csv",
    "transport_applied": "egypt_final_target_applied_credit_transport_results.csv",

    "repeat_k30": "egypt_final_target_k30_repeated_seed_summary.csv",
    "repeat_formal": "egypt_final_target_formal_credit_repeated_seed_summary.csv",
    "repeat_applied": "egypt_final_target_applied_credit_repeated_seed_summary.csv",

    "ablation_k30": "egypt_final_target_k30_recency_ablation.csv",
    "ablation_formal": "egypt_final_target_formal_credit_recency_ablation.csv",
    "ablation_applied": "egypt_final_target_applied_credit_recency_ablation.csv",

    "subgroup_k30": "egypt_final_target_k30_subgroup_screen.csv",
    "subgroup_formal": "egypt_final_target_formal_credit_subgroup_screen.csv",
    "subgroup_applied": "egypt_final_target_applied_credit_subgroup_screen.csv",
}

# ============================================================
# HELPERS
# ============================================================
def read_csv_if_exists(filename):
    path = BASE_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

def save_figure(fig, stem):
    out_path = FIG_DIR / f"{stem}.{OUTPUT_FORMAT}"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {out_path}")

def pretty_target_name(target_key):
    mapping = {
        "target_k30": "Severe financing obstacle",
        "target_formal_credit": "Formal credit access",
        "target_applied_credit": "Credit application",
    }
    return mapping.get(target_key, target_key)

def short_target_name(target_key):
    mapping = {
        "target_k30": "Obstacle",
        "target_formal_credit": "Formal credit",
        "target_applied_credit": "Credit apply",
    }
    return mapping.get(target_key, target_key)

def pretty_model_name(name):
    mapping = {
        "Core_Logistic_balanced": "Core logistic",
        "Core_Threshold_Logistic_balanced": "Core threshold",
        "Rich_Logistic_balanced": "Rich logistic",
        "Rich_Logistic_recentWeighted": "Rich logistic + recency",
        "Rich_XGB_recentWeighted": "Rich XGBoost + recency",
        "Stacked_Meta_Logistic": "Stacked meta-logit",
        "Rich_Logistic_balanced_train_2013_2016_2020": "2013+2016+2020",
        "Rich_Logistic_balanced_train_2016_2020": "2016+2020",
        "Rich_Logistic_balanced_train_2020_only": "2020 only",
    }
    return mapping.get(name, name)

def wrap_labels(labels, width=18):
    wrapped = []
    for lab in labels:
        words = str(lab).split()
        line = ""
        lines = []
        for w in words:
            if len(line) + len(w) + 1 <= width:
                line = (line + " " + w).strip()
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        wrapped.append("\n".join(lines))
    return wrapped

def clean_axes(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.18, linewidth=0.8)
    ax.grid(axis="y", visible=False)

def outcome_color(label):
    if label == "Obstacle":
        return COLORS["navy"]
    if label == "Formal credit":
        return COLORS["orange"]
    if label == "Credit apply":
        return COLORS["teal"]
    return COLORS["gray3"]

def model_color(model_name):
    mapping = {
        "Core_Logistic_balanced": COLORS["gray2"],
        "Core_Threshold_Logistic_balanced": COLORS["gray4"],
        "Rich_Logistic_balanced": COLORS["navy"],
        "Rich_Logistic_recentWeighted": COLORS["sky"],
        "Rich_XGB_recentWeighted": COLORS["purple"],
        "Stacked_Meta_Logistic": COLORS["gold"],
    }
    return mapping.get(model_name, COLORS["gray3"])

def objective_bar_color(model_name):
    mapping = {
        "Rich_Logistic_balanced": COLORS["navy"],
        "Rich_Logistic_recentWeighted": COLORS["sky"],
        "Stacked_Meta_Logistic": COLORS["gold"],
        "Core_Logistic_balanced": COLORS["gray3"],
        "Rich_XGB_recentWeighted": COLORS["purple"],
    }
    return mapping.get(model_name, COLORS["gray3"])

def subgroup_color(target_key):
    return COLORS["teal"] if target_key == "target_applied_credit" else COLORS["orange"]

# ============================================================
# LOAD DATA
# ============================================================
target_summary = read_csv_if_exists(FILES["target_summary"])

transport = {
    "target_k30": read_csv_if_exists(FILES["transport_k30"]),
    "target_formal_credit": read_csv_if_exists(FILES["transport_formal"]),
    "target_applied_credit": read_csv_if_exists(FILES["transport_applied"]),
}

repeated = {
    "target_k30": read_csv_if_exists(FILES["repeat_k30"]),
    "target_formal_credit": read_csv_if_exists(FILES["repeat_formal"]),
    "target_applied_credit": read_csv_if_exists(FILES["repeat_applied"]),
}

ablation = {
    "target_k30": read_csv_if_exists(FILES["ablation_k30"]),
    "target_formal_credit": read_csv_if_exists(FILES["ablation_formal"]),
    "target_applied_credit": read_csv_if_exists(FILES["ablation_applied"]),
}

subgroup = {
    "target_k30": read_csv_if_exists(FILES["subgroup_k30"]),
    "target_formal_credit": read_csv_if_exists(FILES["subgroup_formal"]),
    "target_applied_credit": read_csv_if_exists(FILES["subgroup_applied"]),
}

# ============================================================
# SAVE TABLES AGAIN
# ============================================================
if target_summary is not None:
    ts = target_summary.rename(columns={
        "survey_year": "Survey year",
        "target_k30": "Severe financing obstacle",
        "target_formal_credit": "Formal credit access",
        "target_applied_credit": "Credit application",
    }).copy()
    ts.to_csv(TAB_DIR / "table1_target_prevalence_by_wave.csv", index=False)

main_tables = []
for target_key, df in transport.items():
    if df is not None:
        out = df[["model", "ROC_AUC", "PR_AUC", "F1", "Recall", "Precision", "Accuracy"]].copy()
        out["model"] = out["model"].map(pretty_model_name)
        out.insert(0, "Target", pretty_target_name(target_key))
        main_tables.append(out)
if main_tables:
    pd.concat(main_tables, ignore_index=True).to_csv(TAB_DIR / "table2_main_2025_transport_results.csv", index=False)

ablation_tables = []
for target_key, df in ablation.items():
    if df is not None and not df.empty:
        out = df[["model", "ROC_AUC", "PR_AUC", "F1", "Recall", "Precision", "train_years", "n_train"]].copy()
        out["model"] = out["model"].map(pretty_model_name)
        out.insert(0, "Target", pretty_target_name(target_key))
        ablation_tables.append(out)
if ablation_tables:
    pd.concat(ablation_tables, ignore_index=True).to_csv(TAB_DIR / "table3_recency_ablation.csv", index=False)

rep_tables = []
for target_key, df in repeated.items():
    if df is not None and not df.empty:
        out = df.copy()
        out["model"] = out["model"].map(pretty_model_name)
        out.insert(0, "Target", pretty_target_name(target_key))
        rep_tables.append(out)
if rep_tables:
    pd.concat(rep_tables, ignore_index=True).to_csv(TAB_DIR / "table4_repeated_seed_summary.csv", index=False)

sub_tables = []
for target_key, df in subgroup.items():
    if df is not None and not df.empty:
        out = df.sort_values(["ROC_AUC", "PR_AUC"], ascending=False).head(8).copy()
        out.insert(0, "Target", pretty_target_name(target_key))
        sub_tables.append(out)
if sub_tables:
    pd.concat(sub_tables, ignore_index=True).to_csv(TAB_DIR / "table5_top_subgroup_screen.csv", index=False)

# ============================================================
# FIGURE 1: PREVALENCE ACROSS WAVES
# ============================================================
if target_summary is not None:
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.plot(
        target_summary["survey_year"], target_summary["target_k30"],
        marker="o", linewidth=2.6, markersize=8, color=outcome_color("Obstacle"), label="Obstacle"
    )
    ax.plot(
        target_summary["survey_year"], target_summary["target_formal_credit"],
        marker="o", linewidth=2.6, markersize=8, color=outcome_color("Formal credit"), label="Formal credit"
    )
    ax.plot(
        target_summary["survey_year"], target_summary["target_applied_credit"],
        marker="o", linewidth=2.6, markersize=8, color=outcome_color("Credit apply"), label="Credit apply"
    )
    ax.set_xlabel("Survey year")
    ax.set_ylabel("Rate")
    ax.set_xticks(target_summary["survey_year"])
    clean_axes(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save_figure(fig, "figure1_target_prevalence_by_wave_color_v2")

# ============================================================
# FIGURE 2: ALL TARGETS, 2025 ROC AUC
# ============================================================
plot_rows = []
for target_key, df in transport.items():
    if df is not None:
        tmp = df.copy()
        tmp["target"] = pretty_target_name(target_key)
        plot_rows.append(tmp)

if plot_rows:
    roc_df = pd.concat(plot_rows, ignore_index=True)
    models = [
        "Core_Logistic_balanced",
        "Core_Threshold_Logistic_balanced",
        "Rich_Logistic_balanced",
        "Rich_Logistic_recentWeighted",
        "Rich_XGB_recentWeighted",
        "Stacked_Meta_Logistic",
    ]
    roc_df = roc_df[roc_df["model"].isin(models)].copy()
    targets = ["Severe financing obstacle", "Formal credit access", "Credit application"]

    x = np.arange(len(targets))
    width = 0.12

    fig, ax = plt.subplots(figsize=(10.6, 5.1))
    for i, model in enumerate(models):
        vals = []
        for target in targets:
            sub = roc_df[(roc_df["target"] == target) & (roc_df["model"] == model)]
            vals.append(sub["ROC_AUC"].iloc[0] if len(sub) else np.nan)
        ax.bar(
            x + (i - (len(models)-1)/2) * width,
            vals,
            width=width,
            label=pretty_model_name(model),
            color=model_color(model),
            edgecolor="white",
            linewidth=0.4
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["Obstacle", "Formal credit", "Credit apply"])
    ax.set_ylabel("ROC AUC")
    clean_axes(ax)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    save_figure(fig, "figure2_2025_transport_rocauc_by_target_model_color_v2")

# ============================================================
# FIGURE 3A AND 3B: OBJECTIVE TARGETS
# ============================================================
objective_targets = ["target_formal_credit", "target_applied_credit"]
obj_rows = []
for target_key in objective_targets:
    df = transport.get(target_key)
    if df is not None:
        tmp = df.copy()
        tmp["target_key"] = target_key
        obj_rows.append(tmp)

if obj_rows:
    obj_df = pd.concat(obj_rows, ignore_index=True)
    chosen_models = [
        "Rich_Logistic_balanced",
        "Rich_Logistic_recentWeighted",
        "Stacked_Meta_Logistic",
        "Core_Logistic_balanced",
        "Rich_XGB_recentWeighted",
    ]

    for metric, stem in [
        ("ROC_AUC", "figure3a_objective_targets_rocauc_color_v2"),
        ("PR_AUC", "figure3b_objective_targets_prauc_color_v2"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharex=False)
        for ax, target_key in zip(axes, objective_targets):
            sub = obj_df[(obj_df["target_key"] == target_key) & (obj_df["model"].isin(chosen_models))].copy()
            sub = sub.sort_values(metric, ascending=True)

            y = np.arange(len(sub))
            colors = [objective_bar_color(m) for m in sub["model"]]
            labels = [pretty_model_name(m) for m in sub["model"]]

            ax.barh(y, sub[metric], color=colors, edgecolor="white", linewidth=0.4)
            ax.set_yticks(y)
            ax.set_yticklabels(wrap_labels(labels, 16))
            ax.set_xlabel(metric.replace("_", " "))
            clean_axes(ax)
            ax.text(0.02, 1.02, pretty_target_name(target_key), transform=ax.transAxes, fontsize=11)

        save_figure(fig, stem)

# ============================================================
# FIGURE 4: RECENCY ABLATION
# ============================================================
for target_key in objective_targets:
    df = ablation.get(target_key)
    if df is not None and not df.empty:
        plot_df = df.copy()
        plot_df["label"] = plot_df["model"].map(pretty_model_name)
        plot_df = plot_df.sort_values("ROC_AUC", ascending=True)

        fig, ax = plt.subplots(figsize=(7.3, 4.5))
        y = np.arange(len(plot_df))
        bar_color = COLORS["orange"] if target_key == "target_formal_credit" else COLORS["teal"]
        ax.barh(y, plot_df["ROC_AUC"], color=bar_color, alpha=0.78, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(plot_df["label"])
        ax.set_xlabel("ROC AUC")
        clean_axes(ax)
        ax.text(0.02, 1.02, pretty_target_name(target_key), transform=ax.transAxes, fontsize=11)
        save_figure(fig, f"figure4_recency_ablation_{target_key}_color_v2")

# ============================================================
# FIGURE 5: REPEATED-SEED SUMMARY
# ============================================================
rep_obj = []
for target_key in objective_targets:
    df = repeated.get(target_key)
    if df is not None and not df.empty:
        tmp = df.copy()
        tmp["target"] = pretty_target_name(target_key)
        rep_obj.append(tmp)

if rep_obj:
    rep_obj_df = pd.concat(rep_obj, ignore_index=True)
    keep_models = ["Core_Logistic_balanced", "Rich_Logistic_balanced", "Rich_Logistic_recentWeighted"]
    rep_obj_df = rep_obj_df[rep_obj_df["model"].isin(keep_models)].copy()

    labels = []
    means = []
    errs = []
    colors = []
    for _, row in rep_obj_df.iterrows():
        lab = "Formal credit" if row["target"] == "Formal credit access" else "Credit apply"
        labels.append(f"{lab} | {pretty_model_name(row['model'])}")
        means.append(row["ROC_AUC_mean"])
        errs.append(row["ROC_AUC_std"])
        colors.append(objective_bar_color(row["model"]))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    y = np.arange(len(labels))
    ax.barh(y, means, xerr=errs, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(wrap_labels(labels, 20))
    ax.set_xlabel("Mean ROC AUC")
    clean_axes(ax)
    save_figure(fig, "figure5_repeated_seed_objective_targets_color_v2")

# ============================================================
# FIGURE 6: TOP SUBGROUPS
# ============================================================
for target_key in objective_targets:
    df = subgroup.get(target_key)
    if df is not None and not df.empty:
        plot_df = df.sort_values("ROC_AUC", ascending=True).head(8).copy()
        labels = plot_df["group_value"].astype(str)
        bar_color = subgroup_color(target_key)

        fig, ax = plt.subplots(figsize=(7.0, 4.3))
        y = np.arange(len(plot_df))
        ax.barh(y, plot_df["ROC_AUC"], color=bar_color, alpha=0.60, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("ROC AUC")
        clean_axes(ax)
        ax.text(0.02, 1.02, pretty_target_name(target_key), transform=ax.transAxes, fontsize=11)
        save_figure(fig, f"figure6_top_subgroups_{target_key}_color_v2")

# ============================================================
# CAPTION STUBS
# ============================================================
captions = [
    {"file": "figure1_target_prevalence_by_wave_color_v2", "caption": "Prevalence of the three study outcomes across Egypt Enterprise Survey waves."},
    {"file": "figure2_2025_transport_rocauc_by_target_model_color_v2", "caption": "Comparison of 2025 holdout ROC AUC across targets and model classes."},
    {"file": "figure3a_objective_targets_rocauc_color_v2", "caption": "ROC AUC for the objective outcomes under competing model specifications."},
    {"file": "figure3b_objective_targets_prauc_color_v2", "caption": "PR AUC for the objective outcomes under competing model specifications."},
    {"file": "figure4_recency_ablation_target_formal_credit_color_v2", "caption": "Recency ablation for formal credit access."},
    {"file": "figure4_recency_ablation_target_applied_credit_color_v2", "caption": "Recency ablation for credit application."},
    {"file": "figure5_repeated_seed_objective_targets_color_v2", "caption": "Repeated-seed summary for the objective outcomes."},
    {"file": "figure6_top_subgroups_target_formal_credit_color_v2", "caption": "Top sector subgroups for formal credit access."},
    {"file": "figure6_top_subgroups_target_applied_credit_color_v2", "caption": "Top sector subgroups for credit application."},
]
pd.DataFrame(captions).to_csv(TAB_DIR / "figure_caption_stubs_color_v2.csv", index=False)

print("\nDone.")
print(f"Figures saved in: {FIG_DIR.resolve()}")
print(f"Tables saved in: {TAB_DIR.resolve()}")
