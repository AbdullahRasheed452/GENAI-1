import csv
import json
import shutil
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
DBDIR = ROOT / "results_raw" / "db"
FIG = ROOT / "report" / "figures"
TAB = ROOT / "report" / "tables"

NICE = {
    "clean": "Clean", "salt_pepper": "Salt-and-pepper", "blur": "Gaussian blur", "occlusion": "Occlusion",
    "all corrupted": "All corrupted", "none": "--", "low": "Low", "medium": "Medium", "high": "High",
    "all": "All", "identity": "Identity", "ok": "healthy", "INACTIVE": "inactive",
    "DOMINATES OTHER TYPES": "dominates other types",
}
KINDS = ("salt_pepper", "blur", "occlusion")


def nice(text):
    return NICE.get(text, text).replace("_", r"\_")


def find(name):
    hits = sorted(RESULTS.rglob(name))
    if not hits:
        raise SystemExit(f"cannot find {name} under {RESULTS}")
    return hits[0]


def read_json(name):
    return json.loads(find(name).read_text())


def fmt_psnr(v):
    if v is None:
        return "--"
    return r"100$^{*}$" if v >= 99.99 else f"{v:.2f}"


def write_table(name, lines):
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / name).write_text("\n".join(lines) + "\n")


def load_run(db_file, experiment, run_name):
    con = sqlite3.connect(DBDIR / db_file)
    rows = con.execute(
        "SELECT r.run_uuid FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id "
        "WHERE e.name = ? AND r.name = ? AND r.lifecycle_stage = 'active' ORDER BY r.start_time",
        (experiment, run_name),
    ).fetchall()
    if not rows:
        con.close()
        raise SystemExit(f"no run named {run_name} in experiment {experiment} of {db_file}")
    uuid = rows[-1][0]
    metrics = {}
    for key, step, value in con.execute(
        "SELECT key, step, value FROM metrics WHERE run_uuid = ? ORDER BY step, timestamp", (uuid,)
    ):
        metrics.setdefault(key, {})[step] = value
    params = dict(con.execute("SELECT key, value FROM params WHERE run_uuid = ?", (uuid,)).fetchall())
    con.close()
    curves = {k: (np.array(sorted(v)), np.array([v[s] for s in sorted(v)])) for k, v in metrics.items()}
    steps = max(len(v[0]) for v in curves.values())
    print(f"  {db_file} / {run_name}: {len(rows)} run(s) with this name, using the latest, {steps} logged epochs")
    return curves, params


def draw(ax, curves, key, label=None, **kw):
    x, y = curves[key]
    ax.plot(x + 1, y, label=label, **kw)


def best_marker(ax, curves, key, mode="min", start=0):
    x, y = curves[key]
    idx = np.flatnonzero(x >= start)
    pick = idx[int(np.argmin(y[idx]) if mode == "min" else np.argmax(y[idx]))]
    ax.axvline(x[pick] + 1, color="gray", linestyle=":", linewidth=1)
    ax.text(x[pick] + 1, ax.get_ylim()[1], f" best: epoch {x[pick] + 1}", va="top", ha="left", fontsize=7, color="gray")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=200)
    plt.close(fig)


def fig_task1():
    c, _ = load_run("task1_mlflow.db", "task1_universal_ae", "final_training")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    draw(ax[0], c, "train_loss")
    ax[0].set_title("Training loss (L1 and SSIM mix)")
    draw(ax[1], c, "objective")
    ax[1].set_title("Validation objective (lower is better)")
    best_marker(ax[1], c, "objective")
    draw(ax[2], c, "ssim")
    ax[2].set_title("Validation SSIM")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    save(fig, "training_task1.png")


def fig_classifier():
    c, _ = load_run("task2_classifier_mlflow.db", "task2_classifier", "final_classifier")
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.8))
    draw(ax[0], c, "train_loss", "training loss")
    draw(ax[0], c, "val_ce", "validation cross-entropy")
    draw(ax[1], c, "accuracy", "accuracy")
    draw(ax[1], c, "macro_f1", "macro F1")
    ax[1].set_ylim(0.93, 1.002)
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=7)
    save(fig, "training_classifier.png")


def fig_specialists():
    fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for j, kind in enumerate(KINDS):
        c, _ = load_run("task2_specialists_mlflow.db", f"task2_specialist_{kind}", f"final_{kind}")
        draw(ax[0, j], c, "train_loss")
        ax[0, j].set_title(f"{NICE[kind]}: training loss")
        draw(ax[1, j], c, "objective")
        ax[1, j].set_title("validation objective")
        best_marker(ax[1, j], c, "objective")
        ax[1, j].set_xlabel("epoch")
        for a in (ax[0, j], ax[1, j]):
            a.grid(alpha=0.3)
    save(fig, "training_specialists.png")


def fig_task3():
    c, p = load_run("task3_mlflow.db", "task3_soft_moe", "moe_run")
    warm = int(float(p.get("warmup_epochs", 0)))
    fig, ax = plt.subplots(2, 2, figsize=(9, 5.4))
    draw(ax[0, 0], c, "train_loss")
    ax[0, 0].set_title("Training loss")
    draw(ax[0, 1], c, "objective")
    ax[0, 1].set_title("Validation objective (lower is better)")
    best_marker(ax[0, 1], c, "objective", start=warm)
    for key, label in (("mean_weight_clean", "Identity"), ("mean_weight_salt_pepper", "Salt-and-pepper expert"),
                       ("mean_weight_blur", "Blur expert"), ("mean_weight_occlusion", "Occlusion expert")):
        draw(ax[1, 0], c, key, label)
    ax[1, 0].set_title("Mean routing weight per branch (validation)")
    ax[1, 0].legend(fontsize=7)
    draw(ax[1, 1], c, "gate_acc")
    ax[1, 1].set_title("Gate top choice equals true corruption")
    for a in ax.ravel():
        a.axvline(warm + 0.5, color="tab:red", linestyle="--", linewidth=1)
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    save(fig, "training_task3.png")


def fig_task4():
    c, _ = load_run("task4_mlflow.db", "task4_face_sketch_cgan", "cgan_run")
    fig, ax = plt.subplots(2, 2, figsize=(9, 5.4))
    draw(ax[0, 0], c, "train_d_real", "real")
    draw(ax[0, 0], c, "train_d_fake", "fake")
    ax[0, 0].set_title("Discriminator loss")
    ax[0, 0].legend(fontsize=7)
    draw(ax[0, 1], c, "train_g_adv")
    ax[0, 1].set_title("Generator adversarial loss")
    draw(ax[1, 0], c, "train_g_rec")
    ax[1, 0].set_title("Generator reconstruction loss (L1, images in [-1, 1])")
    draw(ax[1, 1], c, "val_l1", "validation L1")
    draw(ax[1, 1], c, "val_ssim", "validation SSIM")
    ax[1, 1].set_title("Validation metrics (images in [0, 1])")
    ax[1, 1].legend(fontsize=7)
    best_marker(ax[1, 1], c, "objective")
    for a in ax.ravel():
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    save(fig, "training_task4.png")


STUDIES = [
    ("Task 1", "task1_trials.csv", "task1_best.json", "min"),
    ("Task 2 classifier", "task2_classifier_trials.csv", "task2_classifier_best.json", "max"),
    ("Task 2 specialists", "task2_specialists_trials.csv", "task2_specialists_best.json", "min"),
    ("Task 3", "task3_trials.csv", "task3_best.json", "min"),
    ("Task 4", "task4_trials.csv", "task4_best.json", "min"),
]


def read_trials(name):
    with open(find(name), newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        value = r["value"]
        out.append({"number": int(r["number"]), "state": r["state"], "value": float(value) if value not in ("", None) else None})
    return out


def fig_optuna():
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.4))
    for ax, (label, csv_name, _, mode) in zip(axes.ravel(), STUDIES):
        rows = read_trials(csv_name)
        done = [(r["number"], r["value"]) for r in rows if r["state"] == "COMPLETE"]
        xs, ys = [n for n, _ in done], [v for _, v in done]
        ax.scatter(xs, ys, label="completed trial")
        best = np.minimum.accumulate(ys) if mode == "min" else np.maximum.accumulate(ys)
        ax.step(xs, best, where="post", color="tab:red", label="best so far")
        pruned = sum(1 for r in rows if r["state"] == "PRUNED")
        ax.set_title(f"{label}: {len(done)} completed, {pruned} pruned", fontsize=9)
        ax.set_xlabel("trial")
        ax.set_ylabel("validation objective" if mode == "min" else "validation macro F1")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    axes.ravel()[-1].axis("off")
    save(fig, "optuna_history.png")


def fmt_param(k, v):
    if isinstance(v, float):
        return f"{nice(k)} = {v:.3g}"
    return f"{nice(k)} = {nice(str(v))}"


def optuna_tables():
    summary = [r"\begin{tabular}{lrrrrr}", r"\toprule",
               r"Study & Trials & Completed & Pruned & Best trial & Best value \\", r"\midrule"]
    best_lines = [r"\begin{tabular}{lp{0.68\linewidth}}", r"\toprule", r"Study & Best parameters \\", r"\midrule"]
    for label, csv_name, best_name, mode in STUDIES:
        rows = read_trials(csv_name)
        best = read_json(best_name)
        done = [r for r in rows if r["state"] == "COMPLETE"]
        pruned = sum(1 for r in rows if r["state"] == "PRUNED")
        number = next(r["number"] for r in done if abs(r["value"] - best["value"]) < 1e-12)
        summary.append(f"{label} & {len(rows)} & {len(done)} & {pruned} & {number} & {best['value']:.4f} \\\\")
        best_lines.append(f"{label} & " + ", ".join(fmt_param(k, v) for k, v in best["params"].items()) + r" \\")
    write_table("optuna_summary.tex", summary + [r"\bottomrule", r"\end{tabular}"])
    write_table("optuna_best_params.tex", best_lines + [r"\bottomrule", r"\end{tabular}"])


def restoration_table(rows, name):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Condition & Severity & PSNR & SSIM & In. PSNR & In. SSIM \\", r"\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["condition"] != prev:
            lines.append(r"\midrule")
        prev = r["condition"]
        lines.append(f"{nice(r['condition'])} & {nice(r['severity'])} & {fmt_psnr(r['psnr'])} & {r['ssim']:.3f} & "
                     f"{fmt_psnr(r['input_psnr'])} & {r['input_ssim']:.3f} \\\\")
    write_table(name, lines + [r"\bottomrule", r"\end{tabular}"])


def summary_table(t1, t2, t3):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Condition & Input & Task 1 & Task 2 & Task 3 \\", r"\midrule"]
    for cond in ("clean", "salt_pepper", "blur", "occlusion", "all corrupted"):
        sev = "none" if cond == "clean" else "all"

        def pick(rows):
            return next(r for r in rows if r["condition"] == cond and r["severity"] == sev)

        def cell(r):
            return f"{fmt_psnr(r['psnr'])} / {r['ssim']:.3f}"

        a = pick(t1)
        inp = "-- / 1.000" if cond == "clean" else f"{fmt_psnr(a['input_psnr'])} / {a['input_ssim']:.3f}"
        lines.append(f"{nice(cond)} & {inp} & {cell(a)} & {cell(pick(t2))} & {cell(pick(t3))} \\\\")
    write_table("results_summary.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def classifier_tables():
    r = read_json("classifier_results.json")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule", r"Class & Precision & Recall & F1 \\", r"\midrule"]
    for name in ("clean", "salt_pepper", "blur", "occlusion"):
        m = r["per_class"][name]
        lines.append(f"{nice(name)} & {m['precision']:.4f} & {m['recall']:.4f} & {m['f1']:.4f} \\\\")
    lines += [r"\midrule",
              f"Macro average & {r['macro_precision']:.4f} & {r['macro_recall']:.4f} & {r['macro_f1']:.4f} \\\\",
              f"Accuracy & & & {r['accuracy']:.4f} \\\\", r"\bottomrule", r"\end{tabular}"]
    write_table("classifier_results.tex", lines)
    sev = [r"\begin{tabular}{lrrr}", r"\toprule", r"Corruption & Low & Medium & High \\", r"\midrule"]
    for kind in KINDS:
        v = [r["recall_by_severity"][f"{kind}_{s}"] for s in ("low", "medium", "high")]
        sev.append(f"{nice(kind)} & {v[0]:.4f} & {v[1]:.4f} & {v[2]:.4f} \\\\")
    write_table("classifier_severity.tex", sev + [r"\bottomrule", r"\end{tabular}"])


def routing_tables():
    r = read_json("task3_routing_weights.json")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"True condition & Severity & Identity & Salt-and-pepper & Blur & Occlusion \\", r"\midrule"]
    prev = None
    for g in r["groups"]:
        if prev is not None and g["condition"] != prev:
            lines.append(r"\midrule")
        prev = g["condition"]
        w = g["mean_weights"]
        lines.append(f"{nice(g['condition'])} & {nice(g['severity'])} & {w[0]:.3f} & {w[1]:.3f} & {w[2]:.3f} & {w[3]:.3f} \\\\")
    write_table("routing_weights.tex", lines + [r"\bottomrule", r"\end{tabular}"])
    health = [r"\begin{tabular}{lrrrl}", r"\toprule",
              r"Branch & Mean weight & Top choice & Weight on other types & Status \\", r"\midrule"]
    for name, h in r["expert_health"].items():
        health.append(f"{nice(name)} & {h['mean_weight']:.3f} & {h['top_choice_share']:.3f} & "
                      f"{h['mean_weight_on_other_types']:.3f} & {nice(h['flag'])} \\\\")
    write_table("expert_health.tex", health + [r"\bottomrule", r"\end{tabular}"])


def cgan_table():
    rows = read_json("cgan_results.json")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule", r"Group & Pairs & L1 & SSIM & PSNR \\", r"\midrule"]
    for r in rows:
        lines.append(f"{nice(r['group'])} & {r['n']} & {r['l1']:.4f} & {r['ssim']:.3f} & {r['psnr']:.2f} \\\\")
    write_table("cgan_results.tex", lines + [r"\bottomrule", r"\end{tabular}"])


def tables():
    t1 = read_json("task1_results.json")
    t2o = read_json("routed_oracle_results.json")
    t2p = read_json("routed_predicted_results.json")
    t3 = read_json("task3_results.json")
    restoration_table(t1, "results_task1.tex")
    restoration_table(t2o, "results_task2_oracle.tex")
    restoration_table(t2p, "results_task2_predicted.tex")
    restoration_table(t3, "results_task3.tex")
    summary_table(t1, t2p, t3)
    classifier_tables()
    routing_tables()
    cgan_table()
    optuna_tables()


COPY = [
    "classifier_confusion_matrix.png", "task1_examples_1.png", "task1_examples_2.png", "task1_failures.png",
    "routed_examples_1.png", "routed_examples_2.png", "routed_failures.png", "task3_examples_1.png",
    "task3_examples_2.png", "task3_failures.png", "task3_weight_heatmap.png", "task3_weight_histograms.png",
    "task3_dominant_cases.png", "task3_spread_cases.png", "cgan_examples.png", "cgan_failures.png",
    "cgan_style_control.png",
]


def copy_figures():
    FIG.mkdir(parents=True, exist_ok=True)
    for name in COPY:
        shutil.copy(find(name), FIG / name)
    images = [Image.open(find(f"samples_epoch_{e:03d}.png")).convert("RGB") for e in (10, 30, 60)]
    montage = Image.new("RGB", (images[0].width, sum(i.height for i in images)))
    top = 0
    for img in images:
        montage.paste(img, (0, top))
        top += img.height
    montage.save(FIG / "task4_samples_over_training.png")


def main():
    for fn in (fig_task1, fig_classifier, fig_specialists, fig_task3, fig_task4, fig_optuna, tables, copy_figures):
        try:
            fn()
            print("ok     ", fn.__name__)
        except BaseException as error:
            print("FAILED ", fn.__name__, "->", error)
    print("\nfigures:", len(list(FIG.glob("*.png"))), " tables:", len(list(TAB.glob("*.tex"))))


if __name__ == "__main__":
    main()
