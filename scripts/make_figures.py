"""Generate research-grade figures from artifacts/results/ into docs/figures/.

All numbers and curves are read exclusively from evaluation.json artifacts.
Never hand-edited or hard-coded.
Color palette: Okabe-Ito (colorblind safe)
- Truth: #000000 (black)
- ARX: #0072B2 (blue)
- Prony: #E69F00 (orange)
- Neural ODE: #D55E00 (vermillion)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Spec Section 11 & Decision D13 Okabe-Ito palette
COLORS = {
    "truth": "#000000",
    "arx": "#0072B2",
    "prony": "#E69F00",
    "neural_ode": "#D55E00",
}


def plot_bode(
    plant_id: str,
    data: dict,
    output_dir: Path,
) -> None:
    """Generate Bode magnitude and phase plot with shaded training band."""
    freq_data = data["frequency_response"]
    omega = np.array(freq_data["omega"])

    mag_true = np.array(freq_data["mag_true"])
    phase_true = np.array(freq_data["phase_true"])

    mag_arx = np.array(freq_data["mag_arx"])
    phase_arx = np.array(freq_data["phase_arx"])

    mag_prony = np.array(freq_data["mag_prony"])
    phase_prony = np.array(freq_data["phase_prony"])

    mag_node = np.array(freq_data["mag_neural_ode"])
    phase_node = np.array(freq_data["phase_neural_ode"])

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=300)

    # Shaded excitation band [0.1, 15] rad/s
    ax_mag.axvspan(0.1, 15.0, color="#E0E0E0", alpha=0.5, label="Training band")
    ax_phase.axvspan(0.1, 15.0, color="#E0E0E0", alpha=0.5, label="Training band")

    # Magnitude
    db_true = 20.0 * np.log10(np.maximum(mag_true, 1e-12))
    db_arx = 20.0 * np.log10(np.maximum(mag_arx, 1e-12))
    db_prony = 20.0 * np.log10(np.maximum(mag_prony, 1e-12))
    db_node = 20.0 * np.log10(np.maximum(mag_node, 1e-12))

    ax_mag.semilogx(omega, db_true, color=COLORS["truth"], label="Truth", linewidth=2.0)
    ax_mag.semilogx(
        omega, db_arx, color=COLORS["arx"], label="ARX", linestyle="--", linewidth=1.5
    )
    ax_mag.semilogx(
        omega,
        db_prony,
        color=COLORS["prony"],
        label="Prony",
        linestyle="-.",
        linewidth=1.5,
    )
    ax_mag.semilogx(
        omega,
        db_node,
        color=COLORS["neural_ode"],
        label="Neural ODE",
        linestyle=":",
        linewidth=2.0,
    )

    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True, which="both", linestyle=":", alpha=0.6)
    ax_mag.legend(loc="best")
    ax_mag.set_title(f"Frequency Response: {plant_id.upper()}")

    # Phase
    ax_phase.semilogx(
        omega, phase_true, color=COLORS["truth"], label="Truth", linewidth=2.0
    )
    ax_phase.semilogx(
        omega,
        phase_arx,
        color=COLORS["arx"],
        label="ARX",
        linestyle="--",
        linewidth=1.5,
    )
    ax_phase.semilogx(
        omega,
        phase_prony,
        color=COLORS["prony"],
        label="Prony",
        linestyle="-.",
        linewidth=1.5,
    )
    ax_phase.semilogx(
        omega,
        phase_node,
        color=COLORS["neural_ode"],
        label="Neural ODE",
        linestyle=":",
        linewidth=2.0,
    )

    ax_phase.set_xlabel("Frequency [rad/s]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, which="both", linestyle=":", alpha=0.6)

    fig.tight_layout()
    out_file = output_dir / f"bode_{plant_id}.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file)


def plot_impulse(
    plant_id: str,
    data: dict,
    output_dir: Path,
) -> None:
    """Generate impulse response overlay plot."""
    imp_data = data["impulse_response"]
    t = np.array(imp_data["t"])
    h_true = np.array(imp_data["h_true"])
    h_arx = np.array(imp_data["h_arx"])
    h_prony = np.array(imp_data["h_prony"])
    h_node = np.array(imp_data["h_neural_ode"])

    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.plot(t, h_true, color=COLORS["truth"], label="Truth", linewidth=2.0)
    ax.plot(t, h_arx, color=COLORS["arx"], label="ARX", linestyle="--", linewidth=1.5)
    ax.plot(
        t,
        h_prony,
        color=COLORS["prony"],
        label="Prony",
        linestyle="-.",
        linewidth=1.5,
    )
    ax.plot(
        t,
        h_node,
        color=COLORS["neural_ode"],
        label="Neural ODE",
        linestyle=":",
        linewidth=2.0,
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Impulse Response h(t)")
    ax.set_title(f"Impulse Response: {plant_id.upper()}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best")

    fig.tight_layout()
    out_file = output_dir / f"impulse_{plant_id}.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file)


def plot_ood_bar(
    plant_id: str,
    data: dict,
    output_dir: Path,
) -> None:
    """Generate OOD generalization bar chart across splits."""
    ood_data = data["ood"]
    splits = ["ood_amplitude", "ood_frequency", "ood_waveform"]
    labels = ["Amplitude (x2)", "Frequency (x2.5)", "Waveforms (Sq/Chirp)"]

    arx_fits = [ood_data[s]["arx"]["fit_percent"] for s in splits]
    prony_fits = [ood_data[s]["prony"]["fit_percent"] for s in splits]
    node_fits = [ood_data[s]["neural_ode"]["fit_percent"] for s in splits]

    x = np.arange(len(splits))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.bar(x - width, arx_fits, width, label="ARX", color=COLORS["arx"])
    ax.bar(x, prony_fits, width, label="Prony", color=COLORS["prony"])
    ax.bar(x + width, node_fits, width, label="Neural ODE", color=COLORS["neural_ode"])

    ax.set_ylabel("Goodness of Fit [%]")
    ax.set_title(f"Out-of-Distribution Generalization: {plant_id.upper()}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="best")
    ax.set_ylim(bottom=float(min(min(arx_fits + prony_fits + node_fits) - 10.0, 0.0)), top=105.0)

    fig.tight_layout()
    out_file = output_dir / f"ood_{plant_id}.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file)


def plot_noise_sweep(
    data: dict,
    output_dir: Path,
) -> None:
    """Generate fit versus SNR curve for mass-spring-damper."""
    noise_data = data["noise_sweep"]
    snr_keys = ["inf", "40", "30", "20", "10"]
    x_positions = np.arange(len(snr_keys))

    arx_fits = [noise_data[k]["arx"]["fit_percent"] for k in snr_keys]
    prony_fits = [noise_data[k]["prony"]["fit_percent"] for k in snr_keys]

    node_snr_keys = ["inf", "40", "20"]
    node_x = [snr_keys.index(k) for k in node_snr_keys]
    node_means = [
        noise_data[k]["neural_ode"]["fit_percent_mean"] for k in node_snr_keys
    ]
    node_stds = [noise_data[k]["neural_ode"]["fit_percent_std"] for k in node_snr_keys]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ax.plot(
        x_positions,
        arx_fits,
        "o-",
        color=COLORS["arx"],
        label="ARX (least squares)",
        linewidth=2,
    )
    ax.plot(
        x_positions,
        prony_fits,
        "s-",
        color=COLORS["prony"],
        label="Prony (impulse)",
        linewidth=2,
    )
    ax.errorbar(
        node_x,
        node_means,
        yerr=node_stds,
        fmt="^-",
        color=COLORS["neural_ode"],
        label="Neural ODE (mean +- std, 3 seeds)",
        linewidth=2,
        capsize=5,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Clean (inf)", "40 dB", "30 dB", "20 dB", "10 dB"])
    ax.set_xlabel("Measurement Noise Level (Output SNR)")
    ax.set_ylabel("Goodness of Fit on Clean Test IID [%]")
    ax.set_title("Noise Robustness Sweep: Mass-Spring-Damper (MSD)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")

    fig.tight_layout()
    out_file = output_dir / "noise_sweep_msd.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file)


def plot_param_variation(
    data: dict,
    output_dir: Path,
) -> None:
    """Generate parameter-variation error box plot across 10 perturbed MSD plants."""
    p_data = data["param_variation"]
    if p_data is None:
        return

    arx_errs = p_data["arx_errors_percent"]
    prony_errs = p_data["prony_errors_percent"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    box = ax.boxplot(
        [arx_errs, prony_errs],
        tick_labels=["ARX", "Prony"],
        patch_artist=True,
        widths=0.4,
    )

    colors = [COLORS["arx"], COLORS["prony"]]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(2)

    ax.set_ylabel("Mean Parameter Error [%]")
    ax.set_title("Parameter Variation Study (10 Perturbed MSD Plants)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    out_file = output_dir / "param_variation_msd.png"
    fig.savefig(out_file)
    plt.close(fig)
    logger.info("Saved %s", out_file)


def main() -> None:
    """Produce all research figures from evaluation.json artifacts."""
    parser = argparse.ArgumentParser(description="Generate research figures.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/results"),
        help="Results directory with evaluation.json (default: artifacts/results/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Figures output directory (default: docs/figures/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plants = ["rc", "motor", "msd"]
    for plant_id in plants:
        eval_path = args.results_dir / plant_id / "evaluation.json"
        if not eval_path.exists():
            logger.warning("File not found: %s", eval_path)
            continue

        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)

        plot_bode(plant_id, data, args.output_dir)
        plot_impulse(plant_id, data, args.output_dir)
        plot_ood_bar(plant_id, data, args.output_dir)

        if plant_id == "msd":
            plot_noise_sweep(data, args.output_dir)
            plot_param_variation(data, args.output_dir)


if __name__ == "__main__":
    main()
