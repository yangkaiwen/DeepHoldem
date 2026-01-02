import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re
import os


def plot_combined(roi_csv_path, loss_csv_path):
    # --- Load ROI Data ---
    df_roi = None
    if os.path.exists(roi_csv_path):
        try:
            df_roi = pd.read_csv(roi_csv_path)

            # Extract iteration
            def extract_iteration(model_str):
                match = re.search(r"(\d+)", str(model_str))
                if match:
                    return int(match.group(1))
                return 0

            df_roi["Iteration"] = df_roi["Model"].apply(extract_iteration)
            df_roi = df_roi.sort_values("Iteration")
        except Exception as e:
            print(f"Error reading ROI CSV: {e}")
    else:
        print(f"File not found: {roi_csv_path}")

    # --- Load Loss Data ---
    df_loss = None
    if os.path.exists(loss_csv_path):
        try:
            df_loss = pd.read_csv(loss_csv_path)
            df_loss = df_loss.sort_values("step")
        except Exception as e:
            print(f"Error reading Loss CSV: {e}")
    else:
        print(f"File not found: {loss_csv_path}")

    # --- Setup Plot ---
    fig = plt.figure(figsize=(20, 12))
    # 6 rows, 4 columns
    # Left panel uses columns 0-1 (width 2)
    # Right panel uses columns 2-3 (width 2)
    gs = gridspec.GridSpec(6, 4, figure=fig)

    # Left Panel: 3x1 (ROI)
    # Each plot takes 2 rows height
    ax_roi_1 = fig.add_subplot(gs[0:2, 0:2])
    ax_roi_2 = fig.add_subplot(gs[2:4, 0:2], sharex=ax_roi_1)
    ax_roi_3 = fig.add_subplot(gs[4:6, 0:2], sharex=ax_roi_1)
    roi_axes = [ax_roi_1, ax_roi_2, ax_roi_3]

    # Right Panel: 3x1 (Losses)
    # Each plot takes 2 rows height, 2 columns width (spanning columns 2 and 3)
    ax_loss_1 = fig.add_subplot(gs[0:2, 2:4])  # Top
    ax_loss_2 = fig.add_subplot(gs[2:4, 2:4], sharex=ax_loss_1)  # Middle
    ax_loss_3 = fig.add_subplot(gs[4:6, 2:4], sharex=ax_loss_1)  # Bottom
    loss_axes = [ax_loss_1, ax_loss_2, ax_loss_3]

    # --- Plot ROI ---
    if df_roi is not None:
        metrics = ["Main_Mean_ROI", "Main_BB_100", "Main_Cum_NROI"]
        for i, metric in enumerate(metrics):
            ax = roi_axes[i]
            if metric in df_roi.columns:
                ax.plot(
                    df_roi["Iteration"],
                    df_roi[metric],
                    marker="o",
                    linestyle="-",
                    color="b",
                    label=metric,
                )
                ax.set_title(f"{metric} vs Iterations")
                ax.set_ylabel(metric)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)

                # Trend line
                x = df_roi["Iteration"]
                y = df_roi[metric]
                mask = ~np.isnan(y)
                x_clean = x[mask]
                y_clean = y[mask]
                if len(x_clean) > 1:
                    z = np.polyfit(x_clean, y_clean, 1)
                    p = np.poly1d(z)
                    ax.plot(
                        x_clean,
                        p(x_clean),
                        "r--",
                        label=f"Trend (y={z[0]:.2e}x + {z[1]:.2f})",
                    )
                ax.legend()
        roi_axes[-1].set_xlabel("Training Iterations")

    # --- Plot Losses ---
    if df_loss is not None:
        loss_metrics = ["policy_loss", "value_loss", "entropy_loss"]
        for i, metric in enumerate(loss_metrics):
            if i < len(loss_axes):
                ax = loss_axes[i]
                if metric in df_loss.columns:
                    ax.plot(
                        df_loss["step"],
                        df_loss[metric],
                        color="g",
                        alpha=0.5,
                        label="Raw",
                    )
                    ax.set_title(f"{metric}")
                    if i == 2:  # Bottom row
                        ax.set_xlabel("Steps")
                    ax.set_ylabel(metric)
                    ax.grid(True, linestyle="--", alpha=0.5)

                    # Moving average
                    window = 50
                    if len(df_loss) > window:
                        ma = df_loss[metric].rolling(window=window).mean()
                        ax.plot(
                            df_loss["step"],
                            ma,
                            color="darkgreen",
                            linewidth=2,
                            label=f"MA({window})",
                        )
                        ax.legend()

    plt.tight_layout()
    output_file = "combined_plot.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.show()


if __name__ == "__main__":
    plot_combined("evaluation_vs_random.csv", "log/training_losses.csv")
