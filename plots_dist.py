import matplotlib.pyplot as plt
import numpy as np

BNP_GREEN, GREY, DARK = "#00915A", "#7E8C94", "#006748"


def _panel(ax, y_true, y_pred, title, bins, clip):
    """One overlaid histogram of true vs estimated WER."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    beyond = 0

    upper = min(clip, max(y_true.max(), y_pred.max()) * 1.05)
    beyond = int((y_true > upper).sum() + (y_pred > upper).sum())
    edges = np.linspace(0, upper, bins + 1)
    ax.hist(np.clip(y_true, 0, upper), bins=edges, color=GREY, alpha=0.75,
            label=f"True  (mean {y_true.mean():.2f}, sd {y_true.std():.2f})")
    ax.hist(np.clip(y_pred, 0, upper), bins=edges, histtype="step", lw=2.2,
            color=BNP_GREEN,
            label=f"Estimated  (mean {y_pred.mean():.2f}, sd {y_pred.std():.2f})")

    ax.axvline(y_true.mean(), color=DARK, lw=1, ls="--")
    ax.axvline(y_pred.mean(), color=BNP_GREEN, lw=1, ls="--")

    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_xlim(0, upper)
    ax.set_xlabel("WER" + (f"  ({beyond} values above {upper:.1f}, folded in)"
                          if beyond else ""))
    ax.set_ylabel("Number of items")
    ax.legend(frameon=False, fontsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)


def plot_wer_distributions(result, calls, clip=1.5, bins=40, path=None):
    """
    True vs estimated WER distributions, at segment and call level.

    The gap between the two standard deviations shown in the legend is the
    shrinkage: a narrower estimated distribution means the model plays safe and
    stays near the mean instead of committing to extreme values. That is the
    visual counterpart of std_ratio in r2_decomposition.

    Values above clip are folded into the last bin so the plot stays readable
    when short segments carry unbounded WER; their count is reported on the axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    _panel(axes[0], result["y_wer"], result["pred_wer"],
           "Segment level", bins, clip)
    _panel(axes[1],
           [c["wer_true"] for c in calls], [c["wer_pred"] for c in calls],
           "Call level", max(bins // 3, 8), clip)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig