"""Cross-species figure, four arms.

    python3 make_fig.py            # 1x4, full text width (figure*)
    python3 make_fig.py --1col     # 2x2, one ICASSP column (figure)

`xspecies_all.json` carries both denominators. The paper prints
octave_correct_pop -- over every clip attempted at that k, so a tracker that
returns no estimate is charged for it, as in the elephant robustness table.
Set KEY = "octave_correct" for the answered denominator.
"""
import argparse, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY = "octave_correct_pop"
d = json.load(open("results/xspecies_all.json"))["summary"]

C = {"comb": "#2a78d6", "comb_noexempt": "#eb6834",
     "pyin": "#1baf7a", "crepe": "#8250c4"}
M = {"comb": ("o", "-"), "comb_noexempt": ("s", "--"),
     "pyin": ("^", ":"), "crepe": ("D", "-.")}
LBL = {"comb": "comb + exemption", "comb_noexempt": "comb, no exemption",
       "pyin": "pYIN", "crepe": "CREPE"}
ARMS = ("comb", "comb_noexempt", "pyin", "crepe")

# ordered by resolved harmonics in band -- the explanatory variable
PANELS = [
    ("long-billed_hermits", "long-billed hermits", "4.5 kHz",
     "4.7 harmonics, 8 peaks/frame", "4.7 harm., 8 peaks/fr."),
    ("monk_parakeets", "monk parakeets", "1.8 kHz",
     "11.9 harmonics, 17 peaks/frame", "11.9 harm., 17 peaks/fr."),
    ("lions", "lions", "150 Hz",
     "27 harmonics, 27 peaks/frame", "27 harm., 27 peaks/fr."),
    ("spotted_hyenas", "spotted hyenas", "312 Hz",
     "27 harmonics, 40 peaks (saturated)", "27 harm., 40 peaks (sat.)"),
]


def style(ax, fs, ks=(0, 2, 4, 6)):
    """``ks`` is the species' own sweep, so each panel spans only the range it
    was measured over. Hermits admit k <= 3 alone: at k = 4 the high-pass
    cutoff (k + 1/2) F0 would exceed Nyquist and remove_low_harmonics returns
    nothing, so drawing that panel out to 6 would imply a measurement that
    cannot exist."""
    ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100])
    lo, hi = min(ks), max(ks)
    pad = 0.055 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad); ax.set_xticks(list(ks))
    ax.grid(axis="y", color="#d8d7d2", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8a8984")
    ax.tick_params(colors="#52514e", length=2.2, labelsize=fs)


def wide():
    plt.rcParams.update({"font.size": 7.2, "font.family": "serif",
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6})
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), sharey=True)
    for ax, (sp, name, f0, note, _s) in zip(axes, PANELS):
        for arm in ARMS:
            ks = sorted(int(k) for k in d[sp][arm])
            y = [d[sp][arm][str(k)][KEY] * 100 for k in ks]
            mk, ls = M[arm]
            ax.plot(ks, y, ls, color=C[arm], marker=mk, markersize=3.4,
                    linewidth=1.3, markeredgecolor="white",
                    markeredgewidth=0.5, clip_on=False, zorder=3)
        style(ax, 7.0, ks)
        ax.set_title(f"{name}  ({f0})", fontsize=7.4, color="#0b0b0b", pad=8)
        ax.text(0.5, 1.005, note, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.1, color="#52514e")
        ax.set_xlabel("$k$ harmonics removed", fontsize=7, color="#52514e")
    axes[0].set_ylabel("octave-correct (%)", fontsize=7, color="#52514e")
    lab = {"comb": (3.55, 92, "comb + exemption"),
           "comb_noexempt": (3.55, 78, "no exemption"),
           "pyin": (3.55, 64, "pYIN"), "crepe": (3.55, 50, "CREPE")}
    for arm, (x, y, t) in lab.items():
        axes[0].plot([x - 0.38], [y], marker=M[arm][0], color=C[arm],
                     markersize=3.4, markeredgecolor="white",
                     markeredgewidth=0.5, clip_on=False)
        axes[0].text(x, y, t, fontsize=5.8, color=C[arm], va="center",
                     ha="left")
    handles = [plt.Line2D([], [], color=C[a], marker=M[a][0], linestyle=M[a][1],
                          markersize=3.4, linewidth=1.3, label=LBL[a])
               for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, -0.035), handlelength=2.4,
               columnspacing=1.6)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.80, bottom=0.30,
                        wspace=0.14)
    return fig, "fig_xspecies"


def one_col():
    """2x2 at 8.6 cm. Everything shrinks; the panel notes are abbreviated and
    the direct labels go, since at 4 cm per panel there is no clear space to
    put them. The legend carries identity instead, and marker shape and dash
    pattern still distinguish the arms in greyscale."""
    plt.rcParams.update({"font.size": 6.0, "font.family": "serif",
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5})
    fig, axes = plt.subplots(2, 2, figsize=(3.39, 3.02), sharey=True)
    for ax, (sp, name, f0, _n, short) in zip(axes.ravel(), PANELS):
        for arm in ARMS:
            ks = sorted(int(k) for k in d[sp][arm])
            y = [d[sp][arm][str(k)][KEY] * 100 for k in ks]
            mk, ls = M[arm]
            ax.plot(ks, y, ls, color=C[arm], marker=mk, markersize=2.6,
                    linewidth=1.0, markeredgecolor="white",
                    markeredgewidth=0.4, clip_on=False, zorder=3)
        style(ax, 5.6, ks)
        ax.set_title(f"{name}  ({f0})", fontsize=6.2, color="#0b0b0b", pad=7)
        ax.text(0.5, 1.005, short, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=5.0, color="#52514e")
    # Every panel is labelled and keeps its own ticks: the hermit sweep is
    # k = 0..3 and the others 0..6, so the top row cannot borrow the bottom
    # row's tick labels.
    for ax in axes.ravel():
        ax.set_xlabel("$k$ harmonics removed", fontsize=5.8, color="#52514e")
    for ax in axes[:, 0]:
        ax.set_ylabel("octave-correct (%)", fontsize=5.8, color="#52514e")
    handles = [plt.Line2D([], [], color=C[a], marker=M[a][0], linestyle=M[a][1],
                          markersize=2.6, linewidth=1.0, label=LBL[a])
               for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=5.8, bbox_to_anchor=(0.5, 0.004), handlelength=2.0,
               columnspacing=1.2, labelspacing=0.28)
    fig.subplots_adjust(left=0.128, right=0.985, top=0.895, bottom=0.200,
                        wspace=0.13, hspace=0.62)
    return fig, "fig_xspecies_1col"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--1col", dest="one", action="store_true")
    a = ap.parse_args()
    fig, stem = one_col() if a.one else wide()
    fig.savefig(stem + ".pdf", metadata={"CreationDate": None})
    fig.savefig(stem + ".png", dpi=260)
    print("wrote %s.pdf / .png  (denominator: %s)" % (stem, KEY))
