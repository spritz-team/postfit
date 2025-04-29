import os
from copy import deepcopy

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from utils import cmap_pastel, darker_color, get_analysis_dict
import argparse

np.seterr(divide="ignore", invalid="ignore")
d = deepcopy(hep.style.CMS)


parser = argparse.ArgumentParser(
    description="Plot post-fit shapes for VBFZ EFT analysis",
)

parser.add_argument(
    "-p",
    "--an_path",
    type=str,
    help="Path to the analysis configuration file",
)

parser.add_argument(
    "-f",
    "--fit_path",
    type=str,
    help="Path to the fit_diagnostics file",
)

parser.add_argument(
    "-ops",
    "--ops",
    help="Operators to plot",
    nargs="+",
)

parser.add_argument(
    "-prefit",
    help="Do prefit, default false",
    action="store_true",
)

parser.add_argument(
    "--skip-lin",
    help="Skip lin, default false",
    action="store_true",
)

parser.add_argument(
    "--ratio-option",
    type=int,
    choices=[0, 1, 2],
    default=0,
    help="0: no data-bkg, 1: data-SM, 2: data-bkg",
)

args = parser.parse_args()
ops = args.ops
fitdiag_path = args.fit_path

do_prefit = args.prefit

do_lin = True
if args.skip_lin:
    do_lin = False

ratio_option = args.ratio_option

an_dict = get_analysis_dict(f"{args.an_path}")
samples = an_dict["samples"]
regions = an_dict["regions"]
years = an_dict["years"]
year_region_label = an_dict["year_region_label"]


f = uproot.open(fitdiag_path)

post_fit_folder = f"plots/plots_postfit_combined_{'_'.join(ops)}"
os.makedirs(post_fit_folder, exist_ok=True)


# get wilson coeffs from tree_fit_sb, which is a TTree
tree_fit_sb = f["tree_fit_sb"]

ops_dict = {}
for op in ops:
    op_val = tree_fit_sb[f"k_{op}"].array(library="np")[0]
    ops_dict[op] = op_val


sm_factor = 0.0
for key in ops_dict:
    sm_factor += ops_dict[key]

sm_factor = 1.0 * (1.0 - sm_factor)  # r here is fixed!

maxbins = max([regions[region]["nbins"] for region in regions])

fit_dirs_name = list(zip(["shapes_fit_s"], ["postfit"]))
if do_prefit:
    fit_dirs_name = list(zip(["shapes_prefit", "shapes_fit_s"], ["prefit", "postfit"]))

scales = ["log"]
if do_lin:
    scales = ["lin", "log"]

for scale in scales:
    for directory, name in fit_dirs_name:
        for region in regions:
            histos = {}

            for year in years:
                region_year = year_region_label(year, region)
                for sample in samples:
                    for subsample in samples[sample]["samples_group"]:
                        try:
                            val = f[f"{directory}/{region_year}/{subsample}"].values()
                        except Exception as _:
                            print(
                                f"Subsample {subsample} in {region_year} missing. Skipping ... "
                            )
                            continue

                        if sample not in histos:
                            histos[sample] = val.copy()
                        else:
                            histos[sample] += val

                sig_val = f[f"{directory}/{region_year}/total_signal"].values()
                if "sig" not in histos:
                    histos["sig"] = sig_val.copy()
                else:
                    histos["sig"] += sig_val

                mc_val = f[f"{directory}/{region_year}/total"].values()
                mc_err = f[f"{directory}/{region_year}/total"].variances()

                if "total" not in histos:
                    histos["total"] = mc_val.copy()
                    histos["total_err"] = mc_err.copy()
                else:
                    histos["total"] += mc_val
                    histos["total_err"] += mc_err

                data_val = f[f"{directory}/{region_year}/data"].values(axis=1)
                data_err = np.square(
                    f[f"{directory}/{region_year}/data"].errors(which="high", axis=1)
                )
                if "data" not in histos:
                    histos["data"] = data_val.copy()
                    histos["data_err"] = data_err.copy()
                else:
                    histos["data"] += data_val
                    histos["data_err"] += data_err

            sm_signals = []
            for sample in samples:
                if not samples[sample].get("is_signal", False):
                    continue
                histos[sample] = histos[sample] / sm_factor
                sm_signals.append(histos[sample])

            histos["sig"] = histos["sig"] - np.sum(sm_signals, axis=0)

            for key in histos:
                if "err" in key:
                    histos[key] = np.sqrt(histos[key])

            hlast = 0
            hmin = 10000

            edges = np.linspace(0, maxbins, maxbins + 1)
            centers = (edges[:-1] + edges[1:]) / 2

            if ratio_option not in [1, 2]:
                fig, ax = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                    dpi=200,
                    figsize=(6, 6),
                )  # figsize=(5,5), dpi=200)
            else:
                fig, ax = plt.subplots(
                    3,
                    1,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1, 1]},
                    dpi=200,
                    figsize=(6, 6),
                )  # figsize=(5,5), dpi=200)

            fig.tight_layout(pad=-0.5)
            region_label = regions[region].get("label", region)
            hep.cms.label(
                region_label,
                data=True,
                lumi=round(138, 2),
                ax=ax[0],
                year="Full Run II",
            )  # ,fontsize=16)

            for i, sample in enumerate(list(samples.keys())):
                if sample not in histos:
                    continue
                vals = histos[sample]

                if isinstance(hlast, int):
                    hlast = vals.copy()
                else:
                    hlast += vals.copy()

                # hmin = min(hmin, np.min(vals))
                hmin = min(hmin, np.min(hlast))

                integral = round(float(np.sum(vals)), 2)
                color = samples[sample]["color"]

                sample_label = samples[sample].get("label", sample)
                ax[0].stairs(
                    hlast,
                    edges,
                    label=sample_label + f" [{integral}]",
                    fill=True,
                    zorder=-i,
                    linewidth=1.0,
                    color=color,
                    edgecolor=darker_color(color),
                )

            vals_sig = histos["sig"]
            integral = round(float(np.sum(vals_sig)), 2)

            color = cmap_pastel[1]

            ax[0].stairs(
                hlast + vals_sig,
                edges,
                linewidth=1,
                color=color,
                fill=True,
                zorder=-len(list(histos.keys())),
                edgecolor=darker_color(color),
            )

            eft_zorder = len(histos) + 1

            # superimposed from hlast
            ax[0].stairs(
                hlast + vals_sig,
                edges,
                zorder=eft_zorder,
                linewidth=2,
                color=color,
                label=f"EFT [{integral}]",
            )

            # superimposed from 0
            ax[0].stairs(
                vals_sig,
                edges,
                zorder=eft_zorder,
                linewidth=2,
                linestyle="dashed",
                color=color,
            )

            vals = histos["total"]
            errs = histos["total_err"]
            color = "gray"
            integral = round(float(np.sum(vals)), 2)

            # its already present
            ax[0].stairs(
                vals,
                edges,
                zorder=+len(histos) + 1,
                linewidth=0,
                color=color,
                alpha=0,
                label=f"Total MC [{integral}]",
            )

            # its already present
            ax[0].stairs(
                vals + errs,
                edges,
                baseline=vals - errs,
                zorder=+len(histos) + 1,
                hatch="///",
                linewidth=0,
                color=color,
                # label=f"Total MC [{integral}]",
            )

            x = centers
            ys = np.array([histos["data"], histos["data_err"], histos["data_err"]])
            integral = round(float(np.sum(ys[0, :])), 2)
            ax[0].errorbar(
                x,
                ys[0, :],
                yerr=[ys[1, :], ys[2, :]],
                fmt="ko",
                markersize=4,
                label="Data" + f" [{integral}]",
                zorder=len(histos) + 3,
            )

            nbins = regions[region]["nbins"]

            ax[0].set_xlim(0, nbins)

            if scale == "log":
                ax[0].set_ylim(
                    max(0.5, hmin), np.max(hlast) * 5e3
                )  # FIXME remove comment
                ax[0].set_yscale("log")
            else:
                ax[0].set_ylim(
                    None, np.max(hlast) + (np.max(hlast) - hmin)
                )  # FIXME remove comment

            ax[0].legend(
                loc="upper center",
                # loc=(0.016, 0.75),
                frameon=True,
                ncols=3,
                framealpha=0.8,
                fontsize=8,
            )

            ax[1].errorbar(
                x,
                ys[0, :] / vals,
                yerr=[ys[1, :] / vals, ys[2, :] / vals],
                fmt="ko",
                markersize=4,
            )

            ax[1].plot(edges, np.ones_like(edges), color="black", linestyle="dashed")

            errs = histos["total_err"]

            ax[1].stairs(
                (vals + errs) / vals,
                edges,
                baseline=(vals - errs) / vals,
                # hatch="///",
                color="lightgray",
                # zorder=9,
                fill=True,
            )

            data_err = (ys[1, :] + ys[2, :]) / 2
            data_err = np.min((ys[1, :], ys[2, :]), axis=0)

            # need to mask only on meaningful bins for this region
            _mask = np.array([False] * nbins + [True] * (len(vals) - nbins))
            exp = np.exp(-np.square(ys[0, :] - vals) / (2 * data_err**2))[~_mask]
            chi2 = -2 * np.log(np.prod(exp))

            if name == "postfit":
                ax[1].set_ylim(0.90, 1.1)
                ax[1].legend(
                    [],
                    title="$\\chi^2$ = {:.2f}".format(chi2),
                    loc="upper center",
                    frameon=True,
                    ncols=3,
                    framealpha=0.8,
                    fontsize=8,
                )

            if ratio_option == 1:
                # ax 2, data - bkg - sm  to compare EFT
                vals_bkg = vals - vals_sig

                ax[2].errorbar(
                    x,
                    (ys[0, :] - vals_bkg),
                    yerr=[ys[1, :], ys[2, :]],
                    fmt="ko",
                    markersize=4,
                )

                ax[2].stairs(
                    np.zeros_like(vals_sig),
                    edges,
                    color="green",
                    linestyle="dashed",
                    label="SM",
                )
                ax[2].stairs(vals_sig, edges, color="red", label="EFT")
                ax[2].legend()
                ax[2].set_ylabel("Data - SM", fontsize=10, loc="center")
            elif ratio_option == 2:
                # ax 2, data - bkg - sm  to compare EFT
                vals_sm = np.sum(sm_signals, axis=0)
                vals_bkg = vals - vals_sig - vals_sm

                ax[2].errorbar(
                    x,
                    (ys[0, :] - vals_bkg),
                    yerr=[ys[1, :], ys[2, :]],
                    fmt="ko",
                    markersize=4,
                )

                ax[2].stairs(vals_sm, edges, color="green", label="SM")
                ax[2].stairs(vals_sm + vals_sig, edges, color="red", label="SM + EFT")
                ax[2].legend()
                ax[2].set_ylabel("Data - Bkg", fontsize=10, loc="center")

            ax[0].set_ylabel("Events", fontsize=12, loc="top")
            ax[1].set_ylabel("Data / Pred.", fontsize=10, loc="center")

            # variable label
            variable_label = regions[region].get("var_label", region)
            ax[-1].set_xlabel(variable_label, fontsize=10, loc="right")

            if "var_bins" in regions[region]:
                var_bins = regions[region]["var_bins"]
                ax[1].set_xticks(np.linspace(0, len(var_bins) - 1, len(var_bins)))
                ax[1].set_xticklabels(var_bins, rotation=0, ha="center", fontsize=8)

            if "var_splits" in regions[region]:
                var_splits = regions[region]["var_splits"]
                for split in var_splits:
                    for i in range(len(ax)):
                        ax[i].axvline(split, color="gray", linestyle="--", linewidth=0.5)
                    # ax[1].axvline(split, color="gray", linestyle="--", linewidth=0.5)

            plt.savefig(
                f"{post_fit_folder}/{scale}_{name}_{region}.png",
                facecolor="white",
                pad_inches=0.1,
                bbox_inches="tight",
            )
            plt.close()
