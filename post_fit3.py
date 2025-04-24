import os
from copy import deepcopy

import matplotlib as mpl
import mplhep as hep
import numpy as np
import uproot
from spritz.framework.framework import cmap_pastel, get_analysis_dict

mpl.use("Agg")
import matplotlib.pyplot as plt

d = deepcopy(hep.style.CMS)

d["font.size"] = 12
d["figure.figsize"] = (5, 5)

plt.style.use(d)


def darker_color(color):
    rgb = list(mpl.colors.to_rgba(color)[:-1])
    darker_factor = 4 / 5
    rgb[0] = rgb[0] * darker_factor
    rgb[1] = rgb[1] * darker_factor
    rgb[2] = rgb[2] * darker_factor
    return tuple(rgb)


an_dict = get_analysis_dict("../configs/vbfz-eft-2018/")
colors = an_dict["colors"]
samples = an_dict["samples"]

op = "cHDD"

# f = uproot.open("../configs/vbfz-eft-2018/fits/fitDiagnosticsTest_run2_fullsyst.root")
# f = uproot.open("../configs/vbfz-eft-2018/fits/fitDiagnostics.Test.root")
f = uproot.open("../configs/vbfz-eft-2018/fits/fitDiagnosticsTest.root")
# f = uproot.open("../configs/vbfz-eft-2018/fits/fitDiagnosticsTest_cW.root")
dir_key = "shapes_fit_s"
print(f[dir_key].keys())
years = ["vbfz_2016pre", "vbfz_2016post", "vbfz_2017", "vbfz_2018"]
year_labels = ["2016 HIPM", "2016 noHIPM", "2017", "2018"]
lumis = [19.65, 16.98, 41.48, 59.83]
# years = ["vbfz_2018"]
# year_labels = ["2018"]
# lumis = [59.83]
regions = ["top", "dypu", "sr"]
regions = [
    "sr",
    # "dypu",
    "top",
]  # , "sr_detajj", "sr_mjj", "sr_ptj1", "sr_ptj2"]
region_labels = [
    "VBF-Z SR",
    # "VBF-Z DY PU CR",
    "VBF-Z Top CR",
    # "VBF-Z SR",
    # "VBF-Z SR",
    # "VBF-Z SR",
    # "VBF-Z SR",
]

nbins_total = 25
nbins_region = {}
variables_region = {}
new_regions = []
new_lumis = []
new_year_labels = []
new_region_labels = []
force_blind_masks = []
force_bin_edges = []
for year, lumi, year_label in zip(years, lumis, year_labels[:]):
    for region, region_label in zip(regions, region_labels):
        for cat in ["ee", "mm"]:
            y = year.split("_")
            s = region.split("_")
            if len(s) == 1:
                new_region = f"{y[0]}_{s[0]}_{cat}_{y[1]}"
            else:
                new_region = f"{y[0]}_{s[0]}_{cat}_{s[1]}_{y[1]}"
            print(new_region)
            new_regions.append(new_region)
            new_lumis.append(lumi)
            new_year_labels.append(year_label)
            new_region_labels.append(f"{region_label} {cat}")
            if region == "top":
                nbins_region[new_region] = 1
                variables_region[new_region] = "Events"
                force_blind_mask = [False] * nbins_total
                bin_edges = None
            if region == "dypu":
                nbins_region[new_region] = nbins_total
                variables_region[new_region] = "$\\Delta \\phi _{jj}$ : DNN"
                force_blind_mask = [False] * nbins_total
                bin_edges = None
            if "sr" in region:
                if len(s) == 1:
                    nbins_region[new_region] = nbins_total
                    variables_region[new_region] = "$\\Delta \\phi _{jj}$ : DNN"
                    # force_blind_mask = [False] * 20 + [True] * 10
                    force_blind_mask = [False] * nbins_total  # + [True] * 10
                    bin_edges = None
                # else:
                #     if s[1] == "detajj":
                #         nbins_region[new_region] = 6
                #         variables_region[new_region] = "$\\Delta \\eta _{jj}$"
                #         force_blind_mask = [False] * 5 + [True] + [False] * 24
                #         bin_edges = np.linspace(0, 8.5, 7)
                #     elif s[1] == "mjj":
                #         nbins_region[new_region] = 30
                #         variables_region[new_region] = "$m _{jj}$"
                #         force_blind_mask = [False] * 24 + [True] * 6
                #         bin_edges = np.linspace(200, 1500, 31)
                #     elif s[1] == "ptj1":
                #         nbins_region[new_region] = 30
                #         variables_region[new_region] = "$p ^{T} _{j1}$"
                #         force_blind_mask = [True] * 2 + [False] * 28
                #         bin_edges = np.linspace(30, 500, 31)
                #     elif s[1] == "ptj2":
                #         nbins_region[new_region] = 30
                #         variables_region[new_region] = "$p ^{T} _{j2}$"
                #         force_blind_mask = [True] * 2 + [False] * 28
                #         bin_edges = np.linspace(30, 500, 31)
            force_blind_mask = np.array(force_blind_mask)
            force_blind_masks.append(force_blind_mask)
            if bin_edges is None:
                bin_edges = np.linspace(
                    0, nbins_region[new_region], nbins_region[new_region] + 1
                )
            force_bin_edges.append(bin_edges)

print(nbins_region)


# regions = ["vbfz_16post_sr_ee"]
# samples = ["DY_hard", "DY_PU", "Zjj_fiducial", "Zjj_outfiducial", "Int", "VV", "Top"]
# samples = ["Int", "VV", "Zjj_outfiducial", "Top", "DY_PU", "DY_hard", "Zjj_fiducial"]

regions = new_regions
lumis = new_lumis
year_labels = new_year_labels
region_labels = new_region_labels
print(regions)
print(new_year_labels)

post_fit_folder = "plots_postfit5"
os.makedirs(post_fit_folder, exist_ok=True)

exps = []
blind = True

# year_label = "Full Run II"
for region, lumi, year_label, region_label, force_blind_mask, force_bin_edge in zip(
    regions, lumis, year_labels, region_labels, force_blind_masks, force_bin_edges
):
    # sample = "Zjj_fiducial"
    # signal = f[f"{dir_key}/{region}/{sample}"].values()
    sample = "total"
    tot = f[f"{dir_key}/{region}/{sample}"].values()
    # blind_mask = np.zeros_like(tot) != 0.0

    fig, ax = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        dpi=200,
        figsize=(6, 6),
    )  # figsize=(5,5), dpi=200)
    fig.tight_layout(pad=-0.5)
    hep.cms.label(
        region_label, data=True, lumi=round(lumi, 2), ax=ax[0], year=year_label
    )  # ,fontsize=16)

    hlast = 0
    hmin = 10000
    tot_bins = nbins_total
    print(force_bin_edge)
    print([force_bin_edge[-1] + i for i in range(tot_bins + 1 - len(force_bin_edge))])
    edges = np.array(
        list(force_bin_edge)
        + [force_bin_edge[-1] + i for i in range(tot_bins + 1 - len(force_bin_edge))]
    )
    print(edges)
    for i, sample in enumerate(samples):
        if samples[sample].get("is_signal", False):
            continue
        if "data" in sample.lower():
            continue
        vals = f[f"{dir_key}/{region}/{sample}"].values()
        if "ptj1" in region or "ptj2" in region:
            vals[2] = vals[:3].sum()
            vals[:2] = 0.0
        if isinstance(hlast, int):
            hlast = vals.copy()
        else:
            hlast += vals.copy()

        hmin = min(hmin, np.min(vals))
        integral = round(float(np.sum(vals)), 2)
        color = colors[sample]

        ax[0].stairs(
            hlast,
            edges,
            label=sample + f" [{integral}]",
            fill=True,
            zorder=-i,
            linewidth=1.0,
            color=color,
            edgecolor=darker_color(color),
        )

    vals_sig = 0
    color = cmap_pastel[0]
    for i, sample in enumerate(["sm"]):
        _vals = f[f"{dir_key}/{region}/{sample}"].values()
        if "ptj1" in region or "ptj2" in region:
            _vals[2] = _vals[:3].sum()
            _vals[:2] = 0.0
        if isinstance(vals_sig, int):
            vals_sig = _vals.copy()
        else:
            vals_sig += _vals

    sample = "total_signal"
    vals_sig = f[f"{dir_key}/{region}/{sample}"].values()

    # # FIXME using signal prefit
    # sample = "total_signal"
    # _dir_key = "shapes_prefit"
    # vals_sig = f[f"{_dir_key}/{region}/{sample}"].values()

    # print(vals_sig)
    ax[0].stairs(
        hlast + vals_sig,
        edges,
        linewidth=1,
        color=color,
        fill=True,
        zorder=-len(samples),
        edgecolor=darker_color(color),
    )

    integral = round(float(np.sum(vals_sig)), 2)
    ax[0].stairs(
        vals_sig,
        edges,
        zorder=50,
        linewidth=2,
        color=color,
        label=f"SM + EFT [{integral}]",
    )

    sample = "total"
    vals = f[f"{dir_key}/{region}/{sample}"].values()
    print(vals)

    if "ptj1" in region or "ptj2" in region:
        vals[2] = vals[:3].sum()
        vals[:2] = 0.0

    errs = f[f"{dir_key}/{region}/{sample}"].errors()
    if "ptj1" in region or "ptj2" in region:
        errs[2] = errs[:3].sum()
        errs[:2] = 0.0

    # FIXME using signal prefit
    sample = "total_background"
    vals = f[f"{dir_key}/{region}/{sample}"].values() + vals_sig
    errs = f[f"{dir_key}/{region}/{sample}"].errors()

    # edges = f[f"{dir_key}/{region}/{sample}"].axis().edges()
    # ax.stairs(vals, edges, label=sample)
    unc_down = round(float(np.sum(errs) / np.sum(vals) * 100), 2)
    unc_up = round(float(np.sum(errs) / np.sum(vals) * 100), 2)

    vals_bkg = vals - vals_sig

    significance = vals_sig / vals_bkg
    # blind_mask = (significance > 0.10) | force_blind_mask
    blind_mask = force_blind_mask.copy()
    print(region, blind_mask.shape)
    print(blind_mask)

    ax[0].stairs(
        vals + errs,
        edges,
        baseline=vals - errs,
        hatch="///",
        facecolor="none",
        linewidth=0,
        color="darkgrey",
        label=f"Syst [-{unc_down}, +{unc_up}]%",
        zorder=len(samples),
    )

    integral = round(float(np.sum(vals)), 2)
    # ax[0].stairs(
    #     vals,
    #     edges,
    #     label=f"Tot MC [{integral}]",
    #     color="darkgrey",
    #     linewidth=1,
    #     zorder=len(samples),
    # )

    sample = "data"
    # x = f[f"{dir_key}/{region}/{sample}"].values(axis=0)
    x = (edges[:-1] + edges[1:]) / 2
    y = f[f"{dir_key}/{region}/{sample}"].values(axis=1)
    yerr_up = f[f"{dir_key}/{region}/{sample}"].errors(which="high", axis=1)
    yerr_do = f[f"{dir_key}/{region}/{sample}"].errors(which="low", axis=1)
    ys = np.array([y, yerr_do, yerr_up])
    print(ys.shape)

    if "ptj1" in region or "ptj2" in region:
        ys[0, 2] = np.sum(ys[0, :3])
        ys[0, :2] = 0.0

        for i in range(1, 3):
            ys[i, 2] = np.sum(ys[i, :3]) / 2
            ys[i, :2] = 0.0

    for i in range(3):
        ys[i, :] = np.where(blind_mask, 0.0, ys[i, :])
    vals = np.where(blind_mask, 0.0, vals)
    errs = np.where(blind_mask, 0.0, errs)

    integral = round(float(np.sum(y)), 2)
    ax[0].errorbar(
        x,
        ys[0, :],
        yerr=[ys[1, :], ys[2, :]],
        fmt="ko",
        markersize=4,
        label="Data" + f" [{integral}]",
        zorder=len(samples) + 1,
    )

    # Ratio
    data_err = (ys[1, :] + ys[2, :]) / 2
    data_err = np.min((ys[1, :], ys[2, :]), axis=0)
    print(data_err)
    mc_err = errs

    nbins = nbins_total
    if "top" in region:
        nbins = 1

    # chi2 = (
    #     np.sum((np.square(ys[0, :] - vals) / (data_err**2 + mc_err**2))[~blind_mask])
    #     / nbins
    # )
    # chi2 = np.sum((np.square(ys[0, :] - vals) / (data_err**2))[~blind_mask]) / nbins
    _mask = np.array([False] * nbins + [True] * (nbins_total - nbins))
    exp = np.exp(-np.square(ys[0, :] - vals) / (2 * data_err**2))[~_mask]
    exps.append(exp)
    chi2 = -2 * np.log(np.prod(exp))

    # print(ys[0, :])
    # print(vals)
    # print((ys[0, :] - vals) / np.max([ys[1, :], ys[2, :]], axis=0))
    # print(len(ys[0, :][~blind_mask]))

    ax[1].errorbar(
        x,
        ys[0, :] / vals,
        yerr=[ys[1, :] / vals, ys[2, :] / vals],
        fmt="ko",
        markersize=4,
    )

    ax[1].plot(edges, np.ones_like(edges), color="black", linestyle="dashed")

    ax[1].stairs(
        (vals + errs) / vals,
        edges,
        baseline=(vals - errs) / vals,
        # hatch="///",
        color="lightgray",
        # zorder=9,
        fill=True,
    )

    ax[0].legend(
        loc="upper center",
        # loc=(0.016, 0.75),
        frameon=True,
        ncols=3,
        framealpha=0.8,
        fontsize=10,
    )
    ax[1].legend(
        # title="$\\chi^2_0$ / ndof = {:.2f}".format(chi2),
        title="$\\chi^2$ = {:.2f}".format(chi2),
        loc="upper center",
        # loc=(0.016, 0.75),
        frameon=True,
        ncols=3,
        framealpha=0.8,
        fontsize=8,
    )

    # rwgt = vals_sig / vals

    # err_data_minus_mc = (ys[0, :] + data_err) * rwgt - vals_bkg * rwgt
    # # print(err_data_minus_mc)
    # # print(data_err * rwgt)
    # # ax[2].errorbar(x, (ys[0, :] - vals_bkg) * rwgt, yerr=err_data_minus_mc, fmt="ko")
    # ax[2].errorbar(
    #     x,
    #     (ys[0, :] - vals_bkg) * rwgt,
    #     yerr=data_err * rwgt,
    #     fmt="ko",
    #     markersize=4,
    #     label="Data",
    # )
    # ax[2].stairs(
    #     vals_sig * rwgt, edges, zorder=50, linewidth=2, color=color, label="SM + EFT"
    # )
    # ax[2].stairs(
    #     mc_err * rwgt,
    #     edges,
    #     baseline=-mc_err * rwgt,
    #     color="lightgray",
    #     label="Syst",
    #     fill=True,
    # )
    # ax[2].plot(edges, np.zeros_like(edges), color="black", linestyle="dashed")
    # ax[2].legend(loc="upper center", frameon=True, fontsize=8, framealpha=0.8)
    # ax[2].set_ylabel("Data - Bkg", loc="center")

    # ax[1].set_xticks(np.linspace(0, len(force_bin_edges) - 1, len(force_bin_edges)))
    # ax[1].set_xticklabels(force_bin_edges)
    ax[0].set_ylim(max(0.5, hmin), np.max(hlast) * 1e3)
    ax[1].set_ylim(0.90, 1.1)
    ax[0].set_yscale("log")
    # ax[0].set_xlim(0, nbins_region[region])
    if "ptj1" in region or "ptj2" in region:
        ax[0].set_xlim(force_bin_edge[2], force_bin_edge[-1])
    else:
        ax[0].set_xlim(force_bin_edge[0], force_bin_edge[-1])

    ax[0].set_ylabel("Events")
    ax[1].set_ylabel("Data / MC", loc="center")
    ax[1].set_xlabel(variables_region[region])

    plt.savefig(
        f"{post_fit_folder}/{region}.png",
        facecolor="white",
        pad_inches=0.1,
        bbox_inches="tight",
    )
    plt.close()

    # print(region)
    # break

exps = np.array([x for exp in exps for x in exp])
print(-2 * np.log(np.prod(exps)))
