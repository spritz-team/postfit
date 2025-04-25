from utils import cmap_pastel, cmap_petroff, darker_color

lumis = [19.648, 16.97, 41.48, 59.83]
years = {}

for i, year in enumerate(["2016pre", "2016post", "2017", "2018"]):
    years[year] = {
        "label": year,
        "lumi": lumis[i],
    }

regions = {}

for cat in ["ee", "mm"]:
    regions[f"dypu_{cat}"] = {
        "nbins": 25,
        "label": "$\\Delta\\phi _{jj}$ : DNN",
    }

    regions[f"sr_{cat}"] = {
        "nbins": 25,
        "label": "$\\Delta\\phi _{jj}$ : DNN",
    }


def year_region_label(year, region):
    return f"vbfz_{region}_{year}"


samples = {}


# for j in range(5):
#     base_color = cmap_petroff[3]
#     factor = 1.0 / 1 - ((j + 1) / 5) / 2

#     samples[f"DY_hard_{j}"] = {
#         "samples_group": [f"DY_hard_{j}"],
#         "color": darker_color(base_color, factor),
#         "label": f"DY hard {j}",
#     }

#     base_color = cmap_petroff[0]
#     samples[f"DY_PU_{j}"] = {
#         "samples_group": [f"DY_PU_{j}"],
#         "color": darker_color(base_color, factor),
#         "label": f"DY PU {j}",
#     }

base_color = cmap_petroff[0]
samples["DY_PU"] = {
    "samples_group": [f"DY_PU_{j}" for j in range(5)],
    "color": base_color,
    "label": "DY PU",
}

base_color = cmap_petroff[3]
samples["DY_hard"] = {
    "samples_group": [f"DY_hard_{j}" for j in range(5)],
    "color": base_color,
    "label": "DY hard",
}

samples["Zjj"] = {
    "samples_group": ["sm"],
    "color": cmap_pastel[0],
    "is_signal": True,
    "label": "VBF-Z",
}
