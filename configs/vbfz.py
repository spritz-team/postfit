from utils import cmap_pastel, cmap_petroff, darker_color

lumis = [19.648, 16.97, 41.48, 59.83]
years = {}

for i, year in enumerate(["2016pre", "2016post", "2017", "2018"]):
    years[year] = {
        "label": year,
        "lumi": lumis[i],
    }



var_bins = ["$\\pi$", 2.29, 1.58, 0.96, 0.38] * 5 + [0.0]
var_bins = [round(f, 1) if not isinstance(f, str) else f for f in var_bins]

# Define regions
regions = {}

for cat in ["ee", "mm"]:
    regions[f"top_{cat}"] = {
        "nbins": 1,
        "label": f"Top {cat}",
        "var_bins": [0, 1],
        "var_label": "Events",
    }

    regions[f"dypu_{cat}"] = {
        "nbins": 25,
        "label": f"DY PU {cat}",
        "var_bins": var_bins,
        "var_label": "$\\Delta\\phi _{jj}$ : DNN",
        "var_splits": [5, 10, 15, 20],
    }

    regions[f"sr_{cat}"] = {
        "nbins": 25,
        "label": f"SR {cat}",
        "var_bins": var_bins,
        "var_label": "$\\Delta\\phi _{jj}$ : DNN",
        "var_splits": [5, 10, 15, 20],
    }


# This function is exported to fetch the datacard folder in fitDiag.
def year_region_label(year, region):
    return f"vbfz_{region}_{year}"


# Define sample
samples = {}

samples["Int"] = {
    "samples_group": ["Int"],
    "color": cmap_petroff[4],
    "label": "Int",
}


samples["VV"] = {
    "samples_group": ["VV"],
    "color": cmap_petroff[2],
    "label": "VV",
}

samples["Top"] = {
    "samples_group": ["Top"],
    "color": cmap_petroff[1],
    "label": "Top",
}

for key, i_color in zip(["PU", "hard"], [3, 0]):
    for j in range(5):
        base_color = cmap_petroff[i_color]
        factor = 1.0 / 1 - ((j + 1) / 5) / 2

        samples[f"DY_{key}_{j}"] = {
            "samples_group": [f"DY_{key}_{j}"],
            "color": darker_color(base_color, factor),
            "label": f"DY {key} {j}",
        }

samples["Zjj"] = {
    "samples_group": ["sm"],
    "color": cmap_pastel[0],
    "is_signal": True,
    "label": "VBF-Z",
}
