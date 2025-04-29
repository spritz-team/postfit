from utils import cmap_pastel, cmap_petroff, darker_color

lumis = [19.648, 16.97, 41.48, 59.83]
years = {}

for i, year in enumerate(["2016_HIPM", "2016_noHIPM", "2017", "2018"]):
    years[year] = {
        "label": year,
        "lumi": lumis[i],
    }

regions = {}

regions["ssww"] = {
    "nbins": 12,
    "label": "SSWW SR",
    "var_label": r"$m_{ll} : m_{jj}$",
    "var_splits": [3,6,9],
}

regions["wz"] = {
    "nbins": 6,
    "label": "WZ SR",
    "var_label": r"$m_{jj} : mT_{WZ}$",
    "var_splits": [2,4],
}

regions["wzb"] = {
    "nbins": 4,
    "label":"WZb CR",
    "var_bins": [500,800,1200,1800,3000],
    "var_label": r"$m_{jj} [GeV]$",
}


def year_region_label(year, region):
    return f"{region}{year}"


samples = {}


#samples["wz_sm"] = {
#    "samples_group": ["wz_sm"],
#    "color": "#AEBA8F",
#    "is_signal": True,
#    "label": "wz sm",
#}

samples["WpWp_QCD"] = {
    "samples_group": ["WpWp_QCD"],
    "color": "#AEBA8F",
    "is_signal": False,
    "label": "SSWW QCD",
}

samples["WZ_QCD"] = {
    "samples_group": ["WZ_QCD"],
    "color": "#DB8998",
    "is_signal": False,
    "label": "WZ QCD",
}

samples["ZZ"] = {
    "samples_group": ["ZZ"],
    "color": "#C1DFFE",
    "is_signal": False,
    "label": "ZZ",
}

samples["tVx"] = {
    "samples_group": ["tVx"],
    "color": "#AC6FFE",
    "is_signal": False,
    "label": "tVx",
}

samples["wrong-sign"] = {
    "samples_group": ["WW", "Top", "Higgs"],
    "color": "#82DAFE",
    "is_signal": False,
    "label": "wrong-sign",
}

samples["VVV"] = {
    "samples_group": ["VVV"],
    "color": "#AEBA8F",
    "is_signal": False,
    "label": "VVV",
}

samples["Fake_lep"] = {
    "samples_group": ["Fake_lep"],
    "color": "#9E9DBE",
    "is_signal": False,
    "label": "non-prompt",
}

samples["ww_sm"] = {
    "samples_group": ["ww_sm", "wz_sm"],
    "color": "burlywood",
    "is_signal": True,
    "label": "ww + wz sm",
}


