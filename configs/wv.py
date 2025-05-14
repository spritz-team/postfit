cmap_petroff = [
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
]
cmap_pastel = [
    "#A1C9F4",
    "#FFB482",
    "#8DE5A1",
    "#FF9F9B",
    "#D0BBFF",
    "#DEBB9B",
    "#FAB0E4",
    "#CFCFCF",
    "#FFFEA3",
    "#B9F2F0",
]

lumis = [35.0, 41.48, 59.83]
years = {}

for i, year in enumerate(["2017"]):
    years[year] = {
        "label": year,
        "lumi": lumis[i],
    }

# This function is exported to fetch the datacard folder in fitDiag.
def year_region_label(year, region):
        return f"{region}_{year}"

bins_regions = {
    'boost_sig_ele_DNN_geq_0p5': 8,
    'boost_sig_ele_DNN_leq_0p5': 8,
    'boost_sig_mu_DNN_geq_0p5': 8,
    'boost_sig_mu_DNN_leq_0p5': 8,
    'boost_topcr_ele': 1,
    'boost_topcr_mu': 1,
    'boost_wjetcr_ele': 7,
    'boost_wjetcr_mu': 7,
    'res_sig_ele_DNN_geq_0p5': 6,
    'res_sig_ele_DNN_leq_0p5': 6,
    'res_sig_mu_DNN_geq_0p5': 6,
    'res_sig_mu_DNN_leq_0p5': 6,
    'res_topcr_ele': 1,
    'res_topcr_mu': 1,
    'res_wjetcr_ele': 21,
    'res_wjetcr_mu': 21

}

# Define regions
regions = {}

lep_flav = ["mu", "ele"]
regime =  ["res"]

for cat in lep_flav:
    for cat2 in regime:
        regions[f"{cat2}_topcr_{cat}"] = {
            "nbins": bins_regions[f"{cat2}_topcr_{cat}"],
            "label": f"{cat2} Top {cat}",
            "var_bins": [0, 1],
            "var_label": "Events",
        }
        regions[f"{cat2}_sig_{cat}_DNN_geq_0p5"] = {
            "nbins": bins_regions[f"{cat2}_sig_{cat}_DNN_geq_0p5"],
            "label": f"{cat2} sig {cat} DNN > 0.5",
            "var_label": "mWV"
        }
        regions[f"{cat2}_sig_{cat}_DNN_leq_0p5"] = {
            "nbins": bins_regions[f"{cat2}_sig_{cat}_DNN_leq_0p5"],
            "label": f"{cat2} sig {cat} DNN < 0.5",
            "var_label": "mWV"
        }
        regions[f"{cat2}_wjetcr_{cat}"] = {
            "nbins": bins_regions[f"{cat2}_wjetcr_{cat}"],
            "label": f"{cat2} wjetcr {cat}",
            "var_label": "WJetsCR var"
        }



wjets_res_bins = []
wjets_boost_bins = []
for ir in range(1,22):
    wjets_res_bins.append("Wjets_res_"+str(ir))
for ir in range(1,8):
    wjets_boost_bins.append("Wjets_boost_"+str(ir))

wjets_bins = wjets_res_bins + wjets_boost_bins


samples = {}
colors = {}

samples["Data"] = {
    "samples_group": ["Data"],
    "is_data": True,
}


samples['VV+VVV']  = {  
#    'samples'  : ['VVV', 'VV_ssWW', 'VV_osWW', 'VV_WZjj', 'VV_WZll', 'VV_ZZ' ,'ggWW'],
    'samples_group'  : ['VVV', 'VV_ssWW', 'VV_osWW', 'VV_WZjj', 'VV_WZll', 'ggWW'],
    "is_data": False,
    "is_signal": False,
    'color': cmap_petroff[0]
}
colors['VV+VVV'] = cmap_petroff[0]


samples['DY']  = {  
    'samples_group'  : ['DY'],
    "is_data": False,
    "is_signal": False,
    'color': cmap_petroff[1]
}
colors['DY'] = cmap_petroff[1]

samples['Others']  = {  
    'samples_group'  : ['VBF-V_dipole', 'Vg','VgS' ],
    "is_data": False,
    "is_signal": False,
    'color': cmap_petroff[2]
}
colors['Others'] = cmap_petroff[2]

samples['Fake']  = {    
    'samples_group'  : ['Fake'],
    "is_data": False,
    "is_signal": False,
    'color': cmap_petroff[3]
}
colors['Fake'] = cmap_petroff[3]

samples['top']  = {    
    'samples_group'  : ['top'],
    "is_data": False,
    "is_signal": False,
    'color': cmap_petroff[4]
}
colors['top'] = cmap_petroff[4]


samples["Wjets"]  = {  
    'samples_group'  : wjets_bins,
    "is_data": False,
    "is_signal": False,
    'color': cmap_petroff[5]
}
colors['Wjets'] = cmap_petroff[5]

samples['VBS']  = {
    'samples_group'  : ["sm"],
    "is_data": False,
    "is_signal": True,
    'color': cmap_pastel[6]
}
colors['VBS'] = cmap_pastel[6]
