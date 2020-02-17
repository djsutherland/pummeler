from collections import OrderedDict

import numpy as np
import pandas as pd

from .data import fod_codes

_naics_cat = {
    "11": "Agriculture",
    "21": "Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "3M": "Manufacturing",
    "42": "Wholesale",
    "44": "Retail",
    "45": "Retail",
    "4M": "Retail",
    "48": "Transportation",
    "49": "Transportation",
    "51": "Information",
    "52": "Financial Services",
    "53": "Rental and Leasing",
    "54": "Science and Technical",
    "55": "Management",
    "56": "Misc Professions",
    "61": "Education",
    "62": "Medical",
    "71": "Entertainment",
    "72": "Tourism",
    "81": "Services",
    "92": "Administration",
    "99": "Unemployed",
}


def naics_cat(s):
    return _naics_cat[str(s)[:2]]


occ_cats = [
    (0, "N/A"),
    (1, "Management, Business, Science, and Arts Occupations"),
    (500, "Business Operations Specialists"),
    (800, "Financial Specialists"),
    (1000, "Computer and Mathematical Occupations"),
    (1300, "Architecture and Engineering Occupations"),
    (1600, "Life, Physical, and Social Science Occupations"),
    (2000, "Community and Social Services Occupations"),
    (2100, "Legal Occupations"),
    (2200, "Education, Training, and Library Occupations"),
    (2600, "Arts, Design, Entertainment, Sports, and Media Occupations"),
    (3000, "Healthcare Practitioners and Technical Occupations"),
    (3600, "Healthcare Support Occupations"),
    (3700, "Protective Service Occupations"),
    (4000, "Food Preparation and Serving Occupations"),
    (4200, "Building and Grounds Cleaning and Maintenance Occupations"),
    (4300, "Personal Care and Service Occupations"),
    (4700, "Sales and Related Occupations"),
    (5000, "Office and Administrative Support Occupations"),
    (6000, "Farming, Fishing, and Forestry Occupations"),
    (6200, "Construction and Extraction Occupations"),
    (6800, "Extraction Workers"),
    (7000, "Installation, Maintenance, and Repair Workers"),
    (7700, "Production Occupations"),
    (9000, "Transportation and Material Moving Occupations"),
    (9800, "Military Specific Occupations"),
]
occ_codes, occ_names = zip(*occ_cats)
occ_codes = np.array(occ_codes)


def occ_cat(x):
    return occ_names[occ_codes.searchsorted(float(x), side="right") - 1]


fod_cats = {k: v for k, v in fod_codes().cat_code.items()}
# Other changes that need to be done in sort (:|):
# income recoding (log-scale, percentages for categories?)
# povpip recoding (0-500 can be real, but 501 needs to be discrete)


_ethnicity_map = {
    1: "white",
    2: "black",
    3: "amerindian",
    4: "amerindian",
    5: "amerindian",
    6: "asian",
    7: "amerindian",
    8: "other/biracial",
    9: "other/biracial",
    "hispanic": "hispanic",
}

_old_format = frozenset({"2006-10", "2007-11"})

_schg_remap = {
    float(k): float(v)
    for k, v in {  # new codes to old
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 4,
        10: 4,
        11: 5,
        12: 5,
        13: 5,
        14: 5,
        15: 6,
        16: 7,
    }.items()
}

_schl_remap = {
    float(k): float(v)
    for k, v in {  # new codes to old
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 3,
        9: 3,
        10: 4,
        11: 4,
        12: 5,
        13: 6,
        14: 7,
        15: 8,
        16: 9,
        17: 9,  # assuming GED falls under high school grad here?
        18: 10,
        19: 11,
        20: 12,
        21: 13,
        22: 14,
        23: 15,
        24: 16,
    }.items()
}

_mil_remap = {
    float(k): float(v)
    for k, v in {1: 1, 2: 2, 3: 2, 4: 3, 5: 4}.items()  # old codes to new
}

_fer_remap = {
    float(k): float(v)
    for k, v in {  # new to old, sort of
        1: 1,
        2: 2,
        8: np.nan
        # code 8 is because a few of these results were suppressed in
        # some PUMAS in FL/GA/KS/MT/NC/OH/TX in 2012;
        # arbitrarily call these no, I guess
    }.items()
}

_lanp_new_remap = {
    float(k): float(v)
    for k, v in {  # new to old
        602: 989,  # Krio => Other African
        675: 986,  # Sindhi => Other Asian
        689: 986,  # Uighur => Other Asian
        694: 986,  # Mongolian => Other Asian
        750: 988,  # Micronesian => Other Pacific Island
        761: 988,  # Trukese => Other Pacific Island
        819: 993,  # Ojibwa => Other North American Indian
    }.items()
}
_lanp_old_remap = {  # new to old
    float(k): float(v)
    for k, v in {966: 993}.items()  # "American Indian" => Other North American Indian
}

_ancp_new_remap = {
    float(k): float(v)
    for k, v in {
        94: 87,  # Irish Scotch => Scotch Irish
        131: 176,  # Montenegrin => Yugoslavian
        146: 144,  # Moldavian => Romanian
        168: 434,  # Turkestani => Turkish  :/  (no other Central Asian thing?)
        169: 434,  # Uzbeg => Turkish
        181: 32,  # Central European => German
        185: 51,  # Southern European => Italian
        194: 32,  # Germanic => German
        219: 215,  # Mexican Indian => Mexican American Indian
        411: 499,  # North African => Other Arab
        427: 499,  # Saudia Arabian => Other Arab
        515: 587,  # Congolese => Other Subsaharan African
        588: 587,  # Ugandan => Other Subsaharan African
        607: 799,  # Bhutanese => Other Asian
        714: 799,  # Tibetan => Other Asian
        825: 899,  # Marshallese => Other Pacific
        940: 939,  # United States => American or United States
    }.items()
}
_ancp_old_remap = {
    float(k): float(v)
    for k, v in {
        794: 995,  # Amerasian => Mixture
        936: 935,  # Acadian => French Canadian
    }.items()
}

for d in [_lanp_new_remap, _lanp_old_remap, _ancp_new_remap, _ancp_old_remap]:
    for k in map(float, range(1000)):
        d.setdefault(k, k)

_rac2p_old_remap = {
    float(k): v
    for k, v in {
        1: "White",
        2: "Black or African American",
        3: "Apache",
        4: "Blackfeet",
        5: "Cherokee",
        6: "Cheyenne",
        7: "Chickasaw",
        8: "Chippewa",
        9: "Choctaw",
        10: "Other specified American Indian tribes",  # "Colville"
        11: "Comanche",
        12: "Creek",
        13: "Crow",
        14: "Other specified American Indian tribes",  # "Delaware"
        15: "Other specified American Indian tribes",  # "Houma"
        16: "Iroquois",
        17: "Lumbee",
        18: "Other specified American Indian tribes",  # "Menominee",
        19: "Navajo",
        20: "Other specified American Indian tribes",  # "Paiute",
        21: "Pima",
        22: "Potawatomi",
        23: "Pueblo",
        24: "Puget Sound Salish",
        25: "Seminole",
        26: "Sioux",
        27: "Tohono O'Odham",
        28: "Other specified American Indian tribes",  # "Yakama",
        29: "Yaqui",
        30: "Other specified American Indian tribes",  # "Yuman",
        31: "Other specified American Indian tribes",
        # 32: "Combinations of American Indian tribes only",
        32: "All other specified American Indian tribe combinations",
        # 33: "American Indian or Alaska Native, tribe not specified, or
        #      American Indian and Alaska Native",
        33: "American Indian and Alaska Native, not specified",
        34: "Alaskan Athabascan",
        35: "Aleut",
        36: "Inupiat",
        37: "Tlingit-Haida",
        38: "Other Alaska Native",
        39: "American Indian and Alaska Native, not specified",
        40: "Asian Indian",
        41: "Bangladeshi",
        42: "Cambodian",
        43: "Chinese",
        44: "Filipino",
        45: "Hmong",
        46: "Indonesian",
        47: "Japanese",
        48: "Korean",
        49: "Laotian",
        50: "Malaysian",
        51: "Pakistani",
        52: "Sri Lankan",
        53: "Thai",
        54: "Vietnamese",
        55: "Other Asian",  # "Other specified Asian",
        56: "Other Asian",  # "Asian, not specified",
        # 57: "Combinations of Asian groups only",
        57: "All combinations of Asian races only",
        58: "Native Hawaiian",
        59: "Samoan",
        60: "Tongan",
        # 61: "Other Polynesian alone or in combination with other
        #      Polynesian groups",
        61: "Other Native Hawaiian and Other Pacific Islander",
        62: "Guamanian or Chamorro",
        # 63: "Other Micronesian alone or in combination with other
        #      Micronesian groups",
        63: "Other Native Hawaiian and Other Pacific Islander",
        # 64: "Melanesian alone or in combination with other Melanesian groups",
        64: "Other Native Hawaiian and Other Pacific Islander",
        65: "Other Native Hawaiian and Other Pacific Islander",
        66: "Some Other Race",
        67: "Two or More Races",
    }.items()
}
_rac2p_new_remap = {
    float(k): v
    for k, v in {
        1: "White",
        2: "Black or African American",
        3: "Apache",
        4: "Blackfeet",
        5: "Cherokee",
        6: "Cheyenne",
        7: "Chickasaw",
        8: "Chippewa",
        9: "Choctaw",
        10: "Comanche",
        11: "Creek",
        12: "Crow",
        13: "Other specified American Indian tribes",  # Hopi
        14: "Iroquois",
        15: "Lumbee",
        16: "Other specified American Indian tribes",  # Mexican American Indian
        17: "Navajo",
        18: "Pima",
        19: "Potawatomi",
        20: "Pueblo",
        21: "Puget Sound Salish",
        22: "Seminole",
        23: "Sioux",
        24: "Other specified American Indian tribes",  # South American Indian
        25: "Tohono O'Odham",
        26: "Yaqui",
        27: "Other specified American Indian tribes",
        28: "All other specified American Indian tribe combinations",
        # 29: "American Indian, tribe not specified",
        29: "American Indian and Alaska Native, not specified",
        30: "Alaskan Athabascan",
        31: "Tlingit-Haida",
        32: "Inupiat",
        33: "Other Alaska Native",  # "Yup'ik",
        34: "Aleut",
        35: "Other Alaska Native",
        # 36: "Other American Indian and Alaska Native specified",
        36: "All other specified American Indian tribe combinations",
        37: "American Indian and Alaska Native, not specified",
        38: "Asian Indian",
        39: "Bangladeshi",
        40: "Other Asian",  # "Bhutanese",
        41: "Other Asian",  # "Burmese",
        42: "Cambodian",
        43: "Chinese",  # Chinese, except Taiwanese
        44: "Chinese",  # "Taiwanese",
        45: "Filipino",
        46: "Hmong",
        47: "Indonesian",
        48: "Japanese",
        49: "Korean",
        50: "Laotian",
        51: "Malaysian",
        52: "Other Asian",  # "Mongolian",
        53: "Other Asian",  # "Nepalese",
        54: "Pakistani",
        55: "Sri Lankan",
        56: "Thai",
        57: "Vietnamese",
        58: "Other Asian",
        59: "All combinations of Asian races only",
        60: "Native Hawaiian",
        61: "Samoan",
        62: "Tongan",
        63: "Guamanian or Chamorro",
        64: "Other Native Hawaiian and Other Pacific Islander",  # "Marshallese",
        65: "Other Native Hawaiian and Other Pacific Islander",  # "Fijian",
        66: "Other Native Hawaiian and Other Pacific Islander",
        67: "Some Other Race",
        68: "Two or More Races",
    }.items()
}


class MyPreprocessor:
    def __init__(self, common_feats=False):
        self.common_feats = common_feats

    def handle_stats(self, stats):
        assert not hasattr(self, "stats")
        self.stats = stats
        common_feats = self.common_feats

        info = stats["version_info"]

        stats["do_common"] = common_feats
        new_pumas = info["region_year"] == "10"
        if not common_feats:
            assert new_pumas

        self.my_skip = skip_feats = set(info["alloc_flags"])
        self.inc_feats = inc_feats = set()
        not_feats = set()
        skip = lambda a: skip_feats.update(a.split())
        inc = lambda a: inc_feats.update(a.split())
        not_a_thing = lambda a: not_feats.update(a.split())

        if not common_feats:
            maybe = inc
        else:
            maybe = skip if new_pumas else not_a_thing
        skipp = skip if new_pumas else not_a_thing

        # basic info: keep AGEP, SEX
        inc("AGEP SEX")
        skip("QTRBIR")

        # drop stuff about relationships to the reference person
        skip("RELP OC RC SFN SFR")

        # citizenship
        inc("CIT DECADE NATIVITY NOP")
        maybe("CITWP")
        skip("YOEP")  # DECADE should be good enough

        # employment:
        inc("COW ESR WKHP WKL WKW")
        maybe("WRK")
        skip("INDP" if new_pumas else "indp02 indp07")
        skip("SOCP" if new_pumas else "socp00 socp10")
        inc("NAICSP OCCP")  # will be recoded below
        inc("ESP")  # for kids: are parents employed?
        skip("NWAB NWAV NWLA NWLK NWRE")

        # commute
        inc("JWDP")  # time left for work
        skip("JWAP")  # time arrived at work; fairly redundant with JWDP
        inc("JWMNP")  # travel time
        inc("JWRIP")  # carpooling
        skip("DRIVESP")  # 1/JWRIP
        inc("JWTR")  # commute method
        skip("POWPUMA POWSP")

        # income
        inc("INTP OIP PAP RETP SEMP SSIP SSP WAGP PERNP PINCP POVPIP")
        # NOTE: POVPIP sharply discontinuous, should discretize more

        # kids
        inc("FER")  # woman who gave birth in last year
        inc("PAOC")  # woman with kids' ages
        inc("GCL GCM GCR")  # grandparents living with grandkids

        # education
        inc("SCH SCHG SCHL")
        maybe("SCIENGP SCIENGRLP")
        maybe("FOD1P")
        skipp("FOD2P")  # recoded into categories below
        inc("HASDEGREE")

        # disability
        maybe("DDRS DEAR DEYE DOUT DPHY DREM DRAT DRATX DIS")

        # marital
        inc("MSP")
        maybe("MARHT MARHD MARHM MARHW MARHYP")
        skip("MAR")  # superceded by MSP

        # language
        inc("ENG LANX LANP")

        # health insurance
        maybe("HICOV PRIVCOV PUBCOV")
        skipp("HINS1 HINS2 HINS3 HINS4 HINS5 HINS6 HINS7")

        # migration
        inc("MIG WAOB")  # lived here a year ago, world area of birth
        skip("MIGPUMA MIGSP")
        skip("POBP")

        # military
        inc("MIL VPS")
        skip("MLPA MLPB MLPE MLPH MLPI MLPJ MLPK")
        if new_pumas:
            skip("MLPCD MLPFG")
        else:
            skip("MLPC MLPD MLPF MLPG")

        # ancestry
        inc("HISP")  # 24 levels, area of hispanic origin
        inc("RAC1P RAC2P")
        (skip if common_feats else inc)("RAC3P")
        # many RAC3P levels that changed meanings, probably overfitting anyway
        inc("RACAIAN RACASN RACBLK RACSOR RACWHT RACNUM")
        if "RACNHPI" in info["discrete_feats"]:
            inc("RACNHPI")
        else:
            inc("RACNH RACPI")
        skip("ANC")
        inc("ANC1P")
        skip("ANC2P")  # almost always 999 not reported, would need to recode
        inc("ETHNICITY ANYHISP")

        # modify stats
        vc = stats["value_counts"]
        if new_pumas:
            vc["NAICSP"] = vc["NAICSP"].groupby(naics_cat).sum()
            vc["OCCP"] = vc["OCCP"].groupby(occ_cat).sum()
            vc["FOD1P"] = vc["FOD1P"].groupby(fod_cats).sum()
            vc["FOD2P"] = vc["FOD2P"].groupby(fod_cats).sum()

            if common_feats:
                vc["SCHG"] = vc["SCHG"].groupby(_schg_remap).sum()
                vc["SCHL"] = vc["SCHL"].groupby(_schl_remap).sum()
                vc["FER"] = vc["FER"].groupby(_fer_remap).sum()
                vc["LANP"] = vc["LANP"].groupby(_lanp_new_remap).sum()
                vc["ANC1P"] = vc["ANC1P"].groupby(_ancp_new_remap).sum()
                vc["ANC2P"] = vc["ANC2P"].groupby(_ancp_new_remap).sum()
                vc["RAC2P"] = vc["RAC2P"].groupby(_rac2p_new_remap).sum()
        else:
            vc["NAICSP"] = (
                vc["naicsp02"].groupby(naics_cat).sum()
                + vc["naicsp07"].groupby(naics_cat).sum()
            )
            vc["OCCP"] = (
                vc["occp02"].groupby(occ_cat).sum()
                + vc["occp10"].groupby(occ_cat).sum()
            )
            vc["MIL"] = vc["MIL"].groupby(_mil_remap).sum()
            vc["LANP"] = vc["LANP"].groupby(_lanp_old_remap).sum()
            vc["ANC1P"] = vc["ANC1P"].groupby(_ancp_old_remap).sum()
            vc["ANC2P"] = vc["ANC2P"].groupby(_ancp_old_remap).sum()
            vc["RAC2P"] = vc["RAC2P"].groupby(_rac2p_old_remap).sum()
            del vc["naicsp02"], vc["naicsp07"], vc["occp02"], vc["occp10"]

        cutoff = 12 if common_feats else 20
        vc["HASDEGREE"] = vc["SCHL"].groupby(lambda x: int(x >= cutoff)).sum()
        vc["ANYHISP"] = vc["HISP"].groupby(lambda x: int(x == 1)).sum()
        vc["ETHNICITY"] = pd.Series(
            [stats["n_total"] - 5] + [1] * 5,
            index=[
                "hispanic",
                "white",
                "black",
                "amerindian",
                "asian",
                "other/biracial",
            ],
        )
        # obviously the value counts are a lie, but we don't actually use them

        stats["value_counts"] = OrderedDict(
            sorted((k, v.sort_index()) for k, v in vc.items() if k not in skip_feats)
        )

        all_feats = set(info["real_feats"]) | set(vc) | set(info["alloc_flags"])
        assert not_feats.isdisjoint(all_feats)
        assert all_feats == inc_feats | skip_feats

        stats["real_means"].drop(skip_feats, inplace=True, errors="ignore")
        stats["real_stds"].drop(skip_feats, inplace=True, errors="ignore")

        info["alloc_flags"] = [f for f in info["alloc_flags"] if f not in skip_feats]
        info["real_feats"] = [f for f in info["real_feats"] if f not in skip_feats]
        info["discrete_feats"] = [
            f
            for f in sorted(
                info["discrete_feats"] + ["ANYHISP", "HASDEGREE", "ETHNICITY"]
            )
            if f not in skip_feats
        ]

        self(stats["sample"])

    def always_skip(self, skip_feats):
        self.my_skip.update(skip_feats)
        self.inc_feats -= skip_feats

    def __call__(self, df):
        stats = self.stats
        is_old = stats["version"] in _old_format
        do_common = not is_old and stats["do_common"]
        inc_feats = self.inc_feats

        # get NAICS category
        if "NAICSP" in inc_feats:
            if is_old:
                n = df.naicsp02.where(df.naicsp07.isnull(), df.naicsp07)
            else:
                n = df.NAICSP
            df["NAICSP"] = n.map(naics_cat, na_action="ignore")

        # get OCC categories
        if "OCCP" in inc_feats:
            if is_old:
                o = df.occp02.where(df.occp10.isnull(), df.occp10)
            else:
                o = df.OCCP
            df["OCCP"] = o.map(occ_cat, na_action="ignore")

        # get field of degree categories
        # was averaging these before, but that's a little complicated in this
        # code structure, so whatever
        if "FOD1P" in inc_feats:
            df["FOD1P"] = df.FOD1P.map(fod_cats, na_action="ignore")
        if "FOD2P" in inc_feats:
            df["FOD2P"] = df.FOD2P.map(fod_cats, na_action="ignore")

        # these variables changed meanings; recode to old values
        if do_common:
            df["SCHG"] = df.SCHG.map(_schg_remap, na_action="ignore")
            df["SCHL"] = df.SCHL.map(_schl_remap, na_action="ignore")
            df["LANP"] = df.LANP.map(_lanp_new_remap, na_action="ignore")
            df["ANC1P"] = df.ANC1P.map(_ancp_new_remap, na_action="ignore")
            df["ANC2P"] = df.ANC2P.map(_ancp_new_remap, na_action="ignore")
            df["RAC2P"] = df.RAC2P.map(_rac2p_new_remap, na_action="ignore")
        elif is_old:
            df["MIL"] = df.MIL.map(_mil_remap, na_action="ignore")
            df["LANP"] = df.LANP.map(_lanp_old_remap, na_action="ignore")
            df["ANC1P"] = df.ANC1P.map(_ancp_old_remap, na_action="ignore")
            df["ANC2P"] = df.ANC2P.map(_ancp_old_remap, na_action="ignore")
            df["RAC2P"] = df.RAC2P.map(_rac2p_old_remap, na_action="ignore")

        # recoded variables
        if "ANYHISP" in inc_feats:
            df["ANYHISP"] = (df.HISP > 1).astype(int)
        if "HASDEGREE" in inc_feats:
            df["HASDEGREE"] = (df.SCHL >= 20).astype(int)
        if "ETHNICITY" in inc_feats:
            df["ETHNICITY"] = df.RAC1P.where(df.HISP == 1, "hispanic").map(
                _ethnicity_map
            )

        # drop skipped feats
        df.drop(columns=self.my_skip, inplace=True, errors="ignore")
