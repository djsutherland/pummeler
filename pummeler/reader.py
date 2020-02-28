from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import pandas as pd


weirds = """
    SERIALNO
    indp02 indp07 INDP
    OCCP occp02 occp10 OCCP10 OCCP12
    socp00 socp10 SOCP10 SOCP12 SOCP
    naicsp02 naicsp07 NAICSP
""".split()


def read_chunks(
    fname,
    version,
    chunksize=10 ** 5,
    voters_only=False,
    adj_inc=None,
    adj_hsg=None,
    housing_source=None,  # func from (state, puma) => filename
    housing_cache_size=8,
):
    info = VERSIONS[version]
    dtypes = {}
    for k in info["meta_cols"] + info["discrete_feats"] + info["alloc_flags"]:
        dtypes[k] = "category"
    for k in info["real_feats"]:
        dtypes[k] = "float64"
    for k in info["weight_cols"]:
        dtypes[k] = "Int64"
    dtypes["SERIALNO"] = dtypes["serialno"] = "string"

    if adj_inc and not info.get("to_adjinc"):
        adj_inc = False
    if adj_hsg and not info.get("to_adjhsg"):
        adj_hsg = False

    if housing_source is not None:

        def get_housing_files(st_pumas):
            return pd.concat(
                [
                    load_file(fn)
                    for fn in {housing_source(st, puma) for st, puma in st_pumas}
                ]
            )

        @lru_cache(maxsize=housing_cache_size)
        def load_file(fn):
            fn = Path(fn)
            if fn.suffix in {".pq", ".parquet"}:
                df = pd.read_parquet(fn)
            elif fn.suffix in {".h5", ".hdf5"}:
                df = pd.read_hdf(fn)
            else:
                raise ValueError(f"unknown file format {fn.suffix!r}")
            df.drop(
                columns=["RT", "ST", "PUMA", "ADJINC_orig"],
                errors="ignore",
                inplace=True,
            )
            return df

    chunks = pd.read_csv(
        fname,
        skipinitialspace=True,
        na_values={k: ["N.A.", "N.A.//", "N.A.////"] for k in weirds},
        dtype=dtypes,
        chunksize=chunksize,
    )
    renames = None
    for chunk in chunks:
        if info.get("puma_subset", False):
            puma_key = "PUMA{}".format(info["region_year"])
            chunk = chunk[chunk[puma_key] != -9]
            if chunk.shape[0] == 0:
                continue
            chunk["PUMA"] = chunk[puma_key]
            del chunk["PUMA00"]
            del chunk["PUMA10"]

        if voters_only:
            chunk = chunk[(chunk.AGEP >= 18) & (chunk.CIT != 5)]

        if chunk.shape[0] == 0:
            continue

        if "drop_feats" in info:  # TODO: pass usecols to read_csv instead...
            chunk.drop(info["drop_feats"], axis=1, inplace=True)

        if "renames" in info:
            if renames is None:
                renames = [info["renames"].get(k, k) for k in chunk.columns]
            chunk.columns = renames

        if adj_inc is None:
            if "ADJINC" in chunk:
                adj_inc = True
            elif "ADJINC_orig" in chunk:
                adj_inc = False
            else:
                raise ValueError(
                    "Unclear whether income has been adjusted, "
                    "and adj_inc is None; pass either True or "
                    "False explicitly"
                )
        if adj_inc:
            adj = chunk.ADJINC / 1e6
            for k in info["to_adjinc"]:
                chunk[k] *= adj
            chunk.rename(columns={"ADJINC": "ADJINC_orig"}, inplace=True)

        if adj_hsg is None:
            if "ADJHSG" in chunk:
                adj_hsg = True
            elif "ADJHSG_orig" in chunk:
                adj_hsg = False
            else:
                raise ValueError(
                    "Unclear whether income has been adjusted, "
                    "and adj_hsg is None; pass either True or "
                    "False explicitly"
                )
        if adj_hsg:
            adj = chunk.ADJHSG / 1e6
            for k in info["to_adjhsg"]:
                chunk[k] *= adj
            chunk.rename(columns={"ADJHSG": "ADJHSG_orig"}, inplace=True)

        if housing_source is not None:
            housing = get_housing_files(chunk.groupby(["ST", "PUMA"]).groups)
            chunk = chunk.merge(housing, on="SERIALNO", suffixes=(False, False))

        yield chunk


def _s(s):
    return sorted(s.split())


VERSIONS = {}

VERSIONS["2006-10"] = {
    "weight_cols": ["PWGTP"] + ["PWGTP{}".format(i) for i in range(1, 81)],
    "meta_cols": "RT SPORDER serialno PUMA ST".split(),
    "discrete_feats": _s(
        """
            CIT COW ENG FER GCL GCM GCR JWRIP JWTR LANX MAR MIG MIL MLPA MLPB
            MLPC MLPD MLPE MLPF MLPG MLPH MLPI MLPJ MLPK NWAB NWAV NWLA NWLK
            NWRE RELP SCH SCHG SCHL SEX WKL WKW ANC ANC1P ANC2P DECADE DRIVESP
            ESP ESR HISP indp02 indp07 LANP MIGPUMA MIGSP MSP naicsp02 naicsp07
            NATIVITY NOP OC occp02 occp10 PAOC POBP POWPUMA POWSP QTRBIR RAC1P
            RAC2P RAC3P RACAIAN RACASN RACBLK RACNHPI RACSOR RACWHT RC SFN SFR
            socp00 socp10 VPS WAOB"""
    ),
    "alloc_flags": _s(
        """
            FAGEP FANCP FCITP FCOWP FENGP FESRP FFERP FGCLP FGCMP FGCRP
            FHISP FINDP FINTP FJWDP FJWMNP FJWRIP FJWTRP FLANP FLANXP FMARP
            FMIGP FMIGSP FMILPP FMILSP FOCCP FOIP FPAP FPOBP FPOWSP FRACP
            FRELP FRETP FSCHGP FSCHLP FSCHP FSEMP FSEXP FSSIP FSSP FWAGP
            FWKHP FWKLP FWKWP FYOEP"""
    ),
    "real_feats": _s(
        """
            AGEP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP WKHP YOEP JWAP
            JWDP PERNP PINCP POVPIP RACNUM"""
    ),
    "to_adjinc": _s("INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP"),
    "region_year": "00",
}
VERSIONS["2007-11"] = VERSIONS["2006-10"]

VERSIONS["2010-14_12-14"] = {
    "weight_cols": ["PWGTP"] + ["PWGTP{}".format(i) for i in range(1, 81)],
    "meta_cols": _s("RT SPORDER serialno PUMA ST"),
    "alloc_flags": _s(
        """
            FAGEP FCITP FCITWP FCOWP FDDRSP FDEARP FDEYEP FDOUTP FDPHYP
            FDRATP FDRATXP FDREMP FENGP FESRP FFERP FFODP FGCLP FGCMP FGCRP
            FHINS1P FHINS2P FHINS3C FHINS3P FHINS4C FHINS4P FHINS5C FHINS5P
            FHINS6P FHINS7P FHISP FINDP FINTP FJWDP FJWMNP FJWRIP FJWTRP
            FLANP FLANXP FMARHDP FMARHMP FMARHTP FMARHWP FMARHYP FMARP FMIGP
            FMIGSP FMILPP FMILSP FOCCP FOIP FPAP FPOBP FPOWSP FRACP FRELP
            FRETP FSCHGP FSCHLP FSCHP FSEMP FSEXP FSSIP FSSP FWAGP FWKHP
            FWKLP FWKWP FWRKP FYOEP"""
    ),
    "discrete_feats": _s(
        """
            CIT COW DDRS DEAR DEYE DOUT DPHY DRAT DRATX DREM ENG FER GCL GCM
            GCR HINS1 HINS2 HINS3 HINS4 HINS5 HINS6 HINS7 JWRIP JWTR LANX
            MAR MARHD MARHM MARHT MARHW MARHYP MIG MIL MLPA MLPB MLPCD MLPE
            MLPFG MLPH MLPI MLPJ MLPK NWAB NWAV NWLA NWLK NWRE RELP SCH SCHG
            SCHL SEX WKL WKW WRK ANC ANC1P ANC2P DECADE DIS DRIVESP ESP ESR
            FOD1P FOD2P HICOV HISP INDP LANP MIGPUMA MIGSP MSP NAICSP NATIVITY
            NOP OC OCCP PAOC POBP POWPUMA POWSP PRIVCOV PUBCOV QTRBIR RAC1P
            RAC2P RAC3P RACAIAN RACASN RACBLK RACNHPI RACSOR RACWHT RC SCIENGP
            SCIENGRLP SFN SFR SOCP VPS WAOB"""
    ),
    "real_feats": _s(
        """
            AGEP CITWP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP WKHP YOEP
            JWAP JWDP PERNP PINCP POVPIP RACNUM"""
    ),
    "to_adjinc": _s("INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP"),
    "drop_feats": _s(
        """
            ANC1P05 ANC2P05 FANCP LANP05 MARHYP05 MIGPUMA00 MIGSP05 POBP05
            POWPUMA00 POWSP05 RAC2P05 RAC3P05 SOCP10 YOEP05 CITWP05 OCCP10"""
    ),
    # Always blank for 12-14 data
    "renames": {
        "ANC1P12": "ANC1P",
        "ANC2P12": "ANC2P",
        "LANP12": "LANP",
        "MARHYP12": "MARHYP",
        "MIGPUMA10": "MIGPUMA",
        "MIGSP12": "MIGSP",
        "OCCP12": "OCCP",
        "POBP12": "POBP",
        "POWPUMA10": "POWPUMA",
        "POWSP12": "POWSP",
        "RAC2P12": "RAC2P",
        "RAC3P12": "RAC3P",
        "SOCP12": "SOCP",
        "YOEP12": "YOEP",
        "CITWP12": "CITWP",
    },
    "region_year": "10",
    "puma_subset": True,
}

VERSIONS["2011-15_12-15"] = v = deepcopy(VERSIONS["2010-14_12-14"])
v["alloc_flags"] = sorted(
    v["alloc_flags"] + "FDISP FPINCP FPUBCOVP FPERNP FPRIVCOVP".split()
)

VERSIONS["2012-16"] = {
    "weight_cols": ["PWGTP"] + ["PWGTP{}".format(i) for i in range(1, 81)],
    "meta_cols": _s("RT SPORDER SERIALNO PUMA ST"),
    "alloc_flags": _s(
        """
            FAGEP FANCP FCITP FCITWP FCOWP FDDRSP FDEARP FDEYEP FDISP FDOUTP
            FDPHYP FDRATP FDRATXP FDREMP FENGP FESRP FFERP FFODP FGCLP FGCMP
            FGCRP FHINS1P FHINS2P FHINS3C FHINS3P FHINS4C FHINS4P
            FHINS5C FHINS5P FHINS6P FHINS7P FHISP FINDP FINTP FJWDP FJWMNP
            FJWRIP FJWTRP FLANP FLANXP FMARHDP FMARHMP FMARHTP FMARHWP FMARHYP
            FMARP FMIGP FMIGSP FMILPP FMILSP FOCCP FOIP FPAP FPERNP FPINCP
            FPOBP FPOWSP FPRIVCOVP FPUBCOVP FRACP FRELP FRETP FSCHGP FSCHLP
            FSCHP FSEMP FSEXP FSSIP FSSP FWAGP FWKHP FWKLP FWKWP FWRKP
            FYOEP
        """
    ),
    "discrete_feats": _s(
        """
            CIT COW DDRS DEAR DEYE DOUT DPHY DRAT DRATX DREM ENG FER GCL GCM
            GCR HINS1 HINS2 HINS3 HINS4 HINS5 HINS6 HINS7 JWRIP JWTR LANX MAR
            MARHD MARHM MARHT MARHW MARHYP MIG MIL MLPA MLPB MLPCD MLPE MLPFG
            MLPH MLPI MLPJ MLPK NWAB NWAV NWLA NWLK NWRE RELP SCH SCHG SCHL SEX
            WKL WKW WRK ANC ANC1P ANC2P DECADE DIS DRIVESP ESP ESR FOD1P FOD2P
            HICOV HISP INDP LANP MIGPUMA MIGSP MSP NAICSP NATIVITY NOP OC OCCP
            PAOC POBP POWPUMA POWSP PRIVCOV PUBCOV QTRBIR RAC1P RAC2P RAC3P
            RACAIAN RACASN RACBLK RACNH RACPI RACSOR RACWHT RC SCIENGP SCIENGRLP
            SFN SFR SOCP VPS WAOB
        """
    ),
    "real_feats": _s(
        """
            AGEP CITWP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP WKHP YOEP JWAP
            JWDP PERNP PINCP POVPIP RACNUM
        """
    ),
    "to_adjinc": _s("INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP"),
    "renames": {"pwgtp{}".format(i): "PWGTP{}".format(i) for i in range(1, 81)},
    "region_year": "10",
}


VERSIONS["2013-17"] = v = deepcopy(VERSIONS["2012-16"])
v["alloc_flags"] = sorted(v["alloc_flags"] + ["FHICOVP"])
v["drop_feats"] = sorted(v.get("drop_feats", []) + "REGION DIVISION".split())

VERSIONS["2014-18"] = deepcopy(VERSIONS["2013-17"])

VERSIONS["2015"] = deepcopy(VERSIONS["2013-17"])


VERSIONS["housing_2014-18"] = v = {
    "weight_cols": ["WGTP"] + [f"WGTP{i}" for i in range(1, 81)],
    "meta_cols": _s("RT SERIALNO PUMA ST"),
    "alloc_flags": _s(
        """
        FACCESSP FACRP FAGSP FBATHP FBDSP FBLDP FBROADBNDP FCOMPOTHXP
        FBUSP FCONP FDIALUPP FELEP FFINCP FFSP FFULP FGASP FGRNTP
        FHFLP FHINCP FHISPEEDP FHOTWATP FINSP FKITP FLAPTOPP FMHP FMRGIP
        FMRGP FMRGTP FMRGXP FMVP FOTHSVCEXP FPLMP FPLMPRP FREFRP FRMSP
        FRNTMP FRNTP FRWATP FRWATPRP FSATELLITEP FSINKP FSMARTPHONP
        FSMOCP FSMP FSMXHP FSMXSP FSTOVP FTABLETP FTAXP FTELP FTENP
        FTOILP FVACSP FVALP FVEHP FWATP FYBLP
        """
    ),
    "discrete_feats": _s(
        """
            NP TYPE
            ACCESS ACR AGS BATH BDSP BLD BUS BROADBND COMPOTHX DIALUP
            FS HFL HISPEED HOTWAT LAPTOP
            MRGI MRGT MRGX OTHSVCEX REFR RMSP RNTM RWAT RWATPR
            SATELLITE SINK SMARTPHONE STOV TABLET TEL TEN TOIL VACS
            VEH YBL FES
            FPARC
            HHL HHT HUGCL HUPAC HUPAOC HUPARC KIT LNGI MULTG
            MV NOC NPF NPP NR NRC PARTNER PLM PLMPRP PSF R18 R60 R65
            RESMODE SMX SRNT SSMC SVAL WIF WKEXREL WORKSTAT
            ELEFP FULFP GASFP WATFP
        """
    ),
    "real_feats": _s("VALP GRPIP OCPIP"),
    "to_adjhsg": _s(
        """
            CONP ELEP FULP GASP INSP MHP MRGP RNTP SMP WATP
            GRNTP SMOCP TAXAMT
        """
    ),
    "to_adjinc": _s("FINCP HINCP"),
    "drop_feats": _s("DIVISION REGION"),
    "region_year": "10",
}
v["real_feats"] = sorted(set(v["real_feats"]) | set(v["to_adjhsg"]))
v["real_feats"] = sorted(set(v["real_feats"]) | set(v["to_adjinc"]))


def version_info_with_housing(name, housing_name=None):
    if housing_name is None:
        housing_name = f"housing_{name}"

    h = VERSIONS[housing_name]
    v = deepcopy(VERSIONS[name])
    v["real_feats"] += h["real_feats"]
    v["discrete_feats"] += h["discrete_feats"]
    v["alloc_flags"] += h["alloc_flags"]
    v["weight_cols"] += h["weight_cols"]
    return v
