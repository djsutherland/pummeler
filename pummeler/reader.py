import pandas as pd


weirds = ("indp02 indp07 INDP "
          "OCCP occp02 occp10 OCCP10 OCCP12 "
          "socp00 socp10 SOCP10 SOCP12 SOCP "
          "naicsp02 naicsp07 NAICSP".split())
def read_chunks(fname, version, chunksize=10**5, voters_only=False,
                adj_inc=None):
    info = VERSIONS[version]
    chunks = pd.read_csv(
        fname,
        skipinitialspace=True,
        na_values={k: ['N.A.', 'N.A.//', 'N.A.////'] for k in weirds},
        dtype={k: str for k in weirds},
        chunksize=chunksize,
    )
    renames = None
    for chunk in chunks:
        if info.get('puma_subset', False):
            puma_key = 'PUMA{}'.format(info['region_year'])
            chunk = chunk[chunk[puma_key] != -9]
            if chunk.shape[0] == 0:
                continue
            chunk['PUMA'] = chunk[puma_key]
            del chunk['PUMA00']
            del chunk['PUMA10']

        if voters_only:
            chunk = chunk[(chunk.AGEP >= 18) & (chunk.CIT != 5)]

        if chunk.shape[0] == 0:
            continue
        
        if 'drop_feats' in info:  # TODO: pass usecols to read_csv instead...
            chunk.drop(info['drop_feats'], axis=1, inplace=True)

        if 'renames' in info:
            if renames is None:
                renames = [info['renames'].get(k, k) for k in chunk.columns]
            chunk.columns = renames

        if adj_inc is None:
            if 'ADJINC' in chunk:
                adj_inc = True
            elif 'ADJINC_orig' in chunk:
                adj_inc = False
            else:
                raise ValueError("Unclear whether income has been adjusted, "
                                 "and adj_inc is None; pass either True or "
                                 "False explicitly")

        if adj_inc:
            adj = chunk.ADJINC / 1e6
            for k in info['to_adjinc']:
                chunk[k] *= adj
            chunk['ADJINC_orig'] = chunk.ADJINC
            del chunk['ADJINC']

        yield chunk

_s = lambda s: sorted(s.split())

VERSIONS = {
    '2006-10': {
        'weight_cols': ['PWGTP'] + ['PWGTP{}'.format(i) for i in range(1, 81)],
        'meta_cols': "RT SPORDER serialno PUMA ST".split(),
        'discrete_feats': _s('''
            CIT COW ENG FER GCL GCM GCR JWRIP JWTR LANX MAR MIG MIL MLPA MLPB
            MLPC MLPD MLPE MLPF MLPG MLPH MLPI MLPJ MLPK NWAB NWAV NWLA NWLK
            NWRE RELP SCH SCHG SCHL SEX WKL WKW ANC ANC1P ANC2P DECADE DRIVESP
            ESP ESR HISP indp02 indp07 LANP MIGPUMA MIGSP MSP naicsp02 naicsp07
            NATIVITY NOP OC occp02 occp10 PAOC POBP POWPUMA POWSP QTRBIR RAC1P
            RAC2P RAC3P RACAIAN RACASN RACBLK RACNHPI RACSOR RACWHT RC SFN SFR
            socp00 socp10 VPS WAOB'''),
        'alloc_flags': _s('''
            FAGEP FANCP FCITP FCOWP FENGP FESRP FFERP FGCLP FGCMP FGCRP
            FHISP FINDP FINTP FJWDP FJWMNP FJWRIP FJWTRP FLANP FLANXP FMARP
            FMIGP FMIGSP FMILPP FMILSP FOCCP FOIP FPAP FPOBP FPOWSP FRACP
            FRELP FRETP FSCHGP FSCHLP FSCHP FSEMP FSEXP FSSIP FSSP FWAGP
            FWKHP FWKLP FWKWP FYOEP'''),
        'real_feats': _s('''
            AGEP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP WKHP YOEP JWAP
            JWDP PERNP PINCP POVPIP RACNUM'''),
        'to_adjinc': _s('INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP'),
        'region_year': '00',
    },
    '2010-14_12-14': {
        'weight_cols': ['PWGTP'] + ['PWGTP{}'.format(i) for i in range(1, 81)],
        'meta_cols': _s("RT SPORDER serialno PUMA ST"),
        'alloc_flags': _s('''
            FAGEP FCITP FCITWP FCOWP FDDRSP FDEARP FDEYEP FDOUTP FDPHYP
            FDRATP FDRATXP FDREMP FENGP FESRP FFERP FFODP FGCLP FGCMP FGCRP
            FHINS1P FHINS2P FHINS3C FHINS3P FHINS4C FHINS4P FHINS5C FHINS5P
            FHINS6P FHINS7P FHISP FINDP FINTP FJWDP FJWMNP FJWRIP FJWTRP
            FLANP FLANXP FMARHDP FMARHMP FMARHTP FMARHWP FMARHYP FMARP FMIGP
            FMIGSP FMILPP FMILSP FOCCP FOIP FPAP FPOBP FPOWSP FRACP FRELP
            FRETP FSCHGP FSCHLP FSCHP FSEMP FSEXP FSSIP FSSP FWAGP FWKHP
            FWKLP FWKWP FWRKP FYOEP'''),
        'discrete_feats': _s('''
            CIT COW DDRS DEAR DEYE DOUT DPHY DRAT DRATX DREM ENG FER GCL GCM
            GCR HINS1 HINS2 HINS3 HINS4 HINS5 HINS6 HINS7 JWRIP JWTR LANX
            MAR MARHD MARHM MARHT MARHW MARHYP MIG MIL MLPA MLPB MLPCD MLPE
            MLPFG MLPH MLPI MLPJ MLPK NWAB NWAV NWLA NWLK NWRE RELP SCH SCHG
            SCHL SEX WKL WKW WRK ANC ANC1P ANC2P DECADE DIS DRIVESP ESP ESR
            FOD1P FOD2P HICOV HISP INDP LANP MIGPUMA MIGSP MSP NAICSP NATIVITY
            NOP OC OCCP PAOC POBP POWPUMA POWSP PRIVCOV PUBCOV QTRBIR RAC1P
            RAC2P RAC3P RACAIAN RACASN RACBLK RACNHPI RACSOR RACWHT RC SCIENGP
            SCIENGRLP SFN SFR SOCP VPS WAOB'''),
        'real_feats': _s('''
            AGEP CITWP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP WKHP YOEP
            JWAP JWDP PERNP PINCP POVPIP RACNUM'''),
        'to_adjinc': _s('INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP'),
        'drop_feats': _s('''
            ANC1P05 ANC2P05 FANCP LANP05 MARHYP05 MIGPUMA00 MIGSP05 POBP05
            POWPUMA00 POWSP05 RAC2P05 RAC3P05 SOCP10 YOEP05 CITWP05 OCCP10'''),
            # Always blank for 12-14 data
        'renames': {
            'ANC1P12': 'ANC1P', 'ANC2P12': 'ANC2P',
            'LANP12': 'LANP', 'MARHYP12': 'MARHYP',
            'MIGPUMA10': 'MIGPUMA', 'MIGSP12': 'MIGSP', 'OCCP12': 'OCCP',
            'POBP12': 'POBP', 'POWPUMA10': 'POWPUMA', 'POWSP12': 'POWSP',
            'RAC2P12': 'RAC2P', 'RAC3P12': 'RAC3P',
            'SOCP12': 'SOCP', 'YOEP12': 'YOEP', 'CITWP12': 'CITWP',
        },
        'region_year': '10',
        'puma_subset': True,
    },
    '2015': {
        'weight_cols': ['PWGTP'] + ['PWGTP{}'.format(i) for i in range(1, 81)],
        'meta_cols': _s("RT SPORDER SERIALNO PUMA ST"),
        'alloc_flags': _s('''
            FAGEP FANCP FCITP FCITWP FCOWP FDDRSP FDEARP FDEYEP FDISP FDOUTP
            FDPHYP FDRATP FDRATXP FDREMP FENGP FESRP FFERP FFODP FGCLP FGCMP
            FGCRP FHICOVP FHINS1P FHINS2P FHINS3C FHINS3P FHINS4C FHINS4P
            FHINS5C FHINS5P FHINS6P FHINS7P FHISP FINDP FINTP FJWDP FJWMNP
            FJWRIP FJWTRP FLANP FLANXP FMARHDP FMARHMP FMARHTP FMARHWP FMARHYP
            FMARP FMIGP FMIGSP FMILPP FMILSP FOCCP FOIP FPAP FPERNP FPINCP
            FPOBP FPOWSP FPRIVCOVP FPUBCOVP FRACP FRELP FRETP FSCHGP FSCHLP
            FSCHP FSEMP FSEXP FSSIP FSSP FWAGP FWKHP FWKLP FWKWP FWRKP
            FYOEP'''),
        'discrete_feats': _s('''
            CIT COW DDRS DEAR DEYE DOUT DPHY DRAT DRATX DREM ENG FER GCL GCM
            GCR HINS1 HINS2 HINS3 HINS4 HINS5 HINS6 HINS7 JWRIP JWTR LANX MAR
            MARHD MARHM MARHT MARHW MARHYP MIG MIL MLPA MLPB MLPCD MLPE MLPFG
            MLPH MLPI MLPJ MLPK NWAB NWAV NWLA NWLK NWRE RELP SCH SCHG SCHL SEX
            WKL WKW WRK ANC ANC1P ANC2P DECADE DIS DRIVESP ESP ESR FOD1P FOD2P
            HICOV HISP INDP LANP MIGPUMA MIGSP MSP NAICSP NATIVITY NOP OC OCCP
            PAOC POBP POWPUMA POWSP PRIVCOV PUBCOV QTRBIR RAC1P RAC2P RAC3P
            RACAIAN RACASN RACBLK RACNH RACPI RACSOR RACWHT RC SCIENGP SCIENGRLP
            SFN SFR SOCP VPS WAOB'''),
        'real_feats': _s('''
            AGEP CITWP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP WKHP YOEP JWAP
            JWDP PERNP PINCP POVPIP RACNUM'''),
        'to_adjinc': _s('INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP'),
        'renames': {
            'pwgtp{}'.format(i): 'PWGTP{}'.format(i) for i in range(1, 81)},
        'region_year': '10',
    }
}

VERSIONS['2007-11'] = VERSIONS['2006-10']
VERSIONS['2012-15_manual'] = VERSIONS['2010-14_12-14']

VERSIONS['2011-15_12-15'] = v = VERSIONS['2010-14_12-14'].copy()
v['alloc_flags'] = sorted(
    v['alloc_flags'] + 'FDISP FPINCP FPUBCOVP FPERNP FPRIVCOVP'.split())
