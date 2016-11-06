import pandas as pd


weirds = ("indp02 indp07 occp02 occp10 OCCP10 OCCP12 "
          "socp00 socp10 naicsp02 naicsp07".split())
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
    for chunk in chunks:
        if voters_only:
            chunk = chunk[(chunk.AGEP >= 18) & (chunk.CIT != 5)]
        
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


VERSIONS = {
    '2006-10': {
        'weight_cols': ['PWGTP'] + ['PWGTP{}'.format(i) for i in range(1, 81)],
        'meta_cols': "RT SPORDER serialno PUMA ST".split(),
        'discrete_feats': '''CIT COW ENG FER GCL GCM GCR JWRIP JWTR LANX MAR
                             MIG MIL MLPA MLPB MLPC MLPD MLPE MLPF MLPG MLPH
                             MLPI MLPJ MLPK NWAB NWAV NWLA NWLK NWRE RELP SCH
                             SCHG SCHL SEX WKL WKW ANC ANC1P ANC2P DECADE
                             DRIVESP ESP ESR HISP indp02 indp07 LANP MIGPUMA
                             MIGSP MSP naicsp02 naicsp07 NATIVITY NOP OC occp02 
                             occp10 PAOC POBP POWPUMA POWSP QTRBIR RAC1P RAC2P
                             RAC3P RACAIAN RACASN RACBLK RACNHPI RACSOR RACWHT
                             RC SFN SFR socp00 socp10 VPS WAOB'''.split(),
        'alloc_flags': '''FAGEP FANCP FCITP FCOWP FENGP FESRP FFERP FGCLP
                          FGCMP FGCRP FHISP FINDP FINTP FJWDP FJWMNP FJWRIP
                          FJWTRP FLANP FLANXP FMARP FMIGP FMIGSP FMILPP FMILSP
                          FOCCP FOIP FPAP FPOBP FPOWSP FRACP FRELP FRETP FSCHGP
                          FSCHLP FSCHP FSEMP FSEXP FSSIP FSSP FWAGP FWKHP FWKLP
                          FWKWP FYOEP'''.split(),
        'real_feats': '''AGEP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP WKHP
                         YOEP JWAP JWDP PERNP PINCP POVPIP RACNUM'''.split(),
        'to_adjinc': '''INTP OIP PAP PERNP PINCP RETP SEMP SSIP
                        SSP WAGP'''.split(),
        'region_year': '00',
    },
    '2015': {
        'weight_cols': ['PWGTP'] + ['pwgtp{}'.format(i) for i in range(1, 81)],
        'meta_cols': "RT SPORDER SERIALNO PUMA ST".split(),
        'alloc_flags': '''FAGEP FANCP FCITP FCITWP FCOWP FDDRSP FDEARP FDEYEP
                          FDISP FDOUTP FDPHYP FDRATP FDRATXP FDREMP FENGP FESRP
                          FFERP FFODP FGCLP FGCMP FGCRP FHINS1P FHINS2P FHINS3C
                          FHINS3P FHINS4C FHINS4P FHINS5C FHINS5P FHINS6P
                          FHINS7P FHISP FINDP FINTP FJWDP FJWMNP FJWRIP FJWTRP
                          FLANP FLANXP FMARHDP FMARHMP FMARHTP FMARHWP FMARHYP
                          FMARP FMIGP FMIGSP FMILPP FMILSP FOCCP FOIP FPAP
                          FPERNP FPINCP FPOBP FPOWSP FPRIVCOVP FPUBCOVP FRACP
                          FRELP FRETP FSCHGP FSCHLP FSCHP FSEMP FSEXP FSSIP
                          FSSP FWAGP FWKHP FWKLP FWKWP FWRKP FYOEP'''.split(),
        'discrete_feats': '''CIT COW DDRS DEAR DEYE DOUT DPHY DRAT DRATX
                             DREM ENG FER GCL GCM GCR HINS1 HINS2 HINS3 HINS4
                             HINS5 HINS6 HINS7 JWRIP JWTR LANX MAR MARHD MARHM
                             MARHT MARHW MARHYP MIG MIL MLPA MLPB MLPCD MLPE
                             MLPFG MLPH MLPI MLPJ MLPK NWAB NWAV NWLA NWLK NWRE
                             RELP SCH SCHG SCHL SEX WKHP WKL WKW WRK ANC ANC1P
                             ANC2P DECADE DIS DRIVESP ESP ESR FHICOVP FOD1P
                             FOD2P HICOV HISP INDP LANP MIGPUMA MIGSP MSP
                             NAICSP NATIVITY NOP OC OCCP PAOC POBP POWPUMA
                             POWSP PRIVCOV PUBCOV QTRBIR RAC1P RAC2P RAC3P
                             RACAIAN RACASN RACBLK RACNH RACPI RACSOR RACWHT RC
                             SCIENGP SCIENGRLP SFN SFR SOCP VPS WAOB'''.split(),
        'real_feats': '''AGEP CITWP INTP JWMNP OIP PAP RETP SEMP SSIP SSP WAGP
                         WKHP YOEP JWAP JWDP PERNP PINCP POVPIP
                         RACNUM'''.split(),
        'to_adjinc': 'INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP'.split(),
        'region_year': '10',
    }
}
