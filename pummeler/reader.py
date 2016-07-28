import pandas as pd
import zipfile


weirds = ("indp02 indp07 occp02 occp10 OCCP10 OCCP12 "
          "socp00 socp10 naicsp02 naicsp07".split())
adj_cols = "INTP OIP PAP PERNP PINCP RETP SEMP SSIP SSP WAGP".split()
def read_chunks(fname, chunksize=10**5, voters_only=False, adj_inc=None):
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
            for k in adj_cols:
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
        'region_year': '00',
    }
}

# feat_cols = set(discrete_feats) | set(alloc_flags) | set(real_feats)
# all_cols = set(meta_cols) | feat_cols
