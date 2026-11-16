"""Combine Stage 1's parsed addresses, GPS evidence, and visit reliability."""
from pathlib import Path
import numpy as np
import pandas as pd


def build_master(work_dir=Path('.')):
    work_dir=Path(work_dir)
    parsed=pd.read_csv(work_dir/'ps3_out'/'address_parsed.csv')
    evidence=pd.read_csv(work_dir/'ps3_out'/'address_evidence.csv')
    visits=pd.read_csv(work_dir/'ps3_out'/'visit_features.csv')
    needed=['address_id','geocoder_x','geocoder_y','base_precision','is_surveyed','pt_k','pt_spread_m','pt_x','pt_y','split']
    if not parsed.address_id.is_unique or not evidence.address_id.is_unique:
        raise ValueError('Source address IDs are not unique')
    master=parsed.merge(evidence[needed], on='address_id', how='left', validate='one_to_one')
    grouped=visits.groupby('address_id',dropna=False).agg(
        total_visits=('visit_id','size'),
        clean_visits=('tier',lambda x:(x=='clean').sum()),
        flagged_visits=('tier',lambda x:(x=='flagged').sum()),
        avg_integrity_risk=('integrity_risk','mean'),
        agents_involved=('agent_id','nunique'),
    ).reset_index()
    master=master.merge(grouped,on='address_id',how='left',validate='one_to_one')
    for c in ['total_visits','clean_visits','flagged_visits','agents_involved']:
        master[c]=master[c].fillna(0).astype(int)
    if len(master)!=len(parsed) or master.address_id.isna().any():
        raise AssertionError('Address merge changed cardinality')
    target=work_dir/'ps3_out'/'ps3_master_processed.csv'
    master.to_csv(target,index=False)
    return master

if __name__=='__main__':
    print('Built master:',build_master().shape)
