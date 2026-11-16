"""Directions and offline integration examples, built only from raw PS3 data and fitted predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path('.')
RAW=Path('dataset')
OUT=Path('submission_outputs')

def nfloat(value):
    try: return float(value)
    except (TypeError, ValueError): return float('nan')

def label(value):
    if pd.isna(value) or str(value).strip()=='':return ''
    return str(value).replace('_',' ').strip()

def street_parts(r, localities):
    pieces=[]
    for key,noun in [('house_no','No.'),('cross_no','Cross'),('main_no','Main'),('gali_no','Gali'),('road_no','Road'),('ward_no','Ward')]:
        v=r.get(key)
        if pd.isna(v) or str(v).strip()=='': continue
        try:
            k=float(v);v=str(int(k)) if k.is_integer() else str(v)
        except (ValueError,TypeError):v=str(v)
        pieces.append(f'{noun} {v}')
    loc=r.get('locality_id')
    if pd.notna(loc) and str(loc) in localities:
        pieces.append(localities[str(loc)])
    else:
        town=r.get('town_id')
        if pd.notna(town):pieces.append(f'Town {town}')
    return ', '.join(pieces[:5])

def build_outputs(raw_dir, root=Path('.')):
    raw_dir=Path(raw_dir);root=Path(root);out=root/'submission_outputs';out.mkdir(parents=True,exist_ok=True)
    master=pd.read_csv(root/'ps3_out'/'ps3_master_processed.csv')
    pin=pd.read_csv(root/'stage2_arch52'/'architecture52_all_addresses.csv')
    conf=pd.read_csv(root/'stage2_confidence20'/'confidence20_all_addresses.csv')
    poi=pd.read_csv(raw_dir/'landmarks_poi.csv')
    loc=pd.read_csv(raw_dir/'localities.csv')
    survey=pd.read_csv(raw_dir/'surveyed_addresses.csv')
    visits=pd.read_csv(root/'ps3_out'/'visit_features.csv')
    accounts=pd.read_csv(raw_dir/'accounts.csv')
    if not all(v.address_id.is_unique for v in [master,pin,conf]):raise ValueError('Duplicate address ID')
    cols=['address_id','A52_pin_x','A52_pin_y','A52_method','evidence_tier']
    data=master.merge(pin[cols],on='address_id',validate='one_to_one',how='left')
    data=data.merge(conf[['address_id','confidence20_radius_m','confidence_tier','confidence20_calibration_level']],on='address_id',validate='one_to_one',how='left')
    if len(data)!=len(master):raise AssertionError('A master address was lost')
    data=data.rename(columns={'A52_pin_x':'pin_x','A52_pin_y':'pin_y','confidence20_radius_m':'confidence_radius_m'})
    if (not np.isfinite(data[['pin_x','pin_y','confidence_radius_m']].to_numpy(float)).all()
        or (data.confidence_radius_m <= 0).any()):
        raise AssertionError('Missing pins or invalid confidence radii')
    poimap=poi.set_index('poi_id').to_dict(orient='index')
    locmap=loc.set_index('locality_id').locality_name.astype(str).to_dict()
    types=[];sources=[];quals=[];texts=[];active_poi=[];flags=[];anchor_names=[];distances=[]
    for r in data.to_dict('records'):
        street=street_parts(r,locmap);tail=f' — {street}' if street else ''
        poid=r.get('poi_id'); family=label(r.get('lm_primary'))
        confidence=nfloat(r['confidence_radius_m'])
        d=np.hypot(nfloat(r['pin_x'])-nfloat(r.get('poi_x')), nfloat(r['pin_y'])-nfloat(r.get('poi_y')))
        valid_named=pd.notna(poid) and str(poid) in poimap
        suspect=bool(valid_named and nfloat(r.get('clean_visits'))>0 and d >= max(1000,2*confidence))
        if valid_named and not suspect:
            record=poimap[str(poid)]
            name=str(record.get('name','')).strip() or label(record.get('landmark_type'))
            ptype=label(record.get('landmark_type'))
            t=f'Reference POI: {name} ({ptype}){tail}. Confirm location using the pin; landmark identity is provisional.'
            direction='named_poi_provisional';quality='identity_unverified';source='stage1_matched_poi';assigned=poid
        elif family:
            phrase=('Review needed: ' if suspect else '')+f'Address mentions {family}{tail}. Specific landmark identity is unverified.'
            t=phrase;direction='family_only';quality='family_reference';source='parser_family';assigned=np.nan
        else:
            t=(f'Use the corrected pin{tail}.' if street else 'Use the corrected pin; address has limited reference details.')
            direction='street_only';quality='address_only';source='structured_address';assigned=np.nan
            if suspect:t='Review needed: '+t
        types.append(direction);sources.append(source);quals.append(quality);texts.append(t)
        active_poi.append(assigned);flags.append(suspect);anchor_names.append(str(poimap.get(str(poid),{}).get('name','')) if valid_named else '')
        distances.append(d if valid_named else np.nan)
    data['direction_type']=types;data['direction_quality']=quals;data['landmark_direction']=texts
    data['anchor_source']=sources;data['anchor_poi_id']=active_poi
    data['manual_review']=flags;data['raw_matched_poi_name']=anchor_names
    data['raw_matched_poi_distance_m']=distances
    useful=['address_id','account_id','town_id','pin_x','pin_y','confidence_radius_m','A52_method',
            'evidence_tier','confidence_tier','landmark_direction','direction_type','direction_quality',
            'anchor_source','anchor_poi_id','manual_review','clean_visits','flagged_visits','avg_integrity_risk']
    final=data[useful].copy();final.to_csv(out/'final_predictions.csv',index=False)
    review=data.loc[data.manual_review,['address_id','account_id','poi_id','lm_primary','raw_matched_poi_name',
                                        'raw_matched_poi_distance_m','confidence_radius_m','clean_visits']]
    review.to_csv(out/'named_poi_review_queue.csv',index=False)
    # Local, offline planning features; routing is straight-line only without a road graph.
    planner=final[['address_id','account_id','town_id','pin_x','pin_y','confidence_radius_m','direction_type','clean_visits','flagged_visits']].copy()
    planner['location_uncertainty_score']=np.clip(np.log1p(planner.confidence_radius_m)/np.log1p(5000),0,1)
    planner['verification_priority']=np.clip(.75*planner.location_uncertainty_score + .25*(planner.clean_visits.eq(0)),0,1)
    planner['route_pin_reliable']=planner.confidence_radius_m.le(250) & (~planner.direction_type.eq('street_only'))
    planner.to_csv(out/'visit_planner_features.csv',index=False)
    # Geocoding inputs to an RPC/PS2 model. These are features, not PS2 predictions.
    ps2=final[['address_id','account_id','confidence_radius_m','clean_visits','flagged_visits','direction_type','avg_integrity_risk']].copy()
    ps2=ps2.merge(accounts[['account_id','town_id','portfolio','bucket_start','dpd_start']],on='account_id',how='left',validate='many_to_one')
    ps2['location_confidence_proxy']=1/(1+ps2.confidence_radius_m/500)
    ps2.to_csv(out/'ps2_location_features.csv',index=False)
    oof=pd.read_csv(root/'stage2_arch52'/'architecture52_oof_survey_predictions.csv')
    c_oof=pd.read_csv(root/'stage2_confidence20'/'confidence20_oof_predictions.csv')
    def summarize(s):
        a=pd.to_numeric(s,errors='coerce').dropna();return {'median_m':float(a.median()),'p90_m':float(a.quantile(.9)),'mean_m':float(a.mean())}
    pin_error=summarize(oof['A52_oof_error_m'])
    baseline_error=summarize(oof['baseline_error_m'])
    err_col=next((c for c in ['actual_error_m','A52_oof_error_m'] if c in c_oof.columns),None)
    rad_col=next((c for c in ['conf20_radius_m','confidence20_radius_m','confidence_radius_m'] if c in c_oof.columns),None)
    radius_cov=float((pd.to_numeric(c_oof[err_col],errors='coerce')<=pd.to_numeric(c_oof[rad_col],errors='coerce')).mean()) if err_col and rad_col else None
    metrics={'address_count':int(len(final)),'survey_count':int(len(survey)),
             'pin_oof':pin_error,'baseline_oof':baseline_error,'confidence_oof_coverage':radius_cov,
             'confidence_production_median_radius_m':float(final.confidence_radius_m.median()),
             'direction_counts':final.direction_type.value_counts().to_dict(),
             'direction_note':'Named POIs are provisional, family-only directions do not claim POI identity.',
             'flagged_named_poi_cases':int(final.manual_review.sum()),
             'planner_note':'Visit Planner export is feature interface and straight-line heuristic; no routing uplift estimated.',
             'ps2_note':'PS2 export is a feature interface; no Right-Party Contact outcome model trained.',
             'self_learning_note':'The notebook re-fits from raw visits when rerun; see retrospective feedback demonstration.'}
    (out/'summary_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
    return final,planner,ps2,metrics

if __name__=='__main__':
    print('Import build_outputs(raw_dir) from the notebook.')
