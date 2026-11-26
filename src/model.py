"""Aggregate annual irregular-presence DETECTION rate model.

NOT an estimate of individual immigration status, illegal-presence prevalence,
or a classifier with legal/illegal individual labels.
"""
from __future__ import annotations
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

BASE=Path(__file__).resolve().parent.parent
DATA=BASE/'data'
FEATURES_TOTAL=('country','sex','age')
FEATURES_CIT=('country','citizenship','sex','age')


def read_training(path=DATA/'training.csv'):
    result=[]
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            row['year']=int(row['year'])
            for k in ('detections','population_proxy','rate_per_100k'):
                row[k]=float(row[k])
            row['age_approximate']=bool(int(row['age_approximate']))
            result.append(row)
    return result


def transform(rows, cat_maps, cat_features):
    return np.array([[cat_maps[field].get(str(row[field]), -1) for field in cat_features]
                     + [row['year']] for row in rows], dtype='float64')


def evaluate(rows, model, X):
    mask=np.array([r['year']==2025 for r in rows])
    if not np.any(mask): return {'rows':0,'mae_percentage_points':None}
    y=np.array([r['rate_per_100k'] for r in rows])
    w=np.array([r['population_proxy'] for r in rows])
    predictions=model.predict(X[mask]); obs=y[mask]
    # Weighted absolute error in rate as percentage points, NOT accuracy of
    # individual legal status.
    mae = np.average(np.abs(predictions-obs)/1000,weights=np.sqrt(w[mask]))
    return {'rows':int(mask.sum()),'mae_percentage_points':round(float(mae),4)}


def train(path=DATA/'training.csv',save=DATA/'model.joblib',summary=DATA/'training_summary.json'):
    rows=read_training(path)
    bundles={}; evaluations={}
    for key,features in [('all_citizenships',FEATURES_TOTAL),('specific_citizenship',FEATURES_CIT)]:
        subset=[r for r in rows if (r['citizenship']=='Total') == (key=='all_citizenships')]
        cat_maps={name:{v:i for i,v in enumerate(sorted(set(str(r[name]) for r in subset)))}
                  for name in features}
        X=transform(subset,cat_maps,features)
        y=np.array([r['rate_per_100k'] for r in subset])
        weights=np.sqrt(np.clip(np.array([r['population_proxy'] for r in subset]),1,1e6))
        historical=np.array([r['year']<2025 for r in subset]); latest=~historical
        params=dict(loss='poisson',learning_rate=.065,max_iter=90,max_leaf_nodes=15,
                    min_samples_leaf=20, max_bins=255, l2_regularization=10,
                    categorical_features=list(range(len(features))),early_stopping=False,random_state=42)
        if historical.sum()>100 and latest.sum()>20:
            validation=HistGradientBoostingRegressor(**params)
            validation.fit(X[historical],y[historical],sample_weight=weights[historical])
            evaluations[key]=evaluate(subset,validation,X)
        else:
            evaluations[key]={'rows':0,'mae_percentage_points':None}
        fitted=HistGradientBoostingRegressor(**params)
        fitted.fit(X,y,sample_weight=weights)
        bundles[key]={'model':fitted,'maps':cat_maps,'features':features}
        print(f"Trained {key}: {len(subset):,} rows, test={evaluations[key]}",flush=True)
    joblib.dump(bundles,save,compress=3)
    info={
      'target':'Annual detected irregular-presence cases / population on 1 January',
      'not_a_probability_of_individual_status':True,
      'data_years':sorted(set(r['year'] for r in rows)),
      'training_rows':len(rows),'countries':len(set(r['country'] for r in rows)),
      'citizenships':len(set(r['citizenship'] for r in rows)),
      'age_denominator_assumption':'15–64 population uniformly distributed across 50 ages for Under 18, 18–34, 35+; All ages exact sum of reported population bands',
      'validation_2025':evaluations,
      'warnings':[
       'Population on January 1 is not a verified legal-status population.',
       'Annual detections are events and are not an estimate of every person illegally present.',
       'The population at risk of detection includes short-stay visitors who may be omitted from registered population counts.',
       'Age-specific denominators rely on a strong uniform-age assumption.',
       'Counts are rounded, some missing/suppressed, and outlier cells with detections exceeding population proxy are excluded.',
       'Model outputs MUST NOT be used to judge an individual or make enforcement decisions.'
      ]
    }
    summary.write_text(json.dumps(info,indent=2,ensure_ascii=False),encoding='utf-8')
    return bundles,info


def choose_rows(rows,age='All ages',sex='All',country='All',citizenship='Total',year=2025):
    """For unspecified categories, marginalise via population-weighted summation.
    Citizenship='Total' uses the provider's overall detected third-country
    nationals series (not an unsafe sum of overlapping nationality categories).
    """
    return [r for r in rows if r['age']==age and r['year']==year
            and (sex=='All' or r['sex']==sex)
            and (country=='All' or r['country']==country)
            and r['citizenship']==citizenship]


def summarize_cohort(rows):
    if not rows:return None
    people=sum(r['population_proxy'] for r in rows)
    cases=sum(r['detections'] for r in rows)
    return {
      'detections':round(cases),'population_proxy':round(people),
      'rate_pct':cases/people*100 if people else None,
      'rate_per_100k':cases/people*100_000 if people else None,
      'countries':len({r['country'] for r in rows}),
      'age_approximate':any(r['age_approximate'] for r in rows),
      'n_cells':len(rows)
    }


def predicted_pct(rows,bundles):
    if not rows:return None
    which='all_citizenships' if rows[0]['citizenship']=='Total' else 'specific_citizenship'
    bundle=bundles[which]
    X=transform(rows,bundle['maps'],bundle['features'])
    prediction=bundle['model'].predict(X)
    weights=np.array([r['population_proxy'] for r in rows])
    return float(np.average(prediction,weights=weights)/1000) # per 100k -> percent

if __name__=='__main__':train()
