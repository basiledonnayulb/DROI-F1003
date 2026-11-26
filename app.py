"""Streamlit: aggregate migration-detection statistics (NOT individual status)."""
from __future__ import annotations
import json
import csv
from io import StringIO
from pathlib import Path
import joblib
import sklearn
import plotly.graph_objects as go
import streamlit as st
from src.model import read_training, choose_rows, summarize_cohort, predicted_pct

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
st.set_page_config(page_title='EU Migration Data Lab',page_icon='◉',layout='wide',
                   initial_sidebar_state='expanded')
st.markdown('''<style>
  .block-container {max-width: 1230px; padding-top: 2.1rem}
  [data-testid="stSidebar"] {background:#f1f5f0}
  h1,h2,h3 {letter-spacing:-.028em}
  h1 {font-size:2.35rem !important}
  .eyebrow {font-size:.75rem;letter-spacing:.13em;text-transform:uppercase;font-weight:750;color:#5b7a66}
  .hero {background:linear-gradient(120deg,#eff6ef,#e7f1e8);border:1px solid #d5e5d8;padding:1.6rem 2rem;border-radius:17px;margin: .9rem 0 1.4rem}
  .hero .n {font-weight:750;letter-spacing:-.055em;font-size:3.8rem;color:#1b442d;line-height:1.2}
  .hero .small {color:#53685c;font-size:.93rem}
  .metriclabel {font-size:.85rem;color:#506255}
  div[data-testid="stMetric"] {background:#fff;border:1px solid #dee7e0;border-radius:13px;padding:.95rem 1.2rem}
  .pale {background:#f5f7f4;border:1px solid #e0e8e1;border-radius:10px;padding:13px 17px;font-size:.9rem}
  .pill {display:inline-block;background:#e2f0e4;border-radius:50px;padding:4px 10px;font-weight:700;font-size:.73rem;color:#356344;margin:0 0 10px 0}
</style>''', unsafe_allow_html=True)

@st.cache_resource

def get_model(): return joblib.load(DATA/'model.joblib')
@st.cache_data

def get_rows(): return read_training(DATA/'training.csv')

if sklearn.__version__ != '1.8.0':
    st.error(
        f"The bundled pretrained model requires scikit-learn 1.8.0, "
        f"but this environment has {sklearn.__version__}. "
        "In your activated virtual environment, run "
        "`python -m pip install --force-reinstall scikit-learn==1.8.0` "
        "and restart Streamlit."
    )
    st.stop()

if not (DATA/'training.csv').exists() or not (DATA/'model.joblib').exists():
    st.error('Training artifacts missing. Run `python train.py` first.')
    st.stop()
rows=get_rows(); models=get_model()
summary=json.loads((DATA/'training_summary.json').read_text(encoding='utf-8'))
country_options=sorted({r['country'] for r in rows})
origin_options=sorted({r['citizenship'] for r in rows if r['citizenship']!='Total'})
years=sorted({r['year'] for r in rows},reverse=True)

with st.sidebar:
    st.markdown('### Explorer settings')
    st.caption('Leave categories unrestricted to aggregate over all available records.')
    age=st.selectbox('Age group', ['All ages','Under 18','18–34','35+'],index=0)
    sex=st.selectbox('Sex', ['All','Males','Females'])
    country=st.selectbox('Reporting country', ['All']+country_options)
    origin=st.selectbox('Citizenship', ['All third-country citizenships']+origin_options)
    year=st.select_slider('Reference year', options=sorted(years), value=max(years))
    st.divider()
    st.info('The output is an **aggregate detection-rate benchmark**, not a prediction of anyone’s immigration status.')
    st.caption('Eurostat • 2018–2025 • 27 EU reporting countries')

citizenship='Total' if origin=='All third-country citizenships' else origin
selected=choose_rows(rows,age,sex,country,citizenship,year)
observed=summarize_cohort(selected)
model_pct=predicted_pct(selected,models) if selected else None

st.markdown('<div class="eyebrow">EU MIGRATION DATA LAB / RESEARCH PREVIEW</div>',unsafe_allow_html=True)
st.title('Migration detection-rate explorer')
st.write('Explore annual **detected irregular-presence cases** relative to a population benchmark, using a trained gradient-boosting model and two Eurostat datasets.')
if observed is None:
    st.warning('No matched data exist for this combination. Try another year, reporting country or citizenship.')
    st.stop()

st.markdown(f'''<div class="hero"><span class="pill">MODEL ESTIMATE · AGGREGATE</span>
<div class="small">Estimated annual detections / population benchmark</div>
<div class="n">{model_pct:.3f}%</div>
<div class="small">≈ {model_pct*1000:,.0f} modeled detections per 100,000 population-benchmark members</div>
</div>''',unsafe_allow_html=True)
st.warning('**Not a probability of illegal presence.** Detected cases are annual enforcement observations; the comparator is registered resident population on 1 January. These are not matching individual-level outcome groups.')

c1,c2,c3=st.columns(3)
c1.metric('Observed benchmark',f"{observed['rate_pct']:.3f}%",help='Annual detections divided by resident population benchmark; not a prevalence probability')
c2.metric('Recorded detections',f"{observed['detections']:,}")
c3.metric('Population benchmark',f"{observed['population_proxy']:,}")
st.caption(f"Based on {observed['n_cells']:,} matched records across {observed['countries']} reporting countries. Missing observations are not counted as zero.")
if observed['age_approximate']:
    st.info('**Approximate age denominator:** Population ages 15–64 were split uniformly into single ages to match the irregular-presence age groups. This assumption may be inaccurate. Download finer age bands for a stronger study.')

st.markdown('### Trend across years')
points=[]
for yr in sorted(years):
    same=choose_rows(rows,age,sex,country,citizenship,yr)
    agg=summarize_cohort(same)
    if agg:
        points.append((yr,agg['rate_pct'],predicted_pct(same,models)))
if points:
    graph=go.Figure()
    graph.add_trace(go.Scatter(x=[x[0] for x in points],y=[x[1] for x in points],mode='lines+markers',name='Recorded benchmark',line={'color':'#296646','width':3}))
    graph.add_trace(go.Scatter(x=[x[0] for x in points],y=[x[2] for x in points],mode='lines+markers',name='Model estimate',line={'color':'#9cb9a1','width':3,'dash':'dash'}))
    graph.update_layout(height=325,margin=dict(l=10,r=15,t=15,b=20),
        yaxis_title='Detections per population benchmark (%)',xaxis_title='Year',
        xaxis={'dtick':1},paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h',y=1.15),hovermode='x unified')
    st.plotly_chart(graph,use_container_width=True)

st.markdown('### Compare age groups')
age_rows=[]
for band in ['Under 18','18–34','35+']:
    sample=choose_rows(rows,band,sex,country,citizenship,year)
    agg=summarize_cohort(sample)
    if agg: age_rows.append((band,agg['rate_pct']))
if age_rows:
    bars=go.Figure(go.Bar(x=[x[0] for x in age_rows],y=[x[1] for x in age_rows],
        marker_color=['#aac9b2','#528968','#305b44'],text=[f'{x[1]:.2f}%' for x in age_rows],textposition='outside'))
    bars.update_layout(height=295,margin=dict(l=10,r=15,t=16,b=20),
        yaxis_title='Observed benchmark (%)',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(bars,use_container_width=True)
    st.caption('All age-specific comparisons use approximate denominators; they are not individual legal-status risks.')

st.markdown('### Compare citizenships')
# Compare the top 15 citizenships by recorded detections.
# Combine every remaining citizenship into "Other", including small cohorts.
citizenship_groups = {}
for r in rows:
    if r['citizenship'] == 'Total' or r['year'] != year or r['age'] != age:
        continue
    if sex != 'All' and r['sex'] != sex:
        continue
    if country != 'All' and r['country'] != country:
        continue
    citizenship_groups.setdefault(r['citizenship'], []).append(r)

citizenship_stats = {}
for name, cohort in citizenship_groups.items():
    stats = summarize_cohort(cohort)
    if stats and stats['population_proxy'] > 0:
        citizenship_stats[name] = stats

if citizenship_stats:
    # Keep the original 1,000-person threshold for individually displayed groups.
    # Smaller groups are still counted in "Other"; they are not discarded.
    top_15 = sorted(
        (name for name, stats in citizenship_stats.items()
         if stats['population_proxy'] >= 1000),
        key=lambda name: citizenship_stats[name]['detections'],
        reverse=True,
    )[:15]

    # Aggregate counts BEFORE dividing: never average citizenship percentages.
    other_rows = [
        row
        for name, cohort in citizenship_groups.items()
        if name not in top_15
        for row in cohort
    ]
    other_stats = summarize_cohort(other_rows)

    display_stats = {name: citizenship_stats[name] for name in top_15}
    if other_stats and other_stats['population_proxy'] > 0:
        display_stats['Other'] = other_stats

    # Plotly horizontal bar charts draw the first category at the bottom by
    # default. Reversing the axis puts the top-ranked group first and Other last.
    shown = sorted(
        top_15,
        key=lambda name: display_stats[name]['rate_pct'],
        reverse=True
    )

    if 'Other' in display_stats:
        shown.append('Other')
    
    fig_cit = go.Figure(go.Bar(
        x=[display_stats[name]['rate_pct'] for name in shown],
        y=shown,
        orientation='h',
        text=[f"{display_stats[name]['rate_pct']:.2f}%" for name in shown],
        textposition='outside',
        marker_color=[
            '#a1aaa3' if name == 'Other'
            else '#296646' if name == origin
            else '#78a68a'
            for name in shown
        ],
        customdata=[
            [display_stats[name]['detections'],
             display_stats[name]['population_proxy']]
            for name in shown
        ],
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Observed benchmark: %{x:.2f}%<br>'
            'Recorded detections: %{customdata[0]:,.0f}<br>'
            'Resident population benchmark: %{customdata[1]:,.0f}'
            '<extra></extra>'
        ),
    ))
    fig_cit.update_layout(
        height=max(400, 34 * len(shown) + 115),
        margin=dict(l=10, r=65, t=15, b=20),
        xaxis_title='Observed detection benchmark (%)',
        yaxis_title='Citizenship',
        yaxis=dict(autorange='reversed'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_cit, use_container_width=True)
    st.caption(
        'Top 15 ranked by recorded detections (not by percentage). '
        'Other combines all remaining available citizenship groups, '
        'including groups with fewer than 1,000 benchmark residents; '
        'its percentage is total detections divided by total benchmark population. '
        'These are detection/population ratios, not individual probabilities.'
    )
else:
    st.info('No citizenship comparison data are available for these filters.')

with st.expander('Model quality and data methodology',expanded=False):
    key='all_citizenships' if citizenship=='Total' else 'specific_citizenship'
    val=summary['validation_2025'][key]
    st.write(f"**Model:** histogram-based gradient-boosting Poisson regression · **Training records:** {summary['training_rows']:,} · **Holdout:** year 2025")
    if val['mae_percentage_points'] is not None:
        st.write(f"**2025 holdout mean absolute error:** {val['mae_percentage_points']:.3f} percentage points across {val['rows']:,} cells. This measures aggregate benchmark prediction error, not classification accuracy.")
    st.write('**Age alignment:**',summary['age_denominator_assumption'])
    st.write('**Data definitions:** EU irregular-presence counts are annual detections (and may include multiple observations); Eurostat population is measured on 1 January and includes residents regardless of immigration-law verification. These are not strict disjoint legal/illegal labels.')
    st.write('**Excluded:** missing/suppressed cells, incomplete 3-band population groups and cells where annual detections exceed resident population benchmark.')
    st.write('**Risk of bias:** enforcement and reporting practices, different population scopes and visitor records affect comparisons. Never use these estimates to evaluate people.')
    st.markdown('Sources: [Eurostat migr_eipre](https://ec.europa.eu/eurostat/databrowser/view/migr_eipre/default/table?lang=en) · [Eurostat migr_pop1ctz](https://ec.europa.eu/eurostat/databrowser/view/migr_pop1ctz/default/table?lang=en)')

export=StringIO()
writer=csv.writer(export)
writer.writerow(['country','year','citizenship','sex','age','detections','population_proxy','rate_per_100k'])
for r in selected:
    writer.writerow([r[k] for k in ('country','year','citizenship','sex','age','detections','population_proxy','rate_per_100k')])
st.download_button('Download current cohort as CSV',data=export.getvalue(),
    file_name='migration_detection_cohort.csv',mime='text/csv',use_container_width=False)
