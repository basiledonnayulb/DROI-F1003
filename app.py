"""Streamlit: aggregate migration-detection statistics (NOT individual status)."""
from __future__ import annotations
import json
import csv
from io import StringIO
from pathlib import Path
import joblib
import sklearn
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
import streamlit.components.v1 as components
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
st.caption('Each bar is the total observed detection / population benchmark, '
           'split by the sex recorded in the source data. '
           'Hover to see both the contribution to the total and the sex-specific rate.')

# Use the full sex breakdown regardless of the sidebar's sex selection so
# that women and men can always be compared on the same chart.
AGE_BANDS = ['Under 18', '18–34', '35+']
SEX_SEGMENTS = [
    ('Women', 'Females', '#c68eab'),  # muted pink
    ('Men', 'Males', '#6473b5'),      # matches the map's European blue
    ('Unknown', None, '#8c9599'),   # map's neutral grey
]

age_profiles = []
for band in AGE_BANDS:
    band_rows = choose_rows(rows, band, 'All', country, citizenship, year)
    if not band_rows:
        continue
    population = sum(float(r['population_proxy']) for r in band_rows)
    detections = sum(float(r['detections']) for r in band_rows)
    if population <= 0:
        continue

    female_rows = [r for r in band_rows if r['sex'] == 'Females']
    male_rows = [r for r in band_rows if r['sex'] == 'Males']
    # Any other recorded sex category (if it becomes available in future data)
    # is grouped as Unknown. The CURRENT training data has none.
    unknown_rows = [r for r in band_rows if r['sex'] not in ('Females', 'Males')]

    groups = {'Women': female_rows, 'Men': male_rows, 'Unknown': unknown_rows}
    segments = {}
    for label, sex_rows in groups.items():
        n = sum(float(r['detections']) for r in sex_rows)
        n_pop = sum(float(r['population_proxy']) for r in sex_rows)
        segments[label] = {
            'detections': n,
            'population': n_pop,
            # This CONTRIBUTION is what is stacked: the denominator is the
            # SAME across women, men and unknown, so bars add up to the total.
            'contribution_pct': 100 * n / population,
            # The within-sex rate is different and cannot be stacked.
            'sex_rate_pct': (100 * n / n_pop) if n_pop > 0 else None,
        }

    age_profiles.append({
        'age': band,
        'total_pct': 100 * detections / population,
        'population': population,
        'detections': detections,
        'segments': segments,
    })

if age_profiles:
    age_names = [p['age'] for p in age_profiles]
    bars = go.Figure()
    for label, _sex, color in SEX_SEGMENTS:
        segment_values = [p['segments'][label]['contribution_pct'] for p in age_profiles]
        bars.add_trace(go.Bar(
            name=label,
            x=age_names,
            y=segment_values,
            marker_color=color,
            text=[f'{v:.2f}%' if v > 0 else '' for v in segment_values],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(color='white', size=12),
            customdata=[[
                p['segments'][label]['detections'],
                p['segments'][label]['population'],
                p['segments'][label]['sex_rate_pct']
                if p['segments'][label]['sex_rate_pct'] is not None else -1,
                p['total_pct'],
            ] for p in age_profiles],
            hovertemplate=(
                '<b>%{x} · ' + label + '</b><br>'
                'Contribution to total rate: %{y:.3f}%<br>'
                'Recorded detections: %{customdata[0]:,.0f}<br>'
                'Sex-specific population benchmark: %{customdata[1]:,.0f}<br>'
                'Sex-specific rate: %{customdata[2]:.3f}%<br>'
                'Full age-group rate: %{customdata[3]:.3f}%'
                '<extra></extra>'
            ) if label != 'Unknown' else (
                '<b>%{x} · Unknown</b><br>'
                'Contribution to total rate: %{y:.3f}%<br>'
                'Recorded detections: %{customdata[0]:,.0f}'
                '<extra></extra>'
            ),
        ))

    # Display the complete observed age-group rate above each stacked bar.
    bars.add_trace(go.Scatter(
        x=age_names,
        y=[p['total_pct'] for p in age_profiles],
        mode='text',
        text=[f"<b>{p['total_pct']:.2f}% total</b>" for p in age_profiles],
        textposition='top center',
        textfont=dict(color='#243f37', size=13),
        showlegend=False,
        hoverinfo='skip',
    ))
    ymax = max(p['total_pct'] for p in age_profiles)
    bars.update_layout(
        barmode='stack',
        bargap=0.47,
        height=335,
        margin=dict(l=10, r=15, t=42, b=20),
        yaxis=dict(title='Observed detection benchmark (%)', range=[0, ymax * 1.25] if ymax > 0 else [0, 1]),
        xaxis=dict(title=None),
        legend=dict(orientation='h', y=1.19, x=0, title=None),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
    )
    st.plotly_chart(bars, use_container_width=True)
    st.caption(
        'The coloured segments add up to the full bar because each uses the '
        'same age-group population denominator. Sex-specific detection rates '
        'use separate denominators and are shown on hover, not added together. '
        'The supplied Eurostat export contains only male/female categories; '
        'Unknown is therefore unavailable, not known to be zero. '
        'Age-specific population denominators are approximate. '
        'These are aggregate detection rates, not individual legal-status probabilities.'
    )
else:
    st.info('No age-group data are available for this selection.')

# Covers the citizenship labels occurring in the provided Eurostat training.csv.
# Borderline/transcontinental locations are assigned to a single display region
# for visualisation only; this does not change model estimates.
REGION_MEMBERS = {
    'Europe': 'Albania|Andorra|Belarus|Bosnia and Herzegovina|Croatia|Holy See|Kosovo*|Moldova|Monaco|Montenegro|North Macedonia|Russia|San Marino|Ukraine|United Kingdom',
    'Asia': 'Afghanistan|Armenia|Azerbaijan|Bahrain|Bangladesh|Bhutan|Brunei Darussalam|Cambodia|China|Georgia|India|Indonesia|Iran|Iraq|Israel|Japan|Jordan|Kazakhstan|Kuwait|Kyrgyzstan|Laos|Lebanon|Malaysia|Maldives|Mongolia|Myanmar/Burma|Nepal|North Korea|Oman|Pakistan|Palestine*|Philippines|Qatar|Saudi Arabia|Singapore|Sri Lanka|Syria|Taiwan|Tajikistan|Thailand|Timor-Leste|Türkiye|Turkmenistan|United Arab Emirates|Uzbekistan|Viet Nam|Yemen',
    'Africa': 'Algeria|Angola|Benin|Botswana|Burkina Faso|Burundi|Cabo Verde|Cameroon|Central African Republic|Chad|Comoros|Congo|Côte d’Ivoire|Democratic Republic of the Congo|Djibouti|Egypt|Equatorial Guinea|Eritrea|Eswatini|Ethiopia|Gabon|Gambia, The|Ghana|Guinea|Guinea-Bissau|Kenya|Lesotho|Liberia|Libya|Madagascar|Malawi|Mali|Mauritania|Mauritius|Morocco|Mozambique|Namibia|Niger|Nigeria|Rwanda|São Tomé and Príncipe|Senegal|Seychelles|Sierra Leone|Somalia|South Africa|Sudan|Tanzania|Togo|Tunisia|Uganda|Western Sahara|Zambia|Zimbabwe',
    'Americas': 'Antigua and Barbuda|Argentina|Bahamas|Barbados|Belize|Bolivia|Brazil|Canada|Chile|Colombia|Costa Rica|Cuba|Dominica|Dominican Republic|Ecuador|El Salvador|Grenada|Guatemala|Guyana|Haiti|Honduras|Jamaica|Mexico|Nicaragua|Panama|Paraguay|Peru|Saint Kitts and Nevis|Saint Lucia|Saint Vincent and the Grenadines|Suriname|Trinidad and Tobago|United States|Uruguay|Venezuela',
    'Oceania': 'Australia|Federated States of Micronesia|Fiji|Kiribati|Marshall Islands|New Zealand|Palau|Papua New Guinea|Samoa|Solomon Islands|Tonga|Tuvalu|Vanuatu',
    'Unspecified': 'Recognised non-citizens|Stateless|Unknown',
}
REGION_OF = {
    name: region
    for region, joined_names in REGION_MEMBERS.items()
    for name in joined_names.split('|')
}
REGION_COLOR = {
    'Europe': '#6473b5',
    'Asia': '#9364aa',
    'Africa': '#317e75',
    'Americas': '#4a7cbe',
    'Oceania': '#bf7f55',
    'Unspecified': '#727983',
    'Other regions': '#727983',
}

def lighter(hex_color, strength=0.17):
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    rgb = [round(n + (255 - n) * strength) for n in rgb]
    return '#{:02x}{:02x}{:02x}'.format(*rgb)

# ISO-3166 country codes for citizenship origins, not reporting destinations.
# Stateless / Unknown categories have no geographic polygon.
CITIZENSHIP_ISO3 = {
    'Afghanistan': 'AFG',
    'Albania': 'ALB',
    'Algeria': 'DZA',
    'Andorra': 'AND',
    'Angola': 'AGO',
    'Antigua and Barbuda': 'ATG',
    'Argentina': 'ARG',
    'Armenia': 'ARM',
    'Australia': 'AUS',
    'Azerbaijan': 'AZE',
    'Bahamas': 'BHS',
    'Bahrain': 'BHR',
    'Bangladesh': 'BGD',
    'Barbados': 'BRB',
    'Belarus': 'BLR',
    'Belize': 'BLZ',
    'Benin': 'BEN',
    'Bhutan': 'BTN',
    'Bolivia': 'BOL',
    'Bosnia and Herzegovina': 'BIH',
    'Botswana': 'BWA',
    'Brazil': 'BRA',
    'Brunei Darussalam': 'BRN',
    'Burkina Faso': 'BFA',
    'Burundi': 'BDI',
    'Cabo Verde': 'CPV',
    'Cambodia': 'KHM',
    'Cameroon': 'CMR',
    'Canada': 'CAN',
    'Central African Republic': 'CAF',
    'Chad': 'TCD',
    'Chile': 'CHL',
    'China': 'CHN',
    'Colombia': 'COL',
    'Comoros': 'COM',
    'Congo': 'COG',
    'Costa Rica': 'CRI',
    'Croatia': 'HRV',
    'Cuba': 'CUB',
    'Côte d’Ivoire': 'CIV',
    'Democratic Republic of the Congo': 'COD',
    'Djibouti': 'DJI',
    'Dominica': 'DMA',
    'Dominican Republic': 'DOM',
    'Ecuador': 'ECU',
    'Egypt': 'EGY',
    'El Salvador': 'SLV',
    'Equatorial Guinea': 'GNQ',
    'Eritrea': 'ERI',
    'Eswatini': 'SWZ',
    'Ethiopia': 'ETH',
    'Federated States of Micronesia': 'FSM',
    'Fiji': 'FJI',
    'Gabon': 'GAB',
    'Gambia, The': 'GMB',
    'Georgia': 'GEO',
    'Ghana': 'GHA',
    'Grenada': 'GRD',
    'Guatemala': 'GTM',
    'Guinea': 'GIN',
    'Guinea-Bissau': 'GNB',
    'Guyana': 'GUY',
    'Haiti': 'HTI',
    'Holy See': 'VAT',
    'Honduras': 'HND',
    'India': 'IND',
    'Indonesia': 'IDN',
    'Iran': 'IRN',
    'Iraq': 'IRQ',
    'Israel': 'ISR',
    'Jamaica': 'JAM',
    'Japan': 'JPN',
    'Jordan': 'JOR',
    'Kazakhstan': 'KAZ',
    'Kenya': 'KEN',
    'Kiribati': 'KIR',
    'Kosovo*': 'XKX',
    'Kuwait': 'KWT',
    'Kyrgyzstan': 'KGZ',
    'Laos': 'LAO',
    'Lebanon': 'LBN',
    'Lesotho': 'LSO',
    'Liberia': 'LBR',
    'Libya': 'LBY',
    'Madagascar': 'MDG',
    'Malawi': 'MWI',
    'Malaysia': 'MYS',
    'Maldives': 'MDV',
    'Mali': 'MLI',
    'Marshall Islands': 'MHL',
    'Mauritania': 'MRT',
    'Mauritius': 'MUS',
    'Mexico': 'MEX',
    'Moldova': 'MDA',
    'Monaco': 'MCO',
    'Mongolia': 'MNG',
    'Montenegro': 'MNE',
    'Morocco': 'MAR',
    'Mozambique': 'MOZ',
    'Myanmar/Burma': 'MMR',
    'Namibia': 'NAM',
    'Nepal': 'NPL',
    'New Zealand': 'NZL',
    'Nicaragua': 'NIC',
    'Niger': 'NER',
    'Nigeria': 'NGA',
    'North Korea': 'PRK',
    'North Macedonia': 'MKD',
    'Oman': 'OMN',
    'Pakistan': 'PAK',
    'Palau': 'PLW',
    'Palestine*': 'PSE',
    'Panama': 'PAN',
    'Papua New Guinea': 'PNG',
    'Paraguay': 'PRY',
    'Peru': 'PER',
    'Philippines': 'PHL',
    'Qatar': 'QAT',
    'Russia': 'RUS',
    'Rwanda': 'RWA',
    'Saint Kitts and Nevis': 'KNA',
    'Saint Lucia': 'LCA',
    'Saint Vincent and the Grenadines': 'VCT',
    'Samoa': 'WSM',
    'San Marino': 'SMR',
    'Saudi Arabia': 'SAU',
    'Senegal': 'SEN',
    'Seychelles': 'SYC',
    'Sierra Leone': 'SLE',
    'Singapore': 'SGP',
    'Solomon Islands': 'SLB',
    'Somalia': 'SOM',
    'South Africa': 'ZAF',
    'Sri Lanka': 'LKA',
    'Sudan': 'SDN',
    'Suriname': 'SUR',
    'Syria': 'SYR',
    'São Tomé and Príncipe': 'STP',
    'Taiwan': 'TWN',
    'Tajikistan': 'TJK',
    'Tanzania': 'TZA',
    'Thailand': 'THA',
    'Timor-Leste': 'TLS',
    'Togo': 'TGO',
    'Tonga': 'TON',
    'Trinidad and Tobago': 'TTO',
    'Tunisia': 'TUN',
    'Turkmenistan': 'TKM',
    'Tuvalu': 'TUV',
    'Türkiye': 'TUR',
    'Uganda': 'UGA',
    'Ukraine': 'UKR',
    'United Arab Emirates': 'ARE',
    'United Kingdom': 'GBR',
    'United States': 'USA',
    'Uruguay': 'URY',
    'Uzbekistan': 'UZB',
    'Vanuatu': 'VUT',
    'Venezuela': 'VEN',
    'Viet Nam': 'VNM',
    'Western Sahara': 'ESH',
    'Yemen': 'YEM',
    'Zambia': 'ZMB',
    'Zimbabwe': 'ZWE',
}

LINKED_CHART_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {font-family:Arial, sans-serif;color:#273d32}
  body {margin:0;background:transparent}
  .toolbar {display:flex;justify-content:space-between;align-items:center;gap:12px;padding:4px 2px 12px;flex-wrap:wrap}
  .toolbar-title {font-weight:750;font-size:13px;color:#345a45}
  .toggle-group {display:inline-flex;align-items:center;gap:3px;background:#edf3ee;border:1px solid #dfe9e1;
       border-radius:12px;padding:4px;box-shadow:0 1px 2px #20352a0a}
  .toggle-btn {font:600 12px Arial,sans-serif;cursor:pointer;border:0;border-radius:8px;color:#577061;
      padding:10px 12px;background:transparent;transition:background .25s ease,color .25s ease, box-shadow .25s ease}
  .toggle-btn.active {background:#fff;color:#24543b;box-shadow:0 2px 8px #00000014}
  .toggle-btn:focus-visible {outline:2px solid #296646;outline-offset:2px}
  .charts {display:grid;grid-template-columns:minmax(0,1.26fr) minmax(0,1fr);gap:10px}
  .panel {min-width:0;background:#fff;border:1px solid #dfe7df;border-radius:12px;overflow:hidden}
  .heading {padding:12px 16px 0;font-size:13px;font-weight:700;color:#335a45}
  .sub {padding:3px 16px 0;color:#6b7b70;font-size:11px}
  #world, #sun {height:475px;width:100%}
  #status {box-sizing:border-box;margin-top:8px;min-height:44px;background:#f1f5f2;
      border:1px solid #dde7df;border-radius:10px;padding:12px 15px;
      color:#365846;font-size:12px;line-height:1.55}
  .legend {display:flex;flex-wrap:wrap;gap:12px;margin:9px 5px 0;font-size:11px;color:#4f6257}
  .legend span {display:flex;align-items:center;gap:5px}
  .legend i {width:11px;height:11px;border-radius:3px;display:inline-block}
  /* CSS transitions don't ask Plotly to rebuild the whole SVG on each hover. */
  #world path.choroplethlocation, #sun .slice path.surface {
      transition:fill 340ms cubic-bezier(.22,1,.36,1), opacity 340ms cubic-bezier(.22,1,.36,1),
                 stroke-width 340ms cubic-bezier(.22,1,.36,1),
                 stroke 340ms cubic-bezier(.22,1,.36,1),
                 filter 340ms cubic-bezier(.22,1,.36,1);
  }
  #sun .slice .slicetext {transition:opacity 340ms cubic-bezier(.22,1,.36,1);}
  @media(max-width:680px) {
      .charts {grid-template-columns:1fr} #world,#sun{height:375px}
      .toggle-group {width:100%;display:flex} .toggle-btn {flex:1;padding:10px 4px;font-size:11px}
  }
  @media(prefers-reduced-motion:reduce) {
      #world path.choroplethlocation, #sun .slice path.surface,
      #sun .slice .slicetext, .toggle-btn {transition:none!important}
  }
</style>
<script>/* __PLOTLY_JS__ */</script>
</head><body>
<div class="toolbar">
  <div class="toolbar-title">Explore citizenship statistics <span id="metric-indicator" style="font-weight:500;color:#64776b;margin-left:10px">Showing: detections / population (%)</span></div>
  <div class="toggle-group" role="group" aria-label="Map colour metric">
    <button class="toggle-btn active" type="button" data-metric="rate" aria-pressed="true">Detections / population</button>
    <button class="toggle-btn" type="button" data-metric="count" aria-pressed="false">Recorded detections</button>
  </div>
</div>
<div class="charts">
  <div class="panel"><div class="heading">World map by citizenship</div>
     <div class="sub" id="map-sub">Detection-to-population ratio · all mappable citizenships with sufficient data</div>
     <div id="world"></div></div>
  <div class="panel"><div class="heading">Citizenship distribution</div>
     <div class="sub" id="sun-sub">Labels and shades = detections / population (%) · area = recorded detections</div>
     <div id="sun"></div></div>
</div>
<div class="legend">
 <span><i style="background:#6473b5"></i>Europe</span>
 <span><i style="background:#9364aa"></i>Asia</span>
 <span><i style="background:#317e75"></i>Africa</span>
 <span><i style="background:#4a7cbe"></i>Americas</span>
 <span><i style="background:#bf7f55"></i>Oceania</span>
 <span><i style="background:#8c9599"></i>Other</span>
</div>
<div id="status" aria-live="polite">Hover over a country or sunburst slice to highlight it in both charts.</div>
<script>
'use strict';
const P = __PAYLOAD__;
const world = document.getElementById('world');
const sun = document.getElementById('sun');
const status = document.getElementById('status');
const shortCount = n => Number(n).toLocaleString('en-US');
const bigNames = new Set(P.sunburst_names);
const otherNames = new Set(P.other_names);
const nameByIso = Object.fromEntries(P.map.data.flatMap(t => t.locations.map((iso,i) => [iso, t.customdata[i][0]])));
let metricMode = 'rate';
let selectedFocus = null;
let mapReady = false;
let sunReady = false;
let mapPaths = [];
let sunSlices = [];
let metricBusy = false;
let pendingMetric = null;

// The sunburst keeps additive count-based slice sizes; its shade changes with the metric.
function lighten(hex, strength) {
  const nums = [1,3,5].map(i => parseInt(hex.slice(i,i+2),16));
  return '#' + nums.map(v => Math.round(v+(255-v)*strength).toString(16).padStart(2,'0')).join('');
}
function countryShade(name) {
  const s=P.stats[name];
  if (!s) return '#8c9599';
  const base=P.region_colors[P.region_of[name]] || '#727983';
  const r=metricMode==='rate'
    ? s.rate_pct/Math.max(P.max_rate,1e-9)
    : Math.log1p(s.detections)/Math.max(Math.log1p(P.max_count),1e-9);
  return lighten(base,0.77 - 0.70*Math.max(0,Math.min(1,r)));
}

function mapMatches(name, focus) {
  if (!focus) return true;
  if (focus.type === 'country') return name === focus.name;
  if (focus.type === 'region') return P.region_of[name] === focus.name;
  if (focus.type === 'other') return otherNames.has(name);
  return false;
}
function sunMatches(id, focus) {
  if (!focus || id === 'root') return true;
  if (focus.type === 'other') return id === 'other';
  if (focus.type === 'region') return id === 'r:' + focus.name ||
    (id.startsWith('c:') && P.region_of[id.slice(2)] === focus.name);
  if (focus.type === 'country') {
    const group = bigNames.has(focus.name) ? 'c:' + focus.name : 'other';
    return id === group || (bigNames.has(focus.name) && id === 'r:' + P.region_of[focus.name]);
  }
  return false;
}
function statusText(focus) {
  if (!focus) return metricMode === 'rate'
    ? 'Hover over a country or sunburst slice. Map shading represents detections / resident population.'
    : 'Hover over a country or sunburst slice. Map shading represents the number of recorded detections.';
  if (focus.type === 'region') return focus.name + ' · highlighted region';
  if (focus.type === 'other') {
    const n = P.other_stats?.detections ?? 0;
    return 'Other · ' + shortCount(n) + ' recorded detections across smaller or unmappable categories';
  }
  const s = P.stats[focus.name];
  if (!s) return focus.name;
  const note = bigNames.has(focus.name) ? '' : ' · included in “Other” on the sunburst';
  const primary = metricMode === 'rate'
    ? s.rate_pct.toFixed(2) + '% detections / population'
    : shortCount(s.detections) + ' recorded detections';
  const secondary = metricMode === 'rate'
    ? shortCount(s.detections) + ' recorded detections'
    : s.rate_pct.toFixed(2) + '% detections / population';
  return focus.name + ' · ' + primary + ' · ' + secondary + note;
}

// Read Plotly's attached D3 data so DOM ordering never causes country mismatches.
function indexMapPaths() {
  mapPaths = [];
  const traces = world.querySelectorAll('g.choroplethlayer g.trace');
  traces.forEach((trace, ti) => {
    const fallbackCodes = P.map.data[ti]?.locations || [];
    trace.querySelectorAll('path.choroplethlocation').forEach((path, pi) => {
      const d = path.__data__ || {};
      const iso = d.loc || d.location || d.id || d.properties?.id || fallbackCodes[pi];
      const name = nameByIso[iso];
      if (name) mapPaths.push({path, name});
    });
  });
  // Some Plotly versions omit the trace wrapper in the SVG hierarchy.
  if (!mapPaths.length) {
    world.querySelectorAll('path.choroplethlocation').forEach(path => {
      const d = path.__data__ || {};
      const name = nameByIso[d.loc || d.location || d.id];
      if (name) mapPaths.push({path, name});
    });
  }
}
function indexSunSlices() {
  sunSlices = [];
  sun.querySelectorAll('g.slice').forEach(g => {
    const id = g.__data__?.data?.id || g.__data__?.data?.data?.id;
    const path = g.querySelector('path.surface');
    if (id && path) sunSlices.push({id,path,text:g.querySelector('g.slicetext')});
  });
}

// Crossfade via CSS transitions, not Plotly.restyle on every hover.
function applyFocus(focus) {
  selectedFocus = focus;
  status.textContent = statusText(focus);
  for (const {path,name} of mapPaths) {
    const matched = mapMatches(name,focus);
    path.style.opacity = !focus || matched ? '1' : '.22';
    path.style.stroke = focus && matched ? '#1c332d' : '#ffffff';
    path.style.strokeWidth = focus && matched ? '2.7px' : '.55px';
    path.style.filter = focus && matched ? 'brightness(1.08) saturate(1.16)' : 'none';
    // Set the actual path fill for the selected metric. This crossfades smoothly
    // and does not depend on Plotly rebuilding the map trace.
    path.style.fill = countryShade(name);
  }
  for (const {id,path,text} of sunSlices) {
    const matched = sunMatches(id,focus);
    path.style.opacity = !focus || matched ? '1' : '.22';
    path.style.stroke = focus && matched && id !== 'root' ? '#1c332d' : '#ffffff';
    path.style.strokeWidth = focus && matched && id !== 'root' ? '2.7px' : '1px';
    path.style.filter = focus && matched && id !== 'root' ? 'brightness(1.07) saturate(1.15)' : 'none';
    if (id.startsWith('c:')) path.style.fill = countryShade(id.slice(2));
    if (text) text.style.opacity = !focus || matched ? '1' : '.29';
  }
}
function requestFocus(focus) {
  if (JSON.stringify(focus) !== JSON.stringify(selectedFocus)) applyFocus(focus);
}

const rateHover = '<b>%{customdata[0]}</b><br>' +
  'Detections / population: %{customdata[3]:.2f}%<br>' +
  'Recorded detections: %{customdata[1]:,.0f}<br>' +
  'Resident population benchmark: %{customdata[2]:,.0f}<extra></extra>';
const countHover = '<b>%{customdata[0]}</b><br>' +
  'Recorded detections: %{customdata[1]:,.0f}<br>' +
  'Detections / population: %{customdata[3]:.2f}%<br>' +
  'Resident population benchmark: %{customdata[2]:,.0f}<extra></extra>';

const rateSunHover = '<b>%{label}</b><br>' +
  'Detections / population: %{customdata[1]}<br>' +
  'Recorded detections: %{customdata[0]:,.0f}<br>' +
  'Share of all detections: %{percentRoot:.1%}<extra></extra>';
const countSunHover = '<b>%{label}</b><br>' +
  'Recorded detections: %{customdata[0]:,.0f}<br>' +
  'Detections / population: %{customdata[1]}<br>' +
  'Share of all detections: %{percentRoot:.1%}<extra></extra>';

async function updateMetric(mode) {
  if (mode !== 'rate' && mode !== 'count') return;
  metricMode = mode;
  document.querySelectorAll('[data-metric]').forEach(button => {
    const active = button.dataset.metric === metricMode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });
  document.getElementById('metric-indicator').textContent = metricMode === 'rate'
    ? 'Showing: detections / population (%)'
    : 'Showing: recorded detections (people detected)';
  document.getElementById('map-sub').textContent = metricMode === 'rate'
    ? 'Colour intensity = detection / population ratio (%) · all available countries'
    : 'Colour intensity = recorded detections (logarithmic shading) · all available countries';
  document.getElementById('sun-sub').textContent = metricMode === 'rate'
    ? 'Labels and shades = detection / population ratio (%) · area = recorded detections'
    : 'Labels and shades = recorded detections · area = recorded detections';

  // Apply colours immediately to the existing SVG; their CSS transition gives
  // 340ms crossfades even if Plotly redraws are slow or geographic assets fail.
  applyFocus(selectedFocus);

  if (metricBusy) {pendingMetric = mode; return;}
  metricBusy = true;
  try {
    // Unlike the previous implementation, the sunburst's VISIBLE country
    // values change with the toggle (not only the tooltip and colour).
    if (sunReady) {
      await Plotly.restyle(sun, {
        texttemplate: [metricMode === 'rate'
          ? '%{label}<br>%{customdata[1]}'
          : '%{label}<br>%{customdata[2]}'],
        hovertemplate: [metricMode === 'rate' ? rateSunHover : countSunHover],
      }, [0]);
      indexSunSlices();
      applyFocus(selectedFocus);
    }
    if (mapReady) {
      // Keep Plotly's underlying z array consistent with what the SVG shows.
      // Countries always use the same (global) scale across regions.
      const traces = P.map.data.map(t => t.customdata.map(item =>
        metricMode === 'rate' ? item[3] : Math.log1p(item[1])));
      const zmax = metricMode === 'rate' ? P.max_rate : Math.log1p(P.max_count);
      const indices = P.map.data.map((_,i) => i);
      await Plotly.restyle(world, {
        z: traces,
        zmin: indices.map(() => 0),
        zmax: indices.map(() => Math.max(zmax, 1e-9)),
        hovertemplate: indices.map(() => metricMode === 'rate' ? rateHover : countHover),
      }, indices);
      indexMapPaths();
      applyFocus(selectedFocus);
    }
  } catch(e) {console.warn('Unable to switch metric:', e);}
  metricBusy = false;
  if (pendingMetric && pendingMetric !== mode) {
    const next = pendingMetric;
    pendingMetric = null;
    updateMetric(next);
  } else pendingMetric = null;
}

for (const button of document.querySelectorAll('[data-metric]')) {
  button.addEventListener('click', () => updateMetric(button.dataset.metric));
}

const opts={responsive:true,displayModeBar:false,scrollZoom:false};
// Render independently, so a geographically unavailable map doesn't block the sunburst.
Plotly.newPlot(sun,P.sun.data,P.sun.layout,opts).then(() => {
  sunReady=true;
  indexSunSlices();
  sun.on('plotly_hover',ev => {
    const pt=ev.points?.[0];
    const id=pt?.id || pt?.data?.ids?.[pt?.pointNumber];
    if (!id || id==='root') return requestFocus(null);
    if (id==='other') return requestFocus({type:'other'});
    if (id.startsWith('r:')) return requestFocus({type:'region',name:id.slice(2)});
    if (id.startsWith('c:')) return requestFocus({type:'country',name:id.slice(2)});
  });
  sun.addEventListener('mouseleave', () => requestFocus(null));
  applyFocus(selectedFocus);
  // In case the user changed metrics while Plotly was loading.
  if (metricMode === 'count') updateMetric('count');
}).catch(e => {status.textContent='Could not render sunburst: '+e.message;});
Plotly.newPlot(world,P.map.data,P.map.layout,opts).then(() => {
  mapReady=true;
  indexMapPaths();
  world.on('plotly_hover',ev => {
    const name=ev.points?.[0]?.customdata?.[0];
    if(name) requestFocus({type:'country',name});
  });
  world.addEventListener('mouseleave', () => requestFocus(null));
  // A metric selection can happen before a slow map has finished rendering.
  if (metricMode === 'count') updateMetric('count');
  else applyFocus(selectedFocus);
}).catch(e => {console.warn('Could not render world map:',e);});
window.addEventListener('resize',()=>{
  if(mapReady) Plotly.Plots.resize(world);
  if(sunReady) Plotly.Plots.resize(sun);
  // Keep the index current when Plotly redraws on responsive resize.
  window.requestAnimationFrame(()=>{indexMapPaths();indexSunSlices();applyFocus(selectedFocus);});
});
</script></body></html>"""

# Linked map + sunburst: same data and palette, two-way hover highlighting.
# Map: every mappable citizenship with at least 1,000 benchmark residents.
# Pie: all sufficiently large slices; the remaining countries form "Other".
st.markdown('### Citizenship: linked world map and distribution')
st.caption(
    'Worldwide view of **countries of citizenship**, not EU reporting countries. '
    'Use the metric switch above the map to compare either aggregate detection-to-population '
    'ratios or numbers of recorded detections. Geographic colors represent regions. '
    'Hover over either chart to highlight the corresponding country smoothly.'
)

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
    info = summarize_cohort(cohort)
    if info and info['population_proxy'] > 0:
        citizenship_stats[name] = info

# Leave missing/suppressed cells out. Do not silently interpret them as zero.
MIN_POPULATION_BENCHMARK = 1000
SUNBURST_MIN_DETECTION_SHARE = 0.01  # At least 1% of recorded detections.

mappable_names = sorted(
    (name for name, info in citizenship_stats.items()
     if info['population_proxy'] >= MIN_POPULATION_BENCHMARK
     and name in CITIZENSHIP_ISO3),
    key=lambda name: citizenship_stats[name]['detections'],
    reverse=True,
)

if mappable_names:
    total_detections = sum(info['detections'] for info in citizenship_stats.values())
    sunburst_countries = [
        name for name in mappable_names
        if citizenship_stats[name]['detections'] > 0
        and citizenship_stats[name]['detections'] >= total_detections * SUNBURST_MIN_DETECTION_SHARE
    ]
    other_names = [name for name in citizenship_stats if name not in sunburst_countries]
    other_rows = [r for name in other_names for r in citizenship_groups[name]]
    other_stats = summarize_cohort(other_rows)

    # Each region is a Plotly choropleth trace so it keeps the original palette.
    max_rate = max(citizenship_stats[name]['rate_pct'] for name in mappable_names)
    max_rate = max(max_rate, 1e-9)
    fig_map = go.Figure()
    for region in ['Europe', 'Asia', 'Africa', 'Americas', 'Oceania',
                   'Unspecified', 'Other regions']:
        names_in_region = [
            name for name in mappable_names
            if REGION_OF.get(name, 'Other regions') == region
        ]
        if not names_in_region:
            continue
        base = REGION_COLOR[region]
        fig_map.add_trace(go.Choropleth(
            locations=[CITIZENSHIP_ISO3[name] for name in names_in_region],
            locationmode='ISO-3',
            z=[citizenship_stats[name]['rate_pct'] for name in names_in_region],
            zmin=0,
            zmax=max_rate,
            colorscale=[[0, lighter(base, .77)], [1, base]],
            showscale=False,
            showlegend=False,
            name=region,
            marker_line_color=['#ffffff'] * len(names_in_region),
            marker_line_width=[.55] * len(names_in_region),
            customdata=[[
                name,
                citizenship_stats[name]['detections'],
                citizenship_stats[name]['population_proxy'],
                citizenship_stats[name]['rate_pct'],
            ] for name in names_in_region],
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Detection / population benchmark: %{z:.2f}%<br>'
                'Recorded detections: %{customdata[1]:,.0f}<br>'
                'Population benchmark: %{customdata[2]:,.0f}'
                '<extra></extra>'
            ),
        ))
    fig_map.update_geos(
        scope='world',
        projection_type='natural earth',
        center=dict(lon=0, lat=0),
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor='#ffffff',
        countrywidth=.6,
        showland=True,
        landcolor='#e7ebe9',
        showocean=True,
        oceancolor='#f4f8fa',
        bgcolor='rgba(0,0,0,0)',
    )
    fig_map.update_layout(
        height=475,
        margin=dict(l=0, r=0, t=7, b=7),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=12),
        uirevision='full-world',
    )

    by_region = {}
    for name in sunburst_countries:
        region = REGION_OF.get(name, 'Other regions')
        by_region.setdefault(region, []).append(name)

    ids = ['root']
    labels = ['All citizenships']
    parents = ['']
    values = [total_detections]
    colors = ['#243f37']
    total_population = sum(info['population_proxy'] for info in citizenship_stats.values())
    root_rate = f'{100 * total_detections / total_population:.2f}%' if total_population > 0 else '—'
    info_labels = [[total_detections, root_rate, f'{total_detections:,}']]

    for region in sorted(
        by_region,
        key=lambda reg: sum(citizenship_stats[n]['detections'] for n in by_region[reg]),
        reverse=True,
    ):
        names_in_region = sorted(
            by_region[region],
            key=lambda n: citizenship_stats[n]['detections'],
            reverse=True,
        )
        region_total = sum(citizenship_stats[n]['detections'] for n in names_in_region)
        reg_id = 'r:' + region
        ids.append(reg_id)
        labels.append(region)
        parents.append('root')
        values.append(region_total)
        colors.append(REGION_COLOR[region])
        region_population = sum(citizenship_stats[n]['population_proxy'] for n in names_in_region)
        region_rate = f'{100 * region_total / region_population:.2f}%' if region_population > 0 else '—'
        info_labels.append([region_total, region_rate, f'{region_total:,}'])
        for name in names_in_region:
            stat = citizenship_stats[name]
            ids.append('c:' + name)
            labels.append(name)
            parents.append(reg_id)
            values.append(stat['detections'])
            # Initial country shading reflects the default detection/population metric.
            relative = stat['rate_pct'] / max_rate
            colors.append(lighter(REGION_COLOR[region], .77 - .70 * max(0, min(1, relative))))
            info_labels.append([stat['detections'], f"{stat['rate_pct']:.2f}%", f"{stat['detections']:,}"])

    if other_stats and other_stats['detections'] > 0:
        ids.append('other')
        labels.append('Other')
        parents.append('root')
        values.append(other_stats['detections'])
        colors.append('#8c9599')
        info_labels.append([
            other_stats['detections'],
            f"{other_stats['rate_pct']:.2f}%" if other_stats['rate_pct'] is not None else '—',
            f"{other_stats['detections']:,}"
        ])

    fig_sun = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        branchvalues='total',
        sort=False,
        maxdepth=3,
        marker=dict(colors=colors, line=dict(color=['white'] * len(ids), width=[1] * len(ids))),
        customdata=info_labels,
        insidetextfont=dict(color='#ffffff'),
        texttemplate='%{label}<br>%{customdata[1]}',
        insidetextorientation='radial',
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Detection / population benchmark: %{customdata[1]}<br>'
            'Recorded detections: %{customdata[0]:,.0f}<br>'
            'Share of detections: %{percentRoot:.1%}'
            '<extra></extra>'
        ),
    ))
    fig_sun.update_layout(
        height=475,
        margin=dict(l=0, r=0, t=8, b=8),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=11),
        uniformtext=dict(minsize=9, mode='hide'),
    )

    # Render BOTH charts in one Streamlit HTML component: separate st.plotly_chart()
    # elements cannot share true hover events with one another.
    # get_plotlyjs() embeds JS locally (no CDN dependency when published).
    payload = {
        'map': json.loads(fig_map.to_json()),
        'max_rate': max_rate,
        'max_count': max(citizenship_stats[n]['detections'] for n in mappable_names),
        'region_colors': REGION_COLOR,
        'sun': json.loads(fig_sun.to_json()),
        'sunburst_names': sunburst_countries,
        'other_names': other_names,
        'region_of': {name: REGION_OF.get(name, 'Other regions') for name in mappable_names},
        'stats': {name: {
            'detections': info['detections'],
            'rate_pct': round(info['rate_pct'], 5),
        } for name, info in citizenship_stats.items()},
        'other_stats': other_stats,
    }
    # Protect the embedded script from special characters in country labels.
    safe_json = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    html_page = LINKED_CHART_HTML.replace('__PAYLOAD__', safe_json)
    html_page = html_page.replace('/* __PLOTLY_JS__ */', get_plotlyjs())
    components.html(html_page, height=670, scrolling=True)
    st.caption(
        f'Map: {len(mappable_names)} countries with at least '
        f'{MIN_POPULATION_BENCHMARK:,} benchmark residents, always displayed. '
        f'Sunburst: {len(sunburst_countries)} countries each contributing at least '
        f'{SUNBURST_MIN_DETECTION_SHARE:.0%} of recorded detections; all smaller '
        'categories (including unmappable/unspecified citizenship) appear under Other. '
        'Hovering a small mapped country highlights Other in the sunburst. '
        'The toggle changes both map and sunburst colours, country labels, and metric priority. '
        'Sunburst slice AREA always represents recorded detections: ratios cannot be summed. '
        'These are aggregate detection/population ratios, not individual probabilities.'
    )
else:
    st.info('No mappable citizenship data meet the population coverage threshold for this selection.')

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
