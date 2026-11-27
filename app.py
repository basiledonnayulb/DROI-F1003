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
  @media(max-width:680px) {.charts {grid-template-columns:1fr} #world,#sun{height:375px}}
</style>
<script>/* __PLOTLY_JS__ */</script>
</head><body>
<div class="charts">
  <div class="panel"><div class="heading">World map by citizenship</div>
     <div class="sub">All mappable citizenships with sufficient data</div>
     <div id="world"></div></div>
  <div class="panel"><div class="heading">Citizenship distribution</div>
     <div class="sub">Continent → citizenship · smaller slices grouped as Other</div>
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
<div id="status" aria-live="polite">Hover over a country on the map or a slice on the sunburst to highlight the corresponding area in both charts.</div>
<script>
'use strict';
const P = __PAYLOAD__;
const world = document.getElementById('world');
const sun = document.getElementById('sun');
const status = document.getElementById('status');
const sunTrace = P.sun.data[0];
const allIds = sunTrace.ids;
const baseSunColors = Array.from(sunTrace.marker.colors);
const baseMapWidths = P.map.data.map(t => t.locations.map(() => .55));
const baseMapColors = P.map.data.map(t => t.locations.map(() => '#ffffff'));
const bigNames = new Set(P.sunburst_names);
const otherNames = new Set(P.other_names);
const mapNames = new Set(Object.keys(P.region_of));
const shortCount = n => Number(n).toLocaleString('en-US');
let selectedFocus = null;
let changing = false;
let queued; // undefined means no queued focus; null means reset
let sunReady = false;
let mapReady = false;

// A single focus may identify one country, an entire region or "Other".
function mapMatches(name, focus) {
  if (!focus) return false;
  if (focus.type === 'country') return name === focus.name;
  if (focus.type === 'region') return P.region_of[name] === focus.name;
  if (focus.type === 'other') return otherNames.has(name);
  return false;
}
function sunMatches(id, focus) {
  if (!focus) return true;
  if (id === 'root') return true;
  if (focus.type === 'other') return id === 'other';
  if (focus.type === 'region') {
    if (id === 'r:' + focus.name) return true;
    if (id.startsWith('c:')) return P.region_of[id.slice(2)] === focus.name;
    return false;
  }
  if (focus.type === 'country') {
    const region = P.region_of[focus.name];
    const idToHighlight = bigNames.has(focus.name) ? 'c:' + focus.name : 'other';
    return id === idToHighlight || id === 'r:' + region;
  }
  return false;
}
function statusText(focus) {
  if (!focus) return 'Hover over a country on the map or a slice on the sunburst to highlight the corresponding area in both charts.';
  if (focus.type === 'region') return focus.name + ' · highlighted region';
  if (focus.type === 'other') {
     const n = P.other_stats?.detections ?? 0;
     return 'Other · ' + shortCount(n) + ' recorded detections from all smaller or unmappable categories';
  }
  const s = P.stats[focus.name];
  if (!s) return focus.name;
  const groupNote = bigNames.has(focus.name) ? '' : ' · part of “Other” in the sunburst';
  return focus.name + ' · ' + s.rate_pct.toFixed(2) + '% detected cases / population benchmark · '
       + shortCount(s.detections) + ' recorded detections' + groupNote;
}
async function renderFocus(focus) {
  changing = true;
  selectedFocus = focus;
  status.textContent = statusText(focus);
  const tasks = [];
  for (let ti = 0; mapReady && ti < P.map.data.length; ti++) {
    const names = P.map.data[ti].customdata.map(x => x[0]);
    const widths = names.map(name => focus && mapMatches(name, focus) ? 3 : .55);
    const strokes = names.map(name => focus && mapMatches(name, focus) ? '#1c332d' : '#ffffff');
    tasks.push(Plotly.restyle(world, {
      'marker.line.width': [widths],
      'marker.line.color': [strokes]
    }, [ti]));
  }
  const cols = allIds.map((id, i) => {
    if (!focus || sunMatches(id, focus)) return baseSunColors[i];
    // Fade non-matching areas without changing the continent color scheme.
    return '#e7ece9';
  });
  const widths = allIds.map(id => focus && sunMatches(id, focus) && id !== 'root' ? 3 : 1);
  const strokes = allIds.map(id => focus && sunMatches(id, focus) && id !== 'root' ? '#1c332d' : '#ffffff');
  if (sunReady) tasks.push(Plotly.restyle(sun, {
    'marker.colors': [cols],
    'marker.line.width': [widths],
    'marker.line.color': [strokes]
  }, [0]));
  try { await Promise.all(tasks); } catch (e) { console.warn('Linked chart highlight error:', e); }
  changing = false;
  if (queued !== undefined) {
    const next = queued;
    queued = undefined;
    if (JSON.stringify(next) !== JSON.stringify(selectedFocus)) renderFocus(next);
  }
}
function requestFocus(focus) {
  if (!sunReady) return;
  if (changing) { queued = focus; return; }
  if (JSON.stringify(focus) === JSON.stringify(selectedFocus)) return;
  renderFocus(focus);
}
const opts = {responsive:true, displayModeBar:false, scrollZoom:false};
// Bind each chart independently: a slow/unavailable map tile must not disable the sunburst.
Plotly.newPlot(sun, P.sun.data, P.sun.layout, opts).then(() => {
  sunReady = true;
  sun.on('plotly_hover', ev => {
    const pt = ev.points?.[0];
    if (!pt) return;
    const nodeId = pt.id || pt.data?.ids?.[pt.pointNumber];
    if (!nodeId || nodeId === 'root') return requestFocus(null);
    if (nodeId === 'other') return requestFocus({type:'other'});
    if (nodeId.startsWith('r:')) return requestFocus({type:'region',name:nodeId.slice(2)});
    if (nodeId.startsWith('c:')) return requestFocus({type:'country',name:nodeId.slice(2)});
  });
  sun.addEventListener('mouseleave', () => requestFocus(null));
}).catch(error => {status.textContent = 'Could not render sunburst: ' + error.message;});
Plotly.newPlot(world, P.map.data, P.map.layout, opts).then(() => {
  mapReady = true;
  world.on('plotly_hover', ev => {
    const pt = ev.points?.[0];
    const name = pt?.customdata?.[0];
    if (name) requestFocus({type:'country', name:name});
  });
  world.addEventListener('mouseleave', () => requestFocus(null));
  if (selectedFocus) renderFocus(selectedFocus);
}).catch(error => {console.warn('Could not render world map:', error);});
window.addEventListener('resize', () => {
  if (mapReady) Plotly.Plots.resize(world);
  if (sunReady) Plotly.Plots.resize(sun);
});
</script></body></html>"""

# Linked map + sunburst: same data and palette, two-way hover highlighting.
# Map: every mappable citizenship with at least 1,000 benchmark residents.
# Pie: all sufficiently large slices; the remaining countries form "Other".
st.markdown('### Citizenship: world map and detection distribution')
st.caption(
    'Worldwide view of **countries of citizenship**, not EU reporting countries. '
    'Color families represent regions; intensity represents observed detections '
    'relative to the resident population benchmark. Hover over either view to '
    'highlight the corresponding country in the other view.'
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
    labels = ['All detections']
    parents = ['']
    values = [total_detections]
    colors = ['#243f37']
    info_labels = [[total_detections, '—']]

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
        info_labels.append([region_total, '—'])
        for name in names_in_region:
            stat = citizenship_stats[name]
            ids.append('c:' + name)
            labels.append(name)
            parents.append(reg_id)
            values.append(stat['detections'])
            colors.append(lighter(REGION_COLOR[region], .16))
            info_labels.append([stat['detections'], f"{stat['rate_pct']:.2f}%"])

    if other_stats and other_stats['detections'] > 0:
        ids.append('other')
        labels.append('Other')
        parents.append('root')
        values.append(other_stats['detections'])
        colors.append('#8c9599')
        info_labels.append([
            other_stats['detections'],
            f"{other_stats['rate_pct']:.2f}%" if other_stats['rate_pct'] is not None else '—'
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
        textinfo='label+percent root',
        insidetextorientation='radial',
        hovertemplate=(
            '<b>%{label}</b><br>'
            'Recorded detections: %{customdata[0]:,.0f}<br>'
            'Detection / population benchmark: %{customdata[1]}<br>'
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
