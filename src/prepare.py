"""Stream Eurostat custom wide-format XLSX files without loading entire workbooks.

Input population file is *population on 1 January*, NOT a verified legal-stay
registry. Illegal-presence figures are annual detection counts, NOT the stock
of all people without immigration permission.
"""
from __future__ import annotations
import csv
import os
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

NS = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
COUNTRIES = {'Austria','Belgium','Bulgaria','Croatia','Cyprus','Czechia','Denmark',
             'Estonia','Finland','France','Germany','Greece','Hungary','Ireland',
             'Italy','Latvia','Lithuania','Luxembourg','Malta','Netherlands',
             'Poland','Portugal','Romania','Slovakia','Slovenia','Spain','Sweden'}
NON_EU = 'Non-EU27 countries (from 2020) nor reporting country'
ILLEGAL_AGES = {'Total': 'All ages', 'Less than 18 years': 'Under 18',
                'From 18 to 34 years': '18–34', '35 years or over': '35+'}
LEGAL_AGES = {'Less than 15 years':'under15',
              'From 15 to 64 years':'15to64', '65 years or over':'65plus'}
YEARS = range(2018, 2026)


def _text(cell):
    inline = cell.find(NS+'is')
    if inline is not None:
        return ''.join(x.text or '' for x in inline.iter(NS+'t'))
    v = cell.find(NS+'v')
    return v.text if v is not None and v.text is not None else ''


def parse_wide_data(path: Path, kind: str):
    """Yield (geo,year,citizenship,sex,age,count) from the Data worksheet.
    ':' and confidential/suppressed cells are discarded, not treated as zero.
    """
    with ZipFile(path) as workbook, workbook.open('xl/worksheets/sheet2.xml') as f:
        meta = {}; years = {}; in_geos = False
        for _event, row in iterparse(f, events=('end',)):
            if row.tag != NS+'row':
                continue
            cells = {}
            for cell in row.findall(NS+'c'):
                col = re.match(r'[A-Z]+', cell.attrib['r']).group()
                if col in {'A','C','B','D','F','H','J','L','N','P'}:
                    value = _text(cell)
                    if value: cells[col] = value
            tag = cells.get('A','')
            if tag.startswith('Data extracted on'):
                meta.clear(); years.clear(); in_geos = False
            elif tag == 'GEO (Labels)':
                in_geos = True
            elif tag == 'TIME':
                years = {k:int(v) for k,v in cells.items() if k != 'A' and v.isdigit() and int(v) in YEARS}
            elif tag in {'Special value','Observation flags:'}:
                in_geos = False
            elif tag in {'Country of citizenship','Sex','Age class','Reason','Place of apprehension'} and not in_geos:
                meta[tag] = cells.get('C','')
            elif in_geos and tag in COUNTRIES:
                age = meta.get('Age class','')
                if kind == 'illegal' and age not in ILLEGAL_AGES:
                    row.clear(); continue
                if kind == 'legal' and age not in LEGAL_AGES:
                    row.clear(); continue
                if kind == 'illegal' and (meta.get('Reason')!='Total' or meta.get('Place of apprehension')!='Total'):
                    row.clear(); continue
                for col, year in years.items():
                    raw = cells.get(col,'')
                    if not raw or raw == ':': continue
                    try: amount = float(raw)
                    except (ValueError,TypeError): continue
                    if amount < 0: continue
                    yield (tag,year,meta.get('Country of citizenship',''),meta.get('Sex',''),age,amount)
            row.clear()


def prepare(illegal_file: Path, population_file: Path, database_file: Path, training_file: Path):
    database_file.parent.mkdir(parents=True,exist_ok=True)
    if database_file.exists(): database_file.unlink()
    db = sqlite3.connect(database_file)
    db.executescript('''
        CREATE TABLE observations (kind TEXT, country TEXT, year INTEGER,
          citizenship TEXT, sex TEXT, age TEXT, value REAL);
        CREATE INDEX obs_ix ON observations(kind,country,year,citizenship,sex,age);
    ''')
    for kind,filename in [('illegal',illegal_file),('legal',population_file)]:
        buffer=[]; count=0
        for geo,year,cit,sex,age,value in parse_wide_data(filename,kind):
            # For complete nationality matching, preserve individual citizenships;
            # 'Total' numerator is paired with non-EU population aggregate only.
            if kind=='legal' and cit!=NON_EU and cit not in CITIZENSHIPS:
                continue
            if kind=='illegal' and cit not in CITIZENSHIPS and cit!='Total':
                continue
            buffer.append((kind,geo,year,cit,sex,age,value));count+=1
            if len(buffer)>=8000:
                db.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?,?)',buffer)
                buffer.clear()
        if buffer:db.executemany('INSERT INTO observations VALUES(?,?,?,?,?,?,?)',buffer)
        db.commit();print(f'{kind}: {count:,} numeric cells imported',flush=True)

    # SQL joins on exact country, sex, citizenship and year. An unmatched
    # population group is never silently imputed as zero.
    people=defaultdict(dict)
    for geo,yr,cit,sex,age,v in db.execute('SELECT country,year,citizenship,sex,age,value FROM observations WHERE kind="legal"'):
        people[(geo,yr,cit,sex)][age]=v
    # Split 15–64 population *uniformly by single-year age* for an explicit
    # approximation. No true age-resolved denominator is available in inputs.
    # Never average raw counts or assume all ages have identical legal status.
    exposures={}
    for (geo,yr,cit,sex),bins in people.items():
        if not all(a in bins for a in LEGAL_AGES):continue
        u= bins['Less than 15 years'];middle=bins['From 15 to 64 years'];old=bins['65 years or over']
        if u<0 or middle<0 or old<0: continue
        exposures[(geo,yr,cit,sex)]= {
            'All ages':u+middle+old,
            'Under 18':u+3/50*middle,
            '18–34':17/50*middle,
            '35+':30/50*middle+old,
        }
    del people
    record_count=0; nonzero=0; impossible=0
    cols=['country','year','citizenship','sex','age','detections','population_proxy','rate_per_100k','age_approximate']
    with training_file.open('w',newline='',encoding='utf-8') as f:
        out=csv.DictWriter(f,fieldnames=cols);out.writeheader()
        for geo,yr,cit,sex,age,v in db.execute('SELECT country,year,citizenship,sex,age,value FROM observations WHERE kind="illegal"'):
            legal_cit=NON_EU if cit=='Total' else cit
            age_label=ILLEGAL_AGES[age]
            denominator=exposures.get((geo,yr,legal_cit,sex),{}).get(age_label)
            if not denominator or denominator<=0:continue
            rate=v/denominator*100_000
            if rate>100_000: impossible+=1;continue # numerator outnumbers stock, unusable match
            out.writerow(dict(country=geo,year=yr,citizenship=cit,sex=sex,
                age=age_label,detections=round(v),population_proxy=round(denominator,2),
                rate_per_100k=round(rate,8),age_approximate=int(age_label!='All ages')))
            record_count+=1
            if v>0:nonzero+=1
    db.close()
    print(f'Training observations: {record_count:,}; nonzero: {nonzero:,}; rejected count>stock: {impossible:,}',flush=True)
    return record_count


# Exact nationality whitelist from the supplied irregular-presence summary:
# EU/other broad aggregation categories are excluded to prevent double counting.
CITIZENSHIPS={
'Afghanistan','Albania','Algeria','Andorra','Angola','Antigua and Barbuda','Argentina',
'Armenia','Australia','Azerbaijan','Bahamas','Bahrain','Bangladesh','Barbados',
'Belarus','Belize','Benin','Bhutan','Bolivia','Bosnia and Herzegovina','Botswana',
'Brazil','British overseas countries and territories','Brunei Darussalam',
'Burkina Faso','Burundi','Cabo Verde','Cambodia','Cameroon','Canada',
'Central African Republic','Chad','Chile','China','Colombia','Comoros','Congo',
'Costa Rica','Croatia','Cuba','Côte d’Ivoire','Democratic Republic of the Congo',
'Djibouti','Dominica','Dominican Republic','Ecuador','Egypt','El Salvador',
'Equatorial Guinea','Eritrea','Eswatini','Ethiopia','Federated States of Micronesia',
'Fiji','Gabon','Gambia, The','Georgia','Ghana','Grenada','Guatemala','Guinea',
'Guinea-Bissau','Guyana','Haiti','Holy See','Honduras','Iceland','India','Indonesia',
'Iran','Iraq','Israel','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati',
'Kosovo*','Kuwait','Kyrgyzstan','Laos','Lebanon','Lesotho','Liberia','Libya',
'Liechtenstein','Madagascar','Malawi','Malaysia','Maldives','Mali','Marshall Islands',
'Mauritania','Mauritius','Mexico','Moldova','Monaco','Mongolia','Montenegro',
'Morocco','Mozambique','Myanmar/Burma','Namibia','Nepal','New Zealand',
'Nicaragua','Niger','Nigeria','North Korea','North Macedonia','Norway','Oman',
'Pakistan','Palau','Palestine*','Panama','Papua New Guinea','Paraguay','Peru',
'Philippines','Qatar','Recognised non-citizens','Russia','Rwanda',
'Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines',
'Samoa','San Marino','Saudi Arabia','Senegal','Seychelles','Sierra Leone',
'Singapore','Solomon Islands','Somalia','South Africa','Sri Lanka','Stateless',
'Sudan','Suriname','Switzerland','Syria','São Tomé and Príncipe','Taiwan',
'Tajikistan','Tanzania','Thailand','Timor-Leste','Togo','Tonga',
'Trinidad and Tobago','Tunisia','Türkiye','Turkmenistan','Tuvalu','Uganda',
'Ukraine','United Arab Emirates','United Kingdom','United States','Unknown',
'Uruguay','Uzbekistan','Vanuatu','Venezuela','Viet Nam','Western Sahara',
'Yemen','Zambia','Zimbabwe'}

if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    illegal=Path(sys.argv[1]) if len(sys.argv)>1 else root/'data'/'irregular.xlsx'
    legal=Path(sys.argv[2]) if len(sys.argv)>2 else root/'data'/'population.xlsx'
    prepare(illegal,legal,root/'data'/'source.sqlite',root/'data'/'training.csv')
