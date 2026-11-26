from src.model import choose_rows,summarize_cohort,predicted_pct,read_training


def test_aggregate_rate():
    sample=[dict(country='Belgium',year=2025,citizenship='Total',sex='Males',age='All ages',detections=50,population_proxy=5000,age_approximate=False),
            dict(country='France',year=2025,citizenship='Total',sex='Females',age='All ages',detections=5,population_proxy=5000,age_approximate=False)]
    assert abs(summarize_cohort(sample)['rate_pct']-.55)<1e-10
    assert len(choose_rows(sample,age='All ages',sex='All',country='All',citizenship='Total',year=2025))==2
    assert len(choose_rows(sample,age='All ages',sex='Males',country='All',citizenship='Total',year=2025))==1


def test_age_not_double_counted():
    rows=read_training()
    total=choose_rows(rows,age='All ages',country='Belgium',citizenship='Total',year=2024)
    assert total and all(r['age']=='All ages' for r in total)
    assert len(total)<=2
    for r in total: assert r['population_proxy']>0
