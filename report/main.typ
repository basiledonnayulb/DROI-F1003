#import "template.typ": project

#show: project.with(
  title: "titre",
  authors: (
    (
      name: "Basile DONNAY
Capucine SPEILERS",

      status: "Authors"
    ),
    (
      name: "Nathan Genicot", 
      status: "Professeur"  
    )
  ),
  logo: "ulb_logo.jpg",
  abstract: "text
"
)

= Introduction

intro

= La suite

== la suite sous titre

text

#figure(
  image("images/EDF_vs_FP.png"),
  caption: [This figure is an example of a figure],
)

exemple d'une formule : `O_max + 2 × Hyperperiod`.

exemple d'une citation : @freertos

TODO : choisir les sources et les ajouter ceci dans refs.bib
- https://ec.europa.eu/eurostat/databrowser/view/migr_eipre__custom_24484187/default/table

- https://ec.europa.eu/eurostat/databrowser/view/migr_pop1ctz__custom_24484552/default/table

- 


= Conclusion

text

#bibliography("refs.bib")
