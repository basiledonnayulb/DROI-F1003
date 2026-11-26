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


= Conclusion

text

#bibliography("refs.bib")
