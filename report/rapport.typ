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

= Présentation du prototype

== Objectif de l'application

Dans le cadre de ce projet, nous avons développé un prototype d'intelligence artificielle dont l'objectif initial était d'estimer la probabilité qu'une personne se retrouve en situation de séjour irrégulier dans l'Union européenne, à partir de certaines caractéristiques démographiques.

L'idée initiale était de concevoir un outil inspiré des modèles prédictifs utilisés dans d'autres domaines, notamment les algorithmes de churn prediction, qui cherchent à estimer la probabilité qu'un client quitte un service. Dans notre cas, nous avons cherché à appliquer une approche similaire aux données migratoires afin d'étudier les relations statistiques entre certaines caractéristiques démographiques et les situations de séjour irrégulier recensées dans l'Union européenne.

Le prototype permet ainsi de sélectionner différentes caractéristiques, telles que la tranche d'âge, le sexe, la nationalité ou le pays européen concerné, afin d'obtenir une estimation statistique exprimée en pourcentage. Toutefois, en raison de la nature des données disponibles, le modèle estime actuellement un taux de détection rapporté à une population de référence, et non une véritable probabilité individuelle de séjour irrégulier.

Au-delà de sa dimension technique, le prototype sert également de support à une réflexion sur les possibilités et les limites de l'intelligence artificielle lorsqu'elle est appliquée à des questions sociales sensibles.

== IA ou pas ?

Selon l'article 3, paragraphe 1, du règlement européen sur l'intelligence artificielle (AI Act), un système d'IA est un système automatisé capable de fonctionner avec différents niveaux d'autonomie, qui déduit, à partir des données reçues, comment générer des résultats tels que des prédictions, des recommandations ou des décisions. Il peut également présenter une capacité d'adaptation après son déploiement. @ai_act_2024

Notre prototype peut être considéré comme un système d'intelligence artificielle au sens de cette définition pour plusieurs raisons :

- *L'apprentissage à partir des données :* notre système utilise un algorithme de machine learning, plus précisément un modèle de régression de Poisson fondé sur le gradient boosting. Contrairement à un programme reposant uniquement sur des règles prédéfinies, le modèle apprend des relations statistiques à partir des données historiques d'Eurostat.

- *La capacité de prédiction :* à partir des variables sélectionnées par l'utilisateur, le modèle génère une estimation statistique. Il ne se contente donc pas de rechercher une valeur existante dans une base de données, mais produit un résultat à partir des relations apprises pendant son entraînement.

- *Un fonctionnement autonome :* une fois le modèle entraîné, celui-ci peut générer des estimations sans intervention humaine dans le calcul. L'utilisateur fournit les caractéristiques souhaitées et le système produit automatiquement un résultat.

- *La génération de résultats influençant un environnement virtuel :* les estimations produites sont intégrées à une interface web interactive, qui présente les résultats sous forme de pourcentages et de visualisations.

D'après nous, notre prototype répond donc aux principales caractéristiques de la définition d'un système d'IA présentée dans l'AI Act.

== Variables disponibles

#grid(
  columns: (1.5fr, 1fr),
  gutter: 15pt,
  align: top,
  [
    L'application repose sur plusieurs variables issues des bases de données européennes. L'utilisateur peut les combiner librement afin d'explorer différentes catégories de population.

    Les principales variables sont les suivantes :

    - *Tranche d'âge :* tous âges confondus, moins de 18 ans, 18 à 34 ans et 35 ans ou plus.
    - *Sexe :* hommes, femmes ou ensemble des catégories disponibles.
    - *Nationalité :* sélection d'une nationalité particulière ou de l'ensemble des nationalités de pays tiers présentes dans les données.
    - *Pays déclarant :* sélection d'un État membre de l'Union européenne ou agrégation de tous les pays disponibles.
    - *Année :* sélection d'une année entre 2018 et 2025.
  ],
  [
    #figure(
      image("images/Variables_disponibles.png", width: 100%),
      caption: [Les variables disponibles.],
    )
  ],
)

    Par exemple, il est possible d'étudier les statistiques pour les femmes âgées de 18 à 34 ans en Belgique, ou simplement d'observer les résultats d'une tranche d'âge indépendamment du sexe et du pays déclarant.

    Cette flexibilité permet d'explorer les données à différents niveaux de précision, tout en tenant compte des limites de couverture statistique.

== Interface et visualisation des résultats

Le prototype a été développé en Python, à l'aide du framework Streamlit, qui permet de construire des interfaces web interactives. Les visualisations reposent principalement sur la bibliothèque Plotly.

L'interface a été conçue pour rendre les résultats accessibles sans nécessiter de connaissances en programmation ou en analyse statistique.

Un panneau latéral permet de sélectionner les différentes variables. Les résultats sont ensuite actualisés en fonction de la combinaison choisie.

La partie principale de l'application présente d'abord une estimation produite par le modèle, exprimée en pourcentage. Trois indicateurs complémentaires permettent de contextualiser cette valeur : le taux observé dans les données, le nombre de détections enregistrées et la population de référence utilisée pour le calcul.

#figure(
  image("images/Estimation_modèle_et_indicateurs.png", width: 100%),
  caption: [L'estimation IA et les trois indicateurs complémentaires.],
)

Plusieurs visualisations permettent ensuite d'explorer les résultats.

=== Évolution au fil des années

Un premier graphique présente l'évolution du taux de détection entre 2018 et 2025. Il compare les valeurs calculées à partir des données observées aux estimations produites par le modèle.

Cette représentation permet d'identifier les variations temporelles et d'observer dans quelle mesure le modèle reproduit les tendances historiques.

#figure(
  image("images/Ligne_du_temps_estimation.png", width: 100%),
  caption: [La ligne du temps des estimations.],
)

=== Comparaison des tranches d'âge et du sexe

Un deuxième graphique compare les différentes tranches d'âge sous la forme de barres verticales empilées. Chaque barre représente le taux de détection total de la tranche d'âge concernée. Elle est divisée selon les catégories de sexe.

La hauteur totale de la barre correspond au taux observé pour l'ensemble du groupe d'âge. Les segments représentent la contribution de chaque catégorie au total, tandis que les taux propres aux hommes et aux femmes sont accessibles au survol.

#figure(
  image("images/Comparaison_des_tranches_age.png", width: 100%),
  caption: [Graphique représentant la comparaison des tranches d'âge et de sexe.],
)

=== Carte mondiale et diagramme circulaire interactif

Une troisième partie propose une visualisation géographique des statistiques par nationalité.

Une carte du monde interactive représente les pays de nationalité des personnes recensées dans les données. Elle est accompagnée d'un diagramme circulaire hiérarchique, appelé sunburst, qui regroupe les nationalités par grandes régions géographiques.

Un sélecteur permet de basculer entre deux indicateurs : le taux de détections rapporté à la population et le nombre absolu de détections enregistrées.

Afin de préserver la lisibilité, les nationalités dont la contribution au total des détections est inférieure à 1 % sont regroupées dans une catégorie « Autres » au sein du diagramme circulaire. Elles restent néanmoins visibles individuellement sur la carte lorsqu'elles disposent d'une population de référence d'au moins 1 000 personnes.

La carte illustre les pays de nationalité, et non les lieux dans lesquels les personnes ont été détectées en situation irrégulière. Cette distinction évite de confondre l'origine des populations étudiées avec la localisation des contrôles migratoires.

#figure(
  image("images/Nationalité_carte_du_monde.png", width: 100%),
  caption: [Carte du monde accompagné d'un graphique sunburst représentant la répartition par nationalité.],
)

L'ensemble de ces visualisations vise à rendre les résultats plus compréhensibles tout en permettant une exploration détaillée des données.

== Sources des données et fonctionnement du modèle

Les données utilisées proviennent de deux bases statistiques officielles publiées par Eurostat, l'office statistique de l'Union européenne.

La première, migr_eipre, recense le nombre de ressortissants de pays tiers détectés en situation de présence irrégulière dans les États membres. Les données sont ventilées selon différentes caractéristiques, notamment le pays déclarant, l'année, le sexe, l'âge et la nationalité. @eurostat_migr_eipre

La seconde, migr_pop1ctz, fournit des données relatives à la population résidente au 1er janvier, réparties par âge, sexe, nationalité et pays. Elle permet d'obtenir une population de référence pour calculer des indicateurs relatifs plutôt que de comparer uniquement des nombres absolus de détections. @eurostat_migr_pop1ctz

Les deux bases ont été nettoyées, harmonisées et regroupées en un ensemble de données exploitable pour l'entraînement du modèle. La version actuelle comprend 152 324 observations agrégées, couvrant 27 pays déclarants et la période 2018–2025.

Une difficulté importante concerne la différence entre les tranches d'âge des deux sources. Pour permettre leur rapprochement, une approximation a été appliquée en supposant une répartition uniforme de la population entre 15 et 64 ans. Cette hypothèse facilite les calculs, mais introduit une incertitude supplémentaire dans les comparaisons par âge.

Le modèle IA prédictif repose sur une méthode de gradient boosting adaptée à la modélisation de données de comptage, avec une régression de Poisson. Il apprend à partir des observations historiques afin de produire une estimation du taux agrégé pour les caractéristiques sélectionnées.

Les données de 2025 ont été utilisées comme période de validation temporelle lors du développement du modèle. Le modèle destiné à l'application a ensuite été réentraîné sur l'ensemble de la période disponible.

= Conclusion

text

#bibliography("refs.bib")
