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
      caption: [Les variables disponibles],
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

Plusieurs visualisations permettent ensuite d'explorer les résultats.

=== Évolution au fil des années

Un premier graphique présente l'évolution du taux de détection entre 2018 et 2025. Il compare les valeurs calculées à partir des données observées aux estimations produites par le modèle.

Cette représentation permet d'identifier les variations temporelles et d'observer dans quelle mesure le modèle reproduit les tendances historiques.

=== Comparaison des tranches d'âge et du sexe

Un deuxième graphique compare les différentes tranches d'âge sous la forme de barres verticales empilées.

Chaque barre représente le taux de détection total de la tranche d'âge concernée. Elle est divisée selon les catégories de sexe, avec une couleur rose pour les femmes, bleue pour les hommes et grise pour les données dont le sexe serait inconnu.

La hauteur totale de la barre correspond au taux observé pour l'ensemble du groupe d'âge. Les segments représentent la contribution de chaque catégorie au total, tandis que les taux propres aux hommes et aux femmes sont accessibles au survol.

Dans la version actuelle des données, seules les catégories hommes et femmes sont disponibles séparément. La catégorie inconnue est donc prévue dans l'interface, mais ne dispose pas de valeurs distinctes.

=== Carte mondiale et diagramme circulaire interactif

Une troisième partie propose une visualisation géographique des statistiques par nationalité.

Une carte du monde interactive représente les pays de nationalité des personnes recensées dans les données. Elle est accompagnée d'un diagramme circulaire hiérarchique, appelé sunburst, qui regroupe les nationalités par grandes régions géographiques.

Les deux représentations sont synchronisées : lorsqu'un utilisateur survole un pays sur la carte, la partie correspondante du diagramme est mise en évidence, et inversement. Des transitions progressives facilitent la lecture visuelle.

Un sélecteur permet de basculer entre deux indicateurs : le taux de détections rapporté à la population et le nombre absolu de détections enregistrées.

Afin de préserver la lisibilité, les nationalités dont la contribution au total des détections est inférieure à 1 % sont regroupées dans une catégorie « Other » au sein du diagramme circulaire. Elles restent néanmoins visibles individuellement sur la carte lorsqu'elles disposent d'une population de référence d'au moins 1 000 personnes.

La carte illustre les pays de nationalité, et non les lieux dans lesquels les personnes ont été détectées en situation irrégulière. Cette distinction évite de confondre l'origine des populations étudiées avec la localisation des contrôles migratoires.

L'ensemble de ces visualisations vise à rendre les résultats plus compréhensibles tout en permettant une exploration détaillée des données.

== 4. Sources des données et fonctionnement du modèle

Les données utilisées proviennent de deux bases statistiques officielles publiées par Eurostat, l'office statistique de l'Union européenne.

La première, migr_eipre, recense le nombre de ressortissants de pays tiers détectés en situation de présence irrégulière dans les États membres. Les données sont ventilées selon différentes caractéristiques, notamment le pays déclarant, l'année, le sexe, l'âge et la nationalité.

La seconde, migr_pop1ctz, fournit des données relatives à la population résidente au 1er janvier, réparties par âge, sexe, nationalité et pays. Elle permet d'obtenir une population de référence pour calculer des indicateurs relatifs plutôt que de comparer uniquement des nombres absolus de détections.

Les deux bases ont été nettoyées, harmonisées et regroupées en un ensemble de données exploitable pour l'entraînement du modèle. La version actuelle comprend 152 324 observations agrégées, couvrant 27 pays déclarants et la période 2018–2025.

Une difficulté importante concerne la différence entre les tranches d'âge des deux sources. Pour permettre leur rapprochement, une approximation a été appliquée en supposant une répartition uniforme de la population entre 15 et 64 ans. Cette hypothèse facilite les calculs, mais introduit une incertitude supplémentaire dans les comparaisons par âge.

Le modèle prédictif repose sur une méthode de gradient boosting adaptée à la modélisation de données de comptage, avec une régression de Poisson. Il apprend à partir des observations historiques afin de produire une estimation du taux agrégé pour les caractéristiques sélectionnées.

Les données de 2025 ont été utilisées comme période de validation temporelle lors du développement du modèle. Le modèle destiné à l'application a ensuite été réentraîné sur l'ensemble de la période disponible.

Sources principales :

- Eurostat, [*Third-country nationals found to be illegally present* — migr_eipre](https://ec.europa.eu/eurostat/databrowser/view/migr_eipre/default/table?lang=en).
- Eurostat, [*Population on 1 January by age, sex and citizenship* — migr_pop1ctz](https://ec.europa.eu/eurostat/databrowser/view/migr_pop1ctz/default/table?lang=en).

== 5. Périmètre de l'estimation et limites

L'application produit deux types de résultats : un indicateur observé, directement calculé à partir des données Eurostat, et une estimation de cet indicateur générée par le modèle d'intelligence artificielle.

L'indicateur observé correspond au rapport entre le nombre de détections de présence irrégulière enregistrées pendant une année et la population résidente de référence au début de cette même année, exprimé en pourcentage.

Ce calcul présente plusieurs limites méthodologiques importantes.

Premièrement, les statistiques de détection ne recensent pas nécessairement toutes les personnes en situation irrégulière. Elles dépendent notamment des politiques de contrôle, des moyens mobilisés par les autorités et des pratiques administratives nationales. Deux pays présentant des taux différents ne comptent donc pas nécessairement des proportions différentes de personnes en situation irrégulière.

Deuxièmement, la population utilisée comme dénominateur ne correspond pas exclusivement aux personnes en séjour légal. Eurostat recense une population résidente, sans vérifier individuellement la validité des titres de séjour. Par ailleurs, certains visiteurs de courte durée susceptibles d'être détectés ne figurent pas dans cette population.

Troisièmement, les détections correspondent à des événements enregistrés sur une période, tandis que la population de référence est mesurée à une date précise. Ces deux mesures ne décrivent donc pas exactement le même ensemble statistique.

Enfin, les données peuvent comporter des valeurs manquantes, des catégories insuffisamment représentées et des différences de déclaration entre pays. Les approximations réalisées pour harmoniser les groupes d'âge constituent une source supplémentaire d'incertitude.

Par conséquent, le pourcentage affiché ne doit pas être interprété comme la probabilité qu'un individu donné soit en situation irrégulière. Il s'agit d'un indicateur statistique agrégé, destiné à étudier les données de détection disponibles.

Cette limite définit également le périmètre d'utilisation du prototype : celui-ci est conçu comme un outil expérimental de visualisation et d'analyse, et non comme un instrument de décision administrative, de contrôle migratoire ou de profilage individuel.

Le prototype démontre ainsi la possibilité de combiner des données publiques, des modèles prédictifs et des visualisations interactives. Il met également en évidence une question centrale du projet : dans quelle mesure les résultats d'une intelligence artificielle peuvent-ils être considérés comme fiables lorsque les données utilisées ne représentent qu'une partie d'un phénomène social complexe ?

exemple d'une formule : `O_max + 2 × Hyperperiod`.

= IA, frontières et politiques migratoires

TODO : choisir les sources et les ajouter ceci dans refs.bib

Aussi le 5 est l'article du prof. Faut peut-être l'ajouter.

Voici les données utilisées : 
- https://ec.europa.eu/eurostat/databrowser/view/migr_eipre__custom_24484187/default/table

- https://ec.europa.eu/eurostat/databrowser/view/migr_pop1ctz__custom_24484552/default/table

== Beduschi (2021) — Gouverner la migration à l'ère de l'IA

*Référence.* Beduschi, Ana (2021). « International migration management in the age of artificial intelligence ». _Migration Studies_, 9(3), 576–596.

*Résumé.* L'article examine comment l'intelligence artificielle transforme la gestion internationale des migrations : modernisation administrative, production de connaissances statistiques et renforcement des asymétries de pouvoir entre États. Il souligne que les données quantitatives, présentées comme objectives, peuvent masquer des hypothèses politiques et négliger des informations qualitatives.

*Intérêt pour le projet.* Situer le tableau de bord dans son contexte institutionnel et montrer qu'une prédiction statistique peut orienter les politiques de contrôle.

#link("https://doi.org/10.1093/migration/mnaa003")[Accéder à l'article (DOI)]

== Molnar (2019) — Les migrants comme terrain d'expérimentation

*Référence.* Molnar, Petra (2019). « Technology on the margins: AI and global migration management from a human rights perspective ». _Cambridge International Law Journal_, 8(2), 305–330.

*Résumé.* Molnar examine les technologies automatisées déployées dans les procédures migratoires et les risques qu'elles font peser sur les droits humains. Elle critique le développement de systèmes peu transparents sur des populations ayant souvent peu de moyens de contester une décision et insiste sur les inégalités de pouvoir entre concepteurs, administrations et personnes concernées.

*Intérêt pour le projet.* Questionner la légitimité de prédire une situation juridique sur la base de catégories démographiques et la possibilité d'un usage répressif ultérieur.

#link("https://doi.org/10.4337/cilj.2019.02.07")[Accéder à l'article (DOI)]

== Nalbandian (2022) — Efficacité administrative contre protection des droits

*Référence.* Nalbandian, Lucia (2022). « An eye for an ‘I:’ a critical assessment of artificial intelligence tools in migration and asylum management ». _Comparative Migration Studies_, 10, article 32.

*Résumé.* À travers trois études de cas (HCR, Nouvelle-Zélande et États-Unis), cette recherche compare des usages de l'IA pour l'identification biométrique, le tri des demandes et la surveillance migratoire. Elle montre que les promesses de rapidité et d'efficacité doivent être confrontées à leurs coûts en matière de vie privée, de transparence et de vulnérabilité.

*Intérêt pour le projet.* Construire un tableau « bénéfices attendus / risques sociaux » et discuter les usages possibles de l'application au-delà de son prototype.

#link("https://doi.org/10.1186/s40878-022-00305-0")[Accéder à l'article en libre accès]

== Vavoula (2021) — Le profilage aux frontières Schengen

*Référence.* Vavoula, Niovi (2021). « Artificial Intelligence (AI) at Schengen Borders: Automated Processing, Algorithmic Profiling and Facial Recognition in the Era of Techno-Solutionism ». _European Journal of Migration and Law_, 23(4), 457–484.

*Résumé.* L'article analyse la place croissante du traitement automatisé, du profilage et de la reconnaissance faciale dans les dispositifs européens de contrôle des frontières. Il interroge le recours à des indicateurs de risque, les garanties de protection des données et la tentation de considérer la technologie comme une solution autonome à un problème politique et administratif complexe.

*Intérêt pour le projet.* Discuter l'emploi de l'âge, du sexe ou de la citoyenneté comme variables de profilage, ainsi que le risque de réutilisation des pourcentages dans des contrôles individuels.

#link("https://doi.org/10.1163/15718166-12340114")[Accéder à l'article (DOI)]

== Derave, Genicot et Hetmanska (2022) — Les risques d’une IA « digne de confiance » dans ETIAS

*Référence.* Derave, Charly, Nathan Genicot et Nina Hetmanska (2022). « The Risks of Trustworthy Artificial Intelligence: The Case of the European Travel Information and Authorisation System ». _European Journal of Risk Regulation_, 13, 389–420. DOI : 10.1017/err.2022.5.

*Résumé.* Les auteurs analysent le système européen d'information et d'autorisation concernant les voyages (ETIAS) comme un cas de profilage des personnes étrangères. À partir du dispositif juridique alors prévu, ils montrent que l'analyse de risque reposant sur l'âge, le sexe, la nationalité et d'autres attributs peut défavoriser certains groupes, même si les décisions sont présentées comme objectives. Ils critiquent notamment la reproduction de biais dans les données administratives, l'utilisation de caractéristiques servant de substituts à des catégories protégées (« proxies »), le manque de transparence et les possibles boucles de rétroaction. L'article interroge le décalage entre le discours européen sur une IA « digne de confiance » et la surveillance des frontières.

*Intérêt pour le projet.* C'est une référence particulièrement proche du prototype : notre application compare des statistiques selon l'âge, le sexe et la citoyenneté, des variables qui peuvent faciliter un profilage discriminatoire si un pourcentage de groupe est interprété comme un risque individuel. Les auteurs décrivent aussi un problème directement lié aux données d'entraînement : les chiffres de contrôle et de dépassement de séjour peuvent incorporer des biais de sélection et des situations juridiques mal catégorisées. *Précision :* l'article date de 2022 et étudie principalement le cadre et le fonctionnement anticipé d'ETIAS ; il ne démontre ni que les systèmes actuels utilisent exactement ces techniques ni que notre prototype fonctionne comme ETIAS.

*Passages utiles.* Pages 404–408 : fonctionnement des indicateurs de risque ; pages 412–414 : données d'entraînement biaisées, détections et effets auto-renforçants ; pages 415–417 : choix des variables, discrimination par procuration et effets intersectionnels.

#link("https://doi.org/10.1017/err.2022.5")[Accéder à l'article (DOI)]

== Forti (2024) — Les erreurs algorithmiques ont des conséquences concrètes

*Référence.* Forti, Mirko (2024). « Addressing Algorithmic Errors in Data-Driven Border Control Procedures ». _German Law Journal_, 25(4), 635–645.

*Résumé.* Forti étudie les erreurs produites par des systèmes de traitement de données aux frontières européennes. Il insiste sur les conséquences disproportionnées d'une mauvaise classification pour les personnes migrantes et demande des garanties permettant d'identifier, de corriger et de contester des résultats erronés. L'article critique l'idée selon laquelle une sortie algorithmique serait nécessairement plus fiable qu'une appréciation humaine.

*Intérêt pour le projet.* Justifier l'affichage des incertitudes, les avertissements d'usage et l'interdiction d'utiliser le score comme preuve d'un statut juridique.

#link("https://doi.org/10.1017/glj.2023.102")[Accéder à l'article en libre accès]

== Stewart (2024) — La responsabilité sous le règlement européen sur l'IA

*Référence.* Stewart, Ludivine Sarah (2024). « The regulation of AI-based migration technologies under the EU AI Act: (Still) operating in the shadows? ». _European Law Journal_, 30(1–2), 122–135.

*Résumé.* Cette analyse juridique étudie la capacité du règlement européen sur l'IA à encadrer les technologies appliquées à la migration et aux frontières. Elle examine la qualification des systèmes à risque, la répartition des obligations entre développeurs et utilisateurs et les difficultés de contrôle public. Elle relève des lacunes possibles malgré l'existence de nouvelles exigences réglementaires.

*Intérêt pour le projet.* Construire la partie consacrée à la gouvernance, à la traçabilité, à l'évaluation des risques et à la responsabilité du développeur. La conformité d'un usage concret doit être vérifiée séparément.

#link("https://doi.org/10.1111/eulj.12516")[Accéder à l'article (DOI)]

== Rinaldi et Teo (2025) — Dignité et autonomie des personnes

*Référence.* Rinaldi, Alberto, et Sue Anne Teo (2025). « The Use of Artificial Intelligence Technologies in Border and Migration Control and the Subtle Erosion of Human Rights ». _International & Comparative Law Quarterly_, 74(1), 61–89.

*Résumé.* Les auteurs montrent que l'IA appliquée aux frontières peut affecter les droits humains au-delà des problèmes habituels de confidentialité et de discrimination. Ils examinent notamment la liberté de pensée, la perte de pouvoir d'agir des individus et la dignité humaine. L'étude attire l'attention sur les effets diffus de la normalisation de la surveillance.

*Intérêt pour le projet.* Expliquer pourquoi l'étiquette « risque de séjour irrégulier » peut stigmatiser des personnes même lorsqu'aucune décision automatique n'est prise.

#link("https://doi.org/10.1017/S0020589325000090")[Accéder à l'article en libre accès]

= Profilage, discriminations et équité

== Leese (2014) — Le profilage algorithmique et les protections antidiscriminatoires

*Référence.* Leese, Matthias (2014). « The new profiling: Algorithms, black boxes, and the failure of anti-discriminatory safeguards in the European Union ». _Security Dialogue_, 45(5), 494–511.

*Résumé.* À partir du cas des données passagers dans l'Union européenne, Leese analyse le passage d'un profilage fondé sur des catégories explicites à un profilage construit à partir de régularités statistiques. Il explique pourquoi la complexité et l'opacité de ces catégories rendent plus difficile l'application pratique des protections contre les discriminations.

*Intérêt pour le projet.* Examiner comment des variables telles que la citoyenneté ou l'âge peuvent produire des classements socialement discriminants sans règle discriminatoire explicite dans le code.

#link("https://doi.org/10.1177/0967010614544204")[Accéder à l'article (DOI)]

== Barocas et Selbst (2016) — Des données apparemment neutres peuvent discriminer

*Référence.* Barocas, Solon, et Andrew D. Selbst (2016). « Big Data's Disparate Impact ». _California Law Review_, 104(3), 671–732.

*Résumé.* Les auteurs expliquent comment la collecte des données, le choix des variables, la définition de la cible et l'entraînement d'un algorithme peuvent reproduire ou amplifier des inégalités sociales. Une décision entièrement automatisée peut ainsi créer des effets discriminatoires même sans intention explicite de discriminer.

*Intérêt pour le projet.* Analyser l'utilisation de caractéristiques démographiques sensibles ou corrélées et le risque que les distributions historiques soient interprétées comme des propriétés individuelles.

#link("https://doi.org/10.15779/Z38bg31")[Accéder à l'article (DOI)]

== Wachter, Mittelstadt et Russell (2021) — L'équité n'est pas un simple score

*Référence.* Wachter, Sandra, Brent Mittelstadt et Chris Russell (2021). « Why fairness cannot be automated: Bridging the gap between EU non-discrimination law and AI ». _Computer Law & Security Review_, 41, article 105567.

*Résumé.* L'article confronte les indicateurs statistiques d'équité utilisés en apprentissage automatique aux notions juridiques de discrimination dans l'Union européenne. Il montre qu'aucune métrique unique ne remplace l'analyse du contexte, des effets concrets et des protections juridiques applicables.

*Intérêt pour le projet.* Éviter de conclure qu'un modèle est « équitable » parce que ses erreurs sont similaires entre catégories ; compléter les métriques par une analyse qualitative.

#link("https://doi.org/10.1016/j.clsr.2021.105567")[Accéder à l'article (DOI)]

== Selbst et al. (2019) — Les pièges d'une approche uniquement technique

*Référence.* Selbst, Andrew D., danah boyd, Sorelle A. Friedler, Suresh Venkatasubramanian et Janet Vertesi (2019). « Fairness and Abstraction in Sociotechnical Systems ». _Proceedings of the Conference on Fairness, Accountability, and Transparency_, 59–68.

*Résumé.* Les auteurs identifient les erreurs qui surviennent lorsque l'on isole un modèle mathématique des institutions et des pratiques sociales dans lesquelles il est utilisé. Ils montrent que la correction statistique d'un algorithme ne suffit pas à résoudre des problèmes de justice, d'accès aux droits ou de responsabilité.

*Intérêt pour le projet.* Analyser séparément le modèle, son interface, les décisions qu'un utilisateur pourrait prendre sur la base du score et les personnes susceptibles d'en subir les conséquences.

#link("https://doi.org/10.1145/3287560.3287598")[Accéder à l'article (DOI)]

= Qualité des données et limites statistiques

== Jacobs et Wallach (2021) — Mesurer le bon phénomène

*Référence.* Jacobs, Abigail Z., et Hanna Wallach (2021). « Measurement and Fairness ». _Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency_, 375–385.

*Résumé.* L'article relie l'équité algorithmique à la théorie de la mesure. Il distingue le concept que l'on voudrait étudier et les variables concrètement disponibles, puis propose de vérifier leur validité et leur fiabilité. Une donnée administrativement enregistrée ne représente pas nécessairement le phénomène social qu'on cherche à prédire.

*Intérêt pour le projet.* Distinguer « séjour irrégulier réel », « personne détectée » et « population recensée » ; discuter la validité du numérateur et du dénominateur.

#link("https://doi.org/10.1145/3442188.3445901")[Accéder à l'article (DOI)]

== Suresh et Guttag (2021) — Les biais apparaissent tout au long du cycle de vie

*Référence.* Suresh, Harini, et John Guttag (2021). « A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle ». _Proceedings of the ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization (EAAMO)_ .

*Résumé.* Les auteurs proposent un cadre permettant de repérer différentes sources de préjudice dans les systèmes de machine learning : collecte et représentation des données, étiquetage, modélisation, évaluation et déploiement. Le risque ne se limite pas à la phase d'entraînement ; il peut être introduit ou aggravé par les choix de conception.

*Intérêt pour le projet.* Structurer un audit des fichiers Eurostat, du regroupement des tranches d'âge, des données manquantes, de la métrique affichée et des limites de l'interface.

#link("https://doi.org/10.1145/3465416.3483305")[Accéder à l'article (DOI)]

== Ensign et al. (2018) — Les boucles de rétroaction dans les données de contrôle

*Référence.* Ensign, Danielle, Sorelle A. Friedler, Scott Neville, Carlos Scheidegger et Suresh Venkatasubramanian (2018). « Runaway Feedback Loops in Predictive Policing ». _Proceedings of Machine Learning Research_, 81, 160–171.

*Résumé.* À propos de la police prédictive, cette étude mathématique montre comment un algorithme fondé sur les incidents découverts lors de contrôles peut orienter de nouveaux contrôles vers les mêmes lieux. Les données futures reflètent alors davantage les endroits surveillés, ce qui entretient un cercle de rétroaction au lieu de révéler fidèlement la distribution du phénomène étudié.

*Intérêt pour le projet.* Expliquer par analogie pourquoi les chiffres de détection des séjours irréguliers peuvent dépendre de l'intensité et de la répartition des contrôles, et non uniquement de la prévalence réelle.

#link("https://proceedings.mlr.press/v81/ensign18a.html")[Accéder à l'article en libre accès]

== Lakkaraju et al. (2017) — Le problème des étiquettes observées sélectivement

*Référence.* Lakkaraju, Himabindu, Jon Kleinberg, Jure Leskovec, Jens Ludwig et Sendhil Mullainathan (2017). « The Selective Labels Problem: Evaluating Algorithmic Predictions in the Presence of Unobservables ». _Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_.

*Résumé.* L'article examine les situations où l'on ne connaît le résultat à prédire que pour une sélection non aléatoire des cas. Il explique pourquoi mesurer les performances sur les seuls exemples observables peut être trompeur et présente une méthode d'évaluation adaptée à certains contextes de décision humaine.

*Intérêt pour le projet.* Les personnes non contrôlées peuvent rester absentes de la base des détections. Cette publication aide à démontrer pourquoi une forte performance sur ces données ne prouve pas une bonne estimation de la situation juridique de la population.

#link("https://doi.org/10.1145/3097983.3098066")[Accéder à l'article (DOI)]

= Source de recherche complémentaire : mesurer la migration irrégulière

== Siruno et al. (2024) — MIrreM, flux de migration irrégulière

*Référence.* Siruno, Lalaine, et al. (2024). _MIrreM Working Paper on Irregular Migration Flows_. Projet européen Measuring Irregular Migration and Related Policies (MIrreM). *Document de recherche / working paper : à distinguer des seize publications académiques ci-dessus.*

*Résumé.* Le document décrit les catégories de données disponibles pour étudier les flux migratoires irréguliers, leurs divergences conceptuelles et leurs limites de comparabilité entre pays et périodes. Il souligne que les statistiques de ressortissants de pays tiers trouvés en situation irrégulière ne recensent pas toutes les personnes en situation irrégulière, notamment celles qui ne sont pas détectées.

*Intérêt pour le projet.* Référence directement applicable au jeu de données Eurostat `migr_eipre` : les données de détection ne permettent pas, sans hypothèses supplémentaires solides, de calculer une probabilité individuelle de séjour irrégulier.

#link("https://zenodo.org/records/10702228")[Accéder au working paper et à ses fichiers]

= Proposition de plan pour le rapport

+ *Présentation du prototype.* Objectif de l'application, variables disponibles, interface, sources Eurostat et périmètre de l'estimation.
+ *Validité scientifique de la cible.* Différence entre détection et statut réel, populations de référence, regroupement des âges et conditions nécessaires à une interprétation probabiliste (Jacobs et Wallach ; Lakkaraju et al. ; MIrreM).
+ *Biais et discrimination.* Risques associés au sexe, à l'âge et à la citoyenneté ; sélection des personnes contrôlées ; boucles de rétroaction (Barocas et Selbst ; Leese ; Derave, Genicot et Hetmanska ; Ensign et al.).
+ *Conséquences humaines et sociales.* Stigmatisation, autonomie, confiance, protection des personnes vulnérables et usages détournés (Molnar ; Nalbandian ; Rinaldi et Teo).
+ *Responsabilité et réglementation.* Transparence, contestation des erreurs, protection des données et exigences applicables selon l'usage réel (Forti ; Stewart ; Derave, Genicot et Hetmanska ; Wachter et al.).
+ *Mesures d'atténuation.* Limiter l'outil à des indicateurs agrégés, afficher les dénominateurs et incertitudes, documenter les incompatibilités de données, évaluer les effets par groupe, éviter l'identification ou le ciblage de personnes.

= Questions critiques à traiter explicitement

- *Que prédit-on vraiment ?* Un taux de détection dans une population de référence ou un statut de séjour, qui n'est pas directement observé ?
- *Qui est absent des données ?* Les personnes non détectées, certaines catégories de résidents, des groupes sous-représentés et les périodes sans reporting complet.
- *L'âge et la citoyenneté sont-ils des causes ?* Non : un lien statistique peut résulter de politiques de contrôle, de règles juridiques ou de biais de sélection.
- *Qui pourrait utiliser le résultat ?* Un outil descriptif destiné à l'analyse publique n'a pas les mêmes conséquences qu'un outil de ciblage individuel.
- *Peut-on contester une erreur ?* Si les résultats influencent une décision, les garanties de justification, de contrôle humain et de recours deviennent centrales.


= Conclusion

text

#bibliography("refs.bib")
