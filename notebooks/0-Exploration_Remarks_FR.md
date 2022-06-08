## Lexique

- **BAAC** : Bulletins d’Analyse des Accidents Corporels de la circulation
- **ONISR** : Observatoire National Interministériel de la Sécurité Routière

## Infos sur les fichiers sources
La base Etalab de données des accidents corporels de la circulation d'une année donnée, est répartie 
en 4 rubriques sous la forme pour chacune d'elles d'un fichier au format csv. 
1.  La rubrique CARACTERISTIQUES qui décrit les circonstances générales de l’accident 
2.  La  rubrique  LIEUX  qui  décrit  le  lieu  principal  de  l’accident  même  si  celui-ci  s’est  déroulé  à  une intersection 
3.  La rubrique VEHICULES impliqués 
4.  La rubrique USAGERS impliqués 
 
Chacune des variables contenues dans une rubrique doit pouvoir être reliée aux variables des autres 
rubriques. Le n° d'identifiant de l’accident (Cf. "Num_Acc") présent dans ces 4 rubriques permet d'établir 
un  lien  entre  toutes  les  variables  qui  décrivent  un  accident.  Quand  un  accident  comporte  plusieurs 
véhicules, il faut aussi pouvoir relier chaque véhicule à ses occupants. Ce lien est fait par la variable 
id_vehicule. 

La plupart des variables contenues dans les quatre fichiers précédemment énumérés peuvent contenir 
des cellules vides ou un zéro ou un point. Il s’agit, dans ces trois cas, d’une cellule non renseignée par 
les forces de l'ordre ou sans objet.

> Pour les join entre fichiers, utiliser **Num_Acc**  
> Pour relier les occupants au véhicule, utiliser **id_vehicule**  
> Il faudra remplacer les cellules vides, les point '.' et les zéros '0' par des NaN  

## Géolocalisation

Un accident peut être géolocalisé de plusieurs façons : 
- adresse partielle non normalisée (champ adr)  
- coordonnées gps  (projection WGS84) 
- numéro de la route, PR de rattachement et distance curviligne à ce PR

Attention  :  les  accidents  ne  sont  pas  tous  géolocalisés  de  façon  précise  à  travers  les  informations disponibles dans le Fichier BAAC et restituées ici. **A minima, seule la commune de l'accident est fournie.**

> 3 types de géolocalisation, a minima la commune  
> Est-ce qu'on ne garde que les coordonnées ? Regarder les proportions avec/sans coord. pour faire le choix

## Définitions accident corporel
Un accident corporel (mortel et non mortel) de la circulation routière relevé par les forces de l’ordre : 
- implique au moins une victime, 
- survient sur une voie publique ou privée, ouverte à la circulation publique, 
- implique au moins un véhicule. 
 

Un accident corporel implique un certain nombre d’usagers. Parmi ceux-ci, on distingue : 
- les  personnes  indemnes  :  impliquées  non  décédées  et  dont  l’état  ne  nécessite  aucun  soin 
médical du fait de l’accident, 
- les victimes : impliquées non indemnes. 
  - les personnes tuées : personnes qui décèdent du fait de l’accident, sur le coup ou dans les trente jours qui suivent l’accident, 
  - les personnes blessées : victimes non tuées.  
    - les blessés dits « hospitalisés » : victimes hospitalisées plus de 24 heures, 
    - les blessés légers : victimes ayant fait l'objet de soins médicaux mais n'ayant pas été admises comme patients à l'hôpital plus de 24 heures. 

## DOM-TOM
Les données concernent aussi des DOM-TOM depuis 2012 ou 2019 en fonction des cas
- Si on veut les inclure, il faut remonter à 2019 + 2020, mais plus loin ça posera peut-être problème ?

## Indicateur "blessé  hospitalisé"

Avertissement : Les données sur la qualification de blessé hospitalisé depuis l’année 2018 ne peuvent 
être comparées aux années précédentes suite à des modifications de process de saisie des forces de 
l’ordre.  L’indicateur  « blessé  hospitalisé »  n’est  plus  labellisé  par  l’autorité  de  la  statistique  publique 
depuis 2019.

***
## Questions et remarques TNI

- Est-ce qu'on s'intéresse aux personnes non blessées ?
- Est-ce qu'on regarde les accidents avec au moins un mort/blessé, est-ce qu'on compte les blessés (par véhicule ou au global) ?
  - Est-ce qu'on catégorise la gravité en fonction de ça ?
    - Au moins un décès
    - Au moins un blessé "grave" (hospitalisé > 24h)
    - Au moins un blessé "léger" (hospitalisé < 24h)
    - Aucun blessé (si répertorié, à vérifier)
  - On peut catégoriser aussi le type d'accident
    - Entre véhicules
    - Sur un piéton
    - On peut s'intéresser uniquement aux voitures ou pas...
- Il semblerait qu'on ait presque uniquement des variables catégorielles, vraiment très peu de float

# Idées de projet/besoin

- Carte (heatmap) qui montre les zones à risques avec des paramètres (météo, circu, heure/jour....)
- Indiquer le risque sur un trajet Google Maps / Waze, en se basant sur la succession d'étapes (et les autres infos comme la date, l'heure, la météo....)
  - Peut-être montrer graphiquement les tronçons/zones dangereuses
  - On pourrait se limiter à Paris : juste une ville, et possiblement des données de circulation disponibles
- Prédire le risque en fonction de sa place dans la voiture (ou moto, ou bus)

> - Carte/prédiction des accidents si on est en vélo (risque de se faire percuter à certains endroits)
>   - Cas concret : "j'ai peur d'utiliser Google Maps en vélo, car il me fait passer par des zones dangereuses"
  
## DataViz

- [A] Dangerosité des accidents de vélo p/r au voitures et moto (léger, grave, mortel)
  - Autre formulation : la mortalité par type de véhicule
  - Vérifier que le vélo c'est dangereux
- [A] En agglomération, la dangerosité s'il y a une piste cyclable ou pas (ou bande partagée...)
- [T] Une carte sur la France et/ou Paris qui montre les zones dangereuses pour les vélos (geopandas)
- [H] Montrer les accidents les plus fréquents en vélo (choc avec voiture, ou bien piétons, ou bien seul sur la chaussée...)
- [H] Regarder les données temporelles vis-à-vis des accidents ? Hiver vs été, matin vs soir... ?