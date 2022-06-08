import streamlit as st


title = "Conclusions et limites"
sidebar_name = "Conclusions"

#--------------------------------
# CONTENT OF THE PAGE
#--------------------------------
def run():

    st.title(title)
    
    st.markdown("Dans tout projet, surtout en lien avec du traitement de données et des statistiques, il est important de prendre en compte les effets des biais, limites et hypothèses de notre contexte.\n\n"
                "Avec les données que nous avions à disposition, nous avons réussi à obtenir **un score de 61%, score qui semble plafonner avec les modèles de Machine Learning**.\n"
                "Malgré nos différentes étapes de preprocessing, ce score ne semble pas pouvoir être beaucoup amélioré. Cependant, les limites et biais du modèle semblent être assez clairs et vous trouverez ci-dessous ces aspects détaillés.")
        
    st.subheader("Absence des données de circulation")
    st.markdown("***")

    st.markdown("Nous ne disposons pas des données sur la circulation, c'est-à-dire que nous ne savons pas le nombre de véhicules circulant à un instant t sur une voie X. Cela pose plusieurs problèmes :\n"
                "- Pour une voie X, nous ne pouvons pas corriger le nombre d'accidents pour obtenir une dangerosité. Deux exemples pour mieux comprendre :\n"
                "  - A : Une route très fréquentée (1M voitures/mois) avec 100 accidents par mois n'est pas dangereuse.\n"
                "  - B : Une route peu fréquentée (1000 voitures/mois) avec 10 accidents par mois est dangereuse.\n"
                "- La densité de circulation aurait pu être également une variable d'entrée de notre modèle : il y a des chances qu'un accident soit plus grave s'il y a plus de véhicules (multi-accidents par exemple).")

    st.markdown("Nous pouvons définir la dangerosité comme ceci :")

    st.latex(r"dangerosité = \frac{\sum \text{véhicules accidentés}}{\sum \text{véhicules ayant circulé}}")

    st.markdown("Cette nouvelle variable pourrait s'interpréter comme un pourcentage de chance d'avoir un accident sur la zone concernée si on y circule. Avec nos exemples nous aurions :\n"
                "- Gravité A = 100 / 1 000 000 = 0.01 %\n"
                "- Gravité B = 10 / 1000 = 1 %\n\n"
                "Notre modèle actuel classerait A comme plus dangereux car il y a plus d'accidents dans l'absolu, mais avec l'ajout d'une notion de dangerosité ce serait B qui serait 100 fois plus dangereuse.")


    st.subheader("Machine Learning vs Deep Learning")
    st.markdown("***")
 
    st.markdown("Comme vu dans la partie modélisation, nos modèles de Machine Learning n'ont pas dépassé un score de 60% environ. C'est vraiment peu, surtout par rapport à la frontière de 50%, qui correspondrait à de la pure chance.\n\n"
                "Il est possible que cette limite puisse être repoussée en utilisant plutôt des algorithmes de Deep Learning, qui pourraient mieux interpréter les données et en sortir un modèle plus performant.")


    st.subheader("Prédire le danger n'est pas prédire l'accident")
    st.markdown("***")

    st.markdown("Les données dont nous disposons ont pour cible la gravité d'un accident, sous-entendu l'état de santé d'une personne accidentée.\n"
                "Cependant, nous ne pouvons prédire cette gravité qu'en supposant que l'accident s'est bel et bien produit.\n"
                "Un véritable algorithme intéressant serait de permettre de prédire si un accident risque de se produire ou non, par exemple avec un modèle de régression qui prédirait la probabilité d'un accidenté comme proposé précédemment.\n\n"
                "Mieux encore, avec un modèle à deux sorties, il serait possible de prédire la probabilité d'un accident ainsi que sa gravité :\n"
                "- Target 1 : probabilité d'accident, une valeur entre 0 et 1 (régression).\n"
                "- Target 2 : gravité de l'accident, faible ou grave (classification avec deux classes.\n\n"
                "L'utilisation d'un modèle de Deep Learning customisé serait une bonne piste pour obtenir ces résultats.")


    st.subheader("Peut-on réellement prédire un accident ou sa gravité ?")
    st.markdown("***")

    st.markdown("Dans l'absolu, nous pourrions nous poser la question : \"Comment se produit un accident ?\". Plusieurs possibilités nous viennent à l'esprit quand on regarde d'autres statistiques :\n"
                "- La fatigue, l'inattention du conducteur.\n"
                "- L'alcool ou l'utilisation de drogues.\n"
                "- Le non-respect du code de la route, l'excès de vitesse.\n"
                "- Les conditions environnementales (brouillard, verglas).\n"
                "- D'autres conditions qui peuvent aggraver un accident sans pour autant le déclencher (santé des passagers, angle d'impact...).\n\n"
                "Nous ne disposons pas de ces données, hormis les conditions météos globales. On peut d'ailleurs remarquer que plusieurs facteurs sont purements humains, et ne sont pas mesurables en temps réel.\n\n"
                "Notre conclusion ici est qu'il n'est peut-être tout simplement pas possible de prédire un accident sans avoir des dizaines de variables supplémentaires très difficiles à obtenir ou quantifier.\n\n"
                "**Le mieux reste encore d'être vigilant pour éviter l'accident, d'anticiper les situations à risque et de tout faire pour réduire les dégâts causés (équipements de sécurité, entretien du véhicule, etc.).**")
    
    st.subheader("Notre retour d'expérience")
    st.markdown("***")

    st.markdown("Ce projet nous a permis de mettre en pratique nos acquis durant toute notre formation chez DataScientest. La difficulté liée aux données dont nous disposions nous a poussé à tester d'autres techniques et modèles, "
                "et bien qu'il soit un peu décevant de ne pas obtenir un score supérieur à 61%, l'analyse des biais et limites nous montre qu'il n'était peut-être pas possible d'obtenir de meilleurs résultats.\n\n"
                "Durant ce projet, nous avons pu :\n"
                "- Fusionner puis nettoyer nos données sources,\n"
                "- Rééquilibrer un jeu de données,\n"
                "- Tester plusieurs algorithmes, comparer leurs performances et leur vitesse,\n"
                "- Faire varier les hyper paramètres de ces algorithmes pour en trouver des optimaux,\n"
                "- Tester des algorithmes de Time Series,\n"
                "- **Nouvel acquis** : créer notre propre métrique de score,\n"
                "- **Nouvel acquis** : utiliser des algorithmes de classification que nous ne connaissions pas (LightGBM entre autres),\n"
                "- **Nouvel acquis** : utiliser Streamlit ainsi que de nouvelles librairries (PlotLy, Folium, Geopandas).\n\n"
                "**C'était donc un projet enrichissant et challengeant, mais nous pourrions pousser encore plus loin avec des algorithmes de Deep Learning.**")

    fin= st.button("Merci pour votre attention !")
    if fin ==True: st.balloons()