import streamlit as st

title = "Bicycle Crashing Oracle : prédire la gravité d'un accident de vélo"
sidebar_name = "Introduction"

#--------------------------------
# CONTENT OF THE PAGE
#--------------------------------
def run():

    st.image("assets/header.jpg", width=800)

    st.subheader(title)

    st.markdown("---")

    st.markdown("### Présentation du projet")

    st.markdown("Dans le cadre de notre formation Data Scientist au sein de DataScientest.com, nous avons eu l’occasion de mettre en pratique nos acquis sur l’apprentissage machine, "
                "afin de produire un modèle capable d’anticiper la gravité des accidents sur les routes de France, dans l’espoir de pouvoir sauver les vies.\n\n"
                "Les accidents de la route ont un coût humain et économique très importants. Selon le bilan 2019 l’Observatoire National Interministériel de la Sécurité Routière (ONISR, [source](https://www.onisr.securite-routiere.gouv.fr/sites/default/files/2020-09/Bilan_2019_version_site_internet_24_sept.pdf), p.23), le coût total de l’insécurité routière serait de 50,9 milliards d’Euros, soit 2,2% du PIB. Il apparaît alors une nécessité d’apporter des solutions permettant d’anticiper un accident pour éviter les conséquences dévastatrices qui en résultent.\n\n"
                "Selon le bilan 2020 de l’ONISR (disponible [ici](https://www.onisr.securite-routiere.gouv.fr/etat-de-l-insecurite-routiere/bilans-annuels-de-la-securite-routiere/bilan-2020-de-la-securite-routiere)), qui prend en compte l’impact de la pandémie de Covid-19, la mortalité est en baisse, sauf pour les cyclistes où elle est en augmentation par rapport aux années précédentes. Ce mode s’est développé en milieu urbain pour éviter les transports en commun ainsi qu’en milieu rural pour pratiquer des loisirs de proximité.\n\n"
                "Le graphique ci-dessous fourni par l'ONISR montre les données relatives aux accidents corporels enregistrés par les forces de l'ordre, en France métropolitaine")
    
    st.image("assets/graphe_ONISR.png", width=700)

    st.markdown("### Contenu de cette présentation Streamlit")

    st.markdown("Le présent Streamlit vous permettra de voir rapidement le déroulement de notre projet, et de tester notre algorithme.")
    
    st.markdown("Sommaire :\n\n"
                "- Introduction : présente page, introduisant le projet et son contexte.\n"
                "- Données sources : nos fichiers CSV comme source, leurs liens et leur contenus bruts.\n"
                "- Dataviz' : quelques graphiques explicatifs pour mieux comprendre notre dataset.\n"
                "- Modélisations : notre preprocessing et la construction de notre modèle final.\n"
                "- Simulation : ici vous pourrez choisir quelques paramètres et lancer une simulation sur deux trajets que nous avons prédéfinis.\n"
                "- Conclusions : nos retours sur ce projet, ses limites et nos idées pour aller plus loin.")

    st.markdown("Vous pouvez retrouver notre code source sur [GitHub](https://github.com/DataScientest-Studio/Bicycle-Crashing-Oracle).")


    