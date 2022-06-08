import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

title = "Dataviz'"
sidebar_name = "Dataviz'"

#--------------------------------
# CONTENT OF THE PAGE
#--------------------------------
def run():
    st.title(title)

    st.markdown("La première étape avant de modéliser est de regarder nos données : quelles sont nos variables, la répartition de leurs valeurs, statistiques clés, équilibre des données...\n\n"
                "Ici nous avons choisi de regarder nos données selon 4 thématiques.")

    # Severité
    st.subheader('Répartition par sévérité')

    st.markdown("La gravité d'un accident est mesurée par l'état de santé des personnes impliquées dans l'accident (qu'elles soient au volant ou passagers).")
    st.markdown("Nous disposons de 4 niveaux de gravité dans notre dataset :\n"
                "- Indemne : la personne n'a eu que des blessures légères, n'ayant pas nécessité d'hospitalisation\n"
                "- Blessé léger : la personne a eu besoin de soins en hôpital pour une durée **inférieure à 24 heures**\n"
                "- Blessé lourd : la personne a eu besoin de soins en hôpital pour une durée **supérieure à 24 heures**\n"
                "- Tué : la personne est décédée lors de l'accident")

    st.markdown("La gravité de l'accident est notre variable cible, celle que nous souhaitons prédire.")

    st.markdown("Le graphique ci-dessous montre le nombre et la répartition par gravité.\n\n"
                "Point important : On remarque que nos données sont très déséquilibrées, ce qui sera pris en compte lors de la modélisation.\n\n"
                "*On remarque que la majeure partie des victimes est dans la catégorie 'blessé léger', mais on compte tout de même 3.5% des victimes qui sont décédées à la suite de l'accident.*")

    HtmlFile = open("plots/bicycle_acidents_severity.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=600, width=1100)

    # Age
    st.subheader('Répartition par âge')
    st.markdown("On remaque que la distribution du nombre d'accidents varie selon l'âge: \n\n" 
                "le nombre d'accidents est plus important pour la tranche d'âge 12-20 ans, puis se stabilise jusque 60 ans avant de décroitre.\n\n")
    HtmlFile = open("plots/bike_acidents_by_age.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=500, width=1100)

    st.markdown("La proportion d'accidents graves semble également plus importante pour les personnes âgées (après 60 ans) \n\n")
    HtmlFile = open("plots/bike_acidents_mortality_by_age.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=500, width=1100)

    # Temporel
    st.subheader('Données temporelles')
    st.markdown("#### Par mois dans l'année")
    st.markdown("On constate que le nombre d'accidents est plus important pendant la période d'été entre les mois de mai et octobre.\n\n")
    HtmlFile = open("plots/bike_acidents_by_month.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=500, width=1100) 
    st.markdown("#### Par jour de la semaine")
    st.markdown("On constate que le nombre d'accidents est plus important en milieu de semaine et plus faible le week end.\n\n")
    HtmlFile = open("plots/bike_acidents_by_weekday.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=500, width=1100)  
    st.markdown("#### Par heure de la journée")
    st.markdown("Ce graphique nous permet d'observer un pic du nombre d'accidents le soir en heure de pointe entre 16 heures et 18 heures.\n\n")
    HtmlFile = open("plots/bike_acidents_by_hour.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=520, width=1100)

    # Map Paris
    st.subheader('Carte des accidents dans Paris')
    st.markdown("Il semblait également interessant d'observer la répartition des accidents dans Paris sur une carte qui pourrait permettre de visualiser les zones à risque dans Paris\n"
                "Instinctivement, on pouvait penser que certains lieux à forte circulation montreraient des groupes d’accidents, or c’est plus diffus que cela.\n"
                )
    st.image("assets/graphe_geopandas.png", width=1200)