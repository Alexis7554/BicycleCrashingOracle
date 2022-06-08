import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

title = "Données sources"
sidebar_name = "Données sources"

#--------------------------------
# CONTENT OF THE PAGE
#--------------------------------
def run():
    st.title(title)

    # Présentation des données
    st.subheader('Présentation des données sources')

    st.markdown("Nous avons utilisé les données fournies par le Ministère de l\'Intérieur sur data.gouv.fr (disponibles [ici](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2020/)).\n\n"
                "Ces données concernent les accidents corporels de la circulation, entre 2005 et 2020, sur toute la France, y compris les DROM-COM. Nous avons choisi de nous intéresser aux accidents de vélo, plus particulièrement au sein de Paris (où les données sont plus nombreuses).\n\n"
                "Nous disposons de 4 fichiers, dont voici le contenu :")

    file_details = pd.DataFrame(index = ["caracteristiques-20xx.csv", "lieux-20xx.csv", "usagers-20xx.csv", "vehicules-20xx.csv"],
                                data = [["15","Circonstances générales de l’accident (date, département, météo, intersection)"],["18","Décrit le lieu principal de l’accident (type de route, circulation, surface, infrastructures…)"],["15","Décrit les usagers impliqués (âge, gravité des blessures, position dans le véhicule…)"],["11","Décrit les véhicules impliqués (catégorie, obstacle heurté, point de choc, motorisation…)"]],
                                columns = ["Nombre de colonnes","Contenu"])

    st.table(file_details)

    st.markdown("*Le détail des colonnes de chaque fichier est disponible [ici](https://www.data.gouv.fr/fr/datasets/r/21e46f4e-fa29-40bb-9ec8-905e41e041aa)*")

    st.markdown("Pour un accident, plusieurs véhicules peuvent être impliqués. Chaque véhicule peut avoir un ou plusieurs passagers. Vus sous format UML simplifié, ces fichiers s’articulent ainsi :")

    st.image("assets/vue_classes.png", width = 500)

    # Obtention du dataset complet
    st.subheader("Obtention du dataset complet")

    st.markdown("Nous avons choisi, pour notre dataset, d'avoir une ligne par personne, plutôt que par accident.\n\n"
                "Après fusion des fichiers par année et concaténation des 4 fichiers, nous avons obtenu un dataset conséquent : **près de 2.4 millions de lignes** !\n\n"
                "En voici un rapide aperçu :")

    df = pd.read_csv("data/raw_df_sample.csv", sep=';')
    st.dataframe(df.head(), height = 600)

    st.markdown("Par la suite, nous avons filtré le contenu pour n'avoir que le vélos (catv = 1) sur Paris (en filtrant sur les départements, colonne 'dep').\n\n"
                "Avec ce filtrage, notre dataset 'brut' contient près de **75 000 entrées**.") 