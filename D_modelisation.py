import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

title = "Modélisation"
sidebar_name = "Modélisation"

#--------------------------------
# CONTENT OF THE PAGE
#--------------------------------
def run():
    st.title(title)
    st.markdown("---")
    # Preprocessing
    #--------------------------------
    st.header('Preprocessing')

    st.markdown("#### 1/ Nettoyage de la base de donéees")

    actions_table= pd.DataFrame(data= {'Actions':['Supprimer des colonnes', 'Remplacer des colonnes', 'Limiter les valeurs', 'Remplacer des valeurs absentes par la modalité', 'Supprimer les valeurs manquantes', 'Limiter age à une var. cat. de 5 classes'],
    'Colonnes':['dep, com, col, adr, lat, long, senc, obs, obsm, choc, manv, motor, occutc, voie, v1, v2, pr, pr1, lartpc, larrout, plan, place, locp, actp, etatp, catv, num_veh, Numm_Acc, prof, catu, trajet, secu',
    'an_nais remplacé par age',  'infra , nbv', 'surf , atm','les colonnes restantes', 'age']}
    ).set_index('Actions')
    
    st.table(actions_table)
    
    st.markdown("#### 2/ Limiter les classes cibles\n"
    "Nous avons limité les classes de la variable afin d'équilibrer mieux le dataset et améliorer la performance.")
    st.image("assets/limiting_categories.png", width = 1000)
    
    st.markdown("#### 3/ Répartition et encodage du Dataset\n\n"
    "Puisque toutes les colonnes sont catégorielles, nous avons fait un **Encodage OneHot**.")
    col1, col2, col3= st.columns(3)
    col1.metric("Train set", "80 %", "")
    col2.metric("Test set", "20 %", "")

    st.markdown("#### 4/ Equilibrage du Dataset\n\n"
    "La figure suivante montre la distribution des classes de notre dataset. On voit bien un déséquilibrage:")
    HtmlFile = open("plots/target_classes.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=450)  
    st.markdown("Afin de corriger le déséquilbrage dans notre Dataset et améliorer la performance, nous avons:\n\n"
    "* fait une réduction des features après le OneHot Encoding, grâce à **KSelectBest de Scikit-learn**.\n\n"
    "➜ Nombre de features réduit à **60** au lieu de 78.\n\n"
    "* fait un ré-échantillonage (**Sampling**) grâce à la fonction **UnderSampling de Imbalanced_learn**.\n\n"
    "➜ L'UnderSampling a remarquablement amélioré les matrices de confusion des modèles testés.\n\n")


    #Scorer personnalisé
    #--------------------------------
    st.header('Scorer personnalisé')
    st.markdown("- Etant donné l'importance de détecter la **catégorie positive** (1 : tué/hospitalisé), nous avons voulu **pénaliser** plus les faux négatifs :\n\n ➜ Donner plus de poids à la catgorie 1 pendant l'évalution de la performance.\n\n"
    "- D'où est venue la nécessité de définir une fonction de scoring personnalisée :\n\n"
    " ➜ La métrique choisie est la **moyenne géométrique pondérée (Weighted Geometric Mean WGM)** :"
    )
    st.latex(r'''{\large{WGM_\beta}} = {\large{(P.R^{\beta})}}\raisebox{1em}{$1/1+\beta$}''')
    value = st.checkbox("Afficher la variation du WGM selon Beta, le rappel et la précision")
    if value==True:
        HtmlFile = open("plots/WGM_values.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height=700, width=1200)  

    st.markdown("L'avantage de ce scorer est qu'il dépend plus du rappel et prend en considération la précision s'elle très faible. On a choisi  ***beta***=1.1.\n\n On a inclu le WGM dans le **GridSearchCV** comme métrique par défaut.")

    #Construction du modèle
    #--------------------------------
    st.header('Construction du modèle')
    st.markdown("Afin de trouver le modèle optimal, nous avons essayé une grande variété de modèles de classification supervisée."
    "La table ci-dessous résume le résultat des 8 meilleurs modèles testés :")
    
    results= pd.read_csv('data/Modelisation_results.csv', encoding='iso-8859-1', sep=';').sort_values('WGM test score', ascending=False).reset_index(drop=True)
    st.table(results)

    st.markdown("Pour faire la comparaison, on s'est basé sur : \n\n* **Le score WGM sur le Test set :** c'est le critère principal pour évaluer la performance."
    "\n\n* **Le score WGM sur le Train set :** pour vérifier l'*Overfitting*.\n\n* **Le temps de calcul :** il sert de 2ème critère pour évaluer le modèle à côté de WGM.\n\n"
    "* **Le score AUC :** pour avoir une idée sur le scoring sans pénalisation des faux négatifs, et permet aussi d'évaluer le modèle si le WGM ne suffit pas."
    "\n\nLes deux modèles qui répondent au mieux à nos attentes, sont : \n\n* **Gradient Boosting Classifier**\n\n* **LightGBM**\n\n"
    "Au regard de sa grande rapidité par rapport au 1er, on a choisi le **``LightGBM``**, d'autant plus qu'il n y'a pas de grande différence de score **(~0.1%)**\n\n"
    "Voici le schéma récapitulatif de notre modèle de prédiction (**Pipeline** de l'ensemble des *Transformers* et du Classificateur) :\n\n")
    
    st.image("assets/Pipeline.png", width = 700)
    st.text("")
    st.text("")
    st.markdown("\n\nLe modèle nous renvoie la matrice de confusion suivante lorsqu'il est appliqué sur l'ensemble de test :")
    HtmlFile = open("plots/confusion_matrix.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=500, width=1000)

    st.markdown("\n\n")
    st.markdown("#### ➜    Le score du modèle finale est  **``~ 61 %``** ")
    st.markdown("\n\n***")
    st.markdown("\n\n\n")
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    