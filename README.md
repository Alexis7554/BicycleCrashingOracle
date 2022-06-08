
<img src="images/header.jpg">

# Bicycle Crashing Oracle :crystal_ball:


<table border="0">

 <colgroup>
    <col span="1" style="width: 70%;">
    <col span="1" style="width: 30%;">
 </colgroup>

 <tr>

   <td><img src="images/BCO_logo.png" width=500></td>

  <td>

   [@ Alexis FABRIES](https://github.com/Alexis7554)

   [@ Houssam JADDI](https://github.com/JAHOUD) 

   [@ Thomas NIEDERBERGER](https://github.com/Dharkhar) 
  </td>
 </tr>
</table>
<br/><br/>

## Introduction

As part of our training of  [Data Scientist](https://datascientest.com/en/data-scientist-course) at [*DataScientest*](https://datascientest.com/), we worked on this project in order to put into practice our new acquired skills.

Our project is about creating a prediction model for the bicycle accidents severity :bike: in France :fr:. The main idea is to have a model that can be later implemented in an  app e.g. *Google maps*.

The data is given by the French Interior Ministry via data.gouv.fr ( :link: [here](https://www.data.gouv.fr/en/datasets/bases-de-donnees-annuelles-des-accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2020/)). It contains all the collected informations about road accidents in France's roads :vertical_traffic_light: between 2005 and 2020.

***

## Source Code
Our source code can be found in the [`notebooks/`](notebooks) folder. Each Notebook coresponds to a different step in our project :

### 1/ Data exploration

#### a- Merging Datasets :mag:
- [`1-Merged_Dataset_2005_2020`](notebooks/1-Merged_Dataset_2005_2020.ipynb) :arrow_right: Merging the different files. We had 4 files/tables (users, caracteristics, location, vehicles) for every year between 2005 & 2020. All were assembled in a single CSV file.


#### b- Data Visualisation :bar_chart:
- [`2-Data_Visualization_1_Seaborn`](notebooks/2-Data_Visualization_1_Seaborn.ipynb) || [`2-Data_Visualization_2_Plotly`](notebooks/2-Data_Visualization_2_Plotly.ipynb) ||[`2-Data_Visualization_3_Geopandas`](notebooks/2-Data_Visualization_3_Geopandas.ipynb) :arrow_right:  Using our merged dataset, we made some graphics in order to analye the distribution/correlation/importance of some variables. This had guide us on how to proceed in the data preprocessing.

#### c- Filtering Dataset :scissors:
- [`3-Filtered_Dataset_2005_2020`](notebooks/3-Filtered_Dataset_2005_2020.ipynb)  :arrow_right: Creating a clean & filtered dataset from the raw one.
Modifications were made based on our estimation for the usefulness of each feature & its variables.

### 2/ Preprocessing & Modelling 
#### a- Correlation Analysis & Models Testing :hammer:
- [`4-Correlation_Analysis_&_Prediction_Modeling_1`](notebooks/4-Correlation_Analysis_&_Prediction_Modeling_1.ipynb)  :arrow_right: Analyse the correlation between all the features in our dataset, and building trial prediction models.
#### b- Creating a Customized Scorer & Tuning hyperparameters :wrench:
- [`5-Customized_scorer_&_Prediction_modeling_2`](notebooks/5-Customized_scorer_&_Prediction_modeling_2.ipynb)  :arrow_right: Define a customized scorer for our models, and tuning the hyperparameters of diffrent classification models using GridSearchCV.
#### c- Final Model :robot:
- [`6-Final_Model_&_Application`](notebooks/6-Final_Model_&_Application.ipynb) :arrow_right: Contains the Pipeline for the final classification model with the best parameters.


### 3/ Other Notebooks
The [`other_notebooks/`](notebooks/other_notebooks) folder contains some extra notebooks that were not included in the final model:
- [`5-Prediction_modeling_3`](notebooks/other_notebooks/5-Prediction_modeling_3.ipynb)  :arrow_right:  A second approch for the preprocessing with the same results.
- [`7-Time_Forecasting`](notebooks/other_notebooks/7-Time_Forecasting.ipynb)  :arrow_right:  Time serie forecasting. This wasn't included in the final model because it had no impact on the result.
- ...

## Streamlit App

To run the app :

```shell
cd streamlit
conda create --name BCO-streamlit python=3.9
conda activate BCO-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

**Docker**

You can also run the Streamlit app in a [Docker](https://www.docker.com/) container. To do so, you will first need to build the Docker image :

```shell
cd streamlit
docker build -t streamlit .
```

You can then run the container using :

```shell
docker run --name streamlit -p 8501:8501 streamlit
```

And again, the app should then be available at [localhost:8501](http://localhost:8501).
