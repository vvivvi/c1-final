# kaggle-final-project
Final project of the course" How to Win a Data Science Competition: Learn from Top Kagglers" by National Research University Higher School of Economics.

Author: Ville Viitaniemi

## Overview of the approach

This solution does not aim for the best possible prediction accuracy. Instead, a straightforward solution is described that involves most of the techniques mentioned in the final project requirements. 

### Task
In this competition the input data consists of daily sales data from the 1C Company, separated according to individual shops and items. The input data
is from the time period January 2013 - October 2015. The task is to predict the total sales for every product and store during the month of December 2015.

### Approach
In the  implemented approach
1. The day-level transactional data is aggregated into monthly summaries for each (shop,item) combination. Each monthly summary is considered as a row in a data matrix.  
2. Each row is augmented with additional features describing the particular item and shop. In particular, the items are grouped into categories derived from the item name text with help of text processing techniques, along with the provided item categories. 
3. Each row is augmented with temporally lagged versions of the target variable, i.e. the monthly sales count. In addition to sales history specific to each (item,shop) combination separately, we include also sales history aggregated over the item categories that were added in step 2, both on the level of individual items and an the shop level.
4. Sequential feature selection is performed to select a few subsets of the generated features. The features are selected to maximise the performance of CatBoost regression
models in a validation experiment.
5. The same validation setup is used for searching  optimal hyperparameters for both CatBoost and Random Forest regression models.
6. CatBoost and Random Forest regression models are trained for predicting the target month (December 2015) sales, based on each of the selected feature subsets.
7. The predictions of the models are ensembled using a simple averaging scheme.
8. Data leaking through the leaderboard is used for scaling 
the ensembled predictions optimally.


## Organisation of the work

List of notebooks
1. Exploratory data analysis
2. Data leak analysis
3. Feature generation
4. Feature selection 
5. hyperparameter tuning
6. Prediction model training 
7. Prediction generation 
8. Ensembling of predictions
9. Prediction scaling

It is enough to run the code in the notebooks 6.-8. to generate the final submission file, if the 
readily extracted features and the pre-trained models 
are used. One needs to run the notebooks 3.-9. if one wants 
to extract the features and train the models from scratch. 

## Setup

### Dependencies

requirements.txt specified the Python modules required to run the notebooks. The list may not be minimal,
but should be sufficient.

In conda, the environment can be recreated e.g. with the command line
conda create -n c1env --file requirements.txt

## Review criteria checklist

In this section the review criteria from the assignment instructions are listed and commented one by one. According to the instructions, 
it should be enough to comply with most of the requirements.

### Clarity

- The clear step-by-step instruction on how to produce the final submit file is provided. *Description in this document should suffice.*
- Code has comments where it is needed and meaningful function names. *Up to the reviewer to decide. Some comments and function names are there.*

### Feature preprocessing and generation with respect to models

- Several simple features are generated. *Nearly 100 features are generated. Some of them are simple, e.g. 'is_internet_store' and 'item_name_cyrillic_fraction'.*
- For non-tree-based models preprocessing is used or the absence of it is explained. *Only tree-based models are used here.*

### Feature extraction from text and images

- Features from text are extracted. *Yes.*
- Special preprocessings for text are utilized (TF-IDF, stemming, levenshtening...). *Yes, e.g. TF-IDF and stopword lists*

### EDA
- Several interesting observations about data are discovered and explained. *Yes, for example yearly trend and different store types.*
- Target distribution is visualized, time trend is assessed. *Yes*

### Validation
- Type of train/test split is identified and used for validation
*Yes. VAlidation setup is discussed in Notebook 4. for feature selection and hyperparameter tuning*
- Type of public/private split is identified
*Yes (Notebook 1 on EDA).*

### Data leakages
- Data is investigated for data leakages and investigation process is described. *Yes (Notebook 2.)*
- Found data leakages are utilized
* Yes. For example, the predictions are scaled as a post-processing step based on leaderboard probing.*
### Metrics optimization
- Correct metric is optimized. *Yes. After clipping the target value to the interval [0,20], the metric boils down to standard RMSE, which can be selected as optimisation target in both CatBoost and scikit-learn random forest libraries.*

### Advanced Features I: mean encodings
- Mean-encoding is applied. *No. Mean-encoding was tried out but it did not bring much in addition to the rich information content of the category-wise lagged target variables.*  
- Mean-encoding is set up correctly, i.e. KFold or expanding scheme are utilized correctly. *No, it is not set up at all.*

### Advanced Features II
- At least one feature from this topic is introduced
*Yes. The video "Statistics and distance based features" talks about "Calculating various statistics of one feature grouped by another" as one advanced feature engineering technique. In this solution, monthly sales are aggregated over shops and various item category variables." 
### Hyperparameter tuning
- Parameters of models are roughly optimal. *Yes The hyperparameters are found by a systematic grid search in a validation setup. They can thus be considered "roughly optimal.*

### Ensembles
- Ensembling is utilized (linear combination counts). *Yes.*
- Validation with ensembling scheme is set up correctly, i.e. KFold or Holdout is utilized. *No. Simple linear averaging scheme is utilised with fixed weights, and no validation is necessary.*
- Models from different classes are utilized (at least two from the following: KNN, linear models, RF, GBDT, NN). *Yes. The CatBoost library implements GBDT and random forest implementation from scikit-learn is also utilised.*

## Result summary

The optimally scaled predictions achieve leaderboard scores of 0.961351 and 0.95467 (public and private).  
