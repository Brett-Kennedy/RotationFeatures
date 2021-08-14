# RotationFeatures 

RotationFeatures is a feature generation tool that constructs sets of pairs of new features based on rotations of existing pairs of features. The tool follows a similar API signature as sklearn's PolynomialFeatures.

RotationFeatures is designed specifically for interpretable models such as Decision Trees, Decision Tables, Rule Lists and Rule Sets, though may be used with any model type where feature generation may be useful. Though the features generated may have lower comprehensibility than the original features from which they were created, they may be readily visualized, which often supports greater overall interpertability and allows this tool to contribute to eXplainable AI (XAI). 

## Installation

`
pip install RotationFeatures
`

## Description

The idea behind RotationFeatures, taking classification as an example, is that often, when considering two or more dimensions, an ideal boundary between the target classes is at an oblique angle, and not parallel to any of the axes. Thus, models such as decision trees must use sub-optimal axis-parallel boundaries, which can lead to lower accuracy and larger trees, thus decreasing their interpretability. 

When used with decision trees, RotationFeatures effectively provides a simple, efficient, sklearn-compatible method to generate oblique trees, a now somewhat-forgotten technique to induce more powerful decision trees. The decision trees created may then be used in ensembles through boosting or bagging. As the emphasis here is on XAI, and ensembling, though typically enhancing accuracy, can be counter to interpretability, examples of ensembling are not provided here, though may be in subsequent versions. 

In order to support interpretability and to help ensure transformations are tractable given potentially large numbers of input attributes, rotations are limited to two dimensions. Research conducted by the author suggests this is typically quite sufficient, and where not, nevertheless is generally an improvement over the original features. For now it will suffice to say that two-dimensional spaces are easily comprehended, and that the decision boundaries can be very effectively conveyed visually in these spaces. 

In many cases, RotationFeatures can support higher accuracy in prediction models than using those models with only the original features. Additionally, in many cases the accuracy is similar, but when used with interpretable models such as decision trees, there is an improvement in interpretability, as more concise models may be produced. 

RotationFeatures may generate a large number of features, and therefore following feature generation with feature selection may be important with some datasets and prediction models. The examples here use sklearn’s decision tree, which provides embedded feature selection and may, therefore, skip the explicit feature selection step. 

### Limitation to 2d Spaces
The tool is limited to 2d spaces as its primary purpose is to enhance interpretability and this is compromised when working in higher-dimensional spaces, which can not be visualized easily. 

The limitation to 2d spaces also keeps the tool more scalable, as it must consider only each pair of numeric features. 

Further, we believe, in many cases, capturing interactions based on pairs of features is sufficent to capture the majority of feature insteractions in the dataset. This is not always the case, and more involved feature interactions must be captured to significantly improve accuracy in some cases, but we believe it is a common scenario for 2-way interactions to be sufficient. 

## Example Notebooks

Three example notebooks are provided. 

**Simple Test Rotation-Based Feature Generation** 
This notebook provides examples using RotationFeatures with three toy datasets from sklearn. This is primarily to provide simple examples of using the tool, but also gives examples of the accuracy and interpretability gained, measured by the macro f1 score and tree size respectively. It compares quite favourably compared to using the same model with only the original features. 

**Visualization Examples**. This notebook provides examples of visualizing decision trees generated using RotationFeatures. This uses the GraphTwoDimTree class also included in this package. 

**Accuracy Test Rotation Features** This notebook provides code to recreate additional tests, performed on numerous publicly-available datasets, showing relatively consistent results. As with PolynomialFeatures, and any other feature generation tool, this works for some datasets considerably better than for others, but it frequently improves upon the original features. This requires installing the DatasetsEvaluator tool. 

Note, we define an improvement as:
- Increase in accuracy; and/or
- Greater stability (smaller std dev in accuracy among the cross validation folds); and/or
- In the case of decision trees, smaller trees (allowing for greater interpretability. Similar metrics may be used for other interpretable models. )

This is such that improvements in stability and interpretability are only relevant if the accuracy increases or remains the same. 

## Examples

Given X and y, 

```python
rota_45= RotationFeatures(degree_increment=45, max_cols_created=20)
X_45 = pd.DataFrame(rota_45.fit_transform(X))
dt = tree.DecisionTreeClassifier(min_samples_split=50, max_depth=5, random_state=0)
scores = cross_validate(dt, X_45, y, cv=5, scoring='f1_macro')
test_scores = scores['test_score']
avg_test_score = test_scores.mean()
```

## RotationFeatures Methods

### fit()

fit() simply determines the number of features that will be generated. As the new features are based on rotations, they do not depend on any specific data that must be fit to. 

```
fit(X)
```

#### Parameters

**X**: matrix

#### Return Type

Returns self
## 

### transform()

Generates the new features based on the specifed degree_increment.

```
transform(X)
```

#### Parameters

**X**: matrix

##### Return Type

Returns a new pandas dataframe containing the same rows and columns as the passed matrix X, as well as the additional columns created. 
## 

### fit_transform()

Simply calls fit() and transform()

```
fit_transform(X)
```

#### Parameters

**X**: matrix

**y**: Unused

**fit_params**: Unused

#### Return Type

Returns a new pandas dataframe containing the same rows and columns as the passed matrix X, as well as the additional columns created. 
## 

### get_feature_names()

```
get_feature_names()
```

Returns the list of column names. This includes the original columns and the generated columns. The generated columns have names of the form: "R_" followed by a count. The generated columns have little meaning in themselves except as described as a rotation of two original features. 
## 

### get_feature_sources()

Returns the list of column sources. This has an element for each column. For the original columns, this is empty and for generated columns, this lists the pair of original columns from which it was generated. 

```
get_feature_sources()
```
##

### get_params()

Returns the degree_increment.

```
get_params(deep=True)
```

## 

### set_params()

Accepts degree_increment.
```
set_params(**params)
```
## 

## GraphTwoDimTree Methods

### graph_node()
Presents a series of plots describing a single node. This shows: 
1. A bar chart giving the count for each target class
2. A histogram for the distribution of each target class
3. Optionally repeat 2. on a log scale.
4. The 2d space of two generated features
5. The original 2d space of the two original features before rotation

```
graph_node(
  node_idx, 
  row=None, 
  show_log_scale=False, 
  show_combined_2d_space=False):
```

#### Return Type



### graph_tree()
Calls graph_node() for all nodes in the tree. This, then, presents the full tree in order. 

```
graph_tree(
  show_log_scale=False, 
  show_combined_2d_space=False)
```
#### Parameters

#### Return Type
## 



### graph_decision_path()
Similar to graph_node(), but presents only the nodes appearing on the decision path for the specified row. 
```
graph_decision_path(row=None, show_log_scale=False, show_combined_2d_space=False)
```

#### Parameters

#### Return Type
## 



## Summary of Accuracy & Model Sizes

The gain in accuracy can be sensitive to the number of different degrees of rotation used. 

Overall, both the toy and the real (publicly-available) datasets showed some improvement in at least one sense for at least one setting for the feature engineering. It often, though, required creating over 10,000 features (eg Breast Cancer rotated in 4 degree increments.)

Testing was done using DatasetsEvaluator, which allows for large-scale unbiased evaluation of tools on publicly-avaliable datasets, and this found a noticable improvement in accuracy and/or interpretability for numerous datasets. Overall accuracy was similar with and without RotationFeatures, demonstrating it is also often not useful. 

RotationFeatures is not intended, in itself, to make Decision Trees and other interpretable models competitive with other models in terms of accuracy, though may in many cases make them comparable. The intention is simply to increase the accuracy and interpretability of interpretable models. 
