RotationFeatures is a feature generation tool that constructs sets of pairs of new features based on rotations of existing pairs of features. The tool follows a similar API signature as sklearn's PolynomialFeatures.

RotationFeatures is designed specifically for interpretable models such as Decision Trees, Decision Tables, Rule Lists and Rule Sets, though may be used with any model type where feature generation may be useful. Though the features generated may have lower comprehensibility than the original features from which they were created, they may be readily visualized, which allows this tool to contribute to eXplainable AI (XAI). 

The idea behind RotationFeatures, taking classification as an example, is that often, when considering two or more dimensions, an ideal boundary between the target classes is at an oblique angle, and not parallel to any of the axes. Thus, models such as decision trees must use sub-optimal axis-parallel boundaries, which can lead to lower accuracy and larger trees, which decreases their interpretability. 

When used with decision trees, RotationFeatures effectively provides a simple, efficient, sklearn-compatible method to generate oblique trees, a now somewhat-forgotten technique to induce more powerful decision trees. The decision trees created may then be used in ensembles through boosting or bagging. As the emphasis here is on XAI, and ensembling, though typically enhancing accuracy, can be counter to interpretability, examples of ensembling are not provided here, though may be in subsequent versions. 

In order to support interpretability and to help ensure transformations are tractable given potentially large numbers of input attributes, rotations are limited to two dimensions. Research conducted by the author suggests this is typically quite sufficient, and where not, nevertheless generally an improvement over the original features. For now it will suffice to say that two-dimensional spaces are easily comprehended, and that the decision boundaries can be very effectively conveyed visually in these spaces. 

In many cases, RotationFeatures can support higher accuracy in prediction models than using those models with only the original features. Additionally, in many cases the accuracy is similar, but when used with interpretable models such as decision trees, there is an improvement in interpretability, as more concise models may be produced. 

RotationFeatures may generate a large number of features, and therefore following feature generation with feature selection may be important with some datasets and prediction models. The examples here use sklearn’s decision tree, which provides embedded feature selection and may, therefore, skip the explicit feature selection step. 

Three example notebooks are provided. The first provides examples using this with three toy datasets from sklearn. This is primarily to provide simple examples of using the tool, but also gives examples of the accuracy and interpretability gained, measured by the macro f1 score and tree size respectively. 

The second notebook provides examples of visualizing decision trees generated using RotationFeatures. This uses the GraphTwoDimTree tool. 

The third notebook provides code to recreate additional tests, performed on numerous publicly-available datasets, showing relatively consistent results. As with PolynomialFeatures, and any other feature generation tool, this works for some datasets considerably better than for others, but it frequently improves upon the original features.

Note, we define an improvement as:
- Increase in accuracy; and/or
- Greater stability (smaller std dev in accuracy among the cross validation folds); and/or
- In the case of decision trees, smaller trees (allowing for greater interpretability. Similar metrics may be used for other interpretable models. )

This is such that improvements in stability and interpretability are only relevant if the accuracy increases or remains the same. 

Overall, both the toy and the real (publicly-available) datasets showed some improvement in at least one sense for at least one setting for the feature engineering. It often, though, required creating over 10,000 features (eg Breast Cancer rotated in 4 degree increments.)
