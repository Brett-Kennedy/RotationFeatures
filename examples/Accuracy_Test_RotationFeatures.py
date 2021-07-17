import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.pipeline import Pipeline
from warnings import filterwarnings
import matplotlib.pyplot as plt

# todo: explain how to pip install DatasetsEvaluator, point to github page

# todo: put back when done -- update datasets evaluator on pypi and retest -- i added the drop=True
#from DatasetsEvaluator import DatasetsEvaluator as de

# Todo: remove once have pip install
import sys  
sys.path.insert(0, 'C:\python_projects\RotationFeatures_project\RotationFeatures')
from RotationFeatures import RotationFeatures # todo: fix once have pip install

sys.path.insert(0, 'C:\python_projects\DatasetsEvaluator_project\DatasetsEvaluator')
import DatasetsEvaluator as de

filterwarnings('ignore')
np.random.seed(0)


# These specify how many datasets are used in the tests below. Ideally about 50 to 100 datasets would be used,
# but these may be set lower. Set to 0 to skip tests. 
NUM_DATASETS_CLASSIFICATION_DEFAULT = 5
NUM_DATASETS_REGRESSION_DEFAULT = 100
NUM_DATASETS_CLASSIFICATION_GRID_SEARCH = 10
NUM_DATASETS_REGRESSION_GRID_SEARCH = 50


def print_header(test_name):
	stars = "***************************************"
	print("\n\n")
	print(stars)
	print(test_name)
	print(stars)


def summarize_results(summary_df, accuracy_metric, saved_file_name, results_folder, show_std_dev=False):
	if (len(summary_df)==0):
		return
	p = pd.DataFrame(summary_df.groupby(['Model', 'Feature Engineering Description'])[accuracy_metric].mean())
	if (show_std_dev):
		p['Avg. Std dev between folds'] = summary_df.groupby(['Model', 'Feature Engineering Description'])['Std dev between folds'].mean()
	p['Avg. Train-Test Gap'] = summary_df.groupby(['Model', 'Feature Engineering Description'])['Train-Test Gap'].mean()
	p['Avg. Fit Time'] = summary_df.groupby(['Model', 'Feature Engineering Description'])['Fit Time'].mean()
	p['Avg. Complexity'] = summary_df.groupby(['Model', 'Feature Engineering Description'])['Model Complexity'].mean()
	results_summary_filename = results_folder + "\\" + saved_file_name + "_summarized" + ".csv"
	p.to_csv(results_summary_filename, index=True)


def plot_results(summary_df, accuracy_metric, saved_file_name, results_folder):
	if (len(summary_df)==0):
		return
	
	# Collect the set of all combinations of model type and feature engineering. In this example, there
	# should just be the two.
	combinations_df = summary_df.groupby(['Model', 'Feature Engineering Description']).size().reset_index()

	summary_df = summary_df.dropna(subset=[accuracy_metric])

	# Draw a single plot, with a line for each feature engineering description. Along the x-axis we have each
	# dataset ordered by lowest to highest score when using the original features. 
	fig_width = min(len(summary_df)/4, 20)
	fig_width = max(fig_width, 5)
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width,5))
	for row_idx in range(len(combinations_df)):
		m = combinations_df.iloc[row_idx]['Model']
		h = combinations_df.iloc[row_idx]['Feature Engineering Description']

		# Get the subset of summary_df for the current feature engineering method. 
		if (row_idx==0):
			subset_df_1 = summary_df[(summary_df['Model']==m) & (summary_df['Feature Engineering Description']==h)].sort_values(by=accuracy_metric).reset_index()
			x_coords = subset_df_1.index        
			y_coords = subset_df_1[accuracy_metric]
		else:
			subset_df_2 = summary_df[(summary_df['Model']==m) & (summary_df['Feature Engineering Description']==h)]
			y_coords = []
			for i in range(len(subset_df_1)):
				ds = subset_df_1.iloc[i]['Dataset']
				y_coords.append(subset_df_2[subset_df_2['Dataset']==ds][accuracy_metric])
		ax.plot(x_coords, y_coords, label=m + "(" + h + ")")
	t = subset_df_1['Dataset']
	plt.xticks(range(len(t)), t, size='small', rotation=55)            
	plt.legend()
	plt.title(accuracy_metric + " by dataset")
	results_plot_filename = results_folder + "\\" + saved_file_name + "_plot" + ".png"
	fig.savefig(results_plot_filename, bbox_inches='tight', dpi=150)


def test_classification_default_parameters(datasets_tester, partial_result_folder, results_folder):
	print_header("Classification with default parameters")

	pipe1 = Pipeline([('dt', tree.DecisionTreeClassifier(random_state=0))])
	pipe2 = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeClassifier(random_state=0))])

	# This uses some non-default parameters. 
	summary_df, saved_file_name = datasets_tester.run_tests(
										   estimators_arr = [
											("DT", "Original Features", "Default", pipe1),
											("DT", "Rotation-based Features", "Default", pipe2)],
										   num_cv_folds=3,
										   show_warnings=False,
										   partial_result_folder=partial_result_folder,                                        
										   results_folder=results_folder,
										   run_parallel=True) 

	summarize_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)
	plot_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)


def test_classification_default_parameters_max_five(datasets_tester, partial_result_folder, results_folder):
	print_header("Classification with max depth of 5")

	pipe1 = Pipeline([('dt', tree.DecisionTreeClassifier(max_depth=5, random_state=0))])
	pipe2 = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeClassifier(max_depth=5, random_state=0))])

	# This provides an example using some non-default parameters. 
	summary_df, saved_file_name = datasets_tester.run_tests(
									estimators_arr = [
										("DT", "Original Features", "Default", pipe1),
										("DT", "Rotation-based Features", "Default", pipe2)],
									num_cv_folds=3,
									show_warnings=False,
									results_folder=results_folder,
									partial_result_folder=partial_result_folder,
									run_parallel=True) 

	summarize_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)
	plot_results(summary_df, 'Avg f1_macro', saved_file_name, results_folder)


def test_classification_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder):
	# As this takes much longer than testing with the default parameters, we test with fewer datasets. Note though,
	# run_tests_grid_search() uses CV to evaluate the grid search for the best hyperparameters, it does a train-test 
	# split on the data for evaluation, so evaluates the predictions quickly, though with more variability than if
	# using CV to evaluate as well. 

	# todo: display the best hyperparameters too

	print_header("Classification with grid search for best parameters")
	datasets_tester = de.DatasetsTester()

	matching_datasets = datasets_tester.find_datasets(problem_type = "classification", 
	                                                  min_num_numeric_features=2,
	                                                  max_num_numeric_features=10) # Set lower to be faster

	datasets_tester.collect_data(max_num_datasets_used=NUM_DATASETS_CLASSIFICATION_GRID_SEARCH, 
	                             exclude_list=exclude_list,
	                             preview_data=False, 
	                             save_local_cache=True,
	                             check_local_cache=True,
	                             path_local_cache=cache_folder)

	# orig_parameters = {
	#      'dt__max_depth': (3,4,5,6,100)
	# }

	# rota_parameters = {
	#     'rota__degree_increment': (3,4,10,15,30), 
	# #     'dt__max_depth': (3,4,5,6)
	# }

	# todo: put back
	orig_parameters = {
	     'dt__max_depth': (3,4)
	}

	rota_parameters = {
	    'rota__degree_increment': (20,30), 
	}

	orig_pipe = Pipeline([('dt', tree.DecisionTreeClassifier())])
	rota_pipe = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeClassifier())])

	# This provides an example using some non-default parameters. 
	summary_df, saved_file_name = datasets_tester.run_tests_grid_search(
	        estimators_arr = [
	            ("DT", "Original Features", "", orig_pipe),
	            ("DT", "Rotation-based Features", "", rota_pipe)],
	        parameters_arr=[orig_parameters, rota_parameters],
	        num_cv_folds=3,
	        show_warnings=False,
			results_folder=results_folder,
			partial_result_folder=partial_result_folder,
			run_parallel=True) 

	summarize_results(summary_df, 'f1_macro', saved_file_name, results_folder)
	plot_results(summary_df, 'f1_macro', saved_file_name, results_folder)


def test_regression_default_parameters(datasets_tester, partial_result_folder, results_folder):            
	print_header("Regression with default parameters")

	pipe1 = Pipeline([('dt', tree.DecisionTreeRegressor(random_state=0))])
	pipe2 = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeRegressor(random_state=0))])

	summary_df, saved_file_name = datasets_tester.run_tests(
									estimators_arr = [
                                    	("DT", "Original Features", "Default", pipe1),
                                        ("DT", "Rotation-based Features", "Default", pipe2)],
									num_cv_folds=3,
                                    show_warnings=True,
                                    results_folder=results_folder,
                                    partial_result_folder=partial_result_folder,
                                    run_parallel=True) 
	
	summarize_results(summary_df, 'Avg NRMSE', saved_file_name, results_folder)
	plot_results(summary_df, 'Avg NRMSE', saved_file_name, results_folder)


def test_regression_default_parameters_max_five(datasets_tester, partial_result_folder, results_folder):            
	print_header("Regression with max depth of 5")

	pipe1 = Pipeline([('dt', tree.DecisionTreeRegressor(max_depth=5, random_state=0))])
	pipe2 = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeRegressor(max_depth=5, random_state=0))])

	summary_df, saved_file_name = datasets_tester.run_tests(
									estimators_arr = [
                                    	("DT", "Original Features", "Default", pipe1),
                                        ("DT", "Rotation-based Features", "Default", pipe2)],
									num_cv_folds=3,
                                    show_warnings=True,
                                    results_folder=results_folder,
                                    partial_result_folder=partial_result_folder,
                                    run_parallel=True) 
	
	summarize_results(summary_df, 'Avg NRMSE', saved_file_name, results_folder)
	plot_results(summary_df, 'Avg NRMSE', saved_file_name, results_folder)


def test_regression_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder):
	# As this takes much longer than testing with the default parameters, we test with fewer datasets. Note though,
	# run_tests_grid_search() uses CV to evaluate the grid search for the best hyperparameters, it does a train-test 
	# split on the data for evaluation, so evaluates the predictions quickly, though with more variability than if
	# using CV to evaluate as well. 

	# todo: display the best hyperparameters too

	print_header("Regression with grid search for best parameters")
	datasets_tester = de.DatasetsTester()

	matching_datasets = datasets_tester.find_datasets(problem_type = "regression",
	                                                  min_num_numeric_features=2,
	                                                  max_num_numeric_features=10)

	datasets_tester.collect_data(max_num_datasets_used=NUM_DATASETS_REGRESSION_GRID_SEARCH, 
								 exclude_list=exclude_list,
	                             preview_data=False, 
	                             save_local_cache=True,
	                             check_local_cache=True,
	                             path_local_cache=cache_folder)

	orig_parameters = {
	     'dt__max_depth': (3,4,5,6) # todo: these are different than for classification.
	}

	rota_parameters = {
	     'rota__degree_increment': (3,4,10,15,30),
	     'dt__max_depth': (3,4,5,6)
	}

	orig_pipe = Pipeline([('dt', tree.DecisionTreeRegressor())])
	rota_pipe = Pipeline([('rota', RotationFeatures()), ('dt', tree.DecisionTreeRegressor())])

	# This provides an example using some non-default parameters. 
	summary_df, saved_file_name = datasets_tester.run_tests_grid_search(
	        estimators_arr = [
	            ("DT", "Original Features", "", orig_pipe),
	            ("DT", "Rotation-based Features", "", rota_pipe)],
	        parameters_arr=[orig_parameters, rota_parameters],
	        num_cv_folds=3,
	        show_warnings=False,
	        partial_result_folder=partial_result_folder,
	        results_folder=results_folder,
	        run_parallel=True) 

	summarize_results(summary_df, 'NRMSE', saved_file_name, results_folder)
	plot_results(summary_df, 'NRMSE', saved_file_name, results_folder)


def main():
	cache_folder = "c:\\dataset_cache"
	partial_result_folder = "c:\\intermediate_results"
	results_folder = "c:\\results"

	# These are a bit slower, so excluded for some tests
	exclude_list = ["oil_spill", "fri_c4_1000_50", "fri_c3_1000_50", "fri_c1_1000_50", "fri_c2_1000_50", "waveform-5000", 
				"mfeat-zernikemfeat-zernike", "auml_eml_1_b"]

	# Collect & test with classification datasets
	datasets_tester = de.DatasetsTester()

	matching_datasets = datasets_tester.find_datasets( 
		problem_type = "classification",
		min_num_classes = 2,
		max_num_classes = 20,
		min_num_minority_class = 5,
		max_num_minority_class = np.inf,
		min_num_features = 0,
		max_num_features = np.inf,
		min_num_instances = 500,
		max_num_instances = 5_000,
		min_num_numeric_features = 2,
		max_num_numeric_features = 50,
		min_num_categorical_features=0,
		max_num_categorical_features=50)

	datasets_tester.collect_data(max_num_datasets_used=NUM_DATASETS_CLASSIFICATION_DEFAULT, 
								 preview_data=False, 
								 exclude_list=exclude_list,
								 save_local_cache=True,
								 check_local_cache=True,
								 path_local_cache=cache_folder)                             

	test_classification_default_parameters(datasets_tester, partial_result_folder, results_folder)            
	#test_classification_default_parameters_max_five(datasets_tester, partial_result_folder, results_folder)            
	#test_classification_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder)

	# Collect & test with regression datasets
	matching_datasets = datasets_tester.find_datasets( 
	    problem_type = "regression",
	    min_num_features = 0,
	    max_num_features = np.inf,
	    min_num_instances = 500,
	    max_num_instances = 5_000,
	    min_num_numeric_features = 2,
	    max_num_numeric_features = 50,
	    min_num_categorical_features=0,
	    max_num_categorical_features=50)

	# todo: define an exclude list for regression too
	# datasets_tester.collect_data(max_num_datasets_used=NUM_DATASETS_REGRESSION_DEFAULT, 
	# 							 exclude_list=exclude_list, 
 #                             	 preview_data=False, 
 #                             	 save_local_cache=True,
 #                             	 check_local_cache=True,
 #                             	 path_local_cache=cache_folder)

	#test_regression_default_parameters(datasets_tester, partial_result_folder, results_folder)            
	#test_regression_default_parameters_max_five(datasets_tester, partial_result_folder, results_folder)            
	#test_regression_grid_search(exclude_list, cache_folder, partial_result_folder, results_folder)


if __name__ == "__main__":
	main()                