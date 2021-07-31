import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sin, cos 
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from IPython.core.display import display, HTML

class RotationFeatures():
	"""
	Feature generation tool based on rotating pairs of numeric featuers verious numbers of degrees,
	thereby creating new 2d spaces that may be split by axis-parallel cuts, allowing more effective
	rules to be induced on the data. 
	"""

	def __init__(self, degree_increment=30, determine_numeric_features=True, max_cols_created=np.inf):
		"""
		Parameters
		----------
		degree_increment: float
			Features will be generated based on rotations starting at degree_increment and increasing by
			degree_increment up to 90 degrees. For example, if set to 20, features will be generated for 
			rotations of 20, 40, 60 and 80 degrees. 

		determine_numeric_features: bool
			Features can only be generated for pairs of numeric features. If this is set True, the code
			will determine which are numeric. Otherwise it is assumed all features are numeric.

		max_cols_created: int
			At most max_cols_created features will be created. If more would be based on the the number of
			numeric features and degree_increment, an exception will be thrown. This may be used to avoid
			cases where more features would be generated than would be warranted given the time available.
		"""

		self.degree_increment = degree_increment
		self.determine_numeric_features = determine_numeric_features
		self.max_cols_created = max_cols_created 


	def fit(self, X):
		'''
		fit() simply determines the number of features that will be generated. As the new features are based
		on rotations, they do not depend on any specific data that must be fit to. 

		Parameters
		----------
		X: matrix

		Returns
		-------
		Returns self. 
		'''

		# X = np.array(X)
		# self.X = X
		self.X = np.array(X)
		self.orig_X = pd.DataFrame(X) 
		self.extended_X = None
		self.n_input_features_ = self.X.shape[1]
		self.n_numeric_input_features_ = 0
		self.n_output_features_ = 0
		self.degrees_array = list(range(self.degree_increment, 90, self.degree_increment))
		self.is_numeric_arr = []
		self.feature_names_ = []		
		
		self.scaler_ = MinMaxScaler()
		self.scaled_X_df = pd.DataFrame(self.scaler_.fit_transform(extended_X), columns=extended_X.columns)     
		
		# Determine which features may be considered numeric
		if self.determine_numeric_features:
			self.is_numeric_arr = [1 if is_numeric_dtype(self.orig_X[self.orig_X.columns[c]]) and (self.orig_X[self.orig_X.columns[c]].nunique()>2) else 0 for c in range(len(self.orig_X.columns))]
			self.n_numeric_input_features_ = self.is_numeric_arr.count(1)
		else:
			self.n_numeric_input_features_ = self.n_input_features_

		# Determine the number of features that will be created.
		# We look at each pair of numeric features (i.e., n(n-1)/2 pairs), for each creating 2 new features for each rotation.
		self.n_output_features_ = self.n_numeric_input_features_ * (self.n_numeric_input_features_-1) * len(self.degrees_array)
		
		return self

	def transform(self, X):
		'''
		
		Parameters
		----------
		X: matrix

		Returns
		-------
		Returns a new pandas dataframe containing the same rows and columns as the passed matrix X, as well 
		as the additional columns created. 

		'''
		self.orig_X = pd.DataFrame(X)
		extended_X = pd.DataFrame(X).copy()
		assert len(extended_X.columns) == self.n_input_features_
		self.feature_sources_ = [()]*self.n_input_features_

		# Determine if the number of features generated would be too great
		if self.n_output_features_ > self.max_cols_created:
			raise ValueError (
					"The number of columns passed would result in greater than "
					"the maximum specified number of output columns.")                     
			
		new_feat_idx = 0
		for c1_idx in range(len(self.orig_X.columns)-1):
			if (self.is_numeric_arr[c1_idx] == 0):
				continue
			for c2_idx in range(c1_idx+1, len(self.orig_X.columns)):
				if (self.is_numeric_arr[c2_idx] == 0):
					continue
				for d in self.degrees_array:
					rotated_df_old = self.rotate_data(self.scaled_X_df, c1_idx, c2_idx, d)
					#print("one row method. # rows: ", len(rotated_df))
					#display(rotated_df.head())
					rotated_df = self.rotate_data2(self.scaled_X_df, c1_idx, c2_idx, d)
					#print("dot method. # rows: ", len(rotated_df_dot))
					#display(rotated_df_dot.head())

					new_col_name = "R_" + str(new_feat_idx)
					new_feat_idx += 1
					extended_X[new_col_name] = rotated_df[0].values
					self.feature_sources_.append((c1_idx, c2_idx, d, 0))

					new_col_name = "R_" + str(new_feat_idx)
					new_feat_idx += 1
					extended_X[new_col_name] = rotated_df[1].values
					self.feature_sources_.append((c1_idx, c2_idx, d, 1))
		self.feature_names_ = list(extended_X.columns)
		extended_X = extended_X.fillna(0.0)
		extended_X = extended_X.replace([np.inf, -np.inf], 0.0)                
		return extended_X    

	def fit_transform(self, X, y=None, **fit_params):
		'''
		Calls fit() and transform()

		Parameters
		----------
		X: matrix

		y: Unused

		fit_params: Unused

		Returns
		-------
		Returns a new pandas dataframe containing the same rows and columns as the passed matrix X, as well 
		as the additional columns created. 
		'''

		self.fit(X)
		return self.transform(X) 

	def get_feature_names(self):
		'''
		Returns the list of column names. This includes the original columns and the generated columns. The 
		generated columns have names of the form: "R_" followed by a count. The generated columns have little
		meaning in themselves except as described as a rotation of two original features. 
		'''

		return self.feature_names_ 

	def get_feature_sources(self):
		'''
		Returns the list of column sources. This has an element for each column. For the original columns, this is empty and
		for generated columns, this lists the pair of original columns from which it was generated. 
		'''

		return self.feature_sources_

	def get_params(self, deep=True):		
		return {"degree_increment": self.degree_increment}

	def set_params(self, **params):
		for key, value in params.items():
			setattr(self, key, value)
		return self

	def rotate_data(self, X, col1, col2, degrees):
		# todo: do this as a dot product, not one row at a time. that was working before.

		# # Create the rotation matrix
		# theta = np.radians(degrees)
		# r = np.array(( (np.cos(theta), -np.sin(theta)),
		# 			   (np.sin(theta),  np.cos(theta)) ))

		# Get the specified columns and rotate them
		col_names = [X.columns[col1], X.columns[col2]]
		orig_data = X[col_names]
		# rotated_data = r.dot(orig_data.T)    
		# rotated_data_df = pd.DataFrame(rotated_data).T		
		# return rotated_data_df    

		def rotate_point(x ,y, degrees):
			# todo: we need to specify a point to rotate around instead of the origin I think.
			theta = np.radians(degrees)
			xx = x * np.cos(theta) - y * np.sin(theta) 
			yy = x * np.sin(theta) + y * np.cos(theta)
			#print(x, ",", y, ",", xx, ",", yy)
			return xx,yy

		rotated_data_arr = []
		for row_idx in range(len(orig_data)):
			row = orig_data.iloc[row_idx]
			rotated_data_arr.append(rotate_point(row[0], row[1], degrees))

		rotated_data_df = pd.DataFrame(rotated_data_arr, index=X.index)
		#display(rotated_data_df)
		return rotated_data_df

	def rotate_data2(self, X, col1, col2, degrees):
		# todo: do this as a dot product, not one row at a time. that was working before.

		# # Create the rotation matrix
		theta = np.radians(degrees)
		r = np.array(( (np.cos(theta), -np.sin(theta)),
					   (np.sin(theta),  np.cos(theta)) ))

		# Get the specified columns and rotate them
		col_names = [X.columns[col1], X.columns[col2]]
		orig_data = X[col_names]
		rotated_data = r.dot(orig_data.T)    
		rotated_data_df = pd.DataFrame(rotated_data).T		
		rotated_data_df.index = X.index
		return rotated_data_df    


class GraphTwoDimTree():
	'''
	This generates a series of plots describing an sklearn decision tree generated with either the original featues
	or the features generated using RotationFeatures. 
	'''
	def __init__(self, tree, X_orig, X_extended, y, rota):
		"""
		Parameters
		----------
		tree: sklearn decision tree or other tree supporting the <list methods> # todo: fill in 

		X_orig: pandas dataframe

		X_extended: pandas dataframe

		y: pandas series

		rota: RotationFeatures object used to create the generated features
		"""

		self.tree = tree 
		self.X_orig = X_orig 
		self.X_extended = X_extended
		self.y = y 
		self.rota = rota

		self.feature_sources = self.rota.get_feature_sources()
		if hasattr(self.tree, "feature_sources"):
			self.feature_sources = self.tree.feature_sources
		self.scaler = self.rota.scaler_
		self.class_arr = list(tree.classes_) 
		self.node_indexes = self.get_node_indexes()
		self.tableau_palette_list=["tab:blue", "tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]

	def add_caption(self, ax, caption_text):
		ax.text(.1,-.25, caption_text, fontsize=12, transform=ax.transAxes, linespacing=1.5)								

	# todo: add code to show a row
	# todo: respect the show_log_scale option
	def graph_tree(self, show_log_scale=False, show_combined_2d_space=False): 
		'''
		Graphs all nodes of the tree. Each node is represented by a row of plots, each describing the node in a different manner.

		Parameters
		----------
		Parameters passed through to graph_node()

		Returns
		-------
		'''

		# Graph each node one at a time			
		for node_idx in range(len(self.tree.tree_.feature)):
			self.graph_node(node_idx, row=None, show_log_scale=show_log_scale, show_combined_2d_space=show_combined_2d_space)
		
	def graph_decision_path(self, row=None, show_log_scale=False, show_combined_2d_space=False):
		'''

		Parameters
		----------

		Returns
		-------
		'''
		assert not row is None
		decision_path = self.tree.decision_path([row])
		print(f"Decision Path: {decision_path.indices}")
		for node_idx in decision_path.indices:
			self.graph_node(node_idx, row=row, show_log_scale=show_log_scale, show_combined_2d_space=show_combined_2d_space)

	def graph_incorrect_rows(self, X, y, y_pred, max_rows_shown, show_log_scale=False, show_combined_2d_space=False):
		'''

		Parameters
		----------

		Returns
		-------
		'''
		assert len(X) == len(y) and len(X) == len(y_pred)

		wrong_mask = [x!=y for x,y in zip(y, y_pred)]
		row_num_list = [i for i, value in enumerate(wrong_mask) if value == True][:max_rows_shown]
		print(f"Number of rows: {len(X)}. Number of incorrect: {len(row_num_list)}. Percent incorrect: {round(len(row_num_list)*100.0/len(X))}")
		for i in row_num_list:
			idx = X.iloc[i:i+1].index[0]
			print("\n\n****************************************************************")
			print(f"Displaying decision path for row {idx}. Predicted: {y_pred[i]}. Actual: {y.iloc[i]}")
			print("****************************************************************")
			self.graph_decision_path(row=X.loc[idx], show_log_scale=show_log_scale, show_combined_2d_space=show_combined_2d_space)


	def get_node_indexes(self):
		"""
		Determine the rows at each node
		"""
		node_indexes = [[]]*len(self.tree.tree_.feature)
		node_indexes[0] = self.X_extended.index
		for node_idx in range(len(self.tree.tree_.feature)):
			X_local = self.X_extended.loc[node_indexes[node_idx]]
			feature_idx = self.tree.tree_.feature[node_idx]
			feature_name = self.X_extended.columns[feature_idx]
			if feature_idx == -2:
				continue
			threshold = self.tree.tree_.threshold[node_idx]
			left_child_idx = self.tree.tree_.children_left[node_idx]
			right_child_idx = self.tree.tree_.children_right[node_idx]
			attribute_arr = np.where(X_local[feature_name]<=threshold, 0, 1)            
			node_indexes[left_child_idx] = X_local.iloc[np.where(attribute_arr<=0)[0]].index
			node_indexes[right_child_idx] = X_local.iloc[np.where(attribute_arr>0)[0]].index
		return node_indexes	


	def graph_bar_chart_classes(self, ax, is_leaf, class_counts):
		b = ax.bar(self.class_arr, class_counts, color=self.tableau_palette_list)
		title = "Counts of target classes"
		ax.set_title(title)
		ylim = ax.get_ylim()
		ax.set_ylim(ylim[0], ylim[1]*1.1) 

		ax.minorticks_off()
		ax.tick_params(axis='x', labelrotation=75, direction='in', length=2)

		for idx, rect in enumerate(b):
			height = rect.get_height()
			ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
				class_counts[idx],
				ha='center', 
				va='bottom', 
				rotation=0)

	def graph_histogram(self, ax, node_idx, local_df, local_y, class_counts, feature_idx, threshold, num_original_cols, row=None, log_scale=False, show_caption=True):
		n_bins = 20
		
		for class_idx, class_name in enumerate(self.class_arr):  
			match_ids = (local_y==class_name).values.reshape(1,-1)[0]
			ids_for_class = local_y.loc[match_ids].index
			x1 = local_df.loc[ids_for_class][local_df.columns[feature_idx]]
			ax.hist(x1, n_bins, alpha=0.4, color=self.tableau_palette_list[class_idx], label=class_name)
			if log_scale:
				ax.set_yscale('log')
				ax.set_xscale('log')
		ax.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
		if row is not None:
			ax.axvline(row[feature_idx], color='r', linestyle='solid', linewidth=1)
		if feature_idx < num_original_cols:
			feature_name = self.X_extended.columns[feature_idx]
		else:
			feature_name = "Engineered Feature " + str(self.X_extended.columns[feature_idx])
		title = "Distribution of values in column by class\n" + str(feature_name)
		if log_scale:
			title = "(Log Scale)"
		ax.set_title(title)
		ax.minorticks_off()
		ax.tick_params(axis='x', labelrotation=75, direction='in', length=2)
		ax.legend()
		if show_caption: 
			self.add_caption(ax, "Split into nodes: " + 
							str(self.tree.tree_.children_left[node_idx]) + 
							" and " + 
							str(self.tree.tree_.children_right[node_idx]))
		if log_scale:
			for axis in [ax.xaxis, ax.yaxis]: 
				formatter = ScalarFormatter()
				formatter.set_scientific(False)
				axis.set_major_formatter(formatter)  
				axis.set_minor_formatter(formatter) 
			ax.xaxis.set_major_locator(MaxNLocator(4)) 
			ax.yaxis.set_major_locator(MaxNLocator(5)) 
			ax.minorticks_off()
			ax.tick_params(axis='x', labelrotation=75) 
			
	def get_alpha(self, num_points):
		return min(100/num_points, 1.0)
			
	def graph_scatter_engineered_features(self, ax, feature_idx, local_df, local_y, threshold, row):
		orig_feat1, orig_feat2, degrees, side = self.feature_sources[feature_idx]
		# The column names can occasionally be numbers and not strings, so cast to strings.
		orig_feat1_name = str(self.X_extended.columns[orig_feat1]) 
		orig_feat2_name = str(self.X_extended.columns[orig_feat2])
		
		ax.minorticks_off()
		ax.tick_params(axis='x', labelrotation=75, direction='in', length=2)
		
		# Identify the other feature generated based on the same source columns and rotation and graph these as well
		# The side specifies if this was the 1st or 2nd of the two features generated by rotating the original columns.
		ax.minorticks_off()
		ax.tick_params(axis='x', labelrotation=75, direction='in', length=2)

		if side == 0: 
			other_feature = feature_idx+1
		else:
			other_feature = feature_idx-1
		eng_feat1_name = self.X_extended.columns[feature_idx]
		eng_feat2_name = self.X_extended.columns[other_feature]
		for class_idx, class_name in enumerate(self.class_arr): 
			match_ids = (local_y==class_name).values.reshape(1,-1)[0]
			idx_arr = local_y.loc[match_ids].index
			X_curr_class = local_df.loc[idx_arr]  
			#print("engineerd features. class: ", class_idx, ", ", class_name, ", len: ", len(X_curr_class))
			if (side == 0):                      
				ax.scatter(   X_curr_class[X_curr_class.columns[feature_idx]], 
							  X_curr_class[X_curr_class.columns[other_feature]], 
							  alpha=self.get_alpha(len(local_df)), 
							  c=self.tableau_palette_list[class_idx], 
							  label=class_name)
				ax.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
			else:
				ax.scatter(  X_curr_class[X_curr_class.columns[other_feature]], 
							 X_curr_class[X_curr_class.columns[feature_idx]], 
							 alpha=self.get_alpha(len(local_df)), 
							 c=self.tableau_palette_list[class_idx], 
							 label=class_name)
				ax.axhline(threshold, color='k', linestyle='dashed', linewidth=1)                            

		if side == 0:
			ax.set_title("Engineered Features: " + eng_feat1_name + ", " + eng_feat2_name + "\n(" + orig_feat1_name + ", " + orig_feat2_name + "\nRotated counter-clockwise " + str(degrees) + " Degrees)")
			if (not row is None): 
				ax.plot(row[feature_idx], row[other_feature], marker="*", markersize=15, markeredgecolor="red", markerfacecolor="red")
		else:
			ax.set_title("Engineered Features: " + eng_feat2_name + ", " + eng_feat1_name + "\n(" + orig_feat1_name + ", " + orig_feat2_name + "\nRotated counter-clockwise " + str(degrees) + " Degrees)")			
			if (not row is None): 
				ax.plot(row[other_feature], row[feature_idx], marker="*", markersize=15, markeredgecolor="red", markerfacecolor="red")
		ax.legend()   
		
	def graph_scatter_original_features(self, ax, feature_idx, local_df, local_y, threshold, row, show_scaled=False):
		ax.minorticks_off()
		ax.tick_params(axis='x', labelrotation=75, direction='in', length=2)
		
		def rotate_point(x ,y, degrees):
			theta = np.radians(degrees)
			xx = x * np.cos(theta) - y * np.sin(theta) 
			yy = x * np.sin(theta) + y * np.cos(theta)
			return xx,yy

		if show_scaled:
			orig_cols = self.X_orig.columns
			scaled_X_local = pd.DataFrame(self.scaler.transform(local_df[orig_cols]), columns=orig_cols, index=local_df.index)                        

		orig_feat1, orig_feat2, degrees, side = self.feature_sources[feature_idx]
		orig_feat1_name = str(self.X_extended.columns[orig_feat1]) 
		orig_feat2_name = str(self.X_extended.columns[orig_feat2])
		if side == 0: 
			other_feature = feature_idx+1
		else:
			other_feature = feature_idx-1
		eng_feat1_name = self.X_extended.columns[feature_idx]
		eng_feat2_name = self.X_extended.columns[other_feature]
		for class_idx, class_name in enumerate(self.class_arr):  
			match_ids = (local_y==class_name).values.reshape(1,-1)[0]
			idx_arr = local_y.loc[match_ids].index
			if show_scaled:
				X_curr_class = scaled_X_local.loc[idx_arr]                        
			else:
				X_curr_class = local_df.loc[idx_arr]                        
			ax.scatter( X_curr_class[X_curr_class.columns[orig_feat1]],  \
						X_curr_class[X_curr_class.columns[orig_feat2]], \
						alpha=self.get_alpha(len(local_df)), \
						c=self.tableau_palette_list[class_idx], \
						label=class_name)

			# Draw a line at an oblique angle in the original 2d space to represent the threshold in that space. 
			#print("column name: ", local_df.columns[other_feature])
			#print("data: ", local_df[local_df.columns[other_feature]])                    
			if (side == 0): 
				# Used the 1st engineered feature. When drawing the 2 engineered featuers, we drew a vertical 
				# threshold line. The feature used is on the x-axis and the other feature is on the y-axis.
				rotated_x1, rotated_y1 = rotate_point(threshold, local_df[local_df.columns[other_feature]].min(),  -1*degrees)
				rotated_x2, rotated_y2 = rotate_point(threshold, local_df[local_df.columns[other_feature]].max(),  -1*degrees)
				#print("before points case 0: ", threshold, local_df[local_df.columns[other_feature]].min(), threshold, local_df[local_df.columns[other_feature]].max())
			else: # todo: use parameters instead to reduce code dupliction
				# Used the 2nd engineered feature. When drawing the 2 engineered featuers, we drew a horizontal 
				# threshold line. The feature used is on the y-axis and the other feature is on the x-axis.
				#rotated_x1, rotated_y1 = rotate_point(local_df[local_df.columns[other_feature]].min(), threshold,  -1*degrees)
				#rotated_x2, rotated_y2 = rotate_point(local_df[local_df.columns[other_feature]].max(), threshold,  -1*degrees)                        
				# todo: used something more intelligent that + and - 10
				min_val = local_df[local_df.columns[other_feature]].min()
				max_val = local_df[local_df.columns[other_feature]].max()
				range = max_val-min_val

				rotated_x1, rotated_y1 = rotate_point(min_val - (0.5*range), threshold,  -1*degrees)
				rotated_x2, rotated_y2 = rotate_point(max_val + (0.5*range), threshold,  -1*degrees)                        
				#print("before points case 1: ", local_df[local_df.columns[other_feature]].min(), threshold, local_df[local_df.columns[other_feature]].max(), threshold)
			#print("after reverse rotation points: ", rotated_x1, rotated_y1, rotated_x2, rotated_y2)
			if show_scaled:
				ax.plot([rotated_x1, rotated_x2], [rotated_y1, rotated_y2], c="k", linestyle='dashed', linewidth=1) 
				rotated_x1, rotated_y1 = rotate_point(min_val - (0.5*range), threshold,  -1*degrees)
				rotated_x2, rotated_y2 = rotate_point(max_val + (0.5*range), threshold,  -1*degrees)
				
				# Liz's attempt at contourf - not currently working
				X = np.linspace(rotated_x1,rotated_x2)
				Y = np.linspace(rotated_y1,rotated_y2)
				X_grid, Y_grid = np.meshgrid(X,Y)
				clf = KNeighborsClassifier(n_neighbors=8)
				X = X.reshape(-1,1)
				Y = Y.reshape(-1,1)
				clf.fit(X_grid, Y_grid)
				x_min = ax.get_xlim()[0]
				x_max = ax.get_xlim()[1]
				x_step = (x_max-x_min)/100
				y_min = ax.get_ylim()[0]
				y_max = ax.get_ylim()[1]
				y_step = (y_max-y_min)/100   
				x_mesh, y_mesh = np.meshgrid((np.arange(x_min,x_max,x_step)), (np.arange(y_min,y_max,y_step)))
				df = pd.DataFrame({"0": x_mesh.reshape(-1), "1": y_mesh.reshape(-1)})
				mesh_pred = clf.predict(df)
				Z_grid = mesh_pred.reshape(X_grid.shape)
				plt.contourf(X_grid, Y_grid, Z_grid)
#				plt.show()

			# We're now, after inverse rotation, in the space of the 2 generated columns scaled. We next get the
			# inverse scaling. Take an arbitary pair of rows and replace the 2 columns we're interested in inverse scaling.
			scaled_data = self.X_orig.iloc[:2] 
			scaled_data.iloc[0][orig_feat1_name] = rotated_x1
			scaled_data.iloc[0][orig_feat2_name] = rotated_y1
			scaled_data.iloc[1][orig_feat1_name] = rotated_x2
			scaled_data.iloc[1][orig_feat2_name] = rotated_y2
			#print("scaled_data: ")
			#display(scaled_data)
			rev = self.scaler.inverse_transform(scaled_data)
			#print("rev: ", rev)
			scaled_x1 = rev[0][orig_feat1]
			scaled_y1 = rev[0][orig_feat2]
			scaled_x2 = rev[1][orig_feat1]
			scaled_y2 = rev[1][orig_feat2]                    
			#print("after reverse scaling points: ", scaled_x1, scaled_y1, scaled_x2, scaled_y2)

			# todo: extend the line so it covers the full extent of the axes
			if show_scaled==False:
				ax.plot([scaled_x1, scaled_x2], [scaled_y1, scaled_y2], c="k", linestyle='dashed', linewidth=1) 
			if (not row is None): 
				ax.plot(row[orig_feat1], row[orig_feat2], marker="*", markersize=15, markeredgecolor="red", markerfacecolor="red")
			ax.set_title("Source Features:\n" + orig_feat1_name + ", " + orig_feat2_name)
			if show_scaled==False:
				ax.legend()			


	def graph_node(self, node_idx, row=None, show_log_scale=False, show_combined_2d_space=False):
		'''

		Parameters
		----------
		node_idx: int

		row: array
			The full set of values for a row from the original feature space

		show_log_scale: bool
			If True, the histogram will also be rendered on a log scale, which can in some cases make the separation
			more clear.

		Returns
		-------
		None
		'''
		assert len(self.feature_sources) == len(self.X_extended.columns), "Length of feature_sources is " + str(len(self.feature_sources)) + " but number of columns in X_extended is " + str(len(self.X_extended.columns))
	
		local_df = self.X_extended.loc[self.node_indexes[node_idx]]
		local_y = pd.DataFrame(self.y).loc[self.node_indexes[node_idx]]
		feature_idx = self.tree.tree_.feature[node_idx]
		feature_name = self.X_extended.columns[feature_idx]
		threshold = self.tree.tree_.threshold[node_idx]
		num_original_cols = len(self.X_orig.columns)
		class_counts = [local_y.values.flatten().tolist().count(x) for x in self.class_arr]
		
		# Determine the number of plots, the width of the figure, and the indexes of the axes
		ncols = 1
		if feature_idx >= 0:
			ncols = 2
			if show_log_scale:
				ncols += 1
			if feature_idx >= num_original_cols:
				ncols += 2
				if (show_combined_2d_space) :
					ncols += 1

			scatter_eng_idx = 2
			if show_log_scale:
				scatter_eng_idx += 1
			scatter_orig_idx = scatter_eng_idx + 1
			scatter_combined_idx = scatter_orig_idx +1

		fig_width = ncols * 3.5

		# Render internal nodes
		if feature_idx >= 0:
			fig_height = 3.0
			fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(fig_width,fig_height))
			plt.tight_layout()
			
			# In the first plot, show a bar chart giving the count for each target class
			self.graph_bar_chart_classes(ax[0], is_leaf=False, class_counts=class_counts)

			# In the second plot, show a histogram for the distribution of each target class
			self.graph_histogram(ax[1], node_idx, local_df, local_y, class_counts, feature_idx, threshold, num_original_cols, row, log_scale=False, show_caption=True)
			
			# In the third plot, show a similar histogram, but on a log scale. For some datasets this is clearer, and for some
			# it is less clear. 
			if show_log_scale:
				self.graph_histogram(ax[2], node_idx, local_df, local_y, class_counts, feature_idx, threshold, num_original_cols, row, log_scale=True, show_caption=False)

			# In the 4th and 5th plots, if applicable (if the feature is based on the rotation of 2 other features),
			# show a scatter plot of the two original and the two generated (rotated) features.
			if feature_idx >= num_original_cols: 
				rotated_x1, rotated_y1, rotated_x2, rotated_y2 = [], [], [],[] # temp!! remove

				# The 4th plot shows the 2d space of the 2 generated features
				self.graph_scatter_engineered_features(ax[scatter_eng_idx], feature_idx, local_df, local_y, threshold, row)
				
				# The 5th plot shows the 2d space of the 2 original features before they were rotated
				self.graph_scatter_original_features(ax[scatter_orig_idx], feature_idx, local_df, local_y, threshold, row)

				if show_combined_2d_space:
					self.graph_scatter_engineered_features(ax[scatter_combined_idx], feature_idx, local_df, local_y, threshold, row)	
					self.graph_scatter_original_features(ax[scatter_combined_idx], feature_idx, local_df, local_y, threshold, row, show_scaled=True)
			plt.subplots_adjust(top=0.7)

		# Render leaf nodes
		else:
			fig_height = 2.6
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(fig_width,fig_height))
			self.graph_bar_chart_classes(ax, is_leaf=True, class_counts=class_counts)	
			plt.subplots_adjust(top=0.8)

		title = f"Node: {str(node_idx)}"
		if feature_idx >=0: 
			title += f" -- Split on {str(feature_name)}"
		else:
			title += " (Leaf Node)"
		fig.suptitle(title)
		plt.show()
