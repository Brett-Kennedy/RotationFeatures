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

		X=np.array(X)
		self.X = X
		self.X_df = pd.DataFrame(X) 
		self.n_input_features_ = X.shape[1]
		self.n_numeric_input_features_ = 0
		self.n_output_features_ = 0
		self.degrees_array = list(range(self.degree_increment, 90, self.degree_increment))
		self.is_numeric_arr = []
		self.feature_names_ = []		
		
		# Determine which features may be considered numeric
		if self.determine_numeric_features:
			self.is_numeric_arr = [1 if is_numeric_dtype(self.X_df[self.X_df.columns[c]]) and (self.X_df[self.X_df.columns[c]].nunique()>2) else 0 for c in range(len(self.X_df.columns))]
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

		X_new = pd.DataFrame(X).copy()
		assert len(X_new.columns) == self.n_input_features_
		self.feature_sources_ = [()]*self.n_input_features_

		# Determine if the number of features generated would be too great
		if self.n_output_features_ > self.max_cols_created:
			raise ValueError (
					"The number of columns passed would result in greater than "
					"the maximum specified number of output columns.")
			
		self.scaler_ = MinMaxScaler()
		scaled_X_df = pd.DataFrame(self.scaler_.fit_transform(X), columns=X.columns)                        
			
		new_feat_idx = 0
		for c1_idx in range(len(self.X_df.columns)-1):
			if (self.is_numeric_arr[c1_idx] == 0):
				continue
			for c2_idx in range(c1_idx+1, len(self.X_df.columns)):
				if (self.is_numeric_arr[c2_idx] == 0):
					continue
				for d in self.degrees_array:
					rotated_df = self.__rotate_data(scaled_X_df, c1_idx, c2_idx, d)

					new_col_name = "R_" + str(new_feat_idx)
					new_feat_idx += 1
					X_new[new_col_name] = rotated_df[0].values
					self.feature_sources_.append((c1_idx, c2_idx, d, 0))

					new_col_name = "R_" + str(new_feat_idx)
					new_feat_idx += 1
					X_new[new_col_name] = rotated_df[1].values
					self.feature_sources_.append((c1_idx, c2_idx, d, 1))
		self.feature_names_ = list(X_new.columns)
		X_new = X_new.fillna(0.0)
		X_new = X_new.replace([np.inf, -np.inf], 0.0)                
		return X_new    

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

	def __rotate_data(self, X, col1, col2, degrees):
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
			xx = x * cos(theta) - y * sin(theta) # todo: for consistency use np instead of math
			yy = x * sin(theta) + y * cos(theta)
			#print(x, ",", y, ",", xx, ",", yy)
			return xx,yy

		rotated_data_arr = []
		for row_idx in range(len(orig_data)):
			row = orig_data.iloc[row_idx]
			rotated_data_arr.append(rotate_point(row[0], row[1], degrees))

		rotated_data_df = pd.DataFrame(rotated_data_arr)
		#display(rotated_data_df)
		return rotated_data_df

class GraphTwoDimTree():
	'''
	This generates a series of plots describing an sklearn decision tree generated with either the original featues
	or the features generated using RotationFeatures. 
	'''
	def __init__(self):
		pass

	# todo: add code to show a row
	# todo: respect the show_log_scale option
	def graph_tree(self, tree, X_orig, X_extended, y, class_arr, node_indexes, feature_sources, scaler, row=None, show_log_scale=False): 
		'''
		Graphs all nodes of the tree. Each node is represented by a row of plots, each describing the node in a different manner.

		Parameters
		----------
		tree

		X_orig

		X_extended

		y

		class_arr

		node_indexes

		feature_sources

		scaler

		row

		show_log_scale: bool

		Returns
		-------
		'''

		# Graph each node one at a time			
		for node_idx in range(len(tree.feature)):
			self.graph_node(node_idx, tree, X_orig, X_extended, y, class_arr, node_indexes, feature_sources, scaler, row, show_log_scale)
		
	def graph_decision_path(self):
		'''

		Parameters
		----------

		Returns
		-------
		'''

		pass #todo: fill in

	def graph_node(self, node_idx, tree, X_orig, X_extended, y, class_arr, node_indexes, feature_sources, scaler, row=None, show_log_scale=False):
		'''

		Parameters
		----------

		Returns
		-------
		'''

		assert len(feature_sources) == len(X_extended.columns), "Length of feature_sources is " + str(len(feature_sources)) + " but number of columns in X_extended is " + str(len(X_extended.columns))

		#print("node_idx: ", node_idx)
		
		def add_caption(ax, caption_text):
			ax.text(.1,-.2, caption_text, fontsize=12, transform=ax.transAxes, linespacing=1.5)

		def graph_bar_chart_classes(ax, is_leaf):
			b = ax.bar(class_arr, class_counts, color=tableau_palette_list)
			title = "Counts of target classes"
			if is_leaf:
				title = "(Leaf Node)" + title
			ax.set_title(title)
			ylim = ax.get_ylim()
			ax.set_ylim(ylim[0], ylim[1]*1.1) 
			for idx, rect in enumerate(b):
				height = rect.get_height()
				ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
					class_counts[idx],
					ha='center', 
					va='bottom', 
					rotation=0)
			
		def graph_histogram(ax, log_scale=False, show_caption=True):
			n_bins = 20
			
			for class_idx, class_name in enumerate(class_arr):  
				match_ids = (local_y==class_name).values.reshape(1,-1)[0]
				ids_for_class = local_y.loc[match_ids].index
				x1 = local_df.loc[ids_for_class][local_df.columns[feature_idx]]
				ax.hist(x1, n_bins, alpha=0.4, color=tableau_palette_list[class_idx], label=class_name)
				if log_scale:
					ax.set_yscale('log')
					ax.set_xscale('log')
			ax.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
			if feature_idx < num_original_cols:
				feature_name = X_extended.columns[feature_idx]
			else:
				feature_name = "Engineered Feature " + str(X_extended.columns[feature_idx])
			title = "Distribution of values in column by class\n" + str(feature_name)
			if log_scale:
				title += " (Log Scale)"
			ax.set_title(title)
			ax.legend()
			if show_caption: 
				add_caption(ax, "Split into nodes: " + 
								str(tree.children_left[node_idx]) + 
								" and " + 
								str(tree.children_right[node_idx]))
			if log_scale:
				for axis in [ax.xaxis, ax.yaxis]: # Todo: this leaves a messy set of ticks on the x-axis often
					formatter = ScalarFormatter()
					formatter.set_scientific(False)
					axis.set_major_formatter(formatter)  
					axis.set_minor_formatter(formatter) 
				ax.xaxis.set_major_locator(MaxNLocator(4)) 
				ax.yaxis.set_major_locator(MaxNLocator(5)) 
				ax.minorticks_off()
				ax.tick_params(axis='x', labelrotation=75) 
				
		def get_alpha(num_points):
			return min(100/num_points, 1.0)
				
		def graph_scatter_engineered_features(ax):
			orig_feat1, orig_feat2, degrees, side = feature_sources[feature_idx]
			# The column names can occasionally be numbers and not strings, so cast to strings.
			orig_feat1_name = str(X_extended.columns[orig_feat1]) 
			orig_feat2_name = str(X_extended.columns[orig_feat2])
			# Identify the other feature generated based on the same source columns and rotation and graph these as well
			# The side specifies if this was the 1st or 2nd of the two features generated by rotating the original columns.
			if side == 0: 
				other_feature = feature_idx+1
			else:
				other_feature = feature_idx-1
			eng_feat1_name = X_extended.columns[feature_idx]
			eng_feat2_name = X_extended.columns[other_feature]
			for class_idx, class_name in enumerate(class_arr): 
				match_ids = (local_y==class_name).values.reshape(1,-1)[0]
				idx_arr = local_y.loc[match_ids].index
				X_curr_class = local_df.loc[idx_arr]  
				#print("engineerd features. class: ", class_idx, ", ", class_name, ", len: ", len(X_curr_class))
				if (side == 0):                      
					ax.scatter(   X_curr_class[X_curr_class.columns[feature_idx]], 
								  X_curr_class[X_curr_class.columns[other_feature]], 
								  alpha=get_alpha(len(local_df)), 
								  c=tableau_palette_list[class_idx], 
								  label=class_name)
					ax.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
				else:
					ax.scatter(  X_curr_class[X_curr_class.columns[other_feature]], 
								 X_curr_class[X_curr_class.columns[feature_idx]], 
								 alpha=get_alpha(len(local_df)), 
								 c=tableau_palette_list[class_idx], 
								 label=class_name)
					ax.axhline(threshold, color='k', linestyle='dashed', linewidth=1)                            

			ax.set_title("Engineered Features: " + eng_feat1_name + ", " + eng_feat2_name + "\n(" + orig_feat1_name + ", " + orig_feat2_name + "\nRotated counter-clockwise " + str(degrees) + " Degrees)")
			ax.legend()   
			
		def graph_scatter_original_features(ax):
			
			def rotate_point(x ,y, degrees):
				# todo: we need to specify a point to rotate around instead of the origin I think.
				theta = np.radians(degrees)
				xx = x * cos(theta) - y * sin(theta) # todo: for consistency use np instead of math
				yy = x * sin(theta) + y * cos(theta)
				return xx,yy

			orig_feat1, orig_feat2, degrees, side = feature_sources[feature_idx]
			orig_feat1_name = str(X_extended.columns[orig_feat1]) 
			orig_feat2_name = str(X_extended.columns[orig_feat2])
			if side == 0: 
				other_feature = feature_idx+1
			else:
				other_feature = feature_idx-1
			eng_feat1_name = X_extended.columns[feature_idx]
			eng_feat2_name = X_extended.columns[other_feature]
			for class_idx, class_name in enumerate(class_arr):  
				match_ids = (local_y==class_name).values.reshape(1,-1)[0]
				idx_arr = local_y.loc[match_ids].index
				X_curr_class = local_df.loc[idx_arr]                        
				#print("original features. class: ", class_idx, ", ", class_name, ", len: ", len(X_curr_class))
				ax.scatter( X_curr_class[X_curr_class.columns[orig_feat1]],  \
							X_curr_class[X_curr_class.columns[orig_feat2]], \
							alpha=get_alpha(len(local_df)), \
							c=tableau_palette_list[class_idx], \
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
					rotated_x1, rotated_y1 = rotate_point(local_df[local_df.columns[other_feature]].min(), threshold,  -1*degrees)
					rotated_x2, rotated_y2 = rotate_point(local_df[local_df.columns[other_feature]].max(), threshold,  -1*degrees)                        
					#print("before points case 1: ", local_df[local_df.columns[other_feature]].min(), threshold, local_df[local_df.columns[other_feature]].max(), threshold)
				#print("after reverse rotation points: ", rotated_x1, rotated_y1, rotated_x2, rotated_y2)
				#ax[3].plot([rotated_x1, rotated_x2], [rotated_y1, rotated_y2], c="k", linestyle='dashed', linewidth=1) 

				# We're now, after inverse rotation, in the space of the 2 original columns scaled. We next get the
				# inverse scaling. 
				# scaled_data = X_orig.iloc[:2] # Take an arbitary pair of rows and replace the 2 columns we're interested in inverse scaling.
				# scaled_data.iloc[0][orig_feat1_name] = rotated_x1
				# scaled_data.iloc[0][orig_feat2_name] = rotated_y1
				# scaled_data.iloc[1][orig_feat1_name] = rotated_x2
				# scaled_data.iloc[1][orig_feat2_name] = rotated_y2
				#print("scaled_data: ")
				#display(scaled_data)
				# rev = scaler.inverse_transform(scaled_data)
				# #print("rev", rev)
				# scaled_x1 = rev[0][orig_feat1]
				# scaled_y1 = rev[0][orig_feat2]
				# scaled_x2 = rev[1][orig_feat1]
				# scaled_y2 = rev[1][orig_feat2]                    
				#print("after reverse scaling points: ", scaled_x1, scaled_y1, scaled_x2, scaled_y2)

				# todo: extend the line so it covers the full extent of the axes
				#ax.plot([scaled_x1, scaled_x2], [scaled_y1, scaled_y2], c="k", linestyle='dashed', linewidth=1) 
				if (not row is None): 
					ax.plot(row[orig_feat1], row[orig_feat2], marker="*", markersize=15, markeredgecolor="red", markerfacecolor="red")
				ax.set_title("Source Features:\n" + orig_feat1_name + ", " + orig_feat2_name)
				ax.legend()			
					
		tableau_palette_list=["tab:blue", "tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"]
		local_df = X_extended.loc[node_indexes[node_idx]]
		local_y = pd.DataFrame(y).loc[node_indexes[node_idx]]
		feature_idx = tree.feature[node_idx]
		feature_name = X_extended.columns[feature_idx]
		threshold = tree.threshold[node_idx]
		num_original_cols = len(X_orig.columns)
		class_counts = [local_y.values.tolist().count(x) for x in class_arr]
		
		# Render internal nodes
		if feature_idx >= 0:
			fig, ax = plt.subplots(nrows=1, ncols=6, figsize=(25,5))
			
			# In the first plot, show a bar chart giving the count for each target class
			graph_bar_chart_classes(ax[0], is_leaf=False)

			# In the second plot, show a histogram for the distribution of each target class
			graph_histogram(ax[1], log_scale=False, show_caption=True)
			
			# In the third plot, show a similar histogram, but on a log scale. For some datasets this is clearer, and for some
			# it is less clear. 
			graph_histogram(ax[2], log_scale=True, show_caption=False)

			# In the 4th and 5th plots, if applicable (if the feature is based on the rotation of 2 other features),
			# show a scatter plot of the two original and the two generated (rotated) features.
			if feature_idx >= num_original_cols: 
				rotated_x1, rotated_y1, rotated_x2, rotated_y2 = [], [], [],[] # temp!! remove

				graph_scatter_engineered_features(ax[3])
				# The 5th plot shows the 2d space of the 2 original features before they were rotated
				graph_scatter_original_features(ax[4])

				#TEMP 6th plot -- the original features scaled. just for debugging.
				orig_feat1, orig_feat2, degrees, side = feature_sources[feature_idx]
				orig_feat1_name = str(X_extended.columns[orig_feat1]) 
				orig_feat2_name = str(X_extended.columns[orig_feat2])
				X_scaled = pd.DataFrame(scaler.transform(X_orig), columns=X_orig.columns, index=X_orig.index)
				X_local_scaled = X_scaled.loc[local_df.index]
				if side == 0: 
					other_feature = feature_idx+1
				else:
					other_feature = feature_idx-1

				for class_idx, class_name in enumerate(class_arr):
					match_ids = (local_y==class_name).values.reshape(1,-1)[0]
					idx_arr = local_y.loc[match_ids].index
					X_curr_class = X_local_scaled.loc[idx_arr]                        
					ax[5].scatter(X_curr_class[X_curr_class.columns[orig_feat1]], 
								  X_curr_class[X_curr_class.columns[orig_feat2]], 
								  alpha=get_alpha(len(local_df)), 
								  c=tableau_palette_list[class_idx], 
								  label=class_name)

					X_curr_class = local_df.loc[idx_arr]
					if side == 0:
						ax[5].scatter( X_curr_class[X_curr_class.columns[feature_idx]],  \
							X_curr_class[X_curr_class.columns[other_feature]], \
							alpha=get_alpha(len(local_df)), \
							c=tableau_palette_list[class_idx], \
							marker="s")
					else:
						ax[5].scatter( X_curr_class[X_curr_class.columns[other_feature]],  \
							X_curr_class[X_curr_class.columns[feature_idx]], \
							alpha=get_alpha(len(local_df)), \
							c=tableau_palette_list[class_idx], \
							marker="s")

				#ax[5].plot([rotated_x1, rotated_x2], [rotated_y1, rotated_y2], c="k", )
				ax[5].set_title("Source Features scaled:\n" + orig_feat1_name + ", " + orig_feat2_name)
				ax[5].legend()

			else:
				ax[3].set_visible(False)
				ax[4].set_visible(False)

		# Render leaf nodes
		else:
			fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4,5))
			graph_bar_chart_classes(ax, is_leaf=True)	

		fig.suptitle("Node: " + str(node_idx))
		plt.show()
