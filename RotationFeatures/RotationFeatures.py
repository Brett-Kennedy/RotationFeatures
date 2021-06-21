import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RotationFeatures():
	
	def __init__(self, degree_increment, determine_numeric_features=True, max_cols_created=np.inf):
		self.degree_increment = degree_increment
		self.determine_numeric_features = determine_numeric_features
		self.max_cols_created = max_cols_created 
		self.n_input_features_ = 0
		self.n_numeric_input_features_ = 0
		self.n_output_features_ = 0
		self.degrees_array = []
		self.is_numeric_arr = []
		self.feature_names_ = []
		self.degrees_array = list(range(self.degree_increment, 90, self.degree_increment))

	def fit(self, X):
		X=np.array(X)
		self.X = X
		self.n_input_features_ = X.shape[1]
		self.X_df = pd.DataFrame(X) 
		
		# Determine which features may be considered numeric
		if self.determine_numeric_features:
			self.is_numeric_arr = [1 if is_numeric_dtype(self.X_df[self.X_df.columns[c]]) and (self.X_df[self.X_df.columns[c]].nunique()>2) else 0 for c in range(len(self.X_df.columns))]
		else:
			self.n_numeric_input_features_ = self.n_input_features_

		# Determine the number of features that will be created
		# We look at each pair of numeric features (so n(n-1)/2 pairs), for each creating 2 new features for each rotation 
		# by a given number of degrees.
		self.n_output_features_ = self.n_numeric_input_features_ * (self.n_numeric_input_features_-1) * len(self.degrees_array)

		return self

	def transform(self, X):
		X=np.array(X)
		X_new = pd.DataFrame(X).copy()
		assert len(X_new.columns) == self.n_input_features_

		# Determine if the number of features generated would be too great
		if self.n_output_features_ > self.max_cols_created:
			return X
			
		scaler = MinMaxScaler()
		#scaler.fit(X)
		scaled_X_df = pd.DataFrame(scaler.fit_transform(X))                        
			
		new_feat_idx = 0
		for c1_idx in range(len(self.X_df.columns)-1):
			if (self.is_numeric_arr[c1_idx] == 0):
				continue
			for c2_idx in range(c1_idx+1, len(self.X_df.columns)):
				if (self.is_numeric_arr[c2_idx] == 0):
					continue
				for d in self.degrees_array:
					rotated_df = self.rotate_data(scaled_X_df, c1_idx, c2_idx, d)

					new_col_name = "R_" + str(new_feat_idx)
					new_feat_idx += 1
					X_new[new_col_name] = rotated_df[0]

					new_col_name = "R_" + str(new_feat_idx)
					new_feat_idx += 1
					X_new[new_col_name] = rotated_df[1]
		self.feature_names_ = list(X_new.columns)
		return X_new.values    

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X) 

	def get_feature_names(self):
		return self.feature_names_ 

	def get_params(self):		
		return {"degree_increment": self.degree_increment}

	def set_params(self, **params):
		for key, value in params.items():
            setattr(self, key, value)
		return self

	def rotate_data(self, X, col1, col2, degrees):
		# Create the rotation matrix
		theta = np.radians(degrees)
		r = np.array(( (np.cos(theta), -np.sin(theta)),
					   (np.sin(theta),  np.cos(theta)) ))

		# Get the specified columns and rotate them
		col_names = [X.columns[col1], X.columns[col2]]
		orig_data = X[col_names]
		rotated_data = r.dot(orig_data.T)    
		rotated_data_df = pd.DataFrame(rotated_data).T
		
		return rotated_data_df    