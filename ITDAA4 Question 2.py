#Import Libraries
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Extract and Load the dataset
#------------------------------
df = pd.read_csv(r"C:\Users\reyna\OneDrive - Eduvos\Desktop\icml-topic-modelling\asteroid data and description\asteroid.csv")

# Load selected useful columns
usecols = [
    "neo", "pha", "H", "diameter", "albedo",
    "epoch", "epoch_mjd", "e", "a", "q", "i", "om", "w", "ma",
    "ad", "n", "tp", "per", "per_y", "moid", "moid_ld",
    "sigma_e", "sigma_a", "sigma_q", "sigma_i", "sigma_om",
    "sigma_w", "sigma_ma", "sigma_ad", "sigma_n", "sigma_tp",
    "sigma_per", "class", "rms"
]

df = pd.read_csv(r"C:\Users\reyna\OneDrive - Eduvos\Desktop\icml-topic-modelling\asteroid data and description\asteroid.csv", usecols=usecols)
print(df.head())
print(df.info())
print(df.sample)
print(df.isnull().sum())

#Data Cleaning
#-------------

print("Original shape:", df.shape)

#Remove duplicates
df = df.drop_duplicates()

#Remove rows where target variable missing
df = df.dropna(subset=["diameter"])
print("Shape after cleaning:", df.shape)

#Seperate target variables
X = df.drop(columns=["diameter"])
y = df["diameter"]

print("Missing values after target cleaning:")
print(X.isnull().sum())

#Categorical encoding
#---------------------

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)

#Preprocessing pipeline
#-----------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

X_processed = preprocessor.fit_transform(X)

#Feature selection
#------------------

#Get processed feature names
feature_names = list(numerical_cols)

encoded_cat_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols)
feature_names.extend(encoded_cat_names)
feature_names = np.array(feature_names)

# Select top 12 features
selector = SelectKBest(score_func=f_regression, k=12)

X_selected = selector.fit_transform(X_processed, y)

selected_features = feature_names[selector.get_support()]
selected_scores = np.array(selector.scores_)[selector.get_support()]

selected_feature_df = pd.DataFrame({
    "Feature": selected_features,
    "Score": selected_scores
}).sort_values(by="Score", ascending=False)

print(selected_feature_df)


#Plot selected features
#-----------------------

plt.figure(figsize=(10, 6))
plt.barh(
    selected_feature_df["Feature"][::-1],
    selected_feature_df["Score"][::-1]
)
plt.title("Selected Features for Predicting Asteroid Diameter")
plt.xlabel("Feature Selection Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


#Feature Scaling
#---------------

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


#Train-test split
#------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

#fit Decision Tree Regressor
#---------------------------
decision_tree = DecisionTreeRegressor(
    random_state=42,
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10
    )

decision_tree.fit(X_train, y_train)

print("Decision Tree Regressor trained successfully.")



#Model Evaluation
#----------------

y_pred = decision_tree.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared score:", r2)



#Visualize the Decision Tree
#---------------------------
plt.figure(figsize=(20, 10))
plot_tree(
    decision_tree,
    feature_names=list(selected_features),
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)

plt.title("Decision Tree for predicting Asteroid Diameter")
plt.tight_layout()
plt.show()