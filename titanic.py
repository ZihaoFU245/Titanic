# %%
# 0 = No , 1 = Yes
# pclass , 1 = 1st , 2 = 2nd , 3 = 3rd

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# %%
df = pd.read_csv('./titanic/train.csv')

X = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
X = X.dropna()
labels = X["Survived"]
X = X.drop("Survived", axis=1)

print(X.shape , labels.shape)

# %%
# data preprocess
X["Sex"] = X["Sex"].map({"male" : 0 , "female" : 1}) # male : 0 , female : 1
X["Embarked"] = X["Embarked"].map({"S" : 0 , "Q" : 1 , "C" : 2})
X['AgeBin'] = pd.cut(X['Age'], bins=4, labels=[0, 1, 2, 3])
X['FareBin'] = pd.cut(X['Fare'], bins=5, labels=range(5))

X = X.drop(["Age", "Fare"], axis=1)
X = X.dropna()
X.head()

# %%
# Add this line to check the structure of training data before numpy conversion
print("Training columns:", X.columns)
X_train , X_test , y_train , y_test = train_test_split(X , labels , test_size=0.15 , random_state=42)
print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)
print(type(X_train))
X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()
print(type(X_train), type(y_train))

# %%
model = CategoricalNB(alpha=1.0)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test , y_pred))
mtx = confusion_matrix(y_test , y_pred)
sns.heatmap(mtx , annot=True , fmt="d" , cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test , y_pred))

# %%
pred_df = pd.read_csv('./titanic/test.csv')
id_col = pred_df["PassengerId"].values

# Process test data exactly the same as training data
pred_df = pred_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
# Keep rows with NaN values until after all transformations (matching training process)
pred_df["Sex"] = pred_df["Sex"].map({"male": 0, "female": 1})  # male : 0 , female : 1
pred_df["Embarked"] = pred_df["Embarked"].map({"S": 0, "Q": 1, "C": 2})
pred_df['AgeBin'] = pd.cut(pred_df['Age'], bins=4, labels=[0, 1, 2, 3])
pred_df['FareBin'] = pd.cut(pred_df['Fare'], bins=5, labels=range(5))
pred_df = pred_df.drop(["Age", "Fare"], axis=1)
# Now drop NaN values
pred_df = pred_df.dropna()

# Ensure columns match training data
print("Prediction columns:", pred_df.columns)

# Create a DataFrame with the same columns as the training data
matching_cols = list(X.columns)
for col in matching_cols:
    if col not in pred_df.columns:
        pred_df[col] = 0
pred_df = pred_df[matching_cols]
print("Final prediction columns:", pred_df.columns)
print("Final prediction shape:", pred_df.shape)

# Store the indices of valid rows
valid_indices = pred_df.index
pred_features = pred_df.values
valid_ids = id_col[valid_indices]

print("Prediction features shape:", pred_features.shape)
print("Training data shape:", X_train.shape)

y_pred = model.predict(pred_features)
print(y_pred)
submission = pd.DataFrame({"PassengerId": valid_ids, "Survived": y_pred})
submission.to_csv("submission.csv", index=False)




