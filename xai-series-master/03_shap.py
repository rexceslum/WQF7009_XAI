# %% Imports
from utils import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import shap
import matplotlib.pyplot as plt

# %% Load and preprocess data
data_loader = DataLoader()
data_loader.load_dataset()
data_loader.preprocess_data()
# Split the data for evaluation
X_train, X_test, y_train, y_test = data_loader.get_data_split()
# Oversample the train data
X_train, y_train = data_loader.oversample(X_train, y_train)
print("X_train.shape ->", X_train.shape)
print("X_test.shape ->", X_test.shape)

# %% Fit blackbox model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"F1 Score -> {f1_score(y_test, y_pred, average='macro')}")
print(f"Accuracy -> {accuracy_score(y_test, y_pred)}")

# %% Create SHAP explainer
explainer = shap.TreeExplainer(rf)
# Calculate shapley values for test data
start_index = 1
end_index = 101
X_test_subset = X_test[start_index:end_index]
shap_values = explainer.shap_values(X_test_subset)
print("X_test_subset.shape ->", X_test_subset.shape)

# %% Investigating the values (classification problem)
# class 0 = contribution to class 1
# class 1 = contribution to class 2
print("shap_values.shape ->", shap_values.shape)
print("shap_values[0][0] ->", shap_values[0][0])
print("shap_values[0][0] -> [class 0, class 1]")

# %% >> Visualize local predictions
shap.initjs()

# Force plot
prediction = rf.predict(X_test_subset)[8]
print("subset prediction ->", rf.predict(X_test_subset))
print(f"The RF predicted -> {prediction}")
print("base value of [class 0, class 1] ->", explainer.expected_value)
print("X_test_subset.iloc[8].shape ->", X_test_subset.iloc[8].shape)
print("shap_values[8][:, 1].shape ->", shap_values[1][:, 1].shape)

# force plot for 1 sample
force_plot = shap.force_plot(explainer.expected_value[1], 
                            shap_values[8][:, 1], 
                            X_test_subset.iloc[8])
shap.save_html("./xai-series-master/data/force_plot.html", force_plot)

# %% >> Visualize global features
# Feature summary
summary_plot = shap.summary_plot(shap_values[:,:,1], X_test_subset, show=False)
plt.savefig("./xai-series-master/data/summary_plot.png", bbox_inches="tight")
