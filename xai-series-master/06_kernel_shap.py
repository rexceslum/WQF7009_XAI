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

# Explore explainability using Kernel SHAP
# Select a background dataset (this is a subset of X_train)
background_dataset = X_train.sample(200)
# Define the model's prediction function
def model_predict(X):
    return rf.predict_proba(X)  # For classification, we need the probabilities

# Create Kernel SHAP explainer
kernel_explainer = shap.KernelExplainer(model_predict, background_dataset)
# Calculate shap values
start_index2 = 5
end_index2 = 15
X_test_subset2 = X_test[start_index2:end_index2]
kernel_shap_values = kernel_explainer.shap_values(X_test_subset2)
print("kernel_shap_values.shape ->", kernel_shap_values.shape)
print("kernel_explainer.expected_value ->", kernel_explainer.expected_value)
print("subset prediction ->", rf.predict(X_test_subset2))

# force plot for 1 sample
force_plot2 = shap.force_plot(kernel_explainer.expected_value[1], 
                            kernel_shap_values[8][:, 1], 
                            X_test_subset2.iloc[8])
shap.save_html("./xai-series-master/data/result/kernel_force_plot.html", force_plot2)

# %% >> Visualize global features
# Feature summary
summary_plot2 = shap.summary_plot(kernel_shap_values[:,:,1], X_test_subset2, show=False)
plt.savefig("./xai-series-master/data/result/kernel_summary_plot.png", bbox_inches="tight")