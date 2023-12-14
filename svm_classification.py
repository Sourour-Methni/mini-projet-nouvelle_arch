import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import pickle

# Load your dataset
data = pd.read_csv('features_3_sec.csv', sep=";")  # Replace 'your_data.csv' with your actual data file.

# Split the data into features and labels
X = data.drop('label', axis=1)  # Assuming 'label' is the name of your label column.
y = data['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a PCA instance on the training data
n_components = 21  # Adjust as needed
pca = PCA(n_components=n_components)
pca.fit(X_train)

# Create an SVM classifier
svm_model = SVC(kernel='linear')  # You can choose the appropriate kernel.

# Apply PCA transformation to training and test data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train the classifier on the training data with PCA-transformed features
svm_model.fit(X_train_pca, y_train)

# Make predictions on the test data with PCA-transformed features
y_pred = svm_model.predict(X_test_pca)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
report = classification_report(y_test, y_pred)

# Save the trained classifier and PCA instance to pickle files
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(svm_model, file)

with open('pca_model.pkl', 'wb') as file:
    pickle.dump(pca, file)
