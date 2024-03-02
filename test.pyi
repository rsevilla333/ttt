from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# load database function
def load_data(file_name):
    return np.loadtxt("datasets/tictac_" + file_name + ".txt")


final_bc = load_data("final")
multi_bc = load_data("multi")
single_bc = load_data("single")


def evaluate_classifier(classifier, X, y):
    """
    Function to evaluate a classifier using given data.

    Args:
    classifier: The classifier object.
    X: The feature matrix.
    y: The target vector.

    Returns:
    accuracy: The accuracy score.
    conf_matrix: The confusion matrix.
    cv_scores: The cross-validation scores.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier.fit(X_train, y_train)

    # O X O
    # O O X 
    #  X X

    # Predict the optimal moves for the entire dataset
    y_pred = classifier.predict(X_test)
    print(X_test[0])
    print(y_pred[0])

    # Convert the predicted moves into row and column indices
    # Note: This conversion logic depends on how your Tic Tac Toe board is represented
    # Here's a simple example assuming a 3x3 board represented as a 2D array
    # You may need to adjust this based on your actual implementation
    row, col = divmod(y_pred[0], 3)  # Assuming y_pred contains a single prediction

    # for label in y_pred:
    #     print(label)

    return row, col
    # # Split the dataset into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Train the classifier on the training data
    # classifier.fit(X_train, y_train)

    # # Predict the optimal moves for the O player on the test data
    # y_pred = classifier.predict(X_test)
    # print(y_pred)

    # # Calculate accuracy
    # accuracy = accuracy_score(y_test, y_pred)

    # # Generate confusion matrix
    # conf_matrix = confusion_matrix(y_test, y_pred)

    # # Perform k-fold cross-validation
    # cv_scores = cross_val_score(classifier, X, y, cv=10)

    # # Print results
    # print("Accuracy:", accuracy)
    # print("Confusion Matrix:")
    # print(conf_matrix)
    # print("Cross-validation scores:", cv_scores)
    # print("Mean accuracy:", np.mean(cv_scores))


# Initialize the KNN classifier
print("KNN classifier")
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Extract features (X) and labels (y)
X = single_bc[:, :9]
y = single_bc[:, 9]

# Evaluate the classifier
row, col = evaluate_classifier(knn_classifier, X, y)

print(row)
print(col)

# print("SVM classifier")
# # Initialize the Linear SVM classifier
# svm_classifier = LinearSVC(dual=True)

# # Extract features (X) and labels (y)
# X = single_bc[:, :9]
# y = single_bc[:, 9]

# # Evaluate the classifier
# evaluate_classifier(svm_classifier, X, y)

# print("MLP classifier")
# # Initialize the MLP classifier
# mlp_classifier = MLPClassifier(max_iter=100000)

# # Extract features (X) and labels (y)
# X = single_bc[:, :9]
# y = single_bc[:, 9]

# # Evaluate the classifier
# evaluate_classifier(mlp_classifier, X, y)

