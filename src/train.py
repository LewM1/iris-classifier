import os
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# --- Create outputs folder ---
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# --- Load data ---
iris = load_iris()
X = iris.data
y = iris.target
print(iris.feature_names, iris.target_names)

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Train model ---
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# --- Predictions ---
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

# --- Accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# --- Save confusion matrix as PNG ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
plt.close()

print("Saved confusion_matrix.png")

# --- Save trained model ---
model_path = os.path.join(output_dir, "decision_tree_model.pkl")
joblib.dump(model, model_path)

print("Saved model to:", model_path)