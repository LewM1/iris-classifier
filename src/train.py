import os
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def main():
    
    # Locate the project root (one level above this script's folder)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Create/use outputs folder outside src/
    
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split data
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predictions & accuracy
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix figure to: {cm_path}")

    # Save trained model
    
    model_path = os.path.join(output_dir, "decision_tree_model.joblib")
    joblib.dump(model, model_path)

    print(f"Saved trained model to: {model_path}")

if __name__ == "__main__":
    main()