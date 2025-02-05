import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

from fiztorch.tensor import Tensor
from fiztorch.nn.layers import Linear, ReLU, Sigmoid
from fiztorch.nn.sequential import Sequential
import fiztorch.nn.functional as F
import fiztorch.optim.optimizer as opt
from fiztorch.utils import visual

def load_heart_data():
    try:
        # Load the heart disease dataset
        dataset = load_breast_cancer()
        X, y = dataset.data, dataset.target

        # Normalize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        return (Tensor(X_train), Tensor(y_train), 
                Tensor(X_test), Tensor(y_test))
    except Exception as e:
        print(f"Error loading Heart Disease data: {str(e)}")
        raise

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics using sklearn
    """
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }

def print_detailed_metrics(y_true, y_pred, phase="Training"):
    """
    Print detailed classification metrics and confusion matrix
    """
    print(f"\n{phase} Detailed Metrics:")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    metrics = calculate_metrics(y_true, y_pred)
    print(f"\nSummary Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

def create_model():
    try:
        model = Sequential(
            Linear(30, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 1),
            Sigmoid()
        )
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

def train_epoch(model, optimizer, X_train, y_train, batch_size):
    try:
        indices = np.random.permutation(len(X_train.data))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(X_train.data), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = Tensor(X_train.data[batch_indices])
            y_batch = Tensor(y_train.data[batch_indices])

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = F.binary_cross_entropy(predictions, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.data
            n_batches += 1

        return total_loss / n_batches
    except Exception as e:
        print(f"Error during training epoch: {str(e)}")
        raise

def evaluate(model, X, y):
    """
    Evaluate model and return predictions and accuracy
    """
    try:
        predictions = model(X)
        pred_binary = predictions.data > 0.5
        accuracy = np.mean(pred_binary == y.data)
        return pred_binary, accuracy
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

def main():
    try:
        print("Loading Heart Disease data...")
        X_train, y_train, X_test, y_test = load_heart_data()

        print("Creating model...")
        model = create_model()
        optimizer = opt.Adam(model.parameters(), lr=0.001)

        n_epochs = 100
        batch_size = 32

        train_losses = []
        train_accuracies = []
        test_accuracies = []
        loss_visual = visual.LossVisualizer()

        print("Training started...")
        for epoch in range(n_epochs):
            # Train
            avg_loss = train_epoch(model, optimizer, X_train, y_train, batch_size)
            
            # Evaluate
            train_preds, train_acc = evaluate(model, X_train, y_train)
            test_preds, test_acc = evaluate(model, X_test, y_test)

            # Record metrics
            loss_visual.update(avg_loss)
            train_losses.append(avg_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch + 1}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Training Accuracy: {train_acc:.4f}")
                print(f"Test Accuracy: {test_acc:.4f}")
                
                # Calculate and print detailed metrics
                train_metrics = calculate_metrics(y_train.data, train_preds)
                test_metrics = calculate_metrics(y_test.data, test_preds)
                
                print("\nTraining Metrics:")
                for metric, value in train_metrics.items():
                    print(f"{metric}: {value:.4f}")
                    
                print("\nTest Metrics:")
                for metric, value in test_metrics.items():
                    print(f"{metric}: {value:.4f}")
                print("-" * 50)

        print("\nTraining complete!")
        loss_visual.plot(window_size=5)

        # Final detailed evaluation
        final_train_preds, _ = evaluate(model, X_train, y_train)
        final_test_preds, _ = evaluate(model, X_test, y_test)
        
        print("\nFinal Model Evaluation:")
        print_detailed_metrics(y_train.data, final_train_preds, "Training")
        print_detailed_metrics(y_test.data, final_test_preds, "Test")

    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise