from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
import pandas as pd
import mysql.connector
import optuna.visualization as vis

# Koneksi ke database MySQL di XAMPP
def get_data_from_db():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="hcv"
        )
        query = "SELECT * FROM hcv_data"
        data = pd.read_sql(query, conn)
        conn.close()
        return data
    except Exception as e:
        print("Error connecting to the database:", e)
        return None

# Load dataset dari database
data = get_data_from_db()
if data is None:
    raise ValueError("Failed to load data from the database.")

# Pisahkan fitur dan target
X = data[['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]
y = data['Category']

# Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Optuna optimization function for KNN
def knn_objective(trial):
    # Hyperparameter tuning
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    p = trial.suggest_int('p', 1, 5) if metric == 'minkowski' else 2

    # Initialize KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric, p=p)

    # Cross-validation
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    return scores.mean()

# Optimize KNN model using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(knn_objective, n_trials=1000)

# Best parameters and F1 score
best_params = study.best_params
print("Best Parameters:", best_params)

# Train the final KNN model with the best parameters
final_knn = KNeighborsClassifier(**best_params)
final_knn.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred_final = final_knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final, average='weighted')
recall = recall_score(y_test, y_pred_final, average='weighted')
final_f1 = f1_score(y_test, y_pred_final, average='weighted')

# Print results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Final F1 Score:", final_f1)

# Save evaluation results back to the database
def save_evaluation_to_db(accuracy, precision, recall, f1):
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="hcv"
        )
        cursor = conn.cursor()

        # Perbaikan nama kolom untuk 'precision'
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INT AUTO_INCREMENT PRIMARY KEY,
                accuracy FLOAT,
                precision_score FLOAT,
                recall FLOAT,
                f1_score FLOAT
            )
        """)
        cursor.execute("""
            INSERT INTO evaluation_results (accuracy, precision_score, recall, f1_score)
            VALUES (%s, %s, %s, %s)
        """, (accuracy, precision, recall, f1))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Error saving evaluation to the database:", e)

# Save the evaluation metrics
save_evaluation_to_db(accuracy, precision, recall, final_f1)

# Visualize Optuna study
print("Generating visualizations...")
vis.plot_optimization_history(study).show()
vis.plot_parallel_coordinate(study).show()
vis.plot_param_importances(study).show()
