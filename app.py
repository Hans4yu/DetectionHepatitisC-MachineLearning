from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
import mysql.connector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import seaborn as sns
import os
import pandas as pd
import pickle
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import optuna
import shap
import numpy as np
import io
import base64

app = Flask(__name__)
app.secret_key = "secret_key"  # Diperlukan untuk menampilkan pesan flash

# Konfigurasi login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"  # Tentukan route login jika belum login
bcrypt = Bcrypt(app)

UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MODEL_PATH = os.path.join(MODEL_FOLDER, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_FOLDER, 'scaler.pkl')  
TEMP_FOLDER = 'temp'
app.config['MODEL_FOLDER'] = 'models'
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

os.makedirs(TEMP_FOLDER, exist_ok=True)
# Buat folder 'uploads' jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Model Admin untuk Flask-Login
class Admin(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# Mengambil admin dari database
def get_admin_by_username(username):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="hcv"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin WHERE username = %s", (username,))
    admin = cursor.fetchone()
    conn.close()
    return admin

# Mengambil admin untuk login
@login_manager.user_loader
def load_admin(admin_id):
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="hcv"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin WHERE id = %s", (admin_id,))
    admin = cursor.fetchone()
    conn.close()
    if admin:
        return Admin(id=admin[0], username=admin[1])
    return None

# Route untuk login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="hcv"
        )
        cursor = conn.cursor()

        cursor.execute("SELECT id, username, password FROM admin WHERE username = %s", (username,))
        admin = cursor.fetchone()

        if admin and bcrypt.check_password_hash(admin[2], password):
            login_user(Admin(id=admin[0], username=admin[1]))
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password", "danger")
        
        cursor.close()
        conn.close()

    return render_template('login.html')


# Route untuk logout
@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

# Route utama
@app.route('/')
def index():
    return render_template('index.html')

# In app.py - Update the form route

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        try:
            # Load the model and scaler
            with open(MODEL_PATH, 'rb') as file:
                model = pickle.load(file)
            with open(SCALER_PATH, 'rb') as file:
                scaler = pickle.load(file)

            # Extract and validate form data
            features = {
                'Age': int(request.form['age']),
                'Sex': int(request.form['sex']),
                'ALB': float(request.form['alb']),
                'ALP': float(request.form['alp']),
                'ALT': float(request.form['alt']),
                'AST': float(request.form['ast']),
                'BIL': float(request.form['bil']),
                'CHE': float(request.form['che']),
                'CHOL': float(request.form['chol']),
                'CREA': float(request.form['crea']),
                'GGT': float(request.form['ggt']),
                'PROT': float(request.form['prot'])
            }

            # Validate ranges
            validation_ranges = {
                'Age': (0, 120),
                'Sex': (0, 1),
                'ALB': (0, 200.0),
                'ALP': (0, 200),
                'ALT': (0, 200),
                'AST': (0, 200),
                'BIL': (0, 200),
                'CHE': (0, 200),
                'CHOL': (0, 300),
                'CREA': (0, 300),
                'GGT': (0, 300),
                'PROT': (0, 100)
            }

            for key, value in features.items():
                min_val, max_val = validation_ranges[key]
                if not min_val <= value <= max_val:
                    raise ValueError(f"{key} value {value} is outside valid range ({min_val}, {max_val})")
                
            

            # Convert to DataFrame and scale
            features_df = pd.DataFrame([features])
            features_scaled = scaler.transform(features_df)
            features_scaled_df = pd.DataFrame(features_scaled, columns=features_df.columns)

            # Get prediction and probabilities
            prediction = model.predict(features_scaled_df)[0]
            probabilities = model.predict_proba(features_scaled_df)[0]
            risk_score = round(probabilities[1] * 100, 2)

            # Generate background data for SHAP
            background_data = pd.DataFrame(columns=features_df.columns)
            for col in features_df.columns:
                min_val, max_val = validation_ranges[col]
                background_data[col] = np.random.uniform(min_val, max_val, size=100)
            
            background_scaled = scaler.transform(background_data)

            # Create explainer with improved settings
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1],
                background_scaled,
                link="logit"
            )

            # Calculate SHAP values with increased samples
            shap_values = explainer.shap_values(features_scaled, nsamples=500)

            # If shap_values is a list, take the appropriate values
            if isinstance(shap_values, list):
                plot_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                plot_values = shap_values

            # Create enhanced SHAP visualization
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                plot_values,
                features_df,
                plot_type="bar",
                max_display=12,
                show=False,
                plot_size=(12, 8),
                color_bar_label='Feature value'
            )

            # Enhance plot styling
            plt.gcf().set_size_inches(12, 8)
            plt.title("Impact of Features on HCV Prediction", fontsize=14, pad=20)
            plt.xlabel("SHAP value (impact on model output)", fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)

            # Add enhanced feature importance information
            for i, feature in enumerate(features_df.columns):
                current_value = features_df[feature].iloc[0]
                impact = abs(plot_values[0][i])
                color = 'red' if impact > np.mean(np.abs(plot_values)) else 'gray'
                plt.text(
                    plt.gca().get_xlim()[1], 
                    i,
                    f'Current: {current_value:.2f}',
                    fontsize=9,
                    va='center',
                    ha='left',
                    color=color,
                    alpha=0.8
                )

            # Adjust layout and save
            plt.tight_layout()
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
            plt.close()
            img.seek(0)
            shap_plot_base64 = base64.b64encode(img.getvalue()).decode()

            # Save to history
            conn = mysql.connector.connect(
                host="localhost",
                user="root",
                password="",
                database="hcv"
            )
            cursor = conn.cursor()
            
            # Create history table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    risk_score DECIMAL(10,2),
                    age INT,
                    sex INT,
                    alb DECIMAL(10,2),
                    alp DECIMAL(10,2),
                    alt DECIMAL(10,2),
                    ast DECIMAL(10,2),
                    bil DECIMAL(10,2),
                    che DECIMAL(10,2),
                    chol DECIMAL(10,2),
                    crea DECIMAL(10,2),
                    ggt DECIMAL(10,2),
                    prot DECIMAL(10,2),
                    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert record
            cursor.execute("""
                INSERT INTO detection_history 
                (user_id, risk_score, age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                current_user.id,  # Add user_id
                float(risk_score),
                features['Age'],
                features['Sex'],
                float(features['ALB']),
                float(features['ALP']),
                float(features['ALT']),
                float(features['AST']),
                float(features['BIL']),
                float(features['CHE']),
                float(features['CHOL']),
                float(features['CREA']),
                float(features['GGT']),
                float(features['PROT'])
            ))
            conn.commit()
            cursor.close()
            conn.close()

            return render_template(
                'result.html',
                shap_plot_base64=shap_plot_base64,
                risk_score=risk_score,
                feature_values=features,
                prediction=prediction
            )

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
            return redirect(url_for('form'))

    return render_template('form.html')

@app.route('/riwayat')
@login_required
def riwayat():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="hcv"
    )
    cursor = conn.cursor(dictionary=True)
    
    # Get records for current user only
    cursor.execute("""
        SELECT * FROM detection_history 
        WHERE user_id = %s 
        ORDER BY detection_date DESC
    """, (current_user.id,))
    history = cursor.fetchall()

    for record in history:
        record['Category'] = 1 if float(record['risk_score']) >= 50 else 0

    if history:
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'history_{current_user.id}.csv')
        df = pd.DataFrame(history)
        
        column_mapping = {
            'age': 'Age',
            'sex': 'Sex',
            'alb': 'ALB',
            'alp': 'ALP',
            'alt': 'ALT',
            'ast': 'AST',
            'bil': 'BIL',
            'che': 'CHE',
            'chol': 'CHOL',
            'crea': 'CREA',
            'ggt': 'GGT',
            'prot': 'PROT'
        }

        df.rename(columns=column_mapping, inplace=True)
        df.drop(['id', 'user_id', 'detection_date'], axis=1, inplace=True, errors='ignore')
        df = df[['Category', 'Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]
        df.to_csv(csv_path, index=False)
        csv_available = True
    else:
        csv_available = False
    
    cursor.close()
    conn.close()
    
    return render_template('riwayat.html', history=history, csv_available=csv_available)



# Modified download route
@app.route('/download_history')
@login_required
def download_history():
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], f'history_{current_user.id}.csv')
    if os.path.exists(csv_path):
        return send_file(
            csv_path,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'hcv_detection_history_{current_user.id}.csv'
        )
    else:
        flash("No CSV file available for download", "warning")
        return redirect(url_for('riwayat'))



@app.route('/train_upload', methods=['GET', 'POST'])
@login_required
def train_upload():
    if request.method == 'POST':
        if 'dataset' in request.files:
            file = request.files['dataset']
            if file.filename != '':
                # Save the uploaded file
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)
                
                # Read and process the dataset
                df = pd.read_csv(filepath)
                session['current_dataset'] = filepath
                
                return jsonify({
                    'success': True,
                    'message': 'Dataset uploaded successfully',
                    'data': df.to_dict('records'),
                    'columns': df.columns.tolist()
                })

        elif 'split_data' in request.form:
            try:
                # Convert integer percentage to float decimal
                test_size_percent = int(request.form['test_size'])
                test_size = test_size_percent / 100.0
                
                if 0 < test_size <= 0.4:
                    df = pd.read_csv(session['current_dataset'])
                    X = df[['Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']]
                    y = df['Category']

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    
                    # Save split data
                    os.makedirs('temp', exist_ok=True)
                    with open('temp/X_model.pkl', 'wb') as f:
                        pickle.dump((X_train, X_test), f)
                    with open('temp/y_model.pkl', 'wb') as f:
                        pickle.dump((y_train, y_test), f)

                    # Check class balance
                    class_counts = y_train.value_counts()
                    imbalance_ratio = class_counts.min() / class_counts.max()
                    needs_smote = bool(imbalance_ratio < 0.5)  # Convert to regular bool

                    # Convert class distribution to serializable format
                    class_dist = {str(k): int(v) for k, v in class_counts.items()}

                    # Create and save class distribution chart
                    plt.figure(figsize=(8, 6))
                    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
                    plt.title('Class Distribution in Training Data')
                    plt.xlabel('Category')
                    plt.ylabel('Count')
                    plt.tight_layout()
                    smote_chart_path = 'static/smote_distribution.png'
                    plt.savefig(smote_chart_path)
                    plt.close()

                    return jsonify({
                        'success': True,
                        'needs_smote': int(needs_smote),  # Convert bool to int
                        'class_distribution': class_dist,
                        'train_size': int(len(X_train)),
                        'test_size': int(len(X_test)),
                        'smote_chart_url': smote_chart_path  # Return the chart URL
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Test size must be between 1% and 40%'
                    }), 400
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': f'Error splitting data: {str(e)}'
                }), 500

        elif 'train_model' in request.form:
            # Load split data
            with open('temp/X_model.pkl', 'rb') as f:
                X_train, X_test = pickle.load(f)
            with open('temp/y_model.pkl', 'rb') as f:
                y_train, y_test = pickle.load(f)

            # Get hyperparameter ranges
            n_neighbors_min = int(request.form['n_neighbors_min'])
            n_neighbors_max = int(request.form['n_neighbors_max'])
            p_min = int(request.form['p_min'])
            p_max = int(request.form['p_max'])
            weights = request.form.getlist('weights')
            n_trials = int(request.form['n_trials'])  # Get number of trials from form

            def objective(trial):
                n_neighbors = trial.suggest_int('n_neighbors', n_neighbors_min, n_neighbors_max)
                p = trial.suggest_int('p', p_min, p_max)
                weight = trial.suggest_categorical('weights', weights)
                
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, weights=weight)
                knn.fit(X_train, y_train)
                return -accuracy_score(y_test, knn.predict(X_test))

            study = optuna.create_study()
            study.optimize(objective, n_trials=n_trials)  # Use custom n_trials

            # Train final model with best parameters
            best_knn = KNeighborsClassifier(**study.best_params)
            best_knn.fit(X_train, y_train)
            y_pred = best_knn.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])


            pd.DataFrame(X_train).to_csv('static/trained_data.csv', index=False)

            # Feature Importance (based on feature variance or another measure)
            feature_variances = X_train.var()
            feature_importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_variances
            }).sort_values(by='Importance', ascending=False)

            # Save Feature Importance Chart
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')
            plt.title('Feature Importance (Variance-Based)')
            plt.tight_layout()
            plt.savefig('static/feature_importance.png')
            plt.close()

            # Create confusion matrix visualization
            plt.figure(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, labels=[0, 1], cmap='Blues'
            )
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig('static/confusion_matrix.png')
            plt.close()

            return jsonify({
                'success': True,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                },
                'best_params': study.best_params,
                'n_trials': n_trials
            })

    return render_template('train_upload.html')




@app.route('/get_trained_data', methods=['GET'])
@login_required
def get_trained_data():
    try:
        if 'current_dataset' not in session:
            return jsonify({'success': False, 'message': 'No dataset found. Please upload a dataset.'}), 400

        dataset_path = session['current_dataset']
        df = pd.read_csv(dataset_path)

        page = int(request.args.get('page', 1))
        per_page = 10
        total_rows = len(df)
        start = (page - 1) * per_page
        end = start + per_page

        return jsonify({
            'success': True,
            'data': df.iloc[start:end].to_dict('records'),
            'columns': df.columns.tolist(),
            'total_rows': total_rows,
            'per_page': per_page
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500




# Add to app.py

@app.route('/save_history', methods=['POST'])
def save_history():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="hcv"
    )
    cursor = conn.cursor()
    
    # Create history table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detection_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            category INT,
            age INT,
            sex INT,
            alb FLOAT,
            alp FLOAT,
            alt FLOAT,
            ast FLOAT,
            bil FLOAT,
            che FLOAT,
            chol FLOAT,
            crea FLOAT,
            ggt FLOAT,
            prot FLOAT,
            detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Insert form data
    try:
        cursor.execute("""
            INSERT INTO detection_history 
            (category, age, sex, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            request.form['prediction'],
            request.form['age'],
            request.form['sex'],
            request.form['alb'],
            request.form['alp'],
            request.form['alt'],
            request.form['ast'],
            request.form['bil'],
            request.form['che'],
            request.form['chol'],
            request.form['crea'],
            request.form['ggt'],
            request.form['prot']
        ))
        conn.commit()
        
        # Check if we have 10 or more records
        cursor.execute("SELECT COUNT(*) FROM detection_history")
        count = cursor.fetchone()[0]
        
        if count >= 10:
            # Generate CSV file
            cursor.execute("SELECT * FROM detection_history ORDER BY detection_date DESC LIMIT 10")
            records = cursor.fetchall()
            
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'history.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Category', 'Age', 'Sex', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'])
                for record in records:
                    writer.writerow(record[1:14])  # Exclude id and timestamp
                    
        return jsonify({'success': True})
        
    except Exception as e:
        conn.rollback()
        return jsonify({'success': False, 'error': str(e)})
    finally:
        cursor.close()
        conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            flash("All fields are required.", "danger")
            return redirect(url_for('register'))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('register'))

        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="hcv"
        )
        cursor = conn.cursor()

        # Create the admin table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(100) NOT NULL UNIQUE,
                password VARCHAR(100) NOT NULL
            )
        """)


        # Check for existing username or email
        cursor.execute("SELECT * FROM admin WHERE username = %s OR email = %s", (username, email))
        existing_user = cursor.fetchone()
        if existing_user:
            flash("Username or email already exists.", "danger")
            return redirect(url_for('register'))

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        cursor.execute("INSERT INTO admin (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        conn.commit()
        cursor.close()
        conn.close()
        flash("Registration successful!", "success")
        return redirect(url_for('login'))

    return render_template('register.html')



if __name__ == '__main__':
    app.run(debug=True)

