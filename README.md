
# Sistem Pendeteksi Hepatitis C - MEDHCV

MEDHCV is a web-based system developed to detect Hepatitis C based on various biochemical parameters using machine learning models. This system allows users to input data such as age, sex, and various biochemical markers to predict the likelihood of having Hepatitis C.

## Project Structure

The project consists of the following components:

1. **`app.py`**  
   This is the main Python file that runs the Flask web application. It handles user requests, processes data, and serves the prediction results.

2. **`hcv.sql`**  
   This file contains the SQL schema or data used for the application's database. It is used to set up or populate the database with initial data.

3. **`templates/`**  
   This folder contains the HTML templates used for rendering the front-end of the application. The templates are built with a focus on providing a user-friendly interface for data input and result display.

## Installation

To run this project locally, follow these steps:

1. Clone the repository or download the project files.
   
2. Navigate to the project directory:

   ```bash
   cd /path/to/Projek_Sistem_Pendeteksi_Hepatitis_C
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up the database using the provided SQL file:

   ```bash
   sqlite3 hcv.db < hcv.sql
   ```

   (Note: This assumes you are using SQLite; modify accordingly if using a different database.)

5. Run the Flask application:

   ```bash
   python app.py
   ```

   The application should now be running on `http://127.0.0.1:5000`.

## Usage

Once the application is running, open your web browser and navigate to the local server address (`http://127.0.0.1:5000`).

- **Input Data**: The form will allow you to input various parameters, including age, sex, and biochemical markers (e.g., ALB, ALP, ALT, etc.).
- **Prediction**: After entering the data, click "Mulai Deteksi" (Start Detection) to receive the prediction result for Hepatitis C.

## Model and Prediction

The system uses a machine learning model trained to predict the likelihood of Hepatitis C based on the provided parameters. The model is invoked from the Flask backend, and the prediction is returned to the user.

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
