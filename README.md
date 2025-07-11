# FinOps Cloud Cost Forecaster

This project is a web application designed to forecast cloud costs using various time series models. Users can upload their FinOps cost data (in CSV format), configure preprocessing parameters, select from multiple forecasting models (including ARIMA, Prophet, Exponential Smoothing, and LSTM), and view the forecasts along with model performance metrics.

## Project Structure

finops_forecast_app/ ├── backend/ │ ├── app/ │ │ ├── init.py │ │ ├── api.py # FastAPI routes │ │ ├── data_handler.py # CSV loading, column identification │ │ ├── evaluator.py # Model performance metrics calculation │ │ ├── main.py # FastAPI app initialization │ │ ├── model_trainer.py# Forecasting model implementations │ │ ├── preprocessor.py # Data cleaning, resampling, outlier handling │ │ ├── schemas.py # Pydantic request/response models │ │ └── utils.py # Utility functions (e.g., logger) │ ├── temp_data/ # Temporary storage for uploaded & processed files │ │ └── processed_cache/# Cached preprocessed data files │ └── requirements.txt # Python backend dependencies ├── frontend/ │ ├── public/ │ │ └── index.html # Main HTML page │ │ └── ... # Other static assets │ ├── src/ │ │ ├── components/ # React components (FileUpload, ColumnSelector, etc.) │ │ ├── services/ # API service (apiService.js) │ │ ├── App.css │ │ ├── App.js # Main React application component │ │ ├── index.css │ │ └── index.js # React entry point │ └── package.json # Frontend dependencies and scripts └── README.md # This file


## Backend (FastAPI)

The backend is built using Python and FastAPI.

### Setup and Running

1.  **Navigate to the backend directory:**
    ```bash
    cd finops_forecast_app/backend
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `prophet` and `tensorflow` can sometimes be tricky. If `pip` fails, you might need to consult their specific installation guides, possibly using `conda` or checking for system dependencies.*

4.  **Run the FastAPI server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    The server will typically start on `http://127.0.0.1:8000`.

## Frontend (React)

The frontend is built using React.

### Setup and Running

1.  **Navigate to the frontend directory:**
    ```bash
    cd finops_forecast_app/frontend
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Run the React development server:**
    ```bash
    npm start
    ```
    The application will typically open in your browser at `http://localhost:3000`. API requests from the frontend are proxied to the backend at `http://localhost:8000` (as configured in `frontend/package.json`).

## Running with Docker Compose (Recommended for combined testing)

This is the easiest way to run both backend and frontend together after initial setup.

1.  **Ensure Docker and Docker Compose are installed.**
2.  **Navigate to the project root directory** (the one containing `docker-compose.yml` and the `finops_forecast_app` directory).
3.  **Build and start the services:**
    ```bash
    docker-compose up --build
    ```
    *   The backend will be accessible at `http://localhost:8000`.
    *   The frontend will be accessible at `http://localhost:3000`.

4.  **To stop the services:**
    ```bash
    docker-compose down
    ```

## Basic Usage Workflow

1.  Open the web application in your browser (usually `http://localhost:3000` if running via Docker Compose or `npm start`).
2.  **Upload Data:** Use the "Upload FinOps Data" section to upload your CSV cost file.
3.  **Configure Columns & Frequency:** Once the file is uploaded and columns are parsed, select:
    *   The date column in your dataset.
    *   The target column representing costs.
    *   Optionally, columns to group by (e.g., Resource Group, Subscription).
    *   The desired time series frequency (Daily, Weekly, Monthly).
4.  **Select Models:** Choose one or more forecasting models (ARIMA, Prophet, Exponential Smoothing, LSTM).
5.  **Preprocess Data:** Click the "Preprocess Data" button. This cleans the data, handles missing values/outliers, and resamples it according to your configuration for each group.
6.  **Run Forecasting:** After preprocessing is complete, click the "Run Forecasting" button. The selected models will be trained, forecasts generated, and performance metrics calculated.
7.  **View Results:** The results, including metrics tables and forecast charts, will be displayed for each group and model.
To run the web app (summary):

Method 1: Using Docker Compose (Recommended for ease)

Make sure Docker and Docker Compose are installed.
Navigate to the root directory of the project (where docker-compose.yml is located).
Run docker-compose up --build.
Open your browser and go to http://localhost:3000 for the frontend.
Method 2: Running Backend and Frontend separately

For the Backend:
cd finops_forecast_app/backend
Set up a Python virtual environment and pip install -r requirements.txt.
Run uvicorn app.main:app --reload.
For the Frontend:
cd finops_forecast_app/frontend
npm install
npm start.
Open your browser to http://localhost:3000.
