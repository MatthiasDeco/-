# -
Developed a Python-based system to analyze running training data from Garmin (.fit files), generating advanced performance metrics and visualizations. Applied machine learning for clustering, regression, and anomaly detection to optimize training insights and support athletes in improving performance sustainably.

This looks like a well-organized project\! Let's get that `README.txt` amplified and translated into clear, concise English.

Here's an improved version for your `README.txt`, incorporating the details you provided and making it more user-friendly:

-----

## RunFlow: Your Activity Data Processor

Welcome to RunFlow\! This program helps you process your activity data, particularly from Strava, and generate insightful reports.

### Table of Contents

1.  **System Requirements**
2.  **Initial Setup & First Run (Processing Strava Data)**
3.  **Processing New Activities**
4.  **Project Structure (for Developers)**

-----

### 1\. System Requirements

To run RunFlow, you need:

  * **Python 3** (or a compatible version).
  * All libraries listed in `requirements.txt`. Install them using:
    ```bash
    pip install -r requirements.txt
    ```
  * A compatible version of `deepseek-r1`. Please refer to the `deepseek-r1` documentation for installation instructions.

-----

### 2\. Initial Setup & First Run (Processing Strava Data)

This section guides you through the initial setup and processing of your historical Strava data.

1.  **Download Your Strava Data**:
      * Log in to your Strava account.
      * Navigate to **Settings** \> **Account** \> **Download or Delete Your Account**.
      * Request an export of your data. Strava will send you an email with a download link.
2.  **Prepare Your Data**:
      * Download the `.zip` file from the Strava email.
      * **Rename this downloaded `.zip` file to `strava.zip`**. This is crucial for the processor to find it.
3.  **Place the Data**:
      * Move the renamed `strava.zip` file into the `1. Procesador strava (Python)` directory.
4.  **Generate Input Files**:
      * Open your terminal or command prompt.
      * Navigate to the `1. Procesador strava (Python)` directory.
      * Execute the `strava_to_input.py` script:
        ```bash
        python strava_to_input.py
        ```
      * This script will extract all `.fit` files from `strava.zip` and place them into a newly created `1.Input` folder within the `1. Procesador strava (Python)` directory. Once completed, only the `1.Input` folder (containing the `.fit` files) will remain.
5.  **Process Your Data**:
      * Move the entire `1.Input` folder from `1. Procesador strava (Python)` into the `2. Backend (Python)` directory.
      * Ensure that the `2. Backend (Python)` directory contains:
          * The `1.Input` folder (with your `.fit` files).
          * `Functions_Database.py`
          * `Functions_Individual.py`
          * `RunFlow_processor.py`
      * From your terminal or command prompt, navigate to the `2. Backend (Python)` directory.
      * Execute the `RunFlow_processor.py` script:
        ```bash
        python RunFlow_processor.py
        ```
      * **Note**: The initial processing of a large dataset can take several hours. It's recommended to let the script run and attend to other tasks. Upon completion, `2.Output` and `__pycache__` folders will be created within the `2. Backend (Python)` directory, containing your processed data.

-----

### 3\. Processing New Activities

To update your data with new activities, you have two primary methods:

1.  **Downloading Individual Activities from Strava**:
      * For each new activity, download its `.zip` file from Strava (usually an option on the activity page itself).
      * Extract the `.fit` file from the downloaded `.zip`.
      * Place the extracted `.fit` file directly into the `1.Input` folder located within the `2. Backend (Python)` directory.
2.  **Manual Extraction from Your Device**:
      * Many GPS watches and devices allow you to manually extract `.fit` files when connected to your computer. Refer to your device's instructions for this process.
      * Place these `.fit` files directly into the `1.Input` folder located within the `2. Backend (Python)` directory.

Once you have added the new `.fit` files to the `1.Input` folder (inside `2. Backend (Python)`), simply re-run the `RunFlow_processor.py` script:

```bash
python RunFlow_processor.py
```

This will process the new activities, update the general data, and generate new Excel files for each added activity within the `2.Output/Processed_data` folder.

-----

### 4\. Project Structure (for Developers)

This section outlines the key directories and files within the RunFlow project. Please note that the `0. Otros` (Other) directory is excluded from deployment.

```
C:.
|   index.html
|   informes.html
|   mis actividades.html
|   RunFlow.py
|
+---1. Procesador strava (Python)
|   |   Read me.txt                 (This file)
|   |   requirements.txt            (Python dependencies)
|   |   strava.zip                  (Placeholder for your downloaded Strava data)
|   |   strava_to_input.py          (Script to convert Strava zip to .fit inputs)
|
\---2. Backend (Python)
    |   Functions_Database.py       (Functions for database operations)
    |   Functions_Individual.py     (Functions for individual activity processing)
    |   RunFlow_processor.py        (Main script for data processing)
    |
    +---1.Input
    |       230104.fit              (Example .fit input files)
    |       ...
    |       250503.fit
    |
    +---2.Output
    |   |   processed_files.txt     (Log of processed files)
    |   |
    |   +---Big_data
    |   |       cluster_summary.xlsx  (Summarized data)
    |   |       global.xlsx             (Global processed data)
    |   |
    |   \---Processed_data
    |           23_01_04_10_20.xlsx   (Individual processed activity data)
    |           ...
    |           25_05_03_07_59.xlsx
    |
    \---__pycache__                 (Python compiled bytecode cache)
            Functions_Database.cpython-313.pyc
            Functions_Individual.cpython-313.pyc
