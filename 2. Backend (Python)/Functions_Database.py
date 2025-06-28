import os
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import random

import warnings
warnings.simplefilter("ignore")

# ------------------------- Crear base de datos --------------------------------------------------

def excels_to_database(processed_data_folder_path):
    headers_global = None
    all_global_data = []
    
    files = [f for f in os.listdir(processed_data_folder_path) if f.endswith('.xlsx')]
    for file_name in files:
        current_file_path = os.path.join(processed_data_folder_path, file_name)
                
        # --- Leer y almacenar los datos ---
        global_sheet = pd.read_excel(current_file_path, sheet_name=0, header=0, engine='openpyxl')
        global_values = global_sheet.copy()
        all_global_data.extend(global_values.values.tolist())

        # --- Leer y almacenar los encabezados ---
        if headers_global is None:
            headers_global = global_sheet.columns.tolist()

    global_df = pd.DataFrame(all_global_data, columns=headers_global) if all_global_data else pd.DataFrame()
    return global_df

def filter_database(global_df):
    # Filtrar solo actividades tipo 'Trail' o 'Road'
    global_df = global_df[global_df['activity_type'].isin(['Trail', 'Road'])]

    return global_df

def order_and_aggregate(global_df):
    # Conversión de fechas
    global_df['date'] = pd.to_datetime(global_df['activity_date'], format="%d/%m/%Y", errors='coerce')
    # Rango completo de fechas desde la primera actividad hasta hoy
    start_date = global_df['date'].min()
    end_date = datetime.today()
    full_range = pd.date_range(start=start_date, end=end_date)
    # DataFrame con todos los días
    df_full = pd.merge(
        pd.DataFrame({'date': full_range}), global_df, on='date', how='left'
    )

    # Definir funciones de agregación para cada columna
    agg_map = {
        'activity_type': 'first',
        'activity_city': 'first',
        'activity_date': 'first',
        'inicial_time': 'first',
        'finish_time': 'first',
        'activity_season': 'first',

        'total_time': 'sum',
        'total_distance': 'sum',
        'equivalence_distance': 'sum',
        'total_accumulated_positive_level': 'sum',
        'total_accumulated_negative_level': 'sum',

        'moving_average_pace': lambda s: (s * df_full.loc[s.index, 'total_time']).sum() / df_full.loc[s.index, 'total_time'].sum() if df_full.loc[s.index, 'total_time'].sum() > 0 else pd.NA,
        'real_average_pace': lambda s: (s * df_full.loc[s.index, 'total_time']).sum() / df_full.loc[s.index, 'total_time'].sum() if df_full.loc[s.index, 'total_time'].sum() > 0 else pd.NA,
        'average_rap': lambda s: (s * df_full.loc[s.index, 'total_time']).sum() / df_full.loc[s.index, 'total_time'].sum() if df_full.loc[s.index, 'total_time'].sum() > 0 else pd.NA,
        'deviation_pace': 'mean',

        'average_bpm': lambda s: (s * df_full.loc[s.index, 'total_time']).sum() / df_full.loc[s.index, 'total_time'].sum() if df_full.loc[s.index, 'total_time'].sum() > 0 else pd.NA,
        'max_bpm': 'max',
        'deviation_bpm': 'mean',
        'average_bpmxrap': 'mean',

        'average_cadence': 'mean',
        'average_stride': 'mean',

        'total_energy': 'sum',
        'average_power': 'mean',
        'standardized_power': 'mean',
        'training_load': 'sum',

        'vdot': 'mean',
        'PI': 'mean',
        'intensity_factor': 'mean',
        'average_VAM': 'mean',
        'energy_km': 'mean',

        'TSS_power': 'sum',
        'TSS_bpm': 'sum',
        'FTP_bpm': 'first',
        'FTP_rap': 'first',
        'FTP_power': 'first',

        'inicial_long': 'first',
        'inicial_lat': 'first',
        'weight': 'mean',

        **{f'time_z{i}_bpm': 'sum' for i in range(1,6)},
        **{f'time_z{i}_pace': 'sum' for i in range(1,6)},
        **{f'limit_bpm_z{i}': 'first' for i in range(1,5)},
        **{f'limit_pace_z{i}': 'first' for i in range(1,5)},

        **{col: 'sum' for col in df_full.columns if col.startswith('ritmo_')},
        **{col: 'sum' for col in df_full.columns if col.startswith('potencia_')},

        '3km_daily_mark': 'min',
        '5km_daily_mark': 'min',
        '10km_daily_mark': 'min',
        '21_1km_daily_mark': 'min',
        '42_2km_daily_mark': 'min',

        'activity_summary': 'first'
    }

    # Agregar
    aggregated = df_full.groupby('date').agg(agg_map).reset_index()

    # Reordenar columnas en el orden original proporcionado
    original_order = [
        'date','activity_type','activity_city','activity_date','inicial_time','finish_time','activity_season',
        'total_time','total_distance','equivalence_distance','total_accumulated_positive_level',
        'total_accumulated_negative_level','moving_average_pace','real_average_pace','average_rap',
        'deviation_pace','average_bpm','max_bpm','deviation_bpm','average_bpmxrap','average_cadence',
        'average_stride','total_energy','average_power','standardized_power','training_load','vdot',
        'PI','intensity_factor','average_VAM','energy_km','TSS_power','TSS_bpm','FTP_bpm','FTP_rap',
        'FTP_power','inicial_long','inicial_lat','weight',
        *[f'time_z{i}_bpm' for i in range(1,6)], *[f'time_z{i}_pace' for i in range(1,6)],
        *[f'limit_bpm_z{i}' for i in range(1,5)], *[f'limit_pace_z{i}' for i in range(1,5)],
        *[col for col in aggregated.columns if col.startswith('ritmo_')],
        *[col for col in aggregated.columns if col.startswith('potencia_')],
        '3km_daily_mark','5km_daily_mark','10km_daily_mark','21_1km_daily_mark',
        '42_2km_daily_mark','activity_summary'
    ]
    # Filtrar por original_order para mantener solo esas columnas
    return aggregated.reindex(columns=original_order)

# -------------------------  Predicción de marcas (VDOT) -----------------------------------------

def race_prediction_vdot(global_df):
    # --- Tabla VDOT y distancias objetivo ---
    vdot_data = {
        'VDOT': [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62],
        '1500': [8.5, 8.033, 7.617, 7.233, 6.9, 6.583, 6.317, 6.05, 5.817, 5.5, 5.4, 5.217, 5.033, 4.883, 4.733, 4.583, 4.45],
        '3000': [17.933, 16.983, 16.15, 15.383, 14.683, 14.05, 13.467, 12.917, 12.433, 11.75, 11.55, 11.15, 10.783, 10.45, 10.133, 9.833, 9.55],
        '5000': [30.667, 29.083, 27.65, 26.367, 25.2, 24.133, 23.15, 22.25, 21.417, 20.3, 19.95, 19.283, 18.667, 18.083, 17.55, 17.05, 16.567],
        '10000': [63.767, 60.433, 57.433, 54.733, 52.283, 50.05, 48.017, 46.15, 44.417, 42.067, 41.35, 39.983, 38.7, 37.517, 36.4, 35.367, 34.383],
        '21100': [141.067, 133.817, 127.267, 121.317, 115.917, 110.983, 106.45, 102.283, 98.45, 93.2, 91.583, 88.517, 85.667, 83, 80.5, 78.15, 75.57],
        '42200': [289.283, 274.983, 262.05, 250.317, 239.583, 229.75, 220.717, 212.383, 204.65, 194.1, 190.817, 184.6, 178.783, 173.333, 168.233, 163.417, 158.9],}
    
    vdot_table = pd.DataFrame(vdot_data)
    vdot_table = vdot_table.sort_values('VDOT')
    target_distances = {
        '3km_PB_VDOT': '3000',
        '5km_PB_VDOT': '5000',
        '10km_PB_VDOT': '10000',
        '21.1km_PB_VDOT': '21100',
        '42.2km_PB_VDOT': '42200'}
    
    # --- Calcula el VDOT máximo del último año móvil e interpola distancias equivalentes en nuevas columnas ---
    global_df['vdot_rolling_max'] = global_df['vdot'].rolling(window=365, min_periods=1).max()
    for pb_news_col, table_col in target_distances.items():
        global_df[pb_news_col] = np.interp(global_df['vdot_rolling_max'], vdot_table['VDOT'], vdot_table[table_col])
    
    # --- Elimina la columna VDOT_max, no interesa ---
    global_df.drop(columns=['vdot_rolling_max'], inplace=True)
    return global_df

# -------------------------  Predicción de marcas (ML) -------------------------------------------

def race_prediction_ml(global_df, block_size):

    # --- Columnas a predecir y a considerar ---
    target_columns = ['3km_daily_mark', '5km_daily_mark', '10km_daily_mark', '21_1km_daily_mark', '42_2km_daily_mark']
    features = ['total_distance', 'total_time', 
                'moving_average_pace', 'average_rap', 'average_bpm', 'max_bpm', 
                'average_cadence', 'average_stride', 
                'average_power', 'standardized_power', 
                'training_load', 'vdot', 'PI', 
                'time_z1_pace', 'time_z2_pace', 'time_z3_pace', 'time_z4_pace', 'time_z5_pace']
    
    # ---  Crear la columna de predicción para cada distancia objetivo ---
    for target in target_columns:
        pb_ml_estimation_col = pd.Series(np.nan, index=global_df.index)
        
        # ---  Cálculo de predicción cada block_size días ---
        n_rows = len(global_df)
        for start in range(0, n_rows, block_size):
            end = min(start + block_size, n_rows)
            cutoff_date = global_df.loc[end - 1, "date"]
            train_data = global_df[global_df["date"] <= cutoff_date]
            train_data = train_data.dropna(subset = features + [target])
            
            if train_data.empty:
                potential_best = np.nan
            else:
                X_train = train_data[features]
                y_train = train_data[target]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_train)
                potential_best = np.min(preds)
            
            pb_ml_estimation_col.iloc[start:end] = potential_best
        
        global_df[f'{target.split("_")[0]}_PB_ML'] = pb_ml_estimation_col

    return global_df

# ------------------------- Métricas de carga de entrenamiento -----------------------------------

def calculate_training_metrics(global_df):
    global_df['training_load'] = global_df['training_load'].fillna(0)
    global_df['fitness'] = global_df['training_load'].rolling(window=42, min_periods=1).mean()
    global_df['fatigue'] = global_df['training_load'].rolling(window=7, min_periods=1).mean()
    global_df['forma'] = global_df['fitness'].shift(1) - global_df['fatigue']
    return global_df

# -------------------------  Running Performance Score (RPS)  ------------------------------------

def add_rps_score(global_df):
    # 1. Definir variables de entrada
    features = [
        'total_time',
        'total_distance',
        'total_accumulated_positive_level',
        'moving_average_pace',
        'real_average_pace',
        'average_rap',
        'deviation_pace',
        'average_bpm',
        'max_bpm',
        'deviation_bpm'
    ]
    
    # 2. Filtrar filas con NaNs en features para el entrenamiento
    df_train = global_df[features].dropna()
    
    # 3. Crear seed score (rps_seed) con peso igual
    n = len(features)
    df_train = df_train.assign(
        rps_seed = df_train[features].sum(axis=1) / n
    )
    
    # 4. Preparar X e y
    X = df_train[features]
    y = df_train['rps_seed']
    
    # 5. Normalizar
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 6. Dividir train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 7. Entrenar modelo
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 8. Predecir RPS sobre TODO el DataFrame (rellenar NaNs con la media de cada feature)
    X_full = global_df[features].copy()
    X_full = X_full.fillna(X_full.mean())
    X_full_scaled = scaler.transform(X_full)
    global_df['RPS'] = model.predict(X_full_scaled)
    
    return global_df

# ------------------------- Detección de anomalías -----------------------------------------------

def detect_training_anomalies(global_df):

    # --- DataFrame solo con las columnas a considerar ---
    features = ['total_time','total_distance', 
                'average_rap', 'average_bpm', 'deviation_pace', 'deviation_bpm',
                'standardized_power', 'training_load', 
                'vdot', 'fitness', 'fatigue', 'forma']
    features = [col for col in features if col in global_df.columns]
    if not features: return global_df

    df_features = global_df[features].copy()
    df_features.fillna(df_features.mean(), inplace=True)
    
    # --- Escalar los datos para que tengan media 0 y varianza 1 ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    
    # --- Ejecutar el modelo Isolation Forest ---
    model = IsolationForest(contamination=0.05, random_state=42) # La contaminación es el % de valores anómalos a esperar
    preds = model.fit_predict(X_scaled)
    global_df['anomaly'] = (preds == -1).astype(int) # 1 = anomalía
    
    return global_df

# -------------------------  Clustering ----------------------------------------------------------

def clustering_type_activity(date_global, optimal_clusters_road, optimal_clusters_trail):
    date_global['inicial_time'] = pd.to_datetime(date_global['inicial_time'])

    # --- Columnas y período de entrenamiento ---
    features = ["activity_type", 
                "total_time", "total_distance", "total_accumulated_positive_level", 
                "moving_average_pace", "average_rap", "deviation_pace",
                "average_bpm", "max_bpm", "deviation_bpm", 
                "total_energy", "training_load", "vdot"]
    date_global_pre_2024 = date_global[date_global["date"] < "2025-01-01"].copy()
    database_global = date_global_pre_2024[features]
    
    # --- Diferenciar clusters de tipo 'Road' y 'Trail' ---
    database_global_road = database_global[database_global['activity_type'] == 'Road'].drop(columns=['activity_type'])
    database_global_trail = database_global[database_global['activity_type'] == 'Trail'].drop(columns=['activity_type'])

    # ---  Entrenar clustering para 'Road' y 'Trail' de forma independiente
    def apply_clustering(data, optimal_clusters):
        if len(data) <= optimal_clusters: print(f"Not enough data for {optimal_clusters} clusters."); return None, None, None

        # --- Preparar datos ---
        imputer = SimpleImputer(strategy='mean')  
        numeric_data = data.select_dtypes(include=['number'])
        numeric_data_imputed = imputer.fit_transform(numeric_data)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_data_imputed)

        # --- Clusterizar con KMeans ---
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=123)
        clusters = kmeans.fit_predict(scaled_data)
        data['Cluster'] = clusters

        cluster_summary = data.groupby('Cluster').mean(numeric_only=False).T
        return data, cluster_summary, (imputer, scaler, kmeans)
    
    road_clusters, road_summary, road_models = apply_clustering(database_global_road, optimal_clusters_road)
    trail_clusters, trail_summary, trail_models = apply_clustering(database_global_trail, optimal_clusters_trail)
    
    # --- Assignar clusters a todos los datos ---
    def assign_clusters_to_new_data(new_data, road_models, trail_models):
        if new_data.empty: return new_data
        new_data_with_clusters = new_data.copy()

        # --- Asignación de clusters a 'Road' ---
        road_data = new_data[new_data["activity_type"] == "Road"].copy()
        if road_models is not None and not road_data.empty:
            road_imputer, road_scaler, road_kmeans = road_models
            expected_features = road_imputer.feature_names_in_

            for col in expected_features:
                if col not in road_data.columns: road_data[col] = np.nan

            road_data = road_data[expected_features]

            road_data_imputed = road_imputer.transform(road_data)
            road_data_scaled = road_scaler.transform(road_data_imputed)

            road_clusters_new = road_kmeans.predict(road_data_scaled)
            new_data_with_clusters.loc[road_data.index, "road_cluster"] = road_clusters_new

        # --- Asignación de clusters a 'Trail' ---
        trail_data = new_data[new_data["activity_type"] == "Trail"].copy()
        if trail_models is not None and not trail_data.empty:
            trail_imputer, trail_scaler, trail_kmeans = trail_models
            expected_features = trail_imputer.feature_names_in_

            for col in expected_features:
                if col not in trail_data.columns: trail_data[col] = np.nan

            trail_data = trail_data[expected_features]

            trail_data_imputed = trail_imputer.transform(trail_data)
            trail_data_scaled = trail_scaler.transform(trail_data_imputed)

            trail_clusters_new = trail_kmeans.predict(trail_data_scaled)
            new_data_with_clusters.loc[trail_data.index, "trail_cluster"] = trail_clusters_new

        return new_data_with_clusters
    global_with_clusters = assign_clusters_to_new_data(date_global, road_models, trail_models)
    
    return global_with_clusters, road_summary, trail_summary

# -------------------------  Entrenamiento recomendado -------------------------------------------

def training_recommendation(global_df,road_cluster_dict, trail_cluster_dict):
    # --- Definir entreno de hoy y los de la última semana ---
    def get_training_type(row):
        if not pd.isna(row['road_cluster']):
            return road_cluster_dict.get(row['road_cluster'], "Entreno asfalto")
        elif not pd.isna(row['trail_cluster']):
            return trail_cluster_dict.get(row['trail_cluster'], "Entreno trail")
        else:
            return "Día de descanso"
    global_df['training_type'] = global_df.apply(get_training_type, axis=1)

    n_days_ago = 8 # Mínimo 7 para tener la semana de entrenos
    for i in range(1, n_days_ago): global_df[f'training_{i}_days_ago'] = global_df['training_type'].shift(i)

    # --- Recomendar el próximo entrenamiento ---
    def recommend_next_training(row):
        # --- Listas de referencia para los distintos tipos de entrenamientos ---
        random_road_options = ["Rodaje recuperativo", "Rodaje", "Rodaje extenso", "Entreno umbral", "Intervalos"]
        random_trail_options = ["Rodaje corto (trail)", "Rodaje (trail)", "Rodaje extenso (trail)", "Long run (trail)", "Ritmo intenso (trail)"]
        random_options = 2 * random_road_options + random_trail_options  # Ponderado, doble probabilidad asfalto

        recovery_runs = ["Rodaje recuperativo", "Rodaje"]
        easy_short_runs = ["Rodaje recuperativo", "Rodaje", "Rodaje corto (trail)", "Rodaje (trail)"]

        easy_road_runs = ["Rodaje recuperativo", "Rodaje", "Rodaje extenso"]
        easy_trail_runs = ["Rodaje corto (trail)", "Rodaje (trail)", "Rodaje extenso (trail)"]
        easy_runs = 2 * easy_road_runs + easy_trail_runs  # Ponderado, doble probabilidad asfalto
        easy_runs_y_descanso = easy_runs + ["Día de descanso"]

        trail_workouts = random_trail_options + ["Competicion (trail)"]
        long_trail_workouts = ["Rodaje extenso (trail)", "Long run (trail)"]
        long_workouts = long_trail_workouts + 2*["Rodaje extenso"]

        intense_road_workouts = ["Entreno umbral", "Intervalos"]
        intense_trail_workouts = ["Ritmo intenso (trail)"]
        intense_workouts = intense_road_workouts + intense_trail_workouts

        competicions_workouts = ["Competicion (trail)", "Entreno maratoniano", "Carrera media distancia"]
        hard_workouts = competicions_workouts + intense_workouts

        # --- Extraer entrenamientos de los registros ---
        today = row["training_type"]
        yesterday = row.get("training_1_days_ago", None)
        day_before_yesterday  = row.get("training_2_days_ago", None)
        two_days_before_yesterday  = row.get("training_3_days_ago", None)
        four_day = [two_days_before_yesterday , day_before_yesterday , yesterday, today]
        last_week = [today, yesterday, day_before_yesterday , two_days_before_yesterday , 
            row.get("training_4_days_ago", None), row.get("training_5_days_ago", None), row.get("training_6_days_ago", None)]
        
        # --- Condiciones para definir el entreno ---
        
        # 0. Condicion de recuperacion basica
        if not any(t == "Día de descanso" for t in last_week if t):
            # Si no hubo ningún "Día de descanso" en la última semana, se recomienda un "Día de descanso".
            return "Día de descanso"
        
        # 1. Condiciones basadas en el entrenamiento de HOY
        if today in hard_workouts:
            # Si hoy fue un entrenamiento duro, se recomienda recuperación.
            return random.choice(recovery_runs)
        
        if today in long_trail_workouts:
            # Si hoy fue "Rodaje extenso (trail)" o "Long run (trail)", se recomienda un entrenamiento fácil de carretera.
            return random.choice(easy_road_runs)
        
        # 2. Condiciones basadas en AYER
        if yesterday in competicions_workouts:
            # Si ayer fue competición, recomendar recuperación.
            return random.choice(recovery_runs)
        
        if yesterday in intense_workouts:
            # Si ayer fue un entrenamiento intenso:
            if today not in trail_workouts:
                # y hoy no es de trail, se recomienda un entrenamiento fácil (por ejemplo, en asfalto).
                return random.choice(easy_runs)
            else:
                # Si hoy es de trail, se recomienda recuperación.
                return random.choice(recovery_runs)
        
        if yesterday in long_workouts and today in easy_short_runs:
            # Si ayer fue largo y hoy es una sesión corta fácil, y además en la última semana hubo menos de 2 entrenos duros,
            # se recomienda un entrenamiento intenso.
            if sum(1 for t in last_week if t in hard_workouts) < 2:
                return random.choice(intense_workouts)
        
        if yesterday in easy_short_runs and today in easy_short_runs:
            # Si ayer y hoy fueron sesiones fáciles cortas, y la cantidad de entrenos duros en la última semana es baja,
            # se recomienda un entrenamiento intenso.
            if sum(1 for t in last_week if t in hard_workouts) < 2:
                return random.choice(intense_workouts)
        
        # 3. Condiciones basadas en días anteriores
        if day_before_yesterday  in competicions_workouts:
            # Si anteayer fue competición, se recomienda una sesión fácil.
            return random.choice(easy_runs)
        
        # Condiciones evaluadas en conjunto para los últimos 4 días (anteanteayer, anteayer, ayer y hoy)
        if all(t in easy_runs_y_descanso for t in four_day if t):
            # Si todos fueron entrenamientos fáciles, y además hay menos de 2 entrenos duros en la última semana,
            # se recomienda un entrenamiento intenso.
            if sum(1 for t in last_week if t in hard_workouts) < 2:
                return random.choice(intense_workouts)
        
        if all(t not in hard_workouts for t in four_day if t):
            # Si en los últimos 4 días no hubo entrenamientos duros, y la cuenta de duros en la semana es baja,
            # se recomienda un entrenamiento intenso.
            if sum(1 for t in last_week if t in hard_workouts) < 2:
                return random.choice(intense_workouts)
        
        if sum(1 for t in four_day if t in (hard_workouts + long_workouts)) > 2:
            # Si en los últimos 4 días hubo más de 2 entrenos duros o largos, se recomienda un entrenamiento fácil.
            return random.choice(easy_runs)
        
        # 4. Condiciones basadas en el historial de la última semana (incluyendo hoy)
        if not any(t in trail_workouts for t in last_week if t) and not any(t in long_workouts for t in last_week if t):
            # Si en la última semana (y hoy) no hubo ni entrenos de trail ni largos, se recomienda "Rodaje extenso (trail)" o "Long run (trail)".
            return random.choice(long_trail_workouts)
        
        if any(t in trail_workouts for t in last_week if t) and not any(t in long_workouts for t in last_week if t):
            # Si hubo entrenos de trail pero no largos, se recomienda un entrenamiento largo.
            return random.choice(long_workouts)
        
        if any(t in long_workouts for t in last_week if t) and not any(t in trail_workouts for t in last_week if t):
            # Si hubo entrenos largos pero no de trail, se recomienda un entrenamiento de trail.
            return random.choice(trail_workouts)
        
        if sum(1 for t in last_week if t in hard_workouts) < 2:
            # Si en la última semana hubo menos de 2 entrenos duros, se recomienda un entrenamiento intenso.
            return random.choice(intense_workouts)
        
        # 5. Valor por defecto
        return random.choice(random_options)
    global_df['training_recommendation'] = global_df.apply(recommend_next_training, axis=1)

    global_df = global_df.drop(columns=[f'training_{i}_days_ago' for i in range(1, n_days_ago)])
    return global_df

# -------------------------  Guardar datos -------------------------------------------------------

def save_bigdata_excel(global_df, road_summary, trail_summary, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    global_file = os.path.join(output_dir, "global.xlsx")
    global_df.to_excel(global_file, sheet_name="Global", index=False)

    cluster_file = os.path.join(output_dir, "cluster_summary.xlsx")
    with pd.ExcelWriter(cluster_file, engine="xlsxwriter") as writer:
        road_summary.to_excel(writer, sheet_name="Road_Clusters", index=True)
        trail_summary.to_excel(writer, sheet_name="Trail_Clusters", index=True)