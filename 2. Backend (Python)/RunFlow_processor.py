import asyncio
import os
import pandas as pd
from Functions_Individual import determine_cst, convert_fit_to_rawdata, raw_to_processed, lap_generator, summary_data 
from Functions_Individual import zone_calc, ratios_bpm, daily_times_calc, add_deepseek_summary, save_individual_excel, save_one_excel
from Functions_Database import excels_to_database, order_and_aggregate, filter_database, calculate_training_metrics, clustering_type_activity
from Functions_Database import race_prediction_vdot, race_prediction_ml, detect_training_anomalies, training_recommendation, add_rps_score, save_bigdata_excel
import webbrowser

# ------------------------- 1ra parte - Función individual ---------------------------------------

async def first_part():
    print("----- Starting first part of the program: Processing data recorded -----")

    # --- Constantes (actualizar cada vez que se estime un nuevo umbral láctico) ---
    switch_FTP_2023_2024, switch_FTP_2024_2025 = 295, 599
    weight_2023, FTP_bpm_2023, FTP_rap_2023 = 63.5, 196, 4.2
    weight_2024, FTP_bpm_2024, FTP_rap_2024 = 62.5, 196, 3.8
    weight_2025, FTP_bpm_2025, FTP_rap_2025 = 64, 196, 3.867
    running_efficiency = 0.25  # Rendimiento de un corredor biologicamente
    PI_referencia = FTP_bpm_2023 * FTP_rap_2023 # BPMxritmo primera carrera, una media maraton (parecido al lactate threshold)
    cst_raw_to_processed = {
        "road_running_limit_pace": 7.5,         # Límite de ritmo para Road [min/km]
        "trail_running_limit_pace": 20,         # Límite de ritmo para Trail [min/km]
        "n_level": 10,                          # Intervalo para calcular el desnivel [s]
        "margin_error_lower_level": 0.5,        # Margen de error inferior [m]
        "beach_altitud": 15,                    # Altitud límite para considerar "paseo marítimo" [m]
        "margin_error_upper_level": 20,         # Margen de error superior (para saltos anómalos) [m]
        "running_efficiency": running_efficiency, # [ud]
        "distance_error": 0                     # Contador de archivos con error de distancia [ud]
    }
    
    # --- Rutas de archivos (relativas al programa) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_folder = os.path.join(script_dir, '1.Input')
    if not os.path.exists(input_folder):
        print('Error: Input folder does not exist.')
        exit(1)

    register_file_path = os.path.join(script_dir, '2.Output', 'processed_files.txt')
    processed_data_folder_path = os.path.join(script_dir, '2.Output', 'Processed_data')
    os.makedirs(processed_data_folder_path, exist_ok=True)

    # --- Lectura y procesamiento de los archivos individuales ---
    processed_files = set() 
    if os.path.exists(register_file_path):
        with open(register_file_path, 'r') as f:
            processed_files = {line.strip() for line in f if line.strip()}

    file_counter = len(processed_files)
    # Buscar archivos .fit en input_folder y subcarpetas
    fit_files = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.fit'):
                fit_files.append(os.path.join(root, file))

    for filepath in fit_files:
        filename = os.path.basename(filepath)  # "2025-05-08-07-50-47.fit"
        activity_name = os.path.splitext(filename)[0]
        if activity_name in processed_files:
            continue

        FTP_bpm, FTP_rap, FTP_pace, weight = determine_cst(file_counter, switch_FTP_2023_2024, switch_FTP_2024_2025, FTP_bpm_2023, 
            FTP_rap_2023, weight_2023, FTP_bpm_2024, FTP_rap_2024, weight_2024, FTP_bpm_2025, FTP_rap_2025, weight_2025)
        
        required_columns = ['timestamp', 'position_lat', 'position_long', 'distance', 'enhanced_speed', 
                            'enhanced_altitude', 'heart_rate', 'cadence', 'fractional_cadence']
        raw_data = convert_fit_to_rawdata(filepath, required_columns)
        # Verificar columnas mínimas
        missing_columns = [col for col in required_columns if col not in raw_data.columns]
        if missing_columns:
            print(f"Error: The following columns are missing in the file {filename}: {missing_columns}. Skipping...")
            
            # Registrar archivo omitido
            with open(register_file_path, 'a') as f:
                f.write(f"{activity_name}\n")
            processed_files.add(activity_name)
            
            continue

        activity_name, processed_data, MET_model, activity_type, activity_date, inicial_time, finish_time = raw_to_processed(raw_data, weight, cst_raw_to_processed)

        intervals = [10, 60, 180, 300, 600]
        all_laps = pd.concat([lap_generator(interval, processed_data).assign(id_lap=f'min{interval//60}_lap') for interval in intervals], ignore_index=True)
        all_laps["activity_type"] = activity_type
        
        cst_summary_data = {
            "weight": weight,
            "FTP_bpm": FTP_bpm,
            "FTP_rap": FTP_rap,
            "MET_model": MET_model,
            "activity_date": activity_date,
            "inicial_time": inicial_time,
            "finish_time": finish_time,
            "PI_referencia": PI_referencia,
            "running_efficiency": running_efficiency,
            "activity_type": activity_type
        }
        summary = summary_data(processed_data, cst_summary_data)
        zones = zone_calc(FTP_bpm, FTP_pace, processed_data)
        ratios_lap = lap_generator(30, processed_data)
        ratios = ratios_bpm(ratios_lap)
        daily_mark = daily_times_calc(summary)
        pre_general_data = pd.concat([summary, zones, ratios, daily_mark], axis=1)
        general_data = await add_deepseek_summary(pre_general_data, model_version='1.5b') # 7b (o superior) da buenos resúmenes, pero es muy lento. 1.5b será mucho más rápido pero peor.

        save_individual_excel(activity_name, processed_data_folder_path, processed_data, general_data, all_laps, register_file_path, processed_files, filename)

        file_counter += 1
        progress_first_part = (file_counter / (len(fit_files)) * 100)
        print(f'- Progress first part: {progress_first_part:.2f} % - {activity_name}')

# ------------------------- 2da parte - Función base de datos ------------------------------------

def second_part():
    print("----- Starting second part of the program: Database processing -----")
    
    # --- Rutas de archivos (relativas al programa) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    processed_data_folder_path = os.path.join(script_dir, '2.Output', 'Processed_data')
    big_data_folder_path = os.path.join(script_dir, '2.Output', 'Big_data')
    
    # --- Creación y procesamiento de la base de datos ---
    print("- Uploading Database")
    global_df_original = excels_to_database(processed_data_folder_path)
    global_df_filtered = filter_database(global_df_original)
    global_df_v1 = order_and_aggregate(global_df_filtered)
    
    print("- Calculating race predictions")
    global_df_v2 = race_prediction_ml(global_df_v1, block_size=90)  # block_size=1 calculará la mejor marca posible de cada día y len(global_df_v1) calculará solo la marca actual. El primero
                                                                    # tardará horas, el segundo es rápido pero no permite ver históricos. Lo ideal es un intermedio, cada 90 días, por ejemplo.
    global_df_v3 = race_prediction_vdot(global_df_v2)

    print("- Calculating RPS & TrainingLoad Metrics")
    global_df_v4 = calculate_training_metrics(global_df_v3)
    global_df_v5 = add_rps_score(global_df_v4)

    print("- Looking for anomalis")
    global_df_v6 = detect_training_anomalies(global_df_v5)

    print("- Clustering the activities")
    global_df_v7, road_summary, trail_summary = clustering_type_activity(global_df_v6, optimal_clusters_road=8, optimal_clusters_trail=6) 
    # Hay que definir el número de clústeres. Ya se ha comprobado y lo ideal será separar 6 clústeres de asfalto y 4 clústeres de trail.

    print("- Thinking training recomendations")
    road_cluster_dict = { # Para definir los nombres de los clusters hay que determinar mediante road_summary, trail_summary que es cada cluster.
        0: "Rodaje",
        1: "Intervalos",
        2: "Entreno maratoniano",
        3: "Entreno umbral",
        4: "Rodaje extenso",
        5: "Carrera media distancia",
        6: "Rodaje recuperativo",
        7: "Entreno umbral"
    }
    trail_cluster_dict = {
        0: "Ritmo intenso (trail)",
        1: "Rodaje corto (trail)",
        2: "Rodaje extenso (trail)",
        3: "Competicion (trail)",
        4: "Rodaje (trail)",
        5: "Long run (trail)"
    }
    global_df_v8 = training_recommendation(global_df_v7, road_cluster_dict, trail_cluster_dict)
    
    print("- Saving excels")
    save_bigdata_excel(global_df_v8, road_summary, trail_summary, big_data_folder_path)

    print("----- This is the END my friend -----")

# ------------------------- Solo un archivo 1ra parte ------------------------------------

async def process_specific_activity():
    print("----- Opción 4: Procesar una actividad específica -----")
    file_path = input("Introduce el path completo al archivo .fit que quieres procesar: ").strip().strip('"')

    if not os.path.isfile(file_path) or not file_path.endswith(".fit"):
        print("Error: El archivo no existe o no es un archivo .fit válido.")
        return

    folder_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)

    # --- Constantes y configuraciones (igual que en first_part) ---
    try:
        FTP_bpm = int(input("Introduce tu FTP cardíaco (en bpm): (EJ:196)"))
        FTP_pace = float(input("Introduce tu FTP de ritmo (en min/km): (EJ:4.2) "))
        weight = float(input("Introduce tu peso (en kg): (EJ:64)"))
    except ValueError:
        print("Error: Asegúrate de introducir valores numéricos válidos.")
        return
    FTP_rap = FTP_pace
    running_efficiency = 0.25
    PI_referencia = FTP_bpm * FTP_pace
    cst_raw_to_processed = {
        "road_running_limit_pace": 7.5,
        "trail_running_limit_pace": 20,
        "n_level": 10,
        "margin_error_lower_level": 0.5,
        "beach_altitud": 15,
        "margin_error_upper_level": 20,
        "running_efficiency": running_efficiency,
        "distance_error": 0
    }

    print("\n--- Parámetros de procesamiento ---")
    for key, value in cst_raw_to_processed.items():
        print(f"{key}: {value}")

    confirm = input("\n¿Los parámetros anteriores son correctos? (s/n): ").strip().lower()
    if confirm != "s":
        print("Operación cancelada por el usuario. Revisa los parámetros en el código.")
        return
    
    required_columns = ['timestamp', 'position_lat', 'position_long', 'distance', 'enhanced_speed',
                        'enhanced_altitude', 'heart_rate', 'cadence', 'fractional_cadence']
    raw_data = convert_fit_to_rawdata(file_path, required_columns)
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        print(f"Error: Faltan las siguientes columnas en el archivo {filename}: {missing_columns}.")
        return

    activity_name, processed_data, MET_model, activity_type, activity_date, inicial_time, finish_time = raw_to_processed(
        raw_data, weight, cst_raw_to_processed)

    intervals = [10, 60, 180, 300, 600]
    all_laps = pd.concat([lap_generator(interval, processed_data).assign(id_lap=f'min{interval//60}_lap') for interval in intervals], ignore_index=True)
    all_laps["activity_type"] = activity_type

    cst_summary_data = {
        "weight": weight,
        "FTP_bpm": FTP_bpm,
        "FTP_rap": FTP_rap,
        "MET_model": MET_model,
        "activity_date": activity_date,
        "inicial_time": inicial_time,
        "finish_time": finish_time,
        "PI_referencia": PI_referencia,
        "running_efficiency": running_efficiency,
        "activity_type": activity_type
    }
    summary = summary_data(processed_data, cst_summary_data)
    zones = zone_calc(FTP_bpm, FTP_pace, processed_data)
    ratios_lap = lap_generator(30, processed_data)
    ratios = ratios_bpm(ratios_lap)
    daily_mark = daily_times_calc(summary)
    pre_general_data = pd.concat([summary, zones, ratios, daily_mark], axis=1)
    general_data = await add_deepseek_summary(pre_general_data, model_version='1.5b')

    save_one_excel(activity_name, folder_path, processed_data, general_data, all_laps)

    print("Actividad procesada y guardada correctamente.")

# ------------------------- Ejecución del programa por partes ------------------------------------

def main():
    print("Select an option to execute:")
    print("1. Process all individual activities")
    print("2. Process database")
    print("3. Run all")
    print("4. Process a specific activity")

    option = input("Enter 1, 2, 3 or 4: ")

    if option == "1":
        asyncio.run(first_part())
    elif option == "2":
        second_part()
    elif option == "3":
        asyncio.run(first_part())
        second_part()
    elif option == "4":
        asyncio.run(process_specific_activity())
    else:
        print("Error: Invalid option")

    # Abrir la interfaz HTML al finalizar la ejecución
    current_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.abspath(os.path.join(current_dir, '..', '3. Frontend (HTML offline)', 'index.html'))
    webbrowser.open('file://' + index_path)

main()