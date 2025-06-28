import os
import pandas as pd
import numpy as np
import ollama
from fitparse import FitFile
from googletrans import Translator
from datetime import datetime
import requests
import pytz
import zipfile
import shutil

import warnings
warnings.simplefilter("ignore")

# ------------------------- Determinar constantes ------------------------------------------------

def determine_cst(file_counter, 
                  switch_FTP_2023_2024, switch_FTP_2024_2025, 
                  FTP_bpm_2023, FTP_rap_2023, weight_2023,
                  FTP_bpm_2024, FTP_rap_2024, weight_2024,
                  FTP_bpm_2025, FTP_rap_2025, weight_2025):

    if file_counter < switch_FTP_2023_2024:
        return FTP_bpm_2023, FTP_rap_2023, FTP_rap_2023, weight_2023
    elif file_counter < switch_FTP_2024_2025:
        return FTP_bpm_2024, FTP_rap_2024, FTP_rap_2024, weight_2024
    else:
        return FTP_bpm_2025, FTP_rap_2025, FTP_rap_2025, weight_2025

# ------------------------- De garmin a DataFrame ------------------------------------------------

def convert_fit_to_rawdata(fit_path, required_columns):
    raw_data = []

    try:
        fitfile = FitFile(fit_path)

        # --- Extraer los campos 'record' de interés ---
        for record in fitfile.get_messages("record"):
            data_dict = {}
            for data in record:
                name = data.name
                if name in required_columns:
                    val = data.value
                    # Convertir timestamp FIT (datetime) a pandas.Timestamp
                    if name == 'timestamp':
                        val = pd.to_datetime(val)  
                    data_dict[name] = val
            if data_dict:
                raw_data.append(data_dict)

        if not raw_data:
            print("⚠️ No se extrajo nada: revisa `required_columns`.")
            return pd.DataFrame()

        # --- DataFrame inicial ---
        raw_data_df = pd.DataFrame(raw_data)

        # --- Inicializar arrays con NaN ---
        def init_array(col):
            if col in raw_data_df:
                return raw_data_df[col].to_numpy()
            else:
                # Si no existe, creamos NaN del mismo largo
                return np.full(len(raw_data_df), np.nan)

        timestamp          = init_array('timestamp')
        position_lat       = init_array('position_lat')
        position_long      = init_array('position_long')
        distance           = init_array('distance')
        enhanced_speed     = init_array('enhanced_speed')
        enhanced_altitude  = init_array('enhanced_altitude')
        heart_rate         = init_array('heart_rate')
        cadence            = init_array('cadence')
        fractional_cadence = init_array('fractional_cadence')

        # --- Relleno inteligente de huecos ---
        def smart_fill(arr):
            s = pd.Series(arr)
            if s.notna().any():
                s = s.interpolate(method='linear', limit_direction='both')
                s = s.fillna(method='bfill').fillna(method='ffill')
            return s.to_numpy()

        timestamp          = smart_fill(timestamp)
        position_lat       = smart_fill(position_lat)
        position_long      = smart_fill(position_long)
        distance           = smart_fill(distance)
        enhanced_speed     = smart_fill(enhanced_speed)
        enhanced_altitude  = smart_fill(enhanced_altitude)
        heart_rate         = smart_fill(heart_rate)
        cadence            = smart_fill(cadence)
        fractional_cadence = smart_fill(fractional_cadence)

        # --- Corregir coordenadas iniciales vacías ---
        for arr in (position_lat, position_long):
            idx = np.where(~np.isnan(arr))[0]
            if idx.size:
                arr[:idx[0]] = arr[idx[0]]

        # --- Columnas sin datos pasan a cero ---
        def zero_if_all_nan(arr):
            return np.zeros_like(arr) if np.isnan(arr).all() else arr

        position_lat       = zero_if_all_nan(position_lat)
        position_long      = zero_if_all_nan(position_long)
        distance           = zero_if_all_nan(distance)
        enhanced_speed     = zero_if_all_nan(enhanced_speed)
        enhanced_altitude  = zero_if_all_nan(enhanced_altitude)
        heart_rate         = zero_if_all_nan(heart_rate)
        cadence            = zero_if_all_nan(cadence)
        fractional_cadence = zero_if_all_nan(fractional_cadence)

        # --- Construir DataFrame base (timestamp como datetime) ---
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamp),
            'position_lat': position_lat,
            'position_long': position_long,
            'distance': distance,
            'enhanced_speed': enhanced_speed,
            'enhanced_altitude': enhanced_altitude,
            'heart_rate': heart_rate,
            'cadence': cadence,
            'fractional_cadence': fractional_cadence
        })

        # --- Reindexar a 1 s usando DateTimeIndex ---
        df = df.set_index('timestamp')
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='S')
        df = df.reindex(full_idx)
        df = df.interpolate(method='time')
        df = df.fillna(method='bfill').fillna(method='ffill')

        # --- Columnas finales ---
        df = df.reset_index().rename(columns={'index': 'timestamp'})
        # Columna auxiliar: segundos transcurridos desde el inicio
        df['elapsed'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

        return df

    except Exception as error:
        print(f"Error al procesar el archivo FIT: {error}")
        return pd.DataFrame()

# ------------------------- Datos originales a datos ampliados -----------------------------------

def raw_to_processed(raw_data, weight, cst):
    # --- Extraer columnas de las constantes y de raw_data ---
    road_running_limit_pace = cst["road_running_limit_pace"]
    trail_running_limit_pace = cst["trail_running_limit_pace"]
    n_level = cst["n_level"]
    margin_error_lower_level = cst["margin_error_lower_level"]
    beach_altitud = cst["beach_altitud"]
    margin_error_upper_level = cst["margin_error_upper_level"]
    running_efficiency = cst["running_efficiency"]
    distance_error = cst["distance_error"]

    timestamp = raw_data['timestamp']
    position_lat = raw_data['position_lat']
    position_long = raw_data['position_long']
    distance = raw_data['distance']
    enhanced_speed = raw_data['enhanced_speed']
    enhanced_altitude = raw_data['enhanced_altitude']
    heart_rate = raw_data['heart_rate']
    cadence = raw_data['cadence']
    fractional_cadence = raw_data['fractional_cadence']

    if np.nanmax(distance) > 150 * 1000: # Distancias de mas de 150km significan error de registro
        distance_error += 1
        return None, None, None, None, None
    
    ## --- Cálculos derivados ---

    # --- Coordenadas ---
    position_lat = position_lat*180/(2**31) # [º]
    position_long = position_long*180/(2**31) # [º]

    # --- Tiempo ---
    timestamp_seconds = (timestamp - timestamp.iloc[0]).dt.total_seconds() # [s]
    instant_time = np.concatenate(([0], np.diff(timestamp_seconds) / 60)) # [min]
    accumulated_time = (timestamp_seconds - timestamp_seconds[0]) / 60 # [min]

    # --- Distancia ---
    instant_distance = np.concatenate(([0], np.diff(distance))) # [m]
    accumulated_distance = distance.copy() # [m]

    # --- Desnivel (instantáneos y acumulados) ---
    beach_margin = margin_error_lower_level / n_level * 2 # [m]
    num_points = len(enhanced_altitude)  # [ud]
    instant_level = np.zeros(num_points) # [m]
    accumulated_positive_level = np.zeros(num_points) # [m]
    accumulated_negative_level = np.zeros(num_points) # [m]
    i_idx = 0
    while i_idx < num_points:
        if (i_idx + n_level) <= num_points:
            altitude_delta = (enhanced_altitude[i_idx + n_level - 1] - enhanced_altitude[i_idx]) / n_level
            if enhanced_altitude[i_idx] < beach_altitud and altitude_delta < beach_margin:
                altitude_delta = 0
            if (i_idx + n_level) < num_points:
                diff_val = abs(enhanced_altitude[i_idx] - enhanced_altitude[i_idx + n_level])
                if diff_val <= margin_error_lower_level or diff_val > margin_error_lower_level * n_level:
                    instant_level[i_idx:i_idx+n_level] = 0
                else:
                    instant_level[i_idx:i_idx+n_level] = altitude_delta
            else:
                instant_level[i_idx:i_idx+n_level] = altitude_delta

            if altitude_delta > 0 and altitude_delta < margin_error_upper_level / n_level:
                current_min = np.min(accumulated_negative_level[:i_idx]) if i_idx > 0 else 0
                accumulated_negative_level[i_idx:i_idx+n_level] = current_min
                current_max = np.max(accumulated_positive_level[:i_idx]) if i_idx > 0 else 0
                accumulated_positive_level[i_idx:i_idx+n_level] = current_max + n_level * altitude_delta
            elif altitude_delta < 0 and altitude_delta > -margin_error_upper_level / n_level:
                current_min = np.min(accumulated_negative_level[:i_idx]) if i_idx > 0 else 0
                accumulated_negative_level[i_idx:i_idx+n_level] = current_min + n_level * altitude_delta
                current_max = np.max(accumulated_positive_level[:i_idx]) if i_idx > 0 else 0
                accumulated_positive_level[i_idx:i_idx+n_level] = current_max
            else:
                current_min = np.min(accumulated_negative_level[:i_idx]) if i_idx > 0 else 0
                current_max = np.max(accumulated_positive_level[:i_idx]) if i_idx > 0 else 0
                accumulated_negative_level[i_idx:i_idx+n_level] = current_min
                accumulated_positive_level[i_idx:i_idx+n_level] = current_max
            i_idx += n_level
        else:
            instant_level[i_idx:] = 0
            last_positive = accumulated_positive_level[i_idx-1] if i_idx > 0 else 0
            last_negative = accumulated_negative_level[i_idx-1] if i_idx > 0 else 0
            accumulated_positive_level[i_idx:] = last_positive
            accumulated_negative_level[i_idx:] = last_negative
            break

   
    # Tipo de actividad
    max_top100_enhanced_speed = np.nanmean(np.sort(enhanced_speed)[-100:])
    max_top100_enhanced_altitude = np.nanmean(np.sort(enhanced_altitude)[-100:])
    total_time = np.nanmax(accumulated_time)
    total_distance = np.nanmax(accumulated_distance) / 1000  # en km
    if total_distance > 0:
        mean_pace = total_time / total_distance  # [min/km]
    else:
        mean_pace = 0
    if total_distance > 0:
        ratio_level_distance = np.nanmax(accumulated_positive_level) / total_distance
    else:
        ratio_level_distance = np.nan  # o podrías asignar 0, según cómo lo uses después
    if max_top100_enhanced_speed > 10 and max_top100_enhanced_altitude > 1000:
        activity_type = "Snow"
    elif mean_pace > 12 or total_distance <3.5:
        if total_distance == 0:
            activity_type = "Gym"
        elif mean_pace > 30 and total_distance < 3:
            activity_type = "Escalada"
        elif 12 <= mean_pace <= 25:
            activity_type = "Caminar"
        else:
            activity_type = "Otros"
    elif ratio_level_distance < 10:
        activity_type = "Road"
    else:
        activity_type = "Trail"

    # --- Ritmo ---
    with np.errstate(divide='ignore', invalid='ignore'):
        instant_pace = 60.0 / (enhanced_speed * 3.6) # [min/km]
    if activity_type == 'Road':
        instant_pace[instant_pace > road_running_limit_pace] = 0
    elif activity_type == 'Trail':
        instant_pace[instant_pace > trail_running_limit_pace] = 0
    else:
        pass

    # --- Pulsaciones ---
    instant_bpm = heart_rate.copy() # [bpm]

    # --- BPMxPACE
    instant_bpmxpace = instant_bpm * instant_pace
    instant_bpmxpace[(instant_bpmxpace > 1500) | (instant_bpmxpace < 400)] = 0 # [bpm*min/km]

    # --- Cadencia y cadencia fraccionada ---
    instant_cadence = cadence * 2 # [ppm]
    instant_split_cadence = fractional_cadence * 2 # [pasos]

    # --- Zancada ---
    with np.errstate(divide='ignore', invalid='ignore'): 
        instant_stride = 1000.0 / (instant_pace * instant_cadence)
    instant_stride[(instant_stride > 5) | (instant_stride < 0.2)] = 0 # [m]

    # --- Cálculo de potencia ---
    # Método x distancia:
    instant_power_1 = 4186 * weight * (instant_distance / 1000.0) * running_efficiency  # [W]

    # Método MET:
    MET_paces_table = np.array([8.1, 7.5, 7.1, 6.2, 5.6, 5.3, 5.0, 4.7, 4.3, 4.0, 3.7, 3.4, 3.1, 2.9, 2.7]) # Tabla obtenida del libro: La Dieta Inteligente para Runners
    MET_values_table = np.array([6, 8.3, 9, 9.8, 10.5, 11, 11.5, 11.8, 12.3, 12.8, 14.5, 16, 19, 19.8, 23])

    coeffs = np.polyfit(MET_paces_table, MET_values_table, 3)
    MET_model = np.poly1d(coeffs)
    instant_MET_2 = MET_model(instant_pace)
    
    instant_MET_2[instant_pace > 8.1] = 6
    instant_MET_2[instant_pace <= 2.7] = 23
    mean_MET = np.nanmean(instant_MET_2[np.isfinite(instant_MET_2)])
    instant_MET_2[instant_pace < 2] = mean_MET
    instant_power_2 = weight * (instant_MET_2 * 4186) * (instant_time / 60.0) * running_efficiency  # [W]

    # Promedio
    instant_power_final = instant_power_2 # [W]

    # --- Crear DataFrame con los datos procesados ---
    processed_data = pd.DataFrame({
        'instant_time': instant_time,
        'accumulated_time': accumulated_time,
        'position_long': position_long,
        'position_lat': position_lat,
        'instant_distance': instant_distance,
        'accumulated_distance': accumulated_distance,
        'instant_level': instant_level,
        'accumulated_positive_level': accumulated_positive_level,
        'accumulated_negative_level': accumulated_negative_level,
        'instant_pace': instant_pace,
        'instant_bpm': instant_bpm,
        'instant_bpmxpace': instant_bpmxpace,
        'instant_cadence': instant_cadence,
        'instant_split_cadence': instant_split_cadence,
        'instant_stride': instant_stride,
        'instant_power_1': instant_power_1,
        'instant_MET_2': instant_MET_2,
        'instant_power_2': instant_power_2,
        'instant_power_final': instant_power_final
    })

    # --- Obtener el dia, la hora de inicio y final y el nombre de la actividad ---
    tz_spain = pytz.timezone('Europe/Madrid')
    activity_start_datetime = raw_data['timestamp'].iloc[0].tz_localize('UTC').tz_convert(tz_spain)
    activity_end_datetime = raw_data['timestamp'].iloc[-1].tz_localize('UTC').tz_convert(tz_spain)

    activity_date = activity_start_datetime.strftime('%d/%m/%Y')
    inicial_time = activity_start_datetime.strftime('%H:%M:%S')
    finish_time = activity_end_datetime.strftime('%H:%M:%S')

    activity_name = activity_start_datetime.strftime('%y_%m_%d_%H_%M')

    return activity_name, processed_data, MET_model, activity_type, activity_date, inicial_time, finish_time

# ------------------------- Generador de vueltas -------------------------------------------------

def lap_generator(interval_time,processed_data):
    # --- Variables instantaneas & vueltas ---
    instant_time = processed_data['instant_time']
    accumulated_time = processed_data['accumulated_time']
    accumulated_distance = processed_data['accumulated_distance']
    instant_level = processed_data['instant_level']
    accumulated_positive_level = processed_data['accumulated_positive_level']
    accumulated_negative_level = processed_data['accumulated_negative_level']
    instant_bpm = processed_data['instant_bpm']
    instant_pace = processed_data['instant_pace']
    instant_cadence = processed_data['instant_cadence']
    instant_stride = processed_data['instant_stride']
    instant_power_final = processed_data['instant_power_final']

    initial_time_laps = []
    time_laps = []
    distance_laps = []
    level_laps = []
    accumulated_positive_level_laps = []
    accumulated_negative_level_laps = []
    bpm_laps = []
    moving_pace_laps = []
    real_pace_laps = []
    rap_laps = []
    bpmxrap_laps = []
    cadence_laps = []
    stride_laps = []
    power_laps = []

    # --- Cálculo de cada vuelta ---
    num_laps = int(np.ceil(len(instant_time) / interval_time))
    for k in range(num_laps):
        i_lap = k * interval_time
        j_lap = min((k + 1) * interval_time, len(instant_time))
        
        # --- Tiempo de la vuelta ---
        initial_time_lap = accumulated_time[i_lap]
        final_time_lap = accumulated_time[j_lap - 1]
        time_lap = (final_time_lap - initial_time_lap)*60
        
        # --- Distancia y desnivel---
        distance_lap = accumulated_distance[j_lap - 1] - accumulated_distance[i_lap]
        level_lap = np.sum(instant_level[i_lap:j_lap])
        accumulated_positive_level_lap = accumulated_positive_level[j_lap - 1] - accumulated_positive_level[i_lap]
        accumulated_negative_level_lap = accumulated_negative_level[j_lap - 1] - accumulated_negative_level[i_lap]
        
        # --- Pulsaciones ---
        bpm_lap = np.mean(instant_bpm[i_lap:j_lap])
        
        # --- Ritmos ---
        pace_lap = instant_pace[i_lap:j_lap]
        moving_pace_lap = np.mean(pace_lap[pace_lap != 0]) if np.any(pace_lap != 0) else 0

        real_pace_lap = (time_lap / 60) / (distance_lap / 1000) if distance_lap != 0 else 0
        
        dist_eq_lap = (distance_lap + 5 * accumulated_positive_level_lap - 2.5 * accumulated_negative_level_lap)
        rap_lap = (time_lap / 60) / (dist_eq_lap / 1000) if dist_eq_lap != 0 else 0
        
        moving_pace_lap = 0 if moving_pace_lap > 20 else moving_pace_lap
        real_pace_lap = 0 if real_pace_lap > 20 else real_pace_lap
        rap_lap = 0 if rap_lap > 20 else rap_lap

        # --- BPMxRAP ---
        bpmxrap_lap = bpm_lap * rap_lap
        if bpmxrap_lap > 1500 or bpmxrap_lap < 400:
            bpmxrap_lap = 0
        
        # --- Cadencia & zancadas ---
        all_lap_cadence = instant_cadence[i_lap:j_lap]
        cadence_lap = np.mean(all_lap_cadence[all_lap_cadence != 0]) if np.any(all_lap_cadence != 0) else 0
        
        all_lap_stride = instant_stride[i_lap:j_lap]
        stride_lap = np.mean(all_lap_stride[all_lap_stride != 0]) if np.any(all_lap_stride != 0) else 0
        
        # --- Potencia ---
        all_lap_power = instant_power_final[i_lap:j_lap]
        power_lap = np.mean(all_lap_power[all_lap_power != 0]) if np.any(all_lap_power != 0) else 0
        
        # --- Almacenar resultados ---
        initial_time_laps.append(initial_time_lap)
        time_laps.append(time_lap)
        distance_laps.append(distance_lap)
        level_laps.append(level_lap)
        accumulated_positive_level_laps.append(accumulated_positive_level_lap)
        accumulated_negative_level_laps.append(accumulated_negative_level_lap)
        bpm_laps.append(bpm_lap)
        moving_pace_laps.append(moving_pace_lap)
        real_pace_laps.append(real_pace_lap)
        rap_laps.append(rap_lap)
        bpmxrap_laps.append(bpmxrap_lap)
        cadence_laps.append(cadence_lap)
        stride_laps.append(stride_lap)
        power_laps.append(power_lap)
    
    lap_table = pd.DataFrame({
        'initial_time_laps': initial_time_laps,
        'time_laps': time_laps,
        'distance_laps': distance_laps,
        'level_laps': level_laps,
        'accumulated_positive_level_laps': accumulated_positive_level_laps,
        'accumulated_negative_level_laps': accumulated_negative_level_laps,
        'bpm_laps': bpm_laps,
        'moving_pace_laps': moving_pace_laps,
        'real_pace_laps': real_pace_laps,
        'rap_laps': rap_laps,
        'bpmxrap_laps': bpmxrap_laps,
        'cadence_laps': cadence_laps,
        'stride_laps': stride_laps,
        'power_laps': power_laps,
    })
    return lap_table 

# ------------------------- Resumen actividad ----------------------------------------------------

def summary_data(processed_data, cst):
    # --- Extraer las constantes y variables de cst y processed_data ---
    weight = cst["weight"]
    FTP_bpm = cst["FTP_bpm"]
    FTP_rap = cst["FTP_rap"]
    MET_model = cst["MET_model"]
    activity_date = cst["activity_date"]
    inicial_time = cst["inicial_time"]
    finish_time = cst["finish_time"]
    PI_referencia = cst["PI_referencia"]
    running_efficiency = cst["running_efficiency"]
    activity_type = cst["activity_type"]

    accumulated_time = processed_data['accumulated_time']
    position_long = processed_data['position_long']
    position_lat = processed_data['position_lat']
    accumulated_distance = processed_data['accumulated_distance']
    accumulated_positive_level = processed_data['accumulated_positive_level']
    accumulated_negative_level = processed_data['accumulated_negative_level']
    instant_pace = processed_data['instant_pace']
    instant_bpm = processed_data['instant_bpm']
    instant_cadence = processed_data['instant_cadence']
    instant_stride = processed_data['instant_stride']
    instant_power_1 = processed_data['instant_power_1']
    instant_power_2 = processed_data['instant_power_2']

    # --- Coordenadas iniciales ---
    inicial_long = position_long[0] if len(position_long) > 0 else 0 # [º]
    inicial_lat = position_lat[0] if len(position_lat) > 0 else 0 # [º]

    # --- Tiempos y distancias ---
    total_time = np.max(accumulated_time) # [min]
    total_distance = np.max(accumulated_distance) / 1000 # [km]
    total_accumulated_positive_level = np.max(accumulated_positive_level) # [m]
    total_accumulated_negative_level = np.min(accumulated_negative_level) # [m]
    equivalence_distance = total_distance + 0.792*total_accumulated_positive_level/100

    # --- Ritmos ---
    valid_pace = instant_pace[instant_pace != 0]
    moving_average_pace = np.mean(valid_pace) if valid_pace.size > 0 else 0 # [min/km]

    real_average_pace = total_time / total_distance if total_distance != 0 else 0 # [min/km]

    dist_eq_rap = total_distance + 5 * total_accumulated_positive_level / 1000 - 2.5 * total_accumulated_negative_level / 1000 # [km]
    average_rap = total_time / (dist_eq_rap) if dist_eq_rap != 0 else 0 # [min/km]

    valid_bpm = instant_bpm[(instant_bpm != 0) & (~np.isnan(instant_bpm))]
    average_bpm = np.mean(valid_bpm) if valid_bpm.size > 0 else 0 # [bpm]
    max_bpm = np.max(instant_bpm) # [bpm]
    average_bpmxrap = average_bpm * moving_average_pace # [bpm*min/km]

    deviation_pace_percent = (np.std(instant_pace) / np.mean(instant_pace)) * 100 if np.mean(instant_pace) != 0 else 0 # [%]
    deviation_bpm_percent = (np.std(instant_bpm) / np.mean(instant_bpm)) * 100 if np.mean(instant_bpm) != 0 else 0 # [%]

    average_cadence = np.mean(instant_cadence) # [ppm]
    average_stride = np.mean(instant_stride) # [min]

    PI = average_bpmxrap / PI_referencia
    PI = (1.0 / PI) * 100 if PI != 0 else 0 # [%]

    # --- VDOT calculation ---
    if activity_type in ['Road', 'Trail']:# --- Cálculo de VDOT solo si es Road o Trail ---
        # Se crea el DataFrame con los datos de VDOT y tiempos
        vdot_data = {
            'VDOT': [30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62],
            '1500': [8.5, 8.033, 7.617, 7.233, 6.9, 6.583, 6.317, 6.05, 5.817, 5.5, 5.4, 5.217, 5.033, 4.883, 4.733, 4.583, 4.45],
            '3000': [17.933, 16.983, 16.15, 15.383, 14.683, 14.05, 13.467, 12.917, 12.433, 11.75, 11.55, 11.15, 10.783, 10.45, 10.133, 9.833, 9.55],
            '5000': [30.667, 29.083, 27.65, 26.367, 25.2, 24.133, 23.15, 22.25, 21.417, 20.3, 19.95, 19.283, 18.667, 18.083, 17.55, 17.05, 16.567],
            '10000': [63.767, 60.433, 57.433, 54.733, 52.283, 50.05, 48.017, 46.15, 44.417, 42.067, 41.35, 39.983, 38.7, 37.517, 36.4, 35.367, 34.383],
            '15000': [98.233, 93.117, 88.5, 84.333, 80.55, 77.1, 73.933, 71.033, 68.367, 64.733, 63.6, 61.483, 59.5, 57.65, 55.917, 54.3, 52.783],
            '21100': [141.067, 133.817, 127.267, 121.317, 115.917, 110.983, 106.45, 102.283, 98.45, 93.2, 91.583, 88.517, 85.667, 83, 80.5, 78.15, 75.57],
            '42200': [289.283, 274.983, 262.05, 250.317, 239.583, 229.75, 220.717, 212.383, 204.65, 194.1, 190.817, 184.6, 178.783, 173.333, 168.233, 163.417, 158.9],
            '50000': [359.89, 342.1, 326.01, 311.41, 298.06, 285.83, 274.59, 264.22, 254.6, 241.48, 237.39, 229.66, 222.42, 215.64, 209.29, 203.30, 197.68],
            '150000': [1079.67, 1026.3, 978.03, 934.24, 894.18, 857.48, 823.77, 792.66, 763.8, 724.43, 712.17, 688.97, 667.26, 646.92, 627.89, 609.91, 593.05]
        }
        vdot_table = pd.DataFrame(vdot_data)
        distance_columns = [col for col in vdot_table.columns if col != 'VDOT']
        distances = np.array([float(col) for col in distance_columns])
        sort_idx = np.argsort(distances)
        distances = distances[sort_idx]

        if activity_type == 'Road':
            distance_to_interpol = total_distance * 1000
        else:  # Trail
            distance_to_interpol = equivalence_distance * 1000

        predicted_times = []
        for idx, row in vdot_table.iterrows():
            times = row[distance_columns].values.astype(float)[sort_idx]
            interp_time = np.interp(distance_to_interpol, distances, times)
            predicted_times.append(interp_time)
        predicted_times = np.array(predicted_times)

        vdot_values = vdot_table['VDOT'].values.astype(float)
        order = np.argsort(predicted_times)
        sorted_times = predicted_times[order]
        sorted_vdot = vdot_values[order]
        vdot_final = np.interp(total_time, sorted_times, sorted_vdot)
    else:
        vdot_final = 0

    # --- Cálculo de potencias y energías ---
    average_power_1 = np.mean(instant_power_1)
    average_VAM_1 = average_power_1 / weight
    total_energy_1 = np.sum(instant_power_1) / 4186 / running_efficiency
    energy_km_1 = total_energy_1 / total_distance if total_distance != 0 else 0

    average_power_2 = np.mean(instant_power_2)
    average_VAM_2 = average_power_2 / weight
    total_energy_2 = np.sum(instant_power_2) / 4186 / running_efficiency
    energy_km_2 = total_energy_2 / total_distance if total_distance != 0 else 0

    average_power = average_power_2 # [W]
    standardized_power = average_power * 170 / average_bpm if average_bpm != 0 else 0 # [W]
    average_VAM = average_VAM_2 # [W/kg]
    total_energy = total_energy_2 # [kcal]
    energy_km = energy_km_2 # [kcal/km]

    # --- Factor de intensidad y carga de entrenamiento ---
    MET_FTP = MET_model(FTP_rap)
    if FTP_rap > 8.1:
        MET_FTP = 6
    elif FTP_rap <= 2.7:
        MET_FTP = 23
    elif FTP_rap < 2:
        MET_FTP = np.nanmean([MET_FTP])
    FTP_power = weight * MET_FTP * (4184 / 3600) * running_efficiency

    intensity_factor_power = average_power / FTP_power if FTP_power != 0 else 0
    intensity_factor_bpm = average_bpm / FTP_bpm if FTP_bpm != 0 else 0

    TSS_power = (total_time / 60) * (intensity_factor_power ** 2) * 100
    TSS_bpm = (total_time / 60) * (intensity_factor_bpm ** 2) * 100

    intensity_factor = (intensity_factor_power + intensity_factor_bpm) / 2 # [ud]
    training_load = (TSS_power + TSS_bpm) / 2 # [TSS]

    # --- Ubicación ---
    p_long = next((x for x in position_long if not np.isnan(x)), 0)
    p_lat = next((x for x in position_lat if not np.isnan(x)), 0)
    api_city_key = '8996f4c10ffe45529ab096d18b7a3cf5'
    url_city = f'https://api.opencagedata.com/geocode/v1/json?q={p_lat}+{p_long}&key={api_city_key}'
    response_city = requests.get(url_city).json()
    if response_city.get('results') and len(response_city['results']) > 0:
        components = response_city['results'][0].get('components', {})
        activity_city = (
            components.get('city') or
            components.get('town') or
            components.get('village') or
            components.get('municipality') or
            components.get('county') or
            components.get('state') or
            components.get('country', '')
        )
    else:
        activity_city = ''


    try:
        activity_date_dt = datetime.strptime(activity_date, '%d/%m/%Y')
    except Exception:
        activity_date_dt = None
    if activity_date_dt:
        activity_month = activity_date_dt.month
        activity_day = activity_date_dt.day
        if (activity_month == 12 and activity_day >= 21) or (activity_month == 1) or (activity_month == 2) or (activity_month == 3 and activity_day < 21):
            activity_season = 'Invierno'
        elif (activity_month == 3 and activity_day >= 21) or (activity_month == 4) or (activity_month == 5) or (activity_month == 6 and activity_day < 21):
            activity_season = 'Primavera'
        elif (activity_month == 6 and activity_day >= 21) or (activity_month == 7) or (activity_month == 8) or (activity_month == 9 and activity_day < 21):
            activity_season = 'Verano'
        else:
            activity_season = 'Otoño'
    else:
        activity_season = ''

    # Si la actividad no es Road ni Trail, la carga es 0
    if activity_type not in ['Road', 'Trail']:
        training_load = 0

    summary_dict = {
        'activity_type': str(activity_type),
        'activity_city': str(activity_city),
        'activity_date': str(activity_date),
        'inicial_time': str(inicial_time),
        'finish_time': str(finish_time),
        'activity_season': str(activity_season),
        'total_time': total_time,
        'total_distance': total_distance,
        'equivalence_distance': equivalence_distance,
        'total_accumulated_positive_level': total_accumulated_positive_level,
        'total_accumulated_negative_level': total_accumulated_negative_level,
        'moving_average_pace': moving_average_pace,
        'real_average_pace': real_average_pace,
        'average_rap': average_rap,
        'deviation_pace': deviation_pace_percent,
        'average_bpm': average_bpm,
        'max_bpm': max_bpm,
        'deviation_bpm': deviation_bpm_percent,
        'average_bpmxrap': average_bpmxrap,
        'average_cadence': average_cadence,
        'average_stride': average_stride,
        'total_energy': total_energy,
        'average_power': average_power,
        'standardized_power': standardized_power,
        'training_load': training_load,
        'vdot': vdot_final,
        'PI': PI,
        'intensity_factor': intensity_factor,
        'average_VAM': average_VAM,
        'energy_km': energy_km,
        'TSS_power': TSS_power,
        'TSS_bpm': TSS_bpm,
        'FTP_bpm': FTP_bpm,
        'FTP_rap': FTP_rap,
        'FTP_power': FTP_power,
        'inicial_long': inicial_long,
        'inicial_lat': inicial_lat,
        'weight': weight
    }
    summary = pd.DataFrame([summary_dict])
    return summary

# ------------------------- Zonas de entrenamiento -----------------------------------------------

def zone_calc(FTP_bpm, FTP_pace, processed_data):
    bpm_z1 = FTP_bpm * 0.8
    bpm_z2 = FTP_bpm * 0.885
    bpm_z3 = FTP_bpm * 0.925
    bpm_z4 = FTP_bpm * 1.0
    pace_z1 = FTP_pace / 0.775
    pace_z2 = FTP_pace / 0.877
    pace_z3 = FTP_pace / 0.943
    pace_z4 = FTP_pace / 1.0

    if isinstance(processed_data, pd.DataFrame):
        instant_bpm = processed_data['instant_bpm'].to_numpy()
        instant_pace = processed_data['instant_pace'].to_numpy()
    else:
        instant_bpm = np.array(processed_data['instant_bpm'])
        instant_pace = np.array(processed_data['instant_pace'])
    
    # --- Contador de tiempo en zonas ---
    time_z1_bpm = time_z2_bpm = time_z3_bpm = time_z4_bpm = time_z5_bpm = 0
    for bpm_val in instant_bpm:
        if bpm_val > bpm_z4:
            time_z5_bpm += 1
        elif bpm_val > bpm_z3:
            time_z4_bpm += 1
        elif bpm_val > bpm_z2:
            time_z3_bpm += 1
        elif bpm_val > bpm_z1:
            time_z2_bpm += 1
        else:
            time_z1_bpm += 1

    time_z1_pace = time_z2_pace = time_z3_pace = time_z4_pace = time_z5_pace = 0
    for pace_val in instant_pace:
        if pace_val < pace_z4:
            time_z5_pace += 1
        elif pace_val < pace_z3:
            time_z4_pace += 1
        elif pace_val < pace_z2:
            time_z3_pace += 1
        elif pace_val < pace_z1:
            time_z2_pace += 1
        else:
            time_z1_pace += 1

    # --- Tabla timepo en zonas ---
    time_zones_names = ['time_z1_bpm', 'time_z2_bpm', 'time_z3_bpm', 'time_z4_bpm', 'time_z5_bpm',
                        'time_z1_pace', 'time_z2_pace', 'time_z3_pace', 'time_z4_pace', 'time_z5_pace']

    time_zones_values = [time_z1_bpm, time_z2_bpm, time_z3_bpm, time_z4_bpm, time_z5_bpm,
                        time_z1_pace, time_z2_pace, time_z3_pace, time_z4_pace, time_z5_pace]

    # --- Limites ---
    limit_names = ['limit_bpm_z1', 'limit_bpm_z2', 'limit_bpm_z3', 'limit_bpm_z4', 
                'limit_pace_z1', 'limit_pace_z2', 'limit_pace_z3', 'limit_pace_z4']

    limit_values = [bpm_z1, bpm_z2, bpm_z3, bpm_z4, pace_z1, pace_z2, pace_z3, pace_z4]

    # --- Crear el DataFrame con zonas y límites ---
    time_zones = pd.DataFrame([time_zones_values + limit_values], columns=time_zones_names + limit_names)
    return time_zones

# ------------------------- Ratios bpm-ritmo/potencia --------------------------------------------

def ratios_bpm(lap_table):
    bpm_instant = lap_table['bpm_laps'].to_numpy()[10:]
    pace_instant = lap_table['real_pace_laps'].to_numpy()[10:]  # o 'rap_laps'
    power_instant = lap_table['power_laps'].to_numpy()[10:]

    bpm_ranges = [
        (0, 119), (120, 129), (130, 139), (140, 147), (148, 152), (153, 157),
        (158, 162), (163, 167), (168, 172), (173, 177), (178, 182), (183, 187),
        (188, 192), (193, 197), (198, 202), (203, 220)
    ]

    # --- Almacenar los valores de pace y power ---
    pace_dict = {f'pace_{low}_{high}_bpm': np.full(len(bpm_instant), np.nan) for low, high in bpm_ranges}
    power_dict = {f'power_{low}_{high}_bpm': np.full(len(bpm_instant), np.nan) for low, high in bpm_ranges}

    # --- Asignar pace y power a sus respectivos rangos ---
    for i in range(len(bpm_instant)):
        bpm_val = bpm_instant[i]
        for (low, high), key in zip(bpm_ranges, pace_dict.keys()):
            if low <= bpm_val <= high:
                pace_dict[key][i] = pace_instant[i]
                power_dict[key.replace('pace', 'power')][i] = power_instant[i]

    # --- Calcular media sin contar ceros ni NaNs ---
    def safe_mean_no_zero(arr):
        valid = arr[(~np.isnan(arr)) & (arr != 0)]
        return np.mean(valid) if len(valid) > 0 else 0

    # --- Filtrar valores extremos (doble de la media válida) ---
    def calculate_mean_and_filter(values):
        valid_values = values[(~np.isnan(values)) & (values != 0)]
        if len(valid_values) == 0:
            return np.zeros_like(values)
        
        mean_value = np.mean(valid_values)
        threshold = mean_value * 2
        filtered_values = np.where(np.abs(values - mean_value) > threshold, 0, values)
        return filtered_values

    # --- Promedios por rango de bpm ---
    ratio_pace_bpm = np.array([
        safe_mean_no_zero(pace_dict[f'pace_{low}_{high}_bpm']) for low, high in bpm_ranges
    ])
    ratio_pace_bpm = calculate_mean_and_filter(ratio_pace_bpm)

    ratio_power_bpm = np.array([
        safe_mean_no_zero(power_dict[f'power_{low}_{high}_bpm']) for low, high in bpm_ranges
    ])
    ratio_power_bpm = calculate_mean_and_filter(ratio_power_bpm)

    all_ratios_names = [f"ritmo_{low}_{high}bpm" for low, high in bpm_ranges] + \
                       [f"potencia_{low}_{high}bpm" for low, high in bpm_ranges]

    # --- Tabla resultados ---
    ratios_data = np.concatenate([ratio_pace_bpm, ratio_power_bpm])
    ratios_table = pd.DataFrame([ratios_data], columns=all_ratios_names)
    
    return ratios_table

# ------------------------- Marcas del día -------------------------------------------------------

def daily_times_calc(summary):

    # --- Distancias objetivo ---
    race_distances = {
        '3km_daily_mark': 3,
        '5km_daily_mark': 5,
        '10km_daily_mark': 10,
        '21_1km_daily_mark': 21.1,
        '42_2km_daily_mark': 42.2
    }
    daily_mark = {}

    # --- Calcular el tiempo de la distancia hecha ---
    total_time = summary.loc[0, 'total_time']  # [min]
    total_distance = summary.loc[0, 'total_distance']  # [km]
    for key, distance in race_distances.items():
        if total_distance >= distance:
            daily_mark[key] = round((distance * total_time) / total_distance, 2)
        else:
            daily_mark[key] = None 
    daily_mark_df = pd.DataFrame([daily_mark])
    return daily_mark_df

# ------------------------- Resumen de deepseek --------------------------------------------------

async def add_deepseek_summary(general_data, model_version):
    
    # --- Prompt a enviar a deepseek ---
    prompt_activity_info = general_data.iloc[0]
    prompt = f"""
    I have completed a running activity today in {prompt_activity_info['activity_city']} on {prompt_activity_info['activity_date']}, with the following details:

    The activity started at {prompt_activity_info['inicial_time']} and finished at {prompt_activity_info['finish_time']}, during the {prompt_activity_info['activity_season']} season. The total time of the session was {prompt_activity_info['total_time']} minutes, covering a distance of {prompt_activity_info['total_distance']} km ({prompt_activity_info['equivalence_distance']} km adjusted for effort). The total elevation gain was {prompt_activity_info['total_accumulated_positive_level']} meters, while the total descent was {prompt_activity_info['total_accumulated_negative_level']} meters.

    My moving average pace was {prompt_activity_info['moving_average_pace']} min/km, and my real average pace was {prompt_activity_info['real_average_pace']} min/km. Please note that a higher pace number indicates a slower speed, and a lower pace number indicates a faster pace. The average heart rate during the run was {prompt_activity_info['average_bpm']} bpm, peaking at {prompt_activity_info['max_bpm']} bpm. I spent time in the following heart rate zones: {prompt_activity_info['time_z1_bpm']} minutes at {prompt_activity_info['limit_bpm_z1']} bpm, {prompt_activity_info['time_z2_bpm']} minutes at {prompt_activity_info['limit_bpm_z2']} bpm, {prompt_activity_info['time_z3_bpm']} minutes at {prompt_activity_info['limit_bpm_z3']} bpm, and {prompt_activity_info['time_z4_bpm']} minutes at {prompt_activity_info['limit_bpm_z4']} bpm. Similarly, my pace in these zones was {prompt_activity_info['time_z1_pace']} minutes at {prompt_activity_info['limit_pace_z1']} min/km, {prompt_activity_info['time_z2_pace']} minutes at {prompt_activity_info['limit_pace_z2']} min/km, {prompt_activity_info['time_z3_pace']} minutes at {prompt_activity_info['limit_pace_z3']} min/km, and {prompt_activity_info['time_z4_pace']} minutes at {prompt_activity_info['limit_pace_z4']} min/km.

    The total energy expenditure for this run was {prompt_activity_info['total_energy']} kcal, with an average power of {prompt_activity_info['average_power']} W.

    When evaluating the intensity, it is important to note that small changes in heart rate or pace should not be considered significant enough to define intensity changes. For this activity, intensity should be categorized as follows: Zones 3-4 indicate intense effort, while Zones 1-2 indicate a light effort. Given the heart rate and pace distribution, the intensity of the activity is { 'intense' if prompt_activity_info['limit_bpm_z3'] > 0.5 or prompt_activity_info['limit_bpm_z4'] > 0.5 else 'moderate' if prompt_activity_info['limit_bpm_z2'] > 0.5 else 'light' }.

    Generate a brief 1-2 paragraph descriptive summary of the running session, mentioning the overall effort, intensity, and any key observations based on the provided data, making sure the intensity assessment aligns with the heart rate zones and pace limits.

    Ensure that the summary is written without any bold, italics, or special formatting.
    """

    # --- Obtener la respuesta ---
    response = ollama.chat(
        model=f'deepseek-r1:{model_version}',
        messages=[{'role': 'user', 'content': prompt}]
    )
    response_text = response['message']['content']
    cleaned_response = response_text.split("<think>")[0] + response_text.split("</think>")[1].strip()

    # --- Traducir y agregar al dataframe la respuesta ---
    translator = Translator()
    translated_response = await translator.translate(cleaned_response, src='en', dest='es')

    general_data['activity_summary'] = translated_response.text
    return general_data

# ------------------------- Guardar datos --------------------------------------------------------

def save_individual_excel(activity_name, processed_data_folder_path, processed_data, general_date, all_laps, register_file_path, processed_files, filename):
    output_file_path = os.path.join(processed_data_folder_path, f"{activity_name}.xlsx")
    
    # --- Guardar datos en un archivo Excel con múltiples hojas ---
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        general_date.to_excel(writer, sheet_name='Summary', index=False)
        processed_data.to_excel(writer, sheet_name='Instant', index=False)
        all_laps.to_excel(writer, sheet_name='Laps', index=False)

    # --- Registrar el archivo procesado ---
    with open(register_file_path, 'a') as f:
        f.write(f"{(os.path.splitext(filename)[0])}\n")
    
    processed_files.add((os.path.splitext(filename)[0]))

def save_one_excel(activity_name, processed_data_folder_path, processed_data, general_date, all_laps):
    output_file_path = os.path.join(processed_data_folder_path, f"{activity_name}.xlsx")
    
    # --- Guardar datos en un archivo Excel con múltiples hojas ---
    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
        general_date.to_excel(writer, sheet_name='Summary', index=False)
        processed_data.to_excel(writer, sheet_name='Instant', index=False)
        all_laps.to_excel(writer, sheet_name='Laps', index=False)

