import zipfile
import os
import shutil
import gzip

# Definir las rutas relativas
script_dir = os.path.dirname(os.path.realpath(__file__))  # Ruta del script actual
zip_file_path = os.path.join(script_dir, 'strava.zip')  # Ruta al archivo ZIP
activities_folder = 'activities'  # Carpeta 'activities' dentro del ZIP
output_folder = os.path.join(script_dir, '1.Input')  # Carpeta de salida para los .fit

# Crear la carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Abrir el ZIP principal y procesar los archivos .fit.gz dentro de 'activities'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Obtener todos los archivos dentro de la carpeta 'activities'
    activities_path = [f for f in zip_ref.namelist() if f.startswith(activities_folder) and f.endswith('.fit.gz')]

    for file_name in activities_path:
        # Nombre base del archivo .fit final
        fit_file_name = os.path.basename(file_name).replace('.gz', '')
        fit_output_path = os.path.join(output_folder, fit_file_name)

        # Abrir y descomprimir el archivo .gz dentro del ZIP
        with zip_ref.open(file_name) as f_in:
            with gzip.open(f_in, 'rb') as gz_in:
                with open(fit_output_path, 'wb') as f_out:
                    shutil.copyfileobj(gz_in, f_out)

# Imprimir mensaje de éxito
print(f"Los archivos .fit se han descomprimido correctamente en la carpeta '{output_folder}'.")

