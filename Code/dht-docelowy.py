import os
import time
import board
import adafruit_dht
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from shutil import copyfile
from matplotlib.ticker import MultipleLocator

# Ścieżka RAM-dysku
ramdisk_path = '/mnt/ramdisk/Plots'

# Ścieżka do zapisu backupu
timestamp = datetime.now()
year = str(timestamp.year)
month = str(timestamp.month)
day = str(timestamp.day)
hour = str(timestamp.hour)
minute = str(timestamp.minute)

backup_filepath = '/home/adam/Desktop/Backup/'+year+'/'+month+'/'+'temp_humi_plot_'+year+'_'+month+'_'+day + '_' + month +'_' + year + 'time_' + hour + '_' + minute + '.png'
backup_filepath_csv = '/home/adam/Desktop/Backup/'+year+'/'+month+'/'+'temp_humi_csv_'+year+'_'+month+'_'+day + '_' + month +'_' + year + 'time_' + hour + '_' + minute + '.csv'

# Tworzenie katalogów do kopii zapasowych
try:
	os.makedirs('/home/adam/Desktop/Backup/'+year+'/')
except OSError:
	if not os.path.isdir('/home/adam/Desktop/Backup/'+year+'/'):
		raise
 
try:
	os.makedirs('/home/adam/Desktop/Backup/'+year+'/'+month+'/')
except OSError:
	if not os.path.isdir('/home/adam/Desktop/Backup/'+year+'/'+month+'/'):
		raise

# Inicjalizacja urządzenia DHT11
dhtDevice = adafruit_dht.DHT11(board.D4)

# Ścieżka do pliku CSV i HTML
csv_file_path = os.path.join(ramdisk_path, 'DHT11_data.csv')
html_file_path = os.path.join(ramdisk_path, 'sensor_data.html')

# Funkcja do odczytu danych z czujnika
def read_sensor_data():
    try:
        temperature_c = dhtDevice.temperature
        humidity = dhtDevice.humidity
        timestamp = datetime.now()
        return {'timestamp': timestamp, 'humidity': humidity, 'temperature': temperature_c}
    except RuntimeError as error:
        print(f"Błąd: {error.args[0]}")
        time.sleep(2.0)
        return None
    except Exception as error:
        dhtDevice.exit()
        raise error

# Odczyt danych z czujnika i zapis do CSV
reading = read_sensor_data()
if reading is not None:
    # Sprawdzenie, czy plik CSV istnieje, aby określić, czy dopisać nagłówki
    file_exists = os.path.isfile(csv_file_path)
    
    # Zapisz dane do pliku CSV
    df = pd.DataFrame([reading])
    df.to_csv(csv_file_path, mode='a', header=not file_exists, index=False)
    print(f"Dodano nowy pomiar do pliku CSV: {csv_file_path}")

    # Wczytanie danych z pliku CSV do tworzenia wykresu
    data_df = pd.read_csv(csv_file_path, parse_dates=['timestamp'])

    # Tworzenie wykresu na podstawie wszystkich zebranych danych
    plt.figure(figsize=(10, 6))
    plt.plot(data_df['timestamp'], data_df['humidity'], label='Humidity (%)', color='blue')
    plt.plot(data_df['timestamp'], data_df['temperature'], label='Temperature (°C)', color='red')

    # Dodanie kropek w miejscach wartości
    plt.scatter(data_df['timestamp'], data_df['humidity'], color='blue', alpha=0.7, label='Humidity Points')
    plt.scatter(data_df['timestamp'], data_df['temperature'], color='red', alpha=0.7, label='Temperature Points')

    # Ustawienia osi Y z gęstszymi znacznikami
    ax = plt.gca()  # Pobranie aktualnych osi
    ax.yaxis.set_major_locator(MultipleLocator(5))  # Znaczniki co 5 jednostek
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(axis='both')
    plt.title('Humidity and Temperature Over Time')
    plt.legend()

    # Zapisanie wykresu na RAM-dysku
    image_path = os.path.join(ramdisk_path, 'temp_humi_plot.png')
    plt.savefig(image_path)
    plt.close()  # Zamknięcie wykresu, by zwolnić pamięć
    print(f"Obraz zapisany na RAM-dysku: {image_path}")

    

    #Kopia zapasowa wyresów
    #if (timestamp.minute %10 == 0 ):
    #    copyfile(image_path, backup_filepath)
    #    copyfile(csv_file_path, backup_filepath_csv)
    #    print("Zapisano kopie zapasowe ")


