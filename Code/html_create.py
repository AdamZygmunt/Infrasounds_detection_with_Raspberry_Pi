# Skrypt odpowiedzialny za tworzenie pliku html na ramdysku

import os

html_file_path = '/mnt/ramdisk/sensor_data.html'

# Sprawdzenie, czy plik HTML już istnieje
if not os.path.isfile(html_file_path):

    # Tworzenie zawartości HTML tylko raz
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pl">
    <head>
        <meta http-equiv="refresh" content="10">
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sensor Data Plot</title>
    </head>
    <body>
        <div style ="text-align: center">
        <h1>Wizualizacja odczytów infradźwięków, temperatury i wilgotności powietrza</h1>
        <p>Wykres przedstawiający wilgotność i temperature powietrza</p><br>
        <img src="/Plots/temp_humi_plot.png" alt="Temperatura i wilgotność" ><br>
        <p>Wykres zmian ciśnienia w czasie</p><br>
        <img src="/Plots/today_raw_pressure.svg" alt="Today_raw_pressure" width="75%" height="75%"><br>
        <p>Pasma częstotliwości</p><br>
        <img src="/Plots/plotBands.png" alt="Pasma_częstotliwośc" width=60%" height="60%"><br>
        <p>Transformata Fouriera</p><br>
        <img src="/Plots/plotsSimpleFFT.png" alt="Transformata Fouriera" width="50%" height="50%"><br>
        <p>Transformata falkowa</p><br>
        <img src="/Plots/plotWaveletTransform.png" alt="Today_wavelet" width="60%" height="60%"><br>

    </body>
    </html>
    """

    # Zapisanie pliku HTML na RAM-dysku
    with open(html_file_path, 'w', encoding='utf-8') as file:
        file.write(html_content)
    print(f"Plik HTML zapisany na RAM-dysku: {html_file_path}")


