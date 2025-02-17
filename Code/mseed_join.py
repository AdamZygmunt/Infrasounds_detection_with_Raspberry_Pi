# Skrypt służący do łączenia kilku plików .mseed w jeden w celu analizy danych z dłuższego przedziału czasu

from obspy import read, Stream

# Lista plików do połączenia (ścieżki)
file_list = [
    "/home/adam/Desktop/Wiatraki/Data/2025/1/2/21_31-21_46.mseed",
    "/home/adam/Desktop/Wiatraki/Data/2025/1/2/21_48-22_03.mseed"
]

# Zbieranie częstotliwości próbkowania
sampling_rates = []
for file in file_list:
    st = read(file)
    for trace in st:
        sampling_rates.append(trace.stats.sampling_rate)

# Obliczanie docelowej częstotliwości próbkowania (np. średnia)
target_sampling_rate = sum(sampling_rates) / len(sampling_rates)

# Inicjalizacja pustego strumienia
combined_stream = Stream()

# Dopasowanie częstotliwości próbkowania i łączenie danych
for file in file_list:
    st = read(file)
    for trace in st:
        # Dopasowanie częstotliwości próbkowania do docelowej
        trace.interpolate(sampling_rate=target_sampling_rate)
    combined_stream += st

# Łączenie danych czasowo
combined_stream.merge(method=1, fill_value='interpolate')

# Zapisanie do nowego pliku
output_file = "/home/adam/Desktop/Wiatraki/Data/2025/1/2/21_31-22_03.mseed"
combined_stream.write(output_file, format="MSEED")

print(f"Połączone pliki zapisano w: {output_file}")
