# Kod do pomiaru i rysowania wykresów na żywao

import os
import time
import smbus
import numpy as np
import matplotlib
import gc

from datetime import datetime
from obspy import UTCDateTime, read, Trace, Stream
from obspy.signal.filter import bandpass, lowpass, highpass
from obspy.signal.tf_misfit import cwt
from obspy.imaging.cm import obspy_sequential

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from threading import Thread
from shutil import copyfile

#do monitorowania uzycia pamieci
import psutil

matplotlib.use('Agg')  # Przełącza na backend bez GUI

# Ścieżka RAM-dysku (lub lokalna, jeśli RAM-dysk nie jest skonfigurowany)
ramdisk_path = '/mnt/ramdisk'
ramdisk_plots_path = '/mnt/ramdisk/Plots'
os.makedirs(ramdisk_path, exist_ok=True)
os.makedirs(ramdisk_plots_path, exist_ok=True)

html_file_path = os.path.join(ramdisk_path, 'sensor_data.html')



def read_pressure_from_sensor():
	#this function specifically reads data over i2c from a 
	#DLVRF50D1NDCNI3F Amphenol mems differential pressure sensor 
	
	# this routine will need to be changed for a diffent input device
	#such as a voltage via an a/d convertor
	# note: the return value 'pressure' is a floting point number
	
    raw_data = bytearray
    z=bytearray
    raw_pressure= int
    raw_data=[]

    z = [0,0,0,0]
    z = sensor.read_i2c_block_data(addr, 0, 4) #offset 0, 4 bytes

    d_masked = (z[0] & 0x3F)
    raw_data = (d_masked<<8) +  z[1]    

    raw_pressure= int(raw_data)
    pressure = (((raw_pressure-8192.0)/16384.0)*250.0 *1.25) 
    
    return pressure

def plot_daily_raw_pressures(st, frq_low_cut, frq_high_cut):
	#produces a 24hour obspy 'dayplot' saved in svg format.
	#two copies are produced Today.svg and 'date'.svg

	if (st[0].stats.npts > 1000):
		try:
			print("raw pressure plotting...")
			print(f"Active figures: {plt.get_fignums()}")
			here = os.path.dirname(os.path.realpath(__file__))
			start_time = st[0].stats.starttime
			year = str(start_time.year)
			month = str(start_time.month)
			day = str(start_time.day)
			hour = str(start_time.hour)
			minute = str(start_time.minute)
			#year_dir =  'Data' + '/' + year
			station_info= str(st[0].stats.station)+'-'+str(st[0].stats.channel)+'-'+str(st[0].stats.location)

			save_dir=  'Plots/' 
			date_string = str(year) + '_' + str(month) + '_' + str(day) + str(hour) + '_' + str(minute)
			plot_title= 'Raw Pressure ' +':::'+station_info+':::'+' '+date_string+'  '+str(frq_low_cut) + '-' + str(frq_high_cut) + ' Hz'
			filename1 = ('/mnt/ramdisk/Plots/today_raw_pressure.svg')
			filename2 = '/mnt/ramdisk/Plots/' + date_string+'__'+station_info+'__Daily_Raw_Pressure.svg'
			filename3 = '/home/adam/Desktop/Wiatraki/' + date_string+'__'+station_info+'__Daily_Raw_Pressure.svg'
			st.plot(type="dayplot",outfile=filename1, title=plot_title, data_unit='$\Delta$Pa', interval=15, right_vertical_labels=False, one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)

			plt.close()
			
			#copyfile(filename1, filename2) # interval 15 daje lepszy wgląd
			#copyfile(filename1, filename3)
			print ('Plotting of daily raw pressure completed')
		except (ValueError,IndexError):
			print('an  error on plotting dayly raw pressures!')
	
	print(f"Active figures: {plt.get_fignums()}")
	return None

### --------------------------- Funkcje pozostałe --------------------------- ###

def plotBands(tr):
    print('Filtering and plotting bands')
    print(f"Active figures: {plt.get_fignums()}")
    #plt.close('All')
    #fig.clf()
    #plt.close(fig)
    gc.collect() # cczyszczenie pam podrecznej - to pomoglo finalnie rozwiązać problem nakładania na jednej kanwie 2 wykresów
    plt.rcdefaults() #resetuje ustawienia matplotlib - wykresy się nakładały na siebie
    
    trace_end_time = tr.stats.endtime
    trace_start_time = trace_end_time - 10*60 #ostatnie 10 minut
    tr_trim = tr.trim(starttime = trace_start_time, endtime = trace_end_time) #przyciecie danych
    
    legendLoc = 'upper left'
    xMin = 0.0
    xMax = 10.0

    N = len(tr_trim.data)

    samplingFreq = tr_trim.stats.sampling_rate

    yscale = 5.0

    lowCut1 = 0.01
    highCut1 = 0.5
    tr1 = tr_trim.copy()
    tr1.filter('bandpass', freqmin=lowCut1, freqmax=highCut1, corners=4, zerophase=True)

    lowCut2 = 0.5
    highCut2 = 1.0
    tr2 = tr_trim.copy()
    tr2.filter('bandpass', freqmin=lowCut2, freqmax=highCut2, corners=4, zerophase=True)

    lowCut3 = 1.0
    highCut3 = 2.0
    tr3 = tr_trim.copy()
    tr3.filter('bandpass', freqmin=lowCut3, freqmax=highCut3, corners=4, zerophase=True)

    lowCut4 = 2.0
    highCut4 = 3.0
    tr4 = tr_trim.copy()
    tr4.filter('bandpass', freqmin=lowCut4, freqmax=highCut4, corners=4, zerophase=True)

    lowCut5 = 3.0
    highCut5 = 5.0
    tr5 = tr_trim.copy()
    tr5.filter('bandpass', freqmin=lowCut5, freqmax=highCut5, corners=5, zerophase=True)

    lowCut6 = 5.0
    highCut6 = 10.0
    tr6 = tr_trim.copy()
    tr6.filter('bandpass', freqmin=lowCut6, freqmax=highCut6, corners=4, zerophase=True)

    lowCut7 = 10.0
    highCut7 = 15.0
    tr7 = tr_trim.copy()
    tr7.filter('bandpass', freqmin=lowCut7, freqmax=highCut7, corners=4, zerophase=True)

    #os czasu w minutach
    x = np.linspace(0, (N / samplingFreq / 60), N)
    #x = np.divide(x, 3600)  # 3600 dla godzin                                    

    
    fig = plt.figure(figsize=(12,8)) 
    fig.suptitle(str(tr.stats.starttime) + ' Filtered ')
    #fig.canvas.set_window_title('start U.T.C. - ' + str(tr.stats.starttime))

    plt.subplots_adjust(hspace=0.3) #0.001
    gs = gridspec.GridSpec(8, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, tr_trim)
    ax0.set_xlim(xMin, xMax)
    ax0.legend(['raw data'], loc=legendLoc, fontsize=10)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x, tr1)
    ax1.legend([str(lowCut1) + '-' + str(highCut1) + 'Hz'], loc=legendLoc, fontsize=10)

    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.plot(x, tr2)
    ax2.legend([str(lowCut2) + '-' + str(highCut2) + 'Hz'], loc=legendLoc, fontsize=10)

    ax3 = plt.subplot(gs[3], sharex=ax1)
    ax3.plot(x, tr3)
    ax3.legend([str(lowCut3) + '-' + str(highCut3) + 'Hz'], loc=legendLoc, fontsize=10)

    ax4 = plt.subplot(gs[4], sharex=ax1)
    ax4.plot(x, tr4)
    ax4.legend([str(lowCut4) + '-' + str(highCut4) + 'Hz'], loc=legendLoc, fontsize=10)

    ax5 = plt.subplot(gs[5], sharex=ax1)
    ax5.plot(x, tr5)
    ax5.legend([str(lowCut5) + '-' + str(highCut5) + 'Hz'], loc=legendLoc, fontsize=10)

    ax6 = plt.subplot(gs[6], sharex=ax1)
    ax6.plot(x, tr6)
    ax6.legend([str(lowCut6) + '-' + str(highCut6) + 'Hz'], loc=legendLoc, fontsize=10)

    ax7 = plt.subplot(gs[7], sharex=ax1)
    ax7.plot(x, tr7)
    ax7.legend([str(lowCut7) + '-' + str(highCut7) + 'Hz'], loc=legendLoc, fontsize=10)

    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels() + ax3.get_xticklabels() \
                  + ax4.get_xticklabels() + ax5.get_xticklabels() + ax6.get_xticklabels()

    plt.setp(xticklabels, visible=False)

    ax7.set_xlabel('Time [min]', fontsize=12)

    fig.tight_layout()

        # Save figure
    save_path = '/mnt/ramdisk/Plots/plotBands.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # zamykanie kanw 
    plt.close(fig)
    plt.close('All')
    print(f"Wykres zapisano w: {save_path}")
    del fig
    plt.close()
    print(f"Active figures: {plt.get_fignums()}")

    #fig.show()


def plotWaveletTransform2(tr): # to jest prawidłowo działająca funkcja
    print('Calculating Wavelet Transform')
    print(f"Active figures: {plt.get_fignums()}")
    trace_end_time = tr.stats.endtime
    trace_start_time = trace_end_time - 10*60 #ostatnie 10 minut
    tr_trim = tr.trim(starttime = trace_start_time, endtime = trace_end_time) #przyciecie danych
    
    N = len(tr_trim.data)  # Number of samplepoints
    dt = tr.stats.delta

    x0 = 0
    x1 = N - 1

    t = np.linspace(x0, x1, num=N)
    t1 = np.divide(t, (tr_trim.stats.sampling_rate * 60))

    fig = plt.figure()
    fig.suptitle('Wavelet Transform ' + str(tr_trim.stats.starttime.date), fontsize=12)
    #fig.canvas.set_window_title('Wavelet Transform ' + str(tr.stats.starttime.date))
    # ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)

    print ("x1", x1, "len t", len(t), "len t1", len(t1))

    ax1.plot(t1, tr_trim.data, 'k')
    ax1.set_ylabel(r'$\Delta$P - Pa')

    f_min = 0.1
    f_max = 15.0

    scalogram = cwt(tr_trim.data[x0:x1], dt, 8, f_min, f_max)
    
    t1 = t1[:scalogram.shape[1]] # dopasowanie dlugosci tablicy t1 do liczby kolumn w skalogramie
    
    x, y = np.meshgrid(t1, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

    ax2.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential, shading='nearest') #dodano arg shading

    ax2.set_xlabel("Time  [min]")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_yscale('log')
    ax2.set_ylim(f_min, f_max)

    # Save figure
    save_path = '/mnt/ramdisk/Plots/plotWaveletTransform.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    del fig
    print(f"Wykres zapisano w: {save_path}")
    print(f"Active figures: {plt.get_fignums()}")




def plotsSimpleFFT2(tr):          # to jest poprawnie działająca funkcja
    print('plotting FFT....')
    print(f"Active figures: {plt.get_fignums()}")
    #print(tr.stats)

    plt.close('All')
    
    dt = tr.stats.delta
    Fs = 1 / dt  # częstotliwość próbkowania
    tracestart = tr.stats.starttime
    traceend = tr.stats.endtime

    # Ustawienie początku i końca analizy
    endSec = tr.stats.npts * dt  # czas trwania sygnału w sekundach
    startSec = max(0, endSec - 10 * 60)  # ostatnie 10 minut (600 sekund)

    # Wyodrębnienie danych dla tego zakresu czasu
    t = np.arange(startSec, endSec, dt)  # oś czasu
    start_index = int(startSec / dt)
    end_index = int(endSec / dt)
    sigTemp = tr.data[start_index:end_index]  # dane z ostatnich 10 minut
    s = sigTemp[:len(t)]
    
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

    # Wykres sygnału w czasie:
    axes[0, 0].set_title("Signal")
    axes[0, 0].plot(t, s, color='C0')
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude [P - Pa]")
    
    # Wykresy FFT:
    axes[1, 0].set_title("Magnitude Spectrum")
    axes[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')
    axes[1, 0].set_xlabel("Frequency [Hz]")

    axes[1, 1].set_title("Log. Magnitude Spectrum")
    axes[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')
    axes[1, 1].set_xlabel("Frequency [Hz]")

    axes[2, 0].set_title("Phase Spectrum")
    axes[2, 0].phase_spectrum(s, Fs=Fs, color='C2')
    axes[2, 0].set_xlabel("Frequency [Hz]")

    axes[2, 1].set_title("Power Spectrum Density")
    #axes[2, 1].set_xlabel("Frequency [Hz]")
    axes[2, 1].psd(s, 256, Fs, Fc=1)
    axes[2, 1].set_xlabel("Frequency [Hz]")

    axes[0, 1].remove()  # usuń puste pole wykresu

    fig.tight_layout()

    # Zapis wykresu
    save_path = '/mnt/ramdisk/Plots/plotsSimpleFFT.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Wykres zapisano w: {save_path}")
    
    plt.close(fig)
    del fig
    print(f"Active figures: {plt.get_fignums()}")
    
    return

###---------------------------ooo0ooo---------------------------
def create_mseed(readings, start_time, end_time, n_samples, station_id, station_channel, location):

    true_sample_frequency = float(n_samples) / (end_time - start_time)

    # Fill header attributes
    stats = {'network': 'IR', 'station': station_id, 'location': location,
         'channel': station_channel, 'npts': n_samples, 'sampling_rate': true_sample_frequency,
         'mseed': {'dataquality': 'D'}}
    # set current time
    stats['starttime'] = start_time
    stats['endtime'] = end_time
    st = Stream([Trace(data=readings[0:n_samples], header=stats)])
    return st


###---------------------------ooo0ooo---------------------------
def save_and_plot_all(daily_readings, day_start_time, n_daily_samples, station_id, station_channel,location, sample_end_time):	
	
	#create daily stream and plot
	st = create_mseed(daily_readings, day_start_time, sample_end_time, n_daily_samples, station_id, station_channel, location)
	st.filter('bandpass', freqmin=frq_low_cut, freqmax=frq_high_cut, corners=4, zerophase=True)
	plot_daily_raw_pressures(st, frq_low_cut, frq_high_cut)
	#plotWaveletTransform(st[0], frq_low_cut, frq_high_cut)	
	plotBands(st[0])
	plotWaveletTransform2(st[0])
	plotsSimpleFFT2(st[0])
	
	#plot_daily_pwr(st, frq_low_cut, frq_high_cut, dt)
	return None




###--------------------------Main Body--------------------------
###-                                                           -
###-                                                           -
###--------------------------+++++++++--------------------------

sensor = smbus.SMBus(1)    # nr magistrali 
addr = 0x28				   # addres czujnika (z dokumentacji)


os.chdir('/home/adam/Desktop/testowy/NA_BAZIE')  #ścieżka gdzie będa się tworzyły foldery do ewentualnego backupu


# below are station parameters for your station. see SEED manual for
# more details http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf

#-- station parameters
station_id= 'STARF' #to można zmienić na dowolne swoje,poniższe parametry są ustawione dla częstotliwości próbkowania (do 80 Hz)
# channel B=broadband 10-80Hz sampling, D=pressure sensor, F=infrasound
station_channel = 'BDF'  #see SEED format documentation
location = '01'  # 2 digit code to identify specific sensor rig
station_network='IR'

frq_sampling=40.00
n_seconds_in_sampling_period = 1.00/frq_sampling
seconds_in_day=3600*24

n_target_hourly_samples=int(frq_sampling*120*1.1) #bo 2 minut zakres
n_target_daily_samples=int(n_target_hourly_samples*24*30) # dalem zapas na miesiac ale i tak poki co tyle nie bedzie
daily_readings= np.zeros(n_target_daily_samples, np.float32) 

n_daily_samples = 0


minute_start_time = UTCDateTime()
hourly_start_time = UTCDateTime()
day_start_time = UTCDateTime()

dt = 5.0  # time interval seconds to calculate running mean for acoustic pwr
frq_low_cut=0.1         # low cut-off frequency
frq_high_cut=20.0        # high cut-off frequency

### main


while 1:
	
	time.sleep(n_seconds_in_sampling_period)
	
	try:
		temp_pressure=read_pressure_from_sensor()
		daily_readings[n_daily_samples] = temp_pressure
		n_daily_samples = n_daily_samples + 1

		
	except IOError:
		print('read error')
		daily_readings[n_daily_samples] = 0.00#hourly_readings[n_hourly_samples] = 0.00
		n_daily_samples = n_daily_samples + 1
		
	timestamp=datetime.now()


	if (UTCDateTime().minute != minute_start_time.minute): #(timestamp.minute %5 == 0): #(UTCDateTime().hour != hourly_start_time.hour):#(UTCDateTime().minute != minute_start_time.minute):
		
		if(timestamp.minute %2 == 0):
		    sample_end_time=UTCDateTime()
        # funkcja zapisu
		    thread_plot_and_save = Thread(target=save_and_plot_all, args=(daily_readings,  day_start_time, n_daily_samples, station_id, station_channel,location, sample_end_time))#, dt
		    thread_plot_and_save.start()
		    print(timestamp)

		minute_start_time = UTCDateTime()

