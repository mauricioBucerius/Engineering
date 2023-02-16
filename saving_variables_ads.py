import ads
import pwdatatools as pwd
import numpy as np
import os
import time
import datetime
from calc_time import get_time_dif

MAX_TIMESPAN = 15  # mins - time difference between now and the results in past


# package ads comes with numpy, os, sys -> using with ads.np, ads.os, ads.sys
# ADD_INFO = 'GND Drain'.lower().replace(' ', '_')
# ADD_INFO_STR = 'EM LO Merchand Balun'
# ADD_INFO_STR = 'EM Ring Mixer IF 1 GHz'#' P LO 20 dBm IF 1 GHz' #'EM IF 0.1 GHz'
# ADD_INFO_STR = 'EM active Balun final'
ADD_INFO_STR = 'startup active Balun late gate lower upper drain voltage end 5 ms'
# ADD_INFO_STR = 'startup active Balun voltage end 1 us'
LO_MATCH_STR = ''#'without MN'#'RF MN'
PARAMETER = 'Power Gate, parallel power resistor'.lower().replace(' ', '_').replace(".", "_")
ADD_INFO = '_' + ADD_INFO_STR.lower().replace(' ', '_').replace(".", "_")
LO_MATCH = '_' + LO_MATCH_STR.lower().replace(' ', '_').replace(".", "_")

# normed name of the files regarding the simulation method
DIR_S_PARAM = f'S_PARAM{ADD_INFO}_{LO_MATCH}_ums_11.npy'
DIR_S_PARAM_BALUN = f'S_PARAM_active_balun_{ADD_INFO}_ums_11.npy'
DIR_HB_PARAM = f'HB_sim_conversion_gain_isolation{ADD_INFO}{LO_MATCH}_ums_11.npy'
DIR_HB_PARAM_SINGLE = f'HB_sim_{PARAMETER}{ADD_INFO}{LO_MATCH}_ums_11.npy'
DIR_HB_PARAM_UP = f'HB_up_sim_conversion_gain_isolation{ADD_INFO}{LO_MATCH}_ums_11.npy'
DIR_HB_NOISE = f'HB_noise_figure{ADD_INFO}{LO_MATCH}_ums_11.npy'
DIR_LINEARITY = f'XDB_sim_linearity_gain{ADD_INFO}{LO_MATCH}_ums_11.npy'
DIR_DIPL = f'S_PARAM{ADD_INFO}_dipl{LO_MATCH}_ums_11.npy'
DIR_IMD = f'IMD{ADD_INFO}{LO_MATCH}_ums_11.npy'
DIR_TIME = f'TRANSIENT{ADD_INFO}{LO_MATCH}_ums_11.npy'


def seperate_data(data, num_col, **kwargs):
    """
        seperates the input data into num_col columns with aquidistant lenght

    Parameters
    ----------
    data : vector of float/complex
        DESCR.
    num_col : integer
        gives the number of columns in which the data vector has to be splitted.

    Returns
    -------
    data_sep : numpy.array
        well sperated numpy.array

    """
    sweep_param = None
    for value, key in kwargs.items():
        if key=='sweep_param':
            sweep_param = value
    
    if sweep_param is None:
        len_col = int(len(data)/num_col)
        data_sep = np.zeros([len_col, num_col], dtype=float)
        for i in range(num_col):
            for j in range(data_sep.shape[0]):
                data_sep[j][i] = data[j+len_col*i]
        return data_sep.transpose()
    else:
        # data.pop[-1]
        sweep_vec = []
        for col in data:
            if col[0] not in sweep_vec:
                print(col[0])
                sweep_vec.append(col[0])
            
        exit()
        # data_sep = np.zeros([data.shape[1]-1]
    
                            
def rm_trash(array_input):
    """
        removes all NaN/inf inside the Numpy Arrayy
    """
    # print(array_input[1, 6], ads.np.isinf(array_input[1, 6]))
    for idx_row, row in enumerate(array_input):
        for idx_col, col in enumerate(row):
            if ads.np.isinf(col) or ads.np.isnan(col):
                if idx_col > 0:
                    # print(idx_col, row[idx_col-1])
                    array_input[idx_row, idx_col] = row[idx_col-1]
                    # print(array_input[idx_row])
    return array_input


def write_data(path, data, flag):
    """
    Writes the data as numpy array in a dataset with numpy format "*.npy"

    Parameters
    ----------
    path : TYPE
        path to which the dataset will be saved
    data : TYPE
        the ads data
    flag : TYPE
        which kind of data will be saved
    """
    with open(path, 'wb') as output:
        ads.np.save(output, rm_trash(data))
        if len(ADD_INFO) == 0:
            print(f'{flag} Results are saved in File: "{path}"')
        else:
            print(f'{flag} Parameter: {ADD_INFO_STR} - Results are saved in File: "{path}"')


def read_data():
    """
    Reads the data, which is extracted from ads into files "ADS2Pyth_oneway.d",
    "ADS2Pyth_oneway.s". 

    Returns
    -------
    data_ads : TYPE
        DESCRIPTION.
    str_ads : TYPE
        DESCRIPTION.

    """
    d_file = "ADS2Pyth_oneway.d"
    s_file = "ADS2Pyth_oneway.s"
    creation_date_d = os.path.getmtime(d_file)
    creation_date_s = os.path.getmtime(s_file)
    creation_date = datetime.datetime.fromtimestamp(creation_date_d).strftime("%Y-%m-%d")
    if str(datetime.datetime.today().date()) == creation_date:
        # the date must be converted to a string, because the return parameter 
        # of date() is a date object and can't be compared!
        time_m_d = datetime.datetime.fromtimestamp(creation_date_d).strftime("%H:%M")
        time_m_s = datetime.datetime.fromtimestamp(creation_date_s).strftime("%H:%M")
        print("---------------------------------------------\n")
        print('Read all Files:')
        print(f"file: {d_file} - last updated: {time_m_d}")
        print(f"file: {s_file} - last updated: {time_m_s}")
        print("---------------------------------------------\n")
        data_ads, str_ads_list = ads.get()
        # str_ads = str_ads_list[0]
        # try: 
        #     params = str_ads_list[1]
        # except:
        #     params = None
            
        return data_ads, str_ads_list, time_m_d
    else:
        # if the data is generated not today
        time_m_d = datetime.datetime.fromtimestamp(creation_date_d).strftime("%H:%M - %d.%m.%Y")
        time_m_s = datetime.datetime.fromtimestamp(creation_date_s).strftime("%H:%M - %d.%m.%Y")
        print("---------------------------------------------\n")
        print('Read all Files:')
        print(f"file: {d_file} - last updated: {time_m_d}")
        print(f"file: {s_file} - last updated: {time_m_s}")
        print('Updated not today!')
        print("---------------------------------------------\n")
        
        data_ads, str_ads_list = ads.get()
        # str_ads = str_ads_list[0]
        
        # try: 
        #     params = str_ads_list[1]
        # except:
        #     params = None
            
        return data_ads, str_ads_list, None


def save_results():
    data_ads, str_ads_list, creation_time = read_data()
    # return data_ads, str_ads_list
    flag_result = ''
    if len(str_ads_list) >  1:
        print(str_ads_list, str_ads_list[0], str_ads_list[1])
        str_ads = str_ads_list[0]
        sweep_param = str_ads_list[1]
    else:
        str_ads = str_ads_list[0]
        sweep_param = None
        
    # return data_ads, str_ads
    # print(str_ads)
    if str_ads == 's_param':
        # S-Parameter simulation with small signal approximation
        flag_result = 'S-Parameter'
        pathfile = DIR_S_PARAM
        num_cols = 3
        
    elif str_ads == 'hb_sim':     
        # Harmonic Balance simulation with the frequency components
        flag_result = 'HB-Simulation'
        pathfile = DIR_HB_PARAM
        num_cols = 8
        
    elif str_ads == 'hb_sim_up':     
        # Harmonic Balance simulation with the frequency components
        flag_result = 'HB-Simulation Up Conversion'
        pathfile = DIR_HB_PARAM_UP
        num_cols = 2
        
    elif str_ads == 'hb_noise':
        # Harmonic Balance simulation with the frequency components
        flag_result = 'HB-Simulation Noise Figure'
        pathfile = DIR_HB_NOISE
        num_cols = 4
        
    elif str_ads == 'linearity_sim':
        # Gain @ P1dB, Output power @ P1dB and Input power @ P1dB simulation
        flag_result = 'XDB-Simulation'
        pathfile = DIR_LINEARITY
        num_cols = 4
        
    elif str_ads == 's_param_dibl':
        # Diplexer Simulation with only the output of the drain
        flag_result = 'Diplexer Spectrum'
        num_cols = 6
        pathfile = DIR_DIPL
        
    elif str_ads == 'IMD':
        # Intermodulation Distortion Simulation
        flag_result = 'IMD Simulation'
        pathfile = DIR_IMD
        num_cols = 11
    
    elif str_ads == 'time_domain':
        # Intermodulation Distortion Simulation
        flag_result = 'Time Domain Simulation'
        pathfile = DIR_TIME
        num_cols = 4
        # try: 
        #     data_ads = seperate_data(data, num_cols, sweep_param=sweep_param)
        # except:
        #     num_cols = 4
    elif str_ads == 's_param_balun':
        flag_result = 'S-Parameter Balun'
        pathfile = DIR_S_PARAM
        num_cols = 8
        
    elif str_ads == 'hb_sim_pg':  
        # Harmonic Balance simulation with the frequency components
        flag_result = 'HB-Simulation'
        pathfile = DIR_HB_PARAM_SINGLE
        num_cols = 3
        
    elif str_ads == 'active balun':
        # S Parameter Simulation of the amplitude and the phase imbalance
        flag_result = 'S-Parameter active BALUN'
        pathfile = DIR_S_PARAM_BALUN
        num_cols = 5
        
    elif str_ads == 'startup':
        # S Parameter Simulation of the amplitude and the phase imbalance
        flag_result = 'Time-Domain startlup'
        pathfile = DIR_TIME
        num_cols = 5
        
        
    # reshapes the data in case only one column is exportet
    if data_ads.shape[1] == 1:
        data_ads = seperate_data(data_ads, num_cols)
    # else:
    #     nan_counter = 0
    #     for col in range(data_ads.shape[1]):
    #         print(col) #, dir(data_ads[col]))
            
    #         if np.isnan(data_ads[col]):
    #             nan_counter += 1
    #             for row in range(data_ads.shape[0]):
    #                 if not np.isnan(data[row][col]):
    #                     print(row, col, data_ads[row][col])
    #                     exit()
                        
    # Save of the Simulation Results to a Numpy File
    if len(pathfile) >0: 
        if int(get_time_dif(datetime.datetime.fromtimestamp(time.time()).strftime("%H:%M"), creation_time, rv='mins')) >= MAX_TIMESPAN:
            # Asks user, if the data should be saved, when the data is older 
            # then the time MAX_TIMESPAN
            print(f'The data is older then {MAX_TIMESPAN} mins')
            print('Do you want to continue?')
            user_input = input('[y]/[n]: ').lower()
            if user_input == 'n':
                return data_ads, str_ads
        
        if ads.os.path.isfile(pathfile):        
            print(f'The file is already there! Do you want to overwrite the file "{ads.os.path.basename(pathfile)}"?')
            str_answer = input('[y]/[n]/[e]: ').lower().strip()
            if str_answer == 'y':
                write_data(pathfile, data_ads, flag_result)
            elif str_answer == 'n':
                
                print('Do you want to replace the additional informations?')
                str_add = input('[y]/[n]/[e]: ').lower()
                
                if str_add == 'y':
                    ADD_INFO_STR = input('new Info: ')
                    ADD_INFO = ADD_INFO_STR.lower().replace(' ', '_').replace(".", "_")
                    write_data(pathfile, data_ads, flag_result)
                    return data_ads, str_ads
                elif str_add == 'n':
                    return data_ads, str_ads
            elif str_answer == 'e':
                return data_ads, str_ads
            else:
                print("Wrong input!")
                print("Ending without saving...")
        else:
            write_data(pathfile, data_ads, flag_result)
            return data_ads, str_ads
    else: 
        print('No Data available to save...')
        

def open_dataset(filepath):
    print(dir(pwd))
    pwd.file_read(filepath)
    
    
if __name__ == '__main__':
    
    # reads and saves the ADS Data to numpy file
    data = save_results()
