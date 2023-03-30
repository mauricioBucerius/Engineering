import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from regression import *
import os

WIDTH_2_PLOTS = 6
WIDTH = 6
WIDTH_4_PLOTS = 5
HEIGHT = 4
DPI = 500

# Settings for all Plot
# with plt.rcParams could be adjusted the overall plot settings
# Legend
plt.rcParams["legend.fontsize"] = 8

# Axis
# print(plt.rcParams)
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

SKIPROWS = 27

def seperate_by_freq(data, freqs):
    """
    seperates the pandas array into 2 arrays regarding the frequency 4 and 13

    Parameters
    ----------
    data : pandas array
        DESCRIPTION.
    freqs : list of the frequencies
        DESCRIPTION.

    Returns
    -------
    frequency_4 : pandas array
        DESCRIPTION.
    frequency_13 : pandas array
        DESCRIPTION.

    """
    frequencies = [pd.DataFrame(columns=HEADER) for _ in freqs]
    
    print('Seperating the data by the frequency...')
    for idx, row in data.iterrows():
        # creating a pandas object, with the same headers like the original data
        pd_object = pd.DataFrame([row.values], columns=HEADER)
        for idx, freq in enumerate(freqs):
            if row[0] == freq:
                frequencies[idx] = pd.concat([frequencies[idx], pd_object], ignore_index=True)

    return frequencies


def seperate_by_power(data):
    """
    Seperates the Data of the pandas array into a list which matches the 
    requirements

    Parameters
    ----------
    data : pandas array
        DESCRIPTION.

    Returns
    -------
    list_values : TYPE
        DESCRIPTION.

    """
    list_values = []
    cur_val=0
    list_full = False
    sep_str = "Power"
    
    print(f'seperate the data by {sep_str} ...')
    for idx, row in data.iterrows():
        pd_object = pd.DataFrame([row.values], columns=HEADER)
        # print(idx, pd_object["Power"].values, cur_pwr, pd_object["Power"].values==cur_pwr)

        
        if idx==0:
            cur_val = pd_object[sep_str].values
            list_values.append(pd_object)
            cnt=0
            
        elif idx>=0 and pd_object[sep_str].values==cur_val:

            if not list_full:
                list_values.append(pd_object)
            else:
                cnt+=1
                list_values[cnt] = pd.concat([list_values[cnt], pd_object])
                
        elif not pd_object[sep_str].values==cur_val:
            list_full = True    # To stop appending new entries to the list list_values
            cnt=0               # resetting the counter
            
            cur_val = pd_object[sep_str].values     # overwriting the current value
            list_values[cnt] = pd.concat([list_values[cnt], pd_object])     # adding the current row to the first entry in the list
            
    return list_values
        

def seperate_by_att(data):
    """
    

    Parameters
    ----------
    data : pandas array
        DESCRIPTION.

    Returns
    -------
    list_values : list with N pandas array elements inside each element
        DESCRIPTION.

    """
    list_values = []
    cur_val=0
    sep_str = "Att"
    
    print(f'seperate the data by {sep_str} ...')
    for idx, row in data.iterrows():
        pd_object = pd.DataFrame([row.values], columns=HEADER)
    
        if idx==0:
            cur_val = pd_object[sep_str].values
            list_values.append(pd_object)
            cnt = 0
            
        elif idx>=0 and not pd_object[sep_str].values==cur_val:
            list_values[cnt] = pd.concat([list_values[cnt], pd_object])
                
        elif pd_object[sep_str].values==cur_val:
            cnt+=1              # resetting the counter            
            cur_val = pd_object[sep_str].values     # overwriting the current value
            list_values.append(pd_object)     # adding the current row to the first entry in the list
            
    return list_values


def eval_p_out(data, num):
    """
    Evaluates the seperated data to plot "straight lines" when the simulation 
    provides the corresponding data. Starting point for this evaluation is the 
    measurement of the output power when sweeping the input power with constant
    attenuator level or vice versa

    Parameters
    ----------
    data : list, list entries pandas DataFrame
        commits a list in which every entry desribes a DataFrame which contains
        the data to be evaluated.
    num : intetger
        Number of datapoints

    Returns
    -------
    list_values : list, list entries pandas DataFrame
        a pyramidal structure.

    """
    list_values = []
    print('Evaluate the Data')
    # lower triangle
    for i in range(num):
        if i == 0:
            # appending first entry to the list -> pyramidal structure
            pd_object = pd.DataFrame([data[0].iloc[0].values], columns=HEADER)
            list_values.append(pd_object)
        else:
            # generate 60 measure points
            # i = 1: takes from the 1st row the 2nd entry and from the 2nd row the 1st entry
            # i = 2: takes from the 1st row the 3rd entry, from the 2nd row the 2nd entry, from the 3rd row the 1st entry
            # i = 3: and so on for all 60 rows -> looks like a triangle
            
            cnt = i
            for j in range(i+1):         
                pd_object = pd.DataFrame([data[j].iloc[cnt].values], columns=HEADER)
                if j == 0:
                    list_values.append(pd_object)
                else:
                    list_values[i] = pd.concat([list_values[i], pd_object])
                cnt -= 1
    # upper triangle -> reveres the upper triangle
    for i in range(1, num):
        # takes one value less, because the middle won't taken twice
        # ignores the first list entry
        
        # i = 0: takes from the 1st row nothing, from the 2nd row the last entry
        # i = 1: takes from the 3rd row the last entry, from the 4th entry the last - 1 entry
        # i = 2: and so on for 59 rows -> skips the middle lane at 60
        cnt = i
        # print(i)
        for j in range(num-1, i,-1):
            # print(cnt, j)
            pd_object = pd.DataFrame([data[cnt].iloc[j].values], columns=HEADER)
            if j == num-1:
                list_values.append(pd_object)
            else:
                list_values[num+i-1] = pd.concat([list_values[num+i-1], pd_object])     # additional to num entries in list_values the current entry is represented by the num+1-th entry
            cnt += 1
        # return list_values
    return list_values
        

def calc_delta(x, y):
    # calculating the difference between the first und the last value
    # respectively for each axis   
    delta_x = abs(x.max() - x.min()).values
    delta_y = abs(y.max() - y.min()).values/delta_x # normiert auf 1 dB Eingangsleistung
    
    # Creating DataFrame respectively for each axis    
    pd_x = pd.DataFrame([delta_x], columns=[x.columns])
    pd_y = pd.DataFrame([delta_y], columns=[y.columns])
    return pd.concat([pd_x, pd_y], axis=1)


def calc_error(y_reg, y_data):
    """
    calculates the L2-Norm between the y value from the regression and the 
    y value of the data. 

    Parameters
    ----------
    y_reg : numpy.ndarray
        same length like y_data.
    y_data : numpy.ndarray
        the y value of the measuremnt, which is approximated with the regression

    Returns
    -------
    scalar
        represents the error of the regression

    """
    return np.sum(np.abs(y_reg - y_data)**2)/len(y_reg)


def calc_regression_array(data, order):
    print('Fit a regression curve with order', str(order))
    list_fitted = []
    list_delta =  []
    list_error = []
    
    for entry in data:
        if entry.shape[0] > 4:
            # all fitting array should be as an row vector not column
            x_data = np.array([entry["Power"].values])          # x values
            y_data = np.array([entry["Pout_sat"].values])       # y values
            y_fitted = get_regression(x_data, y_data, order)    # ext func
            
            # creating DataFrames, each per row -> to avoid error by creating 
            # DataFrame with not symmetric shape
            x_data_frame = pd.DataFrame(x_data.tolist()[0], columns=['Power'])
            y_data_frame = pd.DataFrame(y_fitted.tolist(), columns=['Pout_sat'])
            
            # add the dataFrame together
            fitted_data_frame = pd.concat([x_data_frame, y_data_frame], axis=1)   
            list_delta.append(calc_delta(x_data_frame, y_data_frame))
            
            # return x_data_frame, y_data_frame
            # append to the list
            list_fitted.append(fitted_data_frame)
            list_error.append(calc_error(y_fitted, y_data))
        else:
            list_error.append(None)
            list_fitted.append(None)
            list_delta.append(None)
        
        
        # mapping the error list into a numpy array
        error = np.zeros(len(list_error), dtype=float)
        for idx, entry in enumerate(list_error):
            if entry is not None:
                error[idx] = entry
            
    return list_fitted, list_delta, error
    

def create_name(str_list):
    name = []
    for entry in str_list:
        try:
            # all letters should be lower case and avoiding space
            if isinstance(entry, str):
                name.append(entry.lower().replace(' ', '_').replace('.', '_').replace(',', '_'))
            elif not isinstance(entry, list):
                # TypeError when not a string -> float or sth else
                name.append(str(entry))
            else:
                name.append(create_name(entry))
                # pass
        except AttributeError:
            # when entry is a list -> recursive function
            name.append(create_name(entry))
            # pass
        
    # return a string which is seperated by underscore
    return '_'.join(name)

    
def plot_eval(data_eval, data_fit, ylim=None, xlim=None,
              save='Output_power_constant_lines', reg_info='',
              title=None, reg=False, width=WIDTH):
    
    if reg:
        list_name = [save, reg_info]
    else:
        list_name = [save]
        
    cmap_eval = plt.get_cmap('viridis', len(data_eval))   # Colormap e with N Entries and the Colormap name 'viridis' (default)
    
    # Normalizer
    cmap_min = data_eval[0]["Pout_sat"].values[0]                   # minimum of the values
    cmap_max = data_eval[-1]["Pout_sat"].values[0]                 # maximum of the values
    norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)     # normalization to the minimum and maximum values
    # ticks = [round(value,0) for value in np.linspace(cmap_min, cmap_max, 7)]
    
    # creating ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap_eval, norm=norm)    # Creating the Color representation which is along the plot
    sm.set_array([])
    
    # Plotting Output Power vs. Input Power - every schar represents an attenuator value
    fig = plt.figure(figsize=[width, HEIGHT], dpi=DPI)
    print('Plotting the Data for the straight lines...')
    for idx, entry in enumerate(data_eval):
        if idx%3==0:
            # plots only every 3rd Value - to improve the visibility
            plt.plot(entry["Power"], entry["Pout_sat"], c=cmap_eval(idx))
            if data_fit[idx] is not None and reg:
                # plots the regression curve only when entry not None and 
                # the input reg
                plt.plot(data_fit[idx]["Power"], data_fit[idx]["Pout_sat"], c=cmap_eval(idx))
    
    plt.xlabel('Input Power/dBm')
    plt.ylabel('Output Power/dBm')
    
    if xlim is None:
        plt.xlim([-27, 7])
    else:
        list_name.append(xlim)
        plt.xlim(xlim)
    
    if ylim is None:
        plt.ylim([-15, 42])
    else:  
        list_name.append(ylim)
        plt.ylim(ylim)
    plt.grid()
    plt.colorbar(sm, ticks=[round(value,0) for value in np.linspace(cmap_min, cmap_max, 7)], 
                  label='Output Power/dBm')
    
    if reg:
        list_name.append('regression')
    if title is not None:
        plt.title(title)
        list_name.append(title)
    plt.savefig(f"{create_name(list_name)}.png")


def plot_curves(data, ylim=None, xlim=None, save='Output_power',
                title=None, reg=False, xlabel=None, ylabel=None, 
                sweep='', skip_rows=1, width=WIDTH_2_PLOTS):
    """
    

    Parameters
    ----------
    data : list with Pandas DataFrame Entries
        contains the data what you want to plot. This parameter is a list with 
        DataFrames. Which are sorted a priori.
    ylim : list of two integer, optional
        to limit the y axis in the plot, e.g. [-10, 5] -> limits the y lim 
        between -10 and 5
    xlim : list of two integer, optional
        to limit the x axis in the plot
    save : string, optional
        DESCRIPTION. The default is 'Output_power'.
    title : TYPE, optional
        DESCRIPTION. The default is None.
    reg : TYPE, optional
        DESCRIPTION. The default is False.
    xlabel : TYPE, optional
        DESCRIPTION. The default is None.
    ylabel : TYPE, optional
        DESCRIPTION. The default is None.
    sweep : TYPE, optional
        DESCRIPTION. The default is ''.
    skip_rows : TYPE, optional
        DESCRIPTION. The default is 1.
    width : TYPE, optional
        DESCRIPTION. The default is WIDTH_2_PLOTS.

    Returns
    -------
    None.

    """
    list_name = [save, sweep]
    
    N = len(data)
    # print('creating Color Map for Input Power Sweep...')
    cmap = plt.get_cmap('viridis', N)   # Colormap e with N Entries and the Colormap name 'viridis' (default)
    
    if sweep == 'Att':
        # With Attenuator sweep -> color map shows the Power level
        var = 'Power'
        cmap_label = 'Input Power/dBm'
        xlabel='Attenuator/dB'
    elif sweep == 'Power':
        # with Power sweep -> color map shows the Attenuator level
        var = 'Att'
        cmap_label = 'Attenuator/dB'

    else:
        var = None
    
    if var is not None:
        cmap_min = data[0][var].values[0]                  # minimum of the values
        cmap_max = data[-1][var].values[0]                 # maximum of the values
        norm = mpl.colors.Normalize(vmin=cmap_min, vmax=cmap_max)     # normalization to the minimum and maximum values
        # ticks = [round(value,0) for value in np.linspace(cmap_min, cmap_max, 7)]
        
        # creating ScalarMappable
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)    # Creating the Color representation which is along the plot
        sm.set_array([])
    
    # Plotting Output Power vs. Input Power - every schar represents an attenuator value
    fig = plt.figure(figsize=[width, HEIGHT], dpi=DPI)
    print('Plotting the Data for the Input Power Sweep...')
    for idx, entry in enumerate(data):
        if idx%skip_rows==0:
            # plots only every 3rd Value - to improve the visibility
            if sweep=='Att' and var is not None:
                # the x value should be negativ, due to the absolut representation
                # of the attenuator level
                plt.plot(-entry[sweep], entry["Pout_sat"], c=cmap(idx))
            elif var is not None:
                plt.plot(entry[sweep], entry["Pout_sat"], c=cmap(idx))
            else:
                plt.plot(entry["Pout_sat"], c=cmap(idx))
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel('Input Power/dBm')
         
    if ylabel is not None:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('Output Power/dBm')
     
    if xlim is None:
        plt.xlim([-30, 10])
    else:
        list_name.append(xlim)
        plt.xlim(xlim)
     
    if ylim is None:
        plt.ylim([-15, 42])
    else:  
        list_name.append(ylim)
        plt.ylim(ylim)
        
    plt.grid()
    plt.colorbar(sm, ticks=[round(value,0) for value in np.linspace(cmap_min, cmap_max, 7)], 
                 label=cmap_label)    # creates the colormap
    if title is not None:
        list_name.append(title)
        plt.title(title)
        
    plt.savefig(f"{create_name(list_name)}.png")

    
if __name__ == '__main__':
    filepath = ''
    
    limit_error = 5.1
    reg_order = 1
    if reg_order == 1:
        reg_str = 'linear'
    elif reg_order == 2:
        reg_str = 'quadratic'
    elif reg_order == 3: 
        reg_str = 'cubic'
    else: 
        reg_str = f'reg {reg_order}th'
        
        
    file = pd.read_excel(filepath, skiprows=SKIPROWS)   # reads the excel sheet
    filename = os.path.basename(filepath)               # extracts the filename from the path
    file_id = filename.split('_')[1]
    
    FOLDER = filename.split('.')[0]                     # extracts the FOLDER name - filename without the .xlsx ending
    
    if not os.path.isdir(FOLDER):
        # creates a new folder, in which all data will be saved
        print(f'Creating new Folder: "{FOLDER}"')
        os.mkdir(FOLDER)
        
    print(f'Change Working Directory to "{FOLDER}"')
    os.chdir(FOLDER) # change the working directory to the new folder
    HEADER = file.columns.values  # extracts the header from the file

    # Data seperated by Frequency
    data_seperated = seperate_by_freq(file, [4, 13])
    
    # Data seperated by Input Power
    data_pwr_4 = seperate_by_power(data_seperated[0])        # Frequency 4
    data_pwr_13 = seperate_by_power(data_seperated[1])        # Frequency 13
    
    data_att_4 = seperate_by_att(data_seperated[0])        # Frequency 4
    data_att_13 = seperate_by_att(data_seperated[1])        # Frequency 13
    
    N = len(data_pwr_4)                     # number of curves - equal for all sweeps
    
    # Evaluation of the output Power 
    eval_data_4 = eval_p_out(data_att_4, N)
    eval_data_13 = eval_p_out(data_att_13, N)
    
    
    # Creating Fitting Curves
    eval_fitted_4, eval_delta_4, eval_error_4 = calc_regression_array(eval_data_4, reg_order)
    eval_fitted_13, eval_delta_13, eval_error_13 = calc_regression_array(eval_data_13, reg_order)
    
    fig = plt.figure(figsize=[WIDTH, HEIGHT], dpi=DPI)
    for idx, entry in enumerate(eval_delta_4):
        if entry is not None:
            plt.scatter(idx, entry["Pout_sat"], color='steelblue')
    plt.xlabel('Index')
    plt.ylabel('delta Output Power pro 1dB Ausgangsleistung')
    plt.grid()
    plt.ylim([0, 0.42])
    plt.title('Frequenz 4')
    savename = f'{file_id}_delta_4_{reg_str}' 
    
    plt.savefig(f'{savename}.png')
    
    
    fig = plt.figure(figsize=[WIDTH, HEIGHT], dpi=DPI)
    for idx, entry in enumerate(eval_delta_13):
        if entry is not None:
            plt.scatter(idx, entry["Pout_sat"], color='steelblue')
    plt.xlabel('Index')
    plt.ylabel('delta Output Power pro 1dB Ausgangsleistung')
    plt.grid()
    plt.ylim([0, 0.42])
    plt.title('Frequenz 13')
    plt.savefig(f'delta_13_{reg_str}.png')
    
