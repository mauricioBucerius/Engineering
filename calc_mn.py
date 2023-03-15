import numpy as np
from scipy.constants import pi
import sys
# sys.path.insert(0, 'K:\\05_Python\\converting-main')
from prefix_converting import zahl2prefix


def fano_crit(Q_0, Q_L):
    """
    with the bode-fano criterium this gives the upper limitation about the best
    possible reflection coefficient, which is with a matching network possible

    Parameters
    ----------
    Q_0 : float
        DESCRIPTION.
    Q_L : float
        loaded quality factor.

    Returns
    -------
    TYPE
        DESCRIPTION.
 
    """
    return np.exp(-pi*Q_0/Q_L)

def get_Q_0(omega_0, delta_omega):
    return omega_0/delta_omega

def get_omega_0(f1, f2):
    return np.sqrt((2*pi)**2 *f1 * f2)

def get_delta_omega(f1, f2):
    return np.abs( 2*pi*(f1 - f2) )

def calc_r_par(q_p, r_ser):
    return (1 + q_p**2)*r_ser

def calc_x_par(q_p, x_ser):
    return x_ser*(1 + 1/q_p**2)

def print_comps(**kwargs):
    
    for key, value in kwargs.items():
        if key == 'L_ser':
            if value > 0:
                print(f'L_ser = {zahl2prefix(value)}H')
            else:
                print(f'C_ser = {zahl2prefix(-value)}F')
                
        elif key == 'C_par':
            if value > 0:
                print(f'C_par = {zahl2prefix(value)}F')
            else:
                print(f'L_par = {zahl2prefix(-value)}H')
                
        elif key == 'L_par':
            if value > 0:
                print(f'L_par = {zahl2prefix(value)}H')
            else:
                print(f'C_par = {zahl2prefix(-value)}F')
            
        elif key == 'C_ser':
                if value > 0:
                    print(f'C_ser = {zahl2prefix(value)}F')
                else:
                    print(f'L_ser = {zahl2prefix(-value)}H')
    print()
        
    # print('Schaltung 2:')
    # print('------------')
    # print(f'C_ser = {zahl2prefix(C_ser)}H')
    # print(f'L_par = {zahl2prefix(L_par)}F')
    
def calc_q(r, **kwargs):
    # r = None
    w = None
    c = None
    l = None
    r_par = None
    r_ser = None
    circuit = None
    
    for key, value in kwargs.items():
        # print(key, value)
        if key=='c':
            c = value
        elif key=='l':
            l = value
        elif key=='circuit':
            circuit=value
        elif key=='r_ser':
            r_ser = value
        elif key=='r_par':
            r_par = value
        # elif key=='r':
        #     r = value
        elif key=='w':
            w = value
        
    # if r_par > r_ser and r_ser is not None and r_par is not None:
    if r_par is not None:
        return np.sqrt(r_par/r-1)
    elif r_ser is not None:
        return np.sqrt(r_ser/r-1)
    # elif r_par < r_ser and r_ser is not None and r_par is not None:
    #     return np.sqrt(r/r_ser-1)
    
    if circuit == 'parallel':
        # quality factor für den Parallelschwingkreis
        if l is not None:
            return r/(w*l)
        elif c is not None:
            return r*w*c
    elif circuit == 'series':
        # quality factor für den Serienschwingkreis
        if l is not None:
            return (w*l)/r
        elif c is not None:
            return 1/(r*w*c)

def calc_r_vir(q_tot, r_s, r_l):
    a = (16*q_tot**4 + 16*q_tot**2)
    b = -(8*q_tot**2*(r_s-r_l) + 16*q_tot**2*r_l)
    c = (r_s - r_l)**2
    # print(a, b, c, b**2-4*a*c)
    r_vir1 = (-b + np.sqrt(b**2-4*a*c)) / (2 * a)
    r_vir2 = (-b - np.sqrt(b**2-4*a*c)) / (2 * a)
    if r_vir1 < 0:
        return r_vir2
    elif r_vir2 < 0:
        return r_vir1
    q1 = np.sqrt(r_s/r_vir1-1)
    q2 = np.sqrt(r_s/r_vir2-1)
    # print(r_vir1, r_vir2)
    if q1<q2:
        return r_vir1
    elif q2<q1:
        return r_vir2
    
def get_l_network(r_l, r_s):
    if r_l > r_s:
        # an source wird der serielle branch angeschlossen
        # x_par is parallel to the load -> reduces the r_l and enlarges r_s
        q = calc_q(r_s, r_ser=r_l)
        x_ser = q*r_s       
        x_par = r_l/q
        Q = r_l/x_par
    else:
        # x_par is parallel to the source -> reduces the source and enlarges load
        q = calc_q(r_l, r_par=r_s)
        x_ser = q*r_l
        x_par = r_s/q
        Q = r_s/x_par
    # print(q)
    return x_ser, x_par, Q

def get_pi_network(q, r_l, r_s):
    r_vir = calc_r_vir(q, r_s, r_l)
    # print(r_vir)
    # 1. L-Network - close to the source
    x_ser1, x_par1 = get_l_network(r_vir, r_s)   # Matching from source to r_vir
    x_ser2, x_par2 = get_l_network(r_l, r_vir)   # Matching from r_vir to load
    # print(x_ser1, x_par1)
    # print(x_ser2, x_par2)
    # print(r_vir)
    return x_ser1, x_ser2, x_par1, x_par2
    
def get_t_network(q, r_l, r_s):
    r_vir = calc_r_vir(q, r_s, r_l)
    
    print(r_vir, q)
    
    # 1. L-Network - close to the source (series Xl1 to source)
    X_ser1, x_par1 = get_l_network(r_l, r_s)
    
def calc_resonator(f_res, **kwargs):
    # inline calculation of the component value
    calc_comp = lambda f, comp: 1/(2*pi*f)**2 * 1/comp  
    
    C = None
    L = None
    
    for key, value in kwargs.items():
        if key == 'C':
            C = value
        elif key == 'L':
            L = value
    if C is None and L is None:
        C = 1e-12
        
    if C is None and L is not None:
        C = calc_comp(f_res, L)
    elif L is None and C is not None:
        L = calc_comp(f_res, C)
        
    return C, L


def main():
    Rs = 50             # source impendance
    
    # Größen des pHEMT bei 8 Finger und 125 um Gatebreite
    R_phemt = 3.2                   # Ohm - Eingangswiderstand des pHEMT
    C_phemt = 2.29e-12              # F - Eingangskapazität des pHEMT
    
    # R_phemt = 62.5                   # Ohm - Eingangswiderstand des pHEMT
    # C_phemt = 34.8e-15         # F - Eingangskapazität des pHEMT
    
    f0 = 12e9            # carrier frequency
    B = 6e9             # bandwidth -> important for q
    MN = 'l-network'   # kind of matching network
    N_HARM = 7          # Anzahl an Harmonics für den Class-F Amplifier
    
    f1 = f0 - B/2       # obere Grenzfrequenz
    f2 = f0 + B/2       # untere Grenzfreqnez
    q_tot = f0/B     # carrier and bandwidth of the circuit determine the overall quality factor
    Z_l = R_phemt - 1j * 1 / (2 * pi *f0 * C_phemt)     # load impendance
    
    Z_l = 0.296 * 50
    
    # print(Z_l)
    # get_t_network(q_tot, Rl, Rs)
    
    
    Q_in = 1/(2*pi*f0*R_phemt*C_phemt)
    omega_0 = get_omega_0(f1, f2)
    delta_omega = get_delta_omega(f1, f2)
    gamma_max = 10*np.log10(fano_crit(get_Q_0(omega_0, delta_omega), Q_in))
    
    print('--------------------------------------------------------------------\n')
    print('Bode-Fano Criterion')
    print(f'Gamma = {round(gamma_max,1)} dB \nB = {zahl2prefix(B)}Hz')
    print(f'total Q = {round(q_tot,1)}')
    print('--------------------------------------------------------------------')
    
    # C_f0, L_f0 = calc_resonator(f0, L=500e-12)
    # C_f3, L_f3 = calc_resonator(3*f0, L=500e-12)
    
    # print(f'C = {zahl2prefix(C_f3)}F')
    # print(f'L = {zahl2prefix(L_f3)}H')
    
    if MN == 'l-network':
        
        if np.imag(Z_l) < 0:
            L_res = -np.imag(Z_l)/(2*pi*f0)
            X_L_res = 2*pi*f0*L_res
            
            print('Resonating the capacitor:')
            print(f'L = {zahl2prefix(L_res)}H \n')
        else:
            X_L_res = 0
            
        # berechnet ein L-Netzwerk zur Anpassung
        x_ser, x_par, q_calc = get_l_network(Rs, np.real(Z_l))
        
        if Rs > np.real(Z_l):
            # parallel branch is up to the source
            print(f'Source > Load - parallel to the Source')
            if X_L_res > 0:
                # Parallele Bauteile zur Source
                C_par = 1/(2*pi*f0*x_par)
                L_par = x_par/(2*pi*f0)
                
                # Serielle Spule/Kapazität zur Last
                L_ser = (x_ser+X_L_res)/(2*pi*f0)
        
                res_x = X_L_res-x_ser
                if res_x > 0:     
                    C_ser = 1/(2*pi*f0*res_x)
                else:
                    C_ser = res_x/(2*pi*f0)
            else:
                L_ser = x_ser/(2*pi*f0)
                C_ser = 1/(2*pi*f0*x_ser)
                
                # parallele Bauteile
                C_par = 1/(2*pi*f0*x_par)
                L_par = x_par/(2*pi*f0)

        else: 
            # parallel branch is up to the load      
            L_ser = (X_L_res-x_ser)/(2*pi*f0)
            C_ser = 1/(2*pi*f0*x_ser)
            
            C_par = 1/(2*pi*f0*x_par)
            L_par = x_par/(2*pi*f0)
        
        print(f'Fehlanpassung wird mittels L-Netzwerk über Q = {round(q_calc,1)} kompensiert:\n')
        print('Schaltung 1:')
        print('------------')
        print_comps(L_ser=L_ser, C_par=C_par)
        
        print('Schaltung 2:')
        print('------------')
        print_comps(C_ser=C_ser, L_par=L_par)
        
    else:
        if Rs > Z_l:
            print('Source > Load -> Pi-Network')
            print('---------------------------')
            
            x_ser1, x_ser2, x_par1, x_par2 = get_pi_network(q_tot, Z_l, Rs)
            
            # Calculating the components
            C_par1 = 1/(2*pi*f0*x_par1)
            C_par2 = 1/(2*pi*f0*x_par2)
            L_ser = x_ser1/(2*pi*f0) + x_ser2/(2*pi*f0)
            print('Circuit 1:')
            print(f'C_par1 = {zahl2prefix(C_par1)}F')
            print(f'L_ser = {zahl2prefix(L_ser)}H')
            print(f'C_par2 = {zahl2prefix(C_par2)}F')
            
            L_par1 = x_par1/(2*pi*f0)
            L_par2 = x_par2/(2*pi*f0)
            C_ser = 1/((2*pi*f0*x_ser1) + (2*pi*f0*x_ser1))
            print('\nCircuit 2:')
            print(f'L_par1 = {zahl2prefix(L_par1)}H')
            print(f'C_ser = {zahl2prefix(C_ser)}F')
            print(f'L_par2 = {zahl2prefix(L_par2)}H')
        elif Rs < Z_l:
            print('Source < Load -> T-Network')
            print('---------------------------')
        
        
if __name__ == '__main__':
    main()
