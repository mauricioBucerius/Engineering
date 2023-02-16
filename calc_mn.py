import numpy as np
from scipy.constants import pi
import sys
# sys.path.insert(0, 'K:\\05_Python\\converting-main')
from prefix_converting import zahl2prefix

Rs = 50     # source impendance
Rl = 10     # load impendance
f = 7.5e9   # carrier frequency
B = 3e9     # bandwidth -> important for q
MN = 'pi-network' # kind of matching network


def calc_r_par(q_p, r_ser):
    return (1 + q_p**2)*r_ser

def calc_x_par(q_p, x_ser):
    return x_ser*(1 + 1/q_p**2)

def print_comps(L_ser, C_par, C_ser, L_par):
    print('Schaltung 1:')
    print('------------')
    print(f'L_ser = {zahl2prefix(L_ser)}H')
    print(f'C_par = {zahl2prefix(C_par)}F\n')
      
    print('Schaltung 2:')
    print('------------')
    print(f'C_ser = {zahl2prefix(C_ser)}H')
    print(f'L_par = {zahl2prefix(L_par)}F')
    
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
        
    else:
        # x_par is parallel to the source -> reduces the source and enlarges load
        q = calc_q(r_l, r_par=r_s)
        x_ser = q*r_l
        x_par = r_s/q
    # print(q)
    return x_ser, x_par

def get_pi_network(q, r_l, r_s):
    r_vir = calc_r_vir(q, Rs, Rl)
    # print(r_vir)
    # 1. L-Network - close to the source
    x_ser1, x_par1 = get_l_network(r_vir, r_s)   # Matching from source to r_vir
    x_ser2, x_par2 = get_l_network(r_l, r_vir)   # Matching from r_vir to load
    # print(x_ser1, x_par1)
    # print(x_ser2, x_par2)
    # print(r_vir)
    return x_ser1, x_ser2, x_par1, x_par2
    
def get_t_network():
    pass

def main():
    q_tot = f/B     # carrier and bandwidth of the circuit determine the overall quality factor
    
    if MN == 'l-network':
        # berechnet ein L-Netzwerk zur Anpassung
        x_ser, x_par = get_l_network()
        # Serielle Bauteile
        L_ser = x_ser/(2*pi*f)
        C_ser = 1/(2*pi*f*x_ser)
        
        # parallele Bauteile
        C_par = 1/(2*pi*f*x_par)
        L_par = x_par/(2*pi*f)
        print(f'Fehlanpassung wird mittels L-Netzwerk über Q = {zahl2prefix(q_tot)}kompensiert:\n')
        print_comps(L_ser, C_par, C_ser, L_par)
        
    else:
        if Rs > Rl:
            print('Source > Load -> Pi-Network')
            print('---------------------------')
            
            x_ser1, x_ser2, x_par1, x_par2 = get_pi_network(q_tot, Rl, Rs)
            
            # Calculating the components
            C_par1 = 1/(2*pi*f*x_par1)
            C_par2 = 1/(2*pi*f*x_par2)
            L_ser = x_ser1/(2*pi*f) + x_ser2/(2*pi*f)
            print('Circuit 1:')
            print(f'C_par1 = {zahl2prefix(C_par1)}F')
            print(f'L_ser = {zahl2prefix(L_ser)}H')
            print(f'C_par2 = {zahl2prefix(C_par2)}F')
            
            L_par1 = x_par1/(2*pi*f)
            L_par2 = x_par2/(2*pi*f)
            C_ser = 1/((2*pi*f*x_ser1) + (2*pi*f*x_ser1))
            print('\nCircuit 2:')
            print(f'L_par1 = {zahl2prefix(L_par1)}H')
            print(f'C_ser = {zahl2prefix(C_ser)}F')
            print(f'L_par2 = {zahl2prefix(L_par2)}H')
        elif Rs < Rl:
            print('Source < Load -> T-Network')
            print('---------------------------')
        
if __name__ == '__main__':
    main()
