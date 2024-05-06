import scipy.integrate as spi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pyDOE import *
import pickle
from IPython.display import display



plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':14})

pd.options.display.float_format = "{:3e}".format



# Import calibrated parameters and define starting population
with open('Data/calibrated_parameters.pkl', 'rb') as data: 
    beta, INPUT1 = pickle.load(data)

with open('Data/literature_parameters.pkl', 'rb') as data2: 
    gamma_u, gamma_v, gamma_b, mu_u, mu_v, mu_b, f, eta_v, eta_b = pickle.load(data2)

time_horizons = [7, 15]

n_groups = 4
n_obj = 4

INPUT = [i for i in INPUT1[:len(INPUT1)-1]]
INPUT.extend([INPUT1[len(INPUT1)-1]*f[0], INPUT1[len(INPUT1)-1]*f[1],
              INPUT1[len(INPUT1)-1]*f[2], INPUT1[len(INPUT1)-1]*f[3]])

pop_size = 20201249



# LYs
L1 = 69.29
L2 = 50.28
L3 = 29.81
L4 = 12.95

# QALYs
q1L1 = 63.02
q2L2 = 45.04
q3L3 = 27.50
q4L4 = 11.22

L = (L1,L2,L3,L4)
qL = (q1L1,q2L2,q3L3,q4L4)

title_obj = ['Minimizing cumulative infections', 'Minimizing deaths', 'Minimizing life years lost', 'Minimizing QALYs lost']



# ODEs
def diff_eqs(INP, t):  
    '''The main set of equations'''
    Y = np.zeros((28)) 
    V = INP
    
    for i in range(4): 
        # Force of infection by group
        lambda_i = 0
        for j in range(4): 
            lambda_i += beta[i,j]*(V[j+12] + (1-eta_v[j])*V[j+16] + (1-eta_b[j])*V[j+20])
        
        # Susceptibles
        # Unvaccinated susceptible
        Y[i] = -V[i]*lambda_i
        # Vaccinated susceptible
        Y[i+4] = -(1-eta_v[i])*V[i+4]*lambda_i + gamma_u[i]*V[i+12]
        # Boosted susceptible
        Y[i+8] = -(1-eta_b[i])*V[i+8]*lambda_i + gamma_v[i]*V[i+16] + gamma_b[i]*V[i+20]
        
        # Infected
        # Unvaccinated infected
        Y[i+12] = V[i]*lambda_i - (gamma_u[i]+mu_u[i])*V[i+12]
        # Vaccinated infected
        Y[i+16] = (1-eta_v[i])*V[i+4]*lambda_i - (gamma_v[i]+mu_v[i])*V[i+16]
        # Boosted infected
        Y[i+20] = (1-eta_b[i])*V[i+8]*lambda_i - (gamma_b[i]+mu_b[i])*V[i+20]
        
        # Dead
        Y[24+i] = mu_u[i]*V[i+12] + mu_v[i]*V[i+16] + mu_b[i]*V[i+20]
    
    return Y   # For odeint

# Model
def model(INPUT,ND,v): # vaccinated (4) and boosted (4)
    t_start = 0.0; t_end = ND; t_inc = 1 #output in days
    t_range = np.arange(t_start, t_end+t_inc, t_inc)
    
    INPUT_v = [INPUT[i] for i in range(len(INPUT))]
    for i in range(8): 
        INPUT_v[i] -= v[i] 
        INPUT_v[i+4] += v[i]
    for i in range(8): 
        if INPUT_v[i]<0: 
            print("error:", i, INPUT_v[i])
            break
    
    RES = spi.odeint(diff_eqs,INPUT_v,t_range)
    
    all_infections = 0
    S = np.zeros(ND+1)
    I = np.zeros(ND+1)
    D = np.zeros(ND+1)
    for i in range(12): 
        S += RES[:,i]
        I += RES[:,i+12]
        all_infections += np.sum(RES[:,i+12])
    for i in range(4): 
        D += RES[:,i+24]
    
    end_pop = RES[ND,:]


    end_deaths = D[ND]-INPUT[24]-INPUT[25]-INPUT[26]-INPUT[27] 
    end_lifeyears = L1*(RES[ND,24]-INPUT[24])+L2*(RES[ND,25]-INPUT[25])+L3*(RES[ND,26]-INPUT[26])+L4*(RES[ND,27]-INPUT[27]) 
    end_qalys = q1L1*(RES[ND,24]-INPUT[24])+q2L2*(RES[ND,25]-INPUT[25])+q3L3*(RES[ND,26]-INPUT[26])+q4L4*(RES[ND,27]-INPUT[27]) 
    
    return(end_pop, [all_infections*pop_size,end_deaths*pop_size,end_lifeyears*pop_size,end_qalys*pop_size])
    


# Threshold
def thresholds(parameters,INPUT,T):
    susceptible = INPUT[0:12]
    infections = INPUT[12:24]
    lambda_ = np.zeros(len(parameters)) #force of infection
    alpha1 = np.zeros(n_groups)
    alpha2 = np.zeros(n_groups)
    for i in range(len(parameters)): 
        for j in range(len(parameters)): 
            lambda_[i] += parameters[i,j]*(infections[j]+(1-eta_v[j])*infections[j+4]+(1-eta_b[j])*infections[j+8])
    for k in range(n_groups): 
        alpha1[k] = max(0,min(susceptible[k], 
                          susceptible[k]-( (gamma_u[k]+mu_u[k])*infections[k]*T-infections[k] )/(lambda_[k]*T)))
        alpha2[k] = max(0,min(susceptible[k+4], susceptible[k+4]-( (gamma_v[k]+mu_v[k])*infections[k+4] )/lambda_[k]                         + infections[k+4]/((1-eta_v[k])*lambda_[k]*T)))
            
    threshold = (sum(alpha1+alpha2))/sum(susceptible[0:8])
    
    return(threshold, np.concatenate((alpha1, alpha2)))

for T in time_horizons: 
    print("T =", T, "days")
    threshold, alpha = thresholds(beta,INPUT,T)
    print('Proportion of the unvaccinated and vaccinated population that can be vaccinated:', threshold)



# Optimal decisions given by approximations   
def conditions(parameters,infections):
    p_infections = np.zeros(2*len(parameters)) #[v,b]
    p_deaths = np.zeros(2*len(parameters))
    p_ly = np.zeros(2*len(parameters))
    p_qaly = np.zeros(2*len(parameters))
    lambda_ = np.zeros(len(parameters)) #force of infection

    for i in range(len(parameters)): 
        for j in range(len(parameters)): 
            lambda_[i] += parameters[i,j]*(infections[j]+(1-eta_v[j])*infections[j+4]+(1-eta_b[j])*infections[j+8])
        
        p_infections[i] = eta_v[i]*lambda_[i]
        p_infections[i+4] = (eta_b[i]-eta_v[i])*lambda_[i]

        p_deaths[i] = (mu_u[i]-(1-eta_v[i])*mu_v[i])*lambda_[i]
        p_deaths[i+4] = ((1-eta_v[i])*mu_v[i]-(1-eta_b[i])*mu_b[i])*lambda_[i]

        p_ly[i] = L[i]*(mu_u[i]-(1-eta_v[i])*mu_v[i])*lambda_[i]
        p_ly[i+4] = L[i]*((1-eta_v[i])*mu_v[i]-(1-eta_b[i])*mu_b[i])*lambda_[i]

        p_qaly[i] = qL[i]*(mu_u[i]-(1-eta_v[i])*mu_v[i])*lambda_[i]
        p_qaly[i+4] = qL[i]*((1-eta_v[i])*mu_v[i]-(1-eta_b[i])*mu_b[i])*lambda_[i]
        
    return(p_infections, p_deaths, p_ly, p_qaly) 

    
def insight(parameters,INPUT):     
    infections = INPUT[12:24]
    p = conditions(parameters,infections)
    
    groups_to_vaccinate = np.zeros((n_obj,2*n_groups))
    for i in range(n_obj):
        groups_to_vaccinate[i,:] = np.argsort(-p[i]) 
    return(groups_to_vaccinate)

groups_to_vaccinate = insight(beta, INPUT) 

print('Optimal vaccination order for first time period (any time horizon, any level of vaccination)')
for i in range(n_obj):
    temp = []
    for j in range(2*n_groups): 
        if groups_to_vaccinate[i,j] < 4: 
            temp.append('v{}'.format(int(groups_to_vaccinate[i,j]+1)))
        else: 
            temp.append('b{}'.format(int(groups_to_vaccinate[i,j]-3)))
    print(title_obj[i]) 
    print(temp)


# # First time period


def approx_function(N,parameters,INPUT,T): 
    groups_to_vaccinate = insight(parameters,INPUT)
    threshold, alpha = thresholds(parameters,INPUT,T)
    v_approx = np.zeros((n_obj,2*n_groups))
    approx_optimal = np.zeros(n_obj)

    for i in range(n_obj): 
        N_temp = N
        v_temp = np.zeros(2*n_groups) 
        for j in range(n_groups): 
            opt_group = int(groups_to_vaccinate[i,j])
            v_temp[opt_group] = min(alpha[opt_group], N_temp)
            N_temp = N_temp - min(alpha[opt_group], N_temp)

        v_approx[i,:] = v_temp            
        end_pop, objectives = model(INPUT,T,v_temp)
        approx_optimal[i] = objectives[i]
        
    return(v_approx, approx_optimal, groups_to_vaccinate)



def delta(N,parameters,INPUT,T): 
    S1u, S2u, S3u, S4u, S1v, S2v, S3v, S4v, S1b, S2b, S3b, S4b, I1u, I2u, I3u, I4u, I1v, I2v, I3v, I4v, I1b, I2b, I3b, I4b, D10, D20, D30, D40 = INPUT
        
    v1_range = np.arange(0, S1u, 0.01)
    v2_range = np.arange(0, S2u, 0.01)
    v3_range = np.arange(0, S3u, 0.01)
    v4_range = np.arange(0, S4u, 0.01)
    
    b1_range = np.arange(0, S1v, 0.01)
    b2_range = np.arange(0, S2v, 0.01)
    b3_range = np.arange(0, S3v, 0.01)
    b4_range = np.arange(0, S4v, 0.01)
    
    difference = np.zeros(n_obj)
    
    v_approx, approx_optimal, groups_to_vaccinate = approx_function(N,beta,INPUT,T)
    v_opt, numerical_optimal, temp = approx_function(N,beta,INPUT,T)
    
    for v1 in v1_range: 
        for v2 in v2_range: 
            for v3 in v3_range: 
                for v4 in v4_range: 
                    for b1 in b1_range: 
                        for b2 in b2_range: 
                            for b3 in b3_range: 
                                b4 = N-v1-v2-v3-v4-b1-b2-b3 
                                if b4 >= 0 and b4 <= S4v: 
                                    v = (v1,v2,v3,v4,b1,b2,b3,b4)
                                    end_pop, objectives = model(INPUT,T,v)
                                    for i in range(n_obj): 
                                        if objectives[i] < numerical_optimal[i]: 
                                            numerical_optimal[i] = objectives[i]
                                            v_opt[i,:] = v
                                            
    for i in range(n_obj):
        difference[i] = (approx_optimal[i] - numerical_optimal[i])/numerical_optimal[i]*100
    
    return(numerical_optimal, difference, v_opt, v_approx, groups_to_vaccinate)



N_range = np.arange(0, 0.101, 0.01)
v_T1 = []

print('First time period')
t = 0
for T in time_horizons: 
    print('T =', T, 'days')
    difference = np.zeros((len(N_range),n_obj))
    v_approx = np.zeros((len(N_range),n_obj,2*n_groups))  
    v_opt = np.zeros((len(N_range),n_obj,2*n_groups))
    numerical_optimal = np.zeros((len(N_range),n_obj))
    
    k=0
    for N in N_range: 
        numerical_optimal[k,:], difference[k,:], v_opt[k,:,:], v_approx[k,:,:], temp = delta(N,beta,INPUT,T)
        k+=1
                
    plt.figure(figsize=(5,4))
    for i in range(difference.shape[1]): #looping through objectives
        plt.plot(N_range, difference[:,i], 'o', label = title_obj[i], alpha=0.7)
        plt.xlabel(r'$\frac{N}{P}$')
        plt.title("Percentage difference")
    plt.legend(loc = 'upper right', bbox_to_anchor=(2, 0.75))
    plt.ylim((-0.1,1))
    plt.savefig("Output/SIS/difference_T1={}.png".format(T), bbox_inches='tight')
    plt.show()
    
    fig, axs = plt.subplots(4, 2, figsize=(14,20))
    
    for i in range(v_opt.shape[1]): #looping through objectives
        for j in range(n_groups): #looping through Su
            axs[i,0].plot(N_range, v_opt[:,i,j], 'o', label = r'$v^{}$'.format(j+1), alpha=0.7)
        for j in range(n_groups): #looping through Sv
            axs[i,0].plot(N_range, v_opt[:,i,j+4], 'o', label = r'$b^{}$'.format(j+1), alpha=0.7)
        axs[i,0].set_xlabel(r'$\frac{N}{P}$')
        axs[i,0].set_ylabel('Optimal vaccine allocation')
        axs[i,0].title.set_text(title_obj[i] + ' (exhaustive search)')
        axs[i,0].legend(bbox_to_anchor=(1.05, 0.75))

        for j in range(n_groups): #looping through Su
            axs[i,1].plot(N_range, v_approx[:,i,j], 'o', label = r'$v^{}$'.format(j+1), alpha=0.7)
        for j in range(n_groups): #looping through Sv
            axs[i,1].plot(N_range, v_approx[:,i,j+4], 'o', label = r'$b^{}$'.format(j+1), alpha=0.7)
        axs[i,1].set_xlabel(r'$\frac{N}{P}$')
        axs[i,1].set_ylabel('Optimal vaccine allocation')
        axs[i,1].title.set_text(title_obj[i] + ' (approximation)')
        axs[i,1].legend(bbox_to_anchor=(1.05, 0.75))

    fig.tight_layout(pad = 3)
    plt.show()

    v_T1.append(v_opt[10])
    


# # Second time period


def delta_2(N,parameters,INPUT,T): 
    difference = np.zeros(n_obj)
    v_opt = np.zeros((n_obj,2*n_groups))
    v_approx = np.zeros((n_obj,2*n_groups))
    numerical_optimal = np.zeros(n_obj)
    groups_to_vaccinate = np.zeros((n_obj,2*n_groups))
    
    v_approx1, approx_optimal1, groups_to_vaccinate1 = approx_function(N,parameters,INPUT,T)
    
    for i in range(n_obj):
        INPUT_T, temp = model(INPUT,T,v_approx1[i,:])
        threshold, alpha = thresholds(parameters, INPUT_T, T)
        if N > threshold: 
            print("Error: N_max > threshold2")
            break
        numerical_optimal2, difference2, v_opt2, v_approx2, groups_to_vaccinate2 = delta(N,parameters,INPUT_T,T) 
        numerical_optimal[i] = numerical_optimal2[i]
        difference[i] = difference2[i]
        v_opt[i,:] = v_opt2[i,:]
        v_approx[i,:] = v_approx2[i,:]
        groups_to_vaccinate[i,:] = groups_to_vaccinate2[i,:]

    return(numerical_optimal, difference, v_opt, v_approx, groups_to_vaccinate)
    


v_T2 = []

print('Second time period')
t = 0
for T in time_horizons: 
    print('T =', T, 'days')
    difference2 = np.zeros((len(N_range),n_obj))
    v_approx2 = np.zeros((len(N_range),n_obj,2*n_groups))  
    v_opt2 = np.zeros((len(N_range),n_obj,2*n_groups))
    numerical_optimal2 = np.zeros((len(N_range),n_obj))
    
    k=0
    for N in N_range: 
        numerical_optimal2[k,:], difference2[k,:], v_opt2[k,:,:], v_approx2[k,:,:], group_to_vaccinate2 = delta_2(N,beta,INPUT,T)
        k+=1
                
    plt.figure(figsize=(5,4))
    for i in range(difference2.shape[1]): #looping through objectives
        plt.plot(N_range, difference2[:,i], 'o', label = title_obj[i], alpha=0.7)
        plt.xlabel(r'$\frac{N}{P}$')
        plt.title("Percentage difference")
    plt.legend(loc = 'upper right', bbox_to_anchor=(2, 0.75))
    plt.ylim((-0.1,1))
    plt.savefig("Output/SIS/difference_T2={}.png".format(T), bbox_inches='tight')
    plt.show()
    
    fig, axs = plt.subplots(4, 2, figsize=(14,20))
    
    for i in range(v_opt.shape[1]): #looping through objectives
        for j in range(n_groups): #looping through Su
            axs[i,0].plot(N_range, v_opt2[:,i,j], 'o', label = r'$v^{}$'.format(j+1), alpha=0.7)
        for j in range(n_groups): #looping through Sv
            axs[i,0].plot(N_range, v_opt2[:,i,j+4], 'o', label = r'$b^{}$'.format(j+1), alpha=0.7)
        axs[i,0].set_xlabel(r'$\frac{N}{P}$')
        axs[i,0].set_ylabel('Optimal vaccine allocation')
        axs[i,0].title.set_text(title_obj[i] + ' (exhaustive search)')
        axs[i,0].legend(bbox_to_anchor=(1.05, 0.75))

        for j in range(n_groups): #looping through Su
            axs[i,1].plot(N_range, v_approx2[:,i,j], 'o', label = r'$v^{}$'.format(j+1), alpha=0.7)
        for j in range(n_groups): #looping through Sv
            axs[i,1].plot(N_range, v_approx2[:,i,j+4], 'o', label = r'$b^{}$'.format(j+1), alpha=0.7)
        axs[i,1].set_xlabel(r'$\frac{N}{P}$')
        axs[i,1].set_ylabel('Optimal vaccine allocation')
        axs[i,1].title.set_text(title_obj[i] + ' (approximation)')
        axs[i,1].legend(bbox_to_anchor=(1.05, 0.75))
        
    fig.tight_layout(pad = 3)
    plt.show()
    
    v_T2.append(v_opt2[10])
    
    



print('Optimal vaccination order for second time period')

for i in range(n_obj):
    print(title_obj[i])
    for T in time_horizons: 
        current = np.zeros(8)
        print('T =', T, 'days')
        k=0
        for N in N_range: 
            temp1, temp2, temp3, temp4, group_to_vaccinate = delta_2(N,beta,INPUT,T)
            k+=1
            temp = []
            for j in range(2*n_groups): 
                if group_to_vaccinate[i,j] < 4: 
                    temp.append('v{}'.format(int(group_to_vaccinate[i,j]+1)))
                else: 
                    temp.append('b{}'.format(int(group_to_vaccinate[i,j]-3)))
            if not np.array_equal(group_to_vaccinate[i,:], current): 
                print('Switch point N =', N, ', approximated optimal order is: \n', temp)
                current = group_to_vaccinate[i,:] 
                


# Plots for the paper 

fig, axs = plt.subplots(1, 2, figsize=(14,5))

for i in range(1): #looping through objectives
    for j in range(n_groups): #looping through Su
        axs[0].plot(N_range, v_opt2[:,i,j], 'o', label = r'$v^{}$'.format(j+1), alpha=0.7)
    for j in range(n_groups): #looping through Sv
        axs[0].plot(N_range, v_opt2[:,i,j+4], 'o', label = r'$b^{}$'.format(j+1), alpha=0.7)
    axs[0].set_xlabel(r'$\frac{N}{P}$')
    axs[0].set_ylabel('Optimal vaccine allocation')
    axs[0].set_title(title_obj[i] + ' (exhaustive search)', pad = 10)
    axs[0].legend(bbox_to_anchor=(1.05, 0.92))
    axs[0].set_ylim((-0.005,0.08))
    
    for j in range(n_groups): #looping through Su
        axs[1].plot(N_range, v_approx2[:,i,j], 'o', label = r'$v^{}$'.format(j+1), alpha=0.7)
    for j in range(n_groups): #looping through Sv
        axs[1].plot(N_range, v_approx2[:,i,j+4], 'o', label = r'$b^{}$'.format(j+1), alpha=0.7)
    axs[1].set_xlabel(r'$\frac{N}{P}$')
    axs[1].set_ylabel('Optimal vaccine allocation')
    axs[1].set_title(title_obj[i] + ' (approximation)', pad = 10)
    axs[1].legend(bbox_to_anchor=(1.05, 0.92))
    axs[1].set_ylim((-0.005,0.08))

fig.tight_layout(pad = 3)
plt.savefig("Output/SIS/v_inf_T2={}.png".format(T), bbox_inches='tight')
plt.show()


# # Calculate outcomes averted with approximation


dict = {'T':[],
        '% of vaccines':[],  
        'Scenario':[], 
        'Infections':[], 
        'Deaths':[], 
        'LYs':[], 
        'QALYs':[]} 

df = pd.DataFrame(dict)
N = 0.1
obj_vacc = np.zeros(n_obj)
obj_num = np.zeros(n_obj)

for k in range(len(time_horizons)): 
    T = time_horizons[k]
    
    # proportional vaccines
    v_prop1 = [INPUT[i]/np.sum(INPUT[0:8])*N for i in range(8)]
    INPUT_T, obj_eq1 = model(INPUT,T,v_prop1)
    v_prop2 = [INPUT_T[i]/np.sum(INPUT_T[0:8])*N for i in range(8)]
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_prop2)
    obj_prop = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
    # proportional initial doses only 
    v_prop_initial1 = np.zeros(8)
    for i in range(4): 
        v_prop_initial1[i] = INPUT[i]/np.sum(INPUT[0:4])*N 
    INPUT_T, obj_eq1 = model(INPUT,T,v_prop_initial1)
    v_prop_initial2 = np.zeros(8)
    for i in range(4): 
        v_prop_initial2[i] = INPUT_T[i]/np.sum(INPUT_T[0:4])*N 
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_prop_initial2)
    obj_prop_initial = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
    # proportional booster doses only 
    v_prop_booster1 = np.zeros(8)
    for i in range(4): 
        v_prop_booster1[i+4] = INPUT[i+4]/np.sum(INPUT[4:8])*N 
    INPUT_T, obj_eq1 = model(INPUT,T,v_prop_booster1)
    v_prop_booster2 = np.zeros(8)
    for i in range(4): 
        v_prop_booster2[i+4] = INPUT_T[i+4]/np.sum(INPUT_T[4:8])*N 
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_prop_booster2)
    obj_prop_booster = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
    # allocate to group 4 (proportionally between vaccine and boosters) - two time periods
    v_g4_1 = [0,0,0,N*INPUT[3]/(INPUT[3]+INPUT[7]),0,0,0,N*INPUT[7]/(INPUT[3]+INPUT[7])]
    INPUT_T, obj_eq1 = model(INPUT,T,v_g4_1)
    N3 = N - INPUT_T[3] - INPUT_T[7]
    v_g4_2 = [0,0,N3*INPUT_T[2]/(INPUT_T[2]+INPUT_T[6]),INPUT_T[3],0,0,N3*INPUT_T[6]/(INPUT_T[2]+INPUT_T[6]),INPUT_T[7]]
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_g4_2)
    obj_g4 = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
    # allocate to group 3 (proportionally between vaccine and boosters)
    v_g3_1 = [0,0,N*INPUT[2]/(INPUT[2]+INPUT[6]),0,0,0,N*INPUT[6]/(INPUT[2]+INPUT[6]),0]
    INPUT_T, obj_eq1 = model(INPUT,T,v_g3_1)
    v_g3_2 = [0,0,N*INPUT_T[2]/(INPUT_T[2]+INPUT_T[6]),0,0,0,N*INPUT_T[6]/(INPUT_T[2]+INPUT_T[6]),0]
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_g3_2)
    obj_g3 = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
    # approximated optimal 
    for i in range(n_obj):
        v_approx1, approx_optimal1, groups_to_vaccinate1 = approx_function(N,beta,INPUT,T)
        INPUT_T, obj1 = model(INPUT,T,v_approx1[i,:])
        v_approx2, approx_optimal2, groups_to_vaccinate2 = approx_function(N,beta,INPUT_T,T)
        INPUT_2T, obj2 = model(INPUT_T,T,v_approx2[i,:])
        obj_vacc[i] = obj1[i] + obj2[i]
    
    # numerical optimal
    for i in range(n_obj): 
        INPUT_T, obj_num1 = model(INPUT,T,v_T1[k][i,:])
        INPUT_2T, obj_num2 = model(INPUT_T,T,v_T2[k][i,:])
        obj_num[i] = obj_num1[i] + obj_num2[i] 
    
    # calculations averted
    inf_averted_prop = (obj_vacc[0] - obj_prop[0])/obj_prop[0]
    deaths_averted_prop = (obj_vacc[1] - obj_prop[1])/obj_prop[1]
    LY_averted_prop = (obj_vacc[2] - obj_prop[2])/obj_prop[2]
    QALY_averted_prop = (obj_vacc[3] - obj_prop[3])/obj_prop[3]
    
    inf_averted_prop_initial = (obj_vacc[0] - obj_prop_initial[0])/obj_prop_initial[0]
    deaths_averted_prop_initial = (obj_vacc[1] - obj_prop_initial[1])/obj_prop_initial[1]
    LY_averted_prop_initial = (obj_vacc[2] - obj_prop_initial[2])/obj_prop_initial[2]
    QALY_averted_prop_initial = (obj_vacc[3] - obj_prop_initial[3])/obj_prop_initial[3]
    
    inf_averted_prop_booster = (obj_vacc[0] - obj_prop_booster[0])/obj_prop_booster[0]
    deaths_averted_prop_booster = (obj_vacc[1] - obj_prop_booster[1])/obj_prop_booster[1]
    LY_averted_prop_booster = (obj_vacc[2] - obj_prop_booster[2])/obj_prop_booster[2]
    QALY_averted_prop_booster = (obj_vacc[3] - obj_prop_booster[3])/obj_prop_booster[3]
    
    inf_averted_g3 = (obj_vacc[0] - obj_g3[0])/obj_g3[0]
    deaths_averted_g3 = (obj_vacc[1] - obj_g3[1])/obj_g3[1]
    LY_averted_g3 = (obj_vacc[2] - obj_g3[2])/obj_g3[2]
    QALY_averted_g3 = (obj_vacc[3] - obj_g3[3])/obj_g3[3]
    
    inf_averted_g4 = (obj_vacc[0] - obj_g4[0])/obj_g4[0]
    deaths_averted_g4 = (obj_vacc[1] - obj_g4[1])/obj_g4[1]
    LY_averted_g4 = (obj_vacc[2] - obj_g4[2])/obj_g4[2]
    QALY_averted_g4 = (obj_vacc[3] - obj_g4[3])/obj_g4[3]
    
    inf_averted_num = (obj_vacc[0] - obj_num[0])/obj_num[0]
    deaths_averted_num = (obj_vacc[1] - obj_num[1])/obj_num[1]
    LY_averted_num = (obj_vacc[2] - obj_num[2])/obj_num[2]
    QALY_averted_num = (obj_vacc[3] - obj_num[3])/obj_num[3]
    
    df.loc[len(df.index)] = [T, N, "Proportional vaccines", inf_averted_prop, deaths_averted_prop, LY_averted_prop, QALY_averted_prop] 
    df.loc[len(df.index)] = [T, N, "Proportional initial doses", inf_averted_prop_initial, deaths_averted_prop_initial, LY_averted_prop_initial, QALY_averted_prop_initial] 
    df.loc[len(df.index)] = [T, N, "Proportional booster doses", inf_averted_prop_booster, deaths_averted_prop_booster, LY_averted_prop_booster, QALY_averted_prop_booster] 
    df.loc[len(df.index)] = [T, N, "Highest initial force of infection", inf_averted_g3, deaths_averted_g3,LY_averted_g3, QALY_averted_g3] 
    df.loc[len(df.index)] = [T, N, "Highest mortality rate", inf_averted_g4, deaths_averted_g4, LY_averted_g4, QALY_averted_g4] 
    df.loc[len(df.index)] = [T, N, "Numerical optimal", inf_averted_num, deaths_averted_num, LY_averted_num, QALY_averted_num] 


pd.set_option('display.float_format', lambda x: f'{x:.3f}')
display(df)
df.to_csv('Output/SIS/Table outcomes.csv', index=False)



