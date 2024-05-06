import scipy.integrate as spi
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pyDOE import *
from IPython.display import display



plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':14})

pd.options.display.float_format = "{:3e}".format



# Parameters

m = 4 # 2 boosters (unvacc, initial, booster 1, booster 2)
n = 2 # 2 population groups
n_obj = 4
# i,k (group and vaccine)

# Proportion of the population in each group
f = np.zeros(n)
f[0] = 0.65
f[1] = 0.35

# Average durations
av_duration = np.array([[7, 7, 6, 6], [7, 7, 6, 6]])

# Infection fatality rate by age group 
IFR = np.zeros((n,m))
# no vaccine
IFR[0,0] = 3/100000
IFR[1,0] = 40/100000
# initial dose
IFR[0,1] = IFR[0,0]/3
IFR[1,1] = IFR[1,0]/8
# booster 1 
IFR[0,2] = IFR[0,0]/5
IFR[1,2] = IFR[1,0]/12
# booster 2
IFR[0,3] = IFR[0,0]/8
IFR[1,3] = IFR[1,0]/16

# omega = 1/average incubation period
omega = np.zeros(n)
omega[0] = 1/14
omega[1] = 1/14

mu = np.zeros((n,m))
gamma = np.zeros((n,m))

for i in range(n): 
    for k in range(m): 
        mu[i,k] = 1/av_duration[i,k]*IFR[i,k]
        gamma[i,k] = 1/av_duration[i,k] - mu[i,k]    

eta = np.zeros((n,m))

eta[0,1] = 0.6
eta[1,1] = 0.6

eta[0,2] = 0.8
eta[1,2] = 0.8

eta[0,3] = 0.9
eta[1,3] = 0.9

# Initialization
N0 = 1
D0 = 0
I0 = 0.01

I1u = I0*f[0]/m
I2u = I0*f[1]/m

I1v = I0*f[0]/m
I2v = I0*f[1]/m

I1b = I0*f[0]/m
I2b = I0*f[1]/m

I1b2 = I0*f[0]/m
I2b2 = I0*f[1]/m

E1u = I1u 
E2u = I2u

E1v = I1v 
E2v = I2v

E1b = I1b
E2b = I2b

E1b2 = I1b2
E2b2 = I2b2

S0 = N0-I1u-I2u-I1v-I2v-I1b-I2b-I1b2-I2b2-E1u-E2u-E1v-E2v-E1b-E2b-E1b2-E2b2-D0
S1u = S0*f[0]/m
S2u = S0*f[1]/m

S1v = S0*f[0]/m
S2v = S0*f[1]/m

S1b = S0*f[0]/m
S2b = S0*f[1]/m

S1b2 = S0*f[0]/m
S2b2 = S0*f[1]/m

INPUT = (S1u, S1v, S1b, S1b2, S2u, S2v, S2b, S2b2, E1u, E1v, E1b, E1b2, E2u, E2v, E2b, E2b2, I1u, I1v, I1b, I1b2, I2u, I2v, I2b, I2b2, D0*f[0], D0*f[1])

beta = np.zeros((n,n))
beta[0,0] = 4
beta[1,1] = 3
beta[0,1] = 1
beta[1,0] = 1
            
# LYs
L1 = 65
L2 = 45

# QALYs
q1L1 = 60
q2L2 = 35

L = (L1,L2)
qL = (q1L1,q2L2)



# Define model parameters 
time_horizons = [7, 15]

title_obj = ['Minimizing cumulative infections', 'Minimizing deaths', 'Minimizing life years lost', 'Minimizing QALYs lost']



# ODEs
def diff_eqs(INP, t):  
    '''The main set of equations'''
    Y=np.zeros((26)) 
    V = INP
    
    for i in range(n): 
        # Force of infection by group
        lambda_i = 0
        for j in range(n): 
            for l in range(m): 
                lambda_i += beta[i,j]*(1-eta[j,l])*V[2*n*m+j*m+l]
        
        for k in range(m): 
            # Susceptibles
            if k == 0: # no vaccines
                Y[i*m+k] = -(1-eta[i,k])*V[i*m+k]*lambda_i
            elif k == (m-1): # booster 2
                Y[i*m+k] = -(1-eta[i,k])*V[i*m+k]*lambda_i + gamma[i,k-1]*V[2*n*m+i*m+k-1] + gamma[i,k]*V[2*n*m+i*m+k]
            else: 
                Y[i*m+k] = -(1-eta[i,k])*V[i*m+k]*lambda_i + gamma[i,k-1]*V[2*n*m+i*m+k-1]
            
            # Exposed
            Y[n*m+i*m+k] = (1-eta[i,k])*V[i*m+k]*lambda_i - omega[i]*V[n*m+i*m+k]
        
            # Infected
            Y[2*n*m+i*m+k] = omega[i]*V[n*m+i*m+k] - (gamma[i,k]+mu[i,k])*V[2*n*m+i*m+k]
        
            # Dead
            Y[3*n*m+i] += mu[i,k]*V[2*n*m+i*m+k] 
    
    return Y   # For odeint

# Model
def model(INPUT,ND,v):
    t_start = 0.0; t_end = ND; t_inc = 1 
    t_range = np.arange(t_start, t_end+t_inc, t_inc)
    
    INPUT_v = [INPUT[i] for i in range(len(INPUT))]
    for i in range(n):
        for k in range(m-1): 
            INPUT_v[i*m+k+1] += v[i*(m-1)+k]
            INPUT_v[i*m+k] -= v[i*(m-1)+k]
    
    RES = spi.odeint(diff_eqs,INPUT_v,t_range)
    
    all_infections = 0
    S = np.zeros(ND+1)
    E = np.zeros(ND+1)
    I = np.zeros(ND+1)
    D = np.zeros(ND+1)
    for i in range(n*m): 
        S += RES[:,i]
        E += RES[:,i+n*m]
        I += RES[:,i+2*n*m]
        all_infections += np.sum(RES[:,i+2*n*m])
    for i in range(n): 
        D += RES[:,i+3*n*m]
    
    end_pop = RES[ND,:]
    end_deaths = D[ND] #total deaths
    end_lifeyears = L1*RES[ND,3*n*m]+L2*RES[ND,3*n*m+1] 
    end_qalys = q1L1*RES[ND,3*n*m]+q2L2*RES[ND,3*n*m+1] 
    
    return(end_pop, [all_infections,end_deaths,end_lifeyears,end_qalys])


# Threshold
def thresholds(parameters,INPUT,sigma):
    S_pop = 0
    susceptible = INPUT[0:8]
    exposed = INPUT[8:16]
    infections = INPUT[16:24]
    lambda_ = np.zeros(len(parameters)) #force of infection
    alpha = np.zeros(n*(m-1))
    for i in range(n): 
        for j in range(n): 
            for l in range(m):
                lambda_[i] += parameters[i,j]*(1-eta[j,l])*infections[j*m+l]
    for i in range(n): 
        for k in range(m-1):
            alpha_temp = susceptible[i*m+k]+ 2*(omega[i]*exposed[i*m+k] - (gamma[i,k]+mu[i,k])*infections[i*m+k])                                          /(omega[i]*(1-eta[i,k])*lambda_[i]*sigma)                                + 2*infections[i*m+k]/(omega[i]*(1-eta[i,k])*lambda_[i]*sigma**2)                                - omega[i]*exposed[i*m+k]/((1-eta[i,k])*lambda_[i])                                - (gamma[i,k]+mu[i,k])*(omega[i]*exposed[i*m+k] - (gamma[i,k]+mu[i,k])*infections[i*m+k])                                         /(omega[i]*(1-eta[i,k])*lambda_[i])
            alpha[i*(m-1)+k] = min(susceptible[i*m+k],alpha_temp)
            if alpha_temp < 0: 
                print("alpha < 0") 
                break
            S_pop += susceptible[i*m+k]
    threshold = (np.sum(alpha))/S_pop
    return(threshold, alpha)

for T in time_horizons: 
    print("T =", T, "days")
    threshold, alpha = thresholds(beta,INPUT,T)
    print('Proportion of the unvaccinated and vaccinated population that can be vaccinated:', threshold)


# Optimal decisions given by approximations   
def conditions(parameters,infections):
    p_infections = np.zeros((m-1)*len(parameters)) #[v,b]
    p_deaths = np.zeros((m-1)*len(parameters))
    p_ly = np.zeros((m-1)*len(parameters))
    p_qaly = np.zeros((m-1)*len(parameters))
    lambda_ = np.zeros(len(parameters)) #force of infection
    
    for i in range(n): 
        for j in range(n): 
            for l in range(m):
                lambda_[i] += parameters[i,j]*(1-eta[j,l])*infections[j*m+l]
        
        for k in range(m-1): 
            p_infections[i*(m-1)+k] = omega[i]*(eta[i,k+1]-eta[i,k])*lambda_[i]
            p_deaths[i*(m-1)+k] = omega[i]*(mu[i,k]*(1-eta[i,k])-mu[i,k+1]*(1-eta[i,k+1]))*lambda_[i]
            p_ly[i*(m-1)+k] = L[i]*omega[i]*(mu[i,k]*(1-eta[i,k])-mu[i,k+1]*(1-eta[i,k+1]))*lambda_[i]
            p_qaly[i*(m-1)+k] = qL[i]*omega[i]*(mu[i,k]*(1-eta[i,k])-mu[i,k+1]*(1-eta[i,k+1]))*lambda_[i]
            
    return(p_infections, p_deaths, p_ly, p_qaly) 

def insight(parameters,INPUT):     
    infections = INPUT[16:24]
    p = conditions(parameters,infections)
    for i in range(n_obj): 
        p_2d = p[i].reshape(n,(m-1))
    groups_to_vaccinate = np.zeros((n_obj,(m-1)*n))
    for i in range(n_obj):
        groups_to_vaccinate[i,:] = np.argsort(-p[i]) 
    return(groups_to_vaccinate)

groups_to_vaccinate = insight(beta, INPUT) 

print('Optimal vaccination order for first time period (any time horizon)')
for l in range(n_obj):
    temp = []
    for j in range((m-1)*n): 
        k = groups_to_vaccinate[l,j]%(m-1)
        i = groups_to_vaccinate[l,j]//(m-1)+1
        temp.append('v_{},{}'.format(int(i),int(k)+1))
    print(title_obj[l])
    print(temp)


# # First time period


def approx_function(N,parameters,INPUT,T): 
    groups_to_vaccinate = insight(parameters,INPUT)
    threshold, alpha = thresholds(parameters,INPUT,T)
    v_approx = np.zeros((n_obj,(m-1)*n))
    approx_optimal = np.zeros(n_obj)

    for i in range(n_obj): 
        N_temp = N
        v_temp = np.zeros((m-1)*n) 
        for j in range((m-1)*n): 
            opt_group = int(groups_to_vaccinate[i,j])
            v_temp[opt_group] = min(alpha[opt_group], N_temp)
            N_temp = N_temp - min(alpha[opt_group], N_temp)
            
        v_approx[i,:] = v_temp            
        end_pop, objectives = model(INPUT,T,v_temp)
        approx_optimal[i] = objectives[i]
        
    return(v_approx, approx_optimal, groups_to_vaccinate)


def delta(N,parameters,INPUT,T): 
    S1u, S1v, S1b, S1b2, S2u, S2v, S2b, S2b2, E1u, E1v, E1b, E1b2, E2u, E2v, E2b, E2b2, I1u, I1v, I1b, I1b2, I2u, I2v, I2b, I2b2, D10, D20 = INPUT
    
    v1_range = np.arange(0, S1u, 0.01)
    v2_range = np.arange(0, S2u, 0.01)
    
    b1_range = np.arange(0, S1v, 0.01)
    b2_range = np.arange(0, S2v, 0.01)

    b2_1_range = np.arange(0, S1b, 0.01)
    b2_2_range = np.arange(0, S2b, 0.01)
    
    difference = np.zeros(n_obj)
    
    v_approx, approx_optimal, groups_to_vaccinate = approx_function(N,beta,INPUT,T)
    v_opt, numerical_optimal, temp = approx_function(N,beta,INPUT,T)
    
    for v1 in v1_range: 
        for v2 in v2_range: 
            for b1 in b1_range: 
                for b2 in b2_range: 
                    for b2_1 in b2_1_range: 
                        b2_2 = N-v1-v2-b1-b2-b2_1 #allocate all vaccines
                        if b2_2 >= 0 and b2_2 <= S2b: 
                            v = (v1,b1,b2_1,v2,b2,b2_2) 
                            end_pop, objectives = model(INPUT,T,v)
                            for i in range(n_obj): 
                                if objectives[i] < numerical_optimal[i]: 
                                    numerical_optimal[i] = objectives[i]
                                    v_opt[i,:] = v
                                            
    for i in range(n_obj):
        difference[i] = (approx_optimal[i] - numerical_optimal[i])/numerical_optimal[i]*100
    
    return(numerical_optimal, difference, v_opt, v_approx, groups_to_vaccinate)



l = [r'$v_{1,1}$', r'$v_{1,2}$', r'$v_{1,3}$', r'$v_{2,1}$', r'$v_{2,2}$', r'$v_{2,3}$']
N_range = np.arange(0, 0.101, 0.01)
v_T1 = []

print('First time period')
t = 0
for T in time_horizons: 
    print('T =', T, 'days')
    difference = np.zeros((len(N_range),n_obj))
    v_approx = np.zeros((len(N_range),n_obj,(m-1)*n))
    v_opt = np.zeros((len(N_range),n_obj,(m-1)*n))
    numerical_optimal = np.zeros((len(N_range),n_obj))
    
    k=0
    for N in N_range: 
        numerical_optimal[k,:], difference[k,:], v_opt[k,:,:], v_approx[k,:,:], group_to_vaccinate = delta(N,beta,INPUT,T)
        k+=1
    
    plt.figure(figsize=(5,4))
    for i in range(difference.shape[1]): #looping through objectives
        plt.plot(N_range, difference[:,i], 'o', label = title_obj[i], alpha=0.7)
        plt.xlabel(r'$\frac{N}{P}$')
        plt.title("Percentage difference".format(T))
    plt.legend(loc = 'upper right', bbox_to_anchor=(2, 0.75))
    plt.ylim((-0.3,3))
    plt.savefig("Output/SEIS/difference_T1={}.png".format(T), bbox_inches='tight')
    plt.show()
    
    fig, axs = plt.subplots(4, 2, figsize=(14,20))
    
    for i in range(v_opt.shape[1]): #looping through objectives
        for j in range((m-1)*n): 
            axs[i,0].plot(N_range, v_opt[:,i,j], 'o', label = l[j], alpha=0.7)
        axs[i,0].set_xlabel(r'$\frac{N}{P}$')
        axs[i,0].set_ylabel('Optimal vaccine allocation')
        axs[i,0].title.set_text(title_obj[i] + ' (exhaustive search)')
        axs[i,0].legend(bbox_to_anchor=(1.05, 0.75))

        for j in range((m-1)*n): 
            axs[i,1].plot(N_range, v_approx[:,i,j], 'o', label = l[j], alpha=0.7)
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
    v_opt = np.zeros((n_obj,(m-1)*n))
    v_approx = np.zeros((n_obj,(m-1)*n))
    numerical_optimal = np.zeros(n_obj)
    groups_to_vaccinate = np.zeros((n_obj,(m-1)*n))
    
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
    v_approx2 = np.zeros((len(N_range),n_obj,(m-1)*n))  
    v_opt2 = np.zeros((len(N_range),n_obj,(m-1)*n))
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
    plt.ylim((-0.3,3))
    plt.savefig("Output/SEIS/difference_T2={}.png".format(T), bbox_inches='tight')
    plt.show()
    
    fig, axs = plt.subplots(4, 2, figsize=(14,20))
    
    for i in range(v_opt.shape[1]):
        for j in range((m-1)*n): 
            axs[i,0].plot(N_range, v_opt2[:,i,j], 'o', label = l[j], alpha=0.7)
        axs[i,0].set_xlabel(r'$\frac{N}{P}$')
        axs[i,0].set_ylabel('Optimal vaccine allocation')
        axs[i,0].title.set_text(title_obj[i] + ' (exhaustive search)')
        axs[i,0].legend(bbox_to_anchor=(1.05, 0.75))

        for j in range((m-1)*n): 
            axs[i,1].plot(N_range, v_approx2[:,i,j], 'o', label = l[j], alpha=0.7)
        axs[i,1].set_xlabel(r'$\frac{N}{P}$')
        axs[i,1].set_ylabel('Optimal vaccine allocation')
        axs[i,1].title.set_text(title_obj[i] + ' (approximation)')
        axs[i,1].legend(bbox_to_anchor=(1.05, 0.75))

    fig.tight_layout(pad = 3)
    plt.show()
    
    v_T2.append(v_opt2[10])
    


for i_obj in range(n_obj):
    print(title_obj[i_obj])
    for T in time_horizons: 
        current = np.zeros(6)
        print('T =', T, 'days')
        k=0
        for N in N_range: 
            temp1, temp2, temp3, temp4, group_to_vaccinate = delta_2(N,beta,INPUT,T)
            k+=1
            temp = []
            for j in range((m-1)*n): 
                k = group_to_vaccinate[i_obj,j]%(m-1)
                i = group_to_vaccinate[i_obj,j]//(m-1)+1
                temp.append('v_{},{}'.format(int(i),int(k)+1))
            if not np.array_equal(group_to_vaccinate[i_obj,:], current): 
                print('Switch point N =', N, ', approximated optimal order is: \n', temp)
                current = group_to_vaccinate[i_obj,:] 
                


fig, axs = plt.subplots(4, 2, figsize=(14,20))
for i in range(v_opt.shape[1]): #looping through objectives
    for j in range((m-1)*n): 
        axs[i,0].plot(N_range, v_opt2[:,i,j], 'o', label = l[j], alpha=0.7)
    axs[i,0].set_xlabel(r'$\frac{N}{P}$')
    axs[i,0].set_ylabel('Optimal vaccine allocation')
    axs[i,0].title.set_text(title_obj[i] + ' (exhaustive search)')
    axs[i,0].legend(bbox_to_anchor=(1.05, 0.75))

    for j in range((m-1)*n): 
        axs[i,1].plot(N_range, v_approx2[:,i,j], 'o', label = l[j], alpha=0.7)
    axs[i,1].set_xlabel(r'$\frac{N}{P}$')
    axs[i,1].set_ylabel('Optimal vaccine allocation')
    axs[i,1].title.set_text(title_obj[i] + ' (approximation)')
    axs[i,1].legend(bbox_to_anchor=(1.05, 0.75))

fig.tight_layout(pad = 3)
plt.savefig("Output/SEIS/v2_all_T2={}.png".format(T), bbox_inches='tight')
plt.show()


# # Outcomes averted


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
    v_prop1 = [INPUT[i]/np.sum(INPUT[0:6])*N for i in range(6)]
    INPUT_T, obj_eq1 = model(INPUT,T,v_prop1)
    v_prop2 = [INPUT_T[i]/np.sum(INPUT_T[0:6])*N for i in range(6)]
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_prop2)
    obj_prop = [sum(x) for x in zip(obj_eq1, obj_eq2)]

    # proportional initial vaccines
    INPUT_T, obj_eq1 = model(INPUT,T,(INPUT[0]/(INPUT[0]+INPUT[4])*N,0,0,INPUT[4]/(INPUT[0]+INPUT[4])*N,0,0))
    INPUT_2T, obj_eq2 = model(INPUT_T,T,(INPUT_T[0]/(INPUT_T[0]+INPUT_T[4])*N,0,0,INPUT_T[4]/(INPUT_T[0]+INPUT_T[4])*N,0,0))
    obj_prop_initial = [sum(x) for x in zip(obj_eq1, obj_eq2)]

    # proportional booster vaccines
    INPUT_T, obj_eq1 = model(INPUT,T,(0,INPUT[1]/(INPUT[1]+INPUT[5])*N,0,0,INPUT[5]/(INPUT[1]+INPUT[5])*N,0))
    INPUT_2T, obj_eq2 = model(INPUT_T,T,(0,INPUT_T[1]/(INPUT_T[1]+INPUT_T[5])*N,0,0,INPUT_T[5]/(INPUT_T[1]+INPUT_T[5])*N,0))
    obj_prop_booster = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
    # proportional booster 2 vaccines
    INPUT_T, obj_eq1 = model(INPUT,T,(0,0,INPUT[2]/(INPUT[2]+INPUT[6])*N,0,0,INPUT[6]/(INPUT[2]+INPUT[6])*N))
    INPUT_2T, obj_eq2 = model(INPUT_T,T,(0,0,INPUT_T[2]/(INPUT_T[2]+INPUT_T[6])*N,0,0,INPUT_T[6]/(INPUT_T[2]+INPUT_T[6])*N))
    obj_prop_booster_2 = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
    # allocate to group 2 (proportionally between vaccine shots) - two time periods
    v_g2_1 = np.zeros(6)
    for i in range(3): 
        v_g2_1[i] = INPUT[i+3]/np.sum(INPUT[3:6])*N
    INPUT_T, obj_eq1 = model(INPUT,T,v_g2_1)
    v_g2_2 = np.zeros(6)
    for i in range(3): 
        v_g2_2[i] = INPUT_T[i+3]/np.sum(INPUT_T[3:6])*N
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_g2_2)
    obj_g2 = [sum(x) for x in zip(obj_eq1, obj_eq2)]

    # allocate to group 1 (proportionally between vaccine shots)
    v_g1_1 = np.zeros(6)
    for i in range(3): 
        v_g1_1[i] = INPUT[i]/np.sum(INPUT[0:3])*N
    INPUT_T, obj_eq1 = model(INPUT,T,v_g1_1)
    v_g1_2 = np.zeros(6)
    for i in range(3): 
        v_g1_2[i] = INPUT_T[i]/np.sum(INPUT_T[0:3])*N
    INPUT_2T, obj_eq2 = model(INPUT_T,T,v_g1_2)
    obj_g1 = [sum(x) for x in zip(obj_eq1, obj_eq2)]
    
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
    
    inf_averted_prop_booster_2 = (obj_vacc[0] - obj_prop_booster_2[0])/obj_prop_booster_2[0]
    deaths_averted_prop_booster_2 = (obj_vacc[1] - obj_prop_booster_2[1])/obj_prop_booster_2[1]
    LY_averted_prop_booster_2 = (obj_vacc[2] - obj_prop_booster_2[2])/obj_prop_booster_2[2]
    QALY_averted_prop_booster_2 = (obj_vacc[3] - obj_prop_booster_2[3])/obj_prop_booster_2[3]
    
    inf_averted_num = (obj_vacc[0] - obj_num[0])/obj_num[0]
    deaths_averted_num = (obj_vacc[1] - obj_num[1])/obj_num[1]
    LY_averted_num = (obj_vacc[2] - obj_num[2])/obj_num[2]
    QALY_averted_num = (obj_vacc[3] - obj_num[3])/obj_num[3]

    inf_averted_g1 = (obj_vacc[0] - obj_g1[0])/obj_g1[0]
    deaths_averted_g1 = (obj_vacc[1] - obj_g1[1])/obj_g1[1]
    LY_averted_g1 = (obj_vacc[2] - obj_g1[2])/obj_g1[2]
    QALY_averted_g1 = (obj_vacc[3] - obj_g1[3])/obj_g1[3]
    
    inf_averted_g2 = (obj_vacc[0] - obj_g2[0])/obj_g2[0]
    deaths_averted_g2 = (obj_vacc[1] - obj_g2[1])/obj_g2[1]
    LY_averted_g2 = (obj_vacc[2] - obj_g2[2])/obj_g2[2]
    QALY_averted_g2 = (obj_vacc[3] - obj_g2[3])/obj_g2[3]
    
    df.loc[len(df.index)] = [T, N, "Proportional vaccines", inf_averted_prop, deaths_averted_prop, LY_averted_prop, QALY_averted_prop] 
    df.loc[len(df.index)] = [T, N, "Proportional first doses", inf_averted_prop_initial, deaths_averted_prop_initial, LY_averted_prop_initial, QALY_averted_prop_initial] 
    df.loc[len(df.index)] = [T, N, "Proportional second doses", inf_averted_prop_booster, deaths_averted_prop_booster, LY_averted_prop_booster, QALY_averted_prop_booster]
    df.loc[len(df.index)] = [T, N, "Proportional third doses", inf_averted_prop_booster_2, deaths_averted_prop_booster_2, LY_averted_prop_booster_2, QALY_averted_prop_booster_2]
    df.loc[len(df.index)] = [T, N, "Highest initial force of infection", inf_averted_g1, deaths_averted_g1,LY_averted_g1, QALY_averted_g1] 
    df.loc[len(df.index)] = [T, N, "Highest mortality rate", inf_averted_g2, deaths_averted_g2, LY_averted_g2, QALY_averted_g2] 
    df.loc[len(df.index)] = [T, N, "Numerical optimal", inf_averted_num, deaths_averted_num, LY_averted_num, QALY_averted_num] 
    
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
display(df)
df.to_csv('Output/SEIS/Table outcomes.csv', index=False)





