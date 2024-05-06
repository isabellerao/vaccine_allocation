# # Calibration

import scipy.integrate as spi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pyDOE import *
import pickle
from scipy.linalg import eigh



plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':14})


# ## Defining uncalibrated parameters (from literature review)


# Parameters for COVID19 - from literature review

# Proportion of the population in each group
f = np.zeros(4)
f[0] = 0.2487 
f[1] = 0.2720 
f[2] = 0.3145 
f[3] = 1-f[0]-f[1]-f[2]

# Average durations
av_duration_mild = 11
av_duration_severe = 8

# Proportion of severe cases
alpha_u = np.zeros(4)
alpha_v = np.zeros(4)
alpha_b = np.zeros(4)

alpha_u[0] = 0.015873016
alpha_u[1] = 0.147863248
alpha_u[2] = 0.262824049
alpha_u[3] = 0.459612278

alpha_v[0] = 0.003993212
alpha_v[1] = 0.037198301
alpha_v[2] = 0.047228041
alpha_v[3] = 0.072266081

alpha_b[0] = 0.000920175
alpha_b[1] = 0.008571782
alpha_b[2] = 0.010882983
alpha_b[3] = 0.016652619

# Infection fatality rate by age group 
IFR_u = np.zeros(4)
IFR_v = np.zeros(4)
IFR_b = np.zeros(4)

IFR_u[0] = 0.0000306267
IFR_u[1] = 0.0001782578
IFR_u[2] = 0.0013621071
IFR_u[3] = 0.0146025583

IFR_v[0] = IFR_u[0]/9.050726505
IFR_v[1] = IFR_u[1]/12.13402384
IFR_v[2] = IFR_u[2]/6.191418464
IFR_v[3] = IFR_u[3]/2.526648598

IFR_b[0] = IFR_u[0]/17.80123112
IFR_b[1] = IFR_u[1]/23.86554965
IFR_b[2] = IFR_u[2]/10.42431911
IFR_b[3] = IFR_u[3]/6.926976422

av_duration_u = np.zeros(4)
av_duration_v = np.zeros(4)
av_duration_b = np.zeros(4)

mu_u = np.zeros(4)
mu_v = np.zeros(4)
mu_b = np.zeros(4)

gamma_u = np.zeros(4)
gamma_v = np.zeros(4)
gamma_b = np.zeros(4)

for i in range(4): 
    av_duration_u[i] = av_duration_mild + alpha_u[i]*av_duration_severe
    av_duration_v[i] = av_duration_mild + alpha_v[i]*av_duration_severe 
    av_duration_b[i] = av_duration_mild + alpha_b[i]*av_duration_severe 
    
    mu_u[i] = 1/av_duration_u[i]*IFR_u[i]
    mu_v[i] = 1/av_duration_v[i]*IFR_v[i]
    mu_b[i] = 1/av_duration_b[i]*IFR_b[i]
    
    gamma_u[i] = 1/av_duration_u[i] - mu_u[i]
    gamma_v[i] = 1/av_duration_v[i] - mu_v[i]
    gamma_b[i] = 1/av_duration_b[i] - mu_b[i]
    
# Vaccine effectiveness
eta_v = np.zeros(4)
eta_b = np.zeros(4)

eta_v[0] = 1-1/3.859753866
eta_v[1] = 1-1/2.601461743
eta_v[2] = 1-1/3.530527975
eta_v[3] = 1-1/7.026663325

eta_b[0] = 1-1/5.651553322
eta_b[1] = 1-1/3.809128837
eta_b[2] = 1-1/5.832992745
eta_b[3] = 1-1/14.9194081

# Initialization
pop_size = 20201249
N0 = 1
D0 = 0.0028


# ## Model (PDEs)


# ODEs
def diff_eqs(INP, t):  
    '''The main set of equations'''
    Y=np.zeros((25)) 
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
        Y[24] += mu_u[i]*V[i+12] + mu_v[i]*V[i+16] + mu_b[i]*V[i+20]
    
    return Y   # For odeint

# Model
def calibration(ND): 
    t_start = 0.0; t_end = ND; t_inc = 1
    t_range = np.arange(t_start, t_end+t_inc, t_inc)
    RES = spi.odeint(diff_eqs,INPUT,t_range)
    
    new_deaths = np.zeros(t_end) 
    new_cases = np.zeros(t_end)
    for i in range(t_end): 
        new_deaths[i] = RES[i+1,24] - RES[i,24]

        for j in range(12): 
            new_cases[i] += RES[i+1,j+12] - RES[i,j+12]
            
    S = np.zeros(ND+1)
    I = np.zeros(ND+1)
    D = np.zeros(ND+1)
    for i in range(12): 
        S += RES[:,i]
        I += RES[:,i+12]
    D = RES[:,24]
    
        
    return([new_deaths*pop_size,new_cases*pop_size])


# ## Importing targets


# Import raw data
df = pd.read_excel('Data/Covid cases and deaths - NY.xlsx', sheet_name = "python")

# Calculate rolling average
window_size = 7
df['rolling_cases'] = df.iloc[:,5].rolling(window=window_size).mean()
df['rolling_deaths'] = df.iloc[:,6].rolling(window=window_size).mean()

for i in range(window_size): 
    df.iloc[i,7] = df.iloc[i,5]
    df.iloc[i,8] = df.iloc[i,6]

df['date'] = pd.to_datetime(df['date'])

start_date = '2021-12-15' 
end_date = '2022-01-25'
raw_data = df[(df['date'] >= start_date) & (df['date'] < end_date)]

num_days_deaths = raw_data.shape[0]
num_days_cases = raw_data.shape[0]
num_days = [num_days_deaths]

targets = raw_data[['rolling_deaths']].values[0:num_days_deaths]



# Plot targets

target_range = raw_data[['date']].values[0:num_days_deaths]

fig, ax = plt.subplots()
ax.plot(target_range, raw_data.rolling_deaths, label = 'Rolling')
ax.plot(target_range, raw_data.new_deaths, label = 'Raw')
date_form = mdates.DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Date')
ax.set_ylabel('New Deaths')
ax.legend(loc=0)
plt.tight_layout()
plt.savefig("Output/Calibration/rolling_deaths.png")
plt.show()

fig, ax = plt.subplots()
ax.plot(target_range, raw_data.rolling_cases, label = 'Rolling')
ax.plot(target_range, raw_data.new_cases, label = 'Raw')
date_form = mdates.DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Date')
ax.set_ylabel('New Cases')
ax.legend(loc=0)
plt.tight_layout()
plt.savefig("Output/Calibration/rolling_cases.png")
plt.show()


# ## LHS Calibration


# Parameters we want to vary: alpha, beta and inital population of infected
names_param = ['beta11', 'beta12', 'beta13', 'beta14', 'beta21', 'beta22', 'beta23', 'beta24',
               'beta31', 'beta32', 'beta33', 'beta34', 'beta41', 'beta42', 'beta43', 'beta44', 'I0']   
num_param = np.shape(names_param)[0]
num_samples = 50000

# Upper and lower bounds for the parameters
lb = [0.5,0.025,0.025,0.025,0.025,0.5,0.025,0.025,0.025,0.025,0.5,0.025,0.025,0.025,0.025,0.5,0.005]
ub = [1.5,0.5,0.5,0.5,0.5,1.5,0.5,0.5,0.5,0.5,1.5,0.5,0.5,0.5,0.5,1.5,0.015]



# Latin Hypercube Sampling (LHS) for exhaustive search
np.random.seed(1)

lhs_unit = lhs(num_param, samples=num_samples)
lhs_scaled = np.zeros((num_samples, num_param)) 

for i in range(num_param): 
    lhs_scaled[:,i] = lb[i] + (ub[i]-lb[i])*lhs_unit[:,i]



# Goodness of fit (GoF)
def GoF_wsse(model_output, targets, num_days):
    gof = 0
    for i in range(np.shape(targets)[1]): 
        gof += np.sum((model_output[i][0:num_days[i]] - targets[0:num_days[i],i])**2)
    return(gof)



# Calculate the goodness of fit for each parameter set
GoF = np.zeros(num_samples)
R_0 = np.zeros(num_samples)
lambda1 = np.zeros(num_samples)
lambda2 = np.zeros(num_samples)
lambda3 = np.zeros(num_samples)
lambda4 = np.zeros(num_samples)

for i in range(num_samples): 
    beta = np.zeros((4,4))
    
    beta[0,0] = lhs_scaled[i,0]
    beta[0,1] = lhs_scaled[i,1]
    beta[0,2] = lhs_scaled[i,2]
    beta[0,3] = lhs_scaled[i,3]
    
    beta[1,0] = lhs_scaled[i,4]
    beta[1,1] = lhs_scaled[i,5]
    beta[1,2] = lhs_scaled[i,6]
    beta[1,3] = lhs_scaled[i,7]
    
    beta[2,0] = lhs_scaled[i,8]
    beta[2,1] = lhs_scaled[i,9]
    beta[2,2] = lhs_scaled[i,10]
    beta[2,3] = lhs_scaled[i,11]
    
    beta[3,0] = lhs_scaled[i,12]
    beta[3,1] = lhs_scaled[i,13]
    beta[3,2] = lhs_scaled[i,14]
    beta[3,3] = lhs_scaled[i,15]
    
    I0 = lhs_scaled[i,16]
    
    I1u = I0*f[0]/3
    I2u = I0*f[1]/3
    I3u = I0*f[2]/3
    I4u = I0*f[3]/3
    
    I1v = I0*f[0]/3
    I2v = I0*f[1]/3
    I3v = I0*f[2]/3
    I4v = I0*f[3]/3
    
    I1b = I0*f[0]/3
    I2b = I0*f[1]/3
    I3b = I0*f[2]/3
    I4b = I0*f[3]/3
    
    S0 = N0-I1u-I2u-I3u-I4u-I1v-I2v-I3v-I4v-I1b-I2b-I3b-I4b-D0
    S1u = S0*f[0]*0.815735223
    S2u = S0*f[1]*0.338994188
    S3u = S0*f[2]*0.260110304
    S4u = S0*f[3]*0.108548946
    S1v = S0*f[0]*0.184264777
    S2v = S0*f[1]*0.582922377
    S3v = S0*f[2]*0.611080913
    S4v = S0*f[3]*0.531736991
    S1b = 0
    S2b = S0*f[1]*0.078083435
    S3b = S0*f[2]*0.128808784
    S4b = S0*f[3]*0.359714063
    
    INPUT = (S1u, S2u, S3u, S4u, S1v, S2v, S3v, S4v, S1b, S2b, S3b, S4b, I1u, I2u, I3u, I4u, I1v, I2v, I3v, I4v, I1b, I2b, I3b, I4b, D0)
    
    GoF[i] = GoF_wsse(calibration(ND=max(num_days)),targets,num_days)
    
    S_u = np.array([S1u, S2u, S3u, S4u])
    S_v = np.array([S1v, S2v, S3v, S4v])
    S_b = np.array([S1b, S2b, S3b, S4b])
    I_u = np.array([I1u, I2u, I3u, I4u])
    I_v = np.array([I1v, I2v, I3v, I4v])
    I_b = np.array([I1b, I2b, I3b, I4b])
    d_u = gamma_u + mu_u
    d_v = gamma_v + mu_v
    d_b = gamma_b + mu_b
    
    FV = np.zeros((12,12))
    for k in range(4): 
        for l in range(4):
            # first column
            FV[k,l]=beta[k,l]*S_u[k]/d_u[l]
            FV[k+4,l]=(1-eta_v[k])*beta[k,l]*S_v[k]/d_u[l]
            FV[k+8,l]=(1-eta_b[k])*beta[k,l]*S_b[k]/d_u[l]
            # second column
            FV[k,l+4]=(1-eta_v[l])*beta[k,l]*S_u[k]/d_v[l]
            FV[k+4,l+4]=(1-eta_v[l])*eta_v[k]*beta[k,l]*S_v[k]/d_v[l]
            FV[k+8,l+4]=(1-eta_v[l])*eta_b[k]*beta[k,l]*S_b[k]/d_v[l]
            # third column
            FV[k,l+8]=(1-eta_b[l])*beta[k,l]*S_u[k]/d_b[l]
            FV[k+4,l+8]=(1-eta_b[l])*(1-eta_v[k])*beta[k,l]*S_v[k]/d_b[l]
            FV[k+8,l+8]=(1-eta_b[l])*(1-eta_b[k])*beta[k,l]*S_b[k]/d_b[l]
    w, v = eigh(FV)
    R_0[i] = np.max(w)
    
    for j in range(4): 
        lambda1[i] += beta[0,j]*(I_u[j] + (1-eta_v[j])*I_v[j] + (1-eta_b[j])*I_b[j]) 
        lambda2[i] += beta[1,j]*(I_u[j] + (1-eta_v[j])*I_v[j] + (1-eta_b[j])*I_b[j]) 
        lambda3[i] += beta[2,j]*(I_u[j] + (1-eta_v[j])*I_v[j] + (1-eta_b[j])*I_b[j]) 
        lambda4[i] += beta[3,j]*(I_u[j] + (1-eta_v[j])*I_v[j] + (1-eta_b[j])*I_b[j]) 



# Clean up results
results = pd.DataFrame(np.column_stack((lhs_scaled, GoF, R_0, lambda1, lambda2, lambda3, lambda4)))
results.columns = ['beta11', 'beta12', 'beta13', 'beta14', 'beta21', 'beta22', 'beta23', 'beta24',
                   'beta31', 'beta32', 'beta33', 'beta34', 'beta41', 'beta42', 'beta43', 'beta44','I0', 
                   'GoF', 'R0', 'lambda1', 'lambda2', 'lambda3', 'lambda4']
results['Admissible'] = np.where(((results['lambda2'] > results['lambda4']) & (results['lambda3'] > results['lambda4'])), 1, 0)
admissible_results = results[results.Admissible.eq(1)]
admissible_results.sort_values(by=['GoF']).head(10)



# Pick the set of parameters with the highest goodness of fit
results_array = admissible_results.to_numpy()
sorted_results = results_array[np.argsort(results_array[:, num_param])]

num = 0
beta[0,0] = sorted_results[num,0]
beta[0,1] = sorted_results[num,1]
beta[0,2] = sorted_results[num,2]
beta[0,3] = sorted_results[num,3]

beta[1,0] = sorted_results[num,4]
beta[1,1] = sorted_results[num,5]
beta[1,2] = sorted_results[num,6]
beta[1,3] = sorted_results[num,7]

beta[2,0] = sorted_results[num,8]
beta[2,1] = sorted_results[num,9]
beta[2,2] = sorted_results[num,10]
beta[2,3] = sorted_results[num,11]

beta[3,0] = sorted_results[num,12]
beta[3,1] = sorted_results[num,13]
beta[3,2] = sorted_results[num,14]
beta[3,3] = sorted_results[num,15]

I0 = sorted_results[num,16]

I1u = I0*f[0]/3
I2u = I0*f[1]/3
I3u = I0*f[2]/3
I4u = I0*f[3]/3

I1v = I0*f[0]/3
I2v = I0*f[1]/3
I3v = I0*f[2]/3
I4v = I0*f[3]/3

I1b = I0*f[0]/3
I2b = I0*f[1]/3
I3b = I0*f[2]/3
I4b = I0*f[3]/3


S0 = N0-I1u-I2u-I3u-I4u-I1v-I2v-I3v-I4v-I1b-I2b-I3b-I4b-D0
S1u = S0*f[0]*0.815735223
S2u = S0*f[1]*0.338994188
S3u = S0*f[2]*0.260110304
S4u = S0*f[3]*0.108548946
S1v = S0*f[0]*0.184264777
S2v = S0*f[1]*0.582922377
S3v = S0*f[2]*0.611080913
S4v = S0*f[3]*0.531736991
S1b = 0
S2b = S0*f[1]*0.078083435
S3b = S0*f[2]*0.128808784
S4b = S0*f[3]*0.359714063
    
INPUT = (S1u, S2u, S3u, S4u, S1v, S2v, S3v, S4v, S1b, S2b, S3b, S4b, I1u, I2u, I3u, I4u, I1v, I2v, I3v, I4v, I1b, I2b, I3b, I4b, D0)

# Run model with best set of parameters
model_calibrated = calibration(ND=365)
GoF_calibrated = GoF_wsse(model_calibrated, targets, num_days)
print('GoF of calibrated model is: ', GoF_calibrated)

FV = np.zeros((12,12))
for k in range(4): 
    for l in range(4):
        # first column
        FV[k,l]=beta[k,l]*S_u[k]/d_u[l]
        FV[k+4,l]=(1-eta_v[k])*beta[k,l]*S_v[k]/d_u[l]
        FV[k+8,l]=(1-eta_b[k])*beta[k,l]*S_b[k]/d_u[l]
        # second column
        FV[k,l+4]=(1-eta_v[l])*beta[k,l]*S_u[k]/d_v[l]
        FV[k+4,l+4]=(1-eta_v[l])*eta_v[k]*beta[k,l]*S_v[k]/d_v[l]
        FV[k+8,l+4]=(1-eta_v[l])*eta_b[k]*beta[k,l]*S_b[k]/d_v[l]
        # third column
        FV[k,l+8]=(1-eta_b[l])*beta[k,l]*S_u[k]/d_b[l]
        FV[k+4,l+8]=(1-eta_b[l])*(1-eta_v[k])*beta[k,l]*S_v[k]/d_b[l]
        FV[k+8,l+8]=(1-eta_b[l])*(1-eta_b[k])*beta[k,l]*S_b[k]/d_b[l]
w, v = eigh(FV)
R_0 = np.max(w)
print(R_0)



# Compare number of deaths to model output
t_range = raw_data[['date']].values[0:num_days_deaths]

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(t_range, raw_data.rolling_deaths[0:num_days_deaths], 'or', label = 'rolling target')
ax.plot(t_range, model_calibrated[0][0:num_days_deaths], label = 'model output')
date_form = mdates.DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Date')
ax.set_ylabel('New Deaths')
ax.legend(loc=0)
plt.tight_layout()
plt.savefig("Output/Calibration/calibration_deaths.png")
plt.show()



# Compare number of cases to model output
t_range = raw_data[['date']].values[0:num_days_cases]

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(t_range, raw_data.rolling_cases[0:num_days_cases], 'or', label = 'rolling')
ax.plot(t_range, 1.5*raw_data.rolling_cases[0:num_days_cases], 'og', label = r'1.5 $\times$ rolling')
ax.plot(t_range, 3*raw_data.rolling_cases[0:num_days_cases], 'ob', label = r'3 $\times$ rolling')
ax.plot(t_range, model_calibrated[1][0:num_days_cases], label = 'model output')
date_form = mdates.DateFormatter("%m-%d")
ax.xaxis.set_major_formatter(date_form)
ax.set_xlabel('Date')
ax.set_ylabel('New Cases')
ax.legend(loc=0)
plt.tight_layout()
plt.savefig("Output/Calibration/calibration_cases.png")
plt.show()




# Export parameters
with open('Data/calibrated_parameters.pkl', 'wb') as data:
    pickle.dump([beta, INPUT], data)

with open('Data/literature_parameters.pkl', 'wb') as data2:
    pickle.dump([gamma_u, gamma_v, gamma_b, mu_u, mu_v, mu_b, f, eta_v, eta_b], data2)





