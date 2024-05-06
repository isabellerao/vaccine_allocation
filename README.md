# Vaccination for communicable endemic diseases

This repository contains the official implementation of Rao IJ, Brandeau ML. Vaccination for communicable endemic diseases: Optimal allocation of initial and booster vaccine doses.  

For some communicable endemic diseases (e.g., influenza, COVID-19), vaccination is an effective means of preventing the spread of infection and reducing mortality, but must be augmented over time with vaccine booster doses. We consider the problem of optimally allocating a limited supply of vaccines over time between different subgroups of a population and between initial versus booster vaccine doses, allowing for multiple booster doses. We first consider an SIS model with interacting population groups and four different objectives: those of minimizing cumulative infections, deaths, life years lost, or quality-adjusted life years lost due to death. We solve the problem sequentially: for each time period, we approximate the system dynamics using Taylor series expansions, and reduce the problem to a piecewise linear convex optimization problem for which we derive intuitive closed-form solutions. We then extend the analysis to the case of an SEIS model. In both cases vaccines are allocated to groups based on their priority order until the vaccine supply is exhausted. Numerical simulations show that our analytical solutions achieve results that are close to optimal with objective function values significantly better than would be obtained using simple allocation rules such as allocation proportional to population group size. In addition to being accurate and interpretable, the solutions are easy to implement in practice. Interpretable models are particularly important in public health decision making. 

## Running the Model

The code can be used to reproduce all graphs and tables in the paper. 
- "SIS COVID Example - Calibration.py" calibrates the model to the COVID-19 example in New York State.
- "SIS COVID Example - Quality of decisions.py" generates all tables and graphs for the SIS example in Section 4. 
- The script "SEIS Example.py" generates all tables and graphs for the SEIS example in Section 6. 
- "requirements.txt" contains a list of all the packages required for the project, along with their versions. It ensures consistent setup across different environments. 
