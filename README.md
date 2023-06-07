# Lyapunov
The Lyapunov exponent for trajectories of time series data in a 6-dimensional space. The Lyapunov exponent is a measure of the rate of divergence or convergence of nearby trajectories in a dynamical system, and it is often used to quantify the degree of chaos in a system.
# A general pipeline 

1) Time series extraction using Shaefer parcellation (n=100)
2) Load hcp functional connectivitty from brainspace toolbox
3) Extract gradient (3 components) 
4) Project Time Series into Gradient Space (Pearson Correlation)
5) Reconstruct the Phase Space (first derivative) - get a 6 dim space by conatenating corr and rate data
6) Split data into half
7) Identify Nearest Neighbors: Intial Condition (D_init)
8) Track Divergence (tLya = 3) 
9) Get the Final distance (D_final)
10) Calculate Lyapunov exponent: Lyapunov = np.log(D_final / D_init) / tLya
Calculate the Lyapunov Exponents: Once you have your phase space, you can calculate the Lyapunov exponents. This involves tracking how small points of initial conditions evolve over time. If the time points expand at an exponential rate, then the system is chaotic and the rate of expansion is given by the Lyapunov exponent
11) Plot results, check if data is normally ditributed (Shapiro test) 
12) T-test for statistical significant difference between two groups
13) Mixed effect models 
14) Interpret the results
