% Script to test the construction of a MultiMSM using trajectories
% generated by a Gillespie algorithm on the toy trimer system. 
clear;

%parameters simulation discretization
discretizationParams = struct();
discretizationParams.dt = 2;
discretizationParams.M = 200;
discretizationParams.disc = [0,0.2,0.5,0.8,1];
discretizationParams.disc = linspace(0,1,9);

%set the assembly parameters in a struct
assemblyParams = struct();

gNuc = -24;                     %original = -7
assemblyParams.gNuc = gNuc;                    %contact energy for intermediates w/ n<nuc
assemblyParams.N = 3;                        %capsid size
assemblyParams.forwardRateConstant = 1.e5;     %in 1/(M s) %original 1e5
assemblyParams.c0 = 1.e-5*50;                  %total subunit concentrations (Molar units)  

%solve the rate equation model and get the sequence of rate and transition
%matrices
[X, Rt, Pt] = solveRateModel(assemblyParams, discretizationParams);
mon_fracs = X(:,1);
plotSolution(X, discretizationParams);

%generate trajectories using the sequeunce of transition matrices
samples = 600;
trajs = generateTrajectories(samples, Pt);
% plot(2 * (0:200), sum(trajs == 3,1)/samples, 'linewidth',2)

%construct a MultiMSM using the sampled trajectory data
lag = 1;
[MultiMSM, counts] = constructMultiMSM(lag, trajs, mon_fracs, discretizationParams.disc);

%solve the forward equation using the MultiMSM approx and compare
Y = solveAveragedModel(MultiMSM, discretizationParams);
compareSolutions(X, Y, discretizationParams);









function R = buildRateMatrix(c, assemblyParams)
    %builds the rate matrix and derivative for the capsid assembly system
    %note: this is for pointwise values of kT, c, and dcdT
    
    %gather the assembly parameters from the struct
    fin    = assemblyParams.forwardRateConstant;
    gNuc   = assemblyParams.gNuc;
    N      = assemblyParams.N;
    c0     = assemblyParams.c0;
    
    %b(2) means rate constant for 2->1, f(2) means rate constant 2->3, etc
    m = 1:1:N+1;
    
    %f(m) and b(m+1) are related by det balance
    b = fin*exp(gNuc * (m>0));
    b(1)=0; %can't have zero subunits
    b(N+1)=0;
    %forward rate constants are all equal, until size N
    f=fin*ones(N,1);
    f(N)=0;
    f(1)=f(1)*0.5;  %avoid double counting rho for the first step
    % I'm going to ignore degeneracy for the rest of the pathway
    
    %init the transition matrix and its derivative
    R = zeros(N); 
    
    %compute all the inner transition matrix entries
    for i = 2:(N-1)
        
        %compute the forward and backward rates for this row, and derivs
        forward  = f(i) * c(1) * c0;   
        backward = (i-1)/i * b(i);     
        to_monomer = 1.0/i * b(i); 
        
        %get the diagonal entries, negative row sum
        S = - (forward+backward+to_monomer); 
        
        %enter into the matrix
        R(i,i-1) = backward;  R(i,i) = S;  R(i,i+1) = forward;  R(i,1) = R(i,1) + to_monomer;
    end
    
    %do the first and last row seperately
    row1sum = 0; 
    for i = 3:N
        R(1,i) = f(i-1) * c(i-1) * c0 / (i-1);    row1sum = row1sum + R(1,i);
    end
    R(1,2) = 2*f(1) * c(1) * c0;    R(1,1) = - (R(1,2)+row1sum);
    R(N,N-1) = (N-1)/N * b(N); R(N,1) = 1/N * b(N); R(N,N) = - (R(N,N-1)+R(N,1));
    
end

function [p2,R] = evolveForward(p1, L, dt)
    %take probability vector to next timestep via rate matrix
    R = expm(L*dt);
    p2 = p1 * R;
end

function [X, Rt, Pt] = solveRateModel(assemblyParams, discretizationParams)
    %solve the rate equation model by constructing rate matrices and
    %propogating the solution in time using a probability transition matrix
    
    %extract the needed disc parameters
    dt = discretizationParams.dt;
    M = discretizationParams.M;
    
    %init storage for the concentrations at each step, and mass-weighted
    %monomer fractions
    c = zeros(1,3); c(1) = assemblyParams.c0;
    x = zeros(1,3); x(1) = 1;
    X = zeros(M+1,3); X(1,:) = x;
    
    %init cells to store all rate and transition matrices
    Rt = cell(M, 1);
    Pt = cell(M, 1);
    
    %solve for the mass fraction
    for i=1:M
        R = buildRateMatrix(c, assemblyParams);
        [x,P] = evolveForward(x, R, dt);
        c = x * assemblyParams.c0;
        X(i+1,:) = x;
        Rt{i} = R; Pt{i} = P;
    end


end

function X = solveAveragedModel(Pt, discretizationParams)

    %extract the needed disc parameters
    dt = discretizationParams.dt;
    M = discretizationParams.M;
    disc = discretizationParams.disc;
    
    %init storage for the concentrations at each step, and mass-weighted
    %monomer fractions
    x = zeros(1,3); x(1) = 1;
    X = zeros(M+1,3); X(1,:) = x;
    
    %solve for the mass fraction
    for i=1:M
        
        %get the current monomer fraction, find its index in disc
        mon_frac = x(1);
        interval = getInterval(mon_frac, disc);
        P = Pt{interval};
        x = x * P;
        X(i+1,:) = x;
    end

end

function trajs = generateTrajectories(samples, Pt)
    %generate trajectories according to the transition matrix sequence
    
    traj_length = length(Pt);
    trajs = zeros(samples, traj_length+1);
    
    for i = 1:samples
        
        state = 1;
        traj = zeros(1, traj_length+1); traj(1) = state;
        
        for j = 1:traj_length
            
            row = Pt{j}(state,:);
            state = sample_distr(row);
            traj(j+1) = state;
            
        end
        
        trajs(i,:) = traj;

    end

end

function [MultiMSM, MultiCounts] = constructMultiMSM(lag, trajs, mon_fracs, disc)
    %construct MSMs in each discretization interval using mon frac data and
    %trajectories
    
    %get the number of samples and length of trajectories
    [samples, traj_length] = size(trajs);
    
    %get the number of intervals in the discretization and init a matrix of
    %zeros in each of them
    num_intervals = length(disc)-1;
    MultiCounts = cell(num_intervals,1);
    for i = 1:num_intervals
        MultiCounts{i} = zeros(max(max(trajs)));
    end
    
    %loop over each sample and log counts of each transition for the given
    %lag time in the appropriate monomer fraction interval
    for sample = 1:samples
        
        traj = trajs(sample,:);
        
        for start = 1:(traj_length-lag)
            
            start_state = traj(start);
            end_state   = traj(start+lag);
            mon_frac    = mon_fracs(start);
            
            interval = getInterval(mon_frac, disc);
            
            MultiCounts{interval}(start_state, end_state) = ...
                MultiCounts{interval}(start_state, end_state) + 1;
             
        end
          
    end

    %normalize each of the rows of the count matrices to construct MSM
    MultiMSM = cell(num_intervals,1);
    for i = 1:num_intervals
        
        if MultiCounts{i}(3,3) < 1
            MultiCounts{i}(3,3) = 1;
        end
        
        MultiMSM{i} = MultiCounts{i} ./ sum(MultiCounts{i},2);
        
    end
    
end



function x = sample_distr(p)
    %sample the discrete probability distribution in p
    
    edges = [0; cumsum(p(:))];
    x = discretize(rand(), edges);   
    
end

function interval = getInterval(number, discretization)
    %takes a number in (0,1) and a discretization of the unit interval and
    %returns the imterval the number falls in
    
    interval = find(discretization < number, 1, 'last');

end


function plotSolution(X, discretizationParams)
    
    dt = discretizationParams.dt;
    M  = discretizationParams.M;
    
    figure(1)
    hold on
    for i = 1:3
        plot(dt * (0:M), X(:,i),'linewidth',2)
    end
    
    legend("Monomer", "Dimer", "Trimer")
    xlabel("Time")
    ylabel("Mass Fraction")
    
end


function compareSolutions(X, Y, discretizationParams)
    
    dt = discretizationParams.dt;
    M  = discretizationParams.M;
    
    figure(2)
    hold on
    for i = 1:3
        plot(dt * (0:M), X(:,i),'linewidth',2)
    end
    for i = 1:3
        plot(dt * (0:M), Y(:,i),'linewidth',2,'linestyle','--')
    end
    
    legend("Monomer", "Dimer", "Trimer")
    xlabel("Time")
    ylabel("Mass Fraction")
    
end



