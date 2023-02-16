%test out how the multi-msm approximation affects the results for an
%idealized trimer model
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
plotSolution(X, discretizationParams)

%construct an averaged rate/transition matrix in each interval of the
%monomer fraction discretization
[Ravg, Pavg] = computeAveragedMatrices(X, Rt, Pt, discretizationParams);

%solve the rate model using the averaged transition matrices
Y = solveAveragedModel(Pavg, assemblyParams, discretizationParams);
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

function X = solveAveragedModel(Pt, assemblyParams, discretizationParams)

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



function [Ravg, Pavg] = computeAveragedMatrices(X, Rt, Pt, discretizationParams)
    %determine which discretization interval a particular monomer fraction
    %in X corresponds to, then average all the matrices from that interval
    
    %extract the needed discretization parameter
    M = discretizationParams.M;
    disc = discretizationParams.disc;
    num_intervals = length(disc) - 1;
    
    %init cells to store the averaged matrices
    Ravg = cell(num_intervals,1);
    Pavg = cell(num_intervals,1);
    
    %init cells of cells to store all observed matrices in an interval
    all_rates = cell(num_intervals,1);
    all_probs = cell(num_intervals,1);
    for i = 1:num_intervals
        all_rates{i} = cell(0);
        all_probs{i} = cell(0);
    end
    
    %loop over all time steps in the discrete trajectory
    for i = 1:M
        
        %determine which interval the current monomer fraction falls in
        mon_frac = X(i,1);
        interval = getInterval(mon_frac, disc);
        
        %add the rate and transition matrices at this time to the interval
        mat_index = length(all_rates{interval});
        all_rates{interval}{mat_index+1} = Rt{i};
        all_probs{interval}{mat_index+1} = Pt{i};
    end
    
    %average all matrices within a cell
    for i = 1:num_intervals
        
        num_samples = length(all_rates{i});
        Psum = all_probs{i}{1}; Rsum = all_rates{i}{1};
        
        for j = 2:num_samples
            
            Psum = Psum + all_probs{i}{j};
            Rsum = Rsum + all_rates{i}{j};
            
        end
        
        Psum = Psum / num_samples; Pavg{i} = Psum;
        Rsum = Rsum / num_samples; Ravg{i} = Rsum;
          
    end

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


