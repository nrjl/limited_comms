function [X_new, F] = metroHaste(density_fun, limits, newPts, stream, burn_in, q_std)
% Limits should be 2*ndim
% X_new will be newPts*ndim

% Constants
if nargin == 4
    burn_in = max(200, floor(newPts/10));
    q_std = diff(limits)./20;
elseif nargin == 5
    q_std = diff(limits)./20;
end

ndim = size(limits,2);


% Q (proposal distribution)


% Initialise sampling
X_new = zeros(newPts+burn_in,ndim);
X_current=limits(1,:) + rand(stream, [1,ndim]).*diff(limits);

t = 1;
 
% RUN METROPOLIS-HASTINGS SAMPLER
while t <= newPts+burn_in
 
    % Generate proposed sample
    X_proposed = X_current + randn(stream, [1,ndim]).*q_std;
    while 1
        inlimits = true;
        for i = 1:ndim
            if X_proposed(i) < limits(1,i) || X_proposed(i) > limits(2,i)
                inlimits = false;
                break
            end
        end
        if ~inlimits
            X_proposed = X_current + randn(stream, [1,ndim]).*q_std;
        else
            break
        end
    end
 
    % Acceptance ratio
    alfa = density_fun(X_proposed)/density_fun(X_current);
 
    % Accept?
    if alfa < 1
        pp = rand(stream);
        if pp > alfa    % Reject
            X_proposed = X_current;
        end
    end
    
    X_new(t,:) = X_proposed;
    t = t+1;
    %disp(t);
    X_current = X_proposed;
end

X_new = X_new(burn_in+1:end,:);
F = 0;