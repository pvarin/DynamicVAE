function generate_trajectories(num_trajectories)
    % load the URDF
    r = RigidBodyManipulator('acrobot.urdf');
    
    % generate a bunch of trajectories
    rng(2);
    j=0;
    for i=1:num_trajectories
        % setup and solve the optimization problem
        x0 = rand(4,1)-.5; % initial state
        xf = [pi;0;0;0]; % final state
        tf0 = 4; % the initial trajector length
        N = 21; % number of knot points
        prog = DircolTrajectoryOptimization(r,N,[2 6]);
        prog = prog.addStateConstraint(ConstantConstraint(x0),1);
        prog = prog.addStateConstraint(ConstantConstraint(xf),N);
        prog = prog.addRunningCost(@acrobot_running_cost);
        prog = prog.addFinalCost(@acrobot_final_cost);

        traj_init.x = PPTrajectory(foh([0,tf0],[double(x0),double(xf)]));
        [xtraj,utraj,z,F,info] = prog.solveTraj(tf0,traj_init);
        
        t = linspace(0,xtraj.tspan(end),N);
        xtraj_data = xtraj.eval(t);
        if info==1
            filename = ['data/acrobot_trajectory_', sprintf('%05d',j)];
            csvwrite(filename,xtraj_data);
            j = j+1;
        end
    end
end