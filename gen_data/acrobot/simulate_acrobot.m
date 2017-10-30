function simulate_acrobot()
    % load the URDF
    r = RigidBodyManipulator('acrobot.urdf');

    % setup and solve the optimization problem
    x0 = zeros(4,1); % initial state
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
    
    % visualize the solution
    v = r.constructVisualizer();
    playback(v,xtraj);
end