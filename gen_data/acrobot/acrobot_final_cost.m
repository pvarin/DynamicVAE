function [h,dh] = acrobot_final_cost(t,x)
    h = t;
    dh = [1,zeros(1,size(x,1))];
    return;

    xd = repmat([pi;0;0;0],1,size(x,2));
    xerr = x-xd;
    xerr(1,:) = mod(xerr(1,:)+pi,2*pi)-pi;

    Qf = 100*diag([10,10,1,1]);
    h = sum((Qf*xerr).*xerr,1);

    if (nargout>1)
      dh = [0, 2*xerr'*Qf];
    end
end