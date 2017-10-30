function [g,dg] = acrobot_running_cost(~,x,u)
    R = 1;
    g = sum((R*u).*u,1);
    dg = [zeros(1,1+size(x,1)),2*u'*R];
    return;

    xd = repmat([pi;0;0;0],1,size(x,2));
    xerr = x-xd;
    xerr(1,:) = mod(xerr(1,:)+pi,2*pi)-pi;

    Q = diag([10,10,1,1]);
    R = 100;
    g = sum((Q*xerr).*xerr + (R*u).*u,1);

    if (nargout>1)
      dgddt = 0;
      dgdx = 2*xerr'*Q;
      dgdu = 2*u'*R;
      dg = [dgddt,dgdx,dgdu];
    end
end