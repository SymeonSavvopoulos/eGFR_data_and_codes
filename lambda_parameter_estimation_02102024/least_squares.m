function val=least_squares(k, mydata, mytime)
y0(1) = mydata(1); y0(2) = (mydata(2)-mydata(1))/(mytime(2)-mytime(1));
modelfun=@(t,x)GFR_Model(t,x,k);
[t ycalc]=ode45(modelfun,mytime,y0);
resid = (ycalc(:,1)-mydata(:,1)).*(ycalc(:,1)-mydata(:,1));
val = sum(sum(resid));
end