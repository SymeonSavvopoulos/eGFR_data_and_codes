function dy = GFR_Model(t,y,k)
dy = zeros(2,1);
alpha = k(1); beta = k(2); gamma = k(3); lambda = k(4);
dy(1,1) = y(2);
dy(2,1)= -lambda*y(2)-(alpha*y(1)*y(1)*y(1)+beta*y(1)*y(1)+gamma*y(1));
end