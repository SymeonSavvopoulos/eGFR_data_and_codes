function [In_max_mean, In_max_std,f1_all]=studies_pat_age_gt50()

A=load('measurements_pat_age_gt50.m');
do_age=load('do_ages_pat_age_gt50.m');
interp_data=load('interp_dat_pat_age_gt50.m');
XX=size(A);

lambda=[0.00733231100000000	1.22000000000000e-06	0.0159178000000000	0.247280386000000	0.0510182110000000	0.000200669000000000	0.0842725070000000	0.287185819000000	0.214383902000000	0.999912339000000	0.0113237310000000	0.999921620000000	0.0262413800000000	0.999960942000000	0.0202877980000000	0.0463455920000000	3.89000000000000e-05	0.000116389000000000	0.269532244000000	0.999845806000000	0.0504214080000000	0.0678144370000000	0.243837996000000	0.0277927290000000	0.0228471670000000	0.0151569970000000	0.201675048000000	0.00933503000000000	0.0197452130000000	0.0256867450000000	0.0534980110000000	0.0457849340000000	0.0307470220000000	0.0449231080000000	0.999980634000000	0.00705027200000000	0.999968127000000	0.0114014480000000	0.0345590080000000	0.0844121470000000	0.633111544000000	0.0480016780000000	0.0224064680000000	0.00113108400000000	0.00355890800000000	0.0336895580000000	0.249595708000000	0.999995482000000	0.999985773000000	0.489993443000000	0.999973828000000	0.119462886000000	0.999997480000000	0.999977031000000	0.999933757000000	0.0148705350000000	0.999961432000000	0.199640014000000	0.0180893540000000	0.209189099000000	0.999776449000000	0.284116500000000	0.160788816000000	0.0317465070000000	3.06000000000000e-05	0.999976926000000	0.118054186000000	0.999971985000000	0.999981209000000	0.999992131000000	0.0246426330000000	0.0239046550000000	0.496747021000000	0.0136660920000000	0.178844461000000	0.999996903000000	0.999962008000000	0.999959832000000	0.0154568880000000	0.539710784000000	0.323328193000000	0.422912169000000	0.999701935000000	0.128149945000000	0.00256946300000000	0.0216995030000000	0.116367157000000	1.81000000000000e-05	0.999997055000000	0.0660000000000000	0.0412620480000000	0.999994478000000	0.0249293210000000	0.0661454380000000	0.00636042400000000	0.0694667010000000	0.999939833000000	0.0881170620000000	0.999958408000000	0.999902565000000	0.0124353050000000	0.0179008500000000	0.999995242000000	0.000200669000000000	0.142503751000000	0.0286101720000000	0.999937864000000	0.999997791000000	0.0170486570000000	0.999993764000000	0.0899141500000000	0.0692229900000000	0.0821465890000000	0.999943333000000	0.0611280250000000	0.999956020000000	0.160137929000000	0.0300597840000000	0.999972118000000	0.169320310000000	0.318935667000000	0.206307912000000	0.0724001020000000	0.0146376210000000	0.000118990000000000	0.117129045000000	0.234884541000000	0.245561604000000	0.999959097000000	0.0153235190000000	0.999993723000000	0.314536219000000	0.999950654000000	0.294957349000000	0.0175281060000000	0.999982190000000	0.130559383000000	0.0503517810000000	0.0995706940000000	0.999999417000000	0.252868609000000	0.172421228000000	0.0159010020000000	0.0119265930000000	0.999995483000000	0.000945779000000000	0.124907585000000	0.999973828000000	0.999954331000000	0.319286081000000	0.150517671000000	0.00654701300000000	0.0127932410000000	0.207455205000000	0.219942863000000	0.0412819960000000	0.383902177000000	0.999892364000000	0.0383664680000000	3.06000000000000e-05	0.00980853700000000	0.137752852000000	0.0142129290000000	0.437960186000000	0.00301000000000000	0.163197173000000	0.0803260130000000	0.0262010550000000	0.122160268000000	9.33000000000000e-06	0.0455335000000000	0.0131874950000000	0.999965626000000	0.0994173590000000	0.299698401000000	0.999837972000000	0.335845591000000	0.198000000000000	0.0147858000000000	0.224279459000000	0.462376642000000	0.999827879000000	0.767630964000000	0.0532601490000000	0.318829455000000	0.999999145000000	0.0243960340000000	0.0113473540000000	0.999957638000000	0.224929413000000	0.999586957000000	0.0238480620000000	0.222481298000000	0.0992588660000000	0.103688389000000	0.0295614740000000	0.331177223000000	0.0357484830000000	0.113539125000000	0.0550541860000000	0.00712160800000000	0.0529489360000000	0.153655925000000	0.152912706000000	0.999982109000000	0.108766024000000	0.149613275000000	0.618350047000000	0.201604353000000	0.999845806000000	0.0133862590000000	0.788157617000000	0.0307385990000000	0.115430579000000	0.999995242000000	1.65000000000000e-05	0.0420843400000000	0.0121318550000000	0.00610489200000000	0.370173893000000	3.62000000000000e-05	0.165294529000000	0.0236208870000000	0.0201318960000000	0.0190905540000000	0.158913947000000	0.0427000000000000	0.0656628790000000	0.196777301000000	0.240734462000000	0.999969254000000	0.399889263000000	0.189000000000000	0.230387984000000	0.999986021000000	0.282076056000000	0.999993723000000	0.999978258000000	0.0296000650000000	0.999982190000000	0.0658642380000000	0.141728596000000	0.0571567260000000	0.0737670580000000	0.0255143870000000	0.145772985000000	0.0732842630000000	0.0300156220000000	0.0871169720000000	0.0144075250000000	0.327166350000000	0.999929204000000	0.999954332000000	0.148750409000000	0.117279262000000	0.999933757000000	1.05000000000000e-05	0.194521462000000	0.0296009050000000	0.244252098000000	0.0674304470000000	0.0458096910000000	0.388358029000000	0.999668668000000	0.999976926000000	0.0174503450000000	0.0731229250000000	0.0182961880000000	0.356547747000000	0.321725554000000	0.999974449000000	0.999951730000000	0.892799300000000	0.141544211000000	0.00284429000000000	0.0414000000000000	0.0197912360000000	0.136346257000000	0.593893476000000	0.169321103000000	0.0695537910000000	0.218880537000000	0.0294263860000000	0.219826991000000	0.999998172000000	0.0195168170000000	0.102575902000000	0.0601304720000000	0.139998696000000	0.0283183820000000	0.999947253000000	0.197360687000000	0.0501642970000000	0.449430964000000	0.0580157710000000	0.999586958000000	0.156540436000000	0.999974987000000	0.215931895000000	0.999917186000000	0.999865775000000	0.334399319000000	0.0442538570000000	0.0119862580000000	0.102291469000000	0.999997264000000	0.0143738080000000	0.999919080000000	0.194317604000000	0.999997293000000	0.242221822000000	0.0870277560000000	0.229544054000000	0.999977551000000	0.309793757000000	0.0224984340000000	0.106514677000000	0.0303177620000000	0.0313767410000000	0.273313388000000	0.999984003000000	0.336561849000000	0.314773286000000	0.0128842620000000	0.999873829000000	0.999994153000000	0.303515198000000	0.0995115630000000	0.999957706000000	0.223444603000000	0.0151768250000000	6.54000000000000e-05	0.125660685000000	0.0397230640000000	0.00315152600000000	0.999911299000000	0.999959528000000	0.0359150950000000	0.999995242000000	0.0934240420000000	0.212485135000000	0.105074226000000	0.311402561000000	0.167274740000000	0.134564608000000	0.226776575000000	0.0876381400000000	0.286671643000000	0.269009974000000	0.00699664600000000	0.999947084000000	0.0211855690000000	0.0334133960000000	0.117929474000000];


cases_min=zeros(length(A)/2,4);
prob_min=zeros(362,1);

cases_max=zeros(362,4);
prob_max=zeros(362,1);

cases=zeros(362,4);
prob=zeros(362,1);

for i=1:length(A)/2
    len(1,i)=nnz(A(2:XX(1),2*i))+1;

i

    mytime=zeros(len(1,i),1);
mydata=zeros(len(1,i),1);

  for k=1:len(1,i)

mytime(k,1) = A(k,2*i-1);
mydata(k,1) =A(k,2*i);
  speed(i)=(A(2,2*i)-A(1,2*i))/(A(2,2*i-1)-A(1,2*i-1));
  end
  
  xq=[0:10:100];
  vq1 = interp1(mytime,mydata,xq);
c_last(i)=19.2214+0.7534*vq1(1)+20.5617*(vq1(7)-vq1(1))/vq1(1)-0.1543*do_age(i);


  
  if c_last(i)<40
      c_last_min(i)=2;
      c_last_max(i)=c_last(i)+40;
  else
      c_last_min(i)=c_last(i)-40;
      c_last_max(i)=c_last(i)+40;
  end
      
gamma=0.000032;
alpha_min(i)=2*gamma/c_last_min(i)^2;
alpha_max(i)=2*gamma/c_last_max(i)^2;

beta_min(i)=-3*(alpha_min(i)*gamma/2)^0.5;
beta_max(i)=-3*(alpha_max(i)*gamma/2)^0.5;

g_a_min(i)=gamma/alpha_min(i);
g_a_max(i)=gamma/alpha_max(i);

c_3_min(i)=c_last_min(i)/3;
c_3_max(i)=c_last_max(i)/3;

theta_min(i)=c_last_min(i)/2;
theta_max(i)=c_last_max(i)/2;






% if A(1,2*i)>theta_min(i) & speed(i)>0
%     cases_min(i,1)=1;
%     prob_min(i,1)=0;
% elseif A(1,2*i)>theta_min(i) & speed(i)<0
%     cases_min(i,2)=1;
    for k=1:length(lambda)
        f_min=@(x,y) [ y(2); -lambda(k)*y(2)-(alpha_min(i)*y(1)*y(1)*y(1)+beta_min(i)*y(1)*y(1)+gamma*y(1))];
    [ts,ys_min] = ode45(f_min,[0,630],[interp_data(7,i);(interp_data(7,i)-interp_data(6,i))/10]);    
    if ys_min(end,1)<theta_min(i)
        reject_min_1(k,i)=1;
    else
        reject_min_1(k,i)=0;
    end
    end
end
    for i=1:length(A)/2
totp_min(i)=sum(reject_min_1(:,i));
end
prob_min=totp_min'/354;
    


%==============================================================================================================================
%==============================================================================================================================
%==============================================================================================================================




for i=1:length(A)/2
    len(1,i)=nnz(A(2:XX(1),2*i))+1;

i

    mytime=zeros(len(1,i),1);
mydata=zeros(len(1,i),1);

  for k=1:len(1,i)

mytime(k,1) = A(k,2*i-1);
mydata(k,1) =A(k,2*i);
  speed(i)=(A(2,2*i)-A(1,2*i))/(A(2,2*i-1)-A(1,2*i-1));
  end
  
  xq=[0:10:100];
  vq1 = interp1(mytime,mydata,xq);
c_last(i)=19.2214+0.7534*vq1(1)+20.5617*(vq1(7)-vq1(1))/vq1(1)-0.1543*do_age(i);



  if c_last(i)<40
      c_last_min(i)=2;
      c_last_max(i)=c_last(i)+40;
  else
      c_last_min(i)=c_last(i)-40;
      c_last_max(i)=c_last(i)+40;
  end
      
gamma=0.000032;
alpha_min(i)=2*gamma/c_last_min(i)^2;
alpha_max(i)=2*gamma/c_last_max(i)^2;

beta_min(i)=-3*(alpha_min(i)*gamma/2)^0.5;
beta_max(i)=-3*(alpha_max(i)*gamma/2)^0.5;

g_a_min(i)=gamma/alpha_min(i);
g_a_max(i)=gamma/alpha_max(i);

c_3_min(i)=c_last_min(i)/3;
c_3_max(i)=c_last_max(i)/3;

theta_min(i)=c_last_min(i)/2;
theta_max(i)=c_last_max(i)/2;




    for k=1:length(lambda)
        f_max=@(x,y) [ y(2); -lambda(k)*y(2)-(alpha_max(i)*y(1)*y(1)*y(1)+beta_max(i)*y(1)*y(1)+gamma*y(1))];
    [ts,ys_max] = ode45(f_max,[0,630],[interp_data(7,i);(interp_data(7,i)-interp_data(6,i))/10]);    
    if ys_max(end,1)<theta_max(i)
        reject_max_1(k,i)=1;
    else
        reject_max_1(k,i)=0;
    end

    
    end
end

 for i=1:length(A)/2
totp_max(i)=sum(reject_max_1(:,i));
end

prob_max=totp_max'/354;



for i=1:length(A)/2
    len(1,i)=nnz(A(2:XX(1),2*i))+1;

i

    mytime=zeros(len(1,i),1);
mydata=zeros(len(1,i),1);

  for k=1:len(1,i)

mytime(k,1) = A(k,2*i-1);
mydata(k,1) =A(k,2*i);
  speed(i)=(A(2,2*i)-A(1,2*i))/(A(2,2*i-1)-A(1,2*i-1));
  end
  
  xq=[0:10:100];
  vq1 = interp1(mytime,mydata,xq);
c_last(i)=19.2214+0.7534*vq1(1)+20.5617*(vq1(7)-vq1(1))/vq1(1)-0.1543*do_age(i);


      
gamma=0.000032;
alpha(i)=2*gamma/c_last(i)^2;

beta(i)=-3*(alpha(i)*gamma/2)^0.5;

g_a(i)=gamma/alpha(i);

c_3(i)=c_last(i)/3;


theta(i)=c_last(i)/2;





    for k=1:length(lambda)
        f=@(x,y) [ y(2); -lambda(k)*y(2)-(alpha(i)*y(1)*y(1)*y(1)+beta(i)*y(1)*y(1)+gamma*y(1))];
    [ts,ys] = ode45(f,[0,630],[interp_data(7,i);(interp_data(7,i)-interp_data(6,i))/10]);    
    if ys(end,1)<theta(i)
        reject_1(k,i)=1;
    else
        reject_1(k,i)=0;
    end
    end
   
end 


 for i=1:length(A)/2
totp(i)=sum(reject_1(:,i));
end
prob=totp'/354;



thre_1=[0.05:0.05:1];

success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=20;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_min(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_min(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_min(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);


end
u20=[xaxis',yaxis'];


success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=30;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_min(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_min(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_min(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_30min(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u30_min=[xaxis',yaxis'];


success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=40;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_min(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_min(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_min(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_40min(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u40_min=[xaxis',yaxis'];

success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=50;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_min(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_min(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_min(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_min(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_50min(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u50_min=[xaxis',yaxis'];


thre_1=[0.05:0.05:1];

success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=20;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);


end
u20=[xaxis',yaxis'];


success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=30;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_30(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u30=[xaxis',yaxis'];


success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=40;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_40(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u40=[xaxis',yaxis'];

success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=50;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_50(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u50=[xaxis',yaxis'];


thre_1=[0.05:0.05:1];

success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=20;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_max(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_max(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_max(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);


end
u20=[xaxis',yaxis'];


success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=30;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_max(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_max(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_max(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_30max(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u30_max=[xaxis',yaxis'];


success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=40;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_max(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_max(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_max(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;    
    
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_40max(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u40_max=[xaxis',yaxis'];

success_and_predicted_success=zeros(362,length(thre_1));
failure_but_predicted_success=zeros(362,length(thre_1));
success_but_predicted_failure=zeros(362,length(thre_1));
failure_and_predicted_failure=zeros(362,length(thre_1));
GFR_critical=50;

for k=1:length(thre_1)
for i=1:length(A)/2
    
if prob_max(i)>thre_1(k) & interp_data(end,i)<GFR_critical
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)>thre_1(k) & interp_data(end,i)>GFR_critical
    success_but_predicted_failure(i,k)=1;
elseif prob_max(i)<=thre_1(k) & interp_data(end,i)>=GFR_critical
    success_and_predicted_success(i,k)=1;
elseif prob_max(i)<thre_1(k) & interp_data(end,i)<GFR_critical
    failure_but_predicted_success(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)<=GFR_critical  
    failure_and_predicted_failure(i,k)=1;
elseif prob_max(i)==1 &  interp_data(end,i)>=GFR_critical
    success_but_predicted_failure(i,k)=1;
end
end
x11(k)=sum(success_and_predicted_success(:,k));
x12(k)=sum(failure_but_predicted_success(:,k));
x21(k)=sum(success_but_predicted_failure(:,k));
x22(k)=sum(failure_and_predicted_failure(:,k));

yaxis(k)=x11(k)/(x11(k)+x21(k));
xaxis(k)=x12(k)/(x12(k)+x22(k)+1e-6);

f1_50max(k)=2*x11(k)/(2*x11(k)+x12(k)+x21(k));
end
u50_max=[xaxis',yaxis'];


u30_min=[0 0; u30_min; 1 1];
u40_min=[0 0; u40_min; 1 1];
u50_min=[0 0; u50_min; 1 1];


u30=[0 0; u30; 1 1];
u40=[0 0; u40; 1 1];
u50=[0 0; u50; 1 1];


u30_max=[0 0; u30_max; 1 1];
u40_max=[0 0; u40_max; 1 1];
u50_max=[0 0; u50_max; 1 1];

I30 = cumtrapz(u30(:,1),u30(:,2));
I40 = cumtrapz(u40(:,1),u40(:,2));
I50 = cumtrapz(u50(:,1),u50(:,2));

I30_min = cumtrapz(u30_min(:,1),u30_min(:,2));
I40_min = cumtrapz(u40_min(:,1),u40_min(:,2));
I50_min = cumtrapz(u50_min(:,1),u50_min(:,2));

I30_max = cumtrapz(u30_max(:,1),u30_max(:,2));
I40_max = cumtrapz(u40_max(:,1),u40_max(:,2));
I50_max = cumtrapz(u50_max(:,1),u50_max(:,2));

thr=[30;40;50];
In_min=[I30_min(end);I40_min(end);I50_min(end)];
In=[I30(end);I40(end);I50(end)];
In_max=[I30_max(end);I40_max(end);I50_max(end)];

In_max_mean=mean(In_max);
In_max_std=std(In_max);



f1_all=[f1_30min;f1_40min;f1_50min;f1_30;f1_40;f1_50;f1_30max;f1_40max;f1_50max];









