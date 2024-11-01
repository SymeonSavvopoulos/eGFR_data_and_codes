clc
clear

%patients sex:males
[dmeansmale,dstdsmale,f1smale]=studies_pat_sex_male()

%donor sex:males
[dmeandsmale,dstddsmale,f1dsmale]=studies_don_sex_male()

%patients age:less 50
[dmeanpagel50,dstdpagel50,f1pagel50]=studies_pat_age_less50()


%patients sex:females
[dmeansfemale,dstdsfemale,f1sfemale]=studies_pat_sex_female()

%donor sex:females
[dmeandsfemale,dstddsfemale,f1dsfemale]=studies_don_sex_female()

%patients age:greater than 50
[dmeanpagegt50,dstdpagegt50,f1pagegt50]=studies_pat_age_gt50()



figure(1)
xlim([0 5])

x1=[1:1:3];
y1=[dmeansmale,dmeandsmale,dmeanpagel50];
err1=[dstdsmale,dstddsmale,dstdpagel50];

errorbar(x1,y1,err1,"o")
names = {'male patients'; 'male donors'; 'patients less than 50'};
set(gca,'xtick',[1:3],'xticklabel',names)

hold on

x2=[1.2:1:3.2];
y2=[dmeansfemale,dmeandsfemale,dmeanpagegt50];
err2=[dstdsfemale,dstddsfemale,dstdpagegt50];
errorbar(x2,y2,err2,"o")
names = {'female patients'; 'female donors'; 'patients greater than 50'};
set(gca,'xtick',[1.2:3.2],'xticklabel',names)





for i=1:3
min_f1smale(i)=mean(f1smale(i+6,2:end));
std_f1smale(i)=std(f1smale(i+6,2:end));
end

maxf1_smale=mean([min_f1smale(1),min_f1smale(2) min_f1smale(3)]);
maxstdf1_smale=mean([std_f1smale(1),std_f1smale(2) std_f1smale(3)]);


for i=1:3
min_f1dsmale(i)=mean(f1dsmale(i+6,2:end));
std_f1dsmale(i)=std(f1dsmale(i+6,2:end));
end

maxf1_dsmale=mean([min_f1dsmale(1),min_f1dsmale(2) min_f1dsmale(3)]);
maxstdf1_dsmale=mean([std_f1dsmale(1),std_f1dsmale(2) std_f1dsmale(3)]);



for i=1:3
min_f1pagel50(i)=mean(f1pagel50(i+6,2:end));
std_f1pagel50(i)=std(f1pagel50(i+6,2:end));
end




maxf1_pagel50=mean([min_f1pagel50(1),min_f1pagel50(2) min_f1pagel50(3)]);
maxstdf1_pagel50=mean([std_f1pagel50(1),std_f1pagel50(2) std_f1pagel50(3)]);


xf1=[maxf1_smale,maxf1_dsmale,maxf1_pagel50]';
yf1=[maxstdf1_smale,maxstdf1_dsmale,maxstdf1_pagel50]';








for i=1:3
min_f1sfemale(i)=mean(f1sfemale(i+6,2:end));
std_f1sfemale(i)=std(f1sfemale(i+6,2:end));
end

maxf1_sfemale=mean([min_f1sfemale(1),min_f1sfemale(2) min_f1sfemale(3)]);
maxstdf1_sfemale=mean([std_f1sfemale(1),std_f1sfemale(2) std_f1sfemale(3)]);


for i=1:3
min_f1dsfemale(i)=mean(f1dsfemale(i+6,2:end));
std_f1dsfemale(i)=std(f1dsfemale(i+6,2:end));
end

maxf1_dsfemale=mean([min_f1dsfemale(1),min_f1dsfemale(2) min_f1dsfemale(3)]);
maxstdf1_dsfemale=mean([std_f1dsfemale(1),std_f1dsfemale(2) std_f1dsfemale(3)]);



for i=1:3
min_f1pagegt50(i)=mean(f1pagegt50(i+6,2:end));
std_f1pagegt50(i)=std(f1pagegt50(i+6,2:end));
end




maxf1_pagegt50=mean([min_f1pagegt50(1),min_f1pagegt50(2) min_f1pagegt50(3)]);
maxstdf1_pagegt50=mean([std_f1pagegt50(1),std_f1pagegt50(2) std_f1pagegt50(3)]);


xf2=[maxf1_sfemale,maxf1_dsfemale,maxf1_pagegt50]';
yf2=[maxstdf1_sfemale,maxstdf1_dsfemale,maxstdf1_pagegt50]';





Table_3_output365_days=[dmeansmale dstdsmale maxf1_smale mean(std_f1smale);...
    dmeansfemale dstdsfemale maxf1_sfemale mean(std_f1sfemale); ...
    dmeandsmale dstdsmale maxf1_dsmale mean(std_f1dsmale); ...
    dmeandsfemale dstdsfemale maxf1_dsfemale mean(std_f1dsfemale); ...
    dmeanpagel50 dstdpagel50 maxf1_pagel50 mean(std_f1pagel50); ...
    dmeanpagegt50 dstdpagegt50 maxf1_pagegt50 mean(std_f1pagegt50)];


























