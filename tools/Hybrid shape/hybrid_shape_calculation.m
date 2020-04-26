close all;
x = linspace(0,2.2,1001);
y = sqrt(max(0,1-x.^2)) + 0 * (x>=1) .* (-sqrt(max(0,1-(x-1).^2))+1);
% y = sqrt(max(0,1-x.^2)) + (x>1) .* (cos(x-1)+sec(x-1)-2);
plot(x,y)
hold on;

plotFLag = true;
% angles = linspace(3,45,8);
% angles = [20,40];
% angles = 30;
% angles = [15,30];
% angles = round(linspace(5,45,3));
% angles = [11,22,33]
% angles = [11,22,33,44];
angles = 3:0.5:45
angles = 45; 
% 
% angles = round(linspace(1,45,10));
% angles = [10,40];
angles = round(linspace(5,45,100));
equalDivided = false;

angles = round(linspace(5,45,500));


xarr = 1
yarr = 0;
counter = 0;
counterAll = 0;
y0 = 0;
marr =[];
for ang = angles
counter = counter + 1;
counterAll = counterAll + 1;
    
x_concave = cosd(ang);
y_concave = sind(ang);

P1 = [x_concave;y_concave];
vec = [-cosd(2*ang),sind(2*ang);-sind(2*ang),-cosd(2*ang)] * [0;1];
P2 = P1 + 10*vec;

m1 = (P2(2)-P1(2)) / (P2(1)-P1(1));
b1 = P2(2)-m1*P2(1);

m2 = tand(ang);
b2 = yarr(counter) - m2 * xarr(counter);

marr(counter+1) = m2;

xint = (b2-b1)/(m1-m2);
if xint < xarr(counter)
    counter = counter - 1;
    continue
end

xarr(counter+1) = xint + (xint-xarr(counter));
yarr(counter+1) = m2 * xarr(counter+1) + b2;

if equalDivided
    xarr(counter+1) = 1 + (2-1) / length(angles);
    yarr(counter+1) = m2 * xarr(counter+1) + b2;
end

P2 = [xint,m2 * xint + b2];
% if (counterAll == length(angles)) && 

% plot
if plotFLag
plot([x_concave,x_concave],[1,y_concave],'r');
% plot normal
plot([P1(1),1.1*P1(1)],[P1(2),1.1*P1(2)],'y--');
% plot reflected
plot(xint,0,'*g');
plot([P1(1),P2(1)],[P1(2),P2(2)],'r');
end

xlim([0,2.2]);
ylim([0,2]);

AngValidation(counterAll) = true;
end

xarr(end) = max(xarr(end),2);
yarr(end) = m2 * xarr(end) + b2;

xq = linspace(1,2.152,1e2);
yq = interp1(xarr,yarr,xq); 
plot(xq,yq);

plot(xarr , yarr,'k')
title(['N = ' num2str(counter)]);
% meshStr = [];
% currH = 0;
% for n=2:length(xarr)
%     meshStr = [meshStr 'min(' num2str(yarr(n)) ',max(' num2str(yarr(n-1)) ',' num2str(marr(n)) '*x-' num2str(-yarr(n)+marr(n)*xarr(n)) ')) + '];
%     currH = yarr(n);
% end
% meshStr(end-2:end) = [];

meshStr = 'z - (';
for n=2:length(xarr)
    func = [num2str(marr(n)) '.*x' num2str(-yarr(n)+marr(n)*xarr(n))];   
%     meshStr = [meshStr 'min(' num2str(yarr(n)) ',max(' num2str(yarr(n-1)) ',' num2str(marr(n)) '*x-1' num2str(-yarr(n)+marr(n)*xarr(n)) ')) + '];
    if n ~= length(xarr)
        meshStr = [meshStr '(x >= ' num2str(xarr(n-1)) ') .* (x < ' num2str(xarr(n)) ') .* (' num2str(marr(n)) '*x-' num2str(-yarr(n)+marr(n)*xarr(n)) ') + '];
    else
        meshStr = [meshStr '(x >= ' num2str(xarr(n-1)) ') .* (' num2str(marr(n)) '.*x-' num2str(-yarr(n)+marr(n)*xarr(n)) ')'];     
    end
end

meshStr = 'z - (';
for n=2:length(xarr)
    func = [num2str(marr(n)) '.*x' num2str(-yarr(n)+marr(n)*xarr(n))];   
%     meshStr = [meshStr 'min(' num2str(yarr(n)) ',max(' num2str(yarr(n-1)) ',' num2str(marr(n)) '*x-1' num2str(-yarr(n)+marr(n)*xarr(n)) ')) + '];
    if n ~= length(xarr)
        meshStr = [meshStr '(x >= ' num2str(xarr(n-1)) ') * (x < ' num2str(xarr(n)) ') * (' num2str(marr(n)) '*x-' num2str(-yarr(n)+marr(n)*xarr(n)) ') + '];
    else
        meshStr = [meshStr '(x >= ' num2str(xarr(n-1)) ') * (' num2str(marr(n)) '*x-' num2str(-yarr(n)+marr(n)*xarr(n)) ')'];     
    end
end
% meshStr = [meshStr '(x >=' num2str(xarr(n)) ') * ' num2str(yarr(n))];
% % meshStr(end-2:end) = [];
meshStr = [meshStr ')'];

meshStr = 'z - (';

for n=2:length(xarr)
    func = [num2str(marr(n)) '.*abs(x)' num2str(-yarr(n)+marr(n)*xarr(n))];   
%     meshStr = [meshStr 'min(' num2str(yarr(n)) ',max(' num2str(yarr(n-1)) ',' num2str(marr(n)) '*x-1' num2str(-yarr(n)+marr(n)*xarr(n)) ')) + '];
    if n ~= length(xarr)
        meshStr = [meshStr '(abs(x) >= ' num2str(xarr(n-1)) ') * (abs(x) < ' num2str(xarr(n)) ') * (' num2str(marr(n)) '*abs(x)-' num2str(-yarr(n)+marr(n)*xarr(n)) ') + '];
    else
        meshStr = [meshStr '(abs(x) >= ' num2str(xarr(n-1)) ') * (' num2str(marr(n)) '*abs(x)-' num2str(-yarr(n)+marr(n)*xarr(n)) ')'];     
    end
end
% meshStr = [meshStr '(x >=' num2str(xarr(n)) ') * ' num2str(yarr(n))];
% % meshStr(end-2:end) = [];
meshStr = [meshStr ')'];

meshStr

meshStr = strrep(meshStr,'abs(x)','sqrt(x^2+y^2)')% x = linspace(1,2,100);
% plot(x,eval(meshStr(5:end)));

% imshow(renderedImages,[]); colorbar
% plot(renderedImages(128,:))


