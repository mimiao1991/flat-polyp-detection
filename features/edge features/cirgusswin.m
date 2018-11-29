function imagew = cirgusswin(image,x,y,r,rga)
ratio = r/40.0;
t = (0:0.1:5*ratio);
g = gaussmf(t,[0.8*ratio,0]);
[m,n] = size(image);
windo = zeros(m,n);
for i = 1:1:m
    for j = 1:1:n
        f = sqrt(((i-x)^2)/rga^2+((j-y)^2)/(r)^2);
        if f<=1
            if f<=7/8
                windo(i,j) = 1;
            else
                windo(i,j) = g(floor(5*10*ratio*f-r/rga)+1);
            end
        end
    end
end
imagew = double(image).*windo;
end
                
            
            
                
