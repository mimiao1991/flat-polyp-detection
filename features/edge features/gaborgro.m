clc
clear all;
filename = ['/Users/miaofan/Desktop/blood vessel/negtidata/frame*.jpg'];
file = dir(filename);
for j = 1:1:114%:1:56%[1,136,139,141,142,145:1:length(file)]%20:1:length(file)    
    imagename = ['/Users/miaofan/Desktop/blood vessel/negtidata/',file(j).name];
    I = imread(imagename);
    I=rgb2gray(I);
    %I = rgb2lab(I);
    %[m,n] = size(I(:,:,3));
    %I = reshape(I(:,:,3),m,n)*255/100;
    %I = imadjust(I);
    %I=rgb2gray(I);
    %I = imadjust(I);
    %I = histeq(I);
    % I_adapthisteq = adapthisteq(I);
    % I = double(I);
    % [IX,IY]=gradient(I);
    % gm=sqrt(IX.*IX+IY.*IY);
    % out1=gm;
    % out2=I;
    % J=find(gm>=5);
    % out2(J)=gm(J);
    % out3=out2;
    % J=find(gm<=10);
    % out3(J)=255;
    % K=find(gm>20);
    % out3(K)=0;
    [m,n] = size(I);
    
    %   LAMBDA - preferred wavelength (period of the cosine factor) [in pixels]
    %   SIGMA - standard deviation of the Gaussian factor [in pixels]
    %   THETA - preferred orientation [in radians]
    %   PHI   - phase offset [in radians] of the cosine factor
    %   GAMMA - spatial aspect ratio (of the x- and y-axis of the Gaussian elipse)
    %   BANDWIDTH - spatial frequency bandwidth at half response,
    %for phi = linspace(-pi,pi,10)% I didn't saw any difference
    phi = 0;
    f_theta = zeros(m*n,37);
    count = 1;
    for theta = linspace(0,pi*2,37)
        maf = [0,0,0,0,0,0];
        maxf = zeros(m,n,6);
        for lambda = linspace(2,20,19)%max lambad
            %lambda = 8;
            gamma = 0.5;
            bandwidth = 2;
            result = gaborKernel2d( lambda, theta, phi, gamma, bandwidth);
            filtered = imfilter(I,result);
            f = abs(filtered);
            savef = f;
            [counts,binLocations] = imhist(f);
            stem(linspace(0,26,27),counts(1:27,:));
            mask_1 = find(f<8);%4
            %mask_2 = find(f>17);
            f(mask_1) = 0;
            %f(mask_2) = 0;
            %sq = round(n*m/172800+3);
            %SE = strel('square',sq);
            %f = imdilate(f,SE);
            %f = imdilate(f,SE);
            imshow(double(f));
            f(find(f>0)) = 1;
            mask = 1-f;
            savef = savef.*mask;
            mask_3 = find(savef<3);%3
            savef(mask_3) = 0;
            [counts,binLocations] = imhist(f);
            stem(linspace(0,26,27),counts(1:27,:));
            imshow(double(savef));
            x = find(maf == min(maf));
            if sum(sum(savef))>min(maf)
                maf(x(1)) = sum(sum(savef));
                maxf(:,:,x(1)) = savef;
            end
            %figure
            %imshow(f/max(f(:)));
        end
        %[counts,binLocations] = imhist(maxf);
        %stem(linspace(0,26,27),counts(1:27,:));
        %[m,n] = size(maxf);
        %         mask_1 = find(maxf<5);
        %         mask_2 = find(maxf>17);
        %         maxf(mask_1) = 0;
        %         maxf(mask_2) = 0;
        maxf = sqrt(sum(maxf.*maxf,3))/6;
        imshow(double(maxf));
        f_theta(:,count) = reshape(maxf,m*n,1);
        count = count+1;
    end
    f_theta = reshape(sqrt(sum((f_theta.*f_theta),2))/37,m,n);
    %figure
    imshow(double(f_theta));
    output = ['/Users/miaofan/Desktop/blood vessel/neg/'];
    mkdir(output);
    wname = ['/Users/miaofan/Desktop/blood vessel/neg/',file(j).name];
    imwrite(double(f_theta),wname);
end
%imlin = lininline(double(f_theta));

load chirp
sound(y,Fs)

