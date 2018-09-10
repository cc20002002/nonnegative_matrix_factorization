red=3;
k=40;

imagefiles = dir('../data/ORL/*/*.pgm');
imagefiles2=struct2cell(imagefiles);
imagefiles=imagefiles((~endsWith(imagefiles2(1,:),'Ambient.pgm'))');
imagefiles2=struct2cell(imagefiles);
A=imread(strcat(imagefiles(1).folder,'\',imagefiles(1).name));
if size(A,1)==112
    A=A(1:111,1:90);
end
A_list=zeros(size(A,1)/red,size(A,2)/red,red);
nfiles = length(imagefiles);    % Number of files found
matrix_image=zeros(prod(size(A))/red^2,nfiles);
temp=struct2cell(imagefiles);
names=temp(1,:)
for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentfilename=strcat(imagefiles(ii).folder,'\',currentfilename);
   currentimage = imread(currentfilename);
   if abs(size(A,1)-112)<=1
    currentimage=currentimage(1:111,1:90);
   end
   
   for i=1:red
    A_list(:,:,i)=currentimage(i:red:end,i:red:end);
   end
   A2=uint8(mean(A_list,3));   
   matrix_image(:,ii) = A2(:);
end

[w h]=NeNMF(matrix_image,k);
V=matrix_image;
[m,n]=size(V);
%%%%%%%%%%%%%%%%%%%%%%%
[W H]=nnmf(V,40,'alg','mult','replicates',5);
norm(V-W*H,'fro')/norm(V,'fro')
noise=normrnd(0,10,size(V));
[W H]=nnmf(V+noise,40,'alg','mult','replicates',5);
norm(V-W*H,'fro')/norm(V,'fro')
V_noise=poissrnd(V);
[W H]=nnmf(V_noise,40,'alg','mult','replicates',5);
norm(V-W*H,'fro')/norm(V,'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%
[W H]=KLNMF(V,40,1000);
norm(V-W*H,'fro')/norm(V,'fro')
[W H]=KLNMF(V+noise,40,1000);
norm(V-W*H,'fro')/norm(V,'fro')
[W H]=KLNMF(V_noise,40,1000);
norm(V-W*H,'fro')/norm(V,'fro')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r=40;
W0=rand(m,r);
H0=rand(r,n);

temp=zeros(r,n);
WH=W0*H0;
for i=1:r
    for j=1:n
        temp(i,j)=sum(W0(:,i).*V(:,j)./WH(:,j),1);
    end
end
norm(temp-W0.'*(V./WH))


idx = kmeans(h',k)
Y_pred=zeros(size(matrix_image,2),1)
namess=str2mat(string(imagefiles2(2,:))')
namess=str2num(namess(:,end-1:end))
for ii=unique(idx)'
   ind= (idx==ii);
   Y_pred(ind)=mode(namess(ind,:));
end
sum(Y_pred==namess)/size(names,2)
nmi(Y_pred,namess)
