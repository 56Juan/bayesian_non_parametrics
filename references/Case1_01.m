clear;  

for tt=1:1
        
    basefname='BHP';
    tempfname1=[basefname, 'in_', num2str(tt),'.txt'];
    tempfname2=[basefname, 'out_', num2str(tt),'.txt'];
    
    %load dataset
    dt=load(tempfname1);
    dt2=load(tempfname2);
    n = size(dt,1);
    n2 = size(dt2, 1);
    dt1=dt;y2=dt2(:,1);
    
    %data standardization
    dtmean = mean(dt, 1);
    dtstd = std(dt, 1);
    dt=(dt-repmat(dtmean,n,1))./repmat(dtstd,n,1);
    dt2=(dt2-repmat(dtmean,n2,1))./repmat(dtstd,n2,1);
    
        y=dt(:,1);Xnoint=dt(:,2:end);
        y1=dt1(:,1);
        n=length(y);
        X=horzcat(ones(n,1),Xnoint);
        p=size(Xnoint,2);
        outX=horzcat(ones(n2,1),dt2(:,2:end));
        outy=dt2(:,1);
        %figure;subplot(4,1,1);plot(Xnoint(:,1),y,'k.');subplot(4,1,2);plot(Xnoint(:,2),y,'k.');
        %subplot(4,1,3);plot(Xnoint(:,3),y,'k.');subplot(4,1,4);plot(Xnoint(:,4),y,'k.');
        
        %---------- Define MCMC parameters
        nsim = 2000;  
        ncur = 1;     
        nrun = 2000-1;
        burn = 200;   

        if ncur==1
            
            N=20;M=50;
            atau=0.5;btau=0.5;
            ag=0.5;bg=0.5;
            apij=ones(p,1);bpij=5*ones(p,1);
            mu=1;mumu=0;taumu=1;
            g=1;
            taupsij=ones(p,1);
            mupsij=zeros(p,1);
            pij=0.5*ones(p,1);
            wj=ones(p,1);pwj=0.5;
            Gstar=(min(min(X)) + ((1:M)/M).*(max(max(X))-min(min(X))))';
            
        
            %------- initialize
            bjrange=[-4,-3,-2,-1.5,-1,0,1,1.5,2,3,4,5]';
            b0range=[-4,-3,-2,-1.5,-1,0,1,1.5,2,3,4,5]';
            betajh=bjrange(unidrnd(7,N,p));
            beta0h=b0range(unidrnd(7,N,1));
            tauh=gamrnd(atau,1/btau,N,1);
            %Si=ones(n,1);
            Si=unidrnd(N,n,1);
            alphah=zeros(N-1,1);
            %psijh=1/3*ones(N-1,p);
            psijh=zeros(N-1,p);
            gammajh=ones(N,p);
            Gloc=unidrnd(M,N-1,p);
            Gammajh=Gstar(Gloc);
            
           
            betajhout=zeros(nsim,N,p,'single');
            beta0hout=zeros(nsim,N,'single');
            tauhout=zeros(nsim,N,'single');
            alphahout=zeros(nsim,N-1,'single');
            Gammajhout=zeros(nsim,N-1,p,'single');
            psijhout=zeros(nsim,N-1,p,'single');
            gammajhout=zeros(nsim,N,p,'single');
            pijout=zeros(nsim,p,'single');
            wjout=zeros(nsim,p,'single');
            N1out=zeros(nsim,1,'single');
            Nout=zeros(nsim,1,'single');
            muout=zeros(nsim,1,'single');
            osumout=zeros(nsim,p,'single');
            inEout=zeros(nsim,n,'single');
            outEout=zeros(nsim,n2,'single');
        end
        

        for gt = ncur:(ncur+nrun)
            
            %Update Zil
            Zil=zeros(n,N);Wil=zeros(n,N);
            for i=1:n
                if Si(i)<N
                    for l=1:Si(i)
                        u=unifrnd(0,1,1,1);
                        m=alphah(l)-sum(psijh(l,:).*abs(Xnoint(i,:)-Gammajh(l,:)),2);
                        v=1;
                        if l<Si(i)
                            Zil(i,l)=m+sqrt(v)*norminv(u*normcdf((0-m)/sqrt(v),0,1),0,1);
                        elseif l==Si(i)
                            Zil(i,l)=m+sqrt(v)*norminv(u+(1-u)*normcdf((0-m)/sqrt(v),0,1),0,1);
                        end
                        Wil(i,l)=Zil(i,l)+sum(psijh(l,:).*abs(X(i,2:end)-Gammajh(l,:)));
                    end
                elseif Si(i)==N
                    for l=1:N-1
                        u=unifrnd(0,1,1,1);
                        m=alphah(l)-sum(psijh(l,:).*abs(Xnoint(i,:)-Gammajh(l,:)),2);
                        v=1;
                        Zil(i,l)=m+sqrt(v)*norminv(u*normcdf((0-m)/sqrt(v),0,1),0,1);
                        Wil(i,l)=Zil(i,l)+sum(psijh(l,:).*abs(X(i,2:end)-Gammajh(l,:)));
                    end
                end 
            end

            
            %Update Si
            phxi=zeros(n,N);phxiout=zeros(n2,N);
            for i=1:n
                vhx=ones(N-1,1);
                phx=ones(N,1);
                for h=1:N-1
                    vhx(h,1)=normcdf(alphah(h,1)-sum(psijh(h,:).*abs(X(i,2:end)-Gammajh(h,:))),0,1);
                    if h==1
                        phx(h)=vhx(h);
                    elseif h>1
                        phx(h)=vhx(h)*prod(1-vhx(1:h-1,1));
                    end
                end
                phx(N)=prod(1-vhx);phxi(i,:)=phx';
                
                phx1=exp(log(phx+realmin)+log(normpdf(y(i),X(i,1)*beta0h(:,1)+betajh(:,:)*X(i,2:end)',1./sqrt(tauh))+realmin));
                phx12=phx1/sum(phx1);
                Si(i)=randsample(N,1,true,phx12);
            end
            
            for i=1:n2
                vhx2=ones(N-1,1);
                phx2=ones(N,1);
                for h=1:N-1
                    vhx2(h,1)=normcdf(alphah(h,1)-sum(psijh(h,:).*abs(outX(i,2:end)-Gammajh(h,:))),0,1);
                    if h==1
                        phx2(h)=vhx2(h);
                    elseif h>1
                        phx2(h)=vhx2(h)*prod(1-vhx2(1:h-1,1));
                    end
                end
                
                phx2(N)=prod(1-vhx2);phxiout(i,:)=phx2';                                           
            end
                
            
            inE=zeros(n,N);outE=zeros(n2,N);
            for h=1:N
                inE(:,h)=phxi(:,h).*(X(:,1)*beta0h(h,1)+X(:,2:end)*betajh(h,:)');
                outE(:,h)=phxiout(:,h).*(outX(:,1)*beta0h(h,1)+outX(:,2:end)*betajh(h,:)');
            end
                
            
            %update betah
            betajh=zeros(N,p);
            for h=1:N
                Xh=X(:,horzcat(1,gammajh(h,:))==1);
                Sh=n/g*inv(Xh'*Xh)/tauh(h);
                Shhat=inv(inv(Sh)+tauh(h)*Xh(Si==h,:)'*Xh(Si==h,:));pgh=size(Shhat,1);
                for l=1:pgh-1
                    for k=l+1:pgh
                        Shhat(l,k)=Shhat(k,l);
                    end
                end
                muhhat=Shhat*(tauh(h)*Xh(Si==h,:)'*y(Si==h)+inv(Sh)*zeros(pgh,1));
                betahtemp=mvnrnd(muhhat,Shhat)';
                beta0h(h,1)=betahtemp(1,1);
                        
                count=2;
                for j=1:p
                    if gammajh(h,j)==1
                        betajh(h,j)=betahtemp(count,1);
                        count=count+1;
                    end
                end
            end
            
            %Update tauh
            for h=1:N
                betagh=horzcat(beta0h(h,1),betajh(h,gammajh(h,:)==1));
                Xh=X(:,horzcat(1,gammajh(h,1:p))==1);
                aa=atau+0.5*size(Si(Si==h),1)+0.5*sum(gammajh(h,1:p))+0.5;
                bb=btau+0.5*(y(Si==h,1)-Xh(Si==h,:)*betagh')'*(y(Si==h,1)-Xh(Si==h,:)*betagh')+0.5/n*g*betagh*Xh'*Xh*betagh';
                tauh(h)=gamrnd(aa,1/bb);
            end
            
            %Update g
            aghat=ag+0.5*(sum(sum(gammajh(:,1:p),1))+N);
            temp=0;
            for h=1:N
                betagh=horzcat(beta0h(h,1),betajh(h,gammajh(h,:)==1));
                Xh=X(:,horzcat(1,gammajh(h,1:p))==1);
                temp=temp+tauh(h)*betagh*Xh'*Xh*betagh';
            end 
            bghat=bg+0.5/n*temp;
            g=gamrnd(aghat, 1/bghat);
            
            %Update w_j
            for j=1:p
                if sum(gammajh(:,j))>0
                    wj(j,1)=1;
                elseif sum(gammajh(:,j))==0
                    b=exp(gammaln(bpij(j,1)+N)+gammaln(apij(j,1)+bpij(j,1))-gammaln(bpij(j,1))-gammaln(apij(j,1)+bpij(j,1)+N));
                    pwjhat=pwj*b/((1-pwj)*1+pwj*b);
                    wj(j,1)=binornd(1,pwjhat,1,1);
                end
            end

            
            %Update alphah
            for h=1:N-1
                v=inv(1+size(Si(ge(Si,h)==1,:),1));
                m=v*(mu+sum(Wil(ge(Si,h)==1,h)));
                alphah(h)=normrnd(m,sqrt(v),1,1);
            end
            
            %Update mu
            taumuhat=N+taumu;
            mumuhat=1/taumuhat*(taumu*mumu+sum(alphah));
            mu=normrnd(mumuhat, sqrt(1/taumuhat));
                                    
            %Update pij
            for j=1:p
                if wj(j,1)==0
                    pij(j,1)=0;
                elseif wj(j,1)==1
                    pij(j)=betarnd(apij(j,1)+sum(gammajh(:,j)),bpij(j,1)+N-sum(gammajh(:,j)));
                end
            end
                        
            %Update Gammajh
            for h=1:N-1
                for j=1:p
                    if gammajh(h,j)==1
                        pm=zeros(M,1);
                        kh=size(Si(ge(Si,h)==1,1),1);
                        for m=1:M
                            pm(m)=exp(1.2*kh+sum(log(normpdf(Zil(ge(Si,h)==1,h),...
                                alphah(h)...
                                -sum(repmat(psijh(h,1:j-1),kh,1).*abs(Xnoint(ge(Si,h)==1,1:j-1)-repmat(Gammajh(h,1:j-1),kh,1)),2)...
                                -sum(repmat(psijh(h,j+1:end),kh,1).*abs(Xnoint(ge(Si,h)==1,j+1:end)-repmat(Gammajh(h,j+1:end),kh,1)),2)...
                                -repmat(psijh(h,j),kh,1).*abs(Xnoint(ge(Si,h)==1,j)-repmat(Gstar(m,1),kh,1))...
                                ,1)+realmin)))+realmin;
                        end
                        pm1=pm/sum(pm);
                        Gammajh(h,j)=Gstar(randsample(M,1,true,pm1),1);
                    end
                end
            end
            
            %Update psijh 
            for h=1:N-1
                for j=1:p
                    if gammajh(h,j)==0
                        psijh(h,j)=0;
                    elseif gammajh(h,j)==1
                        kh=size(Si(ge(Si,h)==1,1),1);
                        Tijh=alphah(h)-Zil(ge(Si,h)==1,h)...
                            -sum(repmat(psijh(h,1:j-1),kh,1).*abs(Xnoint(ge(Si,h)==1,1:j-1)-repmat(Gammajh(h,1:j-1),kh,1)),2)...
                            -sum(repmat(psijh(h,j+1:end),kh,1).*abs(Xnoint(ge(Si,h)==1,j+1:end)-repmat(Gammajh(h,j+1:end),kh,1)),2);
                        v=inv(taupsij(j,1)+(Xnoint(ge(Si,h)==1,j)-Gammajh(h,j))'*(Xnoint(ge(Si,h)==1,j)-Gammajh(h,j)));
                        m=v*(taupsij(j,1)*mupsij(j,1)+sum(Tijh.*abs(Xnoint(ge(Si,h)==1,j)-Gammajh(h,j))));
                        %psijh(h,j)=randraw('normaltrunc',[0, inf, m, sqrt(v)],1);
                        u=unifrnd(0,1,1,1);
                        psijh(h,j)=m+sqrt(v)*norminv(u+(1-u)*normcdf((0-m)/sqrt(v),0,1),0,1);
                        if psijh(h,j)==inf
                            psijh(h,j)=0.01;
                        end
                    end
                end
            end
            
            %Update gammajh
            for h=1:N
                for j=1:p
                    if h<N
                        kh=size(Si(ge(Si,h)==1,1),1);
                        Tijh=alphah(h)-Zil(ge(Si,h)==1,h)...
                            -sum(repmat(psijh(h,1:j-1),kh,1).*abs(Xnoint(ge(Si,h)==1,1:j-1)-repmat(Gammajh(h,1:j-1),kh,1)),2)...
                            -sum(repmat(psijh(h,j+1:end),kh,1).*abs(Xnoint(ge(Si,h)==1,j+1:end)-repmat(Gammajh(h,j+1:end),kh,1)),2);
                        v=inv(taupsij(j,1)+(Xnoint(ge(Si,h)==1,j)-Gammajh(h,j))'*(Xnoint(ge(Si,h)==1,j)-Gammajh(h,j)));
                        m=v*(taupsij(j,1)*mupsij(j,1)+sum(Tijh.*abs(Xnoint(ge(Si,h)==1,j)-Gammajh(h,j))));
                    
                        
                        ystar=y(Si==h)-X(Si==h,1)*beta0h(h)-Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)'-Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)';
                        
                        gammajh1=gammajh(h,1:p);gammajh1(:,j)=[];
                        Xh=X;Xh(:,j+1)=[];
                        Xh1=horzcat(Xnoint(:,j),Xh(:,horzcat(1,gammajh1)==1));
                        sb=n/g*inv(Xh1'*Xh1)/tauh(h);
                        betagh=horzcat(beta0h(h,1),betajh(h,:));betagh(:,j+1)=[];betagh2=betagh(:,horzcat(1,gammajh1)==1);
                            
                        sbj=sb(1,1)-sb(1,2:end)*inv(sb(2:end,2:end))*sb(1,2:end)';taubj=1/sbj;
                        mubj=sb(1,2:end)*inv(sb(2:end,2:end))*betagh2';
                        
                        bjhin=log(1-pij(j,1)+realmin)+sum(log(normpdf(y(Si==h,:),X(Si==h,1)*beta0h(h)...
                            +Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)'+Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)'...
                            ,1/sqrt(tauh(h)))+realmin))...
                            +sum(log(normpdf(Zil(ge(Si,h)==1,h),...
                            alphah(h)-sum(repmat(psijh(h,1:j-1),kh,1).*abs(Xnoint(ge(Si,h)==1,1:j-1)-repmat(Gammajh(h,1:j-1),kh,1)),2)...
                            -sum(repmat(psijh(h,j+1:end),kh,1).*abs(Xnoint(ge(Si,h)==1,j+1:end)-repmat(Gammajh(h,j+1:end),kh,1)),2),1)+realmin));
                        
                        ajhin=log(pij(j,1)+realmin)+sum(log(normpdf(ystar,0,1/sqrt(tauh(h)))+realmin))+log(normpdf(0,mubj,1/sqrt(taubj))+realmin)...
                            -log(normpdf(0,inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj)*(tauh(h)*Xnoint(Si==h,j)'*ystar+taubj*mubj),...
                            sqrt(inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj)))+realmin)...
                            +sum(log(normpdf(0,Tijh,1)+realmin))...
                            +log(normpdf(0,mupsij(j,1),1/sqrt(taupsij(j,1)))+realmin)-log(1-normcdf((0-mupsij(j,1))*sqrt(taupsij(j,1)),0,1)+realmin)...
                            -log(normpdf(0,m,sqrt(v))+realmin)+log(1-normcdf((0-m)/sqrt(v),0,1)+realmin);
                        
                        gammajh(h,j)=binornd(1,1/(1+exp(bjhin-ajhin)),1,1);
                        
                    elseif h==N
                        gammajh1=gammajh(h,1:p);gammajh1(:,j)=[];
                        Xh=X;Xh(:,j+1)=[];
                        Xh1=horzcat(Xnoint(:,j),Xh(:,horzcat(1,gammajh1)==1));
                        sb=n/g*inv(Xh1'*Xh1)/tauh(h);
                        betagh=horzcat(beta0h(h,1),betajh(h,:));betagh(:,j+1)=[];betagh2=betagh(:,horzcat(1,gammajh1)==1);
                            
                        sbj=sb(1,1)-sb(1,2:end)*inv(sb(2:end,2:end))*sb(1,2:end)';taubj=1/sbj;
                        mubj=sb(1,2:end)*inv(sb(2:end,2:end))*betagh2';
                        
                        nh=size(Si(Si==h,:),1);
                        ystar=y(Si==h)-X(Si==h,1)*beta0h(h)-Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)'-Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)';
                        
                        bjh=exp(1.2*nh+log(1-pij(j,1)+realmin)...
                           +sum(log(normpdf(y(Si==h,:),...
                           X(Si==h,1)*beta0h(h)+Xnoint(Si==h,1:j-1)*betajh(h,1:j-1)'+Xnoint(Si==h,j+1:end)*betajh(h,j+1:end)',...
                           1/sqrt(tauh(h)))+realmin)))+realmin;
                            
                        ajh=exp(1.2*nh+log(pij(j)+realmin)+sum(log(normpdf(ystar,0,1/sqrt(tauh(h)))+realmin))+log(normpdf(0,mubj,1/sqrt(taubj))+realmin)...
                            -log(normpdf(0,inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj)*(tauh(h)*Xnoint(Si==h,j)'*ystar+taubj*mubj),...
                            sqrt(inv(tauh(h)*Xnoint(Si==h,j)'*Xnoint(Si==h,j)+taubj)))+realmin));
                        
                        gammajh(h,j)=binornd(1,ajh/(ajh+bjh),1,1);
                    end 
                    
                end
                
            end
            
            osum=(sum((gammajh(1:max(Si),:)==1),1)==0);
            osumout(gt,:)=osum;
            
            gt
            %gammajh
            muout(gt,1)=mu;
            tauhout(gt,:)=tauh';
            beta0hout(gt,:)=beta0h';
            betajhout(gt,:,:)=betajh;
            alphahout(gt,:)=alphah';
            psijhout(gt,:,:)=psijh;
            Gammajhout(gt,:,:)=Gammajh;
            gammajhout(gt,:,:)=gammajh;
            pijout(gt,:)=pij';  
            wjout(gt,:)=wj';
            N1out(gt,1)=max(Si);
            Nout(gt,1)=N;
            inEout(gt,:)=sum(inE,2);
            outEout(gt,:)=sum(outE,2);
            
            
        end
        
        
        gloprob=mean(osumout(burn+1:nsim,:),1);
%        inRMSE=sqrt(mean((mean(inEout(burn+1:nsim,:),1)-(zeros(1,n)-mean(y1))/std(y1)).^2));
%        outRMSE=sqrt(mean((mean(outEout(burn+1:nsim,:),1)-(zeros(1,n)-mean(y2))/std(y2)).^2));
        inPred = mean(y1) + std(y1) * mean(inEout(burn+1:nsim,:),1)';
        outPred = mean(y1) + std(y1) * mean(outEout(burn+1:nsim,:),1)';
        inRMSE=sqrt(mean((inPred - y1).^2));
        outRMSE=sqrt(mean((outPred - y2).^2));
        

        savename=[basefname, '_',num2str(tt),'.mat'];
        
        
        save (savename, 'nsim', 'burn', 'osumout', 'muout', 'tauhout', 'beta0hout', 'betajhout', 'alphahout', 'psijhout', 'Gammajhout', 'gammajhout',...
            'pijout', 'wjout', 'N1out', 'Nout', 'inEout', 'outEout', 'outX', 'y', 'y1', 'y2', 'X', 'outy', 'Xnoint', 'p', 'n', 'gloprob', 'outRMSE',...
            'outPred', 'inRMSE', 'inPred');
        tt
         
end
        
        
        %figure;plot(X(:,2),y,'k.');hold on;plot(X(:,2),mean(inEout(burn+1:nsim,:),1),'bx');plot(X(:,2),zeros(n,1),'ro');
        %figure;plot(outX(:,2),outy,'k.');hold on;plot(outX(:,2),mean(outEout(burn+1:nsim,:),1),'bx');plot(outX(:,2),zeros(n,1),'ro');
        
            
        
            %local hypothesis testing
            %x1g=(-2:0.1:2)';nx1g=size(x1g,1);
            %x2g=(-2:0.1:2)';nx2g=size(x2g,1);
            %yg=(min(y):0.1:max(y))';nyg=size(yg,1);ep=0.05;
            %loc1=zeros(nsim-burn,nx1g);loc2=zeros(nsim-burn,nx2g);
            
            %for t=burn+1:nsim
            %    fyx1=zeros(nx1g,nyg);fyx2=zeros(nx2g,nyg);
            %    for i=1:nx1g
            %            N=Nout(t);
            %            alphah=alphahout(t,1:N-1)';
            %            psijh=reshape(psijhout(t,1:N-1,:),N-1,p);
            %            Gammajh=reshape(Gammajhout(t,1:N-1,:),N-1,p);
            %            betah=horzcat(reshape(beta0hout(t,1:N),N,1),reshape(betajhout(t,1:N,:),N,p));
            %            tauh=tauhout(t,1:N)';
            %            x1gstar=horzcat(x1g(i,1),prctile(Xnoint(:,2:end),50,1));
            %            x2gstar=horzcat(prctile(Xnoint(:,1),50),x2g(i,1),prctile(Xnoint(:,3:end),50,1));
            %        
            %            vhx1=ones(N,1);vhx2=ones(N,1);
            %            phx1=ones(N,1);phx2=ones(N,1);
            %            for h=1:N
            %               if h<N
            %                    vhx1(h,1)=normcdf(alphah(h,1)-sum(psijh(h,:).*abs(x1gstar(1,:)-Gammajh(h,:)),2),0,1);
            %                   vhx2(h,1)=normcdf(alphah(h,1)-sum(psijh(h,:).*abs(x2gstar(1,:)-Gammajh(h,:)),2),0,1);
            %                elseif h==N
            %                   vhx1(h,1)=1;
            %                  vhx2(h,1)=1;
            %              end
            %                 
            %                if h==1
            %                    phx1(h)=vhx1(h);phx2(h)=vhx2(h);
            %                elseif h>1
            %                    phx1(h)=vhx1(h)*prod(1-vhx1(1:h-1,1));
            %                    phx2(h)=vhx2(h)*prod(1-vhx2(1:h-1,1));
            %                end
            %            end
            %            phx11=phx1/sum(phx1);phx21=phx2/sum(phx2);
            %            for k=1:nyg
            %                fyx1(i,k)=sum(phx11.*(normpdf(yg(k),betah(:,1)+betah(:,2:p+1)*x1gstar(1,1:p)',1./sqrt(tauh))+realmin));
            %                fyx2(i,k)=sum(phx21.*(normpdf(yg(k),betah(:,1)+betah(:,2:p+1)*x2gstar(1,1:p)',1./sqrt(tauh))+realmin));
            %            end
            %    end
            %    for i=1:nx1g
            %        loc1(t-burn,i)=(max(fyx1(end,:)-fyx1(i,:))<ep);
            %        loc2(t-burn,i)=(max(fyx2(end,:)-fyx2(i,:))<ep);
            %    end
            %    t
            %end
                        
            
            %figure;
            %subplot(3,1,2);plot(x1g,mean(loc1,1),'k.');
            %title('P(H_{01}(max(x_1),x_1)|Data)');xlabel('x_1');axis([-2 2 -0.1 1.1]);
            %hold on;plot(0,(-0.1:0.1:1.1)','k','Linewidth',5);
            %subplot(3,1,3);plot(x2g,mean(loc2,1),'k.');axis([-2 2 -0.1 1.1]);
            %title('P(H_{02}(max(x_2),x_2)|Data)');xlabel('x_2');axis([-2 2 -0.1 1.1]);
            
            
            
            %density estimation
            %x1star=prctile(Xnoint(:,1), [10 50 90],1);nx1star=size(x1star,1);
            %x2star=prctile(Xnoint(:,2), [10 50 90],1);nx2star=size(x2star,1);
            %yg=(min(y):0.1:max(y))';nyg=size(yg,1);
            %fyxout=zeros(nsim-burn,nx1star,nx2star,nyg);fyxreal=zeros(nx1star,nx2star,nyg);
            %for t=burn+1:nsim
            %    for i=1:nx1star
            %        for j=1:nx2star
            %            N=Nout(t);
            %            alphah=alphahout(t,1:N-1)';
            %            psijh=reshape(psijhout(t,1:N-1,:),N-1,p);
            %            Gammajh=reshape(Gammajhout(t,1:N-1,:),N-1,p);
            %            betah=horzcat(reshape(beta0hout(t,1:N),N,1),reshape(betajhout(t,1:N,:),N,p));
            %            tauh=tauhout(t,1:N)';
            %            xstar=horzcat(x1star(i,1),x2star(j,1),prctile(Xnoint(:,3:end),50,1));
            %       
            %            vhx=ones(N,1);
            %            phx=ones(N,1);
            %            for h=1:N
            %                if h<N
            %                    vhx(h,1)=normcdf(alphah(h,1)-sum(psijh(h,:).*abs(xstar(1,:)-Gammajh(h,:)),2),0,1);
            %                elseif h==N
            %                    vhx(h,1)=1;
            %                end
            %                    
            %                if h==1
            %                    phx(h)=vhx(h);
            %                elseif h>1
            %                    phx(h)=vhx(h)*prod(1-vhx(1:h-1,1));
            %                end
            %            end
            %            phx1=phx/sum(phx);
            %            for k=1:nyg
            %                fyxout(t-burn,i,j,k)=sum(phx1.*(normpdf(yg(k),betah(:,1)+betah(:,2:p+1)*xstar(1,1:p)',1./sqrt(tauh))+realmin));
            %                %fyxreal(i,j,k)=normpdf(yg(k),(0-mean(y1))/std(y1),1/std(y1));
            %                fyxreal(i,j,k)=0.5*normpdf(yg(k),(-1-mean(y1))/std(y1),0.5/std(y1))+0.5*normpdf(yg(k),(1-mean(y1))/std(y1),1/std(y1));
            %                %fyxreal(i,j,k)=0.5*normpdf(yg(k),(10-mean(y1))/std(y1),(2+4*exp(-min(0,xstar(1,1))))/std(y1))...
            %                %    +0.5*normpdf(yg(k),(-10+5*xstar(1,2)-mean(y1))/std(y1),5/std(y1));
            %                %fyxreal(i,j,k)=normpdf(yg(k),(2*xstar(1,1)-3*xstar(1,2)+xstar(1,4)-xstar(1,5)-mean(y1))/std(y1),1/std(y1));
            %                %fyxreal(i,j)=normpdf(yg(j),(max(xstar(i,1),0)-mean(y1))/std(y1),1/std(y1));
            %            end
            %        end
            %    end
            %    t
            %end
                        
            %figure;count=0;
            %for i=1:nx1star
            %    for j=1:nx2star
            %        count=count+1;
            %        subplot(nx1star,nx2star,count);
            %        fyxmean=mean(reshape(fyxout(:,i,j,:),nsim-burn,nyg),1);
            %        fyxlo=prctile(reshape(fyxout(:,i,j,:),nsim-burn,nyg),5);
            %        fyxup=prctile(reshape(fyxout(:,i,j,:),nsim-burn,nyg),95);
            %        fyxr=reshape(fyxreal(i,j,:),nyg,1);
            %        plot(yg,fyxmean,'k--','Linewidth',1.5);axis([-4 4 0 2]);
            %        hold on;plot(yg,fyxr,'r-','Linewidth',1.5);
            %        plot(yg,fyxlo,'k-.','Linewidth',1);
            %        plot(yg,fyxup,'k-.','Linewidth',1);hold off;
            %    end
            %end
            
            
                
            %subplot(4,1,1);plot(psijhout(:,1,1),'k.');
            %subplot(4,1,2);plot(psijhout(:,2,1),'k.');
            %subplot(4,1,3);plot(psijhout(:,3,1),'k.');
            %subplot(4,1,4);plot(psijhout(:,4,1),'k.');
            
            %figure;
            %subplot(4,1,1);plot(gammajhout(:,1,1),'k.');
            %subplot(4,1,2);plot(gammajhout(:,2,1),'k.');
            %subplot(4,1,3);plot(gammajhout(:,3,1),'k.');
            %subplot(4,1,4);plot(gammajhout(:,4,1),'k.');
            %figure;
            %subplot(4,1,1);plot(alphahout(:,1),'k.');
            %subplot(4,1,2);plot(alphahout(:,2),'k.');
            %subplot(4,1,3);plot(alphahout(:,3),'k.');
            %subplot(4,1,4);plot(alphahout(:,4),'k.');
            
        

        
