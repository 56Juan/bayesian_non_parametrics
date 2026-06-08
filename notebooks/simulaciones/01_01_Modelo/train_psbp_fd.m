% ============================================================================
% train_psbp_fd.m  --  Entrenamiento PSBP-FD (UNA regresion, solo train)
% Adaptado de Case1_01.m:
%   * UNA sola regresion: tu construyes y pasas la data, ejecucion 1-a-1
%   * Solo data de entrenamiento: eliminado todo lo de test / fuera de muestra
%   * SIN estandarizacion: la data entra en su escala final (igual que PSBP_FD_v2)
%   * Salida = traces identicas a PSBP_FD_v2.traces (mismos nombres y formas)
% NOTA: se conserva inv() VERBATIM para mantener paridad con el sampler Python v2
%       (ambos comparten el mismo patron); cambiar a Cholesky exigiria hacerlo
%       en AMBOS a la vez para que sigan comparables.
% ============================================================================
clear;  

for tt=1:1
        
    % +-- EDITA AQUI ---------------------------------------------------------+
    IN_CSV  = 'dataset_fpc_1.csv';     % col 1 = respuesta, resto = predictores (con header)
    OUT_MAT = 'psbp_trace_fpc_1.mat';  % traza de salida (.mat v7)
    nsim = 2000;
    burn = 200;
    % +----------------------------------------------------------------------+

    % load dataset (SIN estandarizar; la data entra en su escala final)
    Tbl   = readtable(IN_CSV);
    names = Tbl.Properties.VariableNames;
    dt    = table2array(Tbl);
    n     = size(dt,1);

    y      = dt(:,1);
    Xnoint = dt(:,2:end);
    X      = horzcat(ones(n,1),Xnoint);
    p      = size(Xnoint,2);
    feature_names = strjoin(names(2:end), ",");   % CSV de nombres (robusto para scipy.io.loadmat)

    ncur = 1;
    nrun = nsim - 1;


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
            phxi=zeros(n,N);
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
                
            
            inE=zeros(n,N);
            for h=1:N
                inE(:,h)=phxi(:,h).*(X(:,1)*beta0h(h,1)+X(:,2:end)*betajh(h,:)');
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
            
            
        end

        % == Guardar traza: estructura IDENTICA a PSBP_FD_v2.traces (consumible en Python) ==
        save(OUT_MAT, '-v7', ...
            'betajhout','beta0hout','tauhout','alphahout','psijhout','Gammajhout',...
            'gammajhout','pijout','wjout','muout','osumout','N1out','Nout','inEout',...
            'nsim','burn','N','M','p','n','feature_names');
        disp(['Traza guardada en ', OUT_MAT, '  (', num2str(nsim), ' draws, burn=', num2str(burn), ')']);

end
