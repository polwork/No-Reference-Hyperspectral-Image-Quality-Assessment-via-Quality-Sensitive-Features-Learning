function feat_spe = spectralfeature(structdis)
data = structdis.data;
% % % [m,n,l] = size(data);
% % % window = fspecial('gaussian',7,7/6);
% % % window = window/sum(sum(window));
% % % 
% % % for num=1:size(data,3)
% % %     mu            = filter2(window, data(:,:,num), 'same');
% % %     mu_sq         = mu.*mu;
% % %     sigma         = sqrt(abs(filter2(window, data(:,:,num).*data(:,:,num), 'same') - mu_sq));
% % %     data(:,:,num)     = (data(:,:,num)-mu)./(sigma+1);
% % % end
% % % 
% % % I_vec=reshape(data,m*n,l);
% % % [T1, T2, T3]=svd(I_vec,'econ');
% % % T2_vec=diag(T2);
% % % x=1:l;
% % % x_log=log10(x);
% % % T2_vec_log=log10(T2_vec);
% % % 
% % % [xData, yData] = prepareCurveData( x_log, T2_vec_log );    % original data
% % % ft = fittype( 'a*exp(-b*x)+c', 'independent', 'x', 'dependent', 'y' );
% % % opts = fitoptions( 'Method', 'NonlinearLeastSquares');
% % % % opts.Algorithm = 'Levenberg-Marquardt';
% % % opts.Display = 'Off';
% % % %opts.StartPoint = [0.1 0.1 0.1];
% % % [fitresult, gof] = fit( xData, yData, ft, opts );
% % % a=fitresult.a;b=fitresult.b;c=fitresult.c;
% % % feat_spe=[a,b,c]';


% % % window = fspecial('gaussian',7,7/6);
% % % window = window/sum(sum(window));
% % % 
% % % for num=1:size(data,3)
% % %     mu            = filter2(window, data(:,:,num), 'same');
% % %     mu_sq         = mu.*mu;
% % %     sigma         = sqrt(abs(filter2(window, data(:,:,num).*data(:,:,num), 'same') - mu_sq));
% % %     data(:,:,num)     = (data(:,:,num)-mu)./(sigma+1);
% % % end
% % % 
% % % I_vec=reshape(data,m*n,l);
% % % [T1, T2, T3]=svd(I_vec,'econ');
% % % T2_vec=diag(T2);
% % % T2_vec_log=log10(T2_vec);
% % % feat_spe=T2_vec_log;



[m,n,l] = size(data);
window = fspecial('gaussian',[7 1],11/6);
window = window/sum(sum(window));
gen_spe=[];
for row=1:m
    for col=1:n
        spe=data(row,col,:);
        spe=spe(:);
        mu = imfilter(spe, window, 'replicate');
        mu_sq = mu.*mu;
        sigma = sqrt(abs(imfilter( spe.*spe, window, 'replicate') - mu_sq));
        spe = (spe-mu)./(sigma+0.02);
        gen_spe=[gen_spe,spe];
    end
end
gen_spe=gen_spe(:);
[alpha, betal, betar] = estimateaggdparam(gen_spe);
feat_spe=[alpha,betal,betar];
