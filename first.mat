%% labeling simulation script
xmlfile='simple1';
xlsname='simple1_mea.xlsx';

%% create an EMU model
tic;
[model,memu]=modelinit(xmlfile,{'U'});
simulatefcn=str2func(xmlfile);
toc;

%% read constraints
free_net=zeros(size(model.kernel_net,2),1);
free_xch=zeros(size(model.kernel_xch,2),1);

[ineq,ineqStr]=xlsreadineq(xlsname,model,free_net,free_xch);
[eq,eqStr]=xlsreadeq(xlsname,model,free_net,free_xch);

%% randomize inital fluxes within constraints
n=10000; % the number of sets of initial fluxes for each input metabolite labeling
seed=0;
[free_net_set,free_xch_set]=fluxinit(model,ineq,eq,n,1,seed);
free_set=[free_net_set;free_xch_set]'; % OUTPUT for ML

%% simulate isotope labeling 
m1=3;                                                   % # of atoms in input metabolite 1; may need multiple inpute mets
for j1=2^m1-2:-1:1                                      % generate all nontrivial tracers
    alltracers(j1,:)=de2bi(j1,m1);
    input.molA__U=isotopomervector(alltracers(j1,:),1);
    clear(char(simulatefcn))                            % reset the input metabolite, which is persistent in fcn
%     input.molA__U=isotopomervector([1 1 0],1);
    for i=n:-1:1                                        % run simulation with input metabolite
        simulated(i)=simulatefcn(free_net_set(:,i),free_xch_set(:,i),input);
    end
    simulated_sets(j1)=mergestruct(simulated);          % INPUT for ML (one tracer)
end
[~,simulated_set,lmid]=mergestruct2(simulated_sets);    % INPUT for ML (multiple tracers in struct)
singlemat=struct2mat(simulated_set);                    % INPUT for ML (multiple tracers in a single matrix)
% [singlemat_nan,keepmet,keepexp]=scratchmat(singlemat,length(fieldnames(simulated))-1,lmid,seed,1);        % INPUT for imputation/inpainting (some tracers and some metabolites)

%% randomly erase isotope tracing experiments and metabolite measurements
nexp=1;                                                                                                     % # of isotope tracers used for experiment (realistic situation is 1 to a few)
emumets=fieldnames(simulated);
nmet=length(emumets)-1;
for seeds=20:-1:1
    [singlemat_nan{seeds},keepmet{seeds},keepexp{seeds},mask{seeds}]=scratchmat(singlemat,nmet,lmid,seeds,nexp,-1);     % INPUT for imputation/inpainting (some tracers and some metabolites); last argument '-1' replaces NaN
end

%% convert matrix to grayscale images and visualize
ind=1;
I=reshape(singlemat(ind,:),[],nmet)';
figure(1)
imshow(I,'InitialMagnification','fit')
Inan=reshape(singlemat_nan{1}(ind,:),[],nmet)';
figure(2)
imshow(Inan,'InitialMagnification','fit')

%% place measured data into correct positions in data matrix for imputation/inpainting and flux prediction
exp={'C1' 'C2'};
tracer=[0 1 0; 1 1 0];
[mat,vec,metind,expind]=placeinmat(tracer,mea,exp,alltracers,emumets(2:end),lmid,-1);                       % INPUT for the trained models (both imputation/inpainting and ANN)
Give feedback
