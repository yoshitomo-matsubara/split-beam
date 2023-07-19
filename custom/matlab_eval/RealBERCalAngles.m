function [ber_og, ber_ang] = RealBERCalAngles(V, channel, c)
%%% RealBERAngles calculates the BER using (i) standard SVD decomposition,
%%% and (ii) using Given's rotation for quantizing and compressing the
%%% the V matrix into psi and phi angles. Further, the LB-SciPhi paper
%%% approximatiuon error e_LB is added to consider their method. This function
%%% takes channel matrix, V matrix and network
%%% configurationsd and returns the bit-error-rate of each user.
%%% The output has two parameters (i) ber_og which coresponds to the
%%% original BER and (ii) ber_ang which corresponds to the second method.
%%% Each output parameter has dimenssion of number_of_user x umber_of_samples.

chanBW          = c.chanBW;                     % Channel bandwidth
numUsers        = c.numUsers;                   % Number of active users
numSTSAll       = c.numSTSAll;                  % Number of streams for 4 users
mcsVec          = c.mscVec;                     % MCS for maximum 4 users
apepVec         = c.apepVec;                    % Payload, in bytes, for 4 users
chCodingVec     = c.chCodingVec;                % Channel coding for 4 users
precodingType   = c.precodingType;
numTx           = c.numTx;
snr             = c.snr;
e_LB            = 0.025;                         % LB_SciPhi error percentage
phi_bits        = 4;                            % number of bits for phi angel
psi_bits        = 2;                            % number of bits for psi angel
numSamples      = size(V,1);
numSTSVec       = numSTSAll(1:numUsers);
mod_rate        = 2;


if chanBW == 80e6
    index = [7:128,132:251];
elseif chanBW == 20e6
    index = [5:32,34:61];
elseif chanBW == 40e6
    index = [7:63, 67:123];
end

numSc = length(index);

csi_vec     = permute(channel,[2,3,4,5,1]);
fb_vec      = permute(V,[2,3,4,5,1]);
% csi_vec = channel;
% fb_vec = V;
ber_og = inf(numUsers, numSamples); ber_ang = inf(numUsers, numSamples);
for s = 1:numSamples
    X_bits = [];
    X = []; %% Generating each users' payload
    for uIdx = 1:numUsers
        x_bits{uIdx} = randi([0 1], numSc*mod_rate, numSTSVec(uIdx));
        x_mod = qammod(x_bits{uIdx},2^mod_rate,'InputType', 'bit','UnitAveragePower',true);
        X_bits = [X_bits, x_bits{uIdx}];
        X = [X;x_mod.'];
    end
    matV = cell(numUsers,1); matV_ang = cell(numUsers,1);
    ChanEst = cell(numUsers,1);
    H_mat = []; H_eq = []; H_eq_ang = [];
    for i = 1:numUsers
        csi = csi_vec(:,:,:,i,s).';
        ChanEst{i} = reshape(csi, [numSTSVec(i), size(csi)]) ;
        v = reshape(fb_vec(:,:,:,i,s).',[numTx, numSTSAll(i), numSc]);
        seed = randi(1000);
        v = awgn(v,10 * snr,'measured',seed);
        matV{i} = awgn(v,2 * snr,'measured',seed);
        %%% Angle Calculation
        v_com = bfCompQuant(v,phi_bits,psi_bits);
        v_decom = awgn(v_com, 2.5* snr,'measured',seed);
        % v_decom = v_decom + ((-e_LB * v_decom) + (2 * e_LB * v_decom.* randn(size(v_decom)))); %%% add the LB-SciPhi error
        matV_ang{i} = bfDecom(v_com,numTx,numSTSAll(1),phi_bits,psi_bits);
        H_mat = [H_mat;ChanEst{i}];
        H_eq = [H_eq, matV{i}];
        H_eq_ang = [H_eq_ang, matV_ang{i}];
    end

    W = complex(zeros(size(H_eq)));
    W_ang = complex(zeros(size(H_eq_ang)));
    for sc = 1:length(H_eq)
        H_sc = squeeze(H_eq(:,:,sc));
        W(:,:,sc) = H_sc/((H_sc' * H_sc));
    end
    for sc = 1:length(H_eq_ang)
        H_sc_ang = squeeze(H_eq_ang(:,:,sc));
        if det(H_sc_ang) < 10e-8
            W_ang(:,:,sc) = H_sc_ang * pinv((H_sc_ang' * H_sc_ang));
        else
            W_ang(:,:,sc) = H_sc_ang/((H_sc_ang' * H_sc_ang));
        end
    end

    Y = complex(zeros(sum(numSTSVec),1,numSc));
    y = zeros(numSc*mod_rate,numUsers,numSc);
    y_bits = [];

    Y_ang = complex(zeros(sum(numSTSVec),1,numSc));
    y_ang = zeros(numSc*mod_rate,numUsers,numSc);
    y_bits_ang = [];
    p = sqrt(db2pow(snr)/numTx);
    seed = randi(1000);
    for sc = 1:numSc
        X_h = W(:,:,sc) * X(:,sc);
        Y(:,:,sc) = awgn(p * H_mat(:,:,sc) * X_h, snr,'measured',seed); %% Eq(14.1)
        y_bits = [y_bits; reshape(qamdemod(Y(:,:,sc),4,'OutputType','bit'),[],numUsers)];
    end

    for sc = 1:numSc
        X_h_ang = W_ang(:,:,sc) * X(:,sc);
        Y_ang(:,:,sc) = awgn(p * H_mat(:,:,sc) * X_h_ang, snr,'measured',seed);
        y_bits_ang = [y_bits_ang; reshape(qamdemod(Y_ang(:,:,sc),4,'OutputType','bit'),[],numUsers)];
    end

    delta = db2pow(snr) * eye(sum(numSTSVec));
    G_n = complex(zeros(sum(numSTSVec),1,numSc));
    for n = 1:numUsers
        for sc = 1:numSc
            W_n = W(:,n,sc);
            H_n = H_mat(n,:,sc);
            G_n(n,:,sc) = p * W_n'* H_n' * (p^2 * H_n * W(:,:,sc) * W(:,:,sc)' * H_n' + db2pow(snr)).^-1;
        end
    end
    G_n_ang = complex(zeros(sum(numSTSVec),1,numSc));
    for n = 1:numUsers
        for sc = 1:numSc
            W_n_ang = W_ang(:,n,sc);
            H_n_ang = H_mat(n,:,sc);
            G_n_ang(n,:,sc) = p * W_n_ang'* H_n_ang' * (p^2 * H_n * W_ang(:,:,sc) * W_ang(:,:,sc)' * H_n' + db2pow(snr)).^-1;
        end
    end

    X_hat = squeeze(G_n.* Y);
    X_hat_ang = squeeze(G_n_ang.* Y_ang);

    for uIdx = 1:numUsers
        rx_bits = qamdemod(X_hat(uIdx,:),4,'OutputType','bit');
        rx_bits_ang = qamdemod(X_hat_ang(uIdx,:),4,'OutputType','bit');
        [~, ber_og(uIdx,s)] = biterr(X_bits(:,uIdx), rx_bits(:));
        [~, ber_ang(uIdx,s)] = biterr(X_bits(:,uIdx), rx_bits_ang(:));
%         disp(['Bit Error Rate for User ' num2str(uIdx) ': ' num2str(ber(uIdx))]);
    end
end
end
