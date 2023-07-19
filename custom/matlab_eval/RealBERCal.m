function ber = RealBERCal(V, channel, c)
%%% This function takes channel matrix, V matrix and network
%%% configurationsd and returns the bit-error-rate of each user.
%%% The ber has dimenssion of number_of_user x umber_ofS_samples.
chanBW          = c.chanBW;                     % Channel bandwidth
numUsers        = c.numUsers;                   % Number of active users
numSTSAll       = c.numSTSAll;                  % Number of streams for 4 users
mcsVec          = c.mscVec;                     % MCS for maximum 4 users
apepVec         = c.apepVec;                    % Payload, in bytes, for 4 users
chCodingVec     = c.chCodingVec;                % Channel coding for 4 users
precodingType   = c.precodingType;
numTx           = c.numTx;
snr             = c.snr;
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
ber = inf(numUsers, numSamples);
for s = 1:numSamples
    X_bits = [];
    X = []; %% Generating each users' payload
    for uIdx = 1:numUsers
        x_bits{uIdx} = randi([0 1], numSc*mod_rate, numSTSVec(uIdx));
        x_mod = qammod(x_bits{uIdx},2^mod_rate,'InputType', 'bit','UnitAveragePower',true);
        X_bits = [X_bits, x_bits{uIdx}];
        X = [X;x_mod.'];
    end
    matV = cell(numUsers,1);
    ChanEst = cell(numUsers,1);
    H_mat = []; H_eq = [];
    for i = 1:numUsers
        csi = csi_vec(:,:,:,i,s).';
        ChanEst{i} = reshape(csi, [numSTSVec(i), size(csi)]) ;
        v = reshape(fb_vec(:,:,:,i,s).',[numTx, numSTSAll(i), numSc]);
        seed = randi(1000);
        v = awgn(v,10 * snr,'measured',seed);
        matV{i} = awgn(v,2 * snr,'measured',seed);
        H_mat = [H_mat;ChanEst{i}];
        H_eq = [H_eq, matV{i}];
    end

    W = complex(zeros(size(H_eq)));
    for sc = 1:length(H_eq)
        H_sc = squeeze(H_eq(:,:,sc));
        W(:,:,sc) = H_sc/((H_sc' * H_sc));
    end

    Y = complex(zeros(sum(numSTSVec),1,numSc));
    y = zeros(numSc*mod_rate,numUsers,numSc);
    y_bits = [];

    p = sqrt(db2pow(snr)/numTx);
    seed = randi(1000);
    for sc = 1:numSc
        X_h = W(:,:,sc) * X(:,sc);
        Y(:,:,sc) = awgn(p * H_mat(:,:,sc) * X_h, snr,'measured',seed); %% Eq(14.1)
        y_bits = [y_bits; reshape(qamdemod(Y(:,:,sc),4,'OutputType','bit'),[],numUsers)];
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

    X_hat = squeeze(G_n.* Y);

    for uIdx = 1:numUsers
        rx_bits = qamdemod(X_hat(uIdx,:),4,'OutputType','bit');
        [~, ber(uIdx,s)] = biterr(X_bits(:,uIdx), rx_bits(:));
%         disp(['Bit Error Rate for User ' num2str(uIdx) ': ' num2str(ber(uIdx))]);
    end
end
end
