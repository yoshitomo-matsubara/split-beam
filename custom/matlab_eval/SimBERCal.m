
function BER = SimBERCal(matV,chSeeds,c)
%BERCAL calculates the bit-error-rate (BER) of a receieved payload.
%   The output BER is an array of size number-of-users by number-of-samples

chanBW      = c.chanBW;
numUsers    = c.numUsers;                         % Number of active users
numSTSAll   = c.numSTSAll;
userPos     = c.userPos;                 % User positions for maximum 4 users
mcsVec      = c.mcsVec;                 % MCS for maximum 4 users
apepVec     = c.apepVec;    % Payload, in bytes, for 4 users
chCodingVec = c.chCodingVec;  % Channel coding for 4 users
numTx = c.numTx;


NumTrial = size(matV,5);
BER = zeros(numUsers,NumTrial);
for t = 1:NumTrial
    
    
    V = cell(c.numUsers,1);
%     V{1} = squeeze(V1(t,:,:,:)); V{2} = squeeze(V2(t,:,:,:)); V{3} = squeeze(V3(t,:,:,:));
%     V{1} = squeeze(matV(:,:,:,:)); V{2} = squeeze(V2(t,:,:,:)); V{3} = squeeze(V3(t,:,:,:));

    for i = 1:c.numUsers
        V{i} = matV(:,:,:,i,t);
    end

    % Channel and receiver parameters
    chanMdl       = c.chanMdl;               % TGac fading channel model
    precodingType = c.precodingType;                    % Precoding type; ZF or MMSE
    snr           = c.snr;                      % SNR in dB
    eqMethod      = c.eqMethod;                    % Equalization method

    groupID = c.groupID;


    numSTSVec = numSTSAll(1:numUsers);
%     numTx = sum(numSTSVec);  
    cfgVHTMU = wlanVHTConfig('ChannelBandwidth',chanBW,...
        'NumUsers',numUsers,...
        'UserPositions',userPos(1:numUsers),...
        'NumTransmitAntennas',numTx,...
        'NumSpaceTimeStreams',numSTSVec,...
        'MCS',mcsVec(1:numUsers),...
        'ChannelCoding',chCodingVec(1:numUsers),...
        'APEPLength',apepVec(1:numUsers),...
        'GroupID',groupID);


    %%% Channel Config
    TGAC  = cell(numUsers, 1);
    uIndex    = [10 5 2 1];             % chosen for a maximum of 4 users
    chanDelay = zeros(numUsers, 1);
%     chanSeeds = [randi(1000) randi(2000) randi(3000) randi(4000)];
    chanSeeds = chSeeds(t,:);

    for uIdx = 1:numUsers
        TGAC{uIdx} = wlanTGacChannel(...
            'SampleRate',wlanSampleRate(cfgVHTMU),...
            'DelayProfile',chanMdl,...
            'ChannelBandwidth',chanBW,...
            'TransmitReceiveDistance',5,...
            'UserIndex',uIndex(uIdx),...
            'NumTransmitAntennas',numTx,...
            'NumReceiveAntennas',numSTSVec(uIdx),...
            'Seed',chanSeeds(uIdx),...
            'RandomStream','mt19937ar with seed');
        chanInfo = info(TGAC{uIdx});
        chanDelay(uIdx) = chanInfo.ChannelFilterDelay;
    end

    scNum = size(V{1},1);

    steeringMatrix = zeros(scNum, numTx, sum(numSTSVec)); %Nst-by-Nt-by-Nsts



    for i = 1:numUsers
        bfIdx = sum(numSTSVec(1:i-1))+(1:numSTSVec(i));
        steeringMatrix(:,:,bfIdx) = V{i};
    end

    if strcmp(precodingType,'ZF')
        delta = 0;
    else
        delta = (numTx/(10^(snr/10))) * eye(numTx);
    end

    for i = 1:scNum %% num of suncarriers
        h = squeeze(steeringMatrix(i,:,:));
        steeringMatrix(i,:,:) = h/(h'*h + delta);
    end

    cfgVHTMU.SpatialMapping = 'Custom';
    cfgVHTMU.SpatialMappingMatrix = permute(steeringMatrix,[1 3 2]); 


    % Create data sequences, one for each user
    txDataBits = cell(numUsers, 1);
    psduDataBits = cell(numUsers, 1);
    for uIdx = 1:numUsers
        % Generate payload for each user
        txDataBits{uIdx} = randi([0 1], cfgVHTMU.APEPLength(uIdx)*8, 1, 'int8');

        % Pad payload with zeros to form a PSDU
        psduDataBits{uIdx} = [txDataBits{uIdx}; ...
            zeros((cfgVHTMU.PSDULength(uIdx)-cfgVHTMU.APEPLength(uIdx))*8, 1, 'int8')];
    end


    % Generate the multi-user VHT waveform
    txSig = wlanWaveformGenerator(psduDataBits, cfgVHTMU);

    % Transmit through per-user fading channel 
    rxSig = cell(numUsers, 1);
    for uIdx = 1:numUsers
        % Append zeroes to allow for channel filter delay
        rxSig{uIdx} = TGAC{uIdx}([txSig; zeros(10, numTx)]);
    end

    %%% Recovery
    ind = wlanFieldIndices(cfgVHTMU);

    % Single-user receivers recover payload bits
    rxDataBits = cell(numUsers, 1);


    for uIdx = 1:numUsers
        % Add WGN per receiver
        rxNSig = awgn(rxSig{uIdx}, snr);
        rxNSig = rxNSig(chanDelay(uIdx)+1:end, :);

        % User space-time streams
        stsU = numSTSVec(uIdx);

        % Perform channel estimation based on VHT-LTF
        rxVHTLTF  = rxNSig(ind.VHTLTF(1):ind.VHTLTF(2),:);
        demodVHTLTF = wlanVHTLTFDemodulate(rxVHTLTF, chanBW, numSTSVec);
        chanEst = wlanVHTLTFChannelEstimate(demodVHTLTF, chanBW, numSTSVec);

        % Get single stream channel estimate
        chanEstSSPilots = vhtSingleStreamChannelEstimate(demodVHTLTF,cfgVHTMU);

        % Extract VHT Data samples from the waveform
        rxVHTData = rxNSig(ind.VHTData(1):ind.VHTData(2),:);

        % Estimate the noise power in VHT data field
        nVar = vhtNoiseEstimate(rxVHTData,chanEstSSPilots,cfgVHTMU);

        % Recover information bits in VHT Data field
        [rxDataBits{uIdx}, ~, ~] = wlanVHTDataRecover(rxVHTData, ...
            chanEst, nVar, cfgVHTMU, uIdx, 'EqualizationMethod', eqMethod, ...
            'PilotPhaseTracking', 'None', 'LDPCDecodingMethod', 'layered-bp'); 
    end


    ber = inf(1, numUsers);
    for uIdx = 1:numUsers
        idx = (1:cfgVHTMU.APEPLength(uIdx)*8).';
        [~, ber(uIdx)] = biterr(txDataBits{uIdx}(idx), rxDataBits{uIdx}(idx));
        BER(uIdx,t) = ber(uIdx);
    %     disp(['Bit Error Rate for User ' num2str(uIdx) ': ' num2str(ber(uIdx))]);
    end
end

end
