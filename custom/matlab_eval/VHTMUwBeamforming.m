% Modifying the beamforming example for multiple users
%Scenario: TxAntennas = 6, NumUsers = 3, NSTSPerUser = 2;

%initialization
% Transmission param:
BW = 'CBW20';                  
numUsers = 3;                         
STSVec = [2 2 2];                 
userPos = [0 1 2 3];   
mcsVec = [4 4 4];                
apepVec = [4000 4000 4000];   

% Channel param
chanDelayProf = 'Model-B';             
precType = 'MMSE';                   
snr = 38;     

% Creating VHT format configuration object

if (numUsers==1)
    groupID = 0;
else
    groupID = 2;
end

numTx = sum(STSVec);
cfgVHTMU = wlanVHTConfig('ChannelBandwidth', BW,...
    'NumUsers', numUsers, ...
    'NumTransmitAntennas', numTx, ...
    'GroupID', groupID, ...
    'NumSpaceTimeStreams', STSVec,...
    'UserPositions', userPos(1:numUsers), ...
    'MCS', mcsVec(1:numUsers), ...
    'APEPLength', apepVec(1:numUsers));


%%% Channel config: 3 independent channels
TGac = cell(numUsers, 1);
chanSeeds = [1 2 3 4]; 
uIndex    = [10 5 2 1];    %%% for random angle offset
chDelay = zeros(numUsers, 1);
for uIdx = 1:numUsers
    TGac{uIdx} = wlanTGacChannel(...
        'ChannelBandwidth', cfgVHTMU.ChannelBandwidth,...
        'DelayProfile', chanDelayProf, ...
        'UserIndex', uIndex(uIdx), ...
        'NumTransmitAntennas', numTx, ...
        'NumReceiveAntennas', STSVec(uIdx), ...
        'RandomStream', 'mt19937ar with seed', ...
        'Seed', chanSeeds(uIdx),...
        'SampleRate', wlanSampleRate(cfgVHTMU), ...
        'TransmitReceiveDistance',100);
    chanInfo = info(TGac{uIdx});
    chDelay(uIdx) = chanInfo.ChannelFilterDelay;
end
%%
%%%% Adding beamformer
% channel sounding signals (null data packet transmission)

cfgVHTNDP = wlanVHTConfig('ChannelBandwidth', BW,...
    'NumUsers', 1, ...
    'NumTransmitAntennas', numTx, ...
    'GroupID', 0, ...
    'NumSpaceTimeStreams', sum(STSVec),...
    'MCS', 0, ...
    'APEPLength', 0);

% Generate the null data packet, with no data
txNDPSig = wlanWaveformGenerator([], cfgVHTNDP);


% NPD transmission
txNPDSig = [txNDPSig; zeros(10, numTx)];
rxNPDSig = cell(numUsers, 1);
for uIdx = 1:numUsers
    % Append zeroes to allow for channel filter delay
    rxChan = TGac{uIdx}(txNPDSig);
    rxNPDSigNoise = awgn(rxChan, snr);
    rxNPDSig{uIdx} = rxNPDSigNoise(chDelay(uIdx)+1:end, :);
end

% CSI feedback
mat = cell(numUsers,1);
for uIdx = 1:numUsers
    mat{uIdx} = vhtCSIFeedback(rxNPDSig{uIdx},cfgVHTNDP,uIdx,STSVec);
end
% Number of subcarriers:
numST = length(mat{1});
% Beamforming matrix size: NST-by-NSTS-by-NT
beamformingMat = zeros(numST, sum(STSVec), sum(STSVec));

for uIdx = 1:numUsers
    bfIdx = sum(STSVec(1:uIdx-1))+(1:STSVec(uIdx));
    beamformingMat(:,:,bfIdx) = mat{uIdx};
end

% Zero-forcing or MMSE precoding solution
if strcmp(precType, 'ZF')
    delta = 0; % Zero-forcing
else
    delta = (numTx/(10^(snr/10))) * eye(numTx); % MMSE
end

for i = 1:numST
    h = squeeze(beamformingMat(i,:,:));
    beamformingMat(i,:,:) = h/(h'*h+delta);
end

cfgVHTMU.SpatialMapping = 'Custom';
cfgVHTMU.SpatialMappingMatrix = permute(beamformingMat,[1,3,2]);
%%
% Create data sequences, one for each user
psduData = cell(numUsers, 1);
for uIdx = 1:numUsers
    psduData{uIdx} = randi([0 1], cfgVHTMU.PSDULength(uIdx)*8, 1);
end

% Generate the multi-user VHT waveform
txSig = wlanWaveformGenerator(psduData, cfgVHTMU);

% Transmit through user channel
rxSig = cell(numUsers, 1);
for uIdx = 1:numUsers
    % Append zeroes to allow for channel filter delay
    rxChan = TGac{uIdx}([txSig; zeros(10, numTx)]);
    rxSigNoise = awgn(rxChan, snr);
    rxSig = rxSigNoise(chDelay(uIdx)+1:end, :);
end

%%% Recovery
%%% Get field indices
ind = wlanFieldIndices(cfgVHTMU);
rxData = cell(numUsers, 1);
scaler = zeros(numUsers, 1);
spAxes = gobjects(sum(STSVec), 1);
hfig = figure('Name','Per-stream equalized symbol constellation');

for uIdx = 1:numUsers

    stsU = STSVec(uIdx);

    % Perform channel estimation based on VHT-LTF
    rxVHTLTF  = rxSig(ind.VHTLTF(1):ind.VHTLTF(2),:);
    demodVHTLTF = wlanVHTLTFDemodulate(rxVHTLTF, BW, STSVec);
    chanEst = wlanVHTLTFChannelEstimate(demodVHTLTF, BW, STSVec);

    % Get single stream channel estimate
    chanEstSSPilots = vhtSingleStreamChannelEstimate(demodVHTLTF,cfgVHTMU);

    % Extract VHT Data
    rxVHTData = rxSig(ind.VHTData(1):ind.VHTData(2),:);

    % Estimate the noise power 
    nVar = vhtNoiseEstimate(rxVHTData,chanEstSSPilots,cfgVHTMU);

    % Recover data in VHT Data field
    [rxData{uIdx}, ~, eqsym] = wlanVHTDataRecover(rxVHTData, ...
        chanEst, nVar, cfgVHTMU, uIdx);
    scaler(uIdx) = ceil(max(abs([real(eqsym(:)); imag(eqsym(:))])));
    % Plot equalized symbols for all streams per user
    for i = 1:stsU
        subplot(numUsers, max(STSVec), (uIdx-1)*max(STSVec)+i);
        plot(reshape(eqsym(:,:,i), [], 1), '.');
        axis square
        spAxes(sum([0 STSVec(1:(uIdx-1))])+i) = gca; % Store axes handle
        title(['User ' num2str(uIdx) ', Stream ' num2str(i)]);
        grid on;
    end
end


for i = 1:numel(spAxes)
    xlim(spAxes(i),[-max(scaler) max(scaler)]);
    ylim(spAxes(i),[-max(scaler) max(scaler)]);
end
% pos = get(hfig, 'Position');
% set(hfig, 'Position', [pos(1)*0.7 pos(2)*0.7 1.3*pos(3) 1.3*pos(4)]);




