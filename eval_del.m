% Compute the Validation Accuracy for EfficientNet-B2 for epoch selection.
% by heran, 2023/06/24.

clc;
clear;
close all;

%% ========================================= Basic Setup =======================================
% Parameter Setting.
nclass = 14;
labelSize = 512;

repIDVec = [34023, 34039, 53457, 70814, 84129, 101448, 114800, 123758,
            123936, 134852, 140948, 153037, 164632, 180127, 183630, 195120];
epochIDVec = [5,6,7,8,9,10,11,12,13,14,15];
lthr = 0.64;
hthr = 0.81;

trdom = 'CT';

% Path to Results.
labelPath = '/data/FLARE_Challenge/code/EarlyStopforpseudolabel/experiments/SR3 for MRI_250527_205321_image_label/results/evaluation_time%06d_epoch%d/';

% Data Name.
gtLabelNameFormat = '%s-%s*.npy_tg.npy';
estLabelNameFormat = '%s-%s*.npy_fake.npy';

fileNameAll = dir([sprintf(labelPath, repIDVec(1), epochIDVec(1)), sprintf(gtLabelNameFormat, trdom, '')]);
idstrAll = {};

for i = 1 : length(fileNameAll)

    fileName = fileNameAll(i).name;
    namediv = split(fileName, '-');

    idstrAll(i) = namediv(2);
end

idstrEff = unique(idstrAll);


% ==============================================================================================


%% ==================================== Compute scores =========================================
% iter for repeat times.
for trep = 1 : length(repIDVec)

    repID = repIDVec(trep);

    % iter for Epoch.
    for iepoch = 1 : length(epochIDVec)

        epochID = epochIDVec(iepoch);
        ilabelPath = sprintf(labelPath, repID, epochID);

        if ~exist(ilabelPath, 'dir')
            continue;
        end

        fprintf("Processing times %d epoch %d !\n", repID, epochID);
        
        diceValueMat = zeros(length(idstrEff), nclass-1);
        % iter for Data.
        for jdata = 1 : length(idstrEff)

            idstr = idstrEff{jdata};

            gtLabel_namelist = natsortfiles( dir([ilabelPath, sprintf(gtLabelNameFormat, trdom, idstr)]) );
            estLabel_namelist = natsortfiles( dir([ilabelPath, sprintf(estLabelNameFormat, trdom, idstr)]) );

            sliceNum = length(gtLabel_namelist);
            gtVol  = zeros(labelSize, labelSize, sliceNum, nclass-1);
            estVol = zeros(labelSize, labelSize, sliceNum, nclass-1);

            % iter for Slice.
            for kslice = 1 : sliceNum

                gtLabelacName = [ilabelPath, gtLabel_namelist(kslice).name];
                gtLabel = readNPY(gtLabelacName);
                gtLabel = squeeze(gtLabel) * nclass;

                estLabelacName = [ilabelPath, estLabel_namelist(kslice).name];
                estLabel = readNPY(estLabelacName);
                estLabel = squeeze(estLabel) * nclass;

                % ============================================
                for tcls = 1 : (nclass-1)

                    gt_tmp = zeros(labelSize, labelSize);
                    est_tmp = zeros(labelSize, labelSize);

                    gt_tmp(gtLabel==tcls) = 1;
                    gtVol(:,:,kslice,tcls) = gt_tmp;
                    est_tmp(estLabel==tcls) = 1;
                    estVol(:,:,kslice,tcls) = est_tmp;

                end
                % ============================================

            end

            for tcls = 1 : (nclass-1)
                diceValueMat(jdata, tcls) = eval_compDice(gtVol(:,:,:,tcls), estVol(:,:,:,tcls), 0.5);
            end

        end

        mean(diceValueMat(:))
        
        if mean(diceValueMat(:)) < lthr
            rmdir(ilabelPath, 's');
        elseif mean(diceValueMat(:)) > hthr
            rmdir(ilabelPath, 's');
        end

    end

end
