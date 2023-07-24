close all;
clear all; %#ok<CLALL>

% the sounds i made have sample rate 48,000 Hz and they are 5 seconds long
% the original files are 2MHz sample rate and 1 second long.
% did i convert from 2MHz to 240KHz and then play like they are 48KHz?

% ya got me.
A=load('cyberpowerups_001.mat');
B=load('cortelcophone_001.mat');
sounds=cell(3,1);
N=5;

for ik=1:3
    if ( ik==1 )
        TheData=double(A.data);
        sOut = 'cyberpower.wav';
    elseif ( ik==2 )
        TheData=double(B.data);
        sOut='cortelcophone.wav';
    else
        TheData=0.5*(double(A.data)+double(B.data));
        sOut = 'combined.wav';
    end
    fs=A.samp_rate;
    fs=double(fs); %#ok<NASGU>
    dataLow=TheData;

    % pretend the SR is 2e6/10
    fs=2e6/10;
    fs=fs/4;
    dataLow = lowpass(double(dataLow),100,fs);

    window=hanning(fs/4);

    % data, window, overlap, nfft, fs
    Length=10*fs;
    dataLowPart = dataLow(1:Length);
    
    [S, W, T]=spectrogram(double(dataLowPart),window,fs/8, fs/4, fs);

    figure,imagesc(T,W(1:1000),log(abs(S(1:1000,:))));
    colormap jet;
    dataLowPartNorm = dataLowPart(1:N*fs)/max( max(dataLowPart), abs(min(dataLowPart)) );
    audiowrite(sOut,dataLowPartNorm,fs);

    sounds{ik}=dataLowPartNorm;

end
