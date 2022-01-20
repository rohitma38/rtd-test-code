from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatch
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import scipy.signal as signal
from librosa.display import waveplot, specshow
from IPython.display import Audio
import parselmouth
sns.set_theme(rc={"xtick.bottom" : True, "ytick.left" : False, "xtick.major.size":4, "xtick.minor.size":2, "ytick.major.size":4, "ytick.minor.size":2, "xtick.labelsize": 10, "ytick.labelsize": 10})

#from Nithya
def readCycleAnnotation(cyclePath, numDiv, startTime, endTime):
    '''Function to read cycle annotation and add divisions in the middle if required.

    Parameters:
        cyclePath: path to the cycle annotation file
        numDiv: number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime: start time of audio being analysed
        endTime: end time of audio being analysed

    Returns:
        vibhaags: a numpy array of annotations from the file
        matras: a numpy array of division between annotations
    '''

    cycle_df = pd.read_csv(cyclePath)
    index_values = cycle_df.loc[(cycle_df['Time'] >= startTime) & (cycle_df['Time'] <= endTime)].index.values
    vibhaags = cycle_df.iloc[index_values]
    if vibhaags.shape[0]==1:
        vibhaags = cycle_df.iloc[max(index_values[0]-1, 0):min(index_values[-1]+2, cycle_df.shape[0])]

    # add divisions in the middle
    matras = []
    for ind, vibhaag in enumerate(vibhaags['Time'].values[:-1]):
        matras.extend(np.around(np.linspace(vibhaag, vibhaags['Time'].values[ind+1], num = numDiv, endpoint=False), 2)[1:])
    
    return vibhaags, matras

def drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c='purple', y=None, size=10):
    '''Draws annotations on ax

    Parameters
        cyclePath: path to the cycle annotation file
        numDiv: number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime: start time of audio being analysed
        endTime: end time of audio being analysed
        ax: axis to plot in
        c: colour to plot lines in
        y: y coordinate for annotated cycle number text
        size: font size for annotated text

    Returns
        ax: axis that has been plotted in
    '''
    vibhaags, matras = readCycleAnnotation(cyclePath, numDiv, startTime, endTime)
    if matras is not None:
        for matra in matras:
            if (matra-startTime >= 0) & (matra<endTime):
                ax.axvline(matra - startTime, linestyle='--', c=c, linewidth=0.85)
    if vibhaags is not None:
        for _, vibhaag in vibhaags.iterrows():
            if (vibhaag['Time']-startTime >= 0) & (vibhaag['Time']<endTime):
                ax.axvline((vibhaag['Time']) - startTime, linestyle='-', c=c)
                x = vibhaag['Time']-startTime
                if y is None: y = -0.4
                if np.modf(vibhaag['Cycle'])[0]==0.0:
                    text = int(vibhaag['Cycle'])
                else:
                    text = vibhaag['Cycle']
                ax.annotate(text, (x, y), bbox=dict(facecolor='grey', edgecolor='white'), c='white', fontsize=size)
    return ax

def pitchCountour(audioPath, startTime, endTime, minPitch, maxPitch, notes, tonic, timeStep=0.01, octaveJumpCost=0.9, veryAccurate=True, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, xticks=False, yticks=False, xlabel=True, title='Pitch Contour (cents)'):
    '''Returns pitch contour for the audio file

    Uses `plotPitch` to plot pitch contour if ax is not None.

    Parameters
        audioPath: path to audio file
        startTime: time to start reading audio file
        endTime: time to end reading audio file
        minPitch: minimum pitch to read for contour extraction
        maxPitch: maximum pitch to read for contour extraction
        notes: list of note objects indicating notes present in the raga
        tonic: tonic of the audio
        timeStep: time steps in which audio is extracted
        octaveJumpCost: parameter passed to pitch detection function
        veryAccurate: parameter passed to pitch detection function
        ax: axis to plot the pitch contour in
        freqXlabels: time (in seconds) after which each x label occurs
        annotate: if True, will annotate tala markings
        cyclePath: path to file with tala cycle annotations
        numDiv: number of divisions to put between each annotation marking
        xticks: if True, will plot xticklabels
        yticks: if True, will plot yticklabels

    Returns:
        ax: plot of pitch contour if ax was not None
        pitch: pitch object extracted from audio if ax was None
    '''
    
    audio, sr = librosa.load(audioPath, sr=None, mono=True, offset=startTime, duration = endTime - startTime)
    snd = parselmouth.Sound(audio, sr)
    pitch = snd.to_pitch_ac(time_step=timeStep, pitch_floor=minPitch, very_accurate=veryAccurate, octave_jump_cost=octaveJumpCost, pitch_ceiling=maxPitch)
    
    # plot the contour
    if ax is not None:  return plotPitch(pitch, notes, ax, tonic, startTime, endTime, freqXlabels, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, xticks=xticks, yticks=yticks,  xlabel=xlabel, title=title)
    else:   return pitch

def plotPitch(pitch, notes, ax, tonic, startTime, endTime, freqXlabels=5, xticks=True, yticks=True, xlabel=True, title='Pitch Contour (cents)', annotate=False, cyclePath=None, numDiv=0, cAnnot='purple'):
    '''Converts the pitch contour from Hz to Cents, and plots it

    Parameters
        pitch: pitch object from `pitchCountour`
        notes: object for each note used for labelling y-axis
        ax: axis object on which plot is to be plotted
        tonic: tonic (in Hz) of audio clip
        freqXlabels: time (in seconds) after which each x label occurs
        annotate: if true will mark taala markings 
        xticks: if True, will print x tick labels
        yticks: if True, will print y tick labels
        annotate: if True, 
    Returns
        ax: plotted axis
    '''
    yvals = pitch.selected_array['frequency']
    yvals[yvals==0] = np.nan    # mark unvoiced regions as np.nan
    yvals[~(np.isnan(yvals))] = 1200*np.log2(yvals[~(np.isnan(yvals))]/tonic)   #convert Hz to cents
    xvals = pitch.xs()
    ax = sns.lineplot(x=xvals, y=yvals, ax=ax)
    ax.set(xlabel='Time (s)' if xlabel else '', 
    ylabel='Notes', 
    title=title, 
    xlim=(0, endTime-startTime), 
    xticks=np.arange(0, endTime-startTime, freqXlabels), 
    xticklabels=np.around(np.arange(startTime, endTime, freqXlabels)).astype(int) if xticks else [],
    yticks=[x['cents'] for x in notes if (x['cents'] >= min(yvals[~(np.isnan(yvals))])) & (x['cents'] <= max(yvals[~(np.isnan(yvals))]))] if yticks else [], 
    yticklabels=[x['label'] for x in notes if (x['cents'] >= min(yvals[~(np.isnan(yvals))])) & (x['cents'] <= max(yvals[~(np.isnan(yvals))]))] if yticks else [])

    if annotate:
        ax = drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c=cAnnot)
    return ax

def spectrogram(audioPath, startTime, endTime, cmap, ax, amin=1e-5, freqXlabels=5, xticks=False, yticks=False, title='Spectrogram', annotate=False, cyclePath=None, numDiv=0, cAnnot='purple'):
    '''Plots spectrogram

    Parameters
        audioPath: path to the audio file
        startTime: time to start reading the audio at
        endTime: time to stop reading audio at
        cmap: colormap to use to plot spectrogram
        ax: axis to plot spectrogram in
        amin: controls the contrast of the spectrogram; passed into librosa.power_to_db function
        freqXlabels: time (in seconds) after which each x label occurs
        xticks: if true, will print x labels
        yticks: if true, will print y labels
        annotate: if True, will annotate tala markings
        cyclePath: path to file with tala cycle annotations
        numDiv: number of divisions to put between each annotation marking
        cAnnot: colour for the annotation marking
    '''
    audio, sr = librosa.load(audioPath, sr=None, mono=True, offset=startTime, duration=endTime-startTime)
    audio /= np.max(np.abs(audio))
    
    # stft params
    winsize = int(np.ceil(sr*40e-3))
    hopsize = int(np.ceil(sr*10e-3))
    nfft = int(2**np.ceil(np.log2(winsize)))

    # STFT
    f,t,X = signal.stft(audio, fs=sr, window='hann', nperseg=winsize, noverlap=(winsize-hopsize), nfft=nfft)
    X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=amin)

    specshow(X_dB, x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopsize, ax=ax, cmap=cmap)
    ax.set(ylabel='Frequency (Hz)', 
    xlabel='', 
    title=title,
    xlim=(0, endTime-startTime), 
    xticks=np.arange(0, endTime-startTime, freqXlabels), 
    xticklabels=np.around(np.arange(startTime, endTime, freqXlabels)).astype(int) if xticks else [],
    ylim=(0, 5000),
    yticks=[0, 2e3, 4e3] if yticks else [], 
    yticklabels=['0', '2k', '4k'] if yticks else [])

    if annotate:
        ax = drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c=cAnnot)
    return ax

def drawWave(audioPath, startTime, endTime, ax, xticks=False, freqXlabels=5, title='Waveform', annotate=False, cyclePath=None, numDiv=0, cAnnot='purple', alpha=1):
    '''Plots the wave plot of the audio

    audioPath: path to the audio file
    startTime: time to start reading the audio at
    endTime: time to stop reading the audio at
    ax: axis to plot waveplot in
    xticks: if True, will plot xticklabels
    freqXlabels: time (in seconds) after which each x label occurs
    annotate: if True, will annotate tala markings
    cyclePath: path to file with tala cycle annotations
    numDiv: number of divisions to put between each annotation marking
    cAnnot: colour for the annotation marking
    '''
    audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=endTime-startTime)
    audio /= np.max(np.abs(audio))

    waveplot(audio, sr, ax=ax, alpha=alpha)
    ax.set(xlabel='' if not xticks else 'Time (s)', 
    xlim=(0, endTime-startTime), 
    xticks= np.arange(0, endTime-startTime, freqXlabels),
    xticklabels=[] if not xticks else np.around(np.arange(startTime, endTime, freqXlabels), 2),
    ylabel='Amplitude',
    title=title)
    if annotate:
        ax = drawAnnotation(cyclePath, numDiv, startTime, endTime, ax, c=cAnnot)
    return ax

def playAudio(audioPath, startTime, endTime):
    '''Plays relevant part of audio

    Parameters
        audioPath: file path to audio
        startTime: time to start reading audio at
        endTime: time to stop reading audio at

    Returns:
        iPython.display.Audio object that plays the audio
    '''
    audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=endTime-startTime)
    audio /= np.max(np.abs(audio))
    return Audio(audio, rate=sr)

def generateFig(noRows, figSize, heightRatios):
    '''Generates a matplotlib.pyplot.figure and axes to plot in

    Axes in the plot are stacked vertically in one column, with height of each axis determined by heightRatios

    Parameters
        noRows: number of rows in the figure
        figSize: (width, height) in inches  of the figure
        heightRatios: list of the fraction of height that each axis should take; len(heightRatios) has to be equal to noRows

    Returns:
        fig: figure object
        axs: list of axes objects
    '''
    if len(heightRatios) != noRows:
        Exception("Length of heightRatios has to be equal to noRows")
    fig = plt.figure(figsize=figSize)
    specs = fig.add_gridspec(noRows, 1, height_ratios = heightRatios)
    axs = [fig.add_subplot(specs[i, 0]) for i in range(noRows)]
    return fig, axs

#rohit - onset related
def fadeIn(x,length):
    fade_func = np.ones(len(x))
    fade_func[:length] = np.hanning(2*length)[:length]
    x*=fade_func
    return x

def fadeOut(x,length):
    fade_func = np.ones(len(x))
    fade_func[-length:] = np.hanning(2*length)[length:]
    x*=fade_func
    return x
    
def to_dB(x, C):
    '''Applies logarithmic (base 10) transformation
    
    Parameters
        x: input signal
        C: scaling constant
    
    Returns
        log-scaled x
    '''
    return np.log10(1 + x*C)/(np.log10(1+C))

def subBandEner(X,fs,band):
    '''Computes spectral sub-band energy (suitable for vocal onset detection)

    Parameters
        X: STFT of an audio signal x
        fs: sampling rate
        band: edge frequencies (in Hz) of the sub-band of interest
        
    Returns
        sbe: array with each value representing the magnitude STFT values in a short-time frame squared & summed over the sub-band
    '''

    binLow = int(np.ceil(band[0]*X.shape[0]/(fs/2)))
    binHi = int(np.ceil(band[1]*X.shape[0]/(fs/2)))
    sbe = np.sum(np.abs(X[binLow:binHi])**2, 0)
    return sbe

def biphasicDerivative(x, tHop, norm=1, rectify=1):
    '''Computes a biphasic derivative (See [1] for a detailed explanation of the algorithm)
    
    Parameters
        x: input signal
        tHop: frame- or hop-length used to obtain input signal values (reciprocal of sampling rate of x)
        norm: if output is to be normalized
        rectify: if output is to be rectified to keep only positive values (sufficient for peak-picking)
    
    Returns
        x: after performing the biphasic derivative of input x (i.e, convolving with a biphasic derivative filter)

    [1] Rao, P., Vinutha, T.P. and Rohit, M.A., 2020. Structural Segmentation of Alap in Dhrupad Vocal Concerts. Transactions of the International Society for Music Information Retrieval, 3(1), pp.137–152. DOI: http://doi.org/10.5334/tismir.64
    '''

    n = np.arange(-0.1, 0.1, tHop)
    tau1 = 0.015  # = (1/(T_1*sqrt(2))) || -ve lobe width
    tau2 = 0.025  # = (1/(T_2*sqrt(2))) || +ve lobe width
    d1 = 0.02165  # -ve lobe position
    d2 = 0.005  # +ve lobe position
    A = np.exp(-pow((n-d1)/(np.sqrt(2)*tau1), 2))/(tau1*np.sqrt(2*np.pi))
    B = np.exp(-pow((n+d2)/(np.sqrt(2)*tau2), 2))/(tau2*np.sqrt(2*np.pi))
    biphasic = A-B
    x = np.convolve(x, biphasic, mode='same')
    x = -1*x
    
    if norm==1:
        x/=np.max(x)
        x-=np.mean(x)
    
    if rectify==1:
        x*=(x>0)
    return x


def spectralFlux(X, fs, band, aMin=1e-4, normalize=True):
    '''Computes 1st order rectified spectral flux (difference) of a given STFT input
    
    Parameters
        X: input STFT matrix
        fs: sampling rate of audio signal
        band: frequency band over which to compute flux from STFT (sub-band spectral flux)
        aMin: lower threshold value to prevent log(0)
        normalize: whether to normalize output before returning

    Returns
        specFlux: array with frame-wise spectral flux values
    '''

    X = 20*np.log10(aMin+abs(X)/np.max(np.abs(X)))
    binLow = int(band[0]*X.shape[0]/(fs/2))
    binHi = int(band[1]*X.shape[0]/(fs/2))
    specFlux = np.array([0])
    for hop in range(1,X.shape[1]):
        diff = X[binLow:binHi,hop]-X[binLow:binHi,hop-1]
        diff = (diff + abs(diff))/2
        specFlux=np.append(specFlux,sum(diff))
    if normalize:
        specFlux/=max(specFlux)
    return specFlux

def compute_local_average(x, M, Fs=1):
    """Compute local average of signal

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        M: Determines size (2M+1*Fs) of local average
        Fs: Sampling rate

    Returns:
        local_average: Local average signal
    """
    L = len(x)
    M = int(np.ceil(M * Fs))
    local_average = np.zeros(L)
    for m in range(L):
        a = max(m - M, 0)
        b = min(m + M + 1, L)
        local_average[m] = (1 / (2 * M + 1)) * np.sum(x[a:b])
    return local_average

def spectral_flux_fmp(x, Fs=1, N=1024, W=640, H=80, gamma=100, M=20, norm=1, band=[]):
    """Compute spectral-based novelty function

    Notebook: C6/C6S1_NoveltySpectral.ipynb

    Args:
        x: Signal
        Fs: Sampling rate
        N: Window size
        H: Hope size
        gamma: Parameter for logarithmic compression
        M: Size (frames) of local average
        norm: Apply max norm (if norm==1)
        band: List of lower and upper spectral freq limits

    Returns:
        novelty_spectrum: Energy-based novelty function
        Fs_feature: Feature rate
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=W, window='hanning')
    Fs_feature = Fs / H
    Y = np.log(1 + gamma * np.abs(X))
	
	#if vocal-band SF
    if len(band)!=0: 
        band = np.array(band)*(N/2+1)/Fs
        Y = Y[int(band[0]):int(band[1]),:]

    Y_diff = np.diff(Y)
    Y_diff[Y_diff < 0] = 0
    novelty_spectrum = np.sum(Y_diff, axis=0)
    novelty_spectrum = np.concatenate((novelty_spectrum, np.array([0.0])))
    if M > 0:
        local_average = compute_local_average(novelty_spectrum, M)
        novelty_spectrum = novelty_spectrum - local_average
        novelty_spectrum[novelty_spectrum < 0] = 0.0
    if norm == 1:
        max_value = max(novelty_spectrum)
        if max_value > 0:
            novelty_spectrum = novelty_spectrum / max_value
    return novelty_spectrum

def getOnsetActivation(x=None, fs=16000, win_dur=0.4, hop_dur=0.01, n_fft=1024, source='vocal', audioPath=None, startTime=0, endTime=None):
    '''Computes onset activation function

    Parameters
        x: audio signal
        audioPath: path to the audio file
        startTime: time to start reading the audio at
        endTime: time to stop reading audio at
        fs: sampling rate to read audio at
        win_dur: window duration in seconds for STFT
        hop_dur: hop duration in seconds for STFT
        n_fft: DFT size
        source: choice of instrument - 'vocal' or 'perc' (perc for percussion)
    '''

    win_size = int(np.ceil(win_dur*fs))
    hop_size = int(np.ceil(hop_dur*fs))
    n_fft = int(2**(np.ceil(np.log2(win_size))))
    
    if x is not None:
        x = fadeIn(x,int(0.5*fs))
        x = fadeOut(x,int(0.5*fs))
        if endTime is None: endTime = len(x)/fs
        x = x[int(np.ceil(startTime*fs)):int(np.ceil(endTime*fs))]
    elif audioPath is not None:
        x, _ = librosa.load(audioPath, sr=fs, offset=startTime, duration=endTime-startTime)
    else:
        print('Provide either the audio signal or path to the stored audio file on disk')
        raise

    X,_ = librosa.magphase(librosa.stft(x,win_length=win_size, hop_length=hop_size, n_fft=n_fft))

    if source=='vocal':
        sub_band = [600,2400]
        odf = subBandEner(X, fs, sub_band)
        odf = to_dB(odf,100)
        odf = biphasicDerivative(odf, hop_size/fs, norm=1, rectify=1)

        onsets = librosa.onset.onset_detect(onset_envelope=odf.copy(), sr=fs, hop_length=hop_size, pre_max=4, post_max=4, pre_avg=6, post_avg=6, wait=50, delta=0.12)*hop_size/fs

    else:
        sub_band = [0,fs/2]
        odf = spectral_flux_fmp(x, Fs=fs, N=n_fft, W=win_size, H=hop_size, M=20, band=sub_band)
        #odf = spectral_flux(X,fs,sub_band,amin=1e-4,normalize=True)
        #odf = librosa.onset.onset_strength(x,sr=fs,hop_length=hop_size)

        onsets = librosa.onset.onset_detect(onset_envelope=odf, sr=fs, hop_length=hop_size, pre_max=1, post_max=1, pre_avg=1, post_avg=1, wait=10, delta=0.05)*hop_size/fs

    return odf, onsets

def smooth_downsample_feature_sequence(X, Fs, filt_len=41, down_sampling=10, w_type='boxcar'):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel

    Notebook: C3/C3S1_FeatureSmoothing.ipynb

    Args:
        X: Feature sequence
        Fs: Frame rate of `X`
        filt_len: Length of smoothing filter
        down_sampling: Downsampling factor
        w_type: Window type of smoothing filter

    Returns:
        X_smooth: Smoothed and downsampled feature sequence
        Fs_feature: Frame rate of `X_smooth`
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode='same') / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature

#plotting
def drawAnnotationV2(cyclePath, numDiv, startTime, endTime, ax, columns=['Time', 'Cycle'], c='purple', y=None, size=10):
    '''Draws annotations on ax

    Parameters
        cyclePath: path to the cycle annotation file
        numDiv: number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime: start time of audio being analysed
        endTime: end time of audio being analysed
        ax: axis to plot in
        columns: list of column (header) names in the annotation file 
        c: colour to plot lines in
        y: y coordinate for annotated cycle number text
        size: font size for annotated text

    Returns
        ax: axis that has been plotted in
    '''
    vibhaags, matras = readCycleAnnotation(cyclePath, numDiv, startTime, endTime)

    col0, col1 = columns[0], columns[1]

    if matras is not None:
        for matra in matras:
            if (matra-startTime >= 0) & (matra<endTime):
                ax.axvline(matra - startTime, linestyle='--', c=c, linewidth=0.85)
    if vibhaags is not None:
        for _, vibhaag in vibhaags.iterrows():
            if (vibhaag[col0]-startTime >= 0) & (vibhaag[col0]<endTime):
                ax.axvline((vibhaag[col0]) - startTime, linestyle='-', c=c, linewidth=2.)
                x = vibhaag[col0]-startTime
                if y is None: y = -0.4
                if type(vibhaag[col1]) in ['int', 'float']:
                    if np.modf(vibhaag[col1])[0]==0.0:
                        text = int(vibhaag[col1])
                else:
                    text = vibhaag[col1]
                #ax.annotate(text, (x, y), bbox=dict(facecolor='grey', edgecolor='white'), c='white', fontsize=size)
    return ax

def plot_matrix(X, Fs=1, Fs_F=1, T_coef=None, F_coef=None, xlabel='Time (seconds)', ylabel='Frequency (Hz)', title='',
                dpi=72, colorbar=True, colorbar_aspect=20.0, ax=None, figsize=(6, 3), **kwargs):
    """Plot a matrix, e.g. a spectrogram or a tempogram

    Notebook: B/B_PythonVisualization.ipynb

    Args:
        X: The matrix
        Fs: Sample rate for axis 1
        Fs_F: Sample rate for axis 0
        T_coef: Time coeffients. If None, will be computed, based on Fs.
        F_coef: Frequency coeffients. If None, will be computed, based on Fs_F.
        xlabel: Label for x axis
        ylabel: Label for y axis
        title: Title for plot
        dpi: Dots per inch
        colorbar: Create a colorbar.
        colorbar_aspect: Aspect used for colorbar, in case only a single axes is used.
        ax: Either (1.) a list of two axes (first used for matrix, second for colorbar), or (2.) a list with a single
            axes (used for matrix), or (3.) None (an axes will be created).
        figsize: Width, height in inches
        **kwargs: Keyword arguments for matplotlib.pyplot.imshow

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
        im: The image plot
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax = [ax]
    if T_coef is None:
        T_coef = np.arange(X.shape[1]) / Fs
    if F_coef is None:
        F_coef = np.arange(X.shape[0]) / Fs_F

    if 'extent' not in kwargs:
        x_ext1 = (T_coef[1] - T_coef[0]) / 2
        x_ext2 = (T_coef[-1] - T_coef[-2]) / 2
        y_ext1 = (F_coef[1] - F_coef[0]) / 2
        y_ext2 = (F_coef[-1] - F_coef[-2]) / 2
        kwargs['extent'] = [T_coef[0] - x_ext1, T_coef[-1] + x_ext2, F_coef[0] - y_ext1, F_coef[-1] + y_ext2]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = 'gray_r'
    if 'aspect' not in kwargs:
        kwargs['aspect'] = 'auto'
    if 'origin' not in kwargs:
        kwargs['origin'] = 'lower'

    im = ax[0].imshow(X, **kwargs)

    if len(ax) == 2 and colorbar:
        plt.colorbar(im, cax=ax[1])
    elif len(ax) == 2 and not colorbar:
        ax[1].set_axis_off()
    elif len(ax) == 1 and colorbar:
        plt.sca(ax[0])
        plt.colorbar(im, aspect=colorbar_aspect)

    ax[0].set_xlabel(xlabel, fontsize=14)
    ax[0].set_ylabel(ylabel, fontsize=14)
    ax[0].set_title(title, fontsize=18)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, im

def drawSyllablesBols(filePath, ax, y=None, timeOffset=50e-3, size=10, textColor='blue'):
    data_df = pd.read_csv(filePath)
    if y is None: y = 0
    for i in range(data_df.shape[0]):
        try:
            text = data_df.iloc[i]['Syllable']
        except KeyError:
            text = data_df.iloc[i]['Bol']
        x = data_df.iloc[i]['Time'] + timeOffset
        ax.annotate(text, (x, y), bbox=dict(facecolor='white', edgecolor='grey', linewidth=0.5), c=textColor, fontsize=size)
    return ax

#rhythm-related
def ACF_DFT_sal(signal, t_ACF_lag, t_ACF_frame, t_ACF_hop, fs):
    n_ACF_lag = int(t_ACF_lag*fs)
    n_ACF_frame = int(t_ACF_frame*fs)
    n_ACF_hop = int(t_ACF_hop*fs)
    signal = subsequences(signal, n_ACF_frame, n_ACF_hop)
    ACF = np.zeros((len(signal), n_ACF_lag))
    for i in range(len(ACF)):
        ACF[i][0] = np.dot(signal[i], signal[i])
        for j in range(1, n_ACF_lag):
            ACF[i][j] = np.dot(signal[i][:-j], signal[i][j:])
    DFT = (abs(np.fft.rfft(signal)))
    sal = np.zeros(len(ACF))
    for i in range(len(ACF)):
        sal[i] = max(ACF[i])
    for i in range(len(ACF)):
        if max(ACF[i])!=0:
            ACF[i] = ACF[i]/max(ACF[i])
        if max(DFT[i])!=0:
            DFT[i] = DFT[i]/max(DFT[i])
    return (ACF, DFT, sal)

def subsequences(signal, frame_length, hop_length):
    shape = (int(1 + (len(signal) - frame_length)/hop_length), frame_length)
    strides = (hop_length*signal.strides[0], signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

def tempo_period_comb_filter(ACF, fs, norm=1):
    L = np.shape(ACF)[1]
    min_lag = 10
#    max_lag = L/2    #For madhya laya
    max_lag = L     # For Vil bandish
#    max_lag = 66    # For Drut gats
    N_peaks = 11    # 11 for Madhya laya & 9 for Drut gat  
    
 
    window = zeros((L, L))
    for j in range(min_lag, max_lag):
        C = j*np.arange(1, N_peaks)
        D = np.concatenate((C, C+1, C-1, C+2, C-2, C+3, C-3))
        D = D[D<L]
        norm_factor = len(D)
        if norm == 1:
            window[j][D] = 1.0/norm_factor
        else:
            window[j][D] = 1.0
            
    tempo_candidates = np.dot(ACF, transpose(window))
    
#    re_weight = zeros(L)
#    re_weight[min_lag:max_lag] = linspace(1, 0.5, max_lag-min_lag)
#    tempo_candidates = tempo_candidates*re_weight
    tempo_lag = np.argmax(tempo_candidates, axis=1)/float(fs)
    return (window, tempo_candidates, tempo_lag)

def viterbi_tempo_rhythm(tempo_candidates, fs,transition_penalty):
    T = np.shape(tempo_candidates)[0]
    L = np.shape(tempo_candidates)[1]

    p1=transition_penalty

    cost=ones((T,L//2))*1e8
    m=zeros((T,L//2))

#    cost[:,0]=1000
#    m[0][1]=argmax(tempo_candidates[0])
#    for j in range(1,L):
#        cost[1][j]=abs(60*fs/j-60*fs/m[0][1])/tempo_candidates[1][j]
#        m[1][j]=m[0][1]/fs

    for i in range(1,T):
        for j in range(1,L//2):
            cost[i][j]=cost[i-1][1]+p1*abs(60.0*fs/j-60.0*fs)-tempo_candidates[i][j]
            for k in range(2,L//2):
                if cost[i][j]>cost[i-1][k]+p1*abs(60.0*fs/j-60.0*fs/k)-tempo_candidates[i][j]:
                    cost[i][j]=cost[i-1][k]+p1*abs(60.0*fs/j-60.0*fs/k)-tempo_candidates[i][j]
                    m[i][j]=int(k)
                    
    tempo_period=zeros(T)
    tempo_period[T-1]=argmin(cost[T-1,1:])/float(fs)
    t=int(m[T-1,argmin(cost[T-1,1:])])
    i=T-2
    while(i>=0):
        tempo_period[i]=t/float(fs)
        t=int(m[i][t])
        i=i-1
    return tempo_period

#SSM and novelty detection
def annotationsToFMPFormat(annotPath, duration=None, endTime=None):
    """
    Converts 2 column (Time, Label) format into FMP format ([start time, end time, label])
    
    Parameters:
        annot: path to annotations file
        duration, endTime: provide one of duration or end time of audio to use in last row of converted annotation
    
    Returns:
        annotFMP: annotations in FMP format        
    """
    if (endTime is None) & (duration is None):
        print('Please provide either end time or duration of audio')
        raise
    elif endTime is None: endTime = annot.iloc[0,0]+duration

    annot = pd.read_csv(annotPath)
    annotFMP = []
    for i in range(annot.shape[0]-1):
        annotFMP.append([annot.iloc[i,0], annot.iloc[i+1,0], annot.iloc[i,-1]])

    annotFMP.append([annot.iloc[-1,0], endTime, annot.iloc[-1,-1]])

    return annotFMP

def threshold_matrix(S, thresh, strategy='absolute', scale=False, penalty=0, binarize=False):
    """Treshold matrix in a relative fashion

    Notebook: C4/C4/C4S2_SSM-Thresholding.ipynb

    Args:
        S: Input matrix
        thresh: Treshold (meaning depends on strategy)
        strategy: Thresholding strategy ('absolute', 'relative', 'local')
        scale: If scale=True, then scaling of positive values to range [0,1]
        penalty: Set values below treshold to value specified
        binarize: Binarizes final matrix (positive: 1; otherwise: 0)
        Note: Binarization is applied last (overriding other settings)


    Returns:
        S_thresh: Thresholded matrix
    """
    if np.min(S)<0:
        raise Exception('All entries of the input matrix must be nonnegative')

    S_thresh = np.copy(S)
    N, M = S.shape
    num_cells = N*M

    if strategy == 'absolute':
        thresh_abs = thresh
        S_thresh[S_thresh < thresh] = 0

    if strategy == 'relative':
        thresh_rel = thresh
        num_cells_below_thresh = int(np.round(S_thresh.size*(1-thresh_rel)))
        if num_cells_below_thresh < num_cells:
            values_sorted = np.sort(S_thresh.flatten('F'))
            thresh_abs = values_sorted[num_cells_below_thresh]
            S_thresh[S_thresh < thresh_abs] = 0
        else:
            S_thresh = np.zeros([N,M])

    if strategy == 'local':
        thresh_rel_row = thresh[0]
        thresh_rel_col = thresh[1]
        S_binary_row = np.zeros([N,M])
        num_cells_row_below_thresh = int(np.round(M*(1-thresh_rel_row)))
        for n in range(N):
            row = S[n,:]
            values_sorted = np.sort(row)
            if num_cells_row_below_thresh < M:
                thresh_abs = values_sorted[num_cells_row_below_thresh]
                S_binary_row[n,:] = (row>=thresh_abs)
        S_binary_col = np.zeros([N,M])
        num_cells_col_below_thresh = int(np.round(N*(1-thresh_rel_col)))
        for m in range(M):
            col = S[:,m]
            values_sorted = np.sort(col)
            if num_cells_col_below_thresh < N:
                thresh_abs = values_sorted[num_cells_col_below_thresh]
                S_binary_col[:,m] = (col>=thresh_abs)
        S_thresh =  S * S_binary_row * S_binary_col

    if scale:
        cell_val_zero = np.where(S_thresh==0)
        cell_val_pos = np.where(S_thresh>0)
        if len(cell_val_pos[0])==0:
            min_value = 0
        else:
            min_value = np.min(S_thresh[cell_val_pos])
        max_value = np.max(S_thresh)
        #print('min_value = ', min_value, ', max_value = ', max_value)
        if max_value > min_value:
            S_thresh = np.divide((S_thresh - min_value) , (max_value -  min_value))
            if len(cell_val_zero[0])>0:
                S_thresh[cell_val_zero] = penalty
        else:
            print('Condition max_value > min_value is voliated: output zero matrix')

    if binarize:
        S_thresh[S_thresh > 0] = 1
        S_thresh[S_thresh < 0] = 0
    return S_thresh
    
def compute_SSM(X, L_smooth=16, tempo_rel_set=np.array([1]), shift_set=np.array([0]), strategy = 'relative', scale=1, thresh= 0.15, penalty=0, binarize=0, feature='chroma'):
    """Compute an SSM

    Notebook: C4S2_SSM-Thresholding.ipynb

    Args:
        X: Feature sequence
        L_smooth, tempo_rel_set, shift_set: Parameters for computing SSM
        strategy, scale, thresh, penalty, binarize: Parameters used for tresholding SSM

    Returns:
        S_thresh, I: SSM and index matrix
    """

    S, I = compute_SM_TI(X, X, L=L_smooth, tempo_rel_set=tempo_rel_set, shift_set=shift_set, direction=2)
    S_thresh = threshold_matrix(S, thresh=thresh, strategy=strategy, scale=scale, penalty=penalty, binarize=binarize)
    return S_thresh, I

def compressed_gray_cmap(alpha=5, N=256):
    """Creates a logarithmically or exponentially compressed grayscale colormap

    Notebook: B/B_PythonVisualization.ipynb

    Args:
        alpha: The compression factor. If alpha > 0, it performs log compression (enhancing black colors).
            If alpha < 0, it performs exp compression (enhancing white colors).
            Raises an error if alpha = 0.
        N: The number of rgb quantization levels (usually 256 in matplotlib)

    Returns:
        color_wb: The colormap
    """
    assert alpha != 0

    gray_values = np.log(1 + abs(alpha) * np.linspace(0, 1, N))
    gray_values /= gray_values.max()

    if alpha > 0:
        gray_values = 1 - gray_values
    else:
        gray_values = gray_values[::-1]

    gray_values_rgb = np.repeat(gray_values.reshape(256, 1), 3, axis=1)
    color_wb = LinearSegmentedColormap.from_list('color_wb', gray_values_rgb, N=N)
    return color_wb

def plotSSM(X, Fs_X, S, Fs_S, ann, duration, color_ann=None,
                               title='', label='Time (seconds)', time=True,
                               figsize=(6, 6), fontsize=10, clim_X=None, clim=None):
    """Plot SSM along with annotations (standard setting is time in seconds)
    Notebook: C4/C4S2_SSM.ipynb
    """
    cmap = compressed_gray_cmap(alpha=-10)
    fig, ax = plt.subplots(2, 3, gridspec_kw={'width_ratios': [0.1, 1, 0.05],
                           'wspace': 0.2,
                           'height_ratios': [1, 0.1]},
                           figsize=figsize)
    #plot_matrix(X, Fs=Fs_X, ax=[ax[0,1], ax[0,2]], clim=clim_X,
    #                     xlabel='', ylabel='', title=title)
    #ax[0,0].axis('off')
    plot_matrix(S, Fs=Fs_S, ax=[ax[0,1], ax[0,2]], cmap=cmap, clim=clim,
                         title='', xlabel='', ylabel='', colorbar=True);
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    ax[0,1].grid(False)
    plot_segments(ann, ax=ax[1,1], time_axis=True, fontsize=fontsize,
                           colors=color_ann,
                           time_label=label, time_max=duration)
    ax[1,2].axis('off'), ax[1,0].axis('off')
    plot_segments(ann, ax=ax[0,0], time_axis=True, fontsize=fontsize,
                           direction='vertical', colors=color_ann,
                           time_label=label, time_max=duration)
    return fig, ax

def check_segment_annotations(annot, default_label=''):
    """Checks segment annotation. If label is missing, adds an default label.

    Args:
        annot: A List of the form of [(start_position, end_position, label), ...], or
            [(start_position, end_position), ...]
        default_label: The default label used if label is missing

    Returns:
        annot: A List of tuples in the form of [(start_position, end_position, label), ...]
    """
    assert isinstance(annot[0], (list, np.ndarray, tuple))
    len_annot = len(annot[0])
    assert all(len(a) == len_annot for a in annot)
    if len_annot == 2:
        annot = [(a[0], a[1], default_label) for a in annot]

    return annot

def color_argument_to_dict(colors, labels_set, default='gray'):
    """Creates a color dictionary

    Args:
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap, 3. list or np.ndarray of
            matplotlib color specifications, 4. dict that assigns labels  to colors
        labels_set: List of all labels
        default: Default color, used for labels that are in labels_set, but not in colors

    Returns:
        color_dict: Dictionary that maps labels to colors
    """

    if isinstance(colors, str):
        # FMP colormap
        if colors in FMP_COLORMAPS:
            color_dict = {l: c for l, c in zip(labels_set, FMP_COLORMAPS[colors])}
        # matplotlib colormap
        else:
            cm = plt.get_cmap(colors)
            num_labels = len(labels_set)
            colors = [cm(i / (num_labels + 1)) for i in range(num_labels)]
            color_dict = {l: c for l, c in zip(labels_set, colors)}

    # list/np.ndarray of colors
    elif isinstance(colors, (list, np.ndarray, tuple)):
        color_dict = {l: c for l, c in zip(labels_set, colors)}

    # is already a dict, nothing to do
    elif isinstance(colors, dict):
        color_dict = colors

    else:
        raise ValueError('`colors` must be str, list, np.ndarray, or dict')

    for key in labels_set:
        if key not in color_dict:
            color_dict[key] = default

    return color_dict
    
def plot_segments(annot, ax=None, figsize=(10, 2), direction='horizontal', colors='FMP_1', time_min=None,
                  time_max=None, nontime_min=0, nontime_max=1, time_axis=True, nontime_axis=False, time_label=None,
                  swap_time_ticks=False, edgecolor='k', axis_off=False, dpi=72, adjust_nontime_axislim=True, alpha=None,
                  print_labels=True, label_ticks=False, **kwargs):
    """Creates a multi-line plot for annotation data

    Args:
        annot: A List of tuples in the form of [(start_position, end_position, label), ...]
        ax: The Axes instance to plot on. If None, will create a figure and axes.
        figsize: Width, height in inches
        direction: 'vertical' or 'horizontal'
        colors: Several options: 1. string of FMP_COLORMAPS, 2. string of matplotlib colormap, 3. list or np.ndarray of
            matplotlib color specifications, 4. dict that assigns labels  to colors
        time_min: Minimal limit for time axis. If None, will be min annotation.
        time_max: Maximal limit for time axis. If None, will be max from annotation.
        nontime_min: Minimal limit for non-time axis.
        nontime_max: Maximal limit for non-time axis.
        time_axis: Display time axis ticks or not
        nontime_axis: Display non-time axis ticks or not
        swap_time_ticks: For horizontal: xticks up; for vertical: yticks left
        edgecolor: Color for edgelines of segment box
        axis_off: Calls ax.axis('off')
        dpi: Dots per inch
        adjust_nontime_axislim: Adjust non-time-axis. Usually True for plotting on standalone axes and False for
            overlay plotting
        alpha: Alpha value for rectangle
        print_labels: Print labels inside Rectangles
        label_ticks: Print labels as ticks
        kwargs: Keyword arguments for matplotlib.pyplot.annotate

    Returns:
        fig: The created matplotlib figure or None if ax was given.
        ax: The used axes.
    """
    assert direction in ['vertical', 'horizontal']
    annot = check_segment_annotations(annot)

    if 'color' not in kwargs:
        kwargs['color'] = 'k'
    if 'weight' not in kwargs:
        kwargs['weight'] = 'bold'
        #kwargs['weight'] = 'normal'
    if 'fontsize' not in kwargs:
        kwargs['fontsize'] = 12
    if 'ha' not in kwargs:
        kwargs['ha'] = 'center'
    if 'va' not in kwargs:
        kwargs['va'] = 'center'

    if colors is None:
        colors = 'FMP_1'

    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    labels_set = sorted(set([label for start, end, label in annot]))
    colors = color_argument_to_dict(colors, labels_set)

    nontime_width = nontime_max - nontime_min
    nontime_middle = nontime_min + nontime_width / 2
    all_time_middles = []

    for start, end, label in annot:
        time_width = end - start
        time_middle = start + time_width / 2
        all_time_middles.append(time_middle)

        if direction == 'horizontal':
            rect = mpatch.Rectangle((start, nontime_min), time_width, nontime_width, facecolor=colors[label],
                                    edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (time_middle, nontime_middle), **kwargs)
        else:
            rect = mpatch.Rectangle((nontime_min, start), nontime_width, time_width, facecolor=colors[label],
                                    edgecolor=edgecolor, alpha=alpha)
            ax.add_patch(rect)
            if print_labels:
                ax.annotate(label, (nontime_middle, time_middle), **kwargs)

    if time_min is None:
        time_min = min(start for start, end, label in annot)
    if time_max is None:
        time_max = max(end for start, end, label in annot)

    if direction == 'horizontal':
        ax.set_xlim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_ylim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_yticks([])
        if not time_axis:
            ax.set_xticks([])
        if swap_time_ticks:
            ax.xaxis.tick_top()
        if time_label:
            ax.set_xlabel(time_label)
        if label_ticks:
            ax.set_xticks(all_time_middles)
            ax.set_xticklabels([label for start, end, label in annot])

    else:
        ax.set_ylim([time_min, time_max])
        if adjust_nontime_axislim:
            ax.set_xlim([nontime_min, nontime_max])
        if not nontime_axis:
            ax.set_xticks([])
        if not time_axis:
            ax.set_yticks([])
        if swap_time_ticks:
            ax.yaxis.tick_right()
        if time_label:
            ax.set_ylabel(time_label)
        if label_ticks:
            ax.set_yticks(all_time_middles)
            ax.set_yticklabels([label for start, end, label in annot])

    if axis_off:
        ax.axis('off')

    if fig is not None:
        plt.tight_layout()

    return fig, ax

def normalize_feature_sequence(X, norm='2', threshold=0.0001, v=None):
    """Normalizes the columns of a feature sequence

    Notebook: C3/C3S1_FeatureNormalization.ipynb

    Args:
        X: Feature sequence
        norm: The norm to be applied. '1', '2', 'max' or 'z'
        threshold: An threshold below which the vector `v` used instead of normalization
        v: Used instead of normalization below `threshold`. If None, uses unit vector for given norm

    Returns:
        X_norm: Normalized feature sequence
    """
    assert norm in ['1', '2', 'max', 'z']

    K, N = X.shape
    X_norm = np.zeros((K, N))

    if norm == '1':
        if v is None:
            v = np.ones(K, dtype=np.float64) / K
        for n in range(N):
            s = np.sum(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == '2':
        if v is None:
            v = np.ones(K, dtype=np.float64) / np.sqrt(K)
        for n in range(N):
            s = np.sqrt(np.sum(X[:, n] ** 2))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'max':
        if v is None:
            v = np.ones(K, dtype=np.float64)
        for n in range(N):
            s = np.max(np.abs(X[:, n]))
            if s > threshold:
                X_norm[:, n] = X[:, n] / s
            else:
                X_norm[:, n] = v

    if norm == 'z':
        if v is None:
            v = np.zeros(K, dtype=np.float64)
        for n in range(N):
            mu = np.sum(X[:, n]) / K
            sigma = np.sqrt(np.sum((X[:, n] - mu) ** 2) / (K - 1))
            if sigma > threshold:
                X_norm[:, n] = (X[:, n] - mu) / sigma
            else:
                X_norm[:, n] = v

    return X_norm

def convert_structure_annotation(ann, Fs=1, remove_digits=False, index=False):
    """Convert structure annotations

    Notebook: C4/C4S1_MusicStructureGeneral.ipynb

    Args:
        ann: Structure annotions
        Fs: Sampling rate
        remove_digits: Remove digits from labels

    Returns:
        ann_converted: Converted annotation
    """
    ann_converted = []
    for r in ann:
        s = r[0]*Fs
        t = r[1]*Fs
        if index:
            s = int(np.round(s))
            t = int(np.round(t))
        if remove_digits:
            label = ''.join([i for i in r[2] if not i.isdigit()])
        else:
            label = r[2]
        ann_converted = ann_converted + [[s, t, label]]
    return ann_converted

def compute_SM_dot(X,Y):
    """Computes similarty matrix from feature sequences using dot (inner) product
    Notebook: C4/C4S2_SSM.ipynb
    """
    S = np.dot(np.transpose(Y),X)
    return S

def filter_diag_mult_SM(S, L=1, tempo_rel_set=[1], direction=0):
    """Path smoothing of similarity matrix by filtering in forward or backward direction
    along various directions around main diagonal
    Note: Directions are simulated by resampling one axis using relative tempo values

    Notebook: C4/C4S2_SSM-PathEnhancement.ipynb

    Args:
        S: Self-similarity matrix (SSM)
        L: Length of filter
        tempo_rel_set: Set of relative tempo values
        direction: Direction of smoothing (0: forward; 1: backward)

    Returns:
        S_L_final: Smoothed SM
    """
    N = S.shape[0]
    M = S.shape[1]
    num = len(tempo_rel_set)
    S_L_final = np.zeros((M,N))

    for s in range(0, num):
        M_ceil = int(np.ceil(N/tempo_rel_set[s]))
        resample = np.multiply(np.divide(np.arange(1,M_ceil+1),M_ceil),N)
        np.around(resample, 0, resample)
        resample = resample -1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)
        S_resample = S[:,index_resample]

        S_L = np.zeros((M,M_ceil))
        S_extend_L = np.zeros((M + L, M_ceil + L))

        # Forward direction
        if direction==0:
            S_extend_L[0:M,0:M_ceil] = S_resample
            for pos in range(0,L):
                S_L = S_L + S_extend_L[pos:(M + pos), pos:(M_ceil + pos)]

        # Backward direction
        if direction==1:
            S_extend_L[L:(M+L),L:(M_ceil+L)] = S_resample
            for pos in range(0,L):
                S_L = S_L + S_extend_L[(L-pos):(M + L - pos), (L-pos):(M_ceil + L - pos)]

        S_L = S_L/L
        resample = np.multiply(np.divide(np.arange(1,N+1),N),M_ceil)
        np.around(resample, 0, resample)
        resample = resample-1
        index_resample = np.maximum(resample, np.zeros(len(resample))).astype(np.int64)

        S_resample_inv = S_L[:, index_resample]
        S_L_final = np.maximum(S_L_final, S_resample_inv)

    return S_L_final

def shift_cyc_matrix(X, shift=0):
    """Cyclic shift of features matrix along first dimension

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X: Feature respresentation
        shift: Number of bins to be shifted

    Returns:
        X_cyc: Cyclically shifted feature matrix
    """
    #Note: X_cyc = np.roll(X, shift=shift, axis=0) does to work for jit
    K, N = X.shape
    shift = np.mod(shift, K)
    X_cyc = np.zeros((K,N))
    X_cyc[shift:K, :] = X[0:K-shift, :]
    X_cyc[0:shift, :] = X[K-shift:K, :]
    return X_cyc

def compute_SM_TI(X, Y, L=1, tempo_rel_set=[1], shift_set=[0], direction=2):
    """Compute enhanced similaity matrix by applying path smoothing and transpositions

    Notebook: C4/C4S2_SSM-TranspositionInvariance.ipynb

    Args:
        X, Y: Input feature sequences
        L: Length of filter
        tempo_rel_set: Set of relative tempo values
        shift_set: Set of shift indices
        direction: Direction of smoothing (0: forward; 1: backward; 2: both directions)

    Returns:
        S_TI: Transposition-invariant SM
        I_TI: Transposition index matrix
    """
    for shift in shift_set:
        X_cyc = shift_cyc_matrix(X, shift)
        S_cyc = compute_SM_dot(X,X_cyc)

        if direction==0:
            S_cyc = filter_diag_mult_SM(S_cyc, L, tempo_rel_set, direction=0)
        if direction==1:
            S_cyc = filter_diag_mult_SM(S_cyc, L, tempo_rel_set, direction=1)
        if direction==2:
            S_forward = filter_diag_mult_SM(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=0)
            S_backward = filter_diag_mult_SM(S_cyc, L, tempo_rel_set=tempo_rel_set, direction=1)
            S_cyc = np.maximum(S_forward, S_backward)
        if shift ==  shift_set[0]:
            S_TI = S_cyc
            I_TI = np.ones((S_cyc.shape[0],S_cyc.shape[1])) * shift
        else:
            #jit does not like the following lines
            #I_greater = np.greater(S_cyc, S_TI)
            #I_greater = (S_cyc>S_TI)
            I_TI[S_cyc>S_TI] = shift
            S_TI = np.maximum(S_cyc, S_TI)

    return S_TI, I_TI

def compute_kernel_checkerboard_Gaussian(L, var=1, normalize=True):
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1]
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L: Parameter specifying the kernel size M=2*L+1
        var: Variance parameter determing the tapering (epsilon)

    Returns:
        kernel: Kernel matrix of size M x M
    """
    taper = np.sqrt(1/2)/(L*var)
    axis = np.arange(-L,L+1)
    gaussian1D = np.exp(-taper**2 * (axis**2))
    gaussian2D = np.outer(gaussian1D,gaussian1D)
    kernel_box = np.outer(np.sign(axis),np.sign(axis))
    kernel = kernel_box * gaussian2D
    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))
    return kernel

def compute_novelty_SSM(S, kernel=None, L=10, var=0.5, exclude=False):
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S: SSM
        kernel: Checkerboard kernel (if kernel==None, it will be computed)
        L: Parameter specifying the kernel size M=2*L+1
        var: Variance parameter determing the tapering (epsilon)
        exclude: Sets the first L and last L values of novelty function to zero

    Returns:
        nov: Novelty function
    """    
    if kernel is None:
        kernel = compute_kernel_checkerboard_Gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2*L + 1
    nov = np.zeros(N)
    #np.pad does not work with numba/jit
    S_padded  = np.pad(S,L,mode='constant')
    
    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n:n+M, n:n+M]  * kernel)
    if exclude:
        right = np.min([L,N])
        left = np.max([0,N-L])
        nov[0:right] = 0
        nov[left:N] = 0
        
    return nov

