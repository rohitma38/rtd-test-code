import warnings
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import scipy.signal as sig
from librosa.display import waveplot, specshow
from IPython.display import Audio, Video 
import parselmouth
import math
import soundfile as sf
import ffmpeg
import utils_fmp as fmp

#set seaborn theme parameters for plots
sns.set_theme(rc={"xtick.bottom" : True, "ytick.left" : False, "xtick.major.size":4, "xtick.minor.size":2, "ytick.major.size":4, "ytick.minor.size":2, "xtick.labelsize": 10, "ytick.labelsize": 10})

def readCycleAnnotation(cyclePath, numDiv, startTime, duration):
    """Function to read cycle annotation and add divisions in the middle if required.

    Parameters:
        cyclePath (str): path to the cycle annotation file
        numDiv (int): number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime (float): start time of audio being analysed
        duration (float): duration of the audio to be analysed

    Returns:
        provided (np.ndarray): a numpy array of annotations from the file
        computed (list): a list of division between annotations
    """

    cycle_df = pd.read_csv(cyclePath)
    index_values = cycle_df.loc[(cycle_df['Time'] >= startTime) & (cycle_df['Time'] <= startTime + duration)].index.values
    if len(index_values) == 0:
        return None, None
    provided = cycle_df.iloc[max(index_values[0]-1, 0):min(index_values[-1]+2, cycle_df.shape[0])]
    # add divisions in the middle
    computed = []
    for ind, val in enumerate(provided['Time'].values[:-1]):
        computed.extend(np.around(np.linspace(val, provided['Time'].values[ind+1], num = numDiv, endpoint=False), 2)[1:])
    return [provided], computed

def readOnsetAnnotation(onsetPath, startTime, duration, onsetKeyword=['Inst']):
    '''Function to read cycle annotation and add divisions in the middle if required.

    Parameters:
        onsetPath (str): path to the cycle annotation file
        startTime (float): start time of audio being analysed
        duration (float): duration of the audio to be analysed
        onsetKeyword (list): list of column names in the onset file to take onsets from

    Returns:
        provided (list): list of numpy arrays of annotations from the file
    '''
    
    onset_df = pd.read_csv(onsetPath)
    provided = []   # variable to store onset timestamps
    for keyword in onsetKeyword:
        provided.append(onset_df.loc[(onset_df[keyword] >= startTime) & (onset_df[keyword] <= startTime + duration)])
    return provided

def drawAnnotation(cyclePath=None, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', numDiv=0, startTime=0, duration=None, ax=None, annotLabel=True, c='purple', alpha=0.8, y=0.7, size=10):
    '''Draws annotations on ax

    Parameters:
        cyclePath (str): path to the cycle annotation file
        onsetPath (str): path to onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        numDiv (int): number of equally spaced divisions to add between pairs of annotations (numDiv - 1 timestamps will be added between each pair)
        startTime (float): start time of audio being analysed
        duration (float): duration of the audio to be analysed
        ax (plt.Axes.axis): axis to plot in
        annotLabel (bool): if True, will print annotation label along with line
        c (str or list): list of colour to plot lines in, one for each onsetTimeKeyword (if provided)
        alpha (float): controls opacity of the annotation lines drawn
        y (float): float value from [0, 1] indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.
        size (int): font size for annotated text

    Returns:
        ax (plt.Axes.axis): axis that has been plotted in
    '''

    if cyclePath is not None:
        provided, computed = readCycleAnnotation(cyclePath, numDiv, startTime, duration)
        timeCol = ['Time']    # name of column with time readings
        labelCol = ['Cycle']  # name of column to extract label of annotation from
        c = c if isinstance(c, list) else [c]
    elif onsetPath is not None:
        if annotLabel and type(onsetTimeKeyword) == type(onsetLabelKeyword) and isinstance(onsetTimeKeyword, list):
            # check if length of lists is the same
            if len(onsetTimeKeyword) != len(onsetLabelKeyword):
                raise Exception('Length of onsetTimeKeyword and onsetLabelKeyword should match')
        if type(onsetTimeKeyword) == type(c) and isinstance(onsetTimeKeyword, list):
            # check if length of lists is the same
            if len(onsetTimeKeyword) != len(c):
                raise Exception('Length of onsetTimeKeyword and c should match')   
        timeCol = onsetTimeKeyword if isinstance(onsetTimeKeyword, list) else [onsetTimeKeyword]    # name of column with time readings
        labelCol = onsetLabelKeyword if isinstance(onsetLabelKeyword, list) else [onsetLabelKeyword]  # name of column to extract label of annotation from
        c = c if isinstance(c, list) else [c]
        provided = readOnsetAnnotation(onsetPath, startTime, duration, onsetKeyword=timeCol)
        computed = None
    else:
        raise Exception('A cycle or onset path has to be provided for annotation')
    if computed is not None:
        for computedVal in computed:
            ax.axvline(computedVal - startTime, linestyle='--', c=c[0], alpha=0.4)
    if provided is not None:
        for i, providedListVal in enumerate(provided):
            firstLabel = True   # marker for first line being plotted; to prevent duplicates from occuring in the legend
            for _, providedVal in providedListVal.iterrows():
                ax.axvline((providedVal[timeCol[i]]) - startTime, linestyle='-', c=c[i], label=timeCol[i] if firstLabel and cyclePath is None else '', alpha=alpha)  # add label only for first line of onset for each keyword
                if firstLabel:  firstLabel = False
                if annotLabel:
                    ylims = ax.get_ylim()   # used to set label at 0.7 height of the plot
                    if isinstance(providedVal[labelCol[i]], str):
                        ax.annotate(f"{providedVal[labelCol[i]]}", (providedVal[timeCol[i]]-startTime, (ylims[1]-ylims[0])*y + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c='white')
                    else:
                        ax.annotate(f"{float(providedVal[labelCol[i]]):g}", (providedVal[timeCol[i]]-startTime, (ylims[1]-ylims[0])*y + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c='white')
    if onsetPath is not None and cyclePath is None:     # add legend only is onsets are given, i.e. legend is added
        ax.legend()
    return ax

def pitchCountour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, maxPitch=660, notes=None, tonic=220, timeStep=0.01, octaveJumpCost=0.9, veryAccurate=True, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', xticks=False, yticks=False, xlabel=True, ylabel=True, title='Pitch Contour (Cents)', annotLabel=True, cAnnot='purple', ylim=None, annotAlpha=0.8):
    '''Returns pitch contour (in cents) for the audio

    Uses `plotPitch` to plot pitch contour if ax is not None.

    Parameters:
        audio (np.ndarray): loaded audio time series
        sr (int): sample rate of audio time series/ to load the audio at
        audioPath (str): path to audio file; only needed if audio is None
        startTime (float): time to start reading audio file
        duration (float): duration of the audio file to read
        minPitch (float): minimum pitch to read for contour extraction
        maxPitch (float): maximum pitch to read for contour extraction
        notes (dict): list of note objects indicating notes present in the raga
        tonic (float): tonic of the audio
        timeStep (float): time steps in which audio is extracted
        octaveJumpCost (float): parameter passed to pitch detection function
        veryAccurate (bool): parameter passed to pitch detection function
        ax (matplotlib.axes.Axes): axis to plot the pitch contour in
        freqXlabels (float): time (in seconds) after which each x label occurs
        annotate (bool): if True, will annotate tala markings
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetTimeKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        xticks (bool): if True, will plot xticklabels
        yticks (bool): if True, will plot yticklabels
        xlabel (bool): if True, will print xlabel
        ylabel (bool): if True will pring ylabel
        title (str): Title to add to the plot
        annotLabel: if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        cAnnot: input to the ax.annotate function for the colour (c) parameter
        ylim (tuple): (min, max) limits for the y axis; if None, will be directly interpreted from the data
        annotAlpha (float): controls opacity of the annotation lines

    Returns:
        ax (matplotlib.axes.Axes): plot of pitch contour if ax was not None
        (pitchvals, timevals): tuple with arrays of pitch values (in cents) and time stamps; returned if ax was None
    '''
    
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        # if audio is not given, load audio from audioPath
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration

    snd = parselmouth.Sound(audio, sr)
    pitch = snd.to_pitch_ac(time_step=timeStep, pitch_floor=minPitch, very_accurate=veryAccurate, octave_jump_cost=octaveJumpCost, pitch_ceiling=maxPitch)

    pitchvals = pitch.selected_array['frequency']
    pitchvals[pitchvals==0] = np.nan    # mark unvoiced regions as np.nan
    if tonic is None:   Exception('No tonic provided')
    pitchvals[~(np.isnan(pitchvals))] = 1200*np.log2(pitchvals[~(np.isnan(pitchvals))]/tonic)    #convert Hz to cents
    timevals = pitch.xs()
    if ax is None:
        warnings.warn('ax not provided; returning pitch and time values')
        return (pitchvals, timevals)
    else:
        # plot the contour
        return plotPitch(pitchvals, timevals, notes, ax, tonic, startTime, duration, freqXlabels, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel, title=title, cAnnot=cAnnot, annotLabel=annotLabel, ylim=ylim, annotAlpha=annotAlpha)

def plotPitch(pitchvals=None, timevals=None, notes=None, ax=None, tonic=None, startTime=0, duration=None, freqXlabels=5, xticks=True, yticks=True, xlabel=True, ylabel=True, title='Pitch Contour (Cents)', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='purple', annotLabel=True, ylim=None, annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Plots the pitch contour

    Parameters:
        pitchvals (np.ndarray): pitch values in cents
        timevals (np.ndarray): time values in seconds
        notes (dict): object for each note used for labelling y-axis
        ax (matplotlib.Axes.axes): axis object on which plot is to be plotted
        tonic (float): tonic (in Hz) of audio clip
        startTime (float): start time for x labels in the plot
        duration (float): duration of audio in the plot (used for x labels)
        freqXlabels (int): time (in seconds) after which each x label occurs
        annotate (bool): if true will mark annotations provided
        xticks (bool): if True, will print x tick labels
        yticks (bool): if True, will print y tick labels
        xlabel (bool): if True, will add label to x axis
        ylabel (bool): if True, will add label to y axis
        title (str): title to add to the plot
        annotate (bool): if True, will add beat annotations to the plot 
        cyclePath (bool): path to file with cycle annotations; used only if annotate is True
        numDiv (int): number of divisions to add between each marked cycle; used only if annotate is True
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot: input to the ax.annotate function for the colour (c) parameter; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True
        ylim (tuple): (min, max) limits for the y axis; if None, will be directly interpreted from the data
        annotAlpha (float): controls opacity of the line drawn
        yAnnot (float): float value from [0, 1] indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.
        sizeAnnot (int): font size for annotated text
    Returns:
        ax: plotted axis
    '''

    # Check that all required parameters are present
    if pitchvals is None:
        Exception('No pitch contour provided')
    if timevals is None:
        warnings.warn('No time values provided, assuming 0.01 s time steps in pitch contour')
        timevals = np.arange(0, len(pitchvals)*0.01, 0.01)
    if ax is None:
        Exception('ax parameter has to be provided')
    
    # duration = xvals[-1] + 1    # set duration as last x value + 1
    ax = sns.lineplot(x=timevals, y=pitchvals, ax=ax)
    ax.set(xlabel='Time Stamp (s)' if xlabel else '', 
    ylabel='Notes' if ylabel else '', 
    title=title, 
    xlim=(0, duration), 
    xticks=np.around(np.arange(math.ceil(startTime)-startTime, duration, freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
    xticklabels=np.around(np.arange(startTime, duration+startTime, freqXlabels) ).astype(int) if xticks else [])
    if notes is not None and yticks:
        # add yticks if needed
        ax.set(
        yticks=[x['cents'] for x in notes if (x['cents'] >= min(pitchvals[~(np.isnan(pitchvals))])) & (x['cents'] <= max(pitchvals[~(np.isnan(pitchvals))]))] if yticks else [], 
        yticklabels=[x['label'] for x in notes if (x['cents'] >= min(pitchvals[~(np.isnan(pitchvals))])) & (x['cents'] <= max(pitchvals[~(np.isnan(pitchvals))]))] if yticks else [])
    if ylim is not None:
        ax.set(ylim=ylim)

    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime, duration, ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha, y=yAnnot, size=sizeAnnot)
    return ax

def spectrogram(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, winSize=0.04, hopSize=0.01, n_fft=None, cmap='Blues', ax=None, amin=1e-5, freqXlabels=5, xticks=False, yticks=False, xlabel=True, ylabel=True, title='Spectrogram', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='purple', annotLabel=True):
    '''Plots spectrogram

    Parameters:
        audio (np.ndarray): loaded audio time series
        sr (int): sample rate that audio time series is loaded/ is to be loaded in
        audioPath (str): path to the audio file; only needed if audio is None
        startTime (float): time to start reading the audio at
        duration (float): duration of audio
        winSize (float): size of window for STFT in seconds
        hopSize (float): size of hop for STFT in seconds
        n_fft (int): DFT size
        cmap (matplotlib.colors.Colormap or str): colormap to use to plot spectrogram
        ax (plt.Axes.axes): axis to plot spectrogram in
        amin (float): controls the contrast of the spectrogram; passed into librosa.power_to_db function
        freqXlabels (float): time (in seconds) after which each x label occurs
        xticks (bool): if true, will print x labels
        yticks (bool): if true, will print y labels
        xlabel (bool): if true, will add an xlabel
        ylabel (bool): if true, will add a ylabel
        title (str): title for the plot
        annotate (bool): if True, will annotate either tala or onset markings; if both are provided, tala annotations will be marked
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each tala annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        onsetKeyword (str): column name in the onset file to take onsets from
        onsetLabelKeyword (str): column name with labels for the onsets; if None, no label will be printed
        cAnnot: input to the ax.annotate function for the colour (c) parameter; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration
    
    # convert winSize and hopSize from seconds to samples
    winSize = int(np.ceil(sr*winSize))
    hopSize = int(np.ceil(sr*hopSize))
    if n_fft is None:
        n_fft = int(2**np.ceil(np.log2(winSize)))

    # STFT
    f,t,X = sig.stft(audio, fs=sr, window='hann', nperseg=winSize, noverlap=(winSize-hopSize), nfft=n_fft)
    X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=amin)

    specshow(X_dB, x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopSize, ax=ax, cmap=cmap)
    ax.set(ylabel='Frequency (Hz)' if ylabel else '', 
    xlabel='Time (s)' if xlabel else '', 
    title=title,
    xlim=(0, duration), 
    xticks=np.around(np.arange(math.ceil(startTime)-startTime, duration, freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
    xticklabels=np.around(np.arange(startTime, duration+startTime, freqXlabels) ).astype(int) if xticks else [], 
    ylim=(0, 5000),
    yticks=[0, 2e3, 4e3] if yticks else [], 
    yticklabels=['0', '2k', '4k'] if yticks else [])

    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime, duration, ax, c=cAnnot, annotLabel=annotLabel)
    return ax

def drawWave(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, xticks=False, yticks=True, xlabel=True, ylabel=True, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, cAnnot='purple', annotLabel=True, odf=False, winSize_odf=0.4, hopSize_odf=0.01, nFFT_odf=1024, source_odf='vocal', cOdf='black', title='Waveform'):
    '''Plots the wave plot of the audio

    Parameters:
        audio (np.ndarray): loaded audio time series
        sr (int): sample rate that audio time series is loaded/ is to be loaded in
        audioPath (str): path to the audio file
        startTime (float): time to start reading the audio at
        duration (float): duration of audio to load
        ax (plt.Axes.axes): axis to plot waveplot in
        xticks (bool): if True, will plot xticklabels
        yticks (bool): if True, will plot yticklabels
        xlabel (bool): if True, will add a x label
        ylabel (bool): if True will add a y label
        freqXlabels (float): time (in seconds) after which each x label occurs
        annotate (bool): if True, will annotate tala markings
        cyclePath (str): path to file with tala cycle annotations
        numDiv (int): number of divisions to put between each annotation marking
        onsetPath (str): path to file with onset annotations; only considered if cyclePath is None
        cAnnot: colour for the annotation marking; input to the ax.annotate function for the colour (c) parameter; used only if annotate is True
        annotLabel (bool): if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        odf (bool): if True, will plot the onset detection function over the wave form
        winSize_odf (float): window size in seconds, fed to the onset detection function; valid only if odf is true
        hopSize_odf (float): hop size in seconds, fed to the onset detection function; valid only if odf is true
        nFFT_odf (int): size of DFT used in onset detection function; valid only if odf is true
        source_odf (str): type of instrument - vocal or pakhawaj, fed to odf; valid only if odf is true
        cOdf: colour to plot onset detection function in; valid only if odf is true
        title (str): title of the plot
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration

    waveplot(audio, sr, ax=ax)
    if odf:
        plotODF(audio=audio, sr=sr, startTime=0, duration=None, ax=ax, winSize_odf=winSize_odf, hopSize_odf=hopSize_odf, nFFT_odf=nFFT_odf, source_odf=source_odf, cOdf=cOdf, ylim=True)
    ax.set(xlabel='' if not xlabel else 'Time (s)', 
    ylabel = '' if not ylabel else 'Amplitude',
    xlim=(0, duration), 
    xticks=[] if not xticks else np.around(np.arange(0, duration, freqXlabels)),
    xticklabels=[] if not xticks else np.around(np.arange(startTime, duration+startTime, freqXlabels), 2),
    yticks=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
    yticklabels=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
    title=title)
    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel)
    
    return ax

def plotODF(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, winSize_odf=0.4, hopSize_odf=0.01, nFFT_odf=1024, source_odf='vocal', cOdf='black', freqXlabels=5, ylim=True, xlabel=False, ylabel=False, xticks=False, yticks=False, title='Onset Detection Function'):
    '''
    Plots onset detection function if ax is provided. If not returns a tuple with 2 arrays - onset detection function values and time stamps

        Parameters:
        audio (np.ndarray): loaded audio time series
        sr (int): sample rate that audio time series is loaded/ is to be loaded in
        audioPath (str): path to the audio file
        startTime (float): time to start reading the audio at
        duration (float): duration of audio to load
        ax (plt.Axes.axes): axis to plot waveplot in
        winSize_odf (float): window size in seconds, fed to the onset detection function
        hopSize_odf (float): hop size in seconds, fed to the onset detection function
        nFFT_odf (int): size of DFT used in onset detection function
        source_odf (str): type of instrument - vocal or pakhawaj, fed to odf
        cOdf: colour to plot onset detection function in
        freqXlabels (float): time (in seconds) after which each x label occurs
        ylim (bool): if True, will reset the ylim to the range of the output of the ODF function; this is added because when the ODF is plotted over another plot, say the waveform, it is easier to see if the ylim is readjusted
        xticks (bool): if True, will plot xticklabels
        yticks (bool): if True, will plot yticklabels
        xlabel (bool): if True, will add a x label
        ylabel (bool): if True will add a y label
        title (str): title of the plot

    Returns:
        ax (matplotlib.Axes.axes): if ax is not None, returns a plot
        (odf_vals, time_vals): if ax is None, returns a tuple with ODF values and time stamps.
    '''

    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration

    odf_vals, _, _ = getOnsetActivation(x=audio, audioPath=None, startTime=startTime, duration=duration, fs=sr, winSize=winSize_odf, hopSize=hopSize_odf, nFFT=nFFT_odf, source=source_odf)
    
    # set time and odf values in variables
    time_vals = np.arange(0, duration, hopSize_odf)
    odf_vals = odf_vals[:-1]    # disregard the last frame of odf_vals since it is centered around the frame at time stamp 'duration'

    if ax is None:
        # if ax is None, return (odf_vals, time_vals)
        return (odf_vals, time_vals)
    else:
        ax.plot(time_vals, odf_vals[:-1], c=cOdf)     # plot odf_vals and consider odf_vals for all values except the last frame
        max_abs_val = max(abs(min(odf_vals)), abs(max(odf_vals)))   # find maximum value to set y limits to ensure symmetrical plot
        # set ax parameters only if they are not None
        ax.set(xlabel='' if not xlabel else 'Time (s)', 
        ylabel = '' if not ylabel else 'ODF',
        xlim=(0, duration), 
        xticks=[] if not xticks else np.around(np.arange(0, duration, freqXlabels)),
        xticklabels=[] if not xticks else np.around(np.arange(startTime, duration+startTime, freqXlabels), 2),
        yticks=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
        yticklabels=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
        ylim= ax.get_ylim() if not ylim else (-max_abs_val, max_abs_val),
        title=title) 
        return ax

def plotEnergy(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, xticks=False, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, cAnnot='purple', annotLabel=True, winSize_odf=0.4, hopSize_odf=0.01, nFFT_odf=1024, source_odf='vocal', cOdf='black'):
    '''
    For debugging puposes only - plots energy function used to calculate odf

    Parameters:
        audio: loaded audio time series
        sr: sample rate that audio time series is loaded/ is to be loaded in
        audioPath: path to the audio file
        startTime: time to start reading the audio at
        duration: duration of audio to load
        ax: axis to plot waveplot in
        xticks: if True, will plot xticklabels
        freqXlabels: time (in seconds) after which each x label occurs
        annotate: if True, will annotate tala markings
        cyclePath: path to file with tala cycle annotations
        numDiv: number of divisions to put between each annotation marking
        onsetPath: path to file with onset annotations; only considered if cyclePath is None
        cAnnot: colour for the annotation marking
        annotLabel: if True, will print annotation label along with line; used only if annotate is True; used only if annotate is True
        odf: if True, will plot the onset detection function over the wave form
        winSize_odf: window size, fed to the onset detection function; valid only if odf is true
        hopSize_odf: hop size in seconds, fed to the onset detection function; valid only if odf is true
        nFFT_odf: size of DFT used in onset detection function; valid only if odf is true
        source_odf: type of instrument - vocal or pakhawaj, fed to odf; valid only if odf is true
        cOdf: colour to plot onset detection function in; valid only if odf is true
    '''
    if ax is None:
        Exception('ax parameter has to be provided')
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
        audio /= np.max(np.abs(audio))
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration
    ax.set(xlabel='' if not xticks else 'Time (s)', 
    xlim=(0, duration), 
    xticks=[] if not xticks else np.around(np.arange(0, duration, freqXlabels)),
    xticklabels=[] if not xticks else np.around(np.arange(startTime, duration+startTime, freqXlabels), 2),
    title='Energy Contour', 
    ylabel='dB')

    _, _, energy = getOnsetActivation(x=audio, audioPath=None, startTime=0, duration=duration, fs=sr, winSize=winSize_odf, hopSize=hopSize_odf, nFFT=nFFT_odf, source=source_odf)
    ax.plot(np.arange(0, duration, hopSize_odf), energy[:-1], c=cOdf)
    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel)
    return ax
    
def playAudio(audio=None, sr=16000, audioPath=None, startTime=0, duration=None):
    '''Plays relevant part of audio

    Parameters:
        audio: loaded audio sample
        sr: sample rate of audio; valid only if audio is not None
        audioPath: path to audio file
        startTime: time to start reading audio at
        duration: duration of the audio to load

    Returns:
        iPython.display.Audio object that plays the audio
    '''
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=duration)
    return Audio(audio, rate=sr)

def playAudioWClicks(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, onsetFile=None, onsetLabels=['Inst', 'Tabla'], destPath=None):
    '''Plays relevant part of audio along with clicks at timestamps in onsetTimes

    Parameters:
        audio (np.ndarray): loaded audio sample
        sr (float): sample rate of audio
        audioPath (str): path to audio file
        startTime (float): time to start reading audio at
        duration (float): duration of the audio to load
        onsetFile (str): file path to onset values
        onsetLabels (str): column names in onsetFile to mark with clicks
        destPath (str): path to save audio file at; if None, will not save any audio file

    Returns:
        iPython.display.Audio object that plays the audio
    '''

    if audio is None:
        audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio)
    onsetFileVals = pd.read_csv(onsetFile)
    onsetTimes = []
    for onsetLabel in onsetLabels:
        onsetTimes.append(onsetFileVals.loc[(onsetFileVals[onsetLabel] >= startTime) & (onsetFileVals[onsetLabel] <= startTime+duration), onsetLabel].values)
    clickTracks = [librosa.clicks(onsetTime-startTime, sr=sr, length=len(audio), click_freq=1000*(2*i+1)) for i, onsetTime in enumerate(onsetTimes)]
    audioWClicks = 0.8*audio  # add clicks to this variable
    for clickTrack in clickTracks:
        audioWClicks += 0.2/len(clickTracks)*clickTrack
    if destPath is not None:
        sf.write(destPath, audioWClicks, sr)
    return Audio(audioWClicks, rate=sr)

def playVideo(video=None, videoPath=None, startTime=0, duration=None, destPath='Data/Temp/VideoPart.mp4', videoOffset=0):
    '''Plays relevant part of audio

    Parameters:
        video (np.ndarray): loaded video sample
        videoPath (str): path to video file
        startTime (float): time to start reading the video from
        duration (float): duration of the video to load
        destPath (str): path to store shortened video
        videoOffset (float): number of seconds offset between video and audio; time in audio + videioOffset = time in video
    Returns:
        iPython.display.Video object that plays the video
    '''
    if video is None:
        if duration is None and startTime == 0:
            # play the entire video
            return Video(videoPath, embed=True)
        else:
            # store a shortened video in destPath
            vid = ffmpeg.input(videoPath)
            joined = ffmpeg.concat(
            vid.video.filter('trim', start=startTime+videoOffset, duration=duration).filter('setpts', 'PTS-STARTPTS'),
            vid.audio.filter('atrim', start=startTime+videoOffset, duration=duration).filter('asetpts', 'PTS-STARTPTS'),
            v=1,
            a=1
            ).node
            v3 = joined['v']
            a3 = joined['a']
            out = ffmpeg.output(v3, a3, destPath).overwrite_output()
            out.run()
            return Video(destPath, embed=True)
    else:
        return Video (data=video, embed=True)

def generateFig(noRows, figSize=(14, 7), heightRatios=None):
    '''Generates a matplotlib.pyplot.figure and axes to plot in

    Axes in the plot are stacked vertically in one column, with height of each axis determined by heightRatios

    Parameters:
        noRows: number of rows in the figure
        figSize: (width, height) in inches  of the figure
        heightRatios: list of the fraction of height that each axis should take; len(heightRatios) has to be equal to noRows

    Returns:
        fig: figure object
        axs: list of axes objects
    '''
    if len(heightRatios) != noRows:
        Exception("Length of heightRatios has to be equal to noRows")
    if heightRatios is None:
        # if heightRatios is None
        heightRatios = np.ones(noRows)
    fig = plt.figure(figsize=figSize)
    specs = fig.add_gridspec(noRows, 1, height_ratios = heightRatios)
    axs = [fig.add_subplot(specs[i, 0]) for i in range(noRows)]
    return fig, axs

## Onset detection related
def subBandEner(X,fs,band):
    '''Computes spectral sub-band energy by summing squared magnitude values of STFT over specified spectral band (suitable for vocal onset detection).

    Parameters:
        X (2d array): STFT of an audio signal x
        fs (int or float): sampling rate
        band (list or tuple or 1d array): edge frequencies (in Hz) of the sub-band of interest
        
    Returns:
        sbe (np array): array with each value representing the magnitude STFT values in a short-time frame squared & summed over the sub-band
    '''

    #convert band edge frequencies to bin numbers
    binLow = int(np.ceil(band[0]*X.shape[0]/(fs/2)))
    binHi = int(np.ceil(band[1]*X.shape[0]/(fs/2)))

    #compute sub-band energy
    sbe = np.sum(np.abs(X[binLow:binHi])**2, 0)

    return sbe

def biphasicDerivative(x, hopDur, norm=True, rectify=True):
    '''Computes the biphasic derivative of a signal(See [1] for a detailed explanation of the algorithm).
    
    Parameters:
        x (1d array): input signal
        hopDur (float): sampling interval in seconds of input signal (reciprocal of sampling rate of x)
        norm (bool): if output is to be normalized
        rectify (bool): if output is to be rectified to keep only positive values (sufficient for peak-picking)
    
    Returns:
        x (1d array): after computing the biphasic derivative of input (i.e, convolving with a biphasic derivative filter)

    '''

    #sampling instants
    n = np.arange(-0.1, 0.1, hopDur)

    #filter parameters (see [1] for explanation)
    tau1 = 0.015  # -ve lobe width
    tau2 = 0.025  # +ve lobe width
    d1 = 0.02165  # -ve lobe position
    d2 = 0.005  # +ve lobe position

    #filter
    A = np.exp(-pow((n-d1)/(np.sqrt(2)*tau1), 2))/(tau1*np.sqrt(2*np.pi))
    B = np.exp(-pow((n+d2)/(np.sqrt(2)*tau2), 2))/(tau2*np.sqrt(2*np.pi))
    biphasic = A-B

    #convolve with input and invert
    x = np.convolve(x, biphasic, mode='same')
    x = -1*x

    #normalise and rectify
    if norm:
        x/=np.max(x)
        x-=np.mean(x)

    if rectify:
        x*=(x>0)

    return x

def toDB(x, C):
    '''Applies logarithmic (base 10) transformation (based on [1])
    
    Parameters:
        x (1d array): input signal
        C (int or float): scaling constant
    
    Returns:
        log-scaled x
    '''
    return np.log10(1 + x*C)/(np.log10(1+C))

def getOnsetActivation(x=None, audioPath=None, startTime=0, duration=None, fs=16000, winSize=0.4, hopSize=0.01, nFFT=1024, source='vocal'):
    '''Computes onset activation function from audio signal using short-time spectrum based methods.

    Parameters:
        x (1d array): audio signal array
        audioPath (str): path to the audio file
        startTime (int or float): time to start reading the audio at
        duration (int or float): duration of audio to read
        fs (int or float): sampling rate to read audio at
        winSize (int or float): window size in seconds for STFT
        hopSize (int or float): hop size in seconds for STFT
        nFFT (int): DFT size
        source (str): choice of instrument - one of 'vocal' or 'perc' (percussion)

    Returns:
        odf (1d array): the frame-wise onset activation function (at a sampling rate of 1/hopSize)
        onsets (1d array): time locations of detected onset peaks in the odf (peaks detected using peak picker from librosa)
    '''

    #convert short-time analysis parameters from seconds to frames
    winSize = int(np.ceil(winSize*fs))
    hopSize = int(np.ceil(hopSize*fs))
    nFFT = int(2**(np.ceil(np.log2(winSize))))
    
    #if audio signal is provided
    if x is not None:
        #select segment of interest from audio based on start time and duration
        if duration is None:
            duration = len(x)/fs - startTime
        x = x[int(np.ceil(startTime*fs)):int(np.ceil(startTime+duration))]

    #if audio path is provided
    elif audioPath is not None:
        if duration is None:
            duration = librosa.get_duration(audioPath, sr=fs)
            duration = math.floor(duration)
        x,_ = librosa.load(audioPath, sr=fs, offset=startTime, duration=duration)

    else:
        print('Provide either the audio signal or path to the stored audio file on disk')
        raise

    #fade in and out ends of audio to prevent spurious onsets due to abrupt start and end
    x = fadeIn(x,int(0.5*fs))
    x = fadeOut(x,int(0.5*fs))

    #compute magnitude STFT
    X,_ = librosa.magphase(librosa.stft(x,win_length=winSize, hop_length=hopSize, n_fft=nFFT))

    #use sub-band energy -> log transformation -> biphasic filtering, if vocal onset detection [1]
    if source=='vocal':
        sub_band = [600,2400] #Hz
        odf = subBandEner(X, fs, sub_band)
        odf = toDB(odf, 100)
        odf = biphasicDerivative(odf, hopSize/fs, norm=True, rectify=True)

        #get onset locations using librosa's peak-picking function
        onsets = librosa.onset.onset_detect(onset_envelope=odf.copy(), sr=fs, hop_length=hopSize, pre_max=4, post_max=4, pre_avg=6, post_avg=6, wait=50, delta=0.12)*hopSize/fs

    #use spectral flux method (from FMP notebooks [2])
    elif source=='perc':
        sub_band = [0,fs/2] #full band
        odf = fmp.spectral_flux(x, Fs=fs, N=nFFT, W=winSize, H=hopSize, M=20, band=sub_band)
        onsets = librosa.onset.onset_detect(onset_envelope=odf, sr=fs, hop_length=hopSize, pre_max=1, post_max=1, pre_avg=1, post_avg=1, wait=10, delta=0.05)*hopSize/fs

    return odf, onsets
    
def fadeIn(x, length):
    '''
    Apply fade-in to the beginning of an audio signal using a hanning window.
    
    Parameters:
        x (1d array): signal
        length (int): length of fade (in samples)
    
    Returns:
        faded-in signal (1d array)
    '''
    x[:length] *= np.hanning(2*length)[:length]
    return x

def fadeOut(x, length):
    '''
    Apply fade-out to the end of an audio signal using a hanning window.
    
    Parameters:
        x (1d array): signal
        length (int): length of fade (in samples)
    
    Returns:
        faded-out signal (1d array)
    '''
    x[-length:] *= np.hanning(2*length)[length:]
    return x

def autoCorrelationFunction(x, fs, maxLag, winSize, hopSize):
    '''
    Compute short-time autocorrelation of a signal and normalise every frame by the maximum correlation value.
    
    Parameters:
        x (1d array): input signal
        fs (int or float): sampling rate
        maxLag (int or float): maximum lag in seconds upto which correlation is to be found (ACF is computed for all unit sample shift values lesser than this limit)
        winSize (int or float): length in seconds of the signal selected for short-time autocorrelation
        hopSize (int or float): hop duration in seconds between successive windowed signal segments (not the same as lag/shift)
    
    Returns:
        short-time ACF matrix (2d array of shape = (#frames, #lags))
    '''

    #convert parameters to frames from seconds
    n_ACF_lag = int(maxLag*fs)
    n_ACF_frame = int(winSize*fs)
    n_ACF_hop = int(hopSize*fs)

    #split input signal into windowed segments
    x = subsequences(x, n_ACF_frame, n_ACF_hop)

    #compute ACF for each windowed segment
    ACF = np.zeros((len(x), n_ACF_lag))
    for i in range(len(ACF)):
        ACF[i][0] = np.dot(x[i], x[i])
        for j in range(1, n_ACF_lag):
            ACF[i][j] = np.dot(x[i][:-j], x[i][j:])

    #normalise each ACF vector (every row) by max value
    for i in range(len(ACF)):
        if max(ACF[i])!=0:
            ACF[i] = ACF[i]/max(ACF[i])

    return ACF

def subsequences(x, winSize, hopSize):
    '''
    Split signal into shorter windowed segments with a specified hop between consecutive windows.
    
    Parameters:
        x (1d array): input signal
        winSize (int or float): size of short-time window in seconds
        hopSize (int or float): hop duration in seconds between consecutive windows
    
    Returns:
        matrix containing windowed segments (2d array)
    '''

    #pre-calculate shape of output numpy array (#rows based on #windows obtained using provided window and hop sizes)
    shape = (int(1 + (len(x) - winSize)/hopSize), winSize)

    strides = (hopSize*x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

def tempoPeriodLikelihood(ACF, norm=True):
    '''
    Compute from ACF, the likelihood of each ACF lag being the time-period of the tempo. Likelihood is obtained by taking a dot product between the ACF vector and a comb-filter (see [3] for details).
    
    Parameters:
        ACF (2d array): short-time ACF matrix of a signal
        norm (bool):
    
    Returns:
        matrix with a likelihood value for each candidate lag in each time frame (2d array of same dim as input)
    '''
    L = np.shape(ACF)[1]
    N_peaks = 11 # ?

    window = zeros((L, L))
    for j in range(L):
        C = j*np.arange(1, N_peaks)
        D = np.concatenate((C, C+1, C-1, C+2, C-2, C+3, C-3))
        D = D[D<L]
        norm_factor = len(D)
        if norm:
            window[j][D] = 1.0/norm_factor
        else:
            window[j][D] = 1.0
            
    tempo_period_candidates = np.dot(ACF, transpose(window))
    return tempo_period_candidates

def viterbiSmoothing(tempoPeriodLikelihood, hopDur, transitionPenalty, tempoRange=(30,100)):
    '''
    Apply viterbi smoothing on tempo period (lag) likelihood values to find optimum sequence of tempo values across audio frames (based on [3]).
    
    Parameters:
        tempoPeriodLikelihood (2d array): matrix of likelihood values at each tempo period (lag)
        hopDur (int or float): short-time analysis hop duration in seconds between samples of ACF vector (not hop duration between windowed signal segments taken for ACF)
        transitionPenalty (int or float): multiplicative factor applied to magnitude of change in tempo between frames; high value penalises larger jumps more, suitable for metric tempo that changes gradually across a concert and not abruptly
        tempoRange (tuple or list): expected min and max tempo in BPM
    '''

    #convert tempo range to tempo period (in frames) range
    fs = 1/hopDur
    tempoRange *= (fs/60)

    #initialise cost matrix with very high values
    T,L = np.shape(tempoPeriodLikelihood)
    cost = ones((T,L//2))*1e8

    #
    m = zeros((T,L//2))

    #compute cost value at each lag (within range), at each time frame
    for i in range(1,T):
        for j in range(1,L//2):
            cost[i][j]=cost[i-1][1]+transitionPenalty*abs(60.0*fs/j-60.0*fs)-tempoPeriodLikelihood[i][j]
            for k in range(2,L//2):
                if cost[i][j]>cost[i-1][k]+transitionPenalty*abs(60.0*fs/j-60.0*fs/k)-tempoPeriodLikelihood[i][j]:
                    cost[i][j]=cost[i-1][k]+transitionPenalty*abs(60.0*fs/j-60.0*fs/k)-tempoPeriodLikelihood[i][j]
                    m[i][j]=int(k)
                    
    tempo_period_smoothed=zeros(T)
    tempo_period_smoothed[T-1]=argmin(cost[T-1,1:])/float(fs)
    t=int(m[T-1,argmin(cost[T-1,1:])])
    i=T-2
    while(i>=0):
        tempo_period_smoothed[i]=t/float(fs)
        t=int(m[i][t])
        i=i-1
    return tempo_period_smoothed

'''
References
[1] Rao, P., Vinutha, T.P. and Rohit, M.A., 2020. Structural Segmentation of Alap in Dhrupad Vocal Concerts. Transactions of the International Society for Music Information Retrieval, 3(1), pp.137–152. DOI: http://doi.org/10.5334/tismir.64
[2] Meinard Müller and Frank Zalkow: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing. Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
[3] T.P. Vinutha, S. Suryanarayana, K. K. Ganguli and P. Rao " Structural segmentation and visualization of Sitar and Sarod concert audio ", Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR), Aug 2016, New York, USA
'''
