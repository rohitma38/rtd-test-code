from tkinter import E
import warnings
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import is_color_like
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
import os
import cv2
from collections import defaultdict
import utils_fmp as fmp

#set seaborn theme parameters for plots
sns.set_theme(rc={"xtick.bottom" : True, "ytick.left" : False, "xtick.major.size":4, "xtick.minor.size":2, "ytick.major.size":4, "ytick.minor.size":2, "xtick.labelsize": 10, "ytick.labelsize": 10})

# HELPER FUNCTION
def __check_axes(axes):
    """Check if "axes" is an instance of an axis object. If not, use `plt.gca`.
    
    This function is a modified version from [#]_.

    .. [#] McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. “librosa: Audio and music signal analysis in python.” In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

    """
    if axes is None:
        axes = plt.gca()
        
    elif not isinstance(axes, matplotlib.axes.Axes):
        raise ValueError(
            "`axes` must be an instance of matplotlib.axes.Axes. "
            "Found type(axes)={}".format(type(axes))
        )
    return axes

# ANNOTATION FUNCTION
def readCycleAnnotation(cyclePath, numDiv, startTime, duration, timeCol='Time', labelCol='Cycle'):
    '''Function to read cycle annotation and add divisions between markings if required.

    This function reads the timestamps of provided annotations and adds `numDiv` 'computed' annotations between each provided annotation.

    Parameters
    ----------
        cyclePath    : str
            Path to the cycle annotation csv.

        numDiv    : int
            Number of equally spaced divisions to add between consecutive provided annotations (numDiv - 1 timestamps will be added between each consecutive pair).

        startTime    : float; default=0
            Start time of audio being analysed.

        duration    : float or None; default=None
            Duration of the audio to be analysed.

        timeCol    : str
            Column name of timestamps in cycle annotation file.

        labelCol    : str
            Column name of labels for the annotations in the annotation file.


    Returns
    -------
        provided    : list of one pd.DataFrame element
            Data frame contains the time stamps and respective labels of all provided annotations (i.e. annotations from the annotation csv). 
            
            It is provided in a list to match the output format of the `readOnsetAnnotation`.

        computed    : list 
            List of timestamps of computed annotations (i.e. annotations computed between consecutive provided annotations).

            If `numDiv` is 0 or number of provided annotations is 1, an empty list is returned 

        .. note::
            If there are no provided annotations present during the relevant duration of audio, the function will return (None, None)
    '''
    cycle_df = pd.read_csv(cyclePath)
    index_values = cycle_df.loc[(cycle_df[timeCol] >= startTime) & (cycle_df[timeCol] <= startTime + duration)].index.values
    if len(index_values) == 0:
        return None, None
    provided = cycle_df.iloc[max(index_values[0]-1, 0):min(index_values[-1]+2, cycle_df.shape[0])]     # filter out rows from annotation file that fall within the considered time duration
    provided = provided.loc[:, [timeCol, labelCol]]     # retain only the time and label columns from the data frame
    # add divisions in the middle
    computed = []
    for ind, val in enumerate(provided[timeCol].values[:-1]):
        computed.extend(np.around(np.linspace(val, provided['Time'].values[ind+1], num = numDiv, endpoint=False), 2)[1:])
    return [provided], computed

# ANNOTATION FUNCTION
def readOnsetAnnotation(onsetPath, startTime, duration, timeCol=['Time'], onsetKeyword=['Inst']):
    '''Function to read onset annotations.

    Reads an onset annotation csv file and returns the timestamps and annotation labels for annotations within a given time duration.

    Parameters
    ----------
        onsetPath    : str
            Path to the onset annotation csv file.
        
        startTime    : float; default=0
            Start time of audio being analysed.
        
        duration    : float or None; default=None
            Duration of the audio to be analysed.
        
        timeCol    : str
            Column name of timestamps in onset annotation file. #TODO: fix this; maybe make it a list or make it simpler#

        onsetKeyword    : list or None
            List of column names in the onset file to take onset labels from. For each onsetKeyword, a separate dataframe with onset annotations will be returned.

    Returns
    -------
        provided    : list of pd.DataFrames
            List of data frames with each element corresponding to a keyword from `onsetKeyword`.

            If `onsetKeyword` is None, it will return only annotation time stamps.

            If no onsets are present in the given time duration, None is returned.
    '''
    onset_df = pd.read_csv(onsetPath)
    provided = []   # variable to store onset timestamps
    if onsetKeyword is None:
        # if onsetKeyword is None, return only timestamps
        return [onset_df.loc[(onset_df[timeCol[0]] >= startTime) & (onset_df[timeCol] <= startTime + duration), timeCol[0]]]
    for keyword in onsetKeyword:
        provided.append(onset_df.loc[(onset_df[timeCol[0]] >= startTime) & (onset_df[timeCol[0]] <= startTime + duration), [timeCol[0], keyword]])
    return provided if len(provided) > 0 else None     # return None if no elements are in provided

# ANNOTATION FUNCTION
def drawAnnotation(cyclePath=None, onsetPath=None, onsetTimeKeyword=None, onsetLabelKeyword=None, numDiv=0, startTime=0, duration=None, ax=None, annotLabel=True, c='purple', alpha=0.8, y=0.7, size=10, textColour='white'):
    '''Draws annotations on ax

    Plots annotation labels on `ax` if provided, else creates a new matplotlib.axes.Axes object and adds the labels to that.     

    Parameters
    ----------
        cyclePath    : str
            Path to the cycle annotation csv, used for tala-related annotations.

        onsetPath    : str
            Path to onset annotations, used for non-tala related annotations (example: syllable or performance related annotations).
            
            These annotations are only considered if cyclePath is None.
        
        onsetTimeKeyword    : str
            Column name in the onset file to take onset timestamps from.

            If `onsetLabelKeyword` is a list, the same column is used to determine timestep for every `onsetLabelKeyword` value(s).

            If a list is provided, length of the list should be equal to the length of `onsetLabelKeyword` and `c`. #TODO: reduce the if else statements to make it easier #

        onsetLabelKeyword    : str or list or None
            Column name(s) in the onset file to take annotation labels from. 
            
            If `str` is provided, labels will be drawn only for values from the specified column name. For plotting tala-related annotations with `cyclePath`, only `str` format is valid.

            If `list` is provided, labels will be drawn for each column name in the list. The length of the list should be equal to length of `c`.
            
            If `annotLabel` is False, then `onsetLabelKeyword` can be None.

        numDiv    : int >= 0, default=0
            Number of equally spaced divisions to add between consecutive pairs of annotations (numDiv - 1 timestamps will be added between each pair).

            Used only if `cyclePath` is not None. 

        startTime    : float >= 0, default=0
            Starting timestamp from which to analyse the audio.
        
        duration    : float >= 0 or None
            Duration of audio to be analysed.

            If None, it will analyse the entire audio length.

        ax    : matplotlib.axes.Axes or None
            matplotlib.axes.Axes object to plot in.

            If None, will use `plt.gca()` to use the current matplotlib.axes.Axes object.

        annotLabel    : bool, default=True
            If True, will print annotation label along with a vertical line at the annotation time stamp

            If False, will just add a vertical line at the annotation time stamp without the label.

        c    : color or list (of colors)
            Value is passed as parameter `c` to `plt.axvline()`.

            If a list of colors is provided, one color corresponds to one column name in `onsetLabelKeyword`.::

                len(onsetLabelKeyword) == len(c)

            If cyclePath is not None, c cannot be of type list. Only one value must be provided.
            
        alpha    : scalar or None
            Controls opacity of the annotation lines drawn. Value must be within the range 0-1, inclusive.

            Passed to `plt.axvline()` as the `alpha` parameter.

        y    : float
            Float value from 0-1, inclusive. 
            
            Indicates where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

        size    : int
            Font size for annotated text. Passed as `fontsize` parameter to `matplotlib.axes.Axes.annotate()`.

        textColour    : str or list
            Text colour for annotation. 
            
            Can be a single string or a list of strings for each onsetLabelKeyword.

            If cyclePath is not None, only a single string value is valid.

    Returns
    -------
        ax    : matplotlib.Axes.axis
            axis that has been plotted in

    Raises
    ------
        ValueError
            Raised in any of the following scenarios
                1. If cyclePath is not None and onsetLabelKeyword and/or onsetTimeKeyword is not of types {str, None}

                2. If `onsetPath` is not None and `cyclePath` is None and any of the following
                    a. If `onsetTimeKeyword` is anything but type `str`
                    b. If `onsetLabelKeyword` is None and any of the following
                        i. annotLabel is True
                        ii. c is not of type color
                    c. If `onsetLabelKeyword` is type str and any of the following
                        i. One or more of {onsetTimeKeyword, c, textColour} is not of type str
                    d. If `onsetLabelKeyword` is type list and any of the following
                        i. `c` is a list and length of `c` is not equal to length of `onsetLabelKeyword`.
                        ii. `textColor` is a list and length of `textColor` is not equal to length of `textColor`.
    '''
    if cyclePath is not None:
        if onsetTimeKeyword is None and onsetLabelKeyword is None:
            # time keyword has not been provided, use default timeCol from `readCycleAnnotation`
            provided, computed = readCycleAnnotation(cyclePath, numDiv, startTime, duration)
            timeCol = ['Time']    # name of column with time readings
            labelCol = ['Cycle']     # name of column with label values
        elif onsetTimeKeyword is not None and onsetLabelKeyword is None:
            # if only label keyword is not provided, use default label keyword
            if isinstance(onsetTimeKeyword, str):
                provided, computed = readCycleAnnotation(cyclePath, numDiv, startTime, duration, timeCol=onsetTimeKeyword)
                timeCol = [onsetTimeKeyword]
                labelCol = ['Cycle']  # name of column to extract label of annotation from
            else:
                raise ValueError(f"onsetTimeKeyword has to be of type str or None when cyclePath is not None. Invalid onsetTimeKeyword type: {type(onsetTimeKeyword)}")
        elif onsetTimeKeyword is None and onsetLabelKeyword is not None:
            # if only time keyword is not provided, use default timeCol from `readCycleAnnotation`.
            if isinstance(onsetLabelKeyword, str):
                provided, computed = readCycleAnnotation(cyclePath, numDiv, startTime, duration, labelCol=onsetLabelKeyword)
                timeCol = ['Time']  # name of column to extract timestamps of annotation from
                labelCol = [onsetLabelKeyword]
            else:
                raise ValueError(f"onsetLabelKeyword has to be of type str or None when cyclePath is not None. Invalid onsetLabelKeyword type: {type(onsetLabelKeyword)}")
        else:
            # both onsetTimeKeyword and onsetLabelKeyword are provided.
            if isinstance(onsetTimeKeyword, str) and isinstance(onsetLabelKeyword, str):
                provided, computed = readCycleAnnotation(cyclePath, numDiv, startTime, duration, timeCol=onsetTimeKeyword, labelCol=onsetLabelKeyword)
                timeCol = [onsetTimeKeyword] 
                labelCol = [onsetLabelKeyword]
            else:
                raise ValueError(f"onsetTimeKeyword and onsetLabelKeyword have to be of type str or None when cyclePath is not None. Invalid type(s) of \n1. onsetTimeKeyword type: {type(onsetTimeKeyword)}\n2. onsetLabelKeyword type: {type(onsetLabelKeyword)}")
        
        if isinstance(textColour, str):
            textColours = [textColour]  # colour of text
        else:
            raise ValueError(f"textColour has to be of type str or None when cyclePath is not None. Invalid textColour type: {type(textColour)}")

        if isinstance(c, str):    
            c = [c]  # colour of text
        else:
            raise ValueError(f"c has to be of type str or None when cyclePath is not None. Invalid c type: {type(c)}")
        
    elif onsetPath is not None:
        if onsetTimeKeyword is None:
            timeCol = ['Inst'] # TODO: make this accomodate lists also, for fig 9 #
        else:
            timeCol = onsetTimeKeyword
        if onsetLabelKeyword is None: #TODO: onsetLabelKeyword should be able to take value None also (look at fig 9)#
            labelCol = ['Label']
            textColours = [textColour] if isinstance(textColour, str) else textColour
            c = [c] if isinstance(c, str) else c #TODO fix this #
        # if onsetLabelKeyword is None:
        #     if not annotLabel:
        #         # onsetLabelKeyword can be None only is annotLabel is False
        #         timeCol = [onsetTimeKeyword]
        #         if not is_color_like(c):
        #             # if c is not of type color, then rais error
        #             raise ValueError(f"Invalid type of c: {type(c)}. With onsetPath and annotLabel=False, c has to be of type color.")
        #     else:
        #         # if annotLabel is True, onsetLabelKeyword cannot be None
        #         raise ValueError(f"Invalid type of onsetLabelKeyword. With annotLabel = {annotLabel}, onsetLabelKeyword cannot be {onsetLabelKeyword}") # TODO: fix this and make it more readable#
        elif isinstance(onsetLabelKeyword, str):
            # onsetLabelKeyword is str
            if not (isinstance(onsetTimeKeyword, str) and isinstance(c, str) and isinstance(textColour, str)):
                # if either onsetTimeKeyword, c or textColour is not str
                raise ValueError(f"When using onsetPath, if onsetLabelKeyword is str, onsetTimeKeyword (current type: {type(onsetTimeKeyword)}), c (current type: {type(c)} and textColor (current type: {type(textColour)} have to be of type str.")
            else:
                timeCol = [onsetTimeKeyword]
                labelCol = [onsetLabelKeyword]
                textColours = [textColour]
                c = [c]
        elif isinstance(onsetLabelKeyword, list):
            # onsetLabelKeyword is list
            if isinstance(onsetTimeKeyword, str):
                # if onsetTimeKeyword is a str, duplicate the value into a list of len(onsetLableKeyword)
                timeCol = [onsetTimeKeyword for _ in range(len(onsetLabelKeyword))]
            else:
                raise ValueError(f"When using onsetPath, if onsetLabelKeyword is list, onsetTimeKeyword has to be of type str, not type: {type(onsetTimeKeyword)})")
            
            if is_color_like(c):
                # if c is a color, duplicate the value into a list of len(onsetLabelKeyword)
                c = [c for _ in range(len(onsetLabelKeyword))]
            elif isinstance(c, list):
                if not len(c) == len(onsetLabelKeyword):
                    # if len(c) is not equal to len(onsetLabelKeyword)
                    raise ValueError(f"Length of c: {len(c)} is not equal to length of onsetLabelKeyword: {len(onsetLabelKeyword)}")
            else:
                raise ValueError(f"When using onsetPath, if onsetLabelKeyword is list, c has to be of type color or list, not type: {type(c)})")

            if isinstance(textColour, str):
                # if textColour is a color, duplicate the value into a list of len(onsetLableKeyword)
                textColours = [textColour for _ in range(len(onsetLabelKeyword))]
            elif isinstance(textColour, list):
                textColours = textColour
            else:
                raise ValueError(f"When using onsetPath, if onsetLabelKeyword is list, textColour has to be of type color or list, not type: {type(textColour)})")
        
        else:
            # onsetLabelKeyword is not str or list or None
            raise ValueError(f"When using onsetPath, if annotLabel is True, onsetLabelKeyword has to be of type str or list, not {type(onsetLabelKeyword)}.")

        provided = readOnsetAnnotation(onsetPath, startTime, duration, timeCol=timeCol, onsetKeyword=labelCol)
        computed = None
    else:
        raise Exception('A cycle or onset path has to be provided for annotation')

    # check if ax is None and use current ax if so
    ax = __check_axes(ax)

    if computed is not None:
        # plot computed annotations, valid only when `cyclePath` is not None
        for computedVal in computed:
            ax.axvline(computedVal, linestyle='--', c=c[0], alpha=0.4)
    if provided is not None:
        # plot the annotations from the file
        for i, providedListVal in enumerate(provided):
            firstLabel = True   # marker for first line for each value in  onsetLabelKeyword being plotted; to prevent duplicates from occuring in the legend
            for _, providedVal in providedListVal.iterrows():
                ax.axvline((providedVal[timeCol[i]]), linestyle='-', c=c[i], label=labelCol[i] if firstLabel and cyclePath is None else '', alpha=alpha)  # add label only for first line of onset for each keyword
                if firstLabel:  firstLabel = False     # make firstLabel False after plotting the first line for each value in onsetLabelKeyword
                if annotLabel:
                    ylims = ax.get_ylim()   # used to set label at a height defined by `y`.
                    if isinstance(providedVal[labelCol[i]], str):
                        ax.annotate(f"{providedVal[labelCol[i]]}", (providedVal[timeCol[i]], (ylims[1]-ylims[0])*y + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c=textColours[i], fontsize=size)
                    else:
                        ax.annotate(f"{float(providedVal[labelCol[i]]):g}", (providedVal[timeCol[i]], (ylims[1]-ylims[0])*y + ylims[0]), bbox=dict(facecolor='grey', edgecolor='white'), c=textColours[i], fontsize=size)
    if onsetPath is not None and cyclePath is None:     # add legend only is onsets are given, i.e. legend is added
        ax.legend()
    return ax

# COMPUTATION FUNCTION
def pitchContour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, maxPitch=660, notes=None, tonic=220, timeStep=0.01, octaveJumpCost=0.9, veryAccurate=True, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword=None, onsetLabelKeyword=None, xticks=False, yticks=False, xlabel=True, ylabel=True, title='Pitch Contour (Cents)', annotLabel=True, cAnnot='purple', ylim=None, annotAlpha=0.8):
    '''Returns pitch contour (in cents) for the audio

    Calculates the pitch contour of a given audio sample using autocorrelation method described in _[#]. The implementation of the algorithm is done using [#]_ and it's Python API _[#]. The pitch contour is converted to cents by making the tonic correspond to 0 cents.

    ..[#] Paul Boersma (1993): "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound." Proceedings of the Institute of Phonetic Sciences 17: 97–110. University of Amsterdam.Available on http://www.fon.hum.uva.nl/paul/

    ..[#] Boersma, P., & Weenink, D. (2021). Praat: doing phonetics by computer [Computer program]. Version 6.1.38, retrieved 2 January 2021 from http://www.praat.org/

    ..[#] Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

    The audio signal is given in mono format to the pitch detection algorithm.

    Uses `plotPitch` to plot pitch contour if `ax` is not None.

    Parameters
    ----------
        audio    : ndarray or None
            Loaded audio time series

            Audio signal is converted to mono to compute the pitch.

        sr    : number > 0; default=16000
            If audio is not None, defines sample rate of audio time series 

            If audio is None and audioPath is not None, defines sample rate to load the audio at

        audioPath    : str
            Path to audio file. 

            Used only if audio is None. Audio is loaded as mono.

        startTime    : float; default=0
            Time stamp to consider audio from

        duration    : float or None; default=None
            Duration of the audio to consider

        minPitch    : float; default=98
            Minimum pitch (in Hz) to read for contour extraction.

            Passed as `pitch_floor` parameter to `parselmouth.Sound.to_pitch_ac()`.

        maxPitch    : float
            Maximum pitch to read for contour extraction.

            Passed as `pitch_ceil` parameter to `parselmouth.Sound.to_pitch_ac()`.

        notes    : list
            list of dictionaries with keys ``cents`` and ``label`` for each note present in the raga of the audio.

        tonic    : float
            Tonic of the audio (in Hz).

            Used to compute the pitch contour in cents.

        timeStep    : float; default=0.01
            Time steps (in seconds) in which pitch values are extracted.::

                Example: timeStep = 0.01 implies that pitch values are extracted for every 0.01 s.

        octaveJumpCost    : float
            Degree of disfavouring of pitch changes, relative to maximum possible autocorrelation.

            Passed as `octave_jump_cost` parameter to `praat.Sound.to_pitch_ac()`.

        veryAccurate    : bool
            Passed as `very_accurate` parameter to `praat.Sound.to_pitch_ac()`.

        ax    : matplotlib.axes.Axes or None
            Axis to plot the pitch contour in.

        freqXlabels    : float
            Time (in seconds) after which each x label occurs in the plot

        annotate    : bool
            If True, will annotate markings in either cyclePath or onsetPath with preference to cyclePath.

        cyclePath    : str or None
            Path to file with tala cycle annotations.

            Passed to `drawAnnotation()`.

        numDiv    : int >= 0
            Number of divisions to put between each annotation marking in cyclePath. Used only if cyclePath is not None.

            Passed to `drawAnnotation()`.

        onsetPath    : str or None
            Path to file with onset annotations. Only considered if cyclePath is None.

            Passed to `drawAnnotation()`.

        onsetTimeKeyword    : str
            Column name in the onset file to take time stamps of onsets from.

            Passed to `drawAnnotation()`.

        onsetLabelKeyword    : str or list or None
            Column name with label(s) for the onsets. If None, no label will be printed.

            Passed to `drawAnnotation()`.

        xticks    : bool
            If True, will add xticklabels to plot.

            Passed to `plotPitch()`.

        yticks    : bool
            If True, will add yticklabels to plot.

            Passed to `plotPitch()`.

        xlabel    : bool
            If True, will print xlabel in the plot.

            Passed to `plotPitch()`.

        ylabel    : bool
            If True will print ylabel in the plot.

            Passed to `plotPitch()`.

        title    : str
            Title to add to the plot.

            Passed to `plotPitch()`.

        annotLabel    : bool
            If True, will print annotation label along with the annotation line. Used only if annotate is True.

            Passed to `plotPitch()`.

        cAnnot: color 
            Determines the colour of annotion. Input to the `matplotlib.pyplot.annotate()` function for the `c` parameter.
            
            Passed to `plotPitch()`.

        ylim    : (float, float) or None
            (min, max) limits for the y axis; if None, will be directly interpreted from the data

        annotAlpha    : (float)
            Controls opacity of the annotation lines in the plot. Value has to be in the range [0, 1], inclusive.

    Returns
    -------
        ax : matplotlib.axes.Axes
            Plot of pitch contour if `ax` was not None

        (pitchvals, timevals)    : (ndarray, ndarray)
            Tuple with arrays of pitch values (in cents) and time stamps. Returned if ax was None.

    '''
    
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        # if audio is not given, load audio from audioPath
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    else:
        # if audio is provided, check that it is in mono and convert it to mono if it isn't
        audio = librosa.to_mono(audio)

    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        # duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot #TODO: test and remove #
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration

    snd = parselmouth.Sound(audio, sr)
    pitch = snd.to_pitch_ac(time_step=timeStep, pitch_floor=minPitch, very_accurate=veryAccurate, octave_jump_cost=octaveJumpCost, pitch_ceiling=maxPitch)     # extracting pitch contour (in Hz)

    pitchvals = pitch.selected_array['frequency']
    pitchvals[pitchvals==0] = np.nan    # mark unvoiced regions as np.nan
    if tonic is None:   raise Exception('No tonic provided')
    pitchvals[~(np.isnan(pitchvals))] = 1200*np.log2(pitchvals[~(np.isnan(pitchvals))]/tonic)    # convert Hz to cents
    timevals = pitch.xs() + startTime
    if ax is None:
        warnings.warn('`ax` is None. Returning pitch and time values.')
        return (pitchvals, timevals)
    else:
        # plot the contour
        return plotPitch(pitchvals, timevals, notes=notes, ax=ax, startTime=startTime, duration=duration, freqXlabels=freqXlabels, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel, title=title, cAnnot=cAnnot, annotLabel=annotLabel, ylim=ylim, annotAlpha=annotAlpha)

# PLOTTING FUNCTION
def plotPitch(pitchvals=None, timevals=None, notes=None, ax=None, startTime=0, duration=None, freqXlabels=5, xticks=True, yticks=True, xlabel=True, ylabel=True, title='Pitch Contour (Cents)', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword=None, onsetLabelKeyword=None, cAnnot='purple', annotLabel=True, ylim=None, annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Plots the pitch contour

    Plots the pitch contour passed in the `pitchvals` parameter, computed from `pitchContour()` function.

    Parameters
    ----------
        pitchvals    : ndarray
            Pitch values (in cents).

            Computed from `pitchContour()`

        timevals    : ndarray
            Time stamps (in seconds) corresponding to each value in `pitchvals`.

            Computed from `pitchContour()`.

        notes    : list
            list of dictionaries with keys ``cents`` and ``label`` for each note present in the raga of the audio.

        ax    : matplotlib.axes.Axes or None
            Object on which pitch contour is to be plotted

            If None, will plot in `matplotlib.pyplot.gca()`.

        startTime    : float >= 0
            Offset time (in seconds) from where audio is analysed.

            Sent to `drawAnnotation()`.

        duration    : float >= 0
            Duration of audio in the plot.

            Sent to `drawAnnotation()`.

        freqXlabels    : int
            Time (in seconds) after which each x ticklabel should occur

        xticks    : bool
            If True, will print x ticklabels in the plot.

        yticks    : bool
            If True, will print y ticklabels in the plot.

        xlabel    : bool
            If True, will add label to x axis

        ylabel    : bool
            If True, will add label to y axis

        title    : str
            Title to add to the plot

        annotate    : bool
            If True, will add tala-related/onset annotations to the plot 

        cyclePath    : bool
            Path to csv file with tala-related annotations.
            
            If annotate is True, sent to `drawAnnotation()`.

        numDiv    : int
            Number of divisions to add between each tala-related annotation provided.

            If annotate is True, sent to `drawAnnotation()`.

        onsetPath    : str
            Path to file with onset annotations

            If annotate is True, sent to `drawAnnotation()`. Used only if `cyclePath` is None.

        onsetTimeKeyword    : str
            Column name in the onset file to take onset's timestamps from.

            If annotate is True, sent to `drawAnnotation()`. Used only if `cyclePath` is None.

        onsetLabelKeyword    : str or list or None
            Column name(s) in onsetPath file with labels values for the onsets.
            
            If None, no label will be printed for the onsets. 

            If annotate is True, sent to `drawAnnotation()`. Used only if `cyclePath` is None.

        cAnnot    : color
            Determines the colour of annotation. Sent as input to the `matplotlib.pyplot.annotate()` function for the colour (`c`) parameter.
            
            If `annotate` is True, sent to `drawAnnotation()`.
            
        annotLabel    : bool
            If True, will print annotation label along with line.

            If `annotate` is True, sent to `drawAnnotation()`.

        ylim    : (float, float) or None
            (min, max) limits for the y axis.
            
            If None, will be directly interpreted from the data.

        annotAlpha    : float >= 0
            Controls opacity of the annotation line drawn. Value should range from 0-1, both inclusive

            If `annotate` is True, sent to `drawAnnotation()`.

        yAnnot    : float
            Value ranging from 0-1, both inclusive. 
            
            Indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

            If `annotate` is True, sent to `drawAnnotation()`.

        sizeAnnot    : number 
            Font size for annotated text.

            If `annotate` is True, sent to `drawAnnotation()`.

    Returns
    -------
        ax    : matplotlib.axes.Axes
            Plot of pitch contour.

    Raises
    ------
        ValueError
            If pitchvals is None.
    '''

    # Check that all required parameters are present
    if pitchvals is None:
        ValueError('No pitch contour provided')
    if timevals is None:
        warnings.warn('No time values provided, assuming 0.01 s time steps in pitch contour')
        timevals = np.arange(startTime, len(pitchvals)*0.01, 0.01)
    # Added below block
    if duration is None:
        warnings.warn('No duration provided, assuming last time step in pitch contour as duration')
        duration = timevals[-1]

    
    # if ax is None, use the `plt.gca()` to use current axes object
    ax = __check_axes(ax)
    
    ax = sns.lineplot(x=timevals, y=pitchvals, ax=ax)
    ax.set(xlabel='Time Stamp (s)' if xlabel else '', 
    ylabel='Notes' if ylabel else '', 
    title=title, 
    xlim=(startTime, startTime+duration), 
    xticks=np.around(np.arange(math.ceil(timevals[0]), math.floor(timevals[-1]), freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
    xticklabels=np.around(np.arange(math.ceil(timevals[0]), math.floor(timevals[-1]), freqXlabels)).astype(int) if xticks else [])     # let the labels start from the integer values.
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

# COMPUTATION FUNCTION
def spectrogram(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, winSize=640, hopSize=160, nFFT=1024, cmap='Blues', ax=None, amin=1e-5, freqXlabels=5, xticks=False, yticks=False, xlabel=True, ylabel=True, title='Spectrogram', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword=None, onsetLabelKeyword=None, cAnnot='purple', annotLabel=True, ylim=(0, 5000), annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Computes spectrogram from the audio sample

    Returns a plotted spectrogram if ax is not None, else returns the computed STFT on the audio.

    Uses `librosa.display.specshow()` to plot the spectrogram.

    Parameters
    ----------
        audio    : ndarray or None
            Loaded audio time series.

        sr    : number > 0; default=16000
            If audio is not None, defines sample rate of audio time series 

            If audio is None and audioPath is not None, defines sample rate to load the audio at

        audioPath    : str
            Path to audio file. 

            Used only if audio is None. Audio is loaded as mono.

        startTime    : float; default=0
            Time stamp to consider audio from.

        duration    : float or None; default=None
            Duration of the audio to consider.

        winSize    : int > 0
            Size of window for STFT (in frames)

        hopSize    : int > 0
            Size of hop for STFT (in frames)

        nFFT    : int or None
            DFT size

            If nFFT is None, it takes the value of the closest power of 2 >= winSize (in samples).

        cmap    : matplotlib.colors.Colormap or str
            Colormap to use to plot spectrogram.

            Sent as a parameter to `plotSpectrogram`.

        ax    : matplotlib.axes.Axes or None
            Axes to plot spectrogram in. 

            If None, returns (sample frequencies, segment times, STFT) of audio sample

        amin    : float > 0
            Minimum threshold for `abs(S)` and `ref` in `librosa.power_to_db()`. Controls the contrast of the spectrogram.
            
            Passed into `librosa.power_to_db()` function.

        freqXlabels    : float
            Time (in seconds) after which each x label occurs in the plot

        xticks    : bool
            If True, will add xticklabels to plot.

            Passed to `librosa.display.specshow()`.

        yticks    : bool
            If True, will add yticklabels to plot.

            Passed to `librosa.display.specshow()`.

        xlabel    : bool
            If True, will print xlabel in the plot.

            Passed to `librosa.display.specshow()`.

        ylabel    : bool
            If True will print ylabel in the plot.

            Passed to `librosa.display.specshow()`.

        title    : str
            Title to add to the plot.

            Passed to `librosa.display.specshow()`.
        
        annotate    : bool
            If True, will annotate markings in either cyclePath or onsetPath with preference to cyclePath.

        cyclePath    : str or None
            Path to file with tala cycle annotations.

            Passed to `drawAnnotation()`.

        numDiv    : int >= 0
            Number of divisions to put between each annotation marking in cyclePath. Used only if cyclePath is not None.

            Passed to `drawAnnotation()`.

        onsetPath    : str or None
            Path to file with onset annotations. Only considered if cyclePath is None.

            Passed to `drawAnnotation()`.

        onsetTimeKeyword    : str
            Column name in the onset file to take time stamps of onsets from.

            Passed to `drawAnnotation()`.

        onsetLabelKeyword    : str or list or None
            Column name with label(s) for the onsets. If None, no label will be printed.

            Passed to `drawAnnotation()`.

        cAnnot: color 
            Determines the colour of annotion. Input to the `matplotlib.pyplot.annotate()` function for the `c` parameter.
            
            Passed to `drawAnnotation()`.

        
        annotLabel    : bool
            If True, will print annotation label along with the annotation line. Used only if annotate is True.

            Passed to `drawAnnotation()`.

        ylim    : (float, float) or None
            (min, max) limits for the y axis.
            
            If None, will be directly interpreted from the data.

        annotAlpha    : float >= 0
            Controls opacity of the annotation line drawn. Value should range from 0-1, both inclusive

            If `annotate` is True, sent to `drawAnnotation()`.

        yAnnot    : float
            Value ranging from 0-1, both inclusive. 
            
            Indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

            If `annotate` is True, sent to `drawAnnotation()`.

        sizeAnnot    : number 
            Font size for annotated text.

            If `annotate` is True, sent to `drawAnnotation()`.
        
    Returns
    -------
        ax    : matplotlib.axes.Axes or (ndarray, ndarray, ndarray)
            If `ax` is not None, returns a plot of the spectrogram computed

            If `ax` is None, returns a tuple with (sample frequencies, segment times, STFT of the audio (in dB)) computed by `scipy.signal.stft()`.
    '''
    # if ax is None:
    #     Exception('ax parameter has to be provided') #TODO replace this#
    # startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)     
        # duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot #TODO: test and remove#
        audio = audio[:int(duration*sr)]    # ensure that audio length = duration
    
    # convert winSize and hopSize from seconds to samples
    if nFFT is None:
        nFFT = int(2**np.ceil(np.log2(winSize)))     # set value of `nFFT` if it is None.

    # STFT
    f,t,X = sig.stft(audio, fs=sr, window='hann', nperseg=winSize, noverlap=(winSize-hopSize), nfft=nFFT)
    X_dB = librosa.power_to_db(np.abs(X), ref = np.max, amin=amin)
    t += startTime     # add start time to time segments extracted.

    if ax is None:
        # return f, t, X_dB
        warnings.warn('`ax` is None. Returning frequency, time and STFT values.')
        return (f, t, X_dB)

    else:
        return plotSpectrogram(X_dB, t, f, sr=sr, startTime=startTime, duration=duration, hopSize=hopSize, cmap=cmap, ax=ax, freqXlabels=freqXlabels, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel, title=title, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, cAnnot=cAnnot, annotLabel=annotLabel, ylim=ylim, annotAlpha=annotAlpha, yAnnot=yAnnot, sizeAnnot=sizeAnnot)

# PLOTTING FUNCTION
def plotSpectrogram(X_dB, t, f, sr=16000, startTime=0, duration=None, hopSize=160, cmap='Blues', ax=None, freqXlabels=5, xticks=False, yticks=False, xlabel=True, ylabel=True, title='Spectrogram', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword=None, onsetLabelKeyword=None, cAnnot='purple', annotLabel=True, ylim=(0, 5000), annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Plots spectrogram

    Uses `librosa.display.specshow()` to plot a spectrogram from a computed STFT. Annotations can be added is `annotate` is True.

    Parameters
    ----------
    X_dB    : ndarray
        STFT of audio. Computed in `spectrogram()`.

    t    : ndarray or None
        Time segments corresponding to `X_dB`.

        If None, will assign time steps from 0 (in seconds).
    
    f    : ndarray
        Frequency values. Computed in `spectrogram()`.

        If None, will infer frequency values in a linear scale.

    sr    : number > 0; default=16000
        Sample rate of audio processed in `spectrogram()`.

    startTime    : float; default=0
        Time stamp to consider audio from.

        Used to extract relavant annotation in `drawAnnotation()`.

    duration    : float or None; default=None
        Duration of the audio to consider.

        Used to extract relavant annotation in `drawAnnotation()`.

    hopSize    : int > 0
        Size of hop for STFT (in seconds)

    cmap    : matplotlib.colors.Colormap or str
        Colormap to use to plot spectrogram.

        Sent as a parameter to `plotSpectrogram`.

    ax    : matplotlib.axes.Axes or None
        Axes to plot spectrogram in. 

        If None, plots the spectrogram returned by `plt.gca()`.

    freqXlabels    : float > 0
        Time (in seconds) after which each x label occurs in the plot

    xticks    : bool
        If True, will add xticklabels to plot.

    yticks    : bool
        If True, will add yticklabels to plot.

    xlabel    : bool
        If True, will print xlabel in the plot.

    ylabel    : bool
        If True will print ylabel in the plot.

    title    : str
        Title to add to the plot.

    annotate    : bool
        If True, will annotate markings in either cyclePath or onsetPath with preference to cyclePath.

    cyclePath    : str or None
        Path to file with tala cycle annotations.

        Passed to `drawAnnotation()`.

    numDiv    : int >= 0
        Number of divisions to put between each annotation marking in cyclePath. Used only if cyclePath is not None.

        Passed to `drawAnnotation()`.

    onsetPath    : str or None
        Path to file with onset annotations. Only considered if cyclePath is None.

        Passed to `drawAnnotation()`.

    onsetTimeKeyword    : str
        Column name in the onset file to take time stamps of onsets from.

        Passed to `drawAnnotation()`.

    onsetLabelKeyword    : str or list or None
        Column name with label(s) for the onsets. If None, no label will be printed.

        Passed to `drawAnnotation()`.

    cAnnot: color 
        Determines the colour of annotion. Input to the `matplotlib.pyplot.annotate()` function for the `c` parameter.
        
        Passed to `drawAnnotation()`.

    annotLabel    : bool
        If True, will print annotation label along with the annotation line. Used only if annotate is True.

        Passed to `drawAnnotation()`.

    ylim    : (float, float) or None
        (min, max) limits for the y axis.
        
        If None, will be directly interpreted from the data.

    annotAlpha    : float >= 0
        Controls opacity of the annotation line drawn. Value should range from 0-1, both inclusive

        If `annotate` is True, sent to `drawAnnotation()`.

    yAnnot    : float
        Value ranging from 0-1, both inclusive. 
        
        Indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

        If `annotate` is True, sent to `drawAnnotation()`.

    sizeAnnot    : number 
        Font size for annotated text.

        If `annotate` is True, sent to `drawAnnotation()`.
    
    '''
    # TODO: for some reason, below line is throwing an error due to x_coords and y_coords; I'm passing o/ps X,t,f from spectrogram function
    #specshow(X_dB, x_coords=t, y_coords=f, x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopSize, ax=ax, cmap=cmap)
    specshow(X_dB,x_axis='time', y_axis='linear', sr=sr, fmax=sr//2, hop_length=hopSize, ax=ax, cmap=cmap)

    # set ylim if required
    if ylim is None:
        ylim = ax.get_ylim()
    # Added below block
    if duration is None:
        duration = t[-1]

    # set axes params
    ax.set(ylabel='Frequency (Hz)' if ylabel else '', 
    xlabel='Time (s)' if xlabel else '', 
    title=title,
    xlim=(startTime, startTime+duration), 
    xticks=np.around(np.arange(math.ceil(t[0]), math.floor(t[-1]), freqXlabels)).astype(int),     # start the xticks such that each one corresponds to an integer with xticklabels
    xticklabels=np.around(np.arange(math.ceil(t[0]), math.floor(t[-1]), freqXlabels)).astype(int) if xticks else [],     # let the labels start from the integer values.
    ylim=ylim,
    yticks= np.arange(math.ceil(ylim[0]/1000)*1000, math.ceil(ylim[1]/1000)*1000, 2000) if yticks else [], #TODO: try to see if you can make this more general#
    yticklabels=[f'{(x/1000).astype(int)}k' for x in np.arange(math.ceil(ylim[0]/1000)*1000, math.ceil(ylim[1]/1000)*1000, 2000)]  if yticks else [])

    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime, duration, ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha, y=yAnnot, size=sizeAnnot)

    return ax

# PLOTTING FUNCTION
def drawWave(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, ax=None, xticks=False, yticks=True, xlabel=True, ylabel=True, title='Waveform', freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='purple', annotLabel=True, odf=False, winSize_odf=640, hopSize_odf=160, nFFT_odf=1024, source_odf='vocal', cOdf='black', ylim=None, annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Plots the wave plot of the audio

    Plots the waveform of the given audio using `librosa.display.waveshow()`.

    Parameters
    ----------
        audio    : ndarray or None
            Loaded audio time series

        sr    : number > 0; default=16000
            If audio is not None, defines sample rate of audio time series 

            If audio is None and audioPath is not None, defines sample rate to load the audio at

        audioPath    : str
            Path to audio file. 

            Used only if audio is None. Audio is loaded as mono.

        startTime    : float; default=0
            Time stamp to consider audio from.

        duration    : float or None; default=None
            Duration of the audio to consider.

        ax    : matplotlib.axes.Axes or None
            Axes to plot waveplot in.

            If None, will plot the object in `plt.gca()`

        xticks    : bool
            If True, will add xticklabels to plot.

        yticks    : bool
            If True, will add yticklabels to plot.

        xlabel    : bool
            If True, will print xlabel in the plot.

        ylabel    : bool
            If True will print ylabel in the plot.

        title    : str
            Title to add to the plot.

        freqXlabels    : float > 0
            Time (in seconds) after which each x ticklabel occurs in the plot.

        cyclePath    : str or None
            Path to file with tala cycle annotations.

            Passed to `drawAnnotation()`.

        numDiv    : int >= 0
            Number of divisions to put between each annotation marking in cyclePath. Used only if cyclePath is not None.

            Passed to `drawAnnotation()`.

        onsetPath    : str or None
            Path to file with onset annotations. Only considered if cyclePath is None.

            Passed to `drawAnnotation()`.

        onsetTimeKeyword    : str
            Column name in the onset file to take time stamps of onsets from.

            Passed to `drawAnnotation()`.

        onsetLabelKeyword    : str or list or None
            Column name with label(s) for the onsets. If None, no label will be printed.

            Passed to `drawAnnotation()`.

        cAnnot: color 
            Determines the colour of annotion. Input to the `matplotlib.pyplot.annotate()` function for the `c` parameter.
            
            Passed to `drawAnnotation()`.

        annotLabel    : bool
            If True, will print annotation label along with the annotation line. Used only if annotate is True.

            Passed to `drawAnnotation()`.

        odf    : bool
            If True, will plot the onset detection function over the wave form.

            Uses `getOnsetActivations()` to compute ODF.

        winSize_odf    : int
            Window size (in frames) used by the onset detection function.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        hopSize_odf    : int
            Hop size (in frames) used by the onset detection function.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        nFFT_odf    : int
            Size of DFT used in onset detection function.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        source_odf    : str
            Defines type of instrument in the audio. Accepted values are:
            - 'vocal'
            - 'pakhawaj'
            
            Used in the `getOnsetActivation()` only if `odf` is True.

        cOdf    : color 
            Colour to plot onset detection function in.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        ylim    : (float, float) or None
            (min, max) limits for the y axis.
            
            If None, will be directly interpreted from the data.

        annotAlpha    : float >= 0
            Controls opacity of the annotation line drawn. Value should range from 0-1, both inclusive

            If `annotate` is True, sent to `drawAnnotation()`.

        yAnnot    : float
            Value ranging from 0-1, both inclusive. 
            
            Indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

            If `annotate` is True, sent to `drawAnnotation()`.

        sizeAnnot    : number 
            Font size for annotated text.

            If `annotate` is True, sent to `drawAnnotation()`.
    
    Returns
    -------
        ax    : matplotlib.axes.Axes
            Waveform plot of given audio.

    '''
    
    # startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot #TODO: check and remove#
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio, sr=sr)
        # duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot    #TODO: check and remove#
        # audio = audio[:int(duration*sr)]    # ensure that audio length = duration    #TODO: check and remove#

    waveplot(audio, sr, ax=ax)

    if odf:
        plotODF(audio=audio, sr=sr, startTime=0, duration=None, ax=ax, winSize_odf=winSize_odf, hopSize_odf=hopSize_odf, nFFT_odf=nFFT_odf, source_odf=source_odf, cOdf=cOdf, ylim=True)

    # set ylim if required
    if ylim is None:
        ylim = ax.get_ylim()

    ax.set(xlabel='' if not xlabel else 'Time (s)', 
    ylabel = '' if not ylabel else 'Amplitude',
    xlim=(0, duration), 
    xticks=[] if not xticks else np.around(np.arange(math.ceil(startTime) - startTime, duration, freqXlabels)),
    xticklabels=[] if not xticks else np.around(np.arange(math.ceil(startTime), duration+startTime, freqXlabels)).astype(int),
    yticks=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 1), 
    yticklabels=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 1), 
    ylim=ylim,
    title=title)

    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha, y=yAnnot, size=sizeAnnot)
    
    return ax

# PLOTTING FUNCTION
def plotODF(audio=None, sr=16000, audioPath=None, odf=None, startTime=0, duration=None, ax=None, winSize_odf=640, hopSize_odf=160, nFFT_odf=1024, source_odf='vocal', cOdf='black', xlabel=False, ylabel=False, xticks=False, yticks=False, title='Onset Detection Function', freqXlabels=5, ylim=True):
    '''
    Plots onset detection function if `ax` is provided. Function comes from `getOnsetActivation()`.
    
    If `ax` is None, function returns a tuple with 2 arrays - onset detection function values and time stamps

    Parameters
    ----------
        audio    : ndarray or None
            Loaded audio time series

        sr    : number > 0; default=16000
            If audio is not None, defines sample rate of audio time series 

            If audio is None and audioPath is not None, defines sample rate to load the audio at

        audioPath    : str
            Path to audio file. 

            Used only if audio is None. Audio is loaded as mono.

        odf : ndarray
            Extracted onset detection function, if already available
            
            Can be obtained using `getOnsetActivation()` function

        startTime    : float; default=0
            Time stamp to consider audio from.

        duration    : float or None; default=None
            Duration of the audio to consider.

            If duration is None
                - If `audio` is None, duration is inferred from the audio.
                - If `audio` is None and `audioPath` is not None, the entire song is loaded.

        ax    : matplotlib.axes.Axes
            Axes object to plot waveplot in.

        winSize_odf    : int
            Window size (in frames) used by the onset detection function.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        hopSize_odf    : int
            Hop size (in frames) used by the onset detection function.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        nFFT_odf    : int
            Size of DFT used in onset detection function.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        source_odf    : str
            Defines type of instrument in the audio. Accepted values are:
            - 'vocal'
            - 'pakhawaj'
            
            Used in the `getOnsetActivation()` only if `odf` is True.

        cOdf    : color 
            Colour to plot onset detection function in.

            If `odf` is True, passed to the `getOnsetActivation()` function.

        xticks    : bool
            If True, will add xticklabels to plot.

        yticks    : bool
            If True, will add yticklabels to plot.

        xlabel    : bool
            If True, will print xlabel in the plot.

        ylabel    : bool
            If True will print ylabel in the plot.

        title    : str
            Title to add to the plot.

        freqXlabels    : float > 0
            Time (in seconds) after which each x ticklabel occurs in the plot.

        ylim    : (float, float) or None
            (min, max) limits for the y axis.
            
            If None, will be directly interpreted from the data.

    Returns
    -------
        ax    : matplotlib.axes.Axes)
            If `ax` is not None, returns a plot
        
        (odf_vals, time_vals): (ndarray, ndarray)
            If `ax` is None, returns a tuple with ODF values and time stamps.
    '''

    if odf is None:
        # startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot    # TODO: test and remove#
        if audio is None:
            audio, sr = librosa.load(audioPath, sr=sr, offset=startTime, duration=duration)
        if duration is None:
            duration = librosa.get_duration(audio, sr=sr)
            # duration = math.floor(duration)  # set duration to an integer, for better readability on the x axis of the plot # TODO: remove #
            # audio = audio[:int(duration*sr)]    # ensure that audio length = duration # TODO: remove #

        odf_vals, _ = getOnsetActivation(audio=audio, audioPath=None, startTime=startTime, duration=duration, fs=sr, winSize=winSize_odf, hopSize=hopSize_odf, nFFT=nFFT_odf, source=source_odf)
    else:
        odf_vals = odf.copy()
        duration = len(odf_vals)*hopSize_odf/sr

    # set time and odf values in variables
    time_vals = np.arange(startTime, startTime+duration, hopSize_odf/sr) # changed last argument to hopsize_odf/sr because hopsize_odf is in frames now 
    #odf_vals = odf_vals[:-1]    # disregard the last frame of odf_vals since it is centered around the frame at time stamp `startTime`` + `duration`` # TODO: not sure this is necessary. I got length mismatch errors. We could instead add code to make lengths same like below:
    # odf_vals = odf_vals[: min((len(time_vals), len(time_vals)))]
    # time_vals = time_vals[: min((len(time_vals), len(time_vals)))]
    
    if ax is None:
        # if ax is None, return (odf_vals, time_vals)
        return (odf_vals, time_vals)
    else:
        ax.plot(time_vals, odf_vals, c=cOdf)     # plot odf_vals and consider odf_vals for all values except the last frame
        max_abs_val = max(abs(min(odf_vals)), abs(max(odf_vals)))   # find maximum value to set y limits to ensure symmetrical plot
        # set ax parameters only if they are not None
        ax.set(xlabel='' if not xlabel else 'Time (s)', 
        ylabel = '' if not ylabel else 'ODF',
        xlim=(0, duration), 
        xticks=[] if not xticks else np.around(np.arange(math.ceil(startTime), duration+startTime, freqXlabels)),
        xticklabels=[] if not xticks else np.around(np.arange(math.ceil(startTime), duration+startTime, freqXlabels)).astype(int),
        yticks=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
        yticklabels=[] if not yticks else np.around(np.linspace(min(audio), max(audio), 3), 2), 
        ylim= ax.get_ylim() if ylim is not None else (-max_abs_val, max_abs_val),
        title=title) 
        return ax

# AUDIO MANIPULATION    
def playAudio(audio=None, sr=16000, audioPath=None, startTime=0, duration=None):
    '''Plays relevant part of audio

    Parameters
    ----------
        audio    : ndarray or None
            Loaded audio time series.

        sr    : number > 0; default=16000
            If audio is not None, defines sample rate of audio time series .

            If audio is None and audioPath is not None, defines sample rate to load the audio at.

        audioPath    : str or None
            Path to audio file.

        startTime    : float; default=0
            Time stamp to consider audio from.

        duration    : float or None; default=None
            Duration of the audio to consider.

    Returns
    ----------
        iPython.display.Audio 
            Object that plays the audio.
    '''
    if audio is None:
        audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=duration)
    return Audio(audio, rate=sr)

# AUDIO MANIPULATION
def playAudioWClicks(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, onsetFile=None, onsetLabels=['Inst', 'Tabla'], destPath=None):
    '''Plays relevant part of audio along with clicks at timestamps of each of the onsetLabels provided.

    If `destPath` is not None, generated audio is saved at `destPath`, else the generated audio is returned as a `iPython.display.Audio` object.

    Parameters
    ----------
        audio    : ndarray or None
            Loaded audio time series.

        sr    : number > 0; default=16000
            If audio is not None, defines sample rate of audio time series .

            If audio is None and audioPath is not None, defines sample rate to load the audio at.

        audioPath    : str or None
            Path to audio file.

        startTime    : float
            Time stamp to consider audio from.

        duration    : float
            Duration of the audio to consider.
        
        onsetFile    : str
            File path to csv onset time stamps.

        onsetLabels    : str or list
            Column name(s) in onsetFile with time stamp for different types of onsets.

            If a list is given, then clicks will be generated with different frequencies for each column name in the list.

        destPath    : str or None
            Path to save audio file at.
            
            If None, will not save any audio file.

    Returns
    ----------
        iPython.display.Audio 
            Object that plays the audio with clicks.
    '''

    if audio is None:
        audio, sr = librosa.load(audioPath, sr=None, offset=startTime, duration=duration)
    if duration is None:
        duration = librosa.get_duration(audio)
    onsetFileVals = pd.read_csv(onsetFile)
    onsetTimes = []

    # check if onsetLabels is a str, convert it to a list
    if isinstance(onsetLabels, str):
        onsetLabels = [onsetLabels]
    for onsetLabel in onsetLabels:
        onsetTimes.append(onsetFileVals.loc[(onsetFileVals[onsetLabel] >= startTime) & (onsetFileVals[onsetLabel] <= startTime+duration), onsetLabel].values)
    clickTracks = [librosa.clicks(onsetTime-startTime, sr=sr, length=len(audio), click_freq=1000*(2*i+1)) for i, onsetTime in enumerate(onsetTimes)]     # frequency of each click is set 2000 Hz apart.
    audioWClicks = 0.8*audio  # scale audio amplitude by 0.8
    for clickTrack in clickTracks:
        audioWClicks += 0.2/len(clickTracks)*clickTrack     # add clicks to the audio
    if destPath is not None:
        # write the audio
        sf.write(destPath, audioWClicks, sr)
    return Audio(audioWClicks, rate=sr)

# AUDIO MANIPULATION
def playVideo(video=None, videoPath=None, startTime=0, duration=None, destPath='Data/Temp/VideoPart.mp4', videoOffset=0):
    '''Plays relevant part of a given video.

    If `duration` is None and `startTime` is 0, the entire Video is returned. 
    
    If `duration` is not None or `startTime` is not 0, the video is cut using the `ffmpeg` Python library and is stored in `destPath`. 

    Parameters
    ----------
        video    : ndarray or None
            Loaded video sample. 

            When `video` is not None, all the other parameters in the function are not considered. If a trimmed video is needed, please use `videoPath` instead.

            If None, `videoPath` will be used to load the video.
        videoPath    : str
            Path to video file.

        startTime    : float
            Time to start reading the video from. 
            
            Used only when `video` is None.
        
        duration    : float
            Duration of the video to load.

            Used only when `video` is None.

        destPath    : str or None
            Path to store shortened video.

            Used only when `video` is None.

        videoOffset    : float
            Number of seconds offset between video and audio files. This parameter is useful when the video is present only for an excerpt of the audio file.
            
            ::
                time in audio + `videoOffset` = time in video
    Returns
    -------
        iPython.display.Video 
            Object that plays the video.

    Raises
    ------
        ValueError
            If `destPath` is None, when `startTime` != 0 or `duration` is not None.
    '''
    if video is None:
        if duration is None and startTime == 0:
            # play the entire video
            return Video(videoPath, embed=True)
        else:
            # store a shortened video in destPath
            if destPath is None:
                # if destPath is None, raise an error
                raise ValueError(f'destPath cannot be None if video is to be trimmed before playing. destPath has invalid type of {type(destPath)}.')
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
        return Video(data=video, embed=True)

# PLOTTING FUNCTION
def generateFig(noRows, figSize=(14, 7), heightRatios=None):
    '''Generates a matplotlib.pyplot.figure and axes to plot in.

    Axes in the plot are stacked vertically in one column, with height of each axis determined by heightRatios.

    Parameters
    ----------
        noRows    : int > 0
            Number of plots (i.e. rows) in the figure.

        figSize    : (float, float)
            (width, height) in inches of the figure.

            Passed to `matplotlib.figure.Figure()`.

        heightRatios    : list or None
            List of heights that each plot in the figure should take. Relative height of each row is determined by ``heightRatios[i] / sum(heightRatios)``.

            Passed to `matplotlib.figure.Figure.add_gridspec()` as the parameter `height_ratios`.
            
            ..note::
                len(heightRatios) has to be equal to noRows

    Returns
    -------
        fig    : matplotlib.figure.Figure()
            Figure object with all the plots

        axs    : list of matplotlib.axes.Axes objects 
            List of axes objects. Each object corresponds to one row/plot. 

    Raises
    ------
        Exception
            If ``len(heightRatios) != noRows``.
    '''
    if len(heightRatios) != noRows:
        Exception("Length of heightRatios has to be equal to noRows")
    # if heightRatios is None:
    #     # if heightRatios is None
    #     heightRatios = np.ones(noRows)     #TODO: check that this works and remove#
    fig = plt.figure(figsize=figSize)
    specs = fig.add_gridspec(noRows, 1, height_ratios = heightRatios)
    axs = [fig.add_subplot(specs[i, 0]) for i in range(noRows)]
    return fig, axs

def subBandEner(X,fs,band):
    '''Computes spectral sub-band energy by summing squared magnitude values of STFT over specified spectral band (suitable for vocal onset detection).

    Parameters
    ----------
        X   : ndarray
            STFT of an audio signal x
        fs  : int or float
            Sampling rate
        band    : list or tuple or ndarray
            Edge frequencies (in Hz) of the sub-band of interest
        
    Returns
    ----------
        sbe : ndarray
            Array with each value representing the magnitude STFT values in a short-time frame squared & summed over the sub-band
    '''

    #convert band edge frequencies to bin numbers
    binLow = int(np.ceil(band[0]*X.shape[0]/(fs/2)))
    binHi = int(np.ceil(band[1]*X.shape[0]/(fs/2)))

    #compute sub-band energy
    sbe = np.sum(np.abs(X[binLow:binHi])**2, 0)

    return sbe

def biphasicDerivative(x, hopDur, norm=True, rectify=True):
    '''Computes the biphasic derivative of a signal(See [1] for a detailed explanation of the algorithm).
    
    Parameters
    ----------
        x   : ndarray
            Input signal
        hopDur  : float
            Sampling interval in seconds of input signal (reciprocal of sampling rate of x)
        norm    :bool
            If output is to be normalized
        rectify :bool
            If output is to be rectified to keep only positive values (sufficient for peak-picking)
    
    Returns
    ----------
        x   : ndarray
            Output of convolving input with the biphasic derivative filter

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
    
    Parameters
    ----------
        x   : ndarray
            Input signal
        C   : int or float
            Scaling constant
    
    Returns
    ----------
        log-scaled x
    '''
    return np.log10(1 + x*C)/(np.log10(1+C))

def getOnsetActivation(audio=None, audioPath=None, startTime=0, duration=None, fs=16000, winSize=640, hopSize=160, nFFT=1024, source='vocal'):
    '''Computes onset activation function from audio signal using short-time spectrum based methods.

    Parameters
    ----------
        audio   : ndarray
            Audio signal
        audioPath  : str
            Path to the audio file
        startTime   : int or float
            Time to start reading the audio at
        duration    : int or float
            Duration of audio to read
        fs  : int or float
            Sampling rate to read audio at
        winSize : int
            Window size (in frames) for STFT
        hopSize : int
            Hop size (in frames) for STFT
        nFFT    : int
            DFT size
        source  : str
            Source instrument in audio - one of 'vocal' or 'perc' (percussion)

    Returns
    ----------
        odf : ndarray
            Frame-wise onset activation function (at a sampling rate of 1/hopSize)
        onsets  ndarray
            Time locations of detected onset peaks in the odf (peaks detected using peak picker from librosa)
    '''

    #if audio signal is provided
    if audio is not None:
        #select segment of interest from audio based on start time and duration
        if duration is None:
            audio = audio[int(np.ceil(startTime*fs)):]
        else:
            audio = audio[int(np.ceil(startTime*fs)):int(np.ceil((startTime+duration)*fs))]

    #if audio path is provided
    elif audioPath is not None:
        audio,_ = librosa.load(audioPath, sr=fs, offset=startTime, duration=duration)

    else:
        print('Provide either the audio signal or path to the stored audio file on disk')
        raise

    #fade in and out ends of audio to prevent spurious onsets due to abrupt start and end
    audio = fadeIn(audio,int(0.5*fs))
    audio = fadeOut(audio,int(0.5*fs))

    #compute magnitude STFT
    X,_ = librosa.magphase(librosa.stft(audio, win_length=winSize, hop_length=hopSize, n_fft=nFFT))

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
        odf = fmp.spectral_flux(audio, Fs=fs, N=nFFT, W=winSize, H=hopSize, M=20, band=sub_band)
        onsets = librosa.onset.onset_detect(onset_envelope=odf, sr=fs, hop_length=hopSize, pre_max=1, post_max=1, pre_avg=1, post_avg=1, wait=10, delta=0.05)*hopSize/fs

    return odf, onsets
    
def fadeIn(x, length):
    '''
    Apply fade-in to the beginning of an audio signal using a hanning window.
    
    Parameters
    ----------
        x   : ndarray
            Signal
        length  : int
            Length of fade (in samples)
    
    Returns
    ----------
        x   : ndarray
            Faded-in signal
    '''
    x[:length] *= (np.hanning(2*length))[:length]
    return x

def fadeOut(x, length):
    '''
    Apply fade-out to the end of an audio signal using a hanning window.
    
    Parameters
    ----------
        x   : ndarray
            Signal
        length  : int
            Length of fade (in samples)
    
    Returns
    ----------
        x   : ndarray
            Faded-out signal
    '''
    x[-length:] *= np.hanning(2*length)[length:]
    return x

def autoCorrelationFunction(x, fs, maxLag, winSize, hopSize):
    '''
    Compute short-time autocorrelation of a signal and normalise every frame by the maximum correlation value.
    
    Parameters
    ----------
        x   : ndarray
            Input signal
        fs  : int or float
            Sampling rate
        maxLag  : int or float
            Maximum lag in seconds upto which correlation is to be found (ACF is computed for all unit sample shift values lesser than this limit)
        winSize : int or float
            Length in seconds of the signal selected for short-time autocorrelation
        hopSize : int or float
            Hop duration in seconds between successive windowed signal segments (not the same as lag/shift)
    
    Returns
    ----------
        ACF : ndarray
            short-time ACF matrix [shape=(#frames,#lags)]
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
    
    Parameters
    ----------
        x   : ndarray
            Input signal
        winSize : int or float
            Size of short-time window in seconds
        hopSize : int or float
            Hop duration in seconds between consecutive windows
    
    Returns
    ----------
        x_sub   : ndarray
            2d array containing windowed segments
    '''

    #pre-calculate shape of output numpy array (#rows based on #windows obtained using provided window and hop sizes)
    shape = (int(1 + (len(x) - winSize)/hopSize), winSize)

    strides = (hopSize*x.strides[0], x.strides[0])
    x_sub = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return x_sub

def tempoPeriodLikelihood(ACF, norm=True):
    '''
    Compute from ACF, the likelihood of each ACF lag being the time-period of the tempo. Likelihood is obtained by taking a dot product between the ACF vector and a comb-filter (see [3] for details) in each time frame.
    
    Parameters
    ----------
        ACF : ndarray
            Short-time ACF matrix of a signal
        norm    : bool

    Returns
    ----------
        tempo_period_candidates : ndarray
            2d array with a likelihood value for each lag in each time frame
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
    
    Parameters
    ----------
        tempoPeriodLikelihood   : ndarray
            Likelihood values at each lag (tempo period)
        hopDur  : int or float
            Short-time analysis hop duration in seconds between samples of ACF vector (not hop duration between windowed signal segments taken for ACF)
        transitionPenalty   : int or float
            Penalty factor multiplied with the magnitude of tempo change between frames; high value penalises larger jumps more, suitable for metric tempo that changes gradually across a concert and not abruptly
        tempoRange  : tuple or list
            Expected min and max tempo in BPM

    Returns
    ----------
        tempo_period_smoothed   : ndarray
            Array of chosen tempo period in each time frame
    '''

    #convert tempo range to tempo period (in frames) range
    fs = 1/hopDur
    tempoRange = np.around(np.array(tempoRange)*(fs/60)).astype(int)

    #initialise cost matrix with very high values
    T,L = np.shape(tempoPeriodLikelihood)
    cost = ones((T,L))*1e8

    #matrix to store cost-minimizing lag in previous frame to each lag in current time frame
    m = zeros((T,L))

    #compute cost value at each lag (within range), at each time frame
    #loop over time frames
    for i in range(1,T):
        #loop over lags in current time frame
        for j in range(*tempoRange):
            #loop over lags in prev time frame
            for k in range(*tempoRange):
                if cost[i][j]>cost[i-1][k]+transitionPenalty*abs(60.0*fs/j-60.0*fs/k)-tempoPeriodLikelihood[i][j]:
                    #choose lag 'k' in prev time frame that minimizes cost at lag 'j' in current time frame 
                    cost[i][j]=cost[i-1][k]+transitionPenalty*abs(60.0*fs/j-60.0*fs/k)-tempoPeriodLikelihood[i][j]
                    m[i][j]=int(k)

    #determine least cost path - start at the last frame (pick lag with minimum cost)
    tempo_period_smoothed = zeros(T)
    tempo_period_smoothed[T-1] = argmin(cost[T-1,:])/float(fs)
    t = int(m[T-1,argmin(cost[T-1,:])])

    #loop in reverse till the first frame, reading off values in 'm[i][j]'
    i = T-2
    while(i>=0):
        tempo_period_smoothed[i]=t/float(fs)
        t=int(m[i][t])
        i=i-1
    return tempo_period_smoothed
    
# COMPUTATION FUNCTION
def intensityContour(audio=None, sr=16000, audioPath=None, startTime=0, duration=None, minPitch=98, timeStep=0.01, ax=None, freqXlabels=5, annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', xticks=False, yticks=False, xlabel=True, ylabel=True, title='Intensity Contour', cAnnot='red', annotLabel=True, annotAlpha=0.8):
    '''Calculates the intensity contour for an audio clip.

    Intensity contour is generated for a given audio with [#]_ and it's Python API. [#]_ 

    ..[#] Boersma, P., & Weenink, D. (2021). Praat: doing phonetics by computer [Computer program]. Version 6.1.38, retrieved 2 January 2021 from http://www.praat.org/

    ..[#] Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

    Parameters
    ----------
        audio    : ndarray or None
            Loaded audio time series

        sr    : number > 0; default=16000
            If audio is not None, defines sample rate of audio time series 

            If audio is None and audioPath is not None, defines sample rate to load the audio at

        audioPath    : str
            Path to audio file. 

            Used only if audio is None.

        startTime    : float; default=0
            Time stamp to consider audio from

        duration    : float or None; default=None
            Duration of the audio to consider

        minPitch    : float; default=98
            Minimum pitch (in Hz) to read for contour extraction.

            Passed as `minimum_pitch` parameter to `parselmouth.Sound.to_intensity()`.

        timeStep    : float; default=0.01
            Time steps (in seconds) in which pitch values are extracted.::

                Example: timeStep = 0.01 implies that pitch values are extracted for every 0.01 s.

            Passed as `time_step` parameter to `parselmouth.Sound.to_intensity()`.

        ax    : matplotlib.axes.Axes or None
            Axes object to plot the intensity contour in.

            If None, will return a tuple with (intensity contour, time steps)

        freqXlabels    : float
            Time (in seconds) after which each x label occurs in the plot

        annotate    : bool
            If True, will annotate markings in either cyclePath or onsetPath with preference to cyclePath.

        cyclePath    : str or None
            Path to file with tala cycle annotations.

            Passed to `drawAnnotation()`.

        numDiv    : int >= 0
            Number of divisions to put between each annotation marking in cyclePath. Used only if cyclePath is not None.

            Passed to `drawAnnotation()`.

        onsetPath    : str or None
            Path to file with onset annotations. Only considered if cyclePath is None.

            Passed to `drawAnnotation()`.

        onsetTimeKeyword    : str
            Column name in the onset file to take time stamps of onsets from.

            Passed to `drawAnnotation()`.

        onsetLabelKeyword    : str or list or None
            Column name with label(s) for the onsets. If None, no label will be printed.

            Passed to `drawAnnotation()`.

        xticks    : bool
            If True, will add xticklabels to plot.

            Passed to `plotIntensity()`.

        yticks    : bool
            If True, will add yticklabels to plot.

            Passed to `plotIntensity()`.

        xlabel    : bool
            If True, will print xlabel in the plot.

            Passed to `plotIntensity()`.

        ylabel    : bool
            If True will print ylabel in the plot.

            Passed to `plotIntensity()`.

        title    : str
            Title to add to the plot.

            Passed to `plotIntensity()`.

        annotLabel    : bool
            If True, will print annotation label along with the annotation line. Used only if annotate is True.

            Passed to `plotIntensity()`.

        cAnnot: color 
            Determines the colour of annotion. Input to the `matplotlib.pyplot.annotate()` function for the `c` parameter.
            
            Passed to `plotIntensity()`.

        ylim    : (float, float) or None
            (min, max) limits for the y axis; if None, will be directly interpreted from the data

            Passed to `plotIntensity()`.

        annotAlpha    : (float)
            Controls opacity of the annotation lines in the plot. Value has to be in the range [0, 1], inclusive.

            Passed to `drawAnnotation()`.

    Returns
    -------
        ax : matplotlib.axes.Axes
            Plot of intensity contour if `ax` was not None

        (intensityvals, timevals)    : (ndarray, ndarray)
            Tuple with arrays of intensity values (in dB) and time stamps. Returned if ax was None.
    
    '''
    startTime = math.floor(startTime)   # set start time to an integer, for better readability on the x axis of the plot
    duration = math.ceil(duration)  # set duration to an integer, for better readability on the x axis of the plot
    if audio is None:
        # if audio is not given, load audio from audioPath
        audio, sr = librosa.load(audioPath, sr=sr, mono=True, offset=startTime, duration = duration)
    snd = parselmouth.Sound(audio, sr)
    intensity = snd.to_intensity(time_step=timeStep, minimum_pitch=minPitch)
    intensity_vals = intensity.values[0]
    time_vals = intensity.xs() + startTime
    
    if ax is None:
        # if ax is None, return intensity and time values
        return (intensity_vals, time_vals)
    else:
        # else plot the contour
        return plotIntensity(intensity_vals=intensity_vals, time_vals=time_vals, ax=ax, startTime=startTime, duration=duration, freqXlabels=freqXlabels, xticks=xticks, yticks=yticks, xlabel=xlabel, ylabel=ylabel, title=title, annotate=annotate, cyclePath=cyclePath, numDiv=numDiv, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, cAnnot=cAnnot, annotLabel=annotLabel, annotAlpha=annotAlpha)

# PLOTTING FUNCTION
def plotIntensity(intensity_vals=None, time_vals=None, ax=None, startTime=0, duration=None, freqXlabels=5, xticks=True, yticks=True, xlabel=True, ylabel=True, title='Intensity Contour', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='red', annotLabel=True, ylim=None, annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Function to plot a computed intensity contour from `intensityContour()` function. 

    Parameters
    ----------
        intensity_vals    : ndarray
            Intensity contour from `intensityContour()`

        time_vals    : ndarray or None
            Time steps corresponding to the `intensity_vals` from `intensityContour`

            If None, assumes a time step of 0.01 s
        
        ax    : matplotlib.axes.Axes or None
            Object on which intensity contour is to be plotted

            If None, will plot in `matplotlib.pyplot.gca()`.

        startTime    : float >= 0
            Offset time (in seconds) from where audio is analysed.

            Sent to `drawAnnotation()`.

        duration    : float >= 0 or None
            Duration of audio in the plot.

            If None, will consider the entire audio.

            Sent to `drawAnnotation()`.

        freqXlabels    : int
            Time (in seconds) after which each x ticklabel should occur

        annotate    : bool
            If true will mark annotations provided in the plot.

            Send to `drawAnnotation()`.

        xticks    : bool
            If True, will print x ticklabels in the plot.

        yticks    : bool
            If True, will print y ticklabels in the plot.

        xlabel    : bool
            If True, will add label to x axis

        ylabel    : bool
            If True, will add label to y axis

        title    : str
            Title to add to the plot

        annotate    : bool
            If True, will add tala-related/onset annotations to the plot 

        cyclePath    : bool
            Path to csv file with tala-related annotations.
            
            If annotate is True, sent to `drawAnnotation()`.

        numDiv    : int
            Number of divisions to add between each tala-related annotation provided.

            If annotate is True, sent to `drawAnnotation()`.

        onsetPath    : str
            Path to file with onset annotations

            If annotate is True, sent to `drawAnnotation()`. Used only if `cyclePath` is None.

        onsetTimeKeyword    : str
            Column name in the onset file to take onset's timestamps from.

            If annotate is True, sent to `drawAnnotation()`. Used only if `cyclePath` is None.

        onsetLabelKeyword    : str or list or None
            Column name(s) in onsetPath file with labels values for the onsets.
            
            If None, no label will be printed for the onsets. 

            If annotate is True, sent to `drawAnnotation()`. Used only if `cyclePath` is None.

        cAnnot    : color
            Determines the colour of annotation. Sent as input to the `matplotlib.pyplot.annotate()` function for the colour (`c`) parameter.
            
            If `annotate` is True, sent to `drawAnnotation()`.
            
        annotLabel    : bool
            If True, will print annotation label along with line.

            If `annotate` is True, sent to `drawAnnotation()`.

        ylim    : (float, float) or None
            (min, max) limits for the y axis.
            
            If None, will be directly interpreted from the data.

        annotAlpha    : float >= 0
            Controls opacity of the annotation line drawn. Value should range from 0-1, both inclusive

            If `annotate` is True, sent to `drawAnnotation()`.

        yAnnot    : float
            Value ranging from 0-1, both inclusive. 
            
            Indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

            If `annotate` is True, sent to `drawAnnotation()`.

        sizeAnnot    : number 
            Font size for annotated text.

            If `annotate` is True, sent to `drawAnnotation()`.

    Returns
    -------
        ax    : matplotlib.axes.Axes
            Plot of intensity contour.

    Raises
    ------
        ValueError
            If intensity_vals is not given
    
    '''
    if intensity_vals is None:
        Exception('No intensity contour provided')

    if time_vals is None:
        # if time vals is None, assumes 0.01s time step
        warnings.warn('No time values provided, assuming 0.01 s time steps in intensity contour')
        timevals = np.arange(startTime, len(intensity_vals)*0.01, 0.01)
    
    # check if ax is None
    ax = __check_axes(ax)
    
    ax = sns.lineplot(x=time_vals, y=intensity_vals, ax=ax, color=cAnnot);
    ax.set(xlabel='Time Stamp (s)' if xlabel else '', 
    ylabel='Intensity (dB)' if ylabel else '', 
    title=title, 
    xlim=(startTime, duration+startTime), 
    xticks=(np.arange(math.ceil(startTime), startTime+duration, freqXlabels)), 
    xticklabels=(np.arange(math.ceil(startTime), startTime+duration, freqXlabels) )if xticks else [],
    ylim=ylim if ylim is not None else ax.get_ylim())
    if not yticks:
        ax.set(yticklabels=[])
    if annotate:
        ax = drawAnnotation(cyclePath=cyclePath, onsetPath=onsetPath, onsetTimeKeyword=onsetTimeKeyword, onsetLabelKeyword=onsetLabelKeyword, numDiv=numDiv, startTime=startTime, duration=duration, ax=ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha, yAnnot=yAnnot, sizeAnnot=sizeAnnot)
    return ax

# PLOTTING FUNCTION
def plot_hand(annotationFile=None, startTime=0, duration=None, vidFps=25, ax=None, freqXLabels=5, xticks=False, yticks=False, xlabel=True, ylabel=True, title='Wrist Position Vs. Time', vidOffset=0, lWristCol='LWrist', rWristCol='RWrist', wristAxis='y', annotate=False, cyclePath=None, numDiv=0, onsetPath=None, onsetTimeKeyword='Inst', onsetLabelKeyword='Label', cAnnot='yellow', annotLabel=False, ylim=None, annotAlpha=0.8, yAnnot=0.7, sizeAnnot=10):
    '''Function to plot hand movement.

    Using Openpose annotations, this function plots the height of each hand's wrist vs time. 

    If `ax` is None, this will on `plt.gca()`, i.e. the current axes being used
    
    Parameters
    ----------
        annotationFile    : str
            File path to Openpose annotations.

        startTime    : float
            Start time for x labels in the plot (time stamp with respect to the audio signal).

        duration    : float
            Duration of audio to consider for the plot.
            
        vidFps    : float
            FPS of the video data used in openpose annotation.

        ax    : matplotlib.axes.Axes or None 
        Axes object on which plot is to be plotted.

        If None, uses the current Axes object in use with `plt.gca()`. 

        freqXlabels    : int > 0 
            Time (in seconds) after which each x ticklabel occurs

        xticks    : bool
            If True, will add xticklabels to plot.

        yticks    : bool
            If True, will add yticklabels to plot.

        xlabel    : bool
            If True, will print xlabel in the plot.

        ylabel    : bool
            If True will print ylabel in the plot.

        title    : str
            Title to add to the plot.

        videoOffset    : float
            Number of seconds offset between video and audio::
                time in audio + videioOffset = time in video

        lWristCol    : str
            Name of the column with left wrist data in `annotationFile`.

        rWristCol    : str
            Name of the column with right wrist data in `annotationFile`.

        wristAxis    : str
            Level 2 header in the `annotationFile` denoting axis along which movement is plotted (x, y or z axes).

        annotate    : bool
            If True will mark annotations provided on the plot.

        cyclePath    : str or None
            Path to file with tala cycle annotations.

            Passed to `drawAnnotation()`.

        numDiv    : int >= 0
            Number of divisions to put between each annotation marking in cyclePath. Used only if cyclePath is not None.

            Passed to `drawAnnotation()`.

        onsetPath    : str or None
            Path to file with onset annotations. Only considered if cyclePath is None.

            Passed to `drawAnnotation()`.

        onsetTimeKeyword    : str
            Column name in the onset file to take time stamps of onsets from.

            Passed to `drawAnnotation()`.

        onsetLabelKeyword    : str or list or None
            Column name with label(s) for the onsets. If None, no label will be printed.

            Passed to `drawAnnotation()`.

        cAnnot: color 
            Determines the colour of annotion. Input to the `matplotlib.pyplot.annotate()` function for the `c` parameter.
            
            Passed to `drawAnnotation()`.

        annotLabel    : bool
            If True, will print annotation label along with the annotation line. Used only if annotate is True.

            Passed to `drawAnnotation()`.

        ylim    : (float, float) or None
            (min, max) limits for the y axis.
            
            If None, will be directly interpreted from the data.

        annotAlpha    : float >= 0
            Controls opacity of the annotation line drawn. Value should range from 0-1, both inclusive

            If `annotate` is True, sent to `drawAnnotation()`.

        yAnnot    : float
            Value ranging from 0-1, both inclusive. 
            
            Indicating where the label should occur on the y-axis. 0 indicates the lower ylim, 1 indicates the higher ylim.

            If `annotate` is True, sent to `drawAnnotation()`.

        sizeAnnot    : number 
            Font size for annotated text.

            If `annotate` is True, sent to `drawAnnotation()`.
        
    Returns
    -------
        ax    : matplotlib.axes.Axes
            Axes object with plot

    '''
    startTime = startTime + vidOffset   # convert startTime from time in audio to time in video. See parameter definition of `videoOffset` for more clarity.
    duration = duration
    movements = pd.read_csv(annotationFile, header=[0, 1])
    lWrist = movements[lWristCol][wristAxis].values[startTime*vidFps:int((startTime+duration)*vidFps)]
    rWrist = movements[rWristCol][wristAxis].values[startTime*vidFps:int((startTime+duration)*vidFps)]
    xvals = np.linspace(startTime, startTime+duration, vidFps*duration, endpoint=False)

    # if ax is None, use plt.gca()
    ax = __check_axes(ax)
    ax.plot(xvals, lWrist, label='Left Wrist')
    ax.plot(xvals, rWrist, label='Right Wrist')
    ax.set(xlabel='Time Stamp (s)' if xlabel else '', 
    ylabel='Wrist Position' if ylabel else '', 
    title=title, 
    xlim=(startTime, startTime+duration), 
    xticks=(np.arange(math.ceil(startTime), startTime+duration, freqXLabels)), 
    xticklabels=(np.arange(math.ceil(startTime), duration+startTime, freqXLabels) )if xticks else [],
    ylim=ylim if ylim is not None else ax.get_ylim()
    )
    if not yticks:
        ax.set(yticklabels=[])
    ax.invert_yaxis()    # inverst y-axis to simulate the height of the wrist that we see in real time
    ax.legend()
    if annotate:
        ax = drawAnnotation(cyclePath, onsetPath, onsetTimeKeyword, onsetLabelKeyword, numDiv, startTime-vidOffset, duration, ax, c=cAnnot, annotLabel=annotLabel, alpha=annotAlpha, yAnnot=yAnnot, sizeAnnot=sizeAnnot)
    return ax

# ANNOTATION FUNCTION
def annotateInteraction(axs, keywords, cs, interactionFile, startTime, duration):
    '''Adds interaction annotation to the axes given. 

    Height of each interaction is set randomly using `numpy.random.random()`. This is to prevent clashes between overlapping interaction annotations. #TODO: change this to be more systematic#

    Used in fig 3.

    Parameters
    ----------
        axs    : list of matplotlib.axes.Axes objects
            List of axs to add annotation to.

        keywords    : list
            Keyword corresponding to each Axes object. Value appearing in the 'Type' column in `interactionFile`. 
            
            ..note::
                If len(keywords) = len(axs) + 1, the last keyword is plotted in all Axes objects passed.

        cs    : list 
            List of colours associated with each keyword.

        interactionFile    : str
            Path to csv file with the annotation of the interactions.

        startTime    : float >= 0
            Time to start reading the audio.

        duration    : float >= 0 
            Length of audio to consider

    Returns
    -------
        axs    : matplotlib.axes.Axes
            List of axes with annotation of interaction
    '''

    annotations = pd.read_csv(interactionFile, header=None)
    annotations.columns = ['Type', 'Start Time', 'End Time', 'Duration', 'Label']
    annotations = annotations.loc[((annotations['Start Time'] >= startTime) & (annotations['Start Time'] <= startTime+duration)) &
                                ((annotations['End Time'] >= startTime) & (annotations['End Time'] <= startTime+duration))
                                ]
    for i, keyword in enumerate(keywords):
        if i < len(axs):
            # keyword corresponds to a particular axis
            for _, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
                rand = np.random.random()# random vertical displacement for the label
                lims = axs[i].get_ylim()
                axs[i].annotate('', xy=(annotation['Start Time'], rand*(lims[1] - lims[0] - 100) + lims[0] + 50), xytext=(annotation['End Time'], rand*(lims[1] - lims[0] - 100) + lims[0] + 50), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec': cs[i], 'fc': cs[i]})
                axs[i].annotate(annotation['Label'], (annotation['Start Time'] +annotation['Duration']/2, rand*(lims[1] - lims[0] - 100) + lims[0] + 150), ha='center')
        else:
            # keyword corresponds to all axes
            for ax in axs:
                for _, annotation in annotations.loc[annotations['Type'] == keyword].iterrows():
                    rand = np.random.random()# random vertical displacement for the label
                    ax.annotate('', xy=(annotation['Start Time'], rand*(lims[1] - lims[0] - 100) + lims[0] + 50), xytext=(annotation['End Time'], rand*(lims[1] - lims[0] - 100) + lims[0] + 50), arrowprops={'headlength': 0.4, 'headwidth': 0.2, 'width': 3, 'ec':cs[i], 'fc': cs[i]})
                    ax.annotate(annotation['Label'], (annotation['Start Time'] + annotation['Duration']/2, rand*(lims[1] - lims[0] - 100) + lims[0] + 150), ha='center') 
    return axs

def drawHandTap(ax, handTaps, c='purple'):
    '''Plots the hand taps as vertical lines on the Axes object `ax`. 
    
    Used in fig 9.
    
    Parameters
    ----------
        ax    : matplotlib.axes.Axes or None
            Axes object to add hand taps to

            If None, will plot on `plt.gca()`.

        handTaps    : ndarray
            Array of hand tap timestamps.

        c    : color
            Color of the line

            Passed to `plt.axes.Axes.axvline()`.

    Returns
    -------
        matplotlib.axes.Axes
            Plot with lines
    '''
    for handTap in handTaps:
        ax.axvline(handTap, linestyle='--', c=c, alpha=0.6)
    return ax

# AUDIO MANIPULATION
def generateVideoWSquares(vid_path, tapInfo, dest_path='Data/Temp/vidWSquares.mp4', vid_size=(720, 576)):
    '''Function to genrate a video with rectangles for each hand tap. 
    
    Used in fig 9.
    
    Parameters
    ----------
        vid_path    : str
            Path to the original video file.

        tapInfo    : list
            List of metadata associated with each handtap.
            
            Metadata for each handtap consists of: 
                - time    : float
                    time stamp of hand tap (in seconds).
                - keyword    : str    
                    keyword specifying which hand tap to consider
                - (pos1, pos2)    : ((float, float), (float, float))
                    (x, y) coordinates of opposite corners of the box to be drawn.
                - color    : (int, int, int)
                    tuple with RGB values associated with the colour #TODO: try to add string input also here#

        dest_path    : str
            File path to save video with squares.

        vid_size    : (int, int)
            (width, height) of video to generate #TODO : confirm that this is in pixels#

    Returns
        None
    '''

    cap_vid = cv2.VideoCapture(vid_path)
    fps = cap_vid.get(cv2.CAP_PROP_FPS)
    framesToDraw = defaultdict(list)   # dictionary with frame numbers as keys and properties of square box to draw as list of values
    for timeRow in tapInfo:
        framesToDraw[int(np.around(timeRow[0]*fps))] = timeRow[1:]
    output = cv2.VideoWriter(dest_path, cv2.VideoWriter_fourcc(*"XVID"), fps, vid_size)
    i = 0
    # generate video
    while(cap_vid.isOpened()):
        ret, frame = cap_vid.read()
        if ret == True:
            i+=1
            if i in framesToDraw.keys():
                frame = cv2.rectangle(frame, framesToDraw[i][1][0], framesToDraw[i][1][1], tuple([int(x) for x in framesToDraw[i][2]][::-1]), 3)    # converting color from BGR to RGB
            output.write(frame)
        else:
            # all frames are read
            break
    cap_vid.release()
    output.release()

def combineAudioVideo(vid_path='Data/Temp/vidWSquares.mp4', audio_path='audioWClicks.wav', dest_path='Data/Temp/FinalVid.mp4'):
    '''Function to combine audio and video into a single file. 
    
    Used in fig 9.

    Parameters
    ----------
        vid_path    : str
            File path to the video file with squares.
        
        audio_path    : str
            File path to the audio file with clicks.

        dest_path    : str
            File path to store the combined file at.

    Returns
    -------
        None

    '''
    
    vid_file = ffmpeg.input(vid_path)
    audio_file = ffmpeg.input(audio_path)
    (
        ffmpeg
        .concat(vid_file.video, audio_file.audio, v=1, a=1)
        .output(dest_path)
        .overwrite_output()
        .run()
    )
    print('Video saved at ' + dest_path)

def generateVideo(annotationFile, onsetKeywords, vidPath='Data/Temp/VS_Shree_1235_1321.mp4', tempFolder='Data/Temp/', pos=None, cs=None):
    '''Function to generate video with squares and clicks corresponding to hand taps. 
    
    Used in fig 9.
    
    Parameters
    ----------
        annotationFile    : str
            File path to the annotation file with hand tap timestamps

        onsetKeywords    : list
            List of column names to read from `annotationFile`.

        vidPath    : str
            File path to original video file.

        tempFolder    : str
            File path to temporary directory to store intermediate audio and video files in.

        pos    : list
            list of [pos1, pos2] -> 2 opposite corners of the box for each keyword 

        cs    : list
            list of [R, G, B] colours used for each keyword
    
    Returns
        None
    '''
    annotations = pd.read_csv(annotationFile)
    timeStamps = []
    for i, keyword in enumerate(onsetKeywords):
        for timeVal in annotations[keyword].values[~np.isnan(annotations[keyword].values)]:
            timeStamps.append([timeVal, keyword, pos[i], cs[i]])
    timeStamps.sort(key=lambda x: x[0])

    # generate video 
    generateVideoWSquares(vid_path=vidPath, timeStamps=timeStamps, dest_path=os.path.join(tempFolder, 'vidWSquares.mp4'))

    # generate audio
    playAudioWClicks(audioPath=vidPath, onsetFile=annotationFile, onsetLabels=onsetKeywords, destPath=os.path.join(tempFolder, 'audioWClicks.wav'))

    # combine audio and video
    combineAudioVideo(vid_path=os.path.join(tempFolder, 'vidWSquares.mp4'), audio_path=os.path.join(tempFolder, 'audioWClicks.wav'), dest_path=os.path.join(tempFolder, 'finalVid.mp4'))

'''
References
[1] Rao, P., Vinutha, T.P. and Rohit, M.A., 2020. Structural Segmentation of Alap in Dhrupad Vocal Concerts. Transactions of the International Society for Music Information Retrieval, 3(1), pp.137–152. DOI: http://doi.org/10.5334/tismir.64
[2] Meinard Müller and Frank Zalkow: FMP Notebooks: Educational Material for Teaching and Learning Fundamentals of Music Processing. Proceedings of the International Conference on Music Information Retrieval (ISMIR), Delft, The Netherlands, 2019.
[3] T.P. Vinutha, S. Suryanarayana, K. K. Ganguli and P. Rao " Structural segmentation and visualization of Sitar and Sarod concert audio ", Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR), Aug 2016, New York, USA
'''
