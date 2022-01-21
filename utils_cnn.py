import os
import torch
import numpy as np
import librosa
import torch.nn as nn
import torch.nn.functional as F

def computeLayRatio(audio=None, source='vocal', mode='test', concertName=None, audioPath=None, gpuID=None, pretrainedModelDir='../pretrained_models', hopDur=0.5, smoothTol=5, ):
    '''
    Compute frame-wise lay ratio (surface tempo multiple) estimates using pre-trained models from [1]. There are separate models available for mixture concert audios, as well as source separated vocals and pakhawaj streams. 
    
    Parameters:
        audio (1d array): audio signal sampled at 16kHz
        source (str): one of 'mix', 'voc' or 'pakh' (mixture, vocal or pakhawaj)
        mode (str): 'eval' to predict on audio from original dataset, 'test' for new audios not in dataset
        concertName (str): name of concert audio as it appears in  original dataset; only required if mode is eval
        audioPath (str): path to audio file, if signal not provided
        gpuID (int): serial id/number of gpu device to use, if available
        pretrainedModelDir (str): path to pretrained models (provided in this library)
        hopDur (int or float): intervals at which lay ratios are to be obtained on the audio
        smoothTol (int or float): minimum duration for which a lay ratio value has to be consistently predicted to be retained during the smoothing. If lesser, the prediction is considered erroneous and replaced with neighbouring values.
    
    Returns:
        Frame-wise lay ratio (1d array)
    '''
    fs=16e3

    if (audio is None) and (audioPath is None):
        print('Provide one of audio signal or path to audio')
        raise
    elif audio is None:
        #load input audio
        audio,_ = librosa.load(audioPath, sr=fs)

    #use GPU or CPU
    if gpuID is None:
        device = torch.device("cpu")
    else:
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            device = torch.device("cuda:%d"%gpuID)
        else:
            print("no gpu device found; using cpu")
            device = torch.device("cpu")

    #melgram parameters
    winsize_sec = 0.04
    winsize = int(winsize_sec*fs)
    hopsize_sec = 0.02
    hopsize = int(hopsize_sec*fs)
    nfft = int(2**(np.ceil(np.log2(winsize))))

    input_len_sec = 8
    input_len = int(input_len_sec/hopsize_sec)
    input_hop_sec = hopDur
    input_hop = int(input_hop_sec/hopsize_sec)
    input_height = 40

    #minimum section duration for smoothing s.t.m. estimates
    min_sec_dur = smoothTol #in seconds
    min_sec_dur /= input_hop_sec

    #convert to mel-spectrogram
    melgram = librosa.feature.melspectrogram(audio, sr=fs, n_fft=nfft, hop_length=hopsize, win_length=winsize, n_mels=input_height, fmin=20, fmax=8000)
    melgram = 10*np.log10(1e-10+melgram)
    melgram_chunks = makeChunks(melgram, input_len, input_hop)

    #load model
    classes_dict = {'voc':[1.,2.,4.,8.],'pakh':[1.,2.,4.,8.,16.],'mix':[1.,2.,4.,8.,16.]}
    model_ids = [0,1,2]
    model={}
    for i in model_ids:
        model_path=os.path.join(pretrained_model_dir, source, 'saved_model_fold_%d.pt'%i)
        model[i]=buildModel(input_height,input_len,len(classes_dict[source])).float().to(device)
        model[i].load_state_dict(torch.load(os.path.join(model_path),map_location=device))
        model[i].eval()

    #load splits data if mode is eval
    if mode=='eval':
        splits_data = {}
        for fold in range(3):
            splits_data[fold] = np.loadtxt('../splits/%s/fold_%s.csv'%(source,fold), dtype=str)
        boundaries = np.loadtxt('../annotations/%s/stm.csv'%concertName, usecols=(0,1))

    #predict lay ratio versus time
    stm_vs_time = []
    for i_chunk, chunk in enumerate(melgram_chunks):
        model_in = (torch.tensor(chunk).unsqueeze(0)).unsqueeze(1).float().to(device)

        avg_out = []
        model_out = {}
        if mode == 'eval':
            i_fold = getFoldNum(i_chunk, hopsize_sec, splits_data, boundaries)
            avg_out = (nn.Softmax(1)(model[i_fold].forward(model_in))).detach().numpy()

        elif mode=='test':
            for i in model_ids:
                model_out[i] = (nn.Softmax(1)(model[i].forward(model_in))).detach().numpy()
                if len(avg_out) == 0:
                    avg_out = model_out[i].copy()
                else:
                    avg_out += model_out[i]
            avg_out/=len(model_ids)

        stm_vs_time.append(np.argmax(avg_out))

    #smooth predictions with a minimum section duration of 5s
    stm_vs_time = smoothBoundaries(stm_vs_time,min_sec_dur)
    
    return stm_vs_time

'''Model-related class definitions'''
class sfModule(nn.Module):
    '''Short-filter module used for the early layers in the CNN; inherits torch.nn.Module class properties.
    
    Attributes:
        n_ch_in (int): number of channels in the input mel-spectrogram

    Methods:
        forward: processes the input through the sf module layers    
    '''
    def __init__(self,n_ch_in):
        '''Define the layers in the sfmodule
        
        Parameters:
            n_ch_in (int): number of channels in the input mel-spectrogram        
        '''
        super(sfModule, self).__init__()
        n_filters=16
        self.bn1=nn.BatchNorm2d(n_ch_in,track_running_stats=True)
        self.conv1=nn.Conv2d(n_ch_in, n_filters, (1,5), stride=1, padding=(0,2))
        self.elu=nn.ELU()
        self.do=nn.Dropout(p=0.1)

    def forward(self,x):
        '''Passes the input through the following layers: batch-norm, 2d convolution, ELU activation, dropout.
        
        Parameters:
            x (3d array): input mel-spectrogram
            
        Returns:
            y (3d array): output of the dropout layer
        '''
        y=self.bn1(x)
        y=self.conv1(x)
        y=self.elu(y)
        y=self.do(y)
        return y

class mfModule(nn.Module):
    '''Multi-filter module appearing after the short-filter modules; contains parallel convolution layers of different kernel sizes.

    Attributes:
        pool_height (int): height dimension of the average pool kernel
        n_ch (int): number of channels in the input to the multi-filter module
        kernel_widths (list): kernel widths of each of the parallel conv layers
        n_filters (list or tuple): (#filters in each parallel conv layer, #filters in the 1x1 conv layer following the concatenation of parallel filter outputs)

    Methods:
        forward: processes the input through the mf module layers
    '''
    def __init__(self,pool_height,n_ch,kernel_widths,n_filters):
        '''Define the layers in the mfmodule.
        
        Parameters:
            pool_height (int): height dimension of the average pool kernel
            n_ch (int): number of channels in the input to the multi-filter module
            kernel_widths (list): kernel widths of each of the parallel conv layers
            n_filters (list or tuple): (#filters in each parallel conv layer, #filters in the 1x1 conv layer following the concatenation of parallel filter outputs)
        '''
        super(mfModule, self).__init__()
        self.avgpool1=nn.AvgPool2d((pool_height,1))
        self.bn1=nn.BatchNorm2d(n_ch,track_running_stats=True)

        self.conv1s=nn.ModuleList([])
        for kw in kernel_widths:
            self.conv1s.append(nn.Conv2d(n_ch, n_filters[0], (1,kw), stride=1, padding=(0,kw//2)))

        self.do=nn.Dropout(0.5)
        self.conv2=nn.Conv2d(n_filters[0]*len(kernel_widths),n_filters[1],(1,1),stride=1)

    def forward(self,x):
        '''Passes the input through the following layers: avg-pool, batch-norm, set of parallel 2d conv layers, concatenation, dropout, 1x1 2d conv.

        Parameters:
            x (3d array): input to the multi-filter module (output of the last short-filter module)
            
        Returns:
            y (3d array): output of the 1x1 conv layer
        '''
        y=self.avgpool1(x)
        y=self.bn1(y)
        z=[]
        for conv1 in self.conv1s:
            z.append(conv1(y))
        
        #trim last column to keep width=input_len (needed if filter width is even)
        for i in range(len(z)):
            z[i]=z[i][:,:,:,:-1]

        y=torch.cat(z,dim=1)
        y=self.do(y)
        y=self.conv2(y)
        return y

class denseModule(nn.Module):
    '''Dense layer module appearing after the multi-filter modules.

    Attributes:
        n_ch_in (int): #channels in output of multi-filter module
        input_len (int): width of output of multi-filter module
        input_height (int): width of output of multi-filter module
        n_classes (int): #output classes at output of last dense layer

    Methods:
        forward: processes the input through the dense module layers
    '''
    def __init__(self,n_ch_in,input_len,input_height,n_classes):
        '''Define the layers in the dense module.
        
        Parameters:
            n_ch_in (int): #channels in output of multi-filter module
            input_len (int): width of output of multi-filter module
            input_height (int): width of output of multi-filter module
            n_classes (int): #output classes at output of last dense layer
        '''
        super(denseModule, self).__init__()
        n_linear1_in=n_ch_in*input_height
        
        self.dense_mod=nn.ModuleList([nn.AvgPool2d((1,input_len)), 
        nn.BatchNorm2d(n_ch_in,track_running_stats=True), 
        nn.Dropout(p=0.5), 
        nn.Flatten(), 
        nn.Linear(n_linear1_in,n_classes)]) 
        
    def forward(self,x):
        '''Passes the input through the following layers: avg-pool, batch-norm, dropout, flatten, linear (dense).

        Parameters:
            x (3d array): input to the dense-layer module (output of the multi-filter module)
            
        Returns:
            y (1d array): output of the dense layer (needs application of softmax to convert output class probalities)
        '''
        for layer in self.dense_mod:
            x=layer(x)
        return x

def buildModel(inputHeight,inputLen,nClasses):
    '''Put together all the different modules to build a single model that input mel-spectrogram can be passed through.

    Parameters:
        inputHeight, inputLen (int): height and width of input mel-spectrogram
        nClasses (int): #output classes

    Returns:
        model object containing all the modules (torch.nn.Sequential)        
    '''

    model=nn.Sequential()
    i_module=0
    
    #add sf layers
    sfmod_ch_sizes=[1,16,16]
    for ch in sfmod_ch_sizes:   
        sfmod_i=sfModule(ch)   
        model.add_module(str(i_module),sfmod_i)
        i_module+=1

    #add mfmods
    pool_height=5
    kernel_widths=[16,32,64,96]
    ch_in,ch_out=16,16
    mfmod_n_filters=[12,16]
    
    mfmod_i=mfModule(pool_height,ch_in,kernel_widths,mfmod_n_filters)
    model.add_module(str(i_module),mfmod_i)
    inputHeight//=pool_height
    i_module+=1

    #add densemod
    ch_in=16
    densemod=denseModule(ch_in,inputLen,inputHeight,nClasses)
    model.add_module(str(i_module),densemod)
    return model

def makeChunks(x,duration,hop):
    '''Create overlapping chunks of width N and original height, from the full audio spectrogram.
    
    Parameters:
        x (2d array): audio spectrogram
        duration (int): length of window (in frames)
        hop (int): hop duration between windows (in frames)
    '''
    n_chunks=int(np.floor((x.shape[1]-duration)/hop) + 1)
    y=np.zeros([n_chunks,x.shape[0],duration])
    for i in range(n_chunks):
        y[i]=x[:,i*hop:(i*hop)+duration]
        #normalise
        y[i]=(y[i]-np.min(y[i]))/(np.max(y[i])-np.min(y[i]))
    return y

def smoothBoundaries(x,minDur):
    '''Apply temporal smoothing to predicted lay ratio (stm) estimates by imposing minimum duration for each predicted lay ratio.

    Parameters:
        x (1d array): frame-wise lay ratio estimates
        minDur (int): minimum duration (in frames) that an estimate should last for, else it's reduced to its previous estimate

    Returns:
        x_smu (1d array): smoothed estimates
    '''

    x_smu=np.copy(x)
    prev_stm=x_smu[0]
    curr_stm_dur=1
    i=1
    while i < len(x_smu):
        if x_smu[i]!=x_smu[i-1]:
            if curr_stm_dur>=minDur:
                curr_stm_dur=1
                prev_stm=x_smu[i-1]
                
            else:
                x_smu[i-curr_stm_dur:i]=prev_stm
                curr_stm_dur=1
        else: curr_stm_dur+=1
        i+=1
    return x_smu

def getFoldNum(concertName, frameNum, hopDur, splitsData, boundaries):
    '''If running in 'eval' mode, then get the fold number that the input segment was assigned to in the original CV split of the data (see [1] for details) 
    
    Parameters:
        concertName (str): name of concert audio file in original dataset (see supplementary data of [1] for concert list)
        frameNum (int): index of the audio frame
        hopDur (float): hop duration used for short-time analysis
        splitsData (2d array): annotations file from the dataset
        boundaries (2d array): list of manually annotated section boundaries from the dataset

    Returns:
        fold_num (int): id of the fold (0, 1 or 2)
    '''

    frameNum = frameNum*hopDur
    section_idx = np.where(np.array([bounds[0]<=frameNum<=bounds[1] for bounds in boundaries])==True)[0][0]
    try:
        fold_num = np.where(np.array(['%s_%d'%(concertName, section_idx) in splitsData[i] for i in range(3)])==True)[0][0]
    except:
        print(frameNum)
        fold_num = 0
    return fold_num


'''
References

[1] Rohit M. A., Vinutha T. P., and Preeti Rao " Structural Segmentation of Dhrupad Vocal Bandish Audio Based on Tempo, "Proceedings of ISMIR, October 2020, Montreal, Canada
'''
