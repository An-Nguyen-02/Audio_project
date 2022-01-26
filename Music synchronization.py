import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import librosa.core
import librosa.display
import sounddevice as sd


def plt_t_A(A,B,fs):
    """
    This function use to plot time-amplitude graph of A and B
    :param A: audio A as np.array with shape [sth,]
    :param B: audio B as np.array with shape [sth,]
    :param fs: sampling rate of both A and B
    :return: none
    """
    fig, ax = plt.subplots(nrows=2, sharex=True, sharey=False)
    librosa.display.waveshow(A, sr=fs, ax=ax[0])
    ax[0].set(title='Audio $A$')
    ax[0].label_outer()
    librosa.display.waveshow(B, sr=fs, ax=ax[1])
    ax[1].set(title='Audio $B$')
    plt.show()
def extract_melspec(A,B,fs,hop):
    ""
    A_melspec = librosa.feature.melspectrogram(y=A,sr= fs,hop_length=hop)

    B_melspec = librosa.feature.melspectrogram(y=B, sr=fs, hop_length=hop)
    A_melspec[A_melspec == 0] = 10**-7
    B_melspec[B_melspec == 0] = 10 ** -7
    return A_melspec, B_melspec

def extract_chroma(A,B,fs,hop):
    """
    Use to extract chroma of A and B
    :param A: audio A as np.array with shape [sth,]
    :param B: audio B as np.array with shape [sth,]
    :param fs: sampling rate of both A and B
    :param hop: hop length of chroma
    :return: chroma of A and chroma of B
    """
    A_chroma = librosa.feature.chroma_cqt(y=A, sr=fs,
                                             hop_length=hop)
    B_chroma = librosa.feature.chroma_cqt(y=B, sr=fs,
                                             hop_length=hop)
    # add noise to avoid NaN in cost function of cosine distance.
    B_chroma[B_chroma==0.0] = 10**-7
    A_chroma[A_chroma == 0.0] = 10 ** -7
    return A_chroma, B_chroma

def plt_chroma(A_chroma,B_chroma,hop):
    """
    Use to plot chroma of both audio A and audio B
    :param A_chroma: chroma A extracted from extract chroma
    :param B_chroma: chroma B extracted from extract chroma
    :param hop: hop length of chroma
    :return: none
    """
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 1)
    plt.title('Chroma Representation of $A$')
    librosa.display.specshow(A_chroma, x_axis='time',
                             y_axis='chroma', cmap='gray_r', hop_length=hop)
    plt.colorbar()
    plt.subplot(2, 1, 2)
    plt.title('Chroma Representation of $B$')
    librosa.display.specshow(B_chroma, x_axis='time',
                             y_axis='chroma', cmap='gray_r', hop_length=hop)
    plt.colorbar()
    plt.tight_layout()
    plt.show()

def MFCC_approach(A,B,fs,n_mels):
    mel_a = librosa.feature.melspectrogram(y=A,sr=fs,n_mels=n_mels)


def align_chroma_plt(A_chroma,B_chroma,fs,hop):
    """
    Plot the warping path between A and B
    :param A_chroma: chroma A extracted from extract chroma
    :param B_chroma: chroma B extracted from extract chroma
    :param fs: sampling rate of both A and B
    :param hop: hop length of chroma
    :return: wp_s, the original corresponding time stamp of A and B
    """
    #Align Chroma Sequence.
    # hamming
    D, wp= librosa.sequence.dtw(X=A_chroma, Y=B_chroma, metric='cosine')
    #wp_s = np.asarray(wp) * hop / fs
    wp_s = librosa.frames_to_time(wp, sr=fs, hop_length=hop)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    librosa.display.specshow(D, x_axis='time', y_axis='time',
                             cmap='gray_r', hop_length=hop)
    imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
                     origin='lower', interpolation='nearest', aspect='auto')
    ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
    plt.title('Warping Path on Acc. Cost Matrix $D$')
    plt.colorbar()
    plt.show()
    return wp_s

def visual_connect_time_domain(A,B,fs,wp_s,n_arrows):
    """
    Plot the corresponding time stamp
    :param A: audio A as np.array with shape [sth,]
    :param B: audio B as np.array with shape [sth,]
    :param fs: sampling rate of both A and B
    :param wp_s: the time stamp may have one points to many; array of shape [sth,2]
    :param n_arrows: number of line take from wp_s
    :return: none
    """
    # alternate visualization in time domain
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(8, 4))

    # Plot B
    librosa.display.waveshow(B, sr=fs, ax=ax2)
    ax2.set(title='Audio $B$')

    # Plot A
    librosa.display.waveshow(A, sr=fs, ax=ax1)
    ax1.set(title='Audio $A$')
    ax1.label_outer()

    for tp1, tp2 in wp_s[::len(wp_s)//n_arrows]:
        # Create a connection patch between the aligned time points
        # in each subplot
        con = ConnectionPatch(xyA=(tp1, 0), xyB=(tp2, 0),
                              axesA=ax1, axesB=ax2,
                              coordsA='data', coordsB='data',
                              color='r', linestyle='--',
                              alpha=0.5)
        ax2.add_artist(con)
    plt.show()

def time_stretch_func(B, fs, time_stamp, R):
    corresponding_frame = fs*time_stamp
    B_out = np.array([0])
    for i in range(np.shape(R)[0]-1):
        B_in = B[int(corresponding_frame[i][1]):int(corresponding_frame[i+1][1])]
        B_now = librosa.effects.time_stretch(B_in,1/R[i])
        B_out = np.append(B_out,B_now)
    B_out = np.delete(B_out,0)
    return B_out

def corresponding_time_stamp(A,B,fs,wp_s):
    """
    Removing lines that have single points correspond to many points. Only take
    that point and the last point on the other end
    :param A: audio A as np.array with shape [sth,]
    :param B: audio B as np.array with shape [sth,]
    :param fs: sampling rate of both A and B
    :param wp_s: the time stamp may have one points to many; array of shape [sth,2]
    :return: return time stamp without overlapping points.
    """
    wp_s = np.flip(wp_s,axis=0)
    start = 0.0
    end = 0.0
    # add fisrt corresponding line.
    time_stamp = np.array([[0, 0]])
    for frame in wp_s:
        if (start == 0):
            # for the first frame
            if (frame[0] != frame[1]):
                start = frame[0]
                end = frame[1]
                time_stamp = np.append(time_stamp, np.array([[start, end]]),
                                       axis=0)
        else:  # other frame
            # if one point a in A according to many points in B,
            # take the first point of A and the last point of B to be the line.
            if np.all(start == frame[0] and end != frame[1]):
                end = frame[1]
            elif np.all(start != frame[0] and end == frame[1]):
                start = frame[0]
            else:
                start = frame[0]
                end = frame[1]
                time_stamp = np.append(time_stamp, np.array([[start, end]]),
                                       axis=0)
    time_stamp = np.append(time_stamp,np.array([[np.shape(A)[0]//fs,np.shape(B)[0]//fs]]),axis=0)
    return time_stamp

def call_stretch_arr(time_stamp):
    """
    Return the R array from the given time-stamp
    :param time_stamp: the time-stamp that already eliminated single points many lines
    :return: factor R
    """
    R = np.array([0])
    for i in range(np.shape(time_stamp)[0]-1):
        current_R = (time_stamp[i+1][0]-time_stamp[i][0])/(time_stamp[i+1][1]-time_stamp[i][1])
        R = np.append(R,current_R)
    R = np.delete(R,0)
    return R

def main():
    # Import audio A and B, fs=fs1
    Audio_A= 'A.wav';
    Audio_B= 'B.wav';
    A, fs = librosa.load(Audio_A)
    B, fs1 = librosa.load(Audio_B)
    # Plotting time-amplitude graph of A and B
    #plt_t_A(A,B,fs)
    hop = 1024
    A_chroma, B_chroma = extract_chroma(A,B,fs,hop)
    #A_chroma, B_chroma = extract_melspec(A, B, fs, hop)
    #plt_chroma(A_chroma,B_chroma,hop)
    wp_s = align_chroma_plt(A_chroma, B_chroma, fs, hop)
    # Because wp_s contains all the correspondings line of the 2 audio
    # so representing all of the lines are most correct but may look overwhelming
    n_arrows = np.shape(wp_s)[0]
    # Visualize the connection between 2 audio graph
    visual_connect_time_domain(A,B,fs,wp_s,n_arrows)
    # Because there are few single points correspond to many points
    # So we simplify it by just having each point correspond to another point
    time_stamp = corresponding_time_stamp(A,B,fs,wp_s)
    # If you want to try mel spectrum, use this code line because it will cause R factor error if not do.
    #time_stamp = np.delete(time_stamp,0,0)
    # calculate different time stretch factor for each frame.
    R = call_stretch_arr(time_stamp)
    # Stretch the signal frame by frame
    B_out = time_stretch_func(B,fs,time_stamp,R)
    plt_t_A(A,B_out,fs)
    A_chroma, B_out_chroma = extract_chroma(A, B_out, fs, hop)
    wp_s_out = align_chroma_plt(A_chroma, B_out_chroma, fs, hop)
    #visual_connect_time_domain(A,B_out,fs,wp_s_out,np.shape(wp_s_out)[0])
    #sd.play(B_out2,samplerate=fs,blocking=True)
if __name__ == '__main__':
    main()