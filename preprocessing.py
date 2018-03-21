import librosa

#"sample_data/dev-clean/LibriSpeech/dev-clean/84/121123/84-121123-0000.flac"

def mfcc(path, frame_length, hop_length):
    y, sr = librosa.load(path)
    #print y

    #print librosa.core.get_duration(y=y, sr=sr)

    frames = librosa.util.frame(y=y, frame_length=frame_length, hop_length=hop_length)

    mfcc = [librosa.feature.mfcc(y=frames[value], sr=sr, n_mfcc=12) for value in range (0, frames.shape[0])]

    return (mfcc)
