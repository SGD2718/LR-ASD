import numpy as np
from python_speech_features import mfcc
import base
import sigproc
from expected import expected
from librosa import load
from test_input import test_input


def generate_and_save_data():
    """
    Generates a test signal and its corresponding MFCC features,
    then saves both to files for a C++ program to read and verify.
    """
    print("--- Python: Generating and saving test data ---")

    # --- Test Signal Parameters ---
    samplerate = 16000
    # Use float64 to match C++ double precision
    t = np.linspace(0., 1., samplerate, endpoint=False, dtype=np.float64)
    signal = np.sin(2 * np.pi * 440. * t) + 0.5 * np.sin(2 * np.pi * 880. * t)

    # --- MFCC Parameters (must be identical in C++ test) ---
    winlen = 0.025
    winstep = 0.01
    numcep = 13
    nfilt = 26
    nfft = 512
    lowfreq = 0
    highfreq = None  # Let the library use the default
    preemph = 0.97
    ceplifter = 22
    appendEnergy = True

    # --- Generate MFCCs using the reference library---
    # We use np.hamming for a consistent, standard window function
    py_mfcc = mfcc(signal,
                   samplerate=samplerate,
                   winlen=winlen,
                   winstep=winstep,
                   numcep=numcep,
                   nfilt=nfilt,
                   nfft=nfft,
                   lowfreq=lowfreq,
                   highfreq=highfreq,
                   preemph=preemph,
                   ceplifter=ceplifter,
                   appendEnergy=appendEnergy)

    # --- Save to Files ---
    # Save the arrays as raw binary (double precision)
    signal.astype('float64').tofile('signal.bin')
    py_mfcc.astype('float64').tofile('reference_mfcc.bin')

    # Save the dimensions of the MFCC matrix to a text file for easy reading in C++
    with open('reference_dims.txt', 'w') as f:
        f.write(f'{py_mfcc.shape[0]},{py_mfcc.shape[1]}')

    print(f"Signal ({signal.shape[0]} samples) saved to signal.bin")
    print(f"Reference MFCCs ({py_mfcc.shape[0]}x{py_mfcc.shape[1]}) saved to reference_mfcc.bin")
    print(f"Reference dimensions saved to reference_dims.txt")
    print("\nData generation complete. You can now run the C++ test harness.")


if __name__ == "__main__":
    # generate_and_save_data()
    #audio, sr = load('demo/0004/pyavi/audio.wav', sr=16000, mono=True)
    res = base.mfcc(np.zeros((10000,), float), nfilt=32)
    np.printoptions(threshold=10000)
    print(f"res:\n{res}")
    #print(f"expected:\n{expected}")
    #np.testing.assert_allclose(res, expected, rtol=1e-3, atol=1e-3)

