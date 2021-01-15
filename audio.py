import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import librosa.display

blues_file =  'genres/blues/blues.00000.wav'
classical_file = 'genres/classical/classical.00000.wav'

x_blues, sr_blues = librosa.load(blues_file, duration=30)
x_classical, sr_classical = librosa.load(classical_file, duration=30)
ipd.Audio(x_blues, rate=sr_blues)
ipd.Audio(x_classical, rate=sr_classical)

n_mfcc = 12
mfcc_blues = librosa.feature.mfcc(x_blues, sr=sr_blues, n_mfcc=n_mfcc).T
mfcc_classical = librosa.feature.mfcc(x_classical, sr=sr_classical, n_mfcc=n_mfcc).T

scaler = StandardScaler()
mfcc_blues_scaled = scaler.fit_transform(mfcc_blues)
mfcc_classical_scaled = scaler.fit_transform(mfcc_classical)

mfcc_blues_scaled.mean(axis=0)
mfcc_classical_scaled.mean(axis=0)

features = numpy.vstack((mfcc_blues_scaled, mfcc_classical_scaled))
labels = numpy.concatenate((numpy.zeros(len(mfcc_blues_scaled)),
                            
                            numpy.ones(len(mfcc_classical_scaled))))

model = SVC()
model.fit(features, labels)

predicted_labels = model.predict(mfcc_classical_scaled)
score = model.score(features, labels)
unique_labels, unique_counts = numpy.unique(predicted_labels, return_counts=True)

print(unique_labels, unique_counts)