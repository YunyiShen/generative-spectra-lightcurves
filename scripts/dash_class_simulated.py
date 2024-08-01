import numpy as np
import astrodash
from tqdm import tqdm
import matplotlib.pyplot as plt

def extract_just_class(stringarray):
    '''
    Args: a np.array of strings, that has form of '<something I want>: <something I don't want>'
    Return: a np.array with same size, just having '<something I want>'

    This is used to remove the age from astrodash, before we get the data. 
    '''
    def extract_desired_part(string): # this will just remove the age from astrodash, before I get the age data
        return string.split(":")[0].strip()
    vectorized_func = np.vectorize(extract_desired_part)
    return vectorized_func(stringarray)

def clean_labels(stringarray):
    return np.array([i.replace("IIL", "II").replace("-norm", "").replace("-csm","-CSM").replace("-broad", "-BL").replace("-02cx","x") for i in stringarray])



def aggregate_softmax(SNlabels, softmaxscore):
    '''
    this aggregate softmax scores based on class (since we remove ages)
    Args:
        SNlabels: np.array with just type labels, ages removed
        softmaxscore: softmax score with ages and type labels, will be aggregated based on type
    '''
    unique_labels = np.unique(SNlabels)
    aggregated_values = np.zeros_like(unique_labels, dtype=softmaxscore.dtype)
    for i, label in enumerate(unique_labels):
        indices = np.where(SNlabels == label)  # Get indices where label appears
        aggregated_values[i] = np.sum(softmaxscore[indices])  # Aggregate values
    return unique_labels, aggregated_values

def predict_just_class_for_one_star(filenames, redshift, 
                        classifyHost=False, knownZ=True, 
                        smooth=6, rlapScores=True, aggrclass = True):
    classification = astrodash.Classify([filenames], [redshift], classifyHost=classifyHost, 
                                        knownZ=knownZ, smooth = smooth, rlapScores = rlapScores)
    
    bestTypes, softmaxes, bestLabels, inputImages, inputMinMaxIndexes = classification._input_spectra_info()
    if aggrclass:
        SNlabels, aggr_softmax = aggregate_softmax(extract_just_class(bestTypes[0]), softmaxes[0])
        SNlabels = clean_labels(SNlabels)
        return SNlabels, aggr_softmax
    return bestTypes[0], softmaxes[0]

all_types = ['SN II', 'SN Ia', 'SN Ib', 'SN Ic']
for thetypegt in all_types:

    wavelength = np.load(f"{thetypegt}_wavelength.npy")
    samples = np.load(f"{thetypegt}_simulated.npy")

    #breakpoint()

    Ia_counter = 0
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    for i in range(50):
        #breakpoint()
        spectra = np.array([[wavelength[i][wavelength[i]>1.]],[samples[i][wavelength[i]>1.]]])[:,0,:]
        thetype, thescore =  predict_just_class_for_one_star(spectra, 0.0)
        #print(thetype[np.argsort(thescore)], thescore[np.argsort(thescore)])
        if thetypegt.replace("SN ","") in thetype[np.argsort(thescore)][-1]:
            Ia_counter += 1
            ax[0].plot(wavelength[i][wavelength[i]>1.], samples[i][wavelength[i]>1.], alpha=0.8)
        else:
            ax[1].plot(wavelength[i][wavelength[i]>1.], samples[i][wavelength[i]>1.], alpha=0.8)

    ax[0].set_xlabel("Wavelength")
    ax[0].set_ylabel("Flux")
    ax[1].set_xlabel("Wavelength")
    ax[1].set_ylabel("Flux")
    ax[0].set_title(f"Simulated SN {thetypegt} samples (dash confirmed)")
    ax[1].set_title(f"Simulated SN {thetypegt} samples (dash disagreed)")
    plt.show()
    plt.savefig(f"{thetypegt}_simulated_dash.png")
    print(f"number of hits {Ia_counter} in {thetypegt}")
    #print(Ia_counter)