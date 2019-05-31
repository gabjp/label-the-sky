import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def acc_mag_plot(magnitudes, accuracies, class_names, title=None):
    assert len(accuracies)==len(class_names)
    fig, ax = plt.subplots()
    for ix in range(len(accuracies)):
        scatter = ax.scatter(magnitudes, accuracies[ix], label=class_names[ix], alpha=0.7)
    ax.legend()
    ax.set_ylabel('accuracy')
    ax.set_xlabel('magnitude')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    class_names = ['galaxy', 'star', 'qso']

    # accuracies = [
    #     [1.0, 1.0, 0.9887640449438202, 1.0, 1.0, 0.9958847736625515, 0.9966329966329966, 0.9911242603550295, 0.9939577039274925, 0.9895833333333334, 0.986013986013986, 0.9550827423167849, 0.9337606837606838, 0.8907563025210085, 0.8503118503118503, 0.784741144414169, 0.796812749003984, 0.7241379310344828, 0.7621621621621621, 0.6666666666666666, 0.6428571428571429, 0.5714285714285714],
    #     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.996268656716418, 1.0, 0.9961832061068703, 0.996415770609319, 0.9601593625498008, 0.8960573476702509, 0.9070796460176991, 0.8272727272727273, 0.8863636363636364, 0.7558139534883721, 0.8247422680412371, 0.7586206896551724, 0.7272727272727273, 0.6875, 0.42857142857142855],
    #     [1.0, 1.0, 0.0, 0.4, 0.6666666666666666, 0.75, 1.0, 1.0, 0.8, 0.7894736842105263, 0.8709677419354839, 0.6666666666666666, 0.66, 0.647887323943662, 0.5662650602409639, 0.5833333333333334, 0.5982905982905983, 0.47540983606557374, 0.42758620689655175, 0.29411764705882354, 0.43902439024390244, 0.3888888888888889]
    # ]
    # magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    # acc_mag_plot(magnitudes, accuracies, class_names, 'image classifier trained with magnitudes in [16,19]')

    accuracies = [
        [0.9230769230769231, 0.9113924050632911, 0.9550561797752809, 0.9576271186440678, 0.9724137931034482, 0.9423868312757202, 0.9595959595959596, 0.9230769230769231, 0.9335347432024169, 0.9296875, 0.9114219114219114, 0.8983451536643026, 0.9038461538461539, 0.9285714285714286, 0.9106029106029107, 0.9073569482288828, 0.896414342629482, 0.9386973180076629, 0.9135135135135135, 0.8690476190476191, 0.9761904761904762, 1.0],
        [0.9225352112676056, 0.9115646258503401, 0.841726618705036, 0.9195402298850575, 0.9263157894736842, 0.898989898989899, 0.8854625550660793, 0.8805970149253731, 0.8622047244094488, 0.8282442748091603, 0.7706093189964157, 0.7848605577689243, 0.7383512544802867, 0.6769911504424779, 0.6545454545454545, 0.6590909090909091, 0.6453488372093024, 0.7731958762886598, 0.6206896551724138, 0.5681818181818182, 0.375, 0.2857142857142857],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.25, 0.6666666666666666, 0.5, 0.4, 0.47368421052631576, 0.5483870967741935, 0.5952380952380952, 0.66, 0.7323943661971831, 0.5903614457831325, 0.6388888888888888, 0.7094017094017094, 0.5819672131147541, 0.6689655172413793, 0.7352941176470589, 0.7317073170731707, 0.7777777777777778]
    ]
    magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    acc_mag_plot(magnitudes, accuracies, class_names, 'catalog classifier trained with all magnitudes')

    # accuracies = [
    #     [0.98076923076923073, 0.98734177215189878, 0.9550561797752809, 0.98305084745762716, 0.98620689655172411, 0.98353909465020573, 0.98653198653198648, 0.98224852071005919, 0.98791540785498488, 0.96875, 0.98135198135198132, 0.97399527186761226, 0.97222222222222221, 0.94957983193277307, 0.96881496881496887, 0.92915531335149859, 0.92430278884462147, 0.95019157088122608, 0.91891891891891897, 0.91666666666666663, 0.90476190476190477, 0.8571428571428571],
    #     [1.0, 0.99319727891156462, 0.98561151079136688, 0.9885057471264368, 0.98947368421052628, 0.98989898989898994, 0.99118942731277537, 0.9925373134328358, 0.97637795275590555, 0.95038167938931295, 0.94623655913978499, 0.94023904382470125, 0.91756272401433692, 0.92035398230088494, 0.82272727272727275, 0.85795454545454541, 0.77906976744186052, 0.82474226804123707, 0.73563218390804597, 0.63636363636363635, 0.4375, 0.42857142857142855],
    #     [0.0, 1.0, 0.0, 0.59999999999999998, 0.66666666666666663, 0.25, 1.0, 0.5, 0.5, 0.63157894736842102, 0.61290322580645162, 0.40476190476190477, 0.47999999999999998, 0.40845070422535212, 0.36144578313253012, 0.3611111111111111, 0.33333333333333331, 0.36885245901639346, 0.28965517241379313, 0.40196078431372551, 0.36585365853658536, 0.3888888888888889]
    # ]
    # magnitudes = [16.25, 16.5, 16.75, 17.0, 17.25, 17.5, 17.75, 18.0, 18.25, 18.5, 18.75, 19.0, 19.25, 19.5, 19.75, 20.0, 20.25, 20.5, 20.75, 21.0, 21.25, 21.5]

    # acc_mag_plot(magnitudes, accuracies, class_names, 'image classifier trained with all magnitudes')