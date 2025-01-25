import numpy as np
import matplotlib.pyplot as plt

def display(data, y):
    img_num = min(len(data), 32)
    c, h, w = data[0].shape
    plt.figure(figsize=(img_num/4, img_num/2))
    for i in range(img_num):
        plt.subplot(8,4,i+1)
        plt.imshow(data[i].reshape(h, w, c), cmap='gray')
        plt.title(f'predicted class: {y[i].item()}')
        plt.axis('off')
    plt.show()

def display_stats(training_stats):
    epochs = np.arange(1, len(training_stats["train_acc"])+1)

    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(epochs, training_stats['train_acc'])
    plt.plot(epochs, training_stats['valid_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.subplot(212)
    plt.plot(epochs, training_stats['train_losses'])
    plt.plot(epochs, training_stats['valid_losses'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()