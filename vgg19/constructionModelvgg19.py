import random, os, glob
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Dense, Conv2D, MaxPool2D, Flatten, Reshape, BatchNormalization, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG19
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt
import IPython.display as ipd
from sklearn.metrics import confusion_matrix
import seaborn as sns

def setRandom():
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

filePath = "/kaggle/input/gtzan-fixed/Data/genres_original/blues/blues.00000.wav" # an example file

file, samplingRate = librosa.load(filePath)
example, _ = librosa.effects.trim(file)

hopLength = 512

spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=example, sr=samplingRate, n_fft=2048, hop_length=hopLength, n_mels=128, power=4.0), ref=np.max)

plt.figure(figsize=(3, 2))
librosa.display.specshow(spectrogram, sr=samplingRate, hop_length=hopLength, x_axis="off", y_axis="off")
ipd.Audio(example, rate=samplingRate)
source = "/kaggle/input/gtzan-fixed/Data/images_original/" # source folder path
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

split = [80, 9, 10]
train, val, test = {}, {}, {}
trainLen, valLen, testLen = {}, {}, {}
dictionaries = [train, val, test]

for d in dictionaries:
    if d == train: num = slice(0, split[0])
    elif d == val: num = slice(split[0], split[0] + split[1])
    else: num = slice(split[0] + split[1], split[0] + split[1] + split[2])
    for genre in genres:
        path = os.path.join(source, genre)
        pngs = glob.glob(os.path.join(path, "*.png"))
        selected = pngs[num]
        d[genre] = selected

lenDictionaries = [{genre: len(d[genre]) for genre in genres} for d in dictionaries]        

batchSize = 32
genreMap = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9
}
inverseGenreMap = {value: key for key, value in genreMap.items()}

def createDataset(d):
    imgSize = (288, 432)
    imageList, labelList = [], []
    
    for genre, paths in d.items():
        for path in paths:
            image = tf.cast(tf.image.resize(tf.image.decode_png(tf.io.read_file(path), channels=3), imgSize), tf.float32) / 255.0
            imageList.append(image)
            labelList.append(genreMap[genre])

    dataset = tf.data.Dataset.from_tensor_slices((imageList, labelList)).shuffle(buffer_size=len(imageList)).batch(batchSize)
    return(dataset)

def prep(ds):
    out = (
        ds.map(lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label))
        .cache()
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )
    return out

training, validation, testing = prep(createDataset(train)), prep(createDataset(val)), prep(createDataset(test))

inputShape = [288, 432, 3]

earlyStopping = EarlyStopping(
    min_delta=0.001,
    patience=20,
    restore_best_weights=True
)

baseModel = VGG19(input_shape=inputShape, weights="imagenet", include_top=False, pooling="avg")

for layer in baseModel.layers:
    layer.trainable = False

transfer = Sequential([
    baseModel,
    Flatten(),
    BatchNormalization(),
    Dense(512, activation="relu"),
    Dropout(0.3),
    Dense(256, activation="relu"),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(len(genres), activation="softmax")
])

transfer.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
transferHistory = transfer.fit(training, validation_data=validation, batch_size=batchSize, epochs=2, verbose=1, callbacks=[earlyStopping])

def confusionMatrix(model, name):
    trueLabels = np.concatenate([y for x, y in testing], axis=0)
    predictedLabels = np.argmax(model.predict(testing, verbose=0), axis=1)
    matrix = confusion_matrix(trueLabels, predictedLabels)

    plt.figure()
    sns.heatmap(matrix, annot=True, cmap="Greens", xticklabels=genres, yticklabels=genres)
    plt.xlabel("Predicted Genre")
    plt.ylabel("True Genre")
    plt.title(f"{name} Model: Confusion Matrix")
    plt.show()
    
    trainStats, valStats, testStats = model.evaluate(training, verbose=0), model.evaluate(validation, verbose=0), model.evaluate(testing, verbose=0)
    print(f"{name} Model")
    print(f"Training Accuracy: {round(trainStats[1] * 100, 4)}%\nTrain Loss: {round(trainStats[0], 4)}")
    print(f"Validation Accuracy: {round(valStats[1] * 100, 4)}%\nTest Loss: {round(valStats[0], 4)}")
    print(f"Testing Accuracy: {round(testStats[1] * 100, 4)}%\nTest Loss: {round(testStats[0], 4)}")

confusionMatrix(transfer, "Transfer")

# Sauvegarde du modèle
transfer.save('vgg.h5')
