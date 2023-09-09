import numpy
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout


# load the data
data = []
labels = []
with open("C:\\Users\\Mazen\\Desktop\\NLP_finalproject\\NLP_finalproject\\data\\original-data.txt", 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        labels.append(int(line[0]))
        data.append(line[1])
        
        

# preprocess the data
tokenizer = Tokenizer(num_words=10000) #create model that will swallow 10,000 words at max
tokenizer.fit_on_texts(data) # feed data to the tokenizer to return 10,000 most common tokens
sequences = tokenizer.texts_to_sequences(data) #convert text data to sequences that contain
                                                #integers defining the indices of the tokens
word_index = tokenizer.word_index # get dictionary of word indices
data = pad_sequences(sequences, maxlen=100) # do padding or truncating to keep sequence len = 100

#splitting data
split = int(len(data) * 0.8)
X = data[:split]
y = labels[:split]
y = numpy.array(y)

X_val = data[split:]
y_val = labels[split:]
y_val = numpy.array(y_val) # convert labels from lists into numpy arrays


#defining cnn model
'''
Data preprocessing: The text data must be preprocessed to convert it into a numerical format
that can be fed into a CNN. This typically involves tokenizing the text into individual words
or subwords, and then converting those tokens into numerical vectors.

Embedding layer: The numerical vectors are then typically passed through an embedding layer,
which maps the vectors to a higher-dimensional space where similar words are closer together
and dissimilar words are farther apart. This helps the CNN better capture semantic relationships
between words.

Convolutional layer: The embedded vectors are then passed through one or more convolutional
layers, which apply filters to the sequence of vectors to extract local features.

Pooling layer: The output of the convolutional layer(s) is then typically passed through
a pooling layer, which reduces the dimensionality of the output and helps to capture global
information about the sequence.

Fully connected layer: The output of the pooling layer is then flattened and passed through
one or more fully connected layers, which perform a classification task based on the
extracted features.

Output layer: The output of the final fully connected layer is passed through a sigmoidal 
activation function, which produces a probability distribution over the possible classes.

Training: The CNN is trained on a labeled dataset using backpropagation and gradient descent
to minimize a loss function, such as cross-entropy.
'''
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=100, input_length=100)) #encoding layer using word embedding
model.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=2)) # for reducing the dimensionality of the data matrix
model.add(Flatten()) # for converting data matrix into a 1-d vector before entering the dense layer
model.add(Dense(10, activation='relu')) #fully connected layers with 10 neurons
model.add(Dropout(0.5)) # add a dropout layer with a rate of 0.5
model.add(Dense(1, activation='sigmoid')) #sigmoid is the act. func. for the binary classification task
                                            # it produces a probability distribution from 0 to 1 

#compiling model
# loss func is needed to compute the error
# adam is the optimizer needed to propagate those errors back for weights update
# accuracy is the evalution metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting model
history1 = model.fit(X, y, validation_data=(X_val, y_val), epochs=40, batch_size=32)
train_acc = history1.history['accuracy']
val_acc = history1.history['val_accuracy']

#assess the model
loss, accuracy = model.evaluate(X_val, y_val, batch_size=20)

#model.summary()

#===========================================================================================

# load the augmented data
aug_data = []
aug_labels = []
with open("C:\\Users\\Mazen\\Desktop\\NLP_finalproject\\NLP_finalproject\\data\\eda_original-data.txt", 'r') as ff:
    for line in ff:
        line = line.strip().split('\t')
        aug_data.append(line[1])
        aug_labels.append(int(line[0]))


# preprocess the augmented data
aug_tokenizer = Tokenizer(num_words=10000)
aug_tokenizer.fit_on_texts(aug_data)
aug_sequences = aug_tokenizer.texts_to_sequences(aug_data)
augword_index = aug_tokenizer.word_index
aug_data = pad_sequences(aug_sequences, maxlen=100)


#splitting the augmented data
split_aug = int(len(aug_data) * 0.8)
X_aug = aug_data[:split]
y_aug = aug_labels[:split]
y_aug = numpy.array(y_aug)

X_val_aug = aug_data[split:]
y_val_aug = aug_labels[split:]
y_val_aug = numpy.array(y_val_aug)


#defining cnn model
model_aug = Sequential()
model_aug.add(Embedding(input_dim=10000, output_dim=100, input_length=100))
model_aug.add(Conv1D(filters=16, kernel_size=5, activation='relu'))
model_aug.add(MaxPooling1D(pool_size=2))
model_aug.add(Flatten())
model_aug.add(Dense(10, activation='relu'))
model_aug.add(Dropout(0.5)) # add a dropout layer with a rate of 0.5
model_aug.add(Dense(1, activation='sigmoid'))


#compiling model
model_aug.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#fitting model
history_aug = model_aug.fit(X_aug, y_aug, validation_data=(X_val_aug, y_val_aug), epochs=40, batch_size=32)
train_acc_aug = history_aug.history['accuracy']
val_acc_aug = history_aug.history['val_accuracy']

#assess the model
loss_aug, accuracy_aug = model.evaluate(X_val_aug, y_val_aug, batch_size=20)

#model.summary()

print(accuracy, accuracy_aug)



# Plot the validation accuracies for both models
epochs = range(1, len(val_acc) + 1)
plt.plot(epochs, train_acc, 'r', label='train acc')
plt.plot(epochs, train_acc_aug, 'm', label='train acc_aug')
plt.plot(epochs, val_acc,'g', label='val acc')
plt.plot(epochs, val_acc_aug, 'y', label='val acc_aug')
plt.title('Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


