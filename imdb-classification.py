from keras.datasets import imdb
import numpy as np
from keras import models,layers
import matplotlib.pyplot as plt

(train_data,train_labels), (test_data,test_labels) = imdb.load_data(num_words=10000)

print( train_data[0])

def vectorize_sequences(sequences , dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train[0])

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,partial_y_train,epochs=5,batch_size=512,validation_data=(x_val,y_val))

history_dict = history.history
print( history_dict.keys())

acc = history_dict['accuracy']
loss_vals = history_dict['loss']
val_loss_vals = history_dict['val_loss']

epochs = range(1,len(acc)+1)
plt.plot(epochs,loss_vals,'bo',label='Training Loss')
plt.plot(epochs,val_loss_vals,'b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

results = model.evaluate(x_test,y_test)
print(results)