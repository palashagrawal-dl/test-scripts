from keras.datasets import reuters
import numpy as np
from keras import models,layers
from keras import regularizers
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

(train_data,train_labels), (test_data,test_labels) = reuters.load_data(num_words=10000)

print( train_data[0])

def vectorize_sequences(sequences , dimension = 10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

print(x_train[0])

y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)

model = models.Sequential()
model.add(layers.Dense(128,kernel_regularizer=regularizers.l2(0.001),activation='relu',input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.001),activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

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
