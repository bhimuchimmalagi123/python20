import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

mnist=tf.keras.datasets.mnist
(x_train , y_train),(x_test , y_test)=mnist.load_data()
(x_train , x_test) = x_train/255.0 , x_test/255.0
print("third digit in y_train is:",y_train[2])
plt.imshow(x_train[2],cmap='coolwarm')
plt.show()

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')])
model.compile(optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
logdir ="logs/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath="model_checkpoint.h5",save_best_only=True)
history=model.fit(
    x_train,y_train,
    epochs=6,
    validation_data=(x_test , y_test),
    callbacks=[tensorboard_callback,checkpoint_callback])
loss,acc=model.evaluate(x_test , y_test , verbose=1)
print("Original model accuracy:{:5.2f}%".format(100*acc))
