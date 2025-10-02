import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# load dataset
img_height, img_width = 128, 128
batch_size = 32

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/train",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/test",
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# normalize
normalizer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalizer(x), y))
test_data = test_data.map(lambda x, y: (normalizer(x), y))

# cnn model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(len(train_data.class_names), activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train
history = model.fit(train_data, validation_data=test_data, epochs=10)

# evaluate
test_loss, test_acc = model.evaluate(test_data)
print("Test accuracy:", test_acc)

# plot
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.show()
