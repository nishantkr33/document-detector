from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load pre-trained VGG16 model without top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Build a new model on top of VGG16
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification for forged/genuine
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Image data generators for training and validation datasets
train_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory('data/train', target_size=(128, 128), class_mode='binary')
val_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory('data/validation', target_size=(128, 128), class_mode='binary')

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_gen)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Save the trained model
model.save('forgery_detection_model.h5')
