from flask import Flask, render_template, request

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
#from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50

# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,  LSTM, Reshape,  TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)
# model = ResNet50()

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')


# import opendatasets as od
# od.download("https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset")

# Define parameters
input_shape = (150, 150, 3)  # Adjust the image size as needed
batch_size = 32
epochs = 10
num_classes = 40  # Number of different plant classes
data_dir = "indian-medicinal-leaves-dataset\Indian Medicinal Leaves Image Datasets\Medicinal plant dataset"  # Replace with the path to your dataset folder

# Create a mapping between class indices and plant names
class_mapping = {
    0: 'Aloevera',
    1: 'Amla',
    2: 'Amruta Balli',
    3: 'Arali',
    4: 'Ashoka',
    5: 'Ashwagandha',
    6: 'Avacado',
    7: 'Bamboo',
    8: 'Basale',
    9: 'Betel',
    10: 'Betel_Nut',
    11: 'Brahmi',
    12: 'Castor',
    13: 'Curry Leaf',
    14: 'Doddapatre',
    15: 'Ekka',
    16: 'Ganike',
    17: 'Gauva',
    18: 'Geranium',
    19: 'Henna',
    20: 'Hibiscus',
    21: 'Honge',
    22: 'Insulin',
    23: 'Jasmine',
    24: 'Lemon',
    25: 'Lemon_grass',
    26: 'Mango',
    27: 'Mint',
    28: 'Nagadali',
    29: 'Neem',
    30: 'Nithyapushpa',
    31: 'Nooni',
    32: 'Pappaya',
    33: 'Pepper',
    34: 'Pomegranate',
    35: 'Raktachandini',
    36: 'Rose',
    37: 'Sapota',
    38: 'Tulasi',
    39: 'Wood_sorel',
    # Add mappings for all 40 classes here
}

# # Data preprocessing and augmentation
# datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,     # Normalize pixel values
#     rotation_range=20,      # Randomly rotate images
#     width_shift_range=0.2,  # Randomly shift images horizontally
#     height_shift_range=0.2, # Randomly shift images vertically
#     shear_range=0.2,        # Shear intensity
#     zoom_range=0.2,         # Randomly zoom in on images
#     horizontal_flip=True,   # Randomly flip images horizontally
#     fill_mode='nearest',    # Fill missing pixels with the nearest value
#     validation_split=0.2     # 20% of data will be used for validation
# )

# # Load and augment training data (80%)
# train_generator = datagen.flow_from_directory(
#     data_dir,                 # Root directory containing 40 subfolders
#     target_size=input_shape[:2],   # Resize images to match input_shape
#     batch_size=batch_size,
#     class_mode='categorical',  # Categorical classification
#     shuffle=True,             # Shuffle the data for training
#     subset='training'         # Specify training data subset
# )

# # Load and augment validation data (20%)
# validation_generator = datagen.flow_from_directory(
#     data_dir,                 # Root directory containing 40 subfolders
#     target_size=input_shape[:2],  # Resize images to match input_shape
#     batch_size=batch_size,
#     class_mode='categorical',  # Categorical classification
#     shuffle=False,            # Do not shuffle for validation
#     subset='validation'       # Specify validation data subset
# )



# # Load the DenseNet121 model with pretrained weights
# base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

# # Freeze the pretrained layers
# for layer in base_model.layers:
#     layer.trainable = False

# # Create a new model on top of the pretrained model
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu', name='custom_dense')(x)  # Custom dense layer with a valid name
# predictions = Dense(num_classes, activation='softmax', name='output_layer')(x)

# # Combine the base model and the new classification layers
# model = Model(inputs=base_model.input, outputs=predictions)


# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001),  # Adjust the learning rate if needed
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# # Train the model with validation data
# history = model.fit(
#     train_generator,
#     epochs=20,
#     validation_data=validation_generator,  # Use validation data
# )

# tf.saved_model.save(model, 'plant_classification_model_transfer')

# # Save the trained model
# model.save('plant_classification_model_transfer.h5')

# Load the model for inference
loaded_model = keras.models.load_model('plant_classification_model_transfer.h5')


# image_path = request.files['file']

def predict_plant(img):
    img = img.resize(input_shape)  # Resize the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32') / 255.0  # Normalize
    prediction = loaded_model.predict(img)
    class_index = np.argmax(prediction)
    plant_name = class_mapping.get(class_index, 'Unknown Plant')
    confidence = np.max(prediction)
    return plant_name, confidence

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        plant_name, confidence = predict_plant(file)
        return render_template('index.html', prediction=plant_name, confidence=confidence)

if __name__ == '__main__':
    app.run(port=3000, debug=True)







# @app.route('/', methods=['POST'])

# def predict_plant(image_path):
#     img = image.load_img(image_path, target_size=input_shape[:2])
#     img = image.img_to_array(img)
#     img = img.reshape((1,) + img.shape)  # Reshape for model input
#     img = img.astype('float32') / 255.0  # Normalize the image

#     prediction = loaded_model.predict(img)
#     class_index = np.argmax(prediction)
#     plant_name = class_mapping.get(class_index, 'Unknown Plant')
#     confidence = np.max(prediction)
# # def predict():
# #     imagefile= request.files['imagefile']
# #     image_path = "./images/" + imagefile.filename
# #     imagefile.save(image_path)

# #     image = load_img(image_path, target_size=(224, 224))
# #     image = img_to_array(image)
# #     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# #     image = preprocess_input(image)
# #     yhat = model.predict(image)
# #     label = decode_predictions(yhat)
# #     label = label[0][0]

# #     classification = '%s (%.2f%%)' % (label[1], label[2]*100)


#     return render_template('index.html', prediction=classification)


# if __name__ == '__main__':
#     app.run(port=3000, debug=True)























# from flask import Flask, render_template, request

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# #from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50

# app = Flask(__name__)
# model = ResNet50()

# @app.route('/', methods=['GET'])
# def hello_word():
#     return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     imagefile= request.files['imagefile']
#     image_path = "./images/" + imagefile.filename
#     imagefile.save(image_path)

#     image = load_img(image_path, target_size=(224, 224))
#     image = img_to_array(image)
#     image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
#     image = preprocess_input(image)
#     yhat = model.predict(image)
#     label = decode_predictions(yhat)
#     label = label[0][0]

#     classification = '%s (%.2f%%)' % (label[1], label[2]*100)


#     return render_template('index.html', prediction=classification)


# if __name__ == '__main__':
#     app.run(port=3000, debug=True)