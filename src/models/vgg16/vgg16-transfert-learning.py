import numpy as np
import pandas as pd
from glob import glob
import os

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.tensorflow

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

from vgg16_prepare_dataset import prepareDataset

# Initialisation de l'URL
mlflow_server_uri = "http://192.168.1.19:8080"

# Imports et paramétrage de MLflow
from mlflow import MlflowClient
import mlflow
import setuptools

mlflow.set_tracking_uri(mlflow_server_uri)
mlflow.set_experiment("champi_vgg16") # Le nom du projet

scr_path_origin = "data/vgg-16"
src_path_train = "data/LAYER2/MO/train"
src_path_test = "data/LAYER2/MO/validation"


batch_size = 64
for layer_size1 in [512,256,128]:
  for layer_size2 in [512,256,128]:
    for dropout_rate1 in [0,.1,.2]:
      for dropout_rate2 in [0,.1,.2]:
        IMSIZE = [224,224]
        NBCLASSES = len(os.listdir(src_path_train))

        image_gen = ImageDataGenerator(
            rescale=1 / 255.0
        )

        # create generators
        train_generator = image_gen.flow_from_directory(
          src_path_train,
          target_size=IMSIZE,
          shuffle=True,
          batch_size=batch_size,
        )

        # Convertissez le générateur en un objet tf.data.Dataset
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None, IMSIZE[0], IMSIZE[1], 3]), tf.TensorShape([None, NBCLASSES]))
        )

        # Répétez les données d'entraînement
        #train_dataset = train_dataset.repeat()


        test_generator = image_gen.flow_from_directory(
          src_path_test,
          target_size=IMSIZE,
          batch_size=batch_size,
          shuffle=True
        )


        # Convertissez le générateur en un objet tf.data.Dataset
        test_dataset = tf.data.Dataset.from_generator(
            lambda: test_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([None, IMSIZE[0], IMSIZE[1], 3]), tf.TensorShape([None, NBCLASSES]))
        )

        # Répétez les données d'entraînement
        #test_dataset = test_dataset.repeat()



        train_image_files = glob(src_path_train + '/*/*.jp*g')
        test_image_files = glob(src_path_test + '/*/*.jp*g')


        epochs = 100
        learning_rate = .001
          
        def create_model():
            vgg = VGG16(input_shape=IMSIZE + [3], weights='imagenet', include_top=False)
            vgg.summary()
            # Freeze existing VGG already trained weights
            #for layer in vgg.layers[:15]:
            #    layer.trainable = False
            #for layer in vgg.layers[15:]:
            #    layer.trainable = True
            for layer in vgg.layers:
                layer.trainable = False
            # get the VGG output
            out = vgg.output
            #out = vgg.layers[14].output


            # Add new dense layer at the end
            x = Flatten()(out)
            x = Dense(layer_size1, activation='relu')(x)
            x = Dropout(dropout_rate1)(x)
            x = Dense(layer_size2, activation='relu')(x)
            x = Dropout(dropout_rate2)(x)
            x = Dense(NBCLASSES, activation='softmax')(x)
            
            model = Model(inputs=vgg.input, outputs=x)


            #optimizer = Adam(learning_rate=learning_rate)

            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=['accuracy'])
            
            model.summary()
            
            return model

        mymodel = create_model()

        early_stop = EarlyStopping(monitor='val_accuracy',patience=epochs/10)



        # Démarrer un run MLflow
        run_name = "vgg_16_TRAIN_BLK5_no_augment" # Le nom de la run, nous utiliserons notre propre nomenclature pour le projet

        with mlflow.start_run(run_name=run_name) as run:
          
          checkpoint = ModelCheckpoint('saved_models/'+mlflow.active_run().info.run_id, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

          history = mymodel.fit(
              train_dataset,
              validation_data=test_dataset,
              epochs=epochs,
              callbacks=[early_stop, checkpoint],
              batch_size=batch_size,
              shuffle=True,
              steps_per_epoch=len(train_image_files) // batch_size,
              validation_steps=len(test_image_files) // batch_size,
              validation_batch_size=batch_size 
              )
          

          # Enregistrer les paramètres
          mlflow.log_param("epochs", epochs)
          mlflow.log_param("batch_size", batch_size)
          mlflow.log_param("nb_classes", NBCLASSES)
          mlflow.log_param("nb_images_train", len(train_image_files))
          mlflow.log_param("learning_rate", (learning_rate))
          mlflow.log_param("dropout_rate1", (dropout_rate1))
          mlflow.log_param("layer_size1", (layer_size1))
          mlflow.log_param("dropout_rate2", (dropout_rate2))
          mlflow.log_param("layer_size2", (layer_size2))
          # Enregistrer les métriques
          best_val_accuracy = 0
          for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            if(history.history['val_accuracy'][epoch] > best_val_accuracy):
              best_val_accuracy = history.history['val_accuracy'][epoch]
          # Enregistrer le modèle
          mlflow.log_metric("best_val_accuracy", (best_val_accuracy))

          