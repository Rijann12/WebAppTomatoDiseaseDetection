# vako .h5 model lai .tfLite ma convert 

import tensorflow as tf 
model = tf.keras.models.load_model('trained_plant_disease_model.h5') #h5 model load 
converter =tf.lite.TFLiteConverter.from_keras_model(model)# convert to tflite
converter.optimization = [tf.lite.Optimize.DEFAULT]# optimization garko for smaller size and better speed
tflite_model= converter.convert()#convert gareko
#model save gareko .tflite file ma 
with open('trained_plant_disease_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("conversion complete: model.tflite saved ")