# How to Run the Android Project

## Convert Keras Model to .pb (TensorFlow)
* First, we need to use the keras_to_tensorflow.py file to convert the .h5(keras model) to .pb(tensorflow model).
* eg: python keras_to_tensorflow.py --input_model=model1.h5 --output_model=new.pb

## Convert Frozen Graph (.pb) to Tflite format
* For Reference: https://www.tensorflow.org/lite/convert/cmdline
* Use the Cmd:
```
tflite_convert 
--output_file=test.tflite 
--graph_def_file=tflite_graph.pb 
--input_shape=1,300,300,3
```
* Alternatively, Use the python API : https://www.tensorflow.org/lite/convert/python_api

## Running the application
* Store the .pb file in the asset folder inside the android project.
* Give correct address to the model inside the MainActivity.java file.
* Run the App
