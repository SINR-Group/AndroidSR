# tf-android

# Convert Keras Model to .pb (TensorFlow)
First, we need to use the keras_to_tensorflow.py file to convert the .h5(keras model) to .pb(tensorflow model).
eg: python keras_to_tensorflow.py --input_model=model1.h5 --output_model=new.pb

# Running the application
Store the .pb file in the asset folder inside the android project.
Give correct address to the model inside the MainActivity.java file.
Run the App
