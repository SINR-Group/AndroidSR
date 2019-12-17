# How to Run the Android Project

**Android SDK version==android-ndk-r17c**



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

## Convert Frozen Graph (.pb) to snapdragon Format (.dlc)
* Install the SNPE SDK on the computer
* Overview : https://developer.qualcomm.com/docs/snpe/overview.html
* Please keep in mind not all mobiles are compatible with the snapdragon format.
* Installation Guide : https://developer.qualcomm.com/docs/snpe/setup.html
* After installing the SNPE and SDK, use the cmd: 
```
python ./bin/x86_64-linux-clang//snpe-tensorflow-to-dlc --graph $SNPE_ROOT/bin/x86_64-linux-clang/m.pb --input_dim "time_distributed_29_input" "3,30,120,240,3" --out_node "lstm_15/transpose_1" --dlc m.dlc --allow_unconsumed_nodes
```
Please get the name of the input tensor and output tensor from the frozen graph.

## Running the application
* Store the .pb file in the asset folder inside the android project.
* Give correct address to the model inside the MainActivity.java file.
* Run the App
