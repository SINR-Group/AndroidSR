package com.example.myapplication123;

import android.Manifest;
import android.app.Activity;
import android.app.ActivityManager;
import android.content.ClipData;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ConfigurationInfo;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.Image;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

//import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import org.tensorflow.lite.Interpreter;
//import org.tensorflow.lite.experimental.GpuDelegate;

import java.io.File;
import java.io.FileDescriptor;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;




public class MainActivity extends AppCompatActivity {

    public Bitmap imageBitmap;
    private Button btn;
    private ImageView imageView;
    private TextView txt;
    static final int REQUEST_IMAGE_CAPTURE = 1;
    // PICK_PHOTO_CODE is a constant integer
    public final static int PICK_PHOTO_CODE = 1046;
    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;
    private static final int INPUT_SIZE = 192;
    private int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private static final String MODEL_FILE = "file:///android_asset/m.tflite";



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btn = (Button)findViewById(R.id.button);
        imageView = (ImageView)findViewById(R.id.img);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            btn.setEnabled(false);
            ActivityCompat.requestPermissions(this, new String[] { Manifest.permission.CAMERA }, 0);
        }
    }

    private class AsyncTaskRunner extends AsyncTask<Bitmap, Integer, Bitmap> {

        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
        }

        @Override
        protected Bitmap doInBackground(Bitmap... params) {
            return predict(params[0], MODEL_FILE, getAssets());
//            return results;
        }

        public Bitmap predict(Bitmap bitmap, String modelName, AssetManager assets) {
            double t1 = System.currentTimeMillis();
            final ActivityManager activityManager =
                    (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
            final ConfigurationInfo configurationInfo =
                    activityManager.getDeviceConfigurationInfo();
            System.err.println(Double.parseDouble(configurationInfo.getGlEsVersion()));
            System.err.println(configurationInfo.reqGlEsVersion >= 0x30000);
            System.err.println(String.format("%X", configurationInfo.reqGlEsVersion));

            try {
                // NEW: Prepare GPU delegate.
//                GpuDelegate delegate = new GpuDelegate();
//                Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
                Interpreter tflite = new Interpreter(loadModelFile());
                int[] dims = new int[] {BATCH_SIZE, INPUT_SIZE, INPUT_SIZE, PIXEL_SIZE};
                tflite.resizeInput(0, dims);
                int diff = BATCH_SIZE;
                for(int a=BATCH_SIZE; a<= 30; a = a+diff) {
                    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
                    ByteBuffer imgData = convertBitmapToByteBuffer(bitmap);
                    // 0xff000000 | (R << 16) | (G << 8) | B;
                    float[][][][] result = new float[BATCH_SIZE][INPUT_SIZE][INPUT_SIZE][PIXEL_SIZE];
    //                Map<Integer, Object> outputs = new HashMap<Integer, Object>();
    //                outputs.put(0, result);
                    int[] intValues = new int[INPUT_SIZE*INPUT_SIZE];
    //                Object[] inputs = new Object[]{imgData};
                    tflite.run(imgData, result);
                    int idx = 0;
                    for(int k =0; k<1; ++k) {
                        for (int i = 0; i < INPUT_SIZE; ++i) {
                            for (int j = 0; j < INPUT_SIZE; ++j) {
                                int R = (int)result[k][i][j][0];
                                int G = (int)result[k][i][j][1];
                                int B = (int)result[k][i][j][2];
                                intValues[idx] = 0xff000000 | (R) | (G) | B;
                                idx++;
                            }
                        }
                    }
                    bitmap.setPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
                    if(a+diff > 30){
                        diff = a+diff - 30;
                    }
                }
                // Clean up
//                delegate.close();
                double t2 = System.currentTimeMillis();
                double difference = (t2 - t1)/1000;
                Log.i("Time: ", " in secs: " + difference);
                System.out.println("Batch Size: " + BATCH_SIZE + "  Secs: " + difference);
            } catch(Exception ex){
                Log.w("WARNING: ", ex);
            }
            return bitmap;

        }

        @Override
        protected void onPostExecute(Bitmap result) {
            super.onPostExecute(result);
                imageView.setImageBitmap(result);
        }

        private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
//            ByteBuffer[] byteBuffer = new ByteBuffer[BATCH_SIZE];
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
            int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
            bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
            int pixel = 0;
            for(int k =0; k<BATCH_SIZE; ++k) {

//                byteBuffer[k] = ByteBuffer.allocateDirect(4*BATCH_SIZE * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
//                byteBuffer[k].order(ByteOrder.nativeOrder());

                for (int i = 0; i < INPUT_SIZE; ++i) {
                    for (int j = 0; j < INPUT_SIZE; ++j) {
                        final int val = intValues[i*INPUT_SIZE  + j];
                        byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD); //R
                        byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD); //G
                        byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD); // B

//                        byteBuffer[k].putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD); //R
//                        byteBuffer[k].putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD); //G
//                        byteBuffer[k].putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD); // B


//                        byteBuffer.put((byte) ((val >> 16) & 0xFF));
//                        byteBuffer.put((byte) ((val >> 8) & 0xFF));
//                        byteBuffer.put((byte) (val & 0xFF));
                    }
                }
            }
            return byteBuffer;
        }

        private FileDescriptor openFile(String path)
                throws FileNotFoundException, IOException {
            File file = new File(path);
            FileOutputStream fos = new FileOutputStream(file);
            // remember th 'fos' reference somewhere for later closing it
            fos.write((new Date() + " Beginning of process...").getBytes());
            return fos.getFD();
        }

        /** Memory-map the model file in Assets. */
        private MappedByteBuffer loadModelFile() throws IOException {
            String Model = "m.tflite";
            AssetFileDescriptor fileDescriptor = getAssets().openFd(Model);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }

        //helper class to return the largest value in the output array
        public float arrayMaximum(float[] arr) {
            float max = Float.NEGATIVE_INFINITY;
            for(float cur: arr)
                max = Math.max(max, cur);
            return max;
        }

        // helper class to find the index (and therefore numerical value) of the largest confidence score
        public int getIndexOfLargestValue( float[] array )
        {
            if ( array == null || array.length == 0 ) return -1;
            int largest = 0;
            for ( int i = 1; i < array.length; i++ )
            {if ( array[i] > array[largest] ) largest = i;            }
            return largest;
        }
    }

    //On clicking button
    public void takePicture(View view) {
//        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
//        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
//            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
//        }
//        Intent intent = new Intent(Intent.ACTION_PICK);
//        intent.setType("image/*");
//        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
//        intent.setAction(Intent.ACTION_GET_CONTENT);

        // If you call startActivityForResult() using an intent that no app can handle, your app will crash.
        // So as long as the result is not null, it's safe to use the intent.
//        if (intent.resolveActivity(getPackageManager()) != null) {
            // Bring up gallery to select a photo
//            startActivityForResult(Intent.createChooser(intent,"Select Picture"), PICK_PHOTO_CODE);
//        }
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);

        // If you call startActivityForResult() using an intent that no app can handle, your app will crash.
        // So as long as the result is not null, it's safe to use the intent.
        if (intent.resolveActivity(getPackageManager()) != null) {
            // Bring up gallery to select a photo
            startActivityForResult(intent, PICK_PHOTO_CODE);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == 0) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                btn.setEnabled(true);
            }
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
//        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
//        Bundle extras = data.getExtras();
//        imageBitmap = (Bitmap) extras.get("data");
        Uri photoUri = data.getData();
        try {
//            // Do something with the photo based on Uri
            Bitmap bmp = MediaStore.Images.Media.getBitmap(this.getContentResolver(), photoUri);
            imageBitmap = Bitmap.createScaledBitmap(bmp, 192, 192, true);
        } catch (Exception ex){
            Log.e("Error", "Exception : ",ex);
        }
        imageView.setImageBitmap(imageBitmap);
//        if (data.getClipData() != null) {
//            ClipData mClipData = data.getClipData();
//            ArrayList<Uri> mArrayUri = new ArrayList<Uri>();
//            ArrayList<Bitmap> lbmp = new ArrayList<Bitmap>();
//            for (int i = 0; i < mClipData.getItemCount(); i++) {
//                ClipData.Item item = mClipData.getItemAt(i);
//                Uri uri = item.getUri();
//                mArrayUri.add(uri);
//                Bitmap bmp;
//                try {
//                    // !! You may need to resize the image if it's too large
//                    bmp = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
//                    bmp = Bitmap.createScaledBitmap(bmp, 192, 192, true);
//                    lbmp.add(bmp);
//                } catch (Exception ex){
//                    Log.e("Error", "Exception : ",ex);
//                }
//            }
            AsyncTaskRunner runner = new AsyncTaskRunner();
            runner.execute(imageBitmap);
//        }

//        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
