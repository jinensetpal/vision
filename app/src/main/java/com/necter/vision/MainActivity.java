package com.necter.vision;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;
import com.necter.vision.ml.Deeplabv31Default1;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.concurrent.ExecutionException;


public class MainActivity extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ImageView outputView;
    private PreviewView previewView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
                PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA},
                    50);
        }

        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        previewView = findViewById(R.id.previewView);
        outputView = findViewById(R.id.output);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException ignored) {
            }
        }, ContextCompat.getMainExecutor(this));

        final String ASSOCIATED_AXIS_LABELS = "labels.txt";
        List<String> associatedAxisLabels = null;

        try {
            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }


        Bitmap bitmap = BitmapFactory.decodeResource(this.getResources(), R.drawable.image);
        bitmap = Bitmap.createScaledBitmap(bitmap, 257, 257, true);

        int batchNum = 0;
        float[][][][] input = new float[1][257][257][3];
        for (int x = 0; x < 257; x++) {
            for (int y = 0; y < 257; y++) {
                int pixel = bitmap.getPixel(x, y);
                input[batchNum][x][y][0] = (Color.red(pixel) - 127) / 128.0f;
                input[batchNum][x][y][1] = (Color.green(pixel) - 127) / 128.0f;
                input[batchNum][x][y][2] = (Color.blue(pixel) - 127) / 128.0f;
            }
        }

        try {
            float[] flatInput = new float[198147];
            int c = 0;
            for (int x = 0; x < 257; x++) {
                for (int y = 0; y < 257; y++) {
                    int pixel = bitmap.getPixel(x, y);
                    flatInput[c] = (Color.red(pixel) - 127) / 128.0f;
                    c++;
                    flatInput[c] = (Color.green(pixel) - 127) / 128.0f;
                    c++;
                    flatInput[c] = (Color.blue(pixel) - 127) / 128.0f;
                    c++;
                }
            }


            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(257, 257, ResizeOp.ResizeMethod.BILINEAR))
                            .build();


            TensorImage tImage = new TensorImage(DataType.FLOAT32);
            tImage.load(bitmap);
            tImage = imageProcessor.process(tImage);


            Deeplabv31Default1 model = Deeplabv31Default1.newInstance(this);
            ByteBuffer buffer = ByteBuffer.allocate(bitmap.getByteCount()); // ByteBuffer.allocate(bitmap.getByteCount());
            bitmap.copyPixelsToBuffer(buffer); //Move the byte data to the buffer

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 257, 257, 3}, DataType.FLOAT32);
            inputFeature0.loadArray(flatInput);
//            inputFeature0.loadBuffer(buffer);

            // Runs model inference and gets result.
            Deeplabv31Default1.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            outputView.setImageBitmap(convertByteBufferToBitmap(outputFeature0.getBuffer(), 257, 257));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException ignore) {
        }
    }

    private Bitmap convertByteBufferToBitmap(ByteBuffer byteBuffer, int imgSizeX, int imgSizeY) {
        byteBuffer.rewind();
        byteBuffer.order(ByteOrder.nativeOrder());
        Bitmap bitmap = Bitmap.createBitmap(imgSizeX, imgSizeY, Bitmap.Config.ARGB_4444);
        int[] pixels = new int[imgSizeX * imgSizeY];
        for (int i = 0; i < imgSizeX * imgSizeY; i++)
            if (byteBuffer.getFloat() > 0.5)
                pixels[i] = Color.argb(100, 255, 105, 180);
            else
                pixels[i] = Color.argb(0, 0, 0, 0);

        bitmap.setPixels(pixels, 0, imgSizeX, 0, 0, imgSizeX, imgSizeY);
        return bitmap;
    }

    public static Bitmap getBitmapFromAsset(Context context, String filePath) {
        AssetManager assetManager = context.getAssets();

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
        } catch (IOException e) {
            // handle exception
        }

        return bitmap;
    }


    private Bitmap image() {
        int w = 257, h = 257;
        Bitmap.Config conf = Bitmap.Config.ARGB_8888; // see other conf types
        return Bitmap.createBitmap(w, h, conf); // Temporary Bitmap

//        Button photoButton = (Button) this.findViewById(R.id.button1);
//        photoButton.setOnClickListener(new View.OnClickListener()
//        {
//            @Override
//            public void onClick(View v)
//            {
//                if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
//                {
//                    requestPermissions(new String[]{Manifest.permission.CAMERA}, MY_CAMERA_PERMISSION_CODE);
//                }
//                else
//                {
//                    Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
//                    startActivityForResult(cameraIntent, CAMERA_REQUEST);
//                }
//            }
//        });
    }

    void bindPreview(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview);
    }
}