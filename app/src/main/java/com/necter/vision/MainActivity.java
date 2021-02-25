package com.necter.vision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
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
import com.necter.vision.ml.Deeplab;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.ExecutionException;


public class MainActivity extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private ImageView outputView, inputView, resizedView;
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
        inputView = findViewById(R.id.input);
        resizedView = findViewById(R.id.resized);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException ignored) {
            }
        }, ContextCompat.getMainExecutor(this));

//        final String ASSOCIATED_AXIS_LABELS = "labels.txt";
//        List<String> associatedAxisLabels = null;
//
//        try {
//            associatedAxisLabels = FileUtil.loadLabels(this, ASSOCIATED_AXIS_LABELS);
//        } catch (IOException e) {
//            Log.e("tfliteSupport", "Error reading label file", e);
//        }


        Bitmap bitmap = BitmapFactory.decodeResource(this.getResources(), R.drawable.image);

        try {
            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            .add(new ResizeOp(257, 257, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                            .build();

            TensorImage tImage = new TensorImage(DataType.FLOAT32);
            tImage.load(bitmap);
            tImage = imageProcessor.process(tImage);

            Deeplab model = Deeplab.newInstance(this);
            ByteBuffer buffer = ByteBuffer.allocate(bitmap.getByteCount()); // ByteBuffer.allocate(bitmap.getByteCount());
            bitmap.copyPixelsToBuffer(buffer); //Move the byte data to the buffer

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 257, 257, 3}, DataType.FLOAT32);
//            inputFeature0.loadArray(flatInput);
            inputFeature0.loadBuffer(tImage.getBuffer());

            // Runs model inference and gets result.
            Deeplab.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();

            inputView.setImageBitmap(bitmap);
            resizedView.setImageBitmap(tImage.getBitmap());
            outputView.setImageBitmap(convertByteBufferToBitmap(outputFeature0.getBuffer(), 257, 257));

            // Releases model resources if no longer used.
            model.close();
        } catch (IOException ignore) {
        }
    }

    @SuppressWarnings("SameParameterValue")
    private Bitmap convertByteBufferToBitmap(ByteBuffer byteBuffer, int imgSizeX, int imgSizeY) {
        byteBuffer.rewind();
        byteBuffer.order(ByteOrder.nativeOrder());
        Bitmap bitmap = Bitmap.createBitmap(imgSizeX, imgSizeY, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[imgSizeX * imgSizeY];
        for (int i = 0; i < imgSizeX * imgSizeY; i++)
            if (byteBuffer.getFloat() > 0.5)
                pixels[i] = Color.argb(100, 255, 105, 180);
            else
                pixels[i] = Color.argb(0, 0, 0, 0);

        bitmap.setPixels(pixels, 0, imgSizeX, 0, 0, imgSizeX, imgSizeY);
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
        // noinspection unused
        Camera camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview);
    }
}