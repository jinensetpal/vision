package com.necter.vision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
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
import androidx.core.util.Pair;

import com.google.common.util.concurrent.ListenableFuture;
import com.necter.vision.ml.Deeplab;
import com.necter.vision.ml.Deeplabv3257MvGpu;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import org.tensorflow.lite.task.vision.segmenter.ColoredLabel;
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter;
import org.tensorflow.lite.task.vision.segmenter.Segmentation;

import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;


public class MainActivity extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
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
        ImageView outputView = findViewById(R.id.output);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                bindPreview(cameraProvider);
            } catch (ExecutionException | InterruptedException ignored) {
            }
        }, ContextCompat.getMainExecutor(this));

        Bitmap bitmap = BitmapFactory.decodeResource(this.getResources(), R.drawable.image);

        try {
            TensorImage image = TensorImage.fromBitmap(bitmap);
            ImageSegmenter imageSegmenter = ImageSegmenter.createFromFile(this, "deeplabv3_257_mv_gpu.tflite");

            List<Segmentation> results = imageSegmenter.segment(image);
            Pair<Bitmap, Map<String, Integer>> thing = createMaskBitmapAndLabels(results.get(0), image.getWidth(), image.getHeight());
            outputView.setImageBitmap(stackBitmaps(thing.first,image.getBitmap()));
        } catch (IOException ignore) {
        }
    }

    private Pair<Bitmap, Map<String, Integer>> createMaskBitmapAndLabels(Segmentation result, int inputHeight, int inputWidth) {
        List<ColoredLabel> coloredLabels = result.getColoredLabels();
        int[] colors = new int[coloredLabels.size()];
        int cnt = 0;

        for(ColoredLabel coloredLabel : coloredLabels) {
            int rgb = coloredLabel.getArgb();
            colors[cnt++] = Color.argb(128, Color.red(rgb), Color.green(rgb), Color.blue(rgb));
        }
        colors[0] = Color.TRANSPARENT;

        TensorImage maskTensor = result.getMasks().get(0);
        byte[] maskArray = maskTensor.getBuffer().array();
        int[] pixels = new int[maskArray.length];
        HashMap<String, Integer> itemsFound = new HashMap<>();

        for (int i = 0; i < maskArray.length; i++) {
            int color = colors[maskArray[i]];
            pixels[i] = color;
            itemsFound.put(coloredLabels.get(maskArray[i]).getlabel(), color);
        }

        Bitmap maskBitmap = Bitmap.createBitmap(pixels, maskTensor.getWidth(), maskTensor.getHeight(), Bitmap.Config.ARGB_8888);

        return Pair.create(Bitmap.createScaledBitmap(maskBitmap, inputWidth, inputHeight, true), itemsFound);
    }

    private Bitmap stackBitmaps(Bitmap foreground, Bitmap background) {
        Bitmap merged = Bitmap.createBitmap(foreground.getWidth(), foreground.getHeight(), foreground.getConfig());
        Canvas canvas = new Canvas(merged);
        canvas.drawBitmap(background, 0.0f, 0.0f, null);
        canvas.drawBitmap(foreground, 0.0f, 0.0f, null);
        return merged;
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