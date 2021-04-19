package com.necter.vision;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.util.Pair;

import com.google.common.util.concurrent.ListenableFuture;
import com.quickbirdstudios.yuv2mat.Yuv;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.segmenter.ColoredLabel;
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter;
import org.tensorflow.lite.task.vision.segmenter.Segmentation;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {

    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    private PreviewView previewView;
    private ImageSegmenter imageSegmenter;
    private ImageView outputView;

//    static {
//        System.loadLibrary("opencv_java3");
//    }

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

        try {
            imageSegmenter = ImageSegmenter.createFromFile(this, "deeplabv3_257_mv_gpu.tflite");
        } catch (IOException ignore) {
        }

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
//                bindPreview(cameraProvider);
                getInference(cameraProvider);
            } catch (ExecutionException | InterruptedException ignored) {
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private Pair<Bitmap, Map<String, Integer>> createMaskBitmapAndLabels(Segmentation result, int inputHeight, int inputWidth) {
        List<ColoredLabel> coloredLabels = result.getColoredLabels();
        int[] colors = new int[coloredLabels.size()];
        int cnt = 0;

        for (ColoredLabel coloredLabel : coloredLabels) {
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

    void runInference(Bitmap bitmap) {
        try {
            TensorImage image = TensorImage.fromBitmap(bitmap);

            List<Segmentation> results = imageSegmenter.segment(image);
            Pair<Bitmap, Map<String, Integer>> result = createMaskBitmapAndLabels(results.get(0), image.getWidth(), image.getHeight());
            runOnUiThread(() -> outputView.setImageBitmap(stackBitmaps(Objects.requireNonNull(result.first), image.getBitmap())));
        } catch (Exception e) {
            Log.d("INFERENCE", "shat the bed");
            e.printStackTrace();
        }
    }

    void getInference(@NonNull ProcessCameraProvider cameraProvider) {
        Preview preview = new Preview.Builder()
                .build();

        CameraSelector cameraSelector = new CameraSelector.Builder()
                .requireLensFacing(CameraSelector.LENS_FACING_BACK)
                .build();

        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis imageAnalysis =
                new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(257, 257))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), image -> runInference(imageProxyToBitmap(image)));

        cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis, preview);
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        Log.d("INFERENCE", String.valueOf(image.getFormat()));
        ByteBuffer byteBuffer = image.getPlanes()[0].getBuffer();
        byteBuffer.rewind();
        byte[] bytes = new byte[byteBuffer.capacity()];
        byteBuffer.get(bytes);
        byte[] clonedBytes = bytes.clone();
        return BitmapFactory.decodeByteArray(clonedBytes, 0, clonedBytes.length);
    }

    private Bitmap convertImageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uvBuffer = planes[2].getBuffer();

        int ySize = yBuffer.remaining();
        int uvSize = uvBuffer.remaining();

        byte[] nv21 = new byte[ySize + uvSize];
        yBuffer.get(nv21, 0, ySize);
        uvBuffer.get(nv21, ySize, uvSize);

        YuvImage yuv = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuv.compressToJpeg(new Rect(0, 0, yuv.getWidth(), yuv.getWidth()), 50, out);
        byte[] imageBytes = out.toByteArray();
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }

    private Bitmap conversion(ImageProxy image) {
        Mat mat = Yuv.rgb(image.getImage());

        ImageProxy.PlaneProxy[] planes = image.getPlanes();
        ByteBuffer yBuffer = planes[0].getBuffer();
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();

        yBuffer.position(0);
        uBuffer.position(0);
        vBuffer.position(0);

        int ySize = yBuffer.remaining();
        int uSize = uBuffer.remaining();
        int vSize = vBuffer.remaining();

        ByteBuffer buffer = ByteBuffer.allocateDirect(ySize + uSize + vSize);
        buffer.put(yBuffer);
        buffer.put(uBuffer);
        buffer.put(vBuffer);

        Mat yuvMat = new Mat(257, 257, CvType.CV_8UC1);
        yuvMat.put(0, 0, buffer.array());

        Mat rgbMat = new Mat(yuvMat.rows(), yuvMat.cols(), CvType.CV_8UC4);
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV420sp2RGBA);

        final Bitmap bit = Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rgbMat, bit);

        byte[] nv21 = new byte[ySize + uSize + vSize];

        yBuffer.get(nv21, 0, ySize);
        vBuffer.get(nv21, ySize, vSize);
        uBuffer.get(nv21, ySize + vSize, uSize);

        YuvImage yuvImage = new YuvImage(nv21, ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, yuvImage.getWidth(), yuvImage.getHeight()), 100, out);
        byte[] imageBytes = out.toByteArray();

        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
    }
}