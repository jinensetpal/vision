package com.necter.vision;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Bundle;
import android.renderscript.Allocation;
import android.renderscript.Element;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicYuvToRGB;
import android.renderscript.Type;
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
                        .setTargetResolution(new Size(1280, 720))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();

        imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), image -> {
            runInference(imageProxyToBitmap(image));
            image.close();
        });

        cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis, preview);
    }

//    @Override
//    public void onPreviewFrame(byte[] data, Camera camera) {
//        Bitmap bitmap = Bitmap.createBitmap(camera.getParameters().getPreviewSize().width(), r.height(), Bitmap.Config.ARGB_8888);
//        Allocation bmData = renderScriptNV21ToRGBA8888(
//                this, r.width(), r.height(), data);
//        bmData.copyTo(bitmap);
//    }

    public Allocation renderScriptNV21ToRGBA8888(Context context, int width, int height, byte[] nv21) {
        RenderScript rs = RenderScript.create(context);
        ScriptIntrinsicYuvToRGB yuvToRgbIntrinsic = ScriptIntrinsicYuvToRGB.create(rs, Element.U8_4(rs));

        Type.Builder yuvType = new Type.Builder(rs, Element.U8(rs)).setX(nv21.length);
        Allocation in = Allocation.createTyped(rs, yuvType.create(), Allocation.USAGE_SCRIPT);

        Type.Builder rgbaType = new Type.Builder(rs, Element.RGBA_8888(rs)).setX(width).setY(height);
        Allocation out = Allocation.createTyped(rs, rgbaType.create(), Allocation.USAGE_SCRIPT);

        in.copyFrom(nv21);

        yuvToRgbIntrinsic.setInput(in);
        yuvToRgbIntrinsic.forEach(out);
        return out;
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        Log.d("INFERENCE", String.valueOf(image.getFormat()));

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
//        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);

        ByteBuffer buffer = image.getPlanes()[0].getBuffer();
        byte[] bytes = new byte[buffer.remaining()];
        buffer.get(bytes);

        Bitmap b = Bitmap.createBitmap(image.getHeight(), image.getWidth(), Bitmap.Config.ARGB_8888);

        Mat mat = new Mat(image.getHeight()+image.getHeight()/2, image.getWidth(), CvType.CV_8UC1);
        mat.put(0, 0, bytes);
        Mat rgb = new Mat(image.getHeight(), image.getWidth(), CvType.CV_8UC4);
        Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_YUV420sp2BGR, 4);
        Utils.matToBitmap(rgb, b);

        return b;

//        ByteBuffer byteBuffer = image.getPlanes()[0].getBuffer();
//        byteBuffer.rewind();
//        byte[] bytes = new byte[byteBuffer.capacity()];
//        byteBuffer.get(bytes);
//        byte[] clonedBytes = bytes.clone();
//        return BitmapFactory.decodeByteArray(clonedBytes, 0, clonedBytes.length);
    }
}