package com.lib.vision;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.util.Log;
import android.util.Size;
import android.widget.ImageView;

import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import androidx.core.util.Pair;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.segmenter.ColoredLabel;
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter;
import org.tensorflow.lite.task.vision.segmenter.Segmentation;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

public class Inference {
    private final ImageView outputView;
    private final ImageSegmenter imageSegmenter;

    public Inference(ImageView outputView, ImageSegmenter imageSegmenter) {
        this.outputView = outputView;
        this.imageSegmenter = imageSegmenter;
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

    public Bitmap runInference(Bitmap bitmap) {
        TensorImage image = TensorImage.fromBitmap(bitmap);
        List<Segmentation> results = imageSegmenter.segment(image);
        Pair<Bitmap, Map<String, Integer>> result = createMaskBitmapAndLabels(results.get(0), image.getWidth(), image.getHeight());
        return stackBitmaps(Objects.requireNonNull(result.first), image.getBitmap());
    }

    @SuppressLint("UnsafeExperimentalUsageError")
    public Bitmap imageProxyToBitmap(ImageProxy image) {
        @SuppressLint("UnsafeOptInUsageError") YuvImage yuvImage = new YuvImage(YUV_420_888toNV21(Objects.requireNonNull(image.getImage())), ImageFormat.NV21, image.getWidth(), image.getHeight(), null);
        ByteArrayOutputStream os = new ByteArrayOutputStream();
        yuvImage.compressToJpeg(new Rect(0, 0, image.getWidth(), image.getHeight()), 100, os);
        byte[] jpegByteArray = os.toByteArray();
        return BitmapFactory.decodeByteArray(jpegByteArray, 0, jpegByteArray.length);
    }

    private static byte[] YUV_420_888toNV21(Image image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int ySize = width * height;
        int uvSize = width * height / 4;

        byte[] nv21 = new byte[ySize + uvSize * 2];

        ByteBuffer yBuffer = image.getPlanes()[0].getBuffer(); // Y
        ByteBuffer uBuffer = image.getPlanes()[1].getBuffer(); // U
        ByteBuffer vBuffer = image.getPlanes()[2].getBuffer(); // V

        int rowStride = image.getPlanes()[0].getRowStride();
        if (BuildConfig.DEBUG && !(image.getPlanes()[0].getPixelStride() == 1)) {
            throw new AssertionError("Assertion failed");
        }

        int pos = 0;

        if (rowStride == width) { // likely
            yBuffer.get(nv21, 0, ySize);
            pos += ySize;
        } else {
            long yBufferPos = -rowStride; // not an actual position
            for (; pos < ySize; pos += width) {
                yBufferPos += rowStride;
                yBuffer.position((int) yBufferPos);
                yBuffer.get(nv21, pos, width);
            }
        }

        rowStride = image.getPlanes()[2].getRowStride();
        int pixelStride = image.getPlanes()[2].getPixelStride();

        if (BuildConfig.DEBUG && !(rowStride == image.getPlanes()[1].getRowStride())) {
            throw new AssertionError("Assertion failed");
        }
        if (BuildConfig.DEBUG && !(pixelStride == image.getPlanes()[1].getPixelStride())) {
            throw new AssertionError("Assertion failed");
        }

        if (pixelStride == 2 && rowStride == width && uBuffer.get(0) == vBuffer.get(1)) {
            byte savePixel = vBuffer.get(1);
            try {
                vBuffer.put(1, (byte) ~savePixel);
                if (uBuffer.get(0) == (byte) ~savePixel) {
                    vBuffer.put(1, savePixel);
                    vBuffer.position(0);
                    uBuffer.position(0);
                    vBuffer.get(nv21, ySize, 1);
                    uBuffer.get(nv21, ySize + 1, uBuffer.remaining());

                    return nv21; // shortcut
                }
            } catch (ReadOnlyBufferException ignore) {
            }
            vBuffer.put(1, savePixel);
        }

        for (int row = 0; row < height / 2; row++) {
            for (int col = 0; col < width / 2; col++) {
                int vuPos = col * pixelStride + row * rowStride;
                nv21[pos++] = vBuffer.get(vuPos);
                nv21[pos++] = uBuffer.get(vuPos);
            }
        }

        return nv21;
    }

    public void recurrentInference(Context context, ListenableFuture<ProcessCameraProvider> cameraProviderFuture, CameraSelector cameraSelector, Size size) {
        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                ImageAnalysis imageAnalysis =
                        new ImageAnalysis.Builder()
                                .setTargetResolution(size)
                                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                                .build();

                imageAnalysis.setAnalyzer(Executors.newSingleThreadExecutor(), image -> {
                    Bitmap bmap = runInference(imageProxyToBitmap(image));
                    ((Activity) context).runOnUiThread(() -> outputView.setImageBitmap(bmap));
                    image.close();
                });

                cameraProvider.bindToLifecycle((LifecycleOwner) context, cameraSelector, imageAnalysis);
            } catch (ExecutionException | InterruptedException ignored) {
            }
        }, ContextCompat.getMainExecutor(context));
    }
}
