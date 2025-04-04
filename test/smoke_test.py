"""Run smoke tests"""

import os
import sys
import sysconfig
from pathlib import Path

import torch
import torchvision
from torchvision.io import decode_avif, decode_heic, decode_image, decode_jpeg, read_file
from torchvision.models import resnet50, ResNet50_Weights


SCRIPT_DIR = Path(__file__).parent


def smoke_test_torchvision() -> None:
    print(
        "Is torchvision usable?",
        all(x is not None for x in [torch.ops.image.decode_png, torch.ops.torchvision.roi_align]),
    )


def smoke_test_torchvision_read_decode() -> None:
    img_jpg = decode_image(str(SCRIPT_DIR / "assets" / "encode_jpeg" / "grace_hopper_517x606.jpg"))
    if img_jpg.shape != (3, 606, 517):
        raise RuntimeError(f"Unexpected shape of img_jpg: {img_jpg.shape}")

    img_png = decode_image(str(SCRIPT_DIR / "assets" / "interlaced_png" / "wizard_low.png"))
    if img_png.shape != (4, 471, 354):
        raise RuntimeError(f"Unexpected shape of img_png: {img_png.shape}")

    img_webp = decode_image(str(SCRIPT_DIR / "assets/fakedata/logos/rgb_pytorch.webp"))
    if img_webp.shape != (3, 100, 100):
        raise RuntimeError(f"Unexpected shape of img_webp: {img_webp.shape}")

    if sys.platform == "linux":
        pass
        # TODO: Fix/uncomment below (the TODO below is mostly accurate but we're
        # still observing some failures on some CUDA jobs. Most are working.)
        # if torch.cuda.is_available():
        #     # TODO: For whatever reason this only passes on the runners that
        #     # support CUDA.
        #     # Strangely, on the CPU runners where this fails, the AVIF/HEIC
        #     # tests (ran with pytest) are passing. This is likely related to a
        #     # libcxx symbol thing, and the proper libstdc++.so get loaded only
        #     # with pytest? Ugh.
        #     img_avif = decode_avif(read_file(str(SCRIPT_DIR / "assets/fakedata/logos/rgb_pytorch.avif")))
        #     if img_avif.shape != (3, 100, 100):
        #         raise RuntimeError(f"Unexpected shape of img_avif: {img_avif.shape}")

        #     img_heic = decode_heic(
        #         read_file(str(SCRIPT_DIR / "assets/fakedata/logos/rgb_pytorch_incorrectly_encoded_but_who_cares.heic"))
        #     )
        #     if img_heic.shape != (3, 100, 100):
        #         raise RuntimeError(f"Unexpected shape of img_heic: {img_heic.shape}")
    else:
        try:
            decode_avif(str(SCRIPT_DIR / "assets/fakedata/logos/rgb_pytorch.avif"))
        except RuntimeError as e:
            assert "torchvision-extra-decoders" in str(e)

        try:
            decode_heic(str(SCRIPT_DIR / "assets/fakedata/logos/rgb_pytorch_incorrectly_encoded_but_who_cares.heic"))
        except RuntimeError as e:
            assert "torchvision-extra-decoders" in str(e)


def smoke_test_torchvision_decode_jpeg(device: str = "cpu"):
    img_jpg_data = read_file(str(SCRIPT_DIR / "assets" / "encode_jpeg" / "grace_hopper_517x606.jpg"))
    img_jpg = decode_jpeg(img_jpg_data, device=device)
    if img_jpg.shape != (3, 606, 517):
        raise RuntimeError(f"Unexpected shape of img_jpg: {img_jpg.shape}")


def smoke_test_compile() -> None:
    try:
        model = resnet50().cuda()
        model = torch.compile(model)
        x = torch.randn(1, 3, 224, 224, device="cuda")
        out = model(x)
        print(f"torch.compile model output: {out.shape}")
    except RuntimeError:
        if sys.platform == "win32":
            print("Successfully caught torch.compile RuntimeError on win")
        else:
            raise


def smoke_test_torchvision_resnet50_classify(device: str = "cpu") -> None:
    img = decode_image(str(SCRIPT_DIR / ".." / "gallery" / "assets" / "dog2.jpg")).to(device)

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights, progress=False).to(device)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms(antialias=True)

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    expected_category = "German shepherd"
    print(f"{category_name} ({device}): {100 * score:.1f}%")
    if category_name != expected_category:
        raise RuntimeError(f"Failed ResNet50 classify {category_name} Expected: {expected_category}")


def main() -> None:
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    print(f"{torch.ops.image._jpeg_version() = }")
    if not torch.ops.image._is_compiled_against_turbo():
        msg = "Torchvision wasn't compiled against libjpeg-turbo"
        if os.getenv("IS_M1_CONDA_BUILD_JOB") == "1":
            # When building the conda package on M1, it's difficult to enforce
            # that we build against turbo due to interactions with the libwebp
            # package. So we just accept it, instead of raising an error.
            print(msg)
        else:
            raise ValueError(msg)

    smoke_test_torchvision()
    smoke_test_torchvision_read_decode()
    smoke_test_torchvision_resnet50_classify()
    smoke_test_torchvision_decode_jpeg()
    if torch.cuda.is_available():
        smoke_test_torchvision_decode_jpeg("cuda")
        smoke_test_torchvision_resnet50_classify("cuda")

        #  torch.compile is not supported on Python 3.14+ and Python built with GIL disabled
        if sys.version_info < (3, 14, 0) and not sysconfig.get_config_var("Py_GIL_DISABLED"):
            smoke_test_compile()

    if torch.backends.mps.is_available():
        smoke_test_torchvision_resnet50_classify("mps")


if __name__ == "__main__":
    main()
