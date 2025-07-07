import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageSequence
import cv2  # Using OpenCV for resizing as it's common in video processing
from ASD import ASD  # Assuming ASD class is defined in ASD.py
from test_input import test_input  # Assuming test_input is defined in test_input.py


# ===============================================================
# Model Wrapper for Video-Only Inference
# ===============================================================

class VideoOnlyASDModel(nn.Module):
    """ Wraps the video-only ASD model for inference. """

    def __init__(self, asd_model: ASD):
        super(VideoOnlyASDModel, self).__init__()
        self.asdModel = asd_model.model
        # Use the FC layer from the video-only loss function
        self.fc_layer = asd_model.lossV.FC

    def forward(self, visualFeature):
        """
        Defines the video-only inference path.
        """
        visualEmbed = self.asdModel.forward_visual_frontend(visualFeature)
        outsV = self.asdModel.forward_visual_backend(visualEmbed)
        x = self.fc_layer(outsV)
        # Get the raw score for the positive class
        predScore = x[:, 1]
        return predScore


# ===============================================================
# GIF Processing and Inference Logic
# ===============================================================

def preprocess_frame(frame_image):
    """
    Processes a single frame from a GIF to match the model's input requirements.
    Converts to grayscale, resizes, and crops.
    """
    # Convert PIL image to NumPy array
    frame = np.array(frame_image)

    # Convert to grayscale. If it has an alpha channel, use RGBA2GRAY.
    if frame.ndim == 3 and frame.shape[2] == 4:
        face = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)
    elif frame.ndim == 3:
        face = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
        face = frame  # Already grayscale

    # Resize and crop, similar to the original evaluation script
    face = cv2.resize(face, (224, 224))
    face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
    return face


def run_inference_on_gif(args):
    """
    Loads the PyTorch model, processes a GIF, and runs inference.
    """
    # 1. Load PyTorch Model with trained weights
    print("--- Loading PyTorch model ---")
    complete_model_pt = ASD(device='cpu')
    complete_model_pt.loadParameters(args.weights)
    complete_model_pt.eval()

    video_model_pt = VideoOnlyASDModel(complete_model_pt)
    print("PyTorch video-only model loaded successfully.")

    # 2. Load and process the GIF
    print(f"\n--- Loading and processing GIF: {args.gif_path} ---")
    try:
        gif = Image.open(args.gif_path)
    except FileNotFoundError:
        print(f"❌ ERROR: GIF file not found at {args.gif_path}")
        return

    video_frames = [preprocess_frame(frame.convert('RGB')) for frame in ImageSequence.Iterator(gif)]
    video_feature_np = np.array(video_frames)

    if video_feature_np.shape[0] == 0:
        print("❌ ERROR: Could not extract any frames from the GIF.")
        return

    print(f"Processed {video_feature_np.shape[0]} frames from the GIF.")

    flat = video_feature_np.flatten()

    # 2️⃣ Print as comma-separated float literals
    with open('video_feature_np.txt', 'w') as f:
        f.write(', '.join(f'{x:.6f}' for x in flat))

    # 3. Prepare tensor for the model
    # The model expects a batch dimension, so we add one with unsqueeze(0)
    video_input_pt = torch.from_numpy(video_feature_np).float().unsqueeze(0)
    print("Video input tensor shape:", video_input_pt.shape)

    # 4. Run PyTorch Inference
    print("\n--- Running inference ---")
    with torch.no_grad():
        scores_pt = video_model_pt(video_input_pt)
        scores_pt_np = scores_pt.numpy()

    swift = np.asarray(
        [-1.255859, -1.359375, -1.201172, -1.023438, -0.8579102, -0.9780273, -0.9291992, -0.4914551, -0.3720703,
         -0.730957, -0.3803711, -0.1286621, 0.1263428, -0.05007935, 0.2629395, 0.8803711, 1.268555, 1.426758, 1.334961,
         1.553711, 1.643555, 1.369141, 1.445312, 1.146484, 1.03418])
    np.testing.assert_almost_equal(scores_pt_np, swift, decimal=2)

    # 5. Display results
    print("\n--- Inference Results ---")
    print(f"Model returned {len(scores_pt_np)} scores (one per frame).")
    for i, score in enumerate(scores_pt_np):
        print(f"Frame {i + 1:03d}: Score = {score:.4f}")


def run_inference_on_tensor(video_feature_np, args):
    """
    Loads the PyTorch model, processes a GIF, and runs inference.
    """
    # 1. Load PyTorch Model with trained weights
    print("--- Loading PyTorch model ---")
    complete_model_pt = ASD(device='cpu')
    complete_model_pt.loadParameters(args.weights)
    complete_model_pt.eval()

    video_model_pt = VideoOnlyASDModel(complete_model_pt)
    print("PyTorch video-only model loaded successfully.")

    if video_feature_np.shape[0] == 0:
        print("❌ ERROR: Could not extract any frames from the GIF.")
        return

    print(f"Processed {video_feature_np.shape[0]} frames from the GIF.")

    # 3. Prepare tensor for the model
    # The model expects a batch dimension, so we add one with unsqueeze(0)
    video_input_pt = torch.from_numpy(video_feature_np).float()
    print(video_input_pt.shape)

    # 4. Run PyTorch Inference
    print("\n--- Running inference ---")
    with torch.no_grad():
        scores_pt = video_model_pt(video_input_pt)
        scores_pt_np = scores_pt.numpy()

    # 5. Display results
    print("\n--- Inference Results ---")
    print(f"Model returned {len(scores_pt_np)} scores (one per frame).")
    for i, score in enumerate(scores_pt_np):
        print(f"Frame {i + 1:03d}: Score = {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run video-only ASD inference on a GIF file.")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the .model file with saved PyTorch weights.')
    parser.add_argument('--gif-path', type=str, required=True, help='Path to the input GIF file.')
    args = parser.parse_args()
    run_inference_on_gif(args)
    #run_inference_on_tensor(test_input, args)



if __name__ == '__main__':
    # NOTE: You must have:
    # 1. An 'ASD.py' file containing the 'ASD' class definition.
    # 2. Pillow and OpenCV installed: pip install Pillow opencv-python
    main()
