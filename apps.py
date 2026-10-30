import gradio as gr
import os
import shutil
from typing import *
import torch
import numpy as np
import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Define the pipeline and other settings (assuming it's already set up properly)
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()

# Max seed for random generation
MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)

# Define other helper functions (as per the existing code) ...

def image_to_3d(
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: gr.Request,
) -> Tuple[dict, str, str]:
    """
    Convert an image to a 3D model and return the path to the video and a redirect HTML string.
    """
    # Your existing image-to-3D conversion logic here...

    # Now return the video and a redirect URL
    redirect_html = """
        <html>
            <head>
                <meta http-equiv="refresh" content="2; url=https://huggingface.co/spaces/innoai/TRELLIS" />
                <script type="text/javascript">
                    window.location.href = "https://huggingface.co/spaces/innoai/TRELLIS";
                </script>
            </head>
            <body>
                <p>Redirecting...</p>
            </body>
        </html>
    """
    return state, video_path, redirect_html


with gr.Blocks(delete_cache=(600, 600)) as demo:
    gr.Markdown("""
    ## Image to 3D Asset with [TRELLIS](https://trellis3d.github.io/)
    * Upload an image and click "Generate" to create a 3D asset.
    """)
    
    # Add the input interface...
    # Your existing input and UI components here...

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        image_to_3d,
        inputs=[image_prompt, multiimage_prompt, is_multiimage, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo],
        outputs=[output_buf, video_output, "redirect_html"],  # Adding the new redirect output
    ).then(
        lambda: tuple([gr.Button(interactive=True), gr.Button(interactive=True)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    # Output the HTML for redirection when the model is generated
    gr.HTML().render(output="redirect_html")

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
