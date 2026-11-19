# Let's find objects, from a photo to a live camera

Load a photo, find the objects in it, then try your camera or a video. Adjust the
confidence slider and compare the original frame with the detections. Your photos
and camera frames stay in this browser.

- [Open the WebGPU notebook](https://anutosh21.github.io/dfine/lab/index.html?path=live-dfine-webgpu.ipynb)
- [Open the CPU notebook](https://anutosh21.github.io/dfine/lab/index.html?path=live-dfine-cpu.ipynb)

Run the cells from top to bottom in a fresh kernel, then click **Start camera** and
allow access. You can also use the sample photo or upload your own. Run one notebook
at a time when comparing CPU and WebGPU.

We're using D-FINE Small's pretrained COCO weights, with 640×640 inputs and FP32
values. Model code, weights and the sample photo are included in `content/`.

Shoutout to [D-FINE](https://github.com/Peterande/D-FINE) and the
[ExecuTorch object-detection demo](https://executorch.ai/playground/webgpu/#/demo/detect)
for the inspiration! Built from the
[Xeus-Lite template](https://github.com/jupyterlite/xeus-lite-demo).

Model: Apache-2.0. Photo: Elf; derivatives by Djmirko and FT2, CC BY-SA 3.0.
Displayed photo derivatives retain that license. Licensing and source details are
in `content/LICENSE-D-FINE` and `content/sources.json`.

To deploy, set **Settings → Pages → Source → GitHub Actions**, then push to `main`.
