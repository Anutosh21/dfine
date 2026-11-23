# Let's find objects, from a photo to a live camera

Load a photo, find the objects in it, then try your camera or a video. Adjust the
confidence slider and compare the original frame with the detections. Your photos
and camera frames stay in this browser.

- [Open the WebGPU notebook](https://anutosh21.github.io/dfine/lab/index.html?path=live-dfine-webgpu.ipynb)
- [Open the CPU notebook](https://anutosh21.github.io/dfine/lab/index.html?path=live-dfine-cpu.ipynb)
- [Open the optional benchmark](https://anutosh21.github.io/dfine/lab/index.html?path=benchmark.ipynb)

Both demos include completed before/after outputs from the sample photo. To try
your own photo, video or camera, select **Run → Run All Cells** in a fresh kernel.
Camera access starts only when you click **Start camera**.

The demo goes straight from loading and compiling the model to the detections
and live controls. Eager-reference checks and performance measurements are
separate: `content/benchmark.ipynb` contains completed CPU/WebGPU comparisons,
and `content/benchmark-results.json` keeps the recorded measurements. Run the
benchmark separately when needed; each run saves a device-specific JSON report.

We're using D-FINE Small's pretrained COCO weights, with 640×640 inputs and FP32
values. Model code and the sample photo are included in `content/`.

The model-loading cell downloads the original 39.9 MiB checkpoint from a pinned
GitHub revision and verifies its SHA-256 checksum. Later visits reuse the browser
cache while it remains available. Private browsing, clearing site data or browser
eviction can require another download. The kernel no longer waits for the weights
at startup.

Shoutout to [D-FINE](https://github.com/Peterande/D-FINE) and the
[ExecuTorch object-detection demo](https://executorch.ai/playground/webgpu/#/demo/detect)
for the inspiration! Built from the
[Xeus-Lite template](https://github.com/jupyterlite/xeus-lite-demo).

Model: Apache-2.0. Photo: Elf; derivatives by Djmirko and FT2, CC BY-SA 3.0.
Displayed photo derivatives retain that license. Licensing and source details are
in `content/LICENSE-D-FINE` and `content/sources.json`.

To deploy, set **Settings → Pages → Source → GitHub Actions**, then push to `main`.

The browser bundle omits development headers, static libraries and package tests.
Both notebooks use Turbine/IREE for compilation and execution on their selected device.

To build and check the same site locally before deploying:

```bash
micromamba create -n dfine-site -f .github/build-environment.yml
micromamba run -n dfine-site jupyter lite build --contents content --output-dir dist
micromamba run -n dfine-site python -m http.server 8000 --directory dist
```

Open `http://localhost:8000/lab/index.html`. The saved results are visible without
running the model; run all cells to recreate the live controls.
