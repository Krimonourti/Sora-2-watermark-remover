# Sora 2 Watermark Remover

This project provides a production-ready FastAPI web application that accepts a Sora&nbsp;2 video
with the default "SORA2" watermark and returns a cleaned version of the clip. The backend performs
frame-by-frame watermark detection via template matching and reconstructs occluded pixels using
OpenCV's inpainting algorithms.

## Features

- 🌐 **Modern web UI** with drag-and-drop ready upload form and status messaging.
- 🧠 **Template-matching watermark localisation** driven by OpenCV's `matchTemplate` operator.
- 🎨 **Automatic inpainting** that uses the Telea method to restore content behind the watermark.
- ⚙️ **Configurable pipeline** allowing different watermark templates or detection thresholds.
- 📦 **Stateless processing**—uploads are stored in a temporary directory and cleaned results are
  streamed back to the client.

## How it works

1. A user uploads a Sora&nbsp;2 video through the `/` page.
2. The `/process` endpoint stores the upload and calls `app.watermark_removal.process_video`.
3. The first frame is analysed with `cv2.matchTemplate` using the provided watermark template
   (default file: `assets/sora2_template.pgm`). If the match confidence is high enough, the bounding
   box is reused for subsequent frames.
4. A binary mask is generated for the detected watermark area and dilated to cover the halo around
   the logo.
5. Each frame is cleaned using `cv2.inpaint(..., cv2.INPAINT_TELEA)` and written to the output
   stream.
6. Once processing completes, the resulting MP4 is sent back to the browser as a downloadable file.

### Customising the template

The included `assets/sora2_template.pgm` is a placeholder block rendering of the stock Sora&nbsp;2
watermark. For production, replace it with a high-quality grayscale crop of the actual watermark:

1. Export a still frame from a Sora&nbsp;2 video.
2. Crop the watermark area tightly and convert it to grayscale.
3. Save it in the `assets/` directory as `sora2_template.pgm` or change the `template_path` argument
   when calling `process_video`.
4. Adjust the `threshold` parameter (default `0.35`) if detection is unreliable.

## Getting started

### Requirements

- Python 3.11+
- `ffmpeg` installed and available on the system `PATH` (OpenCV relies on it for some codecs)

Install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the app

```bash
uvicorn app.main:app --reload
```

Visit <http://localhost:8000> and upload a watermarked video. The cleaned file will automatically
download when processing finishes.

### Using the processing module directly

```python
from pathlib import Path
from app.watermark_removal import process_video

output = process_video(Path("input.mp4"), Path("outputs"))
print(f"Cleaned video saved to: {output}")
```

### Troubleshooting

- **Watermark not detected** – Replace the template with an exact crop and lower the threshold to
  `0.25`.
- **Video codec errors** – Ensure `ffmpeg` is installed or re-encode your input to H.264 MP4.
- **Artifacts after inpainting** – Increase the dilation kernel in `_build_inpaint_mask` or use a
  custom mask tailored to the watermark's shape.

## License

MIT
