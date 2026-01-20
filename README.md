# Object Detection Auto-Annotation for Crowd Analysis

An automated annotation tool that uses YOLOv8 to detect and annotate people in images, generating YOLO-format labels for crowd detection datasets.

## Overview

This project automates the process of annotating images containing people using a pre-trained YOLOv8 model. It processes batch images, detects persons, and outputs both annotated images with bounding boxes and YOLO-format label files suitable for training custom object detection models.

## Features

- 🤖 **Automated Detection**: Uses YOLOv8n model for person detection
- 📦 **Batch Processing**: Processes entire directories of images automatically
- 🎯 **YOLO Format Output**: Generates normalized bounding box coordinates in YOLO format
- 🖼️ **Visual Output**: Saves annotated images with drawn bounding boxes and confidence scores
- ⚙️ **Configurable Threshold**: Adjustable confidence threshold for detection filtering
- 📊 **Multiple Format Support**: Supports JPG, JPEG, PNG, and BMP image formats

## Project Structure

```
Object_Detection_Annotation_Crowd/
├── autoannotate.py              # Main annotation script
├── labelimg_installation.txt    # LabelImg setup instructions
├── Dataset/
│   └── images/                  # Input images directory
├── models/
│   └── yolov8n.pt              # YOLOv8 nano model
└── output/
    ├── images/                  # Annotated images with bounding boxes
    └── labels/                  # YOLO format annotation files (.txt)
```

## Requirements

### Python Dependencies

```bash
opencv-python (cv2)
ultralytics
```

### System Requirements

- Python 3.8+
- Sufficient disk space for image processing
- GPU recommended for faster processing (optional)

## Installation

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install opencv-python ultralytics
   ```

3. **Verify model file**: Ensure `models/yolov8n.pt` exists (will be downloaded automatically by ultralytics if missing)

## Usage

### Quick Start

1. Place your images in the `Dataset/images/` directory

2. Run the auto-annotation script:
   ```bash
   python autoannotate.py
   ```

3. Find results in the `output/` directory:
   - `output/images/` - Annotated images with bounding boxes
   - `output/labels/` - YOLO format label files

### Configuration

You can modify these parameters in `autoannotate.py`:

```python
IMAGE_DIR = "Dataset/images"        # Input images directory
OUTPUT_DIR = "output"                # Output directory
MODEL_PATH = "models/yolov8n.pt"    # YOLOv8 model path
PERSON_CLASS_ID = 0                  # Class ID for person (0 in COCO dataset)
CONF_THRESHOLD = 0.25                # Confidence threshold (0.0 - 1.0)
```

### YOLO Label Format

Generated label files follow the YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: Object class (0 for person)
- `x_center`, `y_center`: Normalized center coordinates (0.0 - 1.0)
- `width`, `height`: Normalized box dimensions (0.0 - 1.0)

Example:
```
0 0.512345 0.487654 0.125678 0.234567
0 0.723456 0.345678 0.098765 0.187654
```

## Manual Annotation with LabelImg (Optional)

For manual review or correction of annotations, you can use LabelImg:

### LabelImg Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

2. Clone and install LabelImg:
   ```bash
   git clone https://github.com/HumanSignal/labelImg
   pip install setuptools lxml PyQt5
   cd labelImg
   pyrcc5 -o libs/resources.py resources.qrc
   ```

3. Run LabelImg:
   ```bash
   python labelImg.py
   ```

4. Configure LabelImg:
   - Set "Open Dir" to `output/images/`
   - Set "Change Save Dir" to `output/labels/`
   - Select YOLO format from the menu

## Workflow

1. **Prepare Dataset**: Place raw images in `Dataset/images/`
2. **Auto-Annotate**: Run `autoannotate.py` to generate initial annotations
3. **Review (Optional)**: Use LabelImg to review and correct annotations
4. **Train Model**: Use the annotated dataset to train custom YOLO models

## Output Examples

### Annotated Images
Images are saved with:
- Green bounding boxes around detected persons
- Confidence scores displayed above each box

### Label Files
Each image gets a corresponding `.txt` file with one line per detected person:
```
0 0.345678 0.567890 0.123456 0.234567
0 0.678901 0.234567 0.098765 0.156789
```

## Tips & Best Practices

- **Confidence Threshold**: Adjust `CONF_THRESHOLD` based on your needs:
  - Lower (0.15-0.25): More detections, may include false positives
  - Higher (0.35-0.50): Fewer but more confident detections

- **Model Selection**: YOLOv8n is fast but less accurate. Consider:
  - `yolov8s.pt` - Small (better accuracy)
  - `yolov8m.pt` - Medium (balanced)
  - `yolov8l.pt` - Large (high accuracy)

- **Batch Processing**: Process images in batches to monitor progress
  
- **Manual Review**: Always review auto-annotations for critical applications

## Troubleshooting

### Issue: No detections in output
- Check if images are in supported formats (JPG, PNG, BMP)
- Lower the `CONF_THRESHOLD` value
- Verify images contain people visible enough for detection

### Issue: Model file not found
- The YOLOv8 model will auto-download on first run
- Ensure internet connection for initial download
- Or manually download from [Ultralytics](https://github.com/ultralytics/ultralytics)

### Issue: Out of memory
- Process images in smaller batches
- Resize large images before processing
- Use a smaller model (yolov8n)

## License

This project uses:
- **Ultralytics YOLOv8**: [AGPL-3.0 License](https://github.com/ultralytics/ultralytics)
- **OpenCV**: Apache 2.0 License

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection model
- [LabelImg](https://github.com/HumanSignal/labelImg) - Image annotation tool

## Contact & Support

For issues, questions, or contributions, please refer to the project repository.

---

**Happy Annotating! 🎯**
