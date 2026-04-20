# Low-Cost Football Analytics

A low-cost football analytics project that uses deep learning and computer vision to extract useful insights from football match video. The repository focuses on making video-based football analysis more accessible with open-source Python tools such as PyTorch, Ultralytics, OpenCV, scikit-learn, and Streamlit.

## Overview

`low-cost-football-analytics` is intended for students, analysts, coaches, and football fans who want to experiment with practical football analytics without expensive tracking systems or commercial data subscriptions.

The project can be used as a starting point for workflows such as detecting players and the ball, tracking movement, estimating team behavior, and building simple visual or interactive analysis outputs from video.

## Features

- Computer-vision based football video analysis
- Deep learning workflow using PyTorch and Ultralytics
- Video and image processing with OpenCV and Pillow
- Machine learning utilities with scikit-learn
- Streamlit support for interactive demos or dashboards
- Extensible structure for scouting, match review, and tactical analysis

## Repository Structure

The main project files are currently located in:

```text
Football-Analytics-with-Deep-Learning-and-Computer-Vision-master/
```

Current top-level structure:

```text
.
├── Football-Analytics-with-Deep-Learning-and-Computer-Vision-master/
│   └── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

Recommended setup:

- Python 3.10 or newer
- Git
- A virtual environment
- A local football video file for testing

### Installation

Clone the repository:

```bash
git clone https://github.com/TIANjiming07/low-cost-football-analytics.git
cd low-cost-football-analytics
```

Move into the main project folder:

```bash
cd Football-Analytics-with-Deep-Learning-and-Computer-Vision-master
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

> Note: The dependency file currently lists `python-opencv`. If installation fails, try installing `opencv-python` instead.

## Usage

A typical workflow is:

1. Add or reference a football match video.
2. Run the computer vision pipeline to detect and track objects.
3. Generate annotated video, metrics, or dashboard outputs.
4. Review the results and refine the analysis.

If the project includes a Streamlit app, run it from the main project folder with:

```bash
streamlit run app.py
```

If the main script has a different filename, replace `app.py` with the correct entry point.

## Potential Analysis Outputs

This project can be extended to support:

- Player detection and tracking
- Ball detection and tracking
- Team identification
- Speed and distance estimation
- Possession or phase-of-play analysis
- Annotated match clips
- Interactive dashboards for match review

## Data and Models

This repository may require video files, trained model weights, or other large assets that are not stored directly in GitHub. If you add those files, consider documenting:

- Where the data comes from
- How to download or prepare it
- Where model weights should be placed
- Any licensing or usage restrictions

## Roadmap

- Document the exact entry-point script for running the project.
- Add sample input/output screenshots or videos.
- Add clearer model weight setup instructions.
- Include example commands for common analysis tasks.
- Add troubleshooting notes for dependency installation.
- Add tests or validation scripts for core analysis steps.

## Contributing

Contributions are welcome. Helpful improvements include:

- Better setup instructions
- Bug fixes for dependencies or scripts
- Example videos or screenshots
- New analysis metrics
- Streamlit dashboard improvements
- Documentation for model weights and data sources

## License

No license has been specified yet. Add a license before distributing or reusing this project publicly.
