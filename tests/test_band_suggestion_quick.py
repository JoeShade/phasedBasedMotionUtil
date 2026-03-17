import sys
from pathlib import Path
import numpy as np
from phase_motion_app.core.ffprobe import FfprobeRunner
from phase_motion_app.core.baseline_band import analyze_source_frequency_band

def main():
    input_dir = Path("a:/Documents/phasedBasedMotionUtil/input")
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in {".mp4", ".avi", ".mov"}]
    if not files:
        print("No video files found in input folder.")
        return
    for file in files:
        print(f"\nAnalyzing: {file.name}")
        try:
            probe = FfprobeRunner().probe(str(file))
            suggestion = analyze_source_frequency_band(file, probe)
            print(f"Suggested band: {suggestion.low_hz:.2f} - {suggestion.high_hz:.2f} Hz (peak: {suggestion.peak_hz:.2f} Hz, confidence: {suggestion.confidence:.2f})")
        except Exception as e:
            print(f"Error analyzing {file.name}: {e}")

if __name__ == "__main__":
    main()
