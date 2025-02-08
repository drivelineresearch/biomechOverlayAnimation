"""
Biomechanical Video Analysis Main Script
======================================

Author: Logan Kniss
Date: 2025-02-08
-----------
This script orchestrates the creation of biomechanical analysis videos by combining
motion capture data with force analysis visualizations.

Process Flow:
-----------
1. Load environment variables
2. Connect to database
3. Retrieve session data
4. Process video
5. Generate force analysis animation
6. Combine video and animation

Usage:
-----
python graph_video_overlay.py
"""

from assests import (
    get_db_engine,
    biomech_db,
    get_data,
    trim_video,
    animate_elbow_force,
    create_final_video
)
import os   
import argparse

def parse_arguments() -> dict:
    """Parse command line arguments using argparse.
    
    Returns:
        dict: Dictionary containing parsed arguments
    """
    parser = argparse.ArgumentParser(description='Process command line arguments.')
    parser.add_argument('--session_trial', required=True, help='Session trial id')
    parser.add_argument('--video_path', required=True, help='Path to input video file')
    return vars(parser.parse_args())

def main():
    """
    Main execution function for generating biomechanical analysis videos.
    """
    args = parse_arguments()
    if args['session_trial'] is None:
        raise ValueError("Session trial is required")
    if args['video_path'] is None:
        raise ValueError("Video path is required")
    
    session_trial = args['session_trial']
    video_path = args['video_path']
    
    # Get the desktop path for output files
    desktop_path = os.path.expanduser("~/Desktop")
    
    # Initialize database connection
    engine = get_db_engine(biomech_db)
    
    # Example video path - modify as needed
    video_path = "path/to/your/video.mp4"
    session_trial = "your_session_trial_id"
    
    # Process data and generate visualization
    first_frame, last_frame, df = get_data(session_trial, engine)
    fps, trimmed_video_path = trim_video(video_path, first_frame, last_frame, desktop_path)
    animation_path = animate_elbow_force(df, first_frame, last_frame, fps, desktop_path)
    
    # Create final composite video
    create_final_video(animation_path, trimmed_video_path, desktop_path)

if __name__ == "__main__":
    main()
