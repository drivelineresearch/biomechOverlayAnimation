"""
Biomechanical Analysis Animation Assets
=====================================

This module provides utilities for creating animated visualizations of biomechanical data,
specifically focusing on elbow force analysis during pitching motions.

Key Components:
-------------
1. Database Connectivity
2. Data Processing
3. Video Processing
4. Animation Generation
5. Plot Styling
6. Video Composition

Dependencies:
-----------
- SQLAlchemy for database operations
- Pandas for data manipulation
- Matplotlib for plotting and animation
- OpenCV (cv2) for video processing
- FFmpeg for video composition
"""

import os
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text, Engine
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as path_effects
import cv2
import subprocess
load_dotenv()

## DATABASE THINGS ##
"""Database configuration and connection utilities"""

biomech_db = {
        'host': os.getenv("DB_HOST"),
        'user': os.getenv("DB_USER"),
        'password': quote_plus(os.getenv("DB_PW")),
        'database': os.getenv("PITCHING_DBNAME"),
        'port': os.getenv("DB_PORT")
    }

def create_db_url(user: str, password: str, host: str, port: str, db_name: str) -> str:
    return f'mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}'

def get_db_engine(db_creds: dict):
    engine_string = create_db_url(db_creds['user'], db_creds['password'], db_creds['host'], db_creds['port'], db_creds['database'])
    return create_engine(engine_string)
## DATABASE THINGS ##

## GET THE DATA FOR SESSION TRIAL ##
def get_data(session_trial: str, engine: Engine, show_head: bool = False):
    """
    Retrieves biomechanical data for a specific session trial from the database.
    
    Parameters:
    -----------
    session_trial : str
        The unique identifier for the session trial
    engine : Engine
        SQLAlchemy engine instance for database connection
    show_head : bool, optional
        If True, prints the first few rows of the retrieved data
        
    Returns:
    --------
    tuple
        (first_frame, last_frame, dataframe) containing the analysis window and data
    """
    if engine:
        print("Engine is connected")
    else:
        print("Engine is not connected")
    if not isinstance(session_trial, str):
        print("Session trial is not a string, converting")
        session_trial = str(session_trial)
    else:
        print("Session trial is a string, continuing")
    query = text("""
    SELECT 
        time,

        elbow_force_x,
        elbow_force_y,
        elbow_force_z,
        i_PKH,
        i_BR
    FROM joint_forces
    JOIN events USING (session_trial)
    WHERE session_trial = :session_trial
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"session_trial": session_trial})
        print('Dataframe created')
    if show_head:
        print(df.head())
        
    # Get first and last frame indices from dataframe
    first_frame = int(df.iloc[0]["i_PKH"]) - 100
    print(f'First frame: {first_frame}')
    last_frame = int(df.iloc[-1]["i_BR"]) + 100
    print(f'Last frame: {last_frame}')
    return first_frame, last_frame, df
## GET THE DATA FOR SESSION TRIAL ##

## TRIM VIDEO TO MATCH DATA ##
def trim_video(video_path: str, first_frame: int, last_frame: int, desktop_path: str) -> float:
    """
    Trim video to keep only frames between first_frame and last_frame indices.
    Saves the trimmed video to the user's desktop.

    Args:
        video_path: Path to the video file
        first_frame: Starting frame index to keep
        last_frame: Ending frame index to keep

    Returns:
        float: Frames per second of the video
    """
    print(f'Trimming video from {first_frame} to {last_frame}')
    video_name = os.path.basename(video_path).replace('.mp4', '_trimmed.mp4')
    output_path = os.path.join(desktop_path, video_name)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create video writer to save trimmed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Set video position to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    # Read and write frames between first_frame and last_frame
    current_frame = first_frame
    while cap.isOpened() and current_frame <= last_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Trimmed video saved to {output_path}')
    return fps, output_path
## TRIM VIDEO TO MATCH DATA ##

## PLOTTING STUFF ##
class PlotStyle:
    """Class to manage plot styling configurations"""
    # Color scheme
    BACKGROUND_COLOR = '#1f1f1f'
    TEXT_COLOR = '#ffffff'
    LINE_COLOR = '#00ff99'
    PKH_COLOR = '#ff4757'
    BR_COLOR = '#2ed573'
    GRID_COLOR = '#404040'
    
    # Font configurations
    FONT_FAMILY = 'DejaVu Sans'
    TITLE_SIZE = 14
    LABEL_SIZE = 12
    TICK_SIZE = 10
    
    @classmethod
    def apply_style(cls, fig, ax):
        """Apply consistent styling to figure and axis"""
        # Figure styling
        fig.patch.set_facecolor(cls.BACKGROUND_COLOR)
        
        # Axis styling
        ax.set_facecolor(cls.BACKGROUND_COLOR)
        ax.grid(True, linestyle='--', alpha=0.3, color=cls.GRID_COLOR)
        ax.spines['bottom'].set_color(cls.TEXT_COLOR)
        ax.spines['top'].set_color(cls.TEXT_COLOR)
        ax.spines['left'].set_color(cls.TEXT_COLOR)
        ax.spines['right'].set_color(cls.TEXT_COLOR)
        
        # Text styling
        ax.tick_params(colors=cls.TEXT_COLOR, size=cls.TICK_SIZE)
        ax.xaxis.label.set_color(cls.TEXT_COLOR)
        ax.yaxis.label.set_color(cls.TEXT_COLOR)
        ax.title.set_color(cls.TEXT_COLOR)
        
        # Font styling
        plt.rcParams['font.family'] = cls.FONT_FAMILY
        ax.title.set_size(cls.TITLE_SIZE)
        ax.xaxis.label.set_size(cls.LABEL_SIZE)
        ax.yaxis.label.set_size(cls.LABEL_SIZE)
        
def animate_elbow_force(df: pd.DataFrame, first_frame: int, last_frame: int, fps: float, desktop_path: str, show_plot: bool = False) -> None:
    print('Calculating elbow force magnitude')
    # Calculate elbow force magnitude first
    elbow_force_mag = np.sqrt(df["elbow_force_x"]**2 + df["elbow_force_y"]**2 + df["elbow_force_z"]**2)
    df["elbow_force_mag"] = elbow_force_mag
    print('Elbow force magnitude calculated')
    # Get the PKH and BR indices
    PKH = int(df.iloc[0]["i_PKH"])
    BR = int(df.iloc[0]["i_BR"])
    print(f'PKH: {PKH}, BR: {BR}')
    print('Setting up figure')
    # Adjust figure size for the animation 
    fig = plt.figure(figsize=(10, 8), dpi=120) # portrait orientation (9:16 aspect ratio)
    ax = fig.add_subplot(111)
    print('Figure set up')
    
    # Apply custom styling
    PlotStyle.apply_style(fig, ax)
    
    # Create main line with enhanced styling
    line, = ax.plot([], [], label="Elbow Force Magnitude", 
                    color=PlotStyle.LINE_COLOR, 
                    linewidth=2.5,
                    path_effects=[path_effects.SimpleLineShadow(offset=(1, -1)),
                                path_effects.Normal()])

    # Add vertical lines with enhanced styling
    ax.vlines(PKH, df["elbow_force_mag"].min(), df["elbow_force_mag"].max(), 
              color=PlotStyle.PKH_COLOR, linestyle="--", 
              label="Peak Knee Height", alpha=0.8, linewidth=2)
    ax.vlines(BR, df["elbow_force_mag"].min(), df["elbow_force_mag"].max(), 
              color=PlotStyle.BR_COLOR, linestyle="--", 
              label="Ball Release", alpha=0.8, linewidth=2)

    # Enhanced labels and title
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Elbow Force Magnitude (N)")
    ax.set_title("Dynamic Elbow Force Analysis During Pitch", pad=20)
    
    # Enhanced legend
    ax.legend(facecolor=PlotStyle.BACKGROUND_COLOR, 
             edgecolor=PlotStyle.TEXT_COLOR,
             labelcolor=PlotStyle.TEXT_COLOR,
             loc='upper right')
    plt.tight_layout(pad=1.0)
    
    def init():
        """Initialize the animation"""
        ax.set_xlim(first_frame, last_frame)
        ax.set_ylim(df["elbow_force_mag"].min() * 0.9, 
                    df["elbow_force_mag"].max() * 1.1)
        return line,

    def animate(frame):
        """Update the plot for each frame"""
        x_data = df.index[first_frame:frame]
        y_data = df["elbow_force_mag"][first_frame:frame]
        line.set_data(x_data, y_data)
        return line,

    # Create animation with smoother updating
    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=range(first_frame, last_frame + 1),
        interval=1000/fps,
        blit=True
    )
    
    plt.tight_layout()
    animation_path = os.path.join(desktop_path, 'elbow_force.mp4')
    anim.save(animation_path, writer='ffmpeg', fps=fps)
    if show_plot:
        plt.show()
    print(f'Animation saved to {animation_path}')
    return animation_path


## CREATE FINAL VIDEO ##
def create_final_video(animation_path: str, video_path: str, desktop_path: str) -> None:
    """
    Create a final video by overlaying the animation on top of the video.
    
    Args:
        animation_path: Path to the animation file
        video_path: Path to the video file
        desktop_path: Path to save the final video
    """
    print(f'Creating final video from {animation_path} and {video_path}')
    output_path = os.path.join(desktop_path, 'final_elbow_force_video.mp4')
    # Enhanced ffmpeg command with quality settings
    ffmpeg_command = (
        f'ffmpeg -i "{video_path}" -i "{animation_path}" '
        f'-filter_complex "'
        f'[0:v]scale=1080:608:force_original_aspect_ratio=decrease,pad=1080:608:(ow-iw)/2:(oh-ih)/2[v0];'  # Scale video
        f'[1:v]scale=1080:912:force_original_aspect_ratio=decrease,pad=1080:912:(ow-iw)/2:(oh-ih)/2[v1];'  # Scale animation
        f'[v0][v1]vstack=inputs=2" '  # Stack them vertically
        f'-c:v libx264 '
        f'-preset slower '
        f'-crf 18 '
        f'-maxrate 10M '
        f'-bufsize 20M '
        f'-pix_fmt yuv420p '
        f'"{output_path}"'
    )
    print(f'Running ffmpeg command: {ffmpeg_command}')
    subprocess.run(ffmpeg_command, shell=True)
    print(f'Final video saved to {output_path}')
## CREATE FINAL VIDEO ##
