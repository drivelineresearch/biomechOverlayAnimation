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
hitting_db = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': quote_plus(os.getenv("DB_PW")),
    'database': os.getenv("HITTING_DBNAME"),
    'port': os.getenv("DB_PORT")
}

def create_db_url(user: str, password: str, host: str, port: str, db_name: str) -> str:
    return f'mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}'

def get_db_engine(db_creds: dict):
    engine_string = create_db_url(db_creds['user'], db_creds['password'], db_creds['host'], db_creds['port'], db_creds['database'])
    return create_engine(engine_string)
## DATABASE THINGS ##

## GET THE DATA FOR SESSION TRIAL ##
def get_pitching_data(session_trial: str, engine: Engine, show_head: bool = False):
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
    
    metadata = {}
    query = text("""
    SELECT
        name, date, pitch_speed_mph,
        joint_angles.time,
        joint_angles.shoulder_angle_x,
        i_PKH, i_BR, i_MER
    FROM joint_angles
    JOIN trials USING (session_trial)
    JOIN events USING (session_trial)
    JOIN poi USING (session_trial)
    JOIN sessions USING (session)
    JOIN users using (user)
    WHERE session_trial = :session_trial
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"session_trial": session_trial})
        print('Dataframe created')
    if show_head:
        print(df.head())
    metadata["name"], metadata["date"], metadata["pitch_speed_mph"] = df.iloc[0]["name"], df.iloc[0]["date"], df.iloc[0]["pitch_speed_mph"]
    print(f'Name: {metadata["name"]}, Date: {metadata["date"]}, Pitch Speed: {metadata["pitch_speed_mph"]}')
    # Get first and last frame indices from dataframe
    first_frame = int(df.iloc[0]["i_PKH"]) - 100
    print(f'First frame: {first_frame}')
    last_frame = int(df.iloc[-1]["i_BR"]) + 100
    print(f'Last frame: {last_frame}')
    return first_frame, last_frame, df, metadata

def get_data(mode: str, session_trial: str, engine: Engine, table: str, show_head: bool = False):
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
    
    metadata = {}
    if mode == 'hitting':
        query = text(f"""
        SELECT
            name, date, bat_speed_max as bat_speed_mph, exit_velo_mph,
            {table}.time as joint_time,
            {table}.*,
            load_time,
            ds_time,
            fp_time,
            contact_time
            FROM {table}
            JOIN trials USING (session_trial)
            JOIN events USING (session_trial)
            JOIN poi USING (session_trial)
            JOIN sessions USING (session)
            JOIN users using (user)
            WHERE session_trial = :session_trial
        """)
    elif mode == 'pitching':
        query = text(f"""
        SELECT
            name, date, pitch_speed_mph,
            {table}.time as joint_time,
            {table}.*,
            i_PKH, i_BR, i_MER
            FROM {table}
            JOIN trials USING (session_trial)
            JOIN events USING (session_trial)
            JOIN poi USING (session_trial)
            JOIN sessions USING (session)
            JOIN users using (user)
            WHERE session_trial = :session_trial
        """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection, params={"session_trial": session_trial})
        print('Dataframe created')
    if show_head:
        print(df.head())
    if mode == 'hitting':
        metadata["name"], metadata["date"], metadata["bat_speed_mph"], metadata["exit_velo_mph"] = df.iloc[0]["name"], df.iloc[0]["date"], df.iloc[0]["bat_speed_mph"], df.iloc[0]["exit_velo_mph"]
        print(f'Name: {metadata["name"]}, Date: {metadata["date"]}, Bat Speed: {metadata["bat_speed_mph"]}, Exit Velo: {metadata["exit_velo_mph"]}')
    elif mode == 'pitching':
        metadata["name"], metadata["date"], metadata["pitch_speed_mph"] = df.iloc[0]["name"], df.iloc[0]["date"], df.iloc[0]["pitch_speed_mph"]
        print(f'Name: {metadata["name"]}, Date: {metadata["date"]}, Pitch Speed: {metadata["pitch_speed_mph"]}')
    # Get frame indices where event times are closest to time values
    df = df.reset_index(drop=True)
    if mode == 'hitting':
        # Get frame indices where event times are closest to time values
        df['i_load'] = (df['joint_time'] - df['load_time'].iloc[0]).abs().idxmin()
        df['i_ds'] = (df['joint_time'] - df['ds_time'].iloc[0]).abs().idxmin()
        df['i_fp'] = (df['joint_time'] - df['fp_time'].iloc[0]).abs().idxmin()
        df['i_contact'] = (df['joint_time'] - df['contact_time'].iloc[0]).abs().idxmin()
        first_frame = int(df.iloc[0]["i_load"]) - 125
        last_frame = int(df.iloc[0]["i_contact"]) + 50
    elif mode == 'pitching':
        first_frame = int(df.iloc[0]["i_PKH"]) - 100
        last_frame = int(df.iloc[-1]["i_BR"]) + 100
    print(f'First frame: {first_frame}')
    print(f'Last frame: {last_frame}')
    
    return first_frame, last_frame, df, metadata
## GET THE DATA FOR SESSION TRIAL ##

## TRIM VIDEO TO MATCH DATA ##
def trim_video(video_path: str, first_frame: int, last_frame: int, desktop_path: str, metadata: dict, use_gpu: bool = True) -> tuple:
    """
    Trim video to keep only frames between first_frame and last_frame indices using FFmpeg.
    Saves the trimmed video to the user's desktop.

    Args:
        video_path: Path to the video file
        first_frame: Starting frame index to keep
        last_frame: Ending frame index to keep
        desktop_path: Path to save the trimmed video
        metadata: Dictionary containing metadata about the video
        use_gpu: Whether to attempt using NVIDIA GPU acceleration if available

    Returns:
        tuple: (fps, output_path) - Frames per second of the video and path to trimmed video
    """
    name = metadata["name"]
    date = metadata["date"]
    try:
        mph = metadata["pitch_speed_mph"]
    except:
        mph = metadata["exit_velo_mph"]
    
    # Get video info using OpenCV just to get the FPS
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f'Video info: {total_frames} frames at {fps} FPS')
    print(f'Trimming video from frame {first_frame} to {last_frame}')
    
    # Validate frame indices
    if first_frame < 0:
        print(f"Warning: first_frame ({first_frame}) is negative, setting to 0")
        first_frame = 0
    if last_frame >= total_frames:
        print(f"Warning: last_frame ({last_frame}) exceeds total frames ({total_frames}), setting to {total_frames-1}")
        last_frame = total_frames - 1
    if first_frame >= last_frame:
        print(f"Error: first_frame ({first_frame}) must be less than last_frame ({last_frame})")
        return fps, None
    
    # Calculate time positions based on frame numbers
    start_time = first_frame / fps
    duration = (last_frame - first_frame + 1) / fps
    
    print(f'Start time: {start_time:.3f}s, Duration: {duration:.3f}s')
    
    # Create output filename
    video_name = os.path.basename(video_path).replace('.mp4', f'_{name}_{date}_{mph}_trimmed.mp4')
    output_path = os.path.join(desktop_path, video_name)
    
    # Check for NVIDIA GPU availability if requested
    gpu_available = False
    
    if use_gpu:
        # Try to detect NVIDIA GPU encoder
        try:
            nvenc_check = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if 'h264_nvenc' in nvenc_check.stdout:
                gpu_available = True
                print("NVIDIA GPU encoder (NVENC) detected")
            else:
                print("NVIDIA GPU encoder not detected, using CPU encoding")
        except Exception as e:
            print(f"Error checking for NVIDIA GPU encoder: {e}")
            print("Falling back to CPU encoding")
    
    # Build FFmpeg command with seek parameter before input for faster seeking
    ffmpeg_cmd = ['ffmpeg', '-ss', f'{start_time:.3f}', '-i', video_path]
    
    # Add duration parameter
    ffmpeg_cmd.extend(['-t', f'{duration:.3f}'])
    
    # Configure video codec based on GPU availability
    if gpu_available:
        print("Using NVIDIA GPU acceleration")
        ffmpeg_cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',      # NVENC preset (p1-p7, p7 is highest quality)
            '-profile:v', 'high',
            '-rc:v', 'vbr',       # Variable bitrate
            '-cq:v', '19'         # Quality level (lower is better)
        ])
    else:
        # CPU encoding with libx264
        ffmpeg_cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22'
        ])
    
    # Audio settings (copy if present)
    ffmpeg_cmd.extend(['-c:a', 'aac'])
    
    # Additional settings
    ffmpeg_cmd.extend([
        '-avoid_negative_ts', 'make_zero',
        '-y',  # Overwrite output file if it exists
        output_path
    ])
    
    # Convert command list to string for printing
    cmd_str = ' '.join(ffmpeg_cmd)
    print(f'Running FFmpeg command: {cmd_str}')
    
    # Run FFmpeg command
    try:
        result = subprocess.run(
            ffmpeg_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print("FFmpeg process completed successfully")
        
        # Check if output file exists and has a reasonable size
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f'Trimmed video saved to {output_path}')
            return fps, output_path
        else:
            print(f"Warning: Output file is missing or too small")
            return fps, None
            
    except subprocess.CalledProcessError as e:
        print(f"Error running FFmpeg: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return fps, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return fps, None
## TRIM VIDEO TO MATCH DATA ##

## PLOTTING STUFF ##
class PlotStyle:
    """Class to manage plot styling configurations"""
    # Color scheme
    BACKGROUND_COLOR = '#1f1f1f'
    TEXT_COLOR = '#ffffff'
    LINE_COLOR = '#00ff99'
    # pitching events
    PKH_COLOR = '#ff4757'
    BR_COLOR = '#1e90ff'
    MER_COLOR = '#ffa500'
    # hitting events
    LOAD_COLOR = 'blue'
    DS_COLOR = 'black'
    FP_COLOR = 'red'
    CONTACT_COLOR = 'purple'
    
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
        
def animate_elbow_force(df: pd.DataFrame, first_frame: int, last_frame: int, fps: float, desktop_path: str, metadata: dict, show_plot: bool = False) -> None:
    print('Calculating elbow force magnitude')
    # Calculate elbow force magnitude first
    elbow_force_mag = np.sqrt(df["elbow_force_x"]**2 + df["elbow_force_y"]**2 + df["elbow_force_z"]**2)
    df["elbow_force_mag"] = elbow_force_mag
    print('Elbow force magnitude calculated')
    name = metadata["name"]
    date = metadata["date"]
    pitch_speed_mph = metadata["pitch_speed_mph"]
    # Get the PKH and BR indices
    PKH = int(df.iloc[0]["i_PKH"])
    BR = int(df.iloc[0]["i_BR"])
    MER = int(df.iloc[0]["i_MER"])
    # Calculate normalized positions for PKH and BR
    total_frames = last_frame - first_frame
    PKH_normalized = (PKH - first_frame) / total_frames
    BR_normalized = (BR - first_frame) / total_frames
    MER_normalized = (MER - first_frame) / total_frames
    print(f'PKH: {PKH_normalized:.2f}, BR: {BR_normalized:.2f}, MER: {MER_normalized:.2f}')
    print('Setting up figure')
    # Adjust figure size for the animation - wider but same height
    fig = plt.figure(figsize=(16, 8), dpi=120)
    # Keep some minimal padding for labels and title
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    ax = fig.add_subplot(111)
    print('Figure set up')
    
    # Apply custom styling
    PlotStyle.apply_style(fig, ax)
    
    # Create main line with enhanced styling
    line, = ax.plot([], [], label="Elbow Torque", 
                    color=PlotStyle.LINE_COLOR, 
                    linewidth=2.5,
                    path_effects=[path_effects.SimpleLineShadow(offset=(1, -1)),
                                path_effects.Normal()])

    # Store vertical lines as global variables so we can update them
    vlines = []
    
    # Enhanced labels and title
    ax.set_xlabel("Normalized Time")
    # ax.set_ylabel("Elbow Force Magnitude (N)")
    ax.set_ylabel("Elbow Torque (Nm)")
    ax.set_title("Dynamic Elbow Torque Analysis During Pitch", pad=20)
    
    def init():
        """Initialize the animation"""
        ax.set_xlim(0, 1)  # Set x-axis from 0 to 1
        # ax.set_ylim(df["elbow_force_mag"].min() * 0.9, 
        #             df["elbow_force_mag"].max() * 1.1)
        ax.set_ylim(df["elbow_moment_y"].min() * 1.1, 
                    df["elbow_moment_y"].max() * 1.1)
        
        # Clear any existing vertical lines
        for vline in vlines:
            vline.remove()
        vlines.clear()
        
        # Add vertical lines with normalized positions
        vlines.extend([
            ax.axvline(x=PKH_normalized, color=PlotStyle.PKH_COLOR, linestyle="solid", 
                      label="Peak Knee Height", alpha=0.8, linewidth=2),
            ax.axvline(x=BR_normalized, color=PlotStyle.BR_COLOR, linestyle="solid", 
                      label="Ball Release", alpha=0.8, linewidth=2),
            ax.axvline(x=MER_normalized, color=PlotStyle.MER_COLOR, linestyle="solid", 
                      label="Max External Rotation", alpha=0.8, linewidth=2)
        ])
        
        # Create legend after all plot elements exist
        legend = ax.legend(facecolor=PlotStyle.BACKGROUND_COLOR, 
                          edgecolor=PlotStyle.TEXT_COLOR,
                          labelcolor=PlotStyle.TEXT_COLOR,
                          loc='upper left',
                          prop={'size': 14})
        
        # Set legend title with custom style
        legend.set_title(
            f'Pitch Speed: {pitch_speed_mph} mph',
            prop={
                'family': 'DejaVu Sans',  # More readable sans-serif font
                'size': 14,
                'weight': 'normal'  # Less bold for better readability
            }
        )
        legend.get_title().set_color('yellow')
        
        plt.tight_layout(pad=1.0)
        
        return (line,) + tuple(vlines)

    def animate(frame):
        """Update the plot for each frame"""
        current_length = frame - first_frame
        x_data = [(i - first_frame) / total_frames for i in df.index[first_frame:frame]]
        # y_data = df["elbow_force_mag"][first_frame:frame]
        y_data = df["elbow_moment_y"][first_frame:frame]
        line.set_data(x_data, y_data)
        return (line,) + tuple(vlines)

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
    animation_path = os.path.join(desktop_path, f'elbow_torque_{name}_{date}_{pitch_speed_mph}.mp4')
    anim.save(animation_path, writer='ffmpeg', fps=fps)
    if show_plot:
        plt.show()
    print(f'Animation saved to {animation_path}')
    return animation_path

def animate_graph(mode: str, df: pd.DataFrame, first_frame: int, last_frame: int, fps: float, desktop_path: str, metadata: dict, metric: str, label: str, measurment: str, show_plot: bool = False) -> None:
    print(f'Setting up {metric} animation')
    name = metadata["name"]
    date = metadata["date"]
    total_frames = last_frame - first_frame
    if mode == 'hitting':
        bat_speed_mph = metadata["bat_speed_mph"]
        exit_velo_mph = metadata["exit_velo_mph"]
        
        LOAD = int(df.iloc[0]["i_load"])
        DS = int(df.iloc[0]["i_ds"])
        FP = int(df.iloc[0]["i_fp"])
        CONTACT = int(df.iloc[0]["i_contact"])
        print(f'LOAD: {LOAD}, DS: {DS}, FP: {FP}, CONTACT: {CONTACT}')
        
        LOAD_normalized = (LOAD - first_frame) / total_frames
        DS_normalized = (DS - first_frame) / total_frames
        FP_normalized = (FP - first_frame) / total_frames
        CONTACT_normalized = (CONTACT - first_frame) / total_frames
        print(f'LOAD: {LOAD_normalized:.2f}, DS: {DS_normalized:.2f}, FP: {FP_normalized:.2f}, CONTACT: {CONTACT_normalized:.2f}')

        animation_path = os.path.join(desktop_path, f'{metric}_{name}_{date}_{bat_speed_mph}_{exit_velo_mph}.mp4') if bat_speed_mph is not None else os.path.join(desktop_path, f'{metric}_{name}_{date}_{exit_velo_mph}.mp4')
    elif mode == 'pitching':
        pitch_speed_mph = metadata["pitch_speed_mph"]
        PKH = int(df.iloc[0]["i_PKH"])
        BR = int(df.iloc[0]["i_BR"])
        MER = int(df.iloc[0]["i_MER"])

        PKH_normalized = (PKH - first_frame) / total_frames
        BR_normalized = (BR - first_frame) / total_frames
        MER_normalized = (MER - first_frame) / total_frames
        print(f'PKH: {PKH_normalized:.2f}, BR: {BR_normalized:.2f}, MER: {MER_normalized:.2f}')

        animation_path = os.path.join(desktop_path, f'{metric}_{name}_{date}_{pitch_speed_mph}.mp4')
    # Check for valid FPS value
    if fps <= 0:
        print(f"Warning: Invalid FPS value ({fps}). Setting to default 30 FPS.")
        fps = 30.0
    
    print('Setting up figure')
    # Adjust figure size for the animation - wider but same height
    fig = plt.figure(figsize=(16, 8), dpi=120)
    # Keep some minimal padding for labels and title
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
    ax = fig.add_subplot(111)
    print('Figure set up')
    
    # Apply custom styling
    PlotStyle.apply_style(fig, ax)
    
    # Create main line with enhanced styling
    line, = ax.plot([], [], label=label, 
                    color=PlotStyle.LINE_COLOR, 
                    linewidth=2.5,
                    path_effects=[path_effects.SimpleLineShadow(offset=(1, -1)),
                                path_effects.Normal()])

    # Store vertical lines as global variables so we can update them
    vlines = []
    
    # Enhanced labels and title
    ax.set_xlabel("Normalized Time")
    # ax.set_ylabel("Elbow Force Magnitude (N)")
    if measurment is not None:
        ax.set_ylabel(f"{label} ({measurment})")
    else:
        ax.set_ylabel(label)
    ax.set_title(label, pad=20)
    
    def init():
        """Initialize the animation"""
        ax.set_xlim(0, 1)  # Set x-axis from 0 to 1
        
        # Old version
        # ax.set_ylim(df[metric].min() * 1.1, 
        #             df[metric].max() * 1.1)
        
        # New version with better padding
        y_min = df[metric][first_frame:last_frame].min()
        y_max = df[metric][first_frame:last_frame].max()
        y_range = y_max - y_min
        
        # Add 10% padding on both sides
        ax.set_ylim(y_min - 0.1 * y_range, 
                    y_max + 0.1 * y_range)
        
        # Clear any existing vertical lines
        for vline in vlines:
            vline.remove()
        vlines.clear()
        
        # Add vertical lines with normalized positions
        if mode == 'hitting':
            vlines.extend([
                ax.axvline(x=LOAD_normalized, color=PlotStyle.LOAD_COLOR, linestyle="solid", 
                        label="Load", alpha=0.8, linewidth=2),
                ax.axvline(x=DS_normalized, color=PlotStyle.DS_COLOR, linestyle="solid", 
                        label="Start Swing", alpha=0.8, linewidth=2),
                ax.axvline(x=FP_normalized, color=PlotStyle.FP_COLOR, linestyle="solid", 
                        label="Foot Plant", alpha=0.8, linewidth=2),
                ax.axvline(x=CONTACT_normalized, color=PlotStyle.CONTACT_COLOR, linestyle="solid", 
                        label="Contact", alpha=0.8, linewidth=2)
            ])
            
        elif mode == 'pitching':
            vlines.extend([
                ax.axvline(x=PKH_normalized, color=PlotStyle.PKH_COLOR, linestyle="solid", 
                        label="Peak Knee Height", alpha=0.8, linewidth=2),
                ax.axvline(x=MER_normalized, color=PlotStyle.MER_COLOR, linestyle="solid", 
                        label="Max External Rotation", alpha=0.8, linewidth=2),
                ax.axvline(x=BR_normalized, color=PlotStyle.BR_COLOR, linestyle="solid", 
                        label="Ball Release", alpha=0.8, linewidth=2)
            ])

        # Create legend after all plot elements exist
        legend = ax.legend(facecolor=PlotStyle.BACKGROUND_COLOR, 
                          edgecolor=PlotStyle.TEXT_COLOR,
                          labelcolor=PlotStyle.TEXT_COLOR,
                          loc='upper left',
                          prop={'size': 14})
        
        if mode == 'hitting':
            legend.set_title(
                f'Bat Speed: {bat_speed_mph} mph, Exit Velo: {exit_velo_mph} mph' if bat_speed_mph is not None else f'Exit Velo: {exit_velo_mph} mph',
                prop={
                    'family': 'DejaVu Sans',  # More readable sans-serif font
                    'size': 14,
                    'weight': 'normal'  # Less bold for better readability
                }
            )
        elif mode == 'pitching':
            legend.set_title(
                f'Pitch Speed: {pitch_speed_mph} mph',
                prop={
                    'family': 'DejaVu Sans',  # More readable sans-serif font
                    'size': 14,
                    'weight': 'normal'  # Less bold for better readability
                }
            )
        legend.get_title().set_color('yellow')
        
        plt.tight_layout(pad=1.0)
        
        return (line,) + tuple(vlines)

    def animate(frame):
        """Update the plot for each frame"""
        current_length = frame - first_frame
        x_data = [(i - first_frame) / total_frames for i in df.index[first_frame:frame]]
        y_data = df[metric][first_frame:frame]
        line.set_data(x_data, y_data)
        return (line,) + tuple(vlines)

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
    anim.save(animation_path, writer='ffmpeg', fps=fps)
    if show_plot:
        plt.show()
    print(f'Animation saved to {animation_path}')
    return animation_path


## CREATE FINAL VIDEO ##
def create_final_video(mode: str, animation_path: str, video_path: str, metric, desktop_path: str, metadata: dict) -> None:
    """
    Create a final video by overlaying the animation on top of the video.
    
    Args:
        animation_path: Path to the animation file
        video_path: Path to the video file
        desktop_path: Path to save the final video
    """
    print(f'Creating final video from {animation_path} and {video_path}')
    name = metadata["name"]
    date = metadata["date"]
    label = metric['label']
    try:
        if '-side' in video_path:
            view = 'side'
        elif '-front' in video_path:
            view = 'front'
    except:
        view = None

    if mode == 'hitting':
        bat_speed_mph = metadata["bat_speed_mph"]
        exit_velo_mph = metadata["exit_velo_mph"]
        if view is not None:
            output_path = os.path.join(desktop_path, f'final_{label}_{view}_video_{name}_{date}_{bat_speed_mph}_{exit_velo_mph}.mp4') if bat_speed_mph is not None else os.path.join(desktop_path, f'final_{label}_{view}_video_{name}_{date}_{exit_velo_mph}.mp4')
        else:
            output_path = os.path.join(desktop_path, f'final_{label}_video_{name}_{date}_{bat_speed_mph}_{exit_velo_mph}.mp4') if bat_speed_mph is not None else os.path.join(desktop_path, f'final_{label}_video_{name}_{date}_{exit_velo_mph}.mp4')
    elif mode == 'pitching':    
        pitch_speed_mph = metadata["pitch_speed_mph"]
        output_path = os.path.join(desktop_path, f'final_{label}_video_{name}_{date}_{pitch_speed_mph}.mp4')
    # ffmpeg cmd for normal video
    # ffmpeg_command = (
    #     f'ffmpeg -i "{video_path}" -i "{animation_path}" '
    #     f'-filter_complex "'
    #     f'[0:v]scale=1920:480:force_original_aspect_ratio=decrease,pad=1920:480:(ow-iw)/2:(oh-ih)/2[v0];'  # Scale video (shorter)
    #     f'[1:v]scale=1920:600:force_original_aspect_ratio=decrease,pad=1920:600:(ow-iw)/2:(oh-ih)/2[v1];'  # Scale plot (taller)
    #     f'[v0][v1]vstack=inputs=2:shortest=1,setsar=1" '  # Stack them vertically with no padding
    #     f'-c:v libx264 '
    #     f'-preset slower '
    #     f'-crf 18 '
    #     f'-maxrate 10M '
    #     f'-bufsize 20M '
    #     f'-pix_fmt yuv420p '
    #     f'"{output_path}"'
    # )
    # split video and plot
    ffmpeg_command = (
        f'ffmpeg -i "{video_path}" -i "{animation_path}" '
        f'-filter_complex "'
        f'[0:v]scale=960:-1:force_original_aspect_ratio=1,pad=960:540:(ow-iw)/2:(oh-ih)/2[v0];'  # Scale width only, auto-height
        f'[1:v]scale=960:-1:force_original_aspect_ratio=1,pad=960:540:(ow-iw)/2:(oh-ih)/2[v1];'  # Scale width only, auto-height
        f'[v0][v1]hstack" '  # Simple horizontal stack
        f'-c:v libx264 '
        f'-preset slower '
        f'-crf 18 '
        f'-maxrate 10M '
        f'-bufsize 20M '
        f'-y '  # Force overwrite output file if it exists
        f'"{output_path}"'
    )
        
    print(f'Running ffmpeg command: {ffmpeg_command}')
    subprocess.run(ffmpeg_command, shell=True)
    print(f'Final video saved to {output_path}')
## CREATE FINAL VIDEO ##
