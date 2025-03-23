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
    hitting_db,
    get_pitching_data,
    get_data,
    trim_video,
    animate_elbow_force,
    animate_graph,
    create_final_video
)
import os   
import argparse

metric_dict = {
    'rear_elbow_angle_z': {
        'label': 'Wrist Pronation',
        'measurment': 'deg',
        'table': 'joint_angles',
        'video_path1': r"W:\ArizonaFacility\2025\03\11\Verdugo_A_103757_HittingAssessment_ArizonaFacility_11-Mar-25_1741722199_H7_18_107.1.mp4",
        'video_path2': None
    },
    # 'pelvis_angle_y': {
    #     'label': 'Pelvis Forward Bend',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603346_pelvis-side.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603239_pelvis-front.mp4"
    # },
    # 'pelvis_angle_x': {
    #     'label': 'Pelvis Side Bend',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603346_pelvis-side.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603239_pelvis-front.mp4"
    # },
    # 'torso_angle_z': {
    #     'label': 'Torso Rotation',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603163_torso-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740602797_torso-side.mp4"
    # },
    # 'rear_hip_angle_x': {
    #     'label': 'Hip Flexion',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740604184_hip-flexion-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740604020_hip-flexion-side.mp4"
    # },
    # 'lead_knee_angle_x': {
    #     'label': 'Front Knee Flexion',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740604426_front-knee-flexion-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740611153_front-knee-flexion-side.mp4"
    # },
    # 'rear_elbow_angle_x': {
    #     'label': 'Rear Elbow Flexion',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740604968_rear-elbow-flexion-side.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605138_rear-elbow-flexion-front.mp4"
    # },
    # 'rear_shoulder_angle_y': {
    #     'label': 'Rear Shoulder Adduction',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605310_rear-shoulder-adduction_upper-arm-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605403_rear-shoulder-adduction_upper-arm-side.mp4"
    # },
    # 'pelvis_velo_z': {
    #     'label': 'Pelvis Angular Velocity',
    #     'measurment': 'deg/s',
    #     'table': 'joint_velos',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603346_pelvis-side.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603239_pelvis-front.mp4"
    # },
    # 'torso_velo_z': {
    #     'label': 'Torso Angular Velocity',
    #     'measurment': 'deg/s',
    #     'table': 'joint_velos',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740603163_torso-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740602797_torso-side.mp4"
    # },
    # 'upper_arm_speed_mag_x': {
    #     'label': 'Upper Arm Angular Velocity',
    #     'measurment': 'deg/s',
    #     'table': 'fml_extra_report_cols',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740620204_upper-arm-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740619957_upper-arm-side.mp4"
    # },
    # 'hand_speed_mag_x': {
    #     'label': 'Hand Angular Velocity',
    #     'measurment': 'deg/s',
    #     'table': 'fml_extra_report_cols',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605778_lead-hand-side.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605861_lead-hand-front.mp4"
    # },
    # 'lead_wrist_angle_x': {
    #     'label': 'Wrist Flexion',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605778_lead-hand-side.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605861_lead-hand-front.mp4"
    # },
    # 'rear_elbow_angle_x': {
    #     'label': 'Rear Elbow Flexion',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740605138_rear-elbow-flexion-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740604968_rear-elbow-flexion-side.mp4"
    # },
    # 'rear_wrist_angle_z': {
    #     'label': 'Rear Wrist Pronation',
    #     'measurment': 'deg',
    #     'table': 'joint_angles',
    #     'video_path1': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740620286_rear-hand-front.mp4",
    #     'video_path2': r"C:\Users\logan.kniss\Downloads\animation_2226_9_1740620376_rear-hand-side.mp4"
    # }
}

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
    # args = parse_arguments()
    # if args['session_trial'] is None:
    #     raise ValueError("Session trial is required")
    # if args['video_path'] is None:
    #     raise ValueError("Video path is required")
    
    # session_trial = args['session_trial']
    # video_path = args['video_path']
    session_trial = "2550_18"
    # video_path = r"C:\Users\logan.kniss\Downloads\animation_2226_1_1740526960.mp4"
    
    # Get the desktop path for output files
    desktop_path = os.path.expanduser("~/Desktop")
    desktop_path = desktop_path + "\synced_videos"
    
    
    
    # Process data and generate visualization
    
    # PITCHING
    # Initialize database connection
    # engine = get_db_engine(biomech_db)
    # first_frame, last_frame, df, metadata = get_pitching_data(session_trial, engine)
    # fps, trimmed_video_path = trim_video(video_path, first_frame, last_frame, desktop_path, metadata)
    # animation_path = animate_elbow_force(df, first_frame, last_frame, fps, desktop_path, metadata)
    
    # HITTING
    # Initialize database connection
    engine = get_db_engine(hitting_db)
    for metric in metric_dict:
        first_frame, last_frame, df, metadata = get_data('hitting', session_trial, engine, metric_dict[metric]['table'])
        fps, trimmed_video_path = trim_video(metric_dict[metric]['video_path1'], first_frame, last_frame, desktop_path, metadata)
        print(f'{metric} trimmed video saved to {trimmed_video_path}')
        if metric_dict[metric]['video_path2'] is not None:
            _, trimmed_video_path2 = trim_video(metric_dict[metric]['video_path2'], first_frame, last_frame, desktop_path, metadata)
            print(f'{metric} trimmed video saved to {trimmed_video_path2}')
        # fps, trimmed_video_path = trim_video(video_path, first_frame, last_frame, desktop_path, metadata)
        animation_path = animate_graph('hitting', df, first_frame, last_frame, fps, desktop_path, metadata, metric, metric_dict[metric]['label'], metric_dict[metric]['measurment'])
        print(f'{metric} animation saved to {animation_path}')
        # Create final composite video
        create_final_video('hitting', animation_path, trimmed_video_path, metric_dict[metric], desktop_path, metadata)
        # create_final_video(animation_path, trimmed_video_path2, metric_dict[metric], desktop_path, metadata)

if __name__ == "__main__":
    main()
