
## Configuration

### Video Settings
- Output Resolution: 1080x1520 (608px video + 912px plot)
- Video Codec: H.264
- Quality Preset: slower (high quality)
- CRF: 18 (high quality)
- Maximum Bitrate: 10Mbps

### Plot Styling
The `PlotStyle` class in `assests.py` contains all visualization settings:
- Dark theme with custom color palette
- Professional fonts and sizing
- Grid and axis styling
- Shadow effects for better visibility

## Troubleshooting

Common issues and solutions:

1. FFmpeg not found:
   - Ensure FFmpeg is installed and in system PATH
   - Try running `ffmpeg -version` in terminal

2. Database connection failed:
   - Verify .env file configuration
   - Check database credentials and permissions

3. Video processing errors:
   - Ensure sufficient disk space
   - Verify video file format compatibility
   - Check file permissions

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Author
Logan Kniss