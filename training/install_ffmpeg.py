import subprocess
import sys

def install_ffmpeg():
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                            "--upgrade", "pip"])
    
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "--upgrade", "setuptools"])
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ffmpeg"])
        print("FFmpeg has been installed.")
    except subprocess.CalledProcessError as e:
        print("Error installing FFmpeg.")
    
    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        result = subprocess.run([
            "find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True, text=True
        )

        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed Static FFmpeg")
    
    except Exception as e:
        print(f"Error installing FFmpeg: {str(e)}")

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        print("FFmpeg version: ", result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("FFmpeg not found.")
        return False
