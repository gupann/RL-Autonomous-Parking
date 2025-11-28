import base64
import shutil
import warnings
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from IPython import display as ipythondisplay

try:
    from pyvirtualdisplay import Display
except Exception:
    Display = None


_virtual_display = None
# Disable virtual display to avoid Xvfb hanging issues in WSL2
# Video recording will still work with WSL2's built-in display support
if False:  # Intentionally disabled
    if Display is not None and shutil.which("Xvfb"):
        try:
            _virtual_display = Display(visible=0, size=(1400, 900))
            _virtual_display.start()
        except Exception as exc:
            _virtual_display = None
            warnings.warn(
                f"Could not start virtual display ({exc}). "
                "Video rendering will be disabled."
            )


def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(
        env, video_folder=video_folder, episode_trigger=lambda e: True
    )

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))
