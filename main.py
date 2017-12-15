"""
a simple file of run shadowless
"""
from perception.camera_manager.mono_camera_manager import MonoManager


def shadowless():

    mono_manager = MonoManager()
    mono_manager.serve_local('./videos/road_demo.mp4', is_record=True)


if __name__ == '__main__':
    shadowless()