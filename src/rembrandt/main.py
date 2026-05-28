import datetime
from pathlib import Path

from rembrandt.camera_poses import sample_camera_poses
from rembrandt.scene import Scene

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_OBJECT = PROJECT_ROOT / "test-obj" / "12951_Stone_Chess_Board_v1_L3.obj"


def main() -> None:
    poses = sample_camera_poses(
        n=10,
        azimuth_range=(0.0, 360.0),
        elevation_range=(-10.0, 30.0),
        distance_range=(3.0, 5.0),
        strategy="random",
        seed=42,
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    scene = Scene()
    scene.load_object(SAMPLE_OBJECT)
    scene.center_target()
    scene.add_light(light_type="SUN", location=(2, -3, 5), energy=3.0)
    scene.add_light(light_type="POINT", location=(-2, 2, 3))
    scene.add_camera(focal_length=50)

    for i, pose in enumerate(poses):
        scene.move_camera(location=pose.location, look_at=pose.look_at)

        out = scene.render(f"output/{stamp}/frame_{i:04d}.png", resolution=(640, 640))
        print(f"Rendered frame {i} to {out}")


if __name__ == "__main__":
    main()
