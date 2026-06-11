"""Scene management for synthetic dataset rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import bpy
from mathutils import Vector

from rembrandt.camera.fit import fit_distance
from rembrandt.camera.intrinsics import limiting_fov_from_camera
from rembrandt.camera.orientation import (
    require_nonzero_direction,
    rotation_euler_from_forward,
)
from rembrandt.convention import SourceUpAxis, obj_import_axes
from rembrandt.errors import ModelFileNotFoundError, RenderEngineUnavailableError
from rembrandt.light_poses import DEFAULT_LIGHT_ENERGY
from rembrandt.obj_assets import normalize_obj_mtllibs, resolve_texture_file

_CAMERA_LOOK_AT_ERROR = "Camera location and look_at cannot be the same point."
_LIGHT_LOOK_AT_ERROR = "Light location and look_at cannot be the same point."
_EEVEE_GPU_FAILURE_HINTS = (
    "gpu",
    "egl",
    "opengl",
    "gl context",
    "vulkan",
    "no display",
    "cannot create",
    "failed to initialize",
)


def eevee_failure_is_gpu_context(exc: BaseException) -> bool:
    """Return whether an EEVEE render failure looks like a missing GPU context.

    Args:
        exc: Exception raised during ``bpy.ops.render.render()`` with EEVEE.

    Returns:
        True when the error message suggests a GPU / graphics-context problem.
    """
    message = str(exc).casefold()
    return any(hint in message for hint in _EEVEE_GPU_FAILURE_HINTS)


class Scene:
    """Manages a Blender scene for synthetic dataset rendering.

    Wraps bpy's global scene state and keeps references to objects we
    create or import, so downstream code (bbox projection, randomization)
    can access them without re-querying bpy.
    """

    def __init__(self, *, clear: bool = True) -> None:
        """Initialize the scene.

        Args:
            clear: If True, remove all existing objects from the scene
                on init. Defaults to True since Rembrandt always renders
                from a fresh scene.
        """
        self.targets: list[bpy.types.Object] = []
        self.camera: bpy.types.Object | None = None
        self.lights: list[bpy.types.Object] = []
        self._camera_requested_location: tuple[float, float, float] | None = None
        self._camera_look_at: tuple[float, float, float] | None = None
        self._camera_fit_target = False
        self._camera_fit_margin = 1.2
        if clear:
            self.clear()

    def clear(self) -> None:
        """Remove all objects from the scene and reset tracked references."""
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        self.targets = []
        self.camera = None
        self.lights = []
        self._camera_requested_location = None
        self._camera_look_at = None
        self._camera_fit_target = False

    def clear_lights(self) -> None:
        """Remove all tracked lights (objects and data blocks) from the scene."""
        for light_obj in self.lights:
            light_data = light_obj.data
            bpy.data.objects.remove(light_obj, do_unlink=True)
            bpy.data.lights.remove(light_data)
        self.lights = []
        bpy.context.view_layer.update()

    def load_object(self, obj_path: str | Path, *, up_axis: SourceUpAxis = "Z") -> bpy.types.Object:
        """Load an .obj file as the target object for rendering.

        Args:
            obj_path: Path to the .obj file.
            up_axis: Native up-axis of the source OBJ.

        Returns:
            The imported mesh object.

        Raises:
            ModelFileNotFoundError: If the file does not exist.
            RuntimeError: If the .obj contains no mesh objects.
        """
        path = Path(obj_path).resolve()
        if not path.exists():
            raise ModelFileNotFoundError(str(path))

        asset_dir = path.parent
        normalize_obj_mtllibs(path)

        forward_axis, import_up_axis = obj_import_axes(up_axis)

        # Blender 4.x: bpy.ops.wm.obj_import (replaces import_scene.obj).
        bpy.ops.wm.obj_import(
            filepath=str(path),
            directory=str(asset_dir),
            forward_axis=forward_axis,
            up_axis=import_up_axis,
        )
        self._reload_missing_textures(asset_dir)

        imported = [o for o in bpy.context.selected_objects if o.type == "MESH"]
        if not imported:
            raise RuntimeError(f"No mesh objects found in {path}")

        self.targets = imported
        return imported[0]

    @staticmethod
    def _image_has_pixels(image: bpy.types.Image) -> bool:
        """Return whether an image has loadable pixel data."""
        return image.size[0] > 0 and len(image.pixels) > 0

    def _reload_missing_textures(self, asset_dir: Path) -> None:
        """Reload image textures that Blender could not resolve during OBJ import."""
        for image in bpy.data.images:
            if image.packed_file is not None or self._image_has_pixels(image):
                continue

            filename = Path(bpy.path.basename(image.filepath)).name
            if not filename:
                continue

            resolved = resolve_texture_file(asset_dir, filename)
            if resolved is None:
                continue

            image.filepath = str(resolved)
            image.reload()

        bpy.ops.file.find_missing_files(directory=str(asset_dir))

    def add_camera(
        self,
        location: tuple[float, float, float] = (5.0, 5.0, 5.0),
        look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
        focal_length: float = 50.0,
        fit_target: bool = True,
        fit_margin: float = 1.2,
    ) -> bpy.types.Object:
        """Create a camera, point it at a target, and set it active.

        Args:
            location: Camera position (x, y, z) in world coordinates.
            look_at: World-space point the camera aims at.
            focal_length: Focal length in mm. 50mm is "standard" / human-eye.
            fit_target: If True and a target is loaded, move the camera back
                along the requested view direction until the target fits.
            fit_margin: Extra framing margin around the target.

        Returns:
            The created camera object.
        """
        camera_data = bpy.data.cameras.new("Camera")
        camera_data.lens = focal_length

        camera_obj = bpy.data.objects.new("Camera", camera_data)
        bpy.context.collection.objects.link(camera_obj)
        self.camera = camera_obj
        bpy.context.scene.camera = camera_obj

        return self.move_camera(
            location=location,
            look_at=look_at,
            fit_target=fit_target,
            fit_margin=fit_margin,
        )

    def move_camera(
        self,
        location: tuple[float, float, float],
        look_at: tuple[float, float, float] = (0.0, 0.0, 0.0),
        fit_target: bool = True,
        fit_margin: float = 1.2,
    ) -> bpy.types.Object:
        """Move the existing camera, point it at a target, and keep it active.

        Args:
            location: New camera position (x, y, z) in world coordinates.
            look_at: World-space point the camera aims at.
            fit_target: If True and a target is loaded, move the camera back
                along the requested view direction until the target fits.
            fit_margin: Extra framing margin around the target.

        Returns:
            The existing camera object after repositioning.

        Raises:
            RuntimeError: If no camera has been added to the scene.
        """
        if self.camera is None:
            raise RuntimeError("No camera in the scene. Call add_camera() before move_camera().")

        camera_obj = self.camera
        camera_obj.location = location

        look_at_vec = Vector(look_at)

        if fit_target:
            render = bpy.context.scene.render
            self._fit_camera_to_target(
                camera_obj=camera_obj,
                requested_location=location,
                look_at=look_at_vec,
                fit_margin=fit_margin,
                resolution_x_in_px=render.resolution_x,
                resolution_y_in_px=render.resolution_y,
                pixel_aspect_x=render.pixel_aspect_x,
                pixel_aspect_y=render.pixel_aspect_y,
            )

        self._point_camera_at(camera_obj, look_at_vec)

        self._camera_requested_location = location
        self._camera_look_at = look_at
        self._camera_fit_target = fit_target
        self._camera_fit_margin = fit_margin

        bpy.context.scene.camera = camera_obj
        bpy.context.view_layer.update()
        return camera_obj

    def _point_camera_at(self, camera_obj: bpy.types.Object, look_at: Vector) -> None:
        """Orient the camera so its local -Z axis points at look_at."""
        direction_vec = look_at - Vector(camera_obj.location)
        direction = require_nonzero_direction(
            (direction_vec.x, direction_vec.y, direction_vec.z),
            error_message=_CAMERA_LOOK_AT_ERROR,
        )
        camera_obj.rotation_euler = rotation_euler_from_forward(direction)

    def _fit_camera_to_target(
        self,
        *,
        camera_obj: bpy.types.Object,
        requested_location: tuple[float, float, float],
        look_at: Vector,
        fit_margin: float,
        resolution_x_in_px: int,
        resolution_y_in_px: int,
        pixel_aspect_x: float,
        pixel_aspect_y: float,
    ) -> None:
        """Move the camera back along its view direction until the target fits."""
        if not self.targets:
            return

        bpy.context.view_layer.update()

        corners = self._target_world_corners()
        radius = max((corner - look_at).length for corner in corners)

        requested_direction = look_at - Vector(requested_location)
        require_nonzero_direction(
            (requested_direction.x, requested_direction.y, requested_direction.z),
            error_message=_CAMERA_LOOK_AT_ERROR,
        )

        fov = limiting_fov_from_camera(
            cam=camera_obj.data,
            resolution_x_in_px=resolution_x_in_px,
            resolution_y_in_px=resolution_y_in_px,
            pixel_aspect_x=pixel_aspect_x,
            pixel_aspect_y=pixel_aspect_y,
        )
        min_distance = fit_distance(
            target_radius=radius,
            fov_rad=fov,
            margin=fit_margin,
        )

        distance = max(requested_direction.length, min_distance)
        camera_obj.location = look_at - requested_direction.normalized() * distance

    def _refit_camera_for_current_render_settings(self) -> None:
        """Re-apply target fitting after render settings such as resolution change."""
        if (
            not self._camera_fit_target
            or self.camera is None
            or self._camera_requested_location is None
            or self._camera_look_at is None
        ):
            return

        render = bpy.context.scene.render
        look_at = Vector(self._camera_look_at)
        self._fit_camera_to_target(
            camera_obj=self.camera,
            requested_location=self._camera_requested_location,
            look_at=look_at,
            fit_margin=self._camera_fit_margin,
            resolution_x_in_px=render.resolution_x,
            resolution_y_in_px=render.resolution_y,
            pixel_aspect_x=render.pixel_aspect_x,
            pixel_aspect_y=render.pixel_aspect_y,
        )
        self._point_camera_at(self.camera, look_at)

    def add_light(
        self,
        *,
        light_type: Literal["POINT", "SUN", "AREA"] = "POINT",
        location: tuple[float, float, float] = (5.0, 5.0, 5.0),
        look_at: tuple[float, float, float] = (1.0, 1.0, 1.0),
        energy: float | None = None,
        color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        size: float = 1.0,
    ) -> bpy.types.Object:
        """
        Args:
            light_type: One of "POINT" (omnidirectional), "SUN" (parallel
                directional rays, location-independent), or "AREA"
                (rectangular soft light, good for studio-style renders).
            location: World-space position of the light. For SUN, only the
                direction from `location` to `look_at` matters; the actual
                position is irrelevant for shading.
            look_at: World-space point the light aims at. Used by SUN and
                AREA. Ignored for POINT (omnidirectional).
            energy: Light intensity. Units depend on type:
                POINT and AREA in Watts, SUN in unitless strength.
                If None, uses ``DEFAULT_LIGHT_ENERGY`` from ``light_poses``.
            color: RGB in [0, 1]. White by default.
            size: For AREA lights, the side length in meters. Ignored
                for other types.

        Returns:
            The created light object.
        """
        if energy is None:
            energy = DEFAULT_LIGHT_ENERGY[light_type]

        name = f"Light_{light_type.capitalize()}"
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        light_data.color = color

        if light_type == "AREA":
            light_data.size = size

        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location

        if light_type != "POINT":
            direction_vec = Vector(look_at) - Vector(location)
            direction = require_nonzero_direction(
                (direction_vec.x, direction_vec.y, direction_vec.z),
                error_message=_LIGHT_LOOK_AT_ERROR,
            )
            light_obj.rotation_euler = rotation_euler_from_forward(direction)

        self.lights.append(light_obj)
        bpy.context.view_layer.update()
        return light_obj

    def render(
        self,
        output_path: str | Path,
        *,
        resolution: tuple[int, int] = (256, 256),
        engine: Literal["EEVEE", "CYCLES"] = "EEVEE",
        samples: int = 32,
        transparent_film: bool = False,
    ) -> Path:
        """Renders the current scene to a PNG file.

        Args:
            output_path: Where to write rendered PNG.
            resolution: (width, height) in pixels.
            engine:
                - EEVEE for fast rasterization (good for high-volume data)
                - CYCLES for path-traced realism (slower)
            samples: Render samples. For EEVEE this is TAA samples;
                     for CYCLES, path samples per pixel.
                     Higher = less noise, slower.
            transparent_film: When True, write RGBA with a transparent world
                background for post-render compositing over photo backgrounds.

        Returns:
            The output path as a Path object.

        Raises:
            RuntimeError: If no camera has been added to the scene.
            RenderEngineUnavailableError: If EEVEE cannot acquire a GPU context.
        """
        if self.camera is None:
            raise RuntimeError("No camera in the scene. Call add_camera() before render().")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        engine_id = {
            "EEVEE": "BLENDER_EEVEE_NEXT",
            "CYCLES": "CYCLES",
        }[engine]

        bpy_scene = bpy.context.scene
        bpy_scene.camera = self.camera

        bpy_scene.render.engine = engine_id
        bpy_scene.render.resolution_x = resolution[0]
        bpy_scene.render.resolution_y = resolution[1]
        bpy_scene.render.resolution_percentage = 100
        bpy_scene.render.image_settings.file_format = "PNG"
        if transparent_film:
            bpy_scene.render.film_transparent = True
            bpy_scene.render.image_settings.color_mode = "RGBA"
        else:
            bpy_scene.render.film_transparent = False
            bpy_scene.render.image_settings.color_mode = "RGB"

        if engine == "EEVEE":
            bpy_scene.eevee.taa_render_samples = samples
        else:
            bpy_scene.cycles.samples = samples

        self._refit_camera_for_current_render_settings()

        try:
            bpy.ops.render.render()
        except RuntimeError as exc:
            if engine == "EEVEE" and eevee_failure_is_gpu_context(exc):
                raise RenderEngineUnavailableError() from exc
            raise
        bpy.data.images["Render Result"].save_render(filepath=str(output))

        return output

    def _target_world_corners(self) -> list[Vector]:
        """Return world-space bound-box corners for every imported target mesh."""
        corners: list[Vector] = []
        for target in self.targets:
            corners.extend(target.matrix_world @ Vector(corner) for corner in target.bound_box)
        return corners

    def center_target(self) -> None:
        """Translate the target so its bounding-box center is at (0, 0, 0).

        .obj files don't guarantee where the geometry sits relative to
        the object origin — exporters often put the origin at floor
        level, one corner, or somewhere arbitrary. This normalizes the
        target so camera and light placement relative to the world
        origin actually frames the model.

        Raises:
            RuntimeError: If no target has been loaded.
        """
        if not self.targets:
            raise RuntimeError("No target loaded. Call load_object() first.")

        world_corners = self._target_world_corners()
        min_corner = Vector(
            (
                min(corner.x for corner in world_corners),
                min(corner.y for corner in world_corners),
                min(corner.z for corner in world_corners),
            )
        )
        max_corner = Vector(
            (
                max(corner.x for corner in world_corners),
                max(corner.y for corner in world_corners),
                max(corner.z for corner in world_corners),
            )
        )
        center = (min_corner + max_corner) / 2

        for target in self.targets:
            target.location -= center
        bpy.context.view_layer.update()
