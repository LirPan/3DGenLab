from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path

SUPPORTED_MESH_EXTENSIONS = {".obj", ".ply", ".glb", ".gltf", ".stl"}
DEFAULT_OCIO_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "ocio" / "minimal_config.ocio"


def _validate_mesh_path(mesh_path: str | Path) -> Path:
    mesh_file = Path(mesh_path).resolve()
    if not mesh_file.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
    if mesh_file.suffix.lower() not in SUPPORTED_MESH_EXTENSIONS:
        raise ValueError(
            f"Unsupported mesh format '{mesh_file.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MESH_EXTENSIONS))}"
        )
    return mesh_file


def _import_snippet_for_extension(ext: str) -> str:
    ext_lower = ext.lower()
    if ext_lower == ".obj":
        return "bpy.ops.import_scene.obj(filepath=mesh_path)"
    if ext_lower == ".ply":
        return "bpy.ops.import_mesh.ply(filepath=mesh_path)"
    if ext_lower in {".glb", ".gltf"}:
        return "bpy.ops.import_scene.gltf(filepath=mesh_path)"
    if ext_lower == ".stl":
        return "bpy.ops.import_mesh.stl(filepath=mesh_path)"
    raise ValueError(f"Unsupported mesh extension for Blender import: {ext}")


def _build_blender_script(import_snippet: str) -> str:
    return f"""import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv
if "--" not in argv:
    raise SystemExit("Expected '--' args")
args = argv[argv.index("--") + 1 :]
if len(args) != 7:
    raise SystemExit("Args: <mesh_path> <output_path> <width> <height> <samples> <engine> <background>")

mesh_path, output_path, width, height, samples, engine, background = args
width = int(width)
height = int(height)
samples = int(samples)
background = str(background).lower() in ("1", "true", "yes")

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

{import_snippet}

mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not mesh_objs:
    raise SystemExit("No mesh objects imported")

# Normalize mesh scale and center for stable rendering.
min_corner = Vector((1e9, 1e9, 1e9))
max_corner = Vector((-1e9, -1e9, -1e9))
for obj in mesh_objs:
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        min_corner.x = min(min_corner.x, world_corner.x)
        min_corner.y = min(min_corner.y, world_corner.y)
        min_corner.z = min(min_corner.z, world_corner.z)
        max_corner.x = max(max_corner.x, world_corner.x)
        max_corner.y = max(max_corner.y, world_corner.y)
        max_corner.z = max(max_corner.z, world_corner.z)

center = (min_corner + max_corner) * 0.5
extent = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)
scale = 1.0 if extent <= 1e-6 else 1.4 / extent

for obj in mesh_objs:
    obj.location = (obj.location - center) * scale
    obj.scale = obj.scale * scale
    obj.select_set(True)

bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
bpy.ops.object.shade_smooth()

# Assign a neutral clay material for visibility.
mat = bpy.data.materials.new(name="GenLabClay")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.90, 0.92, 0.95, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.45
    bsdf.inputs["Specular"].default_value = 0.40
    bsdf.inputs["Emission"].default_value = (0.90, 0.92, 0.95, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 40000.0
for obj in mesh_objs:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

bpy.ops.object.camera_add(location=(0.0, -3.2, 1.6), rotation=(1.22, 0.0, 0.0))
camera = bpy.context.active_object
bpy.context.scene.camera = camera
focus_target = mesh_objs[0]
track = camera.constraints.new(type="TRACK_TO")
track.target = focus_target
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"
camera.data.lens = 55.0

bpy.ops.object.light_add(type="SUN", location=(3.0, -3.0, 4.5))
key_light = bpy.context.active_object
key_light.data.energy = 6.0

bpy.ops.object.light_add(type="AREA", location=(-1.5, 1.0, 2.0))
fill_light = bpy.context.active_object
fill_light.data.energy = 350.0
fill_light.data.size = 4.0

# World light and color management.
world = bpy.context.scene.world
world.use_nodes = True
world_nodes = world.node_tree.nodes
world_links = world.node_tree.links
world_nodes.clear()
bg = world_nodes.new(type="ShaderNodeBackground")
world_output = world_nodes.new(type="ShaderNodeOutputWorld")
world_links.new(bg.outputs["Background"], world_output.inputs["Surface"])
bg.inputs[0].default_value = (0.85, 0.85, 0.87, 1.0)
bg.inputs[1].default_value = 1.0

scene = bpy.context.scene
scene.render.engine = engine
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.film_transparent = background
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "8"
scene.render.dither_intensity = 0.0
scene.render.use_compositing = False
scene.use_nodes = False
scene.render.filepath = output_path

def _set_enum_with_fallback(scene_obj, attr_name, prop_name, preferred_values):
    prop = scene_obj.bl_rna.properties.get(prop_name)
    if not prop:
        return
    values = [item.identifier for item in prop.enum_items]
    if not values:
        return
    lookup = {{str(v).lower(): v for v in values}}
    candidates = []
    for candidate in preferred_values:
        hit = lookup.get(str(candidate).lower())
        if hit is not None and hit not in candidates:
            candidates.append(hit)
    for value in values:
        if str(value).lower() == "none":
            continue
        if value not in candidates:
            candidates.append(value)
    for value in candidates:
        try:
            setattr(scene_obj, attr_name, value)
            return
        except Exception:
            pass

# Stabilize color-management in headless environments.
_set_enum_with_fallback(scene.display_settings, "display_device", "display_device", ("sRGB", "ACES", "Rec.709"))
_set_enum_with_fallback(scene.view_settings, "view_transform", "view_transform", ("Raw", "Standard", "Filmic", "AgX"))

look_prop = scene.view_settings.bl_rna.properties.get("look")
if look_prop and "None" in [item.identifier for item in look_prop.enum_items]:
    scene.view_settings.look = "None"

# Keep a conservative camera response for stable clay rendering.
scene.view_settings.exposure = 1.0
scene.view_settings.gamma = 1.0

if engine == "CYCLES":
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
bpy.ops.render.render(write_still=True)

render_result = bpy.data.images.get("Render Result")
result_pixels = []
if render_result is not None and len(render_result.pixels) > 0:
    result_pixels = list(render_result.pixels[:])
else:
    render_image = bpy.data.images.load(output_path, check_existing=False)
    result_pixels = list(render_image.pixels[:])
    bpy.data.images.remove(render_image)

max_rgb = 0.0
for i in range(0, len(result_pixels), 4):
    max_rgb = max(max_rgb, result_pixels[i], result_pixels[i + 1], result_pixels[i + 2])

# Some fallback color-management builds output near-black values.
if max_rgb < 0.05 and result_pixels:
    gain = 0.85 / max(max_rgb, 1e-6)
    gain = min(65535.0, gain)
    for i in range(0, len(result_pixels), 4):
        result_pixels[i] = min(1.0, result_pixels[i] * gain)
        result_pixels[i + 1] = min(1.0, result_pixels[i + 1] * gain)
        result_pixels[i + 2] = min(1.0, result_pixels[i + 2] * gain)
    out_image = bpy.data.images.new(
        name="GenLabNormalized",
        width=width,
        height=height,
        alpha=True,
        float_buffer=False,
    )
    out_image.pixels = result_pixels
    out_image.filepath_raw = output_path
    out_image.file_format = "PNG"
    out_image.save()
    bpy.data.images.remove(out_image)
"""


def _build_mask_fallback_script(import_snippet: str) -> str:
    return f"""import sys
from pathlib import Path

import bpy
from mathutils import Vector

argv = sys.argv
if "--" not in argv:
    raise SystemExit("Expected '--' args")
args = argv[argv.index("--") + 1 :]
if len(args) != 7:
    raise SystemExit("Args: <mesh_path> <output_path> <width> <height> <samples> <engine> <background>")

mesh_path, output_path, width, height, samples, engine, background = args
width = int(width)
height = int(height)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

{import_snippet}

mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
if not mesh_objs:
    raise SystemExit("No mesh objects imported")

min_corner = Vector((1e9, 1e9, 1e9))
max_corner = Vector((-1e9, -1e9, -1e9))
for obj in mesh_objs:
    for corner in obj.bound_box:
        world_corner = obj.matrix_world @ Vector(corner)
        min_corner.x = min(min_corner.x, world_corner.x)
        min_corner.y = min(min_corner.y, world_corner.y)
        min_corner.z = min(min_corner.z, world_corner.z)
        max_corner.x = max(max_corner.x, world_corner.x)
        max_corner.y = max(max_corner.y, world_corner.y)
        max_corner.z = max(max_corner.z, world_corner.z)

center = (min_corner + max_corner) * 0.5
extent = max(max_corner.x - min_corner.x, max_corner.y - min_corner.y, max_corner.z - min_corner.z)
scale = 1.0 if extent <= 1e-6 else 1.4 / extent
for obj in mesh_objs:
    obj.location = (obj.location - center) * scale
    obj.scale = obj.scale * scale
    obj.pass_index = 1
bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)

bpy.ops.object.camera_add(location=(0.0, -3.2, 1.6), rotation=(1.22, 0.0, 0.0))
camera = bpy.context.active_object
bpy.context.scene.camera = camera
track = camera.constraints.new(type="TRACK_TO")
track.target = mesh_objs[0]
track.track_axis = "TRACK_NEGATIVE_Z"
track.up_axis = "UP_Y"

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.use_sequencer = False
scene.render.use_compositing = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.render.image_settings.color_depth = "8"
scene.render.dither_intensity = 0.0
scene.render.filepath = output_path

view_layer = scene.view_layers[0]
view_layer.use_pass_object_index = True

scene.use_nodes = True
nt = scene.node_tree
nt.nodes.clear()
links = nt.links
rl = nt.nodes.new(type="CompositorNodeRLayers")
idm = nt.nodes.new(type="CompositorNodeIDMask")
idm.index = 1
mix = nt.nodes.new(type="CompositorNodeMixRGB")
comp = nt.nodes.new(type="CompositorNodeComposite")
links.new(rl.outputs["IndexOB"], idm.inputs["ID value"])
links.new(idm.outputs["Alpha"], mix.inputs["Fac"])
mix.inputs[1].default_value = (0.93, 0.93, 0.95, 1.0)
mix.inputs[2].default_value = (0.72, 0.74, 0.78, 1.0)
links.new(mix.outputs["Image"], comp.inputs["Image"])

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
bpy.ops.render.render(write_still=True)
"""


def _decode_png_rgb_stats(path: Path) -> dict[str, int]:
    data = path.read_bytes()
    if len(data) < 64 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Rendered output is not a valid PNG: {path}")

    pos = 8
    width = 0
    height = 0
    color_type = -1
    bit_depth = -1
    idat = bytearray()
    while pos + 12 <= len(data):
        chunk_len = int.from_bytes(data[pos : pos + 4], "big")
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data_start = pos + 8
        chunk_data_end = chunk_data_start + chunk_len
        if chunk_data_end + 4 > len(data):
            raise RuntimeError(f"Corrupted PNG chunk boundaries: {path}")
        chunk = data[chunk_data_start:chunk_data_end]
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _comp, _filter, _interlace = struct.unpack(">IIBBBBB", chunk)
        elif chunk_type == b"IDAT":
            idat.extend(chunk)
        elif chunk_type == b"IEND":
            break
        pos = chunk_data_end + 4

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Missing IHDR metadata in PNG: {path}")
    if bit_depth != 8:
        raise RuntimeError(f"Unsupported PNG bit depth ({bit_depth}) in {path}")
    channels_by_color_type = {2: 3, 6: 4}
    channels = channels_by_color_type.get(color_type)
    if channels is None:
        raise RuntimeError(f"Unsupported PNG color type ({color_type}) in {path}")

    raw = zlib.decompress(bytes(idat))
    bytes_per_pixel = channels
    row_stride = width * channels
    expected = height * (1 + row_stride)
    if len(raw) != expected:
        raise RuntimeError(f"Unexpected PNG data size for {path}: got={len(raw)} expected={expected}")

    def paeth(a: int, b: int, c: int) -> int:
        p = a + b - c
        pa = abs(p - a)
        pb = abs(p - b)
        pc = abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        if pb <= pc:
            return b
        return c

    rows: list[list[int]] = []
    prev = [0] * row_stride
    ptr = 0
    for _ in range(height):
        filter_type = raw[ptr]
        ptr += 1
        cur = list(raw[ptr : ptr + row_stride])
        ptr += row_stride

        if filter_type == 1:
            for i in range(row_stride):
                left = cur[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                cur[i] = (cur[i] + left) & 255
        elif filter_type == 2:
            for i in range(row_stride):
                cur[i] = (cur[i] + prev[i]) & 255
        elif filter_type == 3:
            for i in range(row_stride):
                left = cur[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = prev[i]
                cur[i] = (cur[i] + ((left + up) // 2)) & 255
        elif filter_type == 4:
            for i in range(row_stride):
                left = cur[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                up = prev[i]
                up_left = prev[i - bytes_per_pixel] if i >= bytes_per_pixel else 0
                cur[i] = (cur[i] + paeth(left, up, up_left)) & 255
        elif filter_type != 0:
            raise RuntimeError(f"Unsupported PNG filter type ({filter_type}) in {path}")

        rows.append(cur)
        prev = cur

    max_rgb = 0
    min_rgb = 255
    unique_rgb_limited: set[tuple[int, int, int]] = set()
    alpha_nonzero = 0
    for cur in rows:
        for i in range(0, row_stride, channels):
            r, g, b = cur[i], cur[i + 1], cur[i + 2]
            px_max = max(r, g, b)
            px_min = min(r, g, b)
            if px_max > max_rgb:
                max_rgb = px_max
            if px_min < min_rgb:
                min_rgb = px_min
            if len(unique_rgb_limited) < 4096:
                unique_rgb_limited.add((r, g, b))
            if channels == 4 and cur[i + 3] > 0:
                alpha_nonzero += 1

    return {
        "width": width,
        "height": height,
        "max_rgb": max_rgb,
        "min_rgb": min_rgb,
        "unique_rgb": len(unique_rgb_limited),
        "alpha_nonzero": alpha_nonzero,
    }


def _assert_render_png_quality(path: Path) -> None:
    stats = _decode_png_rgb_stats(path)
    if stats["max_rgb"] <= 4:
        raise RuntimeError(
            "Render image is near-black (possible color-management failure): "
            f"{path} max_rgb={stats['max_rgb']} unique_rgb={stats['unique_rgb']}"
        )
    if stats["unique_rgb"] < 2:
        raise RuntimeError(
            f"Render image has too little color variation: {path} unique_rgb={stats['unique_rgb']}"
        )


def _render_with_system_blender(
    mesh_file: Path,
    output_image: Path,
    *,
    blender_bin: str,
    width: int,
    height: int,
    samples: int,
    engine: str,
    transparent_background: bool,
) -> None:
    resolved_bin = shutil.which(blender_bin) or blender_bin
    if not shutil.which(resolved_bin) and not Path(resolved_bin).exists():
        raise FileNotFoundError(f"Blender executable not found: {blender_bin}")

    output_image.parent.mkdir(parents=True, exist_ok=True)
    import_snippet = _import_snippet_for_extension(mesh_file.suffix)
    script_text = _build_blender_script(import_snippet)

    def _invoke_blender(
        background_flag: bool, engine_name: str, script_override: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory(prefix="genlab_blender_") as tmp_dir:
            script_path = Path(tmp_dir) / "render_script.py"
            script_path.write_text(script_override or script_text, encoding="utf-8")

            cmd = [
                resolved_bin,
                "-b",
                "--python-exit-code",
                "1",
                "-P",
                str(script_path),
                "--",
                str(mesh_file),
                str(output_image),
                str(width),
                str(height),
                str(samples),
                engine_name,
                "true" if background_flag else "false",
            ]
            env = os.environ.copy()
            ocio_override = os.environ.get("GENLAB_OCIO_CONFIG", "").strip()
            if ocio_override:
                env["OCIO"] = ocio_override
            elif DEFAULT_OCIO_CONFIG.is_file():
                env["OCIO"] = str(DEFAULT_OCIO_CONFIG)
            return subprocess.run(cmd, capture_output=True, text=True, env=env)

    proc = _invoke_blender(transparent_background, engine)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(f"Blender render failed (exit={proc.returncode}). Tail:\n{tail}")

    if not output_image.is_file():
        logs = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
        raise RuntimeError(
            f"Render completed but output image missing: {output_image}\n"
            f"Blender tail:\n{logs}"
        )
    try:
        _assert_render_png_quality(output_image)
    except RuntimeError as exc:
        # In some headless OCIO setups, transparent output can collapse RGB
        # to near-black. Re-render once with opaque background as fallback.
        if transparent_background and "near-black" in str(exc):
            proc = _invoke_blender(False, engine)
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
                raise RuntimeError(f"Blender fallback render failed (exit={proc.returncode}). Tail:\n{tail}")
            try:
                _assert_render_png_quality(output_image)
                return
            except RuntimeError:
                # Last-resort fallback: use object-index compositor clay preview.
                mask_script = _build_mask_fallback_script(import_snippet)
                proc = _invoke_blender(False, "CYCLES", script_override=mask_script)
                if proc.returncode != 0:
                    tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
                    raise RuntimeError(
                        f"Blender mask fallback failed (exit={proc.returncode}). Tail:\n{tail}"
                    )
                _assert_render_png_quality(output_image)
                return
        raise


def _render_with_python_bpy(
    mesh_file: Path,
    output_image: Path,
    *,
    width: int,
    height: int,
    samples: int,
    engine: str,
    transparent_background: bool,
) -> None:
    # This path is best-effort only. Most deployments should use system Blender.
    import bpy  # type: ignore

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    ext = mesh_file.suffix.lower()
    if ext == ".obj":
        bpy.ops.import_scene.obj(filepath=str(mesh_file))
    elif ext == ".ply":
        bpy.ops.import_mesh.ply(filepath=str(mesh_file))
    elif ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=str(mesh_file))
    elif ext == ".stl":
        bpy.ops.import_mesh.stl(filepath=str(mesh_file))
    else:
        raise ValueError(f"Unsupported mesh extension for bpy import: {ext}")

    bpy.ops.object.camera_add(location=(2.2, -2.2, 1.6), rotation=(1.1, 0.0, 0.78))
    bpy.context.scene.camera = bpy.context.active_object

    bpy.ops.object.light_add(type="SUN", location=(3.0, -3.0, 4.0))
    bpy.context.active_object.data.energy = 3.0
    bpy.ops.object.light_add(type="AREA", location=(-1.5, 1.0, 2.0))
    bpy.context.active_object.data.energy = 300.0

    scene = bpy.context.scene
    scene.render.engine = engine
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = transparent_background
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output_image)
    if engine == "CYCLES":
        scene.cycles.samples = samples

    output_image.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.render.render(write_still=True)
    if not output_image.is_file():
        raise RuntimeError(f"bpy render completed but output image missing: {output_image}")
    try:
        _assert_render_png_quality(output_image)
    except RuntimeError as exc:
        if transparent_background and "near-black" in str(exc):
            scene.render.film_transparent = False
            bpy.ops.render.render(write_still=True)
            _assert_render_png_quality(output_image)
            return
        raise


def render_mesh_with_blender(
    mesh_path: str | Path,
    output_image_path: str | Path,
    *,
    mode: str = "system",
    blender_bin: str = "blender",
    width: int = 768,
    height: int = 768,
    samples: int = 32,
    engine: str = "CYCLES",
    transparent_background: bool = True,
) -> Path:
    mesh_file = _validate_mesh_path(mesh_path)
    output_image = Path(output_image_path).resolve()
    selected_mode = str(mode).strip().lower()
    engine_name = str(engine).strip().upper()

    if selected_mode not in {"system", "python"}:
        raise ValueError("mode must be one of: system, python")
    if engine_name not in {"CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"}:
        raise ValueError("engine must be one of: CYCLES, BLENDER_EEVEE, BLENDER_WORKBENCH")

    if selected_mode == "python":
        try:
            _render_with_python_bpy(
                mesh_file,
                output_image,
                width=width,
                height=height,
                samples=samples,
                engine=engine_name,
                transparent_background=transparent_background,
            )
            return output_image
        except Exception:
            _render_with_system_blender(
                mesh_file,
                output_image,
                blender_bin=blender_bin,
                width=width,
                height=height,
                samples=samples,
                engine=engine_name,
                transparent_background=transparent_background,
            )
            return output_image

    _render_with_system_blender(
        mesh_file,
        output_image,
        blender_bin=blender_bin,
        width=width,
        height=height,
        samples=samples,
        engine=engine_name,
        transparent_background=transparent_background,
    )
    return output_image
