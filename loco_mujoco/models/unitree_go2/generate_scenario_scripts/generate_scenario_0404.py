from __future__ import annotations

import math
import random
from pathlib import Path


UNITREE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = UNITREE_ROOT / "scenarios_0404"
RNG_SEED = 404

OBJECT_TYPES = [
    "chair",
    "couch",
    "bookshelf",
    "table",
    "cardboard_box",
    "trash_can",
]

TIERS = {
    "cardboard_box": 1,
    "trash_can": 1,
    "chair": 2,
    "couch": 2,
    "table": 2,
    "bookshelf": 2,
}

OBJECT_INCLUDE = {
    "chair": "chair.xml",
    "couch": "couch.xml",
    "bookshelf": "bookshelf.xml",
    "table": "table.xml",
    "cardboard_box": "cardboard_box.xml",
    "trash_can": "trash_can.xml",
}

OBJECT_VARIATION = {
    "chair": {"x_shift": -0.15, "y_shift": 0.10, "yaw_shift": 0.18, "mirror": 1},
    "couch": {"x_shift": 0.20, "y_shift": -0.12, "yaw_shift": -0.22, "mirror": -1},
    "bookshelf": {"x_shift": 0.35, "y_shift": 0.16, "yaw_shift": 0.30, "mirror": 1},
    "table": {"x_shift": -0.05, "y_shift": -0.18, "yaw_shift": -0.12, "mirror": -1},
    "cardboard_box": {"x_shift": -0.25, "y_shift": 0.22, "yaw_shift": 0.40, "mirror": 1},
    "trash_can": {"x_shift": 0.10, "y_shift": -0.24, "yaw_shift": -0.35, "mirror": -1},
}

PROFILE_SET = [
    {
        "label": "near-center-clear",
        "distance": "near",
        "difficulty": "easy",
        "target_pos": (2.2, 0.0),
        "target_yaw": 0.0,
        "distractor_positions": [],
        "obstructor_pos": None,
    },
    {
        "label": "near-left-clear",
        "distance": "near",
        "difficulty": "easy",
        "target_pos": (2.6, 1.2),
        "target_yaw": 0.7854,
        "distractor_positions": [],
        "obstructor_pos": None,
    },
    {
        "label": "near-right-distractor",
        "distance": "near",
        "difficulty": "easy",
        "target_pos": (2.5, -1.1),
        "target_yaw": -0.5236,
        "distractor_positions": [(4.4, 1.6)],
        "obstructor_pos": None,
    },
    {
        "label": "medium-center-distractor",
        "distance": "medium",
        "difficulty": "moderate",
        "target_pos": (4.4, 0.0),
        "target_yaw": 1.5708,
        "distractor_positions": [(2.8, -1.5)],
        "obstructor_pos": None,
    },
    {
        "label": "medium-left-two-distractors",
        "distance": "medium",
        "difficulty": "moderate",
        "target_pos": (4.6, 1.5),
        "target_yaw": 0.5236,
        "distractor_positions": [(2.4, -1.2), (6.2, -0.6)],
        "obstructor_pos": None,
    },
    {
        "label": "medium-right-obstructed",
        "distance": "medium",
        "difficulty": "hard",
        "target_pos": (4.8, -1.4),
        "target_yaw": -0.7854,
        "distractor_positions": [(5.8, 1.3)],
        "obstructor_pos": (2.6, -0.8),
    },
    {
        "label": "far-center-obstructed",
        "distance": "far",
        "difficulty": "hard",
        "target_pos": (6.0, 0.3),
        "target_yaw": 1.0472,
        "distractor_positions": [(4.2, -1.9)],
        "obstructor_pos": (3.1, 0.1),
    },
    {
        "label": "far-left-mixed",
        "distance": "far",
        "difficulty": "hard",
        "target_pos": (6.3, 1.8),
        "target_yaw": 2.3562,
        "distractor_positions": [(2.4, -1.3), (5.3, -0.9)],
        "obstructor_pos": (3.6, 0.9),
    },
    {
        "label": "far-right-mixed",
        "distance": "far",
        "difficulty": "hard",
        "target_pos": (6.2, -1.7),
        "target_yaw": -1.0472,
        "distractor_positions": [(2.2, 1.0), (5.4, 1.5)],
        "obstructor_pos": (3.4, -0.9),
    },
    {
        "label": "far-left-offset-mixed",
        "distance": "far",
        "difficulty": "hard",
        "target_pos": (7.0, 1.0),
        "target_yaw": -1.5708,
        "distractor_positions": [(2.6, -1.5), (6.0, -0.8)],
        "obstructor_pos": (3.8, 0.6),
    },
]

SCENE_TEMPLATE = """<mujoco model="scene">

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.871 0.616 0.4 0.5"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>

  <asset>
    <texture builtin="gradient" height="100" rgb1="0.9 0.9 0.9" rgb2="0.55 0.25 0.0" type="skybox" width="100"/>
    <include file="../../room/objects/materials.xml"/>
  </asset>

    <worldbody>
        <geom name="floor" friction="1 .1 .1" pos="0 0 0" size="0 0 0.125" type="plane" material="mat_floor" condim="3" conaffinity="1" contype="1" group="2"/>
        <light cutoff="1000" diffuse="1.5 1.5 1.5" dir="-0 0 -1.3" directional="true" exponent="10" pos="0 0 10.3" specular=".1 .1 .1" castshadow="false"/>

        <geom name="wall_negx" type="box" pos="-15.0 0.0 1.25" size="0.05 15.0 1.25" material="mat_wall"/>
        <geom name="wall_posx" type="box" pos=" 15.0 0.0 1.25" size="0.05 15.0 1.25" material="mat_wall"/>
        <geom name="wall_negy" type="box" pos="0.0 -15.0 1.25" size="15.0 0.05 1.25" material="mat_wall"/>
        <geom name="wall_posy" type="box" pos="0.0  15.0 1.25" size="15.0 0.05 1.25" material="mat_wall"/>

        <geom name="ceiling" type="box"
          pos="0 0 2.55"
          size="15.0 15.0 0.05"
          material="mat_wall"
          contype="1" conaffinity="1"
        />

{body_block}
    </worldbody>
</mujoco>
"""


def fov_ok(pos: tuple[float, float], margin: float = 0.2) -> bool:
    x, y = pos
    return x > 0.0 and abs(y) <= x - margin


def projection_fraction(point: tuple[float, float], target: tuple[float, float]) -> float:
    px, py = point
    tx, ty = target
    target_norm_sq = tx * tx + ty * ty
    return (px * tx + py * ty) / target_norm_sq


def point_line_distance(point: tuple[float, float], target: tuple[float, float]) -> float:
    px, py = point
    tx, ty = target
    numer = abs(ty * px - tx * py)
    denom = math.hypot(tx, ty)
    return numer / denom


def lower_tier_choices(target_type: str) -> list[str]:
    target_tier = TIERS[target_type]
    return [obj for obj in OBJECT_TYPES if TIERS[obj] < target_tier]


def distractor_choices(target_type: str, obstructor_type: str | None, offset: int) -> list[str]:
    candidates = [obj for obj in OBJECT_TYPES if obj not in {target_type, obstructor_type}]
    rotated = candidates[offset:] + candidates[:offset]
    return rotated


def render_body(
    body_name: str,
    object_type: str,
    pos: tuple[float, float],
    yaw: float,
    comment: str,
) -> str:
    lines = [
        f"        <!-- {comment}: {object_type} -->",
        f'        <body name="{body_name}" pos="{pos[0]:.4f} {pos[1]:.4f} 0.0" euler="0 0 {yaw:.4f}">',
    ]
    lines.append(f'          <include file="../../room/objects/{OBJECT_INCLUDE[object_type]}"/>')
    lines.append("        </body>")
    return "\n".join(lines)


def render_scene(spec: dict[str, object]) -> str:
    target_type = spec["target_type"]
    target_pos = spec["target_pos"]
    target_yaw = spec["target_yaw"]
    target_side = spec["target_side"]
    distance = spec["distance"]
    difficulty = spec["difficulty"]
    scenario_name = spec["scenario_name"]

    header = [
        f"        <!-- SCENARIO: {scenario_name} -->",
        f"        <!-- TARGET_BODY_NAME: {target_type}_target -->",
        f"        <!-- LAYOUT: target={target_type}, distance={distance}, side={target_side}, difficulty={difficulty} -->",
        "",
        render_body(
            body_name=f"{target_type}_target",
            object_type=target_type,
            pos=target_pos,
            yaw=target_yaw,
            comment="TARGET",
        ),
        "",
    ]

    obstructor = spec["obstructor"]
    if obstructor is None:
        header.append("        <!-- OBSTRUCTOR: none -->")
    else:
        header.append(
            render_body(
                body_name=f"{obstructor['type']}_obstructor1",
                object_type=obstructor["type"],
                pos=obstructor["pos"],
                yaw=obstructor["yaw"],
                comment="OBSTRUCTOR",
            )
        )
    header.append("")

    distractors = spec["distractors"]
    if not distractors:
        header.append("        <!-- DISTRACTORS: none -->")
    else:
        for index, distractor in enumerate(distractors, start=1):
            header.append(
                render_body(
                    body_name=f"{distractor['type']}_distractor{index}",
                    object_type=distractor["type"],
                    pos=distractor["pos"],
                    yaw=distractor["yaw"],
                    comment="DISTRACTOR",
                )
            )
    return SCENE_TEMPLATE.format(body_block="\n".join(header))


def render_markdown(spec: dict[str, object]) -> str:
    target_pos = spec["target_pos"]
    distractors = spec["distractors"]
    obstructor = spec["obstructor"]
    yaw_deg = math.degrees(spec["target_yaw"])

    lines = [
        f"# {spec['scenario_name']}",
        "",
        f"- Target type: `{spec['target_type']}`",
        f"- Target body name: `{spec['target_type']}_target`",
        f"- Tier: `{TIERS[spec['target_type']]}`",
        f"- Distance bucket: `{spec['distance']}`",
        f"- Side bucket: `{spec['target_side']}`",
        f"- Difficulty: `{spec['difficulty']}`",
        f"- Target pose: `x={target_pos[0]:.2f}, y={target_pos[1]:.2f}, yaw={yaw_deg:.1f} deg`",
        "",
        "## Layout",
    ]

    if obstructor is None:
        lines.append("- Obstructor: none")
    else:
        lines.append(
            "- Obstructor: "
            f"`{obstructor['type']}` at `x={obstructor['pos'][0]:.2f}, y={obstructor['pos'][1]:.2f}`"
        )

    if distractors:
        for distractor in distractors:
            lines.append(
                "- Distractor: "
                f"`{distractor['type']}` at `x={distractor['pos'][0]:.2f}, y={distractor['pos'][1]:.2f}`"
            )
    else:
        lines.append("- Distractors: none")

    lines.extend(
        [
            "",
            "## Notes",
            "- All placed objects remain inside the initial egocentric +/-45 degree horizontal field of view.",
            "- Distractors are positioned away from the straight-line path from the robot start to the target.",
        ]
    )

    if obstructor is None:
        lines.append(
            "- No lower-tier obstructor was used in this scene."
            if TIERS[spec["target_type"]] == 1
            else "- This scene is intentionally unobstructed."
        )
    else:
        lines.append(
            "- The obstructor is lower tier than the target and sits on the nominal path while leaving turning room around it."
        )

    return "\n".join(lines) + "\n"


def side_label(pos: tuple[float, float]) -> str:
    _, y = pos
    if y > 0.4:
        return "left"
    if y < -0.4:
        return "right"
    return "center"


def wrap_yaw(yaw: float) -> float:
    while yaw <= -math.pi:
        yaw += 2 * math.pi
    while yaw > math.pi:
        yaw -= 2 * math.pi
    return yaw


def mirrored_profile(
    profile: dict[str, object],
    rng: random.Random,
) -> dict[str, object]:
    target_x, target_y = profile["target_pos"]
    if abs(target_y) <= 0.4:
        direction = 0
    else:
        direction = rng.choice([-1, 1])

    def mirror_pos(pos: tuple[float, float]) -> tuple[float, float]:
        x, y = pos
        return (x, y if direction == 0 else abs(y) * direction)

    mirrored = dict(profile)
    mirrored["target_pos"] = mirror_pos(profile["target_pos"])
    mirrored["target_yaw"] = profile["target_yaw"] if direction == 0 else -profile["target_yaw"]
    mirrored["distractor_positions"] = [mirror_pos(pos) for pos in profile["distractor_positions"]]
    mirrored["obstructor_pos"] = None if profile["obstructor_pos"] is None else mirror_pos(profile["obstructor_pos"])
    return mirrored


def randomized_variation(
    variation: dict[str, float | int],
    rng: random.Random,
) -> dict[str, float | int]:
    return {
        "x_shift": float(variation["x_shift"]) + rng.uniform(-0.12, 0.12),
        "y_shift": float(variation["y_shift"]) + rng.uniform(-0.12, 0.12),
        "yaw_shift": float(variation["yaw_shift"]) + rng.uniform(-0.10, 0.10),
        "mirror": int(variation["mirror"]) if rng.random() < 0.7 else -int(variation["mirror"]),
    }


def vary_position(
    pos: tuple[float, float],
    variation: dict[str, float | int],
    index: int,
    role: str,
) -> tuple[float, float]:
    x, y = pos
    mirror = int(variation["mirror"])

    if role == "target":
        x += float(variation["x_shift"])
        y += float(variation["y_shift"])
    elif role == "obstructor":
        x += 0.35 * float(variation["x_shift"])
        y += mirror * 0.18 + 0.25 * float(variation["y_shift"])
    else:
        x += 0.45 * float(variation["x_shift"]) + 0.06 * ((index % 3) - 1)
        y = y * mirror + 0.45 * float(variation["y_shift"]) + 0.08 * ((index % 2) * 2 - 1)

    if x < 1.9:
        x = 1.9
    max_abs_y = max(0.0, x - 0.32)
    y = max(-max_abs_y, min(max_abs_y, y))
    return (round(x, 4), round(y, 4))


def varied_yaw(base_yaw: float, variation: dict[str, float | int], index: int, role: str, slot: int = 0) -> float:
    role_scale = {"target": 1.0, "obstructor": 0.6, "distractor": 0.8}[role]
    return round(wrap_yaw(base_yaw + role_scale * float(variation["yaw_shift"]) + 0.09 * ((index % 3) - 1) + 0.08 * slot), 4)


def nudge_into_fov(pos: tuple[float, float]) -> tuple[float, float]:
    x, y = pos
    x = max(1.9, x)
    limit = max(0.0, x - 0.32)
    y = max(-limit, min(limit, y))
    return (round(x, 4), round(y, 4))


def adjust_distractor_position(
    pos: tuple[float, float],
    target_pos: tuple[float, float],
    variation: dict[str, float | int],
    index: int,
) -> tuple[float, float]:
    x, y = pos
    for _ in range(4):
        frac = projection_fraction((x, y), target_pos)
        dist = point_line_distance((x, y), target_pos)
        if not (0.15 <= frac <= 0.92 and dist < 0.9):
            break
        side = 1.0 if y >= 0 else -1.0
        if abs(y) < 0.2:
            side = float(variation["mirror"])
        y += side * (1.05 - dist)
        x += 0.18 if x < target_pos[0] else -0.12
        x, y = nudge_into_fov((x, y))

    return nudge_into_fov((x, y))


def adjust_obstructor_position(pos: tuple[float, float], target_pos: tuple[float, float]) -> tuple[float, float]:
    x, y = pos
    frac = projection_fraction((x, y), target_pos)
    if frac < 0.25:
        x += 0.4
    elif frac > 0.75:
        x -= 0.4

    dist = point_line_distance((x, y), target_pos)
    if dist > 0.35:
        y = round(y * 0.6 + target_pos[1] * 0.4, 4)

    if target_pos[0] - x < 1.4:
        x = target_pos[0] - 1.4
    if x < 2.2:
        x = 2.2
    return nudge_into_fov((x, y))


def separate_from_anchors(
    pos: tuple[float, float],
    anchors: list[tuple[float, float]],
    variation: dict[str, float | int],
    target_pos: tuple[float, float],
) -> tuple[float, float]:
    x, y = pos
    for anchor_x, anchor_y in anchors:
        dist = math.dist((x, y), (anchor_x, anchor_y))
        if dist < 1.0:
            if dist < 1e-6:
                dx, dy = 0.0, float(variation["mirror"])
            else:
                dx, dy = x - anchor_x, y - anchor_y
            norm = math.hypot(dx, dy)
            dx /= norm
            dy /= norm
            push = 1.02 - dist
            x += dx * push
            y += dy * push
    return adjust_distractor_position(nudge_into_fov((x, y)), target_pos, variation, 0)


def build_spec(target_type: str, index: int, profile: dict[str, object]) -> dict[str, object]:
    rng = random.Random(f"{RNG_SEED}:{target_type}:{index}")
    profile = mirrored_profile(profile, rng)
    variation = randomized_variation(OBJECT_VARIATION[target_type], rng)
    target_pos = vary_position(profile["target_pos"], variation, index, "target")
    target_yaw = varied_yaw(profile["target_yaw"], variation, index, "target")
    obstructor = None
    lower_tier = lower_tier_choices(target_type)
    if profile["obstructor_pos"] is not None and lower_tier:
        obstructor_type = lower_tier[index % len(lower_tier)]
        obstructor = {
            "type": obstructor_type,
            "pos": adjust_obstructor_position(vary_position(profile["obstructor_pos"], variation, index, "obstructor"), target_pos),
            "yaw": varied_yaw((index % 4) * 0.5236, variation, index, "obstructor"),
        }
    else:
        obstructor_type = None

    distractor_types = distractor_choices(target_type, obstructor_type, index % 4)
    distractors = []
    for slot, pos in enumerate(profile["distractor_positions"]):
        distractor_pos = adjust_distractor_position(
            vary_position(pos, variation, index + slot, "distractor"),
            target_pos,
            variation,
            index + slot,
        )
        anchors = [target_pos]
        if obstructor is not None:
            anchors.append(obstructor["pos"])
        anchors.extend(d["pos"] for d in distractors)
        distractors.append(
            {
                "type": distractor_types[slot],
                "pos": separate_from_anchors(distractor_pos, anchors, variation, target_pos),
                "yaw": varied_yaw(((index + slot + 1) % 6 - 2) * 0.5236, variation, index, "distractor", slot),
            }
        )

    scenario_name = f"{target_type}_{index + 1:02d}"
    spec = {
        "scenario_name": scenario_name,
        "target_type": target_type,
        "target_pos": target_pos,
        "target_yaw": target_yaw,
        "target_side": side_label(target_pos),
        "distance": profile["distance"],
        "difficulty": profile["difficulty"],
        "obstructor": obstructor,
        "distractors": distractors,
    }
    validate_spec(spec)
    return spec


def validate_spec(spec: dict[str, object]) -> None:
    target_pos = spec["target_pos"]
    assert fov_ok(target_pos), f"target outside fov: {spec['scenario_name']}"

    distractor_types = set()
    all_positions = [target_pos]

    for distractor in spec["distractors"]:
        assert fov_ok(distractor["pos"]), f"distractor outside fov: {spec['scenario_name']}"
        frac = projection_fraction(distractor["pos"], target_pos)
        dist = point_line_distance(distractor["pos"], target_pos)
        if 0.15 <= frac <= 0.92:
            assert dist >= 0.9, f"distractor too close to path: {spec['scenario_name']}"
        distractor_types.add(distractor["type"])
        all_positions.append(distractor["pos"])

    obstructor = spec["obstructor"]
    if obstructor is not None:
        assert TIERS[obstructor["type"]] < TIERS[spec["target_type"]], spec["scenario_name"]
        assert fov_ok(obstructor["pos"]), f"obstructor outside fov: {spec['scenario_name']}"
        frac = projection_fraction(obstructor["pos"], target_pos)
        dist = point_line_distance(obstructor["pos"], target_pos)
        assert 0.25 <= frac <= 0.75, f"obstructor not between robot and target: {spec['scenario_name']}"
        assert dist <= 0.35, f"obstructor not on path: {spec['scenario_name']}"
        assert target_pos[0] - obstructor["pos"][0] >= 1.4, f"target too close to obstructor: {spec['scenario_name']}"
        assert obstructor["pos"][0] >= 2.2, f"obstructor too close to robot: {spec['scenario_name']}"
        all_positions.append(obstructor["pos"])

    assert len(distractor_types) == len(spec["distractors"]), f"duplicate distractor type: {spec['scenario_name']}"

    for i, first in enumerate(all_positions):
        for second in all_positions[i + 1 :]:
            assert math.dist(first, second) >= 1.0, f"objects too close: {spec['scenario_name']}"


def render_index(all_specs: list[dict[str, object]]) -> str:
    lines = [
        "# scenario-0404",
        "",
        "Generated scenario set for the Unitree Go2 room scenes.",
        "",
        "- Total scenarios: `60`",
        "- Targets per object type: `10`",
        "- Target body names follow the pattern `<object_type>_target`.",
        "- Tier-1 targets (`cardboard_box`, `trash_can`) do not include obstructors because there is no lower tier available.",
        "",
        "| Scenario | Target | Tier | Distance | Side | Difficulty | Obstructor | Distractors |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for spec in all_specs:
        obstructor = spec["obstructor"]["type"] if spec["obstructor"] is not None else "none"
        distractors = ", ".join(d["type"] for d in spec["distractors"]) or "none"
        lines.append(
            f"| `{spec['scenario_name']}` | `{spec['target_type']}` | `{TIERS[spec['target_type']]}` | "
            f"`{spec['distance']}` | `{spec['target_side']}` | `{spec['difficulty']}` | "
            f"`{obstructor}` | `{distractors}` |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_specs = []

    for target_type in OBJECT_TYPES:
        for index, profile in enumerate(PROFILE_SET):
            spec = build_spec(target_type, index, profile)
            all_specs.append(spec)
            scenario_dir = OUTPUT_ROOT / spec["scenario_name"]
            scenario_dir.mkdir(parents=True, exist_ok=True)
            (scenario_dir / "scene_room.xml").write_text(render_scene(spec), encoding="utf-8")
            (scenario_dir / "scenario.md").write_text(render_markdown(spec), encoding="utf-8")

    (OUTPUT_ROOT / "README.md").write_text(render_index(all_specs), encoding="utf-8")


if __name__ == "__main__":
    main()
