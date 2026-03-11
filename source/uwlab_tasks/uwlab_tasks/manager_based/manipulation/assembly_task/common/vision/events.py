# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Visual / lighting domain randomization events for assembly tasks."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdLux

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from ..mdp import utils

# ---------------------------------------------------------------------------
# Visual / lighting domain randomization for RGB data collection
# ---------------------------------------------------------------------------


class randomize_hdri(ManagerTermBase):
    """Randomizes the HDRI texture, intensity, and rotation of a DomeLight."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        hdri_config_path = cfg.params.get("hdri_config_path")
        if hdri_config_path is not None:
            self.hdri_paths = utils.load_asset_paths_from_config(
                hdri_config_path, cache_subdir="hdris", skip_validation=False
            )
            logging.info(f"[randomize_hdri] Loaded {len(self.hdri_paths)} HDRI paths.")
        else:
            self.hdri_paths = []

        if not self.hdri_paths:
            raise RuntimeError(
                f"[randomize_hdri] No HDRI paths loaded. Check hdri_config_path={hdri_config_path}"
            )
        self(env, torch.arange(env.num_envs, device=env.device), **cfg.params)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        light_path: str = "/World/skyLight",
        hdri_config_path: str | None = None,
        intensity_range: tuple = (500.0, 1000.0),
        rotation_range: tuple = (0.0, 360.0),
    ) -> None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            raise RuntimeError(
                f"[randomize_hdri] Light prim at '{light_path}' does not exist on the stage."
            )
        dome_light = UsdLux.DomeLight(light_prim)
        if not dome_light:
            raise RuntimeError(f"[randomize_hdri] Prim at '{light_path}' is not a DomeLight.")

        from pxr import Sdf

        random_hdri = random.choice(self.hdri_paths)
        intensity = random.randint(int(intensity_range[0]), int(intensity_range[1]))
        light_prim.GetAttribute("inputs:texture:file").Set(Sdf.AssetPath(random_hdri))
        light_prim.GetAttribute("inputs:intensity").Set(float(intensity))

        from scipy.spatial.transform import Rotation as R
        quat = R.random().as_quat()  # [x, y, z, w] scipy convention
        xformable = UsdGeom.Xformable(light_prim)
        xformable.ClearXformOpOrder()
        xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(float(quat[3]), Gf.Vec3d(float(quat[0]), float(quat[1]), float(quat[2])))
        )
        logging.debug(f"[randomize_hdri] Applied: {random_hdri}, intensity={intensity}")


def randomize_tiled_cameras(
    env,
    env_ids: torch.Tensor,
    camera_path_template: str,
    base_position: tuple,
    base_rotation: tuple,
    position_deltas: dict,
    euler_deltas: dict,
) -> None:
    """Randomizes tiled cameras with XYZ and Euler angle deltas from base values."""
    import omni.usd
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    for env_idx in env_ids:
        env_idx_value = env_idx.item() if hasattr(env_idx, "item") else env_idx
        camera_path = camera_path_template.format(env_idx_value)
        stage = omni.usd.get_context().get_stage()
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            continue

        pos_delta_x = random.uniform(*position_deltas["x"])
        pos_delta_y = random.uniform(*position_deltas["y"])
        pos_delta_z = random.uniform(*position_deltas["z"])
        new_pos = (
            base_position[0] + pos_delta_x,
            base_position[1] + pos_delta_y,
            base_position[2] + pos_delta_z,
        )

        base_quat = Gf.Quatf(base_rotation[0], Gf.Vec3f(base_rotation[1], base_rotation[2], base_rotation[3]))
        base_rot = Gf.Rotation(base_quat)
        delta_pitch = random.uniform(*euler_deltas["pitch"])
        delta_yaw = random.uniform(*euler_deltas["yaw"])
        delta_roll = random.uniform(*euler_deltas["roll"])
        delta_rot = (
            Gf.Rotation(Gf.Vec3d(0, 0, 1), delta_yaw)
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), delta_pitch)
            * Gf.Rotation(Gf.Vec3d(1, 0, 0), delta_roll)
        )
        new_rot = delta_rot * base_rot
        new_quat = new_rot.GetQuat()

        xform = UsdGeom.Xformable(camera_prim)
        xform_ops = xform.GetOrderedXformOps()
        if not xform_ops:
            xform.AddTransformOp()
        xform_ops = xform.GetOrderedXformOps()
        for op in xform_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*new_pos))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                op.Set(new_quat)


def randomize_camera_focal_length(
    env, env_ids: torch.Tensor, camera_path_template: str, focal_length_range: tuple = (0.8, 1.8)
) -> None:
    """Randomizes the focal length of cameras."""
    import omni.usd
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    stage = omni.usd.get_context().get_stage()
    for env_idx in env_ids:
        camera_path = camera_path_template.format(env_idx)
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim.IsValid():
            continue
        focal_length = random.uniform(focal_length_range[0], focal_length_range[1])
        focal_attr = camera_prim.GetAttribute("focalLength")
        if focal_attr.IsValid():
            focal_attr.Set(focal_length)


class randomize_visual_appearance_multiple_meshes(ManagerTermBase):
    """Randomize the visual appearance (texture or color) of multiple mesh bodies using Replicator API.

    Use ``texture_prob`` to control probability of texture vs solid color:
    - ``texture_prob=1.0``: Always use textures
    - ``texture_prob=0.0``: Always use solid colors

    Requires ``InteractiveSceneCfg.replicate_physics = False``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("omni.replicator.core")
        import omni.replicator.core as rep

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        texture_paths = cfg.params.get("texture_paths")
        texture_config_path = cfg.params.get("texture_config_path")
        event_name = cfg.params.get("event_name")
        mesh_names: list[str] = cfg.params.get("mesh_names", [])

        self.texture_prob = cfg.params.get("texture_prob", 1.0)
        self.diffuse_tint_range = cfg.params.get("diffuse_tint_range")
        self.colors = cfg.params.get("colors", {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)})
        self.color_event_name = f"{event_name}_color"
        self._texture_scale_range = cfg.params.get("texture_scale_range", (0.7, 5.0))
        self._roughness_range = cfg.params.get("roughness_range", (0.0, 1.0))
        self._metallic_range = cfg.params.get("metallic_range", (0.0, 1.0))
        self._specular_range = cfg.params.get("specular_range", (0.0, 1.0))

        if texture_config_path is not None:
            texture_paths = utils.load_asset_paths_from_config(
                texture_config_path, cache_subdir="textures", skip_validation=False
            )
            logging.info(f"[{event_name}] Loaded {len(texture_paths)} texture paths.")
        if self.texture_prob > 0 and (texture_paths is None or len(texture_paths) == 0):
            raise RuntimeError(
                f"[{event_name}] texture_prob={self.texture_prob} but no texture paths loaded."
            )
        if texture_paths:
            local_only = [p for p in texture_paths if not p.startswith("http://") and not p.startswith("https://")]
            missing = [p for p in local_only if not os.path.exists(p)]
            if missing:
                raise RuntimeError(
                    f"[{event_name}] {len(missing)}/{len(local_only)} local texture files missing. "
                    f"First 3: {missing[:3]}"
                )

        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual appearance with replicate_physics=True. "
                "Set 'replicate_physics' to False in InteractiveSceneCfg."
            )

        asset = env.scene[asset_cfg.name]
        asset_prim_path = asset.cfg.prim_path

        if len(mesh_names) == 0:
            pattern_with_visuals = f"{asset_prim_path}/.*/visuals"
            matching_prims = sim_utils.find_matching_prim_paths(pattern_with_visuals)
            if matching_prims:
                prim_path_pattern = pattern_with_visuals
            else:
                prim_path_pattern = f"{asset_prim_path}/.*"
        else:
            mesh_prim_paths = []
            for mesh_name in mesh_names:
                if not mesh_name.startswith("/"):
                    mesh_name = "/" + mesh_name
                mesh_prim_paths.append(f"{asset_prim_path}{mesh_name}")
            prim_path_pattern = "|".join(mesh_prim_paths)

        self.texture_paths = texture_paths
        unique_seed = hash(event_name) % (2**31)
        self.texture_rng = rep.rng.ReplicatorRNG(seed=unique_seed)
        self.prim_path_pattern = prim_path_pattern

        stage = sim_utils.SimulationContext.instance().stage
        prims_group = rep.functional.get.prims(path_pattern=prim_path_pattern, stage=stage)
        num_prims = len(prims_group)

        if num_prims == 0:
            raise RuntimeError(
                f"[randomize_visual_appearance_multiple_meshes] No prims found matching: {prim_path_pattern}."
            )

        for prim in prims_group:
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

        self.material_prims = rep.functional.create_batch.material(
            mdl="OmniPBR.mdl", bind_prims=prims_group, count=num_prims, project_uvw=True
        )
        self._stage = stage
        self._texture_verified = False

        from pxr import Sdf, UsdShade
        self._shader_prims = []
        for i, mat_prim in enumerate(self.material_prims):
            mat_path = str(mat_prim.GetPath()) if hasattr(mat_prim, "GetPath") else str(mat_prim)
            shader_prim = stage.GetPrimAtPath(Sdf.Path(f"{mat_path}/Shader"))
            if not shader_prim.IsValid():
                raise RuntimeError(f"[{event_name}] Shader not found at {mat_path}/Shader.")
            self._shader_prims.append(shader_prim)

            material = UsdShade.Material(mat_prim)
            target_prim = prims_group[i]
            UsdShade.MaterialBindingAPI.Apply(target_prim)
            UsdShade.MaterialBindingAPI(target_prim).Bind(
                material, UsdShade.Tokens.strongerThanDescendants
            )

        _required_inputs = {
            "texture_scale": Sdf.ValueTypeNames.Float2,
            "reflection_roughness_constant": Sdf.ValueTypeNames.Float,
            "metallic_constant": Sdf.ValueTypeNames.Float,
            "specular_level": Sdf.ValueTypeNames.Float,
        }
        for shader_prim in self._shader_prims:
            shader = UsdShade.Shader(shader_prim)
            props = shader_prim.GetPropertyNames()
            for attr_name, attr_type in _required_inputs.items():
                if f"inputs:{attr_name}" not in props:
                    shader.CreateInput(attr_name, attr_type)

        if isinstance(self.colors, dict):
            self._color_low = np.array([self.colors[key][0] for key in ["r", "g", "b"]])
            self._color_high = np.array([self.colors[key][1] for key in ["r", "g", "b"]])
        else:
            self._color_list = list(self.colors)
            self._color_low = None
            self._color_high = None

        self(env, torch.arange(env.num_envs, device=env.device), **cfg.params)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str] | None = None,
        texture_config_path: str | None = None,
        mesh_names: list[str] = [],
        texture_prob: float = 1.0,
        colors=None,
        diffuse_tint_range=None,
        texture_scale_range=None,
        roughness_range=None,
        metallic_range=None,
        specular_range=None,
    ):
        if not self._shader_prims:
            return

        from pxr import Sdf

        rng = self.texture_rng.generator
        num_prims = len(self._shader_prims)

        use_texture_mask = rng.random(size=num_prims) < self.texture_prob
        rand_roughness = rng.uniform(self._roughness_range[0], self._roughness_range[1], size=num_prims)
        rand_metallic = rng.uniform(self._metallic_range[0], self._metallic_range[1], size=num_prims)
        rand_specular = rng.uniform(self._specular_range[0], self._specular_range[1], size=num_prims)

        random_textures = None
        if self.texture_paths and use_texture_mask.any():
            random_textures = rng.choice(self.texture_paths, size=num_prims)

        random_colors = None
        if not use_texture_mask.all():
            if self._color_low is not None:
                random_colors = rng.uniform(self._color_low, self._color_high, size=(num_prims, 3))
            else:
                indices = rng.integers(0, len(self._color_list), size=num_prims)
                random_colors = np.array([self._color_list[i] for i in indices])

        with Sdf.ChangeBlock():
            for i, shader_prim in enumerate(self._shader_prims):
                shader_prim.GetAttribute("inputs:reflection_roughness_constant").Set(float(rand_roughness[i]))
                shader_prim.GetAttribute("inputs:metallic_constant").Set(float(rand_metallic[i]))
                shader_prim.GetAttribute("inputs:specular_level").Set(float(rand_specular[i]))

                if use_texture_mask[i] and random_textures is not None:
                    shader_prim.GetAttribute("inputs:diffuse_texture").Set(
                        Sdf.AssetPath(random_textures[i])
                    )
                    s = float(rng.uniform(self._texture_scale_range[0], self._texture_scale_range[1]))
                    shader_prim.GetAttribute("inputs:texture_scale").Set(Gf.Vec2f(s, s))
                    if self.diffuse_tint_range is not None:
                        t = rng.uniform(self.diffuse_tint_range[0], self.diffuse_tint_range[1], size=3)
                        shader_prim.GetAttribute("inputs:diffuse_tint").Set(
                            Gf.Vec3f(float(t[0]), float(t[1]), float(t[2]))
                        )
                else:
                    shader_prim.GetAttribute("inputs:diffuse_texture").Set(Sdf.AssetPath(""))
                    if random_colors is not None:
                        shader_prim.GetAttribute("inputs:diffuse_color_constant").Set(
                            Gf.Vec3f(float(random_colors[i][0]), float(random_colors[i][1]), float(random_colors[i][2]))
                        )

        if not self._texture_verified and random_textures is not None and use_texture_mask.any():
            self._verify_texture_applied(random_textures[int(np.argmax(use_texture_mask))], event_name)
            self._texture_verified = True

    def _verify_texture_applied(self, expected_texture: str, event_name: str):
        shader_prim = self._shader_prims[0]
        shader_path = str(shader_prim.GetPath())
        tex_attr = shader_prim.GetAttribute("inputs:diffuse_texture")
        if not tex_attr or not tex_attr.IsValid():
            raise RuntimeError(
                f"[{event_name}] Texture verify failed: 'inputs:diffuse_texture' not found on {shader_path}."
            )
        current_val = tex_attr.Get()
        if current_val is None or str(current_val) == "":
            raise RuntimeError(
                f"[{event_name}] Texture verify failed: diffuse_texture empty after Set. Expected: {expected_texture}."
            )
