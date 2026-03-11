"""
SurfelGaussianModel: 2D Gaussian surfel primitive for deferred shading.

Each surfel is parametrised by two learnable tangent vectors (_tu, _tv).
All geometry is derived analytically:
  - Normal:   n = normalize(t_u × t_v)          — always valid, never blended
  - Scaling:  [||t_u||, ||t_v||, ε]             — disc semi-axes + thin thickness
  - Rotation: quaternion of frame (t̂_u, t̂_v, n) — consistent with covariance

This satisfies the G-buffer contract for deferred rendering: one crisp surface
per pixel with a geometrically valid normal, enabling correct depth unprojection
for path-traced indirect illumination.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from pytorch3d.transforms import matrix_to_quaternion

from arguments import GroupParams
from utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p

# Minimal disc thickness (world units). Keeps the Gaussian non-degenerate.
_DISC_THICKNESS: float = 1e-3


class SurfelGaussianModel:
    """
    2D Gaussian surfel model for GI-GS deferred shading pipeline.

    Replaces _scaling + _rotation + _normal with _tu + _tv.
    The normal and covariance are derived analytically, fixing the three
    G-buffer failures caused by alpha-blended 3DGS geometry.
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_functions(self) -> None:
        """Register activation functions (subset of GaussianModel's)."""
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.material_activation = torch.sigmoid

    def __init__(self, sh_degree: int) -> None:
        self.active_sh_degree: int = 0
        self.max_sh_degree: int = sh_degree

        # Learnable parameters
        self._xyz: torch.Tensor = torch.empty(0)
        self._features_dc: torch.Tensor = torch.empty(0)
        self._features_rest: torch.Tensor = torch.empty(0)
        self._tu: torch.Tensor = torch.empty(0)   # tangent u [N, 3]
        self._tv: torch.Tensor = torch.empty(0)   # tangent v [N, 3]
        self._opacity: torch.Tensor = torch.empty(0)
        self._albedo: torch.Tensor = torch.empty(0)
        self._roughness: torch.Tensor = torch.empty(0)
        self._metallic: torch.Tensor = torch.empty(0)

        # Densification state (non-parameter tensors)
        self.max_radii2D: torch.Tensor = torch.empty(0)
        self.xyz_gradient_accum: torch.Tensor = torch.empty(0)
        self.xyz_gradient_accum_abs: torch.Tensor = torch.empty(0)
        self.xyz_gradient_accum_abs_max: torch.Tensor = torch.empty(0)
        self.denom: torch.Tensor = torch.empty(0)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.percent_dense: float = 0.0
        self.spatial_lr_scale: float = 0.0

        self.setup_functions()

    # ------------------------------------------------------------------
    # Analytic geometry properties (core contribution)
    # ------------------------------------------------------------------

    @property
    def get_normal(self) -> torch.Tensor:
        """
        Analytic unit normal: n = normalize(t_u × t_v).

        Geometrically valid by construction — no alpha-blending involved.
        Fixes Failure 1 (G-buffer normal noise).
        """
        return F.normalize(torch.cross(self._tu, self._tv, dim=-1), p=2, dim=-1)

    @property
    def get_scaling(self) -> torch.Tensor:
        """
        Disc semi-axes derived from tangent norms: [||t_u||, ||t_v||, ε].

        The thin third axis keeps the Gaussian non-degenerate for the CUDA
        rasterizer while modelling a flat disc.
        """
        tu_norm = torch.clamp_min(self._tu.norm(dim=-1, keepdim=True), 1e-6)  # [N, 1]
        tv_norm = torch.clamp_min(self._tv.norm(dim=-1, keepdim=True), 1e-6)  # [N, 1]
        eps = torch.full_like(tu_norm, _DISC_THICKNESS)                       # [N, 1]
        return torch.cat([tu_norm, tv_norm, eps], dim=-1)                     # [N, 3]

    @property
    def get_rotation(self) -> torch.Tensor:
        """
        Quaternion (w, x, y, z) for the surfel coordinate frame (t̂_u, t̂_v, n).

        Gram-Schmidt orthogonalisation ensures a valid rotation matrix even
        when _tu and _tv drift during optimisation.
        """
        tu_hat = F.normalize(self._tu, p=2, dim=-1)            # [N, 3]
        n = self.get_normal                                     # [N, 3]
        # Re-orthogonalise tv via Gram-Schmidt: tv_orth = n × tu_hat
        tv_hat = F.normalize(torch.cross(n, tu_hat, dim=-1), p=2, dim=-1)
        # Column matrix [tu_hat | tv_hat | n]: column i is the image of basis e_i
        R = torch.stack([tu_hat, tv_hat, n], dim=-1)           # [N, 3, 3]
        return matrix_to_quaternion(R)                         # [N, 4] (w, x, y, z)

    def _raw_rotation(self) -> torch.Tensor:
        """Return quaternion tensor used by densification helpers (no gradient)."""
        with torch.no_grad():
            return self.get_rotation

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """Covariance from derived scaling and rotation (compatibility shim)."""
        L = build_scaling_rotation(scaling_modifier * self.get_scaling, self.get_rotation)
        actual_covariance = L @ L.transpose(1, 2)
        return strip_symmetric(actual_covariance)

    # ------------------------------------------------------------------
    # Accessors that match GaussianModel's interface
    # ------------------------------------------------------------------

    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz

    @property
    def get_features(self) -> torch.Tensor:
        return torch.cat((self._features_dc, self._features_rest), dim=1)

    @property
    def get_opacity(self) -> torch.Tensor:
        return self.opacity_activation(self._opacity)

    @property
    def get_albedo(self) -> torch.Tensor:
        return self.material_activation(self._albedo)

    @property
    def get_roughness(self) -> torch.Tensor:
        return self.material_activation(self._roughness)

    @property
    def get_metallic(self) -> torch.Tensor:
        return self.material_activation(self._metallic)

    def oneupSHdegree(self) -> None:
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def init_normal(self, coe: float) -> None:
        """No-op: surfel normal is derived analytically from tangents."""
        pass

    # ------------------------------------------------------------------
    # Initialisation from SfM point cloud
    # ------------------------------------------------------------------

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float) -> None:
        """
        Initialise surfels from an SfM point cloud.

        Tangent vectors are initialised as axis-aligned unit vectors scaled by
        the nearest-neighbour distance. t_u points along world X, t_v along Y,
        giving a horizontal disc with n = (0, 0, 1) at initialisation.
        """
        self.spatial_lr_scale = spatial_lr_scale
        pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32).cuda()
        colors = RGB2SH(torch.tensor(np.asarray(pcd.colors), dtype=torch.float32).cuda())

        N = pts.shape[0]
        print(f"[SurfelGaussianModel] Number of points at initialisation: {N}")

        features = torch.zeros((N, 3, (self.max_sh_degree + 1) ** 2), device="cuda")
        features[:, :3, 0] = colors

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 1e-7
        )
        scale_init = torch.sqrt(dist2)  # [N]

        # Axis-aligned disc: t_u along X, t_v along Y
        tu = torch.zeros((N, 3), device="cuda")
        tv = torch.zeros((N, 3), device="cuda")
        tu[:, 0] = scale_init  # t_u = scale * e_x
        tv[:, 1] = scale_init  # t_v = scale * e_y

        opacities = inverse_sigmoid(0.1 * torch.ones((N, 1), device="cuda"))
        albedo = torch.ones((N, 3), device="cuda")
        roughness = torch.ones((N, 1), device="cuda")
        metallic = torch.ones((N, 1), device="cuda")

        self._xyz = nn.Parameter(pts.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._tu = nn.Parameter(tu.requires_grad_(True))
        self._tv = nn.Parameter(tv.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._albedo = nn.Parameter(albedo.requires_grad_(True))
        self._roughness = nn.Parameter(roughness.requires_grad_(True))
        self._metallic = nn.Parameter(metallic.requires_grad_(True))
        self.max_radii2D = torch.zeros(N, device="cuda")

    # ------------------------------------------------------------------
    # Optimiser setup
    # ------------------------------------------------------------------

    def training_setup(self, training_args: GroupParams) -> None:
        """Set up Adam optimiser with per-parameter learning rates."""
        self.percent_dense = training_args.percent_dense
        N = self.get_xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((N, 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._tu], "lr": training_args.scaling_lr, "name": "tu"},
            {"params": [self._tv], "lr": training_args.scaling_lr, "name": "tv"},
            {"params": [self._albedo], "lr": training_args.opacity_lr, "name": "albedo"},
            {"params": [self._roughness], "lr": training_args.opacity_lr, "name": "roughness"},
            {"params": [self._metallic], "lr": training_args.opacity_lr, "name": "metallic"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self.BRDF_scheduler_args = get_expon_lr_func(
            lr_init=training_args.opacity_lr,
            lr_final=training_args.BRDF_lr,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=10000,
        )

    def re_setup(self, training_args: GroupParams) -> None:
        """Re-initialise optimiser (used when switching to PBR phase)."""
        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._tu], "lr": training_args.scaling_lr, "name": "tu"},
            {"params": [self._tv], "lr": training_args.scaling_lr, "name": "tv"},
            {"params": [self._albedo], "lr": training_args.opacity_lr, "name": "albedo"},
            {"params": [self._roughness], "lr": training_args.opacity_lr, "name": "roughness"},
            {"params": [self._metallic], "lr": training_args.opacity_lr, "name": "metallic"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def update_learning_rate(self, iteration: int) -> float:
        """Learning rate scheduling per step (mirrors GaussianModel)."""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in ("albedo", "roughness", "metallic"):
                lr = self.BRDF_scheduler_args(iteration - 30000)
                param_group["lr"] = lr
                return lr
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
        return 0.0

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def capture(self) -> Tuple:
        """Serialise model state for checkpointing."""
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._tu,
            self._tv,
            self._opacity,
            self._albedo,
            self._roughness,
            self._metallic,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.xyz_gradient_accum_abs_max,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(
        self, model_args: Tuple, training_args: Optional[GroupParams] = None
    ) -> None:
        """Restore model state from a checkpoint tuple."""
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._tu,
            self._tv,
            self._opacity,
            self._albedo,
            self._roughness,
            self._metallic,
            self.max_radii2D,
            xyz_gradient_accum,
            xyz_gradient_accum_abs,
            xyz_gradient_accum_abs_max,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        if training_args is not None:
            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
            self.xyz_gradient_accum_abs_max = xyz_gradient_accum_abs_max
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

    # ------------------------------------------------------------------
    # PLY I/O
    # ------------------------------------------------------------------

    def construct_list_of_attributes(self) -> List[str]:
        attrs = ["x", "y", "z"]
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append(f"f_dc_{i}")
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append(f"f_rest_{i}")
        attrs.append("opacity")
        attrs += [f"tu_{i}" for i in range(3)]
        attrs += [f"tv_{i}" for i in range(3)]
        attrs += [f"albedo_{i}" for i in range(self._albedo.shape[1])]
        attrs.append("roughness")
        attrs.append("metallic")
        return attrs

    def save_ply(self, path: str) -> None:
        mkdir_p(os.path.dirname(path))
        xyz = self._xyz.detach().cpu().numpy()
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        tu = self._tu.detach().cpu().numpy()
        tv = self._tv.detach().cpu().numpy()
        albedo = self._albedo.detach().cpu().numpy()
        roughness = self._roughness.detach().cpu().numpy()
        metallic = self._metallic.detach().cpu().numpy()

        dtype_full = [(attr, "f4") for attr in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, f_dc, f_rest, opacities, tu, tv, albedo, roughness, metallic), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        PlyData([PlyElement.describe(elements, "vertex")]).write(path)

    def reset_opacity(self) -> None:
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path: str) -> None:
        plydata = PlyData.read(path)
        el = plydata.elements[0]

        xyz = np.stack(
            (np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])), axis=1
        )
        opacities = np.asarray(el["opacity"])[..., np.newaxis]
        tu = np.stack([np.asarray(el[f"tu_{i}"]) for i in range(3)], axis=1)
        tv = np.stack([np.asarray(el[f"tv_{i}"]) for i in range(3)], axis=1)
        albedo = np.stack([np.asarray(el[f"albedo_{i}"]) for i in range(3)], axis=1)
        roughness = np.asarray(el["roughness"])[..., np.newaxis]
        metallic = np.asarray(el["metallic"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(el["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(el["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(el["f_dc_2"])

        extra_f_names = sorted(
            [p.name for p in el.properties if p.name.startswith("f_rest_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(el[attr_name])
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._tu = nn.Parameter(
            torch.tensor(tu, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._tv = nn.Parameter(
            torch.tensor(tv, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._albedo = nn.Parameter(
            torch.tensor(albedo, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._roughness = nn.Parameter(
            torch.tensor(roughness, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._metallic = nn.Parameter(
            torch.tensor(metallic, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")
        self.active_sh_degree = self.max_sh_degree

    # ------------------------------------------------------------------
    # Optimiser tensor management (mirrors GaussianModel exactly)
    # ------------------------------------------------------------------

    def replace_tensor_to_optimizer(self, tensor: torch.Tensor, name: str) -> Dict:
        optimizable_tensors: Dict = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask: torch.Tensor) -> Dict:
        optimizable_tensors: Dict = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask: torch.Tensor) -> None:
        valid_mask = ~mask
        opt = self._prune_optimizer(valid_mask)
        self._xyz = opt["xyz"]
        self._features_dc = opt["f_dc"]
        self._features_rest = opt["f_rest"]
        self._opacity = opt["opacity"]
        self._tu = opt["tu"]
        self._tv = opt["tv"]
        self._albedo = opt["albedo"]
        self._roughness = opt["roughness"]
        self._metallic = opt["metallic"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_mask]
        self.xyz_gradient_accum_abs_max = self.xyz_gradient_accum_abs_max[valid_mask]
        self.denom = self.denom[valid_mask]
        self.max_radii2D = self.max_radii2D[valid_mask]

    def cat_tensors_to_optimizer(self, tensors_dict: Dict) -> Dict:
        optimizable_tensors: Dict = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            ext = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(ext)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(ext)), dim=0
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], ext), dim=0).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], ext), dim=0).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ------------------------------------------------------------------
    # Densification
    # ------------------------------------------------------------------

    def densification_postfix(
        self,
        new_xyz: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_opacities: torch.Tensor,
        new_tu: torch.Tensor,
        new_tv: torch.Tensor,
        new_albedo: torch.Tensor,
        new_roughness: torch.Tensor,
        new_metallic: torch.Tensor,
    ) -> None:
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "tu": new_tu,
            "tv": new_tv,
            "albedo": new_albedo,
            "roughness": new_roughness,
            "metallic": new_metallic,
        }
        opt = self.cat_tensors_to_optimizer(d)
        self._xyz = opt["xyz"]
        self._features_dc = opt["f_dc"]
        self._features_rest = opt["f_rest"]
        self._opacity = opt["opacity"]
        self._tu = opt["tu"]
        self._tv = opt["tv"]
        self._albedo = opt["albedo"]
        self._roughness = opt["roughness"]
        self._metallic = opt["metallic"]

        N = self.get_xyz.shape[0]
        self.xyz_gradient_accum = torch.zeros((N, 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((N, 1), device="cuda")
        self.xyz_gradient_accum_abs_max = torch.zeros((N, 1), device="cuda")
        self.denom = torch.zeros((N, 1), device="cuda")
        self.max_radii2D = torch.zeros(N, device="cuda")

    def densify_and_split(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        grads_abs: torch.Tensor,
        grad_abs_threshold: float,
        scene_extent: float,
        N: int = 2,
    ) -> None:
        """Split large surfels into N smaller children distributed within the disc."""
        n_init = self.get_xyz.shape[0]
        padded_grad = torch.zeros(n_init, device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected = torch.where(padded_grad >= grad_threshold, True, False)

        padded_grad_abs = torch.zeros(n_init, device="cuda")
        padded_grad_abs[: grads_abs.shape[0]] = grads_abs.squeeze()
        selected_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)

        selected = torch.logical_or(selected, selected_abs)
        selected = torch.logical_and(
            selected,
            self.get_scaling.max(dim=1).values > self.percent_dense * scene_extent,
        )

        # Scatter child centres within the parent disc extent
        stds = self.get_scaling[selected].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._raw_rotation()[selected]).repeat(N, 1, 1)
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected].repeat(N, 1)
        )

        # Scale down tangent vectors proportionally (smaller disc)
        scale_factor = 1.0 / (0.8 * N)
        new_tu = self._tu[selected].repeat(N, 1) * scale_factor
        new_tv = self._tv[selected].repeat(N, 1) * scale_factor

        new_features_dc = self._features_dc[selected].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected].repeat(N, 1, 1)
        new_opacity = self._opacity[selected].repeat(N, 1)
        new_albedo = self._albedo[selected].repeat(N, 1)
        new_roughness = self._roughness[selected].repeat(N, 1)
        new_metallic = self._metallic[selected].repeat(N, 1)

        self.densification_postfix(
            new_xyz, new_features_dc, new_features_rest,
            new_opacity, new_tu, new_tv,
            new_albedo, new_roughness, new_metallic,
        )
        prune_filter = torch.cat(
            (selected, torch.zeros(N * selected.sum(), device="cuda", dtype=bool))
        )
        self.prune_points(prune_filter)

    def densify_and_clone(
        self,
        grads: torch.Tensor,
        grad_threshold: float,
        grads_abs: torch.Tensor,
        grad_abs_threshold: float,
        scene_extent: float,
    ) -> None:
        """Clone small high-gradient surfels; new surfel jittered within parent disc."""
        selected = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_abs = torch.where(
            torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False
        )
        selected = torch.logical_or(selected, selected_abs)
        selected = torch.logical_and(
            selected,
            self.get_scaling.max(dim=1).values <= self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected]
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._raw_rotation()[selected])
        new_xyz = (
            torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.get_xyz[selected]
        )

        self.densification_postfix(
            new_xyz,
            self._features_dc[selected],
            self._features_rest[selected],
            self._opacity[selected],
            self._tu[selected],
            self._tv[selected],
            self._albedo[selected],
            self._roughness[selected],
            self._metallic[selected],
        )

    def densify_and_prune(
        self,
        max_grad: float,
        min_opacity: float,
        extent: float,
        max_screen_size: int,
    ) -> None:
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        self.densify_and_clone(grads, max_grad, grads_abs, Q, extent)
        self.densify_and_split(grads, max_grad, grads_abs, Q, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(
        self,
        viewspace_point_tensor: torch.Tensor,
        update_filter: torch.Tensor,
    ) -> None:
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        abs_grad = (
            torch.abs(viewspace_point_tensor.grad[update_filter, :1])
            + torch.abs(viewspace_point_tensor.grad[update_filter, 1:2])
        )
        abs_grad_norm = torch.norm(abs_grad, dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += abs_grad_norm
        self.xyz_gradient_accum_abs_max[update_filter] = torch.max(
            self.xyz_gradient_accum_abs_max[update_filter], abs_grad_norm
        )
        self.denom[update_filter] += 1
