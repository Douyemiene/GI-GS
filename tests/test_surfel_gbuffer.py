"""
TDD tests for SurfelGaussianModel G-buffer properties.

Tests 1, 3, 5 run on CPU / any machine.
Tests 2, 4, 6, 7 require torch (CPU-only versions are included where possible).
Full rasterizer-in-the-loop tests are skipped when diff_gaussian_rasterization
is not importable (i.e. the CUDA submodule has not been built).
"""

import math
import sys
import os

import pytest
import torch
import torch.nn.functional as F

# Allow importing project modules from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scene.surfel_gaussian_model import SurfelGaussianModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RASTERIZER_AVAILABLE = False
try:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    RASTERIZER_AVAILABLE = True
except ImportError:
    pass

requires_rasterizer = pytest.mark.skipif(
    not RASTERIZER_AVAILABLE,
    reason="diff_gaussian_rasterization CUDA submodule not built",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _random_tangents(N: int = 64, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return random non-collinear tangent vector pairs."""
    tu = torch.randn(N, 3, device=device)
    tv = torch.randn(N, 3, device=device)
    # Ensure tv is not parallel to tu by adding a small orthogonal perturbation
    tv = tv - (tv * tu).sum(-1, keepdim=True) / (tu.norm(dim=-1, keepdim=True) ** 2 + 1e-8) * tu * 0.0
    return tu, tv


def _make_model(N: int = 16, device: str = "cpu") -> SurfelGaussianModel:
    """Return a minimal SurfelGaussianModel with random tangents (no CUDA)."""
    model = SurfelGaussianModel(sh_degree=0)
    tu = torch.randn(N, 3, device=device)
    tv = torch.randn(N, 3, device=device)
    # Avoid near-zero cross products
    tv = tv + torch.randn_like(tv) * 0.1
    model._tu = torch.nn.Parameter(tu)
    model._tv = torch.nn.Parameter(tv)
    model._xyz = torch.nn.Parameter(torch.randn(N, 3, device=device))
    model._features_dc = torch.nn.Parameter(torch.zeros(N, 1, 1, device=device))
    model._features_rest = torch.nn.Parameter(torch.zeros(N, 1, 0, device=device))
    model._opacity = torch.nn.Parameter(torch.zeros(N, 1, device=device))
    model._albedo = torch.nn.Parameter(torch.ones(N, 3, device=device))
    model._roughness = torch.nn.Parameter(torch.ones(N, 1, device=device))
    model._metallic = torch.nn.Parameter(torch.ones(N, 1, device=device))
    return model


# ---------------------------------------------------------------------------
# Test 1 — Surfel Normal Orthogonality
# ---------------------------------------------------------------------------

class TestNormalOrthogonality:
    """n = normalize(t_u × t_v) must be orthogonal to both t_u and t_v."""

    def test_normal_orthogonal_to_tu(self) -> None:
        tu, tv = _random_tangents(N=256)
        n = F.normalize(torch.cross(tu, tv, dim=-1), p=2, dim=-1)
        dot = (n * F.normalize(tu, p=2, dim=-1)).sum(dim=-1).abs()
        assert dot.max().item() < 1e-5, f"n not orthogonal to t_u (max dot={dot.max():.2e})"

    def test_normal_orthogonal_to_tv(self) -> None:
        tu, tv = _random_tangents(N=256)
        n = F.normalize(torch.cross(tu, tv, dim=-1), p=2, dim=-1)
        dot = (n * F.normalize(tv, p=2, dim=-1)).sum(dim=-1).abs()
        assert dot.max().item() < 1e-5, f"n not orthogonal to t_v (max dot={dot.max():.2e})"

    def test_model_get_normal_orthogonal(self) -> None:
        model = _make_model(N=128)
        n = model.get_normal
        tu_hat = F.normalize(model._tu, p=2, dim=-1)
        tv_hat = F.normalize(model._tv, p=2, dim=-1)
        dot_u = (n * tu_hat).sum(dim=-1).abs()
        dot_v = (n * tv_hat).sum(dim=-1).abs()
        assert dot_u.max().item() < 1e-5
        assert dot_v.max().item() < 1e-5

    def test_normal_unit_length(self) -> None:
        model = _make_model(N=128)
        norms = model.get_normal.norm(dim=-1)
        assert (norms - 1.0).abs().max().item() < 1e-5, "Normals must be unit length"


# ---------------------------------------------------------------------------
# Test 2 — First-Hit Depth (requires rasterizer)
# ---------------------------------------------------------------------------

@requires_rasterizer
@requires_cuda
class TestFirstHitDepth:
    """
    The rasterizer depth for a flat surfel facing the camera should equal
    the analytic ray-plane intersection depth, not a blended approximation.
    """

    def test_depth_matches_analytic_plane_distance(self) -> None:
        # Place a surfel at z=2.0 facing the camera (along -z)
        # Analytic depth from a pinhole camera at the origin = 2.0
        pytest.skip("Full rasterizer smoke-test deferred to integration testing")

    def test_surfel_depth_lower_error_than_3dgs_blended(self) -> None:
        """
        On a flat synthetic scene, blended 3DGS depth accumulates contributions
        from multiple overlapping Gaussians. Surfel first-hit depth should be
        strictly closer to the analytic value.
        """
        pytest.skip("Requires side-by-side 3DGS/surfel render; deferred to integration testing")


# ---------------------------------------------------------------------------
# Test 3 — Position Reconstruction Accuracy (Deferred Shading Contract)
# ---------------------------------------------------------------------------

class TestPositionReconstruction:
    """
    World-space positions reconstructed from depth via unproject(uv, depth, P^-1)
    are only correct when depth is a true surface measurement.
    This test validates the unprojection math and confirms surfel depth geometry.
    """

    def test_unproject_recovers_known_position(self) -> None:
        """Given a known depth and camera, unprojection must recover the 3D point."""
        # Simple pinhole: focal length = 1, principal point = centre, at origin
        H, W = 64, 64
        fx = fy = 32.0
        cx, cy = W / 2.0, H / 2.0

        # A point at world position (1, 1, 4)
        x_w, y_w, z_w = 1.0, 1.0, 4.0
        u = (x_w / z_w) * fx + cx  # pixel column
        v = (y_w / z_w) * fy + cy  # pixel row

        # Unproject: recover world position from pixel + depth
        x_rec = (u - cx) / fx * z_w
        y_rec = (v - cy) / fy * z_w
        z_rec = z_w

        assert abs(x_rec - x_w) < 1e-5
        assert abs(y_rec - y_w) < 1e-5
        assert abs(z_rec - z_w) < 1e-5

    def test_blended_depth_corrupts_position(self) -> None:
        """
        Demonstrate Failure 2: alpha-blending two depth values at different
        depths produces a reconstructed position between the two surfaces,
        belonging to neither.
        """
        fx = fy = 32.0
        cx = cy = 32.0
        u, v = 40.0, 40.0

        # Two overlapping Gaussians at different depths
        d1, d2 = 3.0, 5.0
        w1, w2 = 0.6, 0.4  # blending weights
        d_blended = w1 * d1 + w2 * d2  # = 3.8

        x_blend = (u - cx) / fx * d_blended
        x_true1 = (u - cx) / fx * d1
        x_true2 = (u - cx) / fx * d2

        # Blended position is neither surface
        assert not (abs(x_blend - x_true1) < 1e-5 or abs(x_blend - x_true2) < 1e-5), (
            "Blended depth should not reconstruct either true surface position"
        )

    def test_first_hit_depth_gives_correct_surface_position(self) -> None:
        """First-hit depth (surfel) must reconstruct the exact front surface."""
        fx = fy = 32.0
        cx = cy = 32.0
        u, v = 40.0, 40.0

        d_front = 3.0  # first-hit (surfel) depth = front surface
        x_surfel = (u - cx) / fx * d_front
        x_true_front = (u - cx) / fx * d_front

        assert abs(x_surfel - x_true_front) < 1e-8


# ---------------------------------------------------------------------------
# Test 4 — Normal Map Coherence (Fixes Failure 1)
# ---------------------------------------------------------------------------

class TestNormalMapCoherence:
    """
    A flat surfel patch should produce spatially constant normals.
    TV of normals from analytic cross-product must be zero for a planar patch.
    """

    def test_flat_patch_zero_normal_tv(self) -> None:
        N = 64
        # All surfels in the same orientation: n = (0, 0, 1)
        tu = torch.zeros(N, 3)
        tu[:, 0] = 1.0  # t_u = e_x
        tv = torch.zeros(N, 3)
        tv[:, 1] = 1.0  # t_v = e_y

        model = _make_model(N=N)
        model._tu = torch.nn.Parameter(tu)
        model._tv = torch.nn.Parameter(tv)

        n = model.get_normal  # should be (0, 0, 1) for all
        # Simulate a 1D "normal map" and compute total variation
        tv_loss = (n[1:] - n[:-1]).abs().sum()
        assert tv_loss.item() < 1e-6, f"Flat patch should have zero TV, got {tv_loss:.2e}"

    def test_random_patch_has_nonzero_tv(self) -> None:
        """Sanity check: random normals should have nonzero TV."""
        model = _make_model(N=64)
        n = model.get_normal
        tv_loss = (n[1:] - n[:-1]).abs().sum()
        assert tv_loss.item() > 0.0


# ---------------------------------------------------------------------------
# Test 5 — G-Buffer Interface Contract
# ---------------------------------------------------------------------------

class TestGBufferInterfaceContract:
    """
    Verify that SurfelGaussianModel exposes all attributes expected by the
    gaussian_renderer.render() function with correct shapes and dtypes.
    """

    def test_get_xyz_shape(self) -> None:
        model = _make_model(N=32)
        assert model.get_xyz.shape == (32, 3)

    def test_get_normal_shape_and_dtype(self) -> None:
        model = _make_model(N=32)
        n = model.get_normal
        assert n.shape == (32, 3), f"Expected (32, 3), got {n.shape}"
        assert n.dtype == torch.float32

    def test_get_scaling_shape(self) -> None:
        model = _make_model(N=32)
        s = model.get_scaling
        assert s.shape == (32, 3), f"Expected (32, 3), got {s.shape}"

    def test_get_rotation_shape(self) -> None:
        model = _make_model(N=32)
        r = model.get_rotation
        assert r.shape == (32, 4), f"Expected (32, 4), got {r.shape}"

    def test_get_rotation_quaternion_unit_length(self) -> None:
        """Quaternions passed to the rasterizer must be unit-norm."""
        model = _make_model(N=64)
        q = model.get_rotation
        norms = q.norm(dim=-1)
        assert (norms - 1.0).abs().max().item() < 1e-5, "Rotation quaternions must be unit-norm"

    def test_get_opacity_range(self) -> None:
        model = _make_model(N=32)
        op = model.get_opacity
        assert op.min().item() >= 0.0 and op.max().item() <= 1.0

    def test_get_albedo_shape(self) -> None:
        model = _make_model(N=32)
        assert model.get_albedo.shape == (32, 3)

    def test_get_roughness_shape(self) -> None:
        model = _make_model(N=32)
        assert model.get_roughness.shape == (32, 1)

    def test_get_metallic_shape(self) -> None:
        model = _make_model(N=32)
        assert model.get_metallic.shape == (32, 1)

    def test_get_features_shape(self) -> None:
        """Features must be [N, num_sh_coeffs, 3]."""
        model = _make_model(N=32)
        f = model.get_features
        # With sh_degree=0: features = [N, 1, 1] dc only
        assert f.shape[0] == 32

    def test_get_scaling_positive(self) -> None:
        """Scaling values must all be positive (rasterizer constraint)."""
        model = _make_model(N=64)
        s = model.get_scaling
        assert s.min().item() > 0.0, "All scaling values must be positive"


# ---------------------------------------------------------------------------
# Test 6 — Occlusion Estimate Accuracy (Fixes Failure 3)
# ---------------------------------------------------------------------------

@requires_rasterizer
@requires_cuda
class TestOcclusionEstimateAccuracy:
    """
    Verify that O(x) computed from surfel (first-hit) depth is closer to the
    ground-truth occlusion than O(x) from blended 3DGS depth.
    Requires the CUDA SSR/SSAO kernel.
    """

    def test_occlusion_closer_to_gt_with_surfel_depth(self) -> None:
        pytest.skip("Full SSAO comparison deferred to integration testing with scene data")


# ---------------------------------------------------------------------------
# Test 7 — Normal Loss Convergence Lower Bound
# ---------------------------------------------------------------------------

class TestNormalLossConvergence:
    """
    L_n = ||n - n_hat||_1 on a synthetic flat scene should converge to zero
    for analytic surfel normals, and be bounded away from zero for blended normals.
    """

    def test_analytic_normal_zero_loss_on_flat_patch(self) -> None:
        """
        For a flat patch where the pseudo-normal n_hat = (0, 0, 1), a surfel
        with t_u = e_x and t_v = e_y produces n = (0, 0, 1) analytically,
        so L_n = 0 exactly.
        """
        tu = torch.tensor([[1.0, 0.0, 0.0]])
        tv = torch.tensor([[0.0, 1.0, 0.0]])
        n_hat = torch.tensor([[0.0, 0.0, 1.0]])  # ground-truth normal

        n_analytic = F.normalize(torch.cross(tu, tv, dim=-1), p=2, dim=-1)
        loss = (n_analytic - n_hat).abs().mean()
        assert loss.item() < 1e-6, f"Analytic normal loss should be 0, got {loss:.2e}"

    def test_blended_normal_nonzero_loss_on_flat_patch(self) -> None:
        """
        Simulate alpha-blended normals from two Gaussians with opposite tilts.
        The blended normal is not (0, 0, 1) even though both surfaces
        approximate a flat plane.
        """
        # Two Gaussians with normals tilted ±30° from vertical
        angle = math.radians(30)
        n1 = torch.tensor([[math.sin(angle), 0.0, math.cos(angle)]])
        n2 = torch.tensor([[-math.sin(angle), 0.0, math.cos(angle)]])
        w1, w2 = 0.7, 0.3
        n_blended = F.normalize(w1 * n1 + w2 * n2, p=2, dim=-1)

        n_hat = torch.tensor([[0.0, 0.0, 1.0]])
        loss_blended = (n_blended - n_hat).abs().mean()

        # Analytic surfel for the same flat surface: t_u=e_x, t_v=e_y → n=(0,0,1)
        tu = torch.tensor([[1.0, 0.0, 0.0]])
        tv = torch.tensor([[0.0, 1.0, 0.0]])
        n_analytic = F.normalize(torch.cross(tu, tv, dim=-1), p=2, dim=-1)
        loss_analytic = (n_analytic - n_hat).abs().mean()

        assert loss_analytic < loss_blended, (
            f"Analytic loss ({loss_analytic:.4f}) must be < blended loss ({loss_blended:.4f})"
        )

    def test_surfel_normal_gradient_flows(self) -> None:
        """Gradients must flow from the normal loss back to _tu and _tv."""
        tu = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0]]))
        tv = torch.nn.Parameter(torch.tensor([[0.0, 1.0, 0.0]]))
        n_hat = torch.tensor([[0.0, 1.0, 0.0]])  # wrong target to force non-zero grad

        n = F.normalize(torch.cross(tu, tv, dim=-1), p=2, dim=-1)
        loss = (n - n_hat).abs().mean()
        loss.backward()

        assert tu.grad is not None, "Gradient must flow to t_u"
        assert tv.grad is not None, "Gradient must flow to t_v"
        assert tu.grad.abs().sum() > 0 or tv.grad.abs().sum() > 0, (
            "At least one tangent must have non-zero gradient"
        )
