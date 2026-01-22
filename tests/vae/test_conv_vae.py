"""
Tests for ConvVAE and its components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

import os
import sys

# Add the src directory to path for imports
src_path = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "src"
)
sys.path.insert(0, os.path.abspath(src_path))

from genofoundation.models.vae.conv import (
    ResBlockDown,
    ResBlockUp,
    ConvEncoder,
    ConvDecoder,
)
from genofoundation.models.vae.conv_vae import (
    Tensor2Image,
    Image2Tensor,
    ConvVAE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def img_params():
    """Basic image parameters for testing."""
    return {"img_h": 32, "img_w": 32}


@pytest.fixture
def px_ind():
    """Pixel indices mapping n_features to image pixels."""
    # Simulate 100 features mapped to specific pixels in a 32x32 image
    np.random.seed(42)
    n_features = 100
    n_pixels = 32 * 32
    return np.random.choice(n_pixels, size=n_features, replace=False)


@pytest.fixture
def small_px_ind():
    """Smaller pixel indices for faster tests."""
    np.random.seed(42)
    n_features = 50
    n_pixels = 16 * 16
    return np.random.choice(n_pixels, size=n_features, replace=False)


@pytest.fixture
def batch_size():
    return 8


@pytest.fixture
def latent_dim():
    return 32


# =============================================================================
# Tests for Tensor2Image
# =============================================================================

class TestTensor2Image:
    """Tests for Tensor2Image module."""

    def test_output_shape(self, px_ind, img_params, batch_size):
        """Test that output has correct shape."""
        t2i = Tensor2Image(px_ind, img_params["img_h"], img_params["img_w"])
        n_features = len(px_ind)
        x = torch.randn(batch_size, n_features)

        img = t2i(x)

        assert img.shape == (batch_size, 1, img_params["img_h"], img_params["img_w"])

    def test_values_placed_correctly(self, px_ind, img_params):
        """Test that values are placed at correct pixel positions."""
        t2i = Tensor2Image(px_ind, img_params["img_h"], img_params["img_w"])
        n_features = len(px_ind)

        # Create input with known values
        x = torch.arange(n_features, dtype=torch.float32).unsqueeze(0)  # [1, n_features]

        img = t2i(x)
        img_flat = img.reshape(1, -1)  # [1, H*W]

        # Check values at pixel indices
        for i, px in enumerate(px_ind):
            assert img_flat[0, px].item() == float(i)

    def test_non_indexed_pixels_are_zero(self, px_ind, img_params):
        """Test that pixels not in px_ind are zero."""
        t2i = Tensor2Image(px_ind, img_params["img_h"], img_params["img_w"])
        n_features = len(px_ind)

        x = torch.ones(1, n_features)
        img = t2i(x)
        img_flat = img.reshape(-1)

        # All pixels not in px_ind should be zero
        all_pixels = set(range(img_params["img_h"] * img_params["img_w"]))
        non_indexed = all_pixels - set(px_ind)

        for px in non_indexed:
            assert img_flat[px].item() == 0.0

    def test_dtype_preserved(self, px_ind, img_params):
        """Test that dtype is preserved."""
        t2i = Tensor2Image(px_ind, img_params["img_h"], img_params["img_w"])
        n_features = len(px_ind)

        x = torch.randn(2, n_features, dtype=torch.float64)
        img = t2i(x)

        assert img.dtype == torch.float64


# =============================================================================
# Tests for Image2Tensor
# =============================================================================

class TestImage2Tensor:
    """Tests for Image2Tensor module."""

    def test_output_shape(self, px_ind, img_params, batch_size):
        """Test that output has correct shape."""
        i2t = Image2Tensor(px_ind)
        n_features = len(px_ind)

        img = torch.randn(batch_size, 1, img_params["img_h"], img_params["img_w"])
        x = i2t(img)

        assert x.shape == (batch_size, n_features)

    def test_values_extracted_correctly(self, px_ind, img_params):
        """Test that correct pixel values are extracted."""
        i2t = Image2Tensor(px_ind)
        n_features = len(px_ind)
        n_pixels = img_params["img_h"] * img_params["img_w"]

        # Create image with known values at each pixel
        img_flat = torch.arange(n_pixels, dtype=torch.float32)
        img = img_flat.reshape(1, 1, img_params["img_h"], img_params["img_w"])

        x = i2t(img)

        # Check extracted values match pixel indices
        for i, px in enumerate(px_ind):
            assert x[0, i].item() == float(px)


# =============================================================================
# Tests for Tensor2Image and Image2Tensor roundtrip
# =============================================================================

class TestTensorImageRoundtrip:
    """Tests for roundtrip conversion between tensor and image."""

    def test_roundtrip_preserves_values(self, px_ind, img_params, batch_size):
        """Test that tensor -> image -> tensor preserves values."""
        t2i = Tensor2Image(px_ind, img_params["img_h"], img_params["img_w"])
        i2t = Image2Tensor(px_ind)
        n_features = len(px_ind)

        x_original = torch.randn(batch_size, n_features)

        img = t2i(x_original)
        x_recovered = i2t(img)

        torch.testing.assert_close(x_original, x_recovered)


# =============================================================================
# Tests for ResBlockDown
# =============================================================================

class TestResBlockDown:
    """Tests for ResBlockDown module."""

    def test_output_shape_with_stride2(self):
        """Test output shape with stride=2 (downsampling)."""
        block = ResBlockDown(in_channels=64, out_channels=128, stride=2, groups=32)
        x = torch.randn(2, 64, 32, 32)

        out = block(x)

        assert out.shape == (2, 128, 16, 16)

    def test_output_shape_with_stride1(self):
        """Test output shape with stride=1 (no downsampling)."""
        block = ResBlockDown(in_channels=64, out_channels=128, stride=1, groups=32)
        x = torch.randn(2, 64, 32, 32)

        out = block(x)

        assert out.shape == (2, 128, 32, 32)

    def test_identity_shortcut(self):
        """Test identity shortcut when channels match and stride=1."""
        block = ResBlockDown(in_channels=64, out_channels=64, stride=1, groups=32)

        assert isinstance(block.shortcut, nn.Identity)


# =============================================================================
# Tests for ResBlockUp
# =============================================================================

class TestResBlockUp:
    """Tests for ResBlockUp module."""

    def test_output_shape_pixelshuffle(self):
        """Test output shape with pixelshuffle upsampling."""
        block = ResBlockUp(in_channels=128, out_channels=64, up_mode='pixelshuffle', groups=32)
        x = torch.randn(2, 128, 16, 16)

        out = block(x)

        assert out.shape == (2, 64, 32, 32)

    def test_output_shape_nearest(self):
        """Test output shape with nearest neighbor upsampling."""
        block = ResBlockUp(in_channels=128, out_channels=64, up_mode='nearest', groups=32)
        x = torch.randn(2, 128, 16, 16)

        out = block(x)

        assert out.shape == (2, 64, 32, 32)


# =============================================================================
# Tests for ConvEncoder
# =============================================================================

class TestConvEncoder:
    """Tests for ConvEncoder module."""

    def test_output_shape(self, img_params, latent_dim, batch_size):
        """Test encoder output shape."""
        encoder = ConvEncoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            latent_dim=latent_dim,
            in_channels=1,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        img = torch.randn(batch_size, 1, img_params["img_h"], img_params["img_w"])
        out = encoder(img)

        assert "mu" in out
        assert "log_var" in out
        assert out["mu"].shape == (batch_size, latent_dim)
        assert out["log_var"].shape == (batch_size, latent_dim)

    def test_with_condition(self, img_params, latent_dim, batch_size):
        """Test encoder with conditioning."""
        condition_dim = 10
        encoder = ConvEncoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            latent_dim=latent_dim,
            in_channels=1,
            condition_dim=condition_dim,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        img = torch.randn(batch_size, 1, img_params["img_h"], img_params["img_w"])
        condition = torch.randn(batch_size, condition_dim)

        out = encoder(img, condition)

        assert out["mu"].shape == (batch_size, latent_dim)
        assert out["log_var"].shape == (batch_size, latent_dim)

    def test_logvar_clamped(self, img_params, latent_dim):
        """Test that log_var is clamped."""
        encoder = ConvEncoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            latent_dim=latent_dim,
            in_channels=1,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        # Large input to potentially produce extreme values
        img = torch.randn(4, 1, img_params["img_h"], img_params["img_w"]) * 100
        out = encoder(img)

        assert out["log_var"].min() >= -10
        assert out["log_var"].max() <= 10


# =============================================================================
# Tests for ConvDecoder
# =============================================================================

class TestConvDecoder:
    """Tests for ConvDecoder module."""

    def test_output_shape(self, img_params, latent_dim, batch_size):
        """Test decoder output shape."""
        # First create encoder to get spatial_shape
        encoder = ConvEncoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            latent_dim=latent_dim,
            in_channels=1,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        decoder = ConvDecoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            spatial_shape=encoder.spatial_shape,
            latent_dim=latent_dim,
            out_channels=1,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        z = torch.randn(batch_size, latent_dim)
        img = decoder(z)

        assert img.shape == (batch_size, 1, img_params["img_h"], img_params["img_w"])

    def test_with_condition(self, img_params, latent_dim, batch_size):
        """Test decoder with conditioning."""
        condition_dim = 10

        encoder = ConvEncoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            latent_dim=latent_dim,
            in_channels=1,
            condition_dim=condition_dim,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        decoder = ConvDecoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            spatial_shape=encoder.spatial_shape,
            latent_dim=latent_dim,
            out_channels=1,
            condition_dim=condition_dim,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        z = torch.randn(batch_size, latent_dim)
        condition = torch.randn(batch_size, condition_dim)

        img = decoder(z, condition)

        assert img.shape == (batch_size, 1, img_params["img_h"], img_params["img_w"])

    def test_output_positive(self, img_params, latent_dim, batch_size):
        """Test that output is positive (due to Softplus)."""
        encoder = ConvEncoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            latent_dim=latent_dim,
            in_channels=1,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        decoder = ConvDecoder(
            img_h=img_params["img_h"],
            img_w=img_params["img_w"],
            spatial_shape=encoder.spatial_shape,
            latent_dim=latent_dim,
            out_channels=1,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        z = torch.randn(batch_size, latent_dim)
        img = decoder(z)

        assert (img > 0).all()


# =============================================================================
# Tests for ConvVAE
# =============================================================================

class TestConvVAE:
    """Tests for ConvVAE module."""

    def test_forward_output_keys(self, small_px_ind, batch_size):
        """Test that forward returns expected keys."""
        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        n_features = len(small_px_ind)
        x = torch.randn(batch_size, n_features)

        out = vae(x)

        expected_keys = {"reconstruction", "latent", "mu", "log_var", "reconstructed_img"}
        assert set(out.keys()) == expected_keys

    def test_forward_output_shapes(self, small_px_ind, batch_size):
        """Test that forward outputs have correct shapes."""
        latent_dim = 16
        n_features = len(small_px_ind)

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=latent_dim,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        x = torch.randn(batch_size, n_features)
        out = vae(x)

        assert out["reconstruction"].shape == (batch_size, n_features)
        assert out["latent"].shape == (batch_size, latent_dim)
        assert out["mu"].shape == (batch_size, latent_dim)
        assert out["log_var"].shape == (batch_size, latent_dim)
        assert out["reconstructed_img"].shape == (batch_size, 1, 16, 16)

    def test_forward_with_condition(self, small_px_ind, batch_size):
        """Test forward pass with conditioning."""
        condition_dim = 8
        n_features = len(small_px_ind)

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            condition_dim=condition_dim,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        x = torch.randn(batch_size, n_features)
        condition = torch.randn(batch_size, condition_dim)

        out = vae(x, condition)

        assert out["reconstruction"].shape == (batch_size, n_features)

    def test_training_vs_eval_mode(self, small_px_ind, batch_size):
        """Test that reparameterization differs in train vs eval mode."""
        n_features = len(small_px_ind)

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        x = torch.randn(batch_size, n_features)

        # In eval mode, latent should equal mu
        vae.eval()
        with torch.no_grad():
            out_eval = vae(x)

        torch.testing.assert_close(out_eval["latent"], out_eval["mu"])

    def test_sample(self, small_px_ind):
        """Test sample generation."""
        n_features = len(small_px_ind)
        num_samples = 4

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        samples = vae.sample(num_samples)

        assert samples.shape == (num_samples, n_features)

    def test_sample_with_condition(self, small_px_ind):
        """Test sample generation with conditioning."""
        n_features = len(small_px_ind)
        num_samples = 4
        condition_dim = 8

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            condition_dim=condition_dim,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        condition = torch.randn(num_samples, condition_dim)
        samples = vae.sample(num_samples, condition=condition)

        assert samples.shape == (num_samples, n_features)

    def test_loss_computation(self, small_px_ind, batch_size):
        """Test loss computation."""
        n_features = len(small_px_ind)

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        x = torch.randn(batch_size, n_features)
        out = vae(x)
        loss = vae.loss(x, out, beta=1.0)

        assert "total" in loss
        assert "reconstruction" in loss
        assert "kl" in loss

        # All losses should be scalars
        assert loss["total"].ndim == 0
        assert loss["reconstruction"].ndim == 0
        assert loss["kl"].ndim == 0

        # Total should equal reconstruction + kl
        torch.testing.assert_close(
            loss["total"],
            loss["reconstruction"] + loss["kl"]
        )

    def test_loss_with_beta(self, small_px_ind, batch_size):
        """Test loss computation with different beta values."""
        n_features = len(small_px_ind)

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        x = torch.randn(batch_size, n_features)
        out = vae(x)

        loss_beta0 = vae.loss(x, out, beta=0.0)
        loss_beta1 = vae.loss(x, out, beta=1.0)
        loss_beta2 = vae.loss(x, out, beta=2.0)

        # With beta=0, total should equal reconstruction
        torch.testing.assert_close(loss_beta0["total"], loss_beta0["reconstruction"])

        # Total should increase with beta (assuming KL > 0)
        assert loss_beta2["total"] >= loss_beta1["total"]

    def test_gradient_flow(self, small_px_ind, batch_size):
        """Test that gradients flow properly through the model."""
        n_features = len(small_px_ind)

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        x = torch.randn(batch_size, n_features)
        out = vae(x)
        loss = vae.loss(x, out)

        loss["total"].backward()

        # Check that gradients exist for encoder and decoder parameters
        for name, param in vae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_reconstruction_positive(self, small_px_ind, batch_size):
        """Test that reconstruction is positive (due to Softplus in decoder)."""
        n_features = len(small_px_ind)

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        x = torch.randn(batch_size, n_features)
        out = vae(x)

        assert (out["reconstruction"] > 0).all()


# =============================================================================
# Integration Tests
# =============================================================================

class TestConvVAEIntegration:
    """Integration tests for ConvVAE."""

    def test_training_step(self, small_px_ind):
        """Test a complete training step."""
        n_features = len(small_px_ind)
        batch_size = 8

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Training step
        vae.train()
        x = torch.randn(batch_size, n_features)

        optimizer.zero_grad()
        out = vae(x)
        loss = vae.loss(x, out)
        loss["total"].backward()
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(loss["total"])

    def test_multiple_training_steps(self, small_px_ind):
        """Test that loss decreases over multiple training steps."""
        n_features = len(small_px_ind)
        batch_size = 16

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        )

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Fixed data for consistent training
        torch.manual_seed(42)
        x = torch.rand(batch_size, n_features) + 0.1  # Positive data

        losses = []
        vae.train()

        for _ in range(20):
            optimizer.zero_grad()
            out = vae(x)
            loss = vae.loss(x, out, beta=0.1)
            loss["total"].backward()
            optimizer.step()
            losses.append(loss["total"].item())

        # Loss should generally decrease
        assert losses[-1] < losses[0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, small_px_ind):
        """Test that model works on CUDA."""
        n_features = len(small_px_ind)
        batch_size = 4

        vae = ConvVAE(
            px_ind=small_px_ind,
            img_h=16,
            img_w=16,
            latent_dim=16,
            base_channels=32,
            num_res_blocks=2,
            groups=16
        ).cuda()

        x = torch.randn(batch_size, n_features).cuda()

        out = vae(x)
        loss = vae.loss(x, out)

        assert out["reconstruction"].device.type == "cuda"
        assert loss["total"].device.type == "cuda"
