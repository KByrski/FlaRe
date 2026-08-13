"""Geometry interchange helpers for editing FlaRe primitives as PLY meshes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


SIDES = 8
CHI_SQUARED_RADIUS = 11.3449


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Exact rotation part of the historical qs2M helper."""
    a, b, c, d = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    s_val = 2.0 / (aa + bb + cc + dd)
    bs, cs, ds = b * s_val, c * s_val, d * s_val
    ab, ac, ad = a * bs, a * cs, a * ds
    bb, bc, bd = bb * s_val, b * cs, b * ds
    cc, cd, dd = cc * s_val, c * ds, dd * s_val

    q11, q12, q13 = 1.0 - cc - dd, bc - ad, bd + ac
    q21, q22, q23 = bc + ad, 1.0 - bb - dd, cd - ab
    q31, q32, q33 = bd - ac, cd + ab, 1.0 - bb - cc
    u = torch.cat((q11, q21, q31), 1).unsqueeze(2)
    v = torch.cat((q12, q22, q32), 1).unsqueeze(2)
    n = torch.cat((q13, q23, q33), 1).unsqueeze(2)
    return torch.cat((u, v, n), 2)


def qs2matrix(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Exact historical qs2M(q, s), including column-wise scaling."""
    return quaternion_to_matrix(q) * s.unsqueeze(1)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Batch version of the Davenport/SVD conversion used in the edit snippet."""
    qxx, qxy, qxz = matrix[:, 0:1, 0:1], matrix[:, 0:1, 1:2], matrix[:, 0:1, 2:3]
    qyx, qyy, qyz = matrix[:, 1:2, 0:1], matrix[:, 1:2, 1:2], matrix[:, 1:2, 2:3]
    qzx, qzy, qzz = matrix[:, 2:3, 0:1], matrix[:, 2:3, 1:2], matrix[:, 2:3, 2:3]
    rows = (
        torch.cat((qxx - qyy - qzz, qyx + qxy, qzx + qxz, qzy - qyz), 2),
        torch.cat((qyx + qxy, qyy - qxx - qzz, qzy + qyz, qxz - qzx), 2),
        torch.cat((qzx + qxz, qzy + qyz, qzz - qxx - qyy, qyx - qxy), 2),
        torch.cat((qzy - qyz, qxz - qzx, qyx - qxy, qxx + qyy + qzz), 2),
    )
    vectors, _, _ = torch.linalg.svd(torch.cat(rows, 1))
    xyzw = vectors[:, :, 0]
    return torch.cat((xyzw[:, 3:4], xyzw[:, :3]), 1).contiguous()


def opacity_scale(a: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Return the renderer's opacity/kappa-dependent radial scale."""
    kappa = 1.0 + torch.nn.functional.softplus(k)
    opacity = torch.sigmoid(a).clamp_min(torch.finfo(a.dtype).tiny)
    power = torch.clamp(kappa * (CHI_SQUARED_RADIUS + 2.0 * torch.log(opacity)), min=0.0)
    return power.pow(1.0 / (2.0 * kappa))


def primitives_to_vertices(
    m: torch.Tensor, s: torch.Tensor, q: torch.Tensor, a: torch.Tensor, k: torch.Tensor
) -> torch.Tensor:
    """Generate the historical eight-point editing proxy."""
    angles = torch.arange(SIDES, dtype=m.dtype, device=m.device) * (2.0 * torch.pi / SIDES)
    circle = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), 1)
    s_clamping = opacity_scale(a, k)
    # TODO: version the PLY convention and migrate new exports to the full
    # Delta-s radius used by SetGeometry and the supplementary material. The
    # historical edit branch used sqrt(Delta-s), which is retained here so its
    # existing PLY assets remain compatible.
    s_transformed = torch.sqrt(s_clamping) * torch.exp(s)
    s_expanded = torch.cat(
        (s_transformed, torch.zeros_like(s_transformed[:, :1])), 1
    )
    matrix = qs2matrix(q, s_expanded)
    x = circle.unsqueeze(0).repeat(m.shape[0], 1, 1).unsqueeze(3)
    return (matrix.unsqueeze(1) @ x).squeeze(3) + m.unsqueeze(1)


def vertices_to_primitives(
    vertices: torch.Tensor,
    a: torch.Tensor,
    k: torch.Tensor,
    fallback_s: torch.Tensor | None = None,
    fallback_q: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit edited eight-vertex polygons back to FlaRe m, log-scales and q."""
    if vertices.ndim != 3 or vertices.shape[1:] != (SIDES, 3):
        raise ValueError(f"Expected [N, {SIDES}, 3] vertices, got {tuple(vertices.shape)}")
    mean = vertices.mean(1, keepdim=True)
    centered = vertices - mean

    covariance = centered.transpose(1, 2) @ centered
    covariance = covariance / SIDES
    basis, _, _ = torch.linalg.svd(covariance)
    basis_u, basis_v = basis[:, :, 0], basis[:, :, 1]
    plane_normal = torch.cross(basis_u, basis_v, dim=1)

    # SVD axes have arbitrary signs and may be swapped. Align them with the
    # corresponding canonical proxy corners so an unchanged edit preserves the
    # learned local radiance-field frame while retaining the SVD geometry fit.
    texture_u = centered[:, 0]
    texture_v = centered[:, SIDES // 4]
    texture_u = texture_u - plane_normal * (texture_u * plane_normal).sum(1, keepdim=True)
    texture_v = texture_v - plane_normal * (texture_v * plane_normal).sum(1, keepdim=True)
    tiny = torch.finfo(vertices.dtype).tiny
    texture_u = texture_u / torch.linalg.vector_norm(texture_u, dim=1, keepdim=True).clamp_min(tiny)
    texture_v = texture_v / torch.linalg.vector_norm(texture_v, dim=1, keepdim=True).clamp_min(tiny)
    direct_score = (basis_u * texture_u).sum(1).abs() + (basis_v * texture_v).sum(1).abs()
    swapped_score = (basis_v * texture_u).sum(1).abs() + (basis_u * texture_v).sum(1).abs()
    swap = (swapped_score > direct_score).unsqueeze(1)
    u = torch.where(swap, basis_v, basis_u)
    v = torch.where(swap, basis_u, basis_v)
    u = torch.where((u * texture_u).sum(1, keepdim=True) < 0, -u, u)
    v = torch.where((v * texture_v).sum(1, keepdim=True) < 0, -v, v)
    normal = torch.cross(u, v, dim=1)
    rotation = torch.stack((u, v, normal), 2)

    points = centered.unsqueeze(3)
    u_projection = u.unsqueeze(1).unsqueeze(3).transpose(2, 3) @ points
    u_extent = torch.max(torch.abs(u_projection), 1, keepdim=True)[0].squeeze(3).squeeze(2)
    v_projection = v.unsqueeze(1).unsqueeze(3).transpose(2, 3) @ points
    v_extent = torch.max(torch.abs(v_projection), 1, keepdim=True)[0].squeeze(3).squeeze(2)

    # TODO: apply the per-primitive 2x2 texture-coordinate correction matrix A
    # described in the supplementary material for sheared/non-rigid edits.

    # Keep the inverse of the historical sqrt(Delta-s) export convention until
    # PLY convention versioning is available; see primitives_to_vertices().
    divisor = torch.sqrt(opacity_scale(a, k))
    extents = torch.cat((u_extent, v_extent), 1)
    invalid = (divisor <= 0).squeeze(1) | (extents <= 0).any(1)
    if torch.any(invalid) and (fallback_s is None or fallback_q is None):
        count = int(invalid.sum().item())
        raise ValueError(f"Cannot recover {count} clipped or collapsed primitive(s)")
    scales = extents / divisor.clamp_min(torch.finfo(vertices.dtype).tiny)
    log_scales = torch.log(scales.clamp_min(torch.finfo(vertices.dtype).tiny))
    quaternion = matrix_to_quaternion(rotation)
    if torch.any(invalid):
        log_scales[invalid] = fallback_s[invalid]
        quaternion[invalid] = fallback_q[invalid]
    return mean[:, 0].contiguous(), log_scales.contiguous(), quaternion.contiguous()


def deform_vertices(
    vertices: torch.Tensor,
    deformation: str | None,
    amplitude: float = 0.1,
    frequency: float = 8.0,
    rotation_z_degrees: float = 0.0,
    phase_shift_degrees: float = 0.0,
) -> torch.Tensor:
    """Rotate around world Z, then apply an optional point-wise deformation."""
    if deformation is None:
        return vertices
    if deformation in ("sin", "sin2"):
        angle = torch.as_tensor(
            rotation_z_degrees * (torch.pi / 180.0),
            dtype=vertices.dtype,
            device=vertices.device,
        )
        cosine, sine = torch.cos(angle), torch.sin(angle)
        phase = torch.as_tensor(
            phase_shift_degrees * (torch.pi / 180.0),
            dtype=vertices.dtype,
            device=vertices.device,
        )
        result = torch.empty_like(vertices)
        result[:, :, 0] = cosine * vertices[:, :, 0] - sine * vertices[:, :, 1]
        result[:, :, 1] = sine * vertices[:, :, 0] + cosine * vertices[:, :, 1]
        result[:, :, 2] = vertices[:, :, 2]
        result[:, :, 2] += amplitude * torch.sin(
            frequency * result[:, :, 0] + phase
        )
        if deformation == "sin2":
            result[:, :, 2] += amplitude * torch.sin(
                frequency * result[:, :, 1] + phase
            )
        return result
    raise ValueError(f"Unknown deformation: {deformation}")


def write_edit_ply(path: Path, vertices: torch.Tensor) -> None:
    """Write polygons plus IDs; IDs allow importers to reorder vertices safely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = vertices.detach().cpu().numpy().astype(np.float32, copy=False)
    count = xyz.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("primitive_id", "i4"), ("corner_id", "i4")]
    records = np.empty(count * SIDES, dtype=dtype)
    records["x"], records["y"], records["z"] = xyz.reshape(-1, 3).T
    records["primitive_id"] = np.repeat(np.arange(count, dtype=np.int32), SIDES)
    records["corner_id"] = np.tile(np.arange(SIDES, dtype=np.int32), count)
    faces = np.empty(count, dtype=[("vertex_indices", "i4", (SIDES,))])
    faces["vertex_indices"] = np.arange(count * SIDES, dtype=np.int32).reshape(count, SIDES)
    PlyData((PlyElement.describe(records, "vertex"), PlyElement.describe(faces, "face")), text=False).write(str(path))


def read_edit_ply(path: Path, expected_primitives: int) -> torch.Tensor:
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    xyz = np.stack((vertex["x"], vertex["y"], vertex["z"]), 1).astype(np.float32)
    expected_vertices = expected_primitives * SIDES
    if len(xyz) != expected_vertices:
        raise ValueError(f"{path}: expected {expected_vertices} vertices, found {len(xyz)}")
    names = {prop.name for prop in vertex.properties}
    if {"primitive_id", "corner_id"} <= names:
        primitive_raw = np.asarray(vertex["primitive_id"])
        corner_raw = np.asarray(vertex["corner_id"])
        if not np.all(primitive_raw == primitive_raw.astype(np.int64)) or not np.all(
            corner_raw == corner_raw.astype(np.int64)
        ):
            raise ValueError(f"{path}: primitive_id/corner_id contains non-integers")
        primitive = primitive_raw.astype(np.int64)
        corner = corner_raw.astype(np.int64)
        if np.any(primitive < 0) or np.any(primitive >= expected_primitives) or np.any(corner < 0) or np.any(corner >= SIDES):
            raise ValueError(f"{path}: primitive_id/corner_id is out of range")
        key = primitive * SIDES + corner
        if len(np.unique(key)) != expected_vertices:
            raise ValueError(f"{path}: duplicate or missing primitive_id/corner_id pairs")
        ordered = np.empty_like(xyz)
        ordered[key] = xyz
        xyz = ordered
    elif "face" in {element.name for element in ply.elements}:
        faces = ply["face"].data["vertex_indices"]
        if len(faces) == expected_primitives and all(
            len(face) == SIDES for face in faces
        ):
            indices = np.stack(faces).astype(np.int64, copy=False)
            if np.any(indices < 0) or np.any(indices >= expected_vertices):
                raise ValueError(f"{path}: face contains an invalid vertex index")
            xyz = xyz[indices].reshape(expected_vertices, 3)
    return torch.from_numpy(xyz.reshape(expected_primitives, SIDES, 3))
