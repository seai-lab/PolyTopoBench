import random

import skimage.draw
import skimage.io
from descartes import PolygonPatch
from matplotlib.collections import PatchCollection
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np
import torch
import shapely.geometry

from lydorn_utils import math_utils
from torch_lydorn import torchvision

VERTEX_FILL_COLOR = (1.0, 1.0, 1.0, 1.0)
VERTEX_OUTLINE_COLOR = (0.0, 0.0, 0.0, 1.0)
HOLE_EDGE_COLOR = (1.0, 220.0 / 255.0, 0.0, 1.0)


def get_seg_display(seg):
    dtype = seg.dtype
    seg_display = np.zeros([seg.shape[0], seg.shape[1], 4], dtype=dtype)
    if len(seg.shape) == 2:
        seg_display[..., 0] = seg
        seg_display[..., 3] = seg
    else:
        for i in range(seg.shape[-1]):
            seg_display[..., i] = seg[..., i]
        clip_max = 255 if dtype == np.uint8 else 1
        seg_display[..., 3] = np.clip(np.sum(seg, axis=-1), 0, clip_max)
    return seg_display


def get_tensorboard_image_seg_display(image, seg, crossfield=None):
    assert len(image.shape) == 4 and image.shape[1] == 3, f"image should be (N, 3, H, W), not {image.shape}."
    assert len(seg.shape) == 4 and seg.shape[1] <= 3, f"image should be (N, C<=3, H, W), not {seg.shape}."
    assert image.shape[0] == seg.shape[0], "image and seg should have the same batch size."
    assert image.shape[2] == seg.shape[2], "image and seg should have the same image height."
    assert image.shape[3] == seg.shape[3], "image and seg should have the same image width."
    if crossfield is not None:
        assert len(crossfield.shape) == 4 and crossfield.shape[
            1] == 4, f"crossfield should be (N, 4, H, W), not {crossfield.shape}."
        assert image.shape[0] == crossfield.shape[0], "image and crossfield should have the same batch size."
        assert image.shape[2] == crossfield.shape[2], "image and crossfield should have the same image height."
        assert image.shape[3] == crossfield.shape[3], "image and crossfield should have the same image width."

    alpha = torch.clamp(torch.sum(seg, dim=1, keepdim=True), 0, 1)

    # Add missing seg channels
    seg_display = torch.zeros_like(image)
    seg_display[:, :seg.shape[1], ...] = seg

    image_seg_display = (1 - alpha) * image + alpha * seg_display
    image_seg_display = image_seg_display.cpu()

    if crossfield is not None:
        np_crossfield = crossfield.cpu().detach().numpy().transpose(0, 2, 3, 1)
        image_plot_crossfield_list = [get_image_plot_crossfield(_crossfield, crossfield_stride=10) for _crossfield in
                                      np_crossfield]
        image_plot_crossfield_list = [torchvision.transforms.functional.to_tensor(image_plot_crossfield).float() / 255
                                      for image_plot_crossfield in image_plot_crossfield_list]
        image_plot_crossfield = torch.stack(image_plot_crossfield_list, dim=0)
        alpha = image_plot_crossfield[:, 3:4, :, :]
        image_seg_display = (1 - alpha) * image_seg_display + alpha * image_plot_crossfield[:, :3, :, :]
        # image_seg_display = image_plot_crossfield[:, :3, :, :]

    return image_seg_display


def plot_crossfield(axis, crossfield, crossfield_stride, alpha=0.5, width=0.5, add_scale=1, invert_y=True):
    x = np.arange(0, crossfield.shape[1], crossfield_stride)
    y = np.arange(0, crossfield.shape[0], crossfield_stride)
    x, y = np.meshgrid(x, y)
    i = y
    if invert_y:
        i = crossfield.shape[0] - 1 - y
    j = x
    scale = add_scale * 1 / crossfield_stride

    c0c2 = crossfield[i, j, :]
    u, v = math_utils.compute_crossfield_uv(c0c2)

    # u_angle = 0.5
    # u.real = np.cos(u_angle)
    # u.imag = np.sin(u_angle)
    # v *= 0

    quiveropts = dict(color=(0, 0, 1, alpha), headaxislength=0, headlength=0, pivot='middle', angles="xy", units='xy',
                      scale=scale, width=width, headwidth=1)
    axis.quiver(x, y, u.imag, -u.real, **quiveropts)
    axis.quiver(x, y, v.imag, -v.real, **quiveropts)


def get_image_plot_crossfield(crossfield, crossfield_stride):
    fig = Figure(figsize=(crossfield.shape[1] / 100, crossfield.shape[0] / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()

    plot_crossfield(ax, crossfield, crossfield_stride, alpha=1.0, width=2.0, add_scale=1)

    ax.axis('off')
    fig.tight_layout(pad=0)
    # To remove the huge white borders
    ax.margins(0)

    canvas.draw()
    image_from_plot = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(canvas.get_width_height()[::-1] + (4,))
    image_from_plot = np.roll(image_from_plot, -1, axis=-1)  # Convert ARGB to RGBA

    # Fix alpha (white to alpha)
    # mask = np.sum(image_from_plot[:, :, :3], axis=2) == 3*255
    # image_from_plot[mask, 3] = 0
    mini = image_from_plot.min()
    image_from_plot[:, :, 3] = np.max(255 - image_from_plot[:, :, :3] + mini, axis=2)

    return image_from_plot


def plot_polygons(axis, polygons, polygon_probs=None, draw_vertices=True, linewidths=2, markersize=10, alpha=0.2,
                  color_choices=None):
    if len(polygons) == 0:
        return
    random.seed(1)
    if color_choices is None:
        color_choices = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0.5, 1, 0, 1],
            [1, 0.5, 0, 1],
            [0.5, 0, 1, 1],
            [1, 0, 0.5, 1],
            [0, 0.5, 1, 1],
            [0, 1, 0.5, 1],
        ]
    colors = random.choices(color_choices, k=len(polygons))

    def plot_ring(coords, color):
        coords = np.asarray(coords)
        if coords.ndim != 2 or coords.shape[0] == 0:
            return
        axis.plot(
            coords[:, 0],
            coords[:, 1],
            color=color,
            linewidth=linewidths,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        if draw_vertices:
            # HiSup-like vertex style: black outline + white fill.
            unique_coords = coords[:-1] if 1 < len(coords) else coords
            if len(unique_coords):
                outline_size = float((markersize + 3) ** 2)
                fill_size = float(max(markersize, 1) ** 2)
                axis.scatter(
                    unique_coords[:, 0],
                    unique_coords[:, 1],
                    s=outline_size,
                    c=[VERTEX_OUTLINE_COLOR],
                    linewidths=0,
                    zorder=5,
                )
                axis.scatter(
                    unique_coords[:, 0],
                    unique_coords[:, 1],
                    s=fill_size,
                    c=[VERTEX_FILL_COLOR],
                    linewidths=0,
                    zorder=6,
                )

    for i, geometry in enumerate(polygons):
        polygon = geometry if isinstance(geometry, shapely.geometry.Polygon) else shapely.geometry.Polygon(geometry)
        if polygon.is_empty:
            continue
        color = colors[i]
        plot_ring(np.array(polygon.exterior.coords), color)
        for interior in polygon.interiors:
            plot_ring(np.array(interior.coords), HOLE_EDGE_COLOR)


def overlay_polygon_fills(image, polygons, alpha=0.2, color_choices=None):
    if len(polygons) == 0:
        return image

    random.seed(1)
    if color_choices is None:
        color_choices = [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [0.5, 1, 0, 1],
            [1, 0.5, 0, 1],
            [0.5, 0, 1, 1],
            [1, 0, 0.5, 1],
            [0, 0.5, 1, 1],
            [0, 1, 0.5, 1],
        ]
    colors = random.choices(color_choices, k=len(polygons))

    image_array = np.asarray(image)
    image_is_float = np.issubdtype(image_array.dtype, np.floating)
    base = image_array.astype(np.float32, copy=True)
    color_scale = 1.0 if image_is_float and base.max() <= 1.0 else 255.0
    height, width = base.shape[:2]

    for i, geometry in enumerate(polygons):
        polygon = geometry if isinstance(geometry, shapely.geometry.Polygon) else shapely.geometry.Polygon(geometry)
        if polygon.is_empty:
            continue

        mask = np.zeros((height, width), dtype=bool)
        exterior = np.asarray(polygon.exterior.coords)
        if exterior.ndim != 2 or exterior.shape[0] < 3:
            continue
        rr, cc = skimage.draw.polygon(exterior[:, 1], exterior[:, 0], shape=(height, width))
        mask[rr, cc] = True
        for interior in polygon.interiors:
            coords = np.asarray(interior.coords)
            if coords.ndim != 2 or coords.shape[0] < 3:
                continue
            rr, cc = skimage.draw.polygon(coords[:, 1], coords[:, 0], shape=(height, width))
            mask[rr, cc] = False

        color = np.array(colors[i][:3], dtype=np.float32) * color_scale
        base[mask] = (1.0 - alpha) * base[mask] + alpha * color

    if image_is_float:
        return base.astype(image_array.dtype, copy=False)
    return np.clip(base, 0, 255).astype(image_array.dtype, copy=False)


def plot_line_strings(axis, line_strings, draw_vertices=True, linewidths=2, markersize=5):
    artists = []
    marker = "o" if draw_vertices else None
    for line_string in line_strings:
        artist, = axis.plot(*line_string.xy, marker=marker, markersize=markersize)
        artists.append(artist)
    return artists


def plot_geometries(axis, geometries, draw_vertices=True, linewidths=2, markersize=3):
    polygons = []
    line_strings = []
    for geometry in geometries:
        if isinstance(geometry, shapely.geometry.Polygon):
            polygons.append(geometry)
        elif isinstance(geometry, shapely.geometry.LineString):
            line_strings.append(geometry)
        elif isinstance(geometry, shapely.geometry.MultiLineString):
            for line_string in geometry:
                line_strings.append(line_string)
        else:
            raise NotImplementedError(f"Geometry type {type(geometry)} not implemented")

    if len(polygons):
        plot_polygons(axis, polygons, draw_vertices=draw_vertices, linewidths=linewidths, markersize=markersize)

    if len(line_strings):
        artists = plot_line_strings(axis, line_strings, draw_vertices=draw_vertices, linewidths=linewidths, markersize=markersize)
        return artists


def save_poly_viz(image, polygons, out_filepath, linewidths=2, markersize=2, alpha=0.2, draw_vertices=True,
                  corners=None, crossfield=None, polygon_probs=None, seg=None, color_choices=None, dpi=100):
    assert isinstance(polygons, list), f"polygons should be of type list, not {type(polygons)}"
    if len(polygons):
        assert (type(polygons[0]) == np.ndarray or type(polygons[0]) == shapely.geometry.Polygon), \
            f"Item of the polygons list should be of type ndarray or shapely Polygon, not {type(polygons[0])}"
    if polygon_probs is not None:
        assert type(polygon_probs) == list
        assert len(polygons) == len(polygon_probs), \
            "len(polygons)={} should be equal to len(polygon_probs)={}".format(len(polygons), len(polygon_probs))
    # Setup plot
    height = image.shape[0]
    width = image.shape[1]
    # Keep the exported PNG at the same pixel dimensions as the input patch.
    figure_dpi = 100
    f, axis = plt.subplots(1, 1, figsize=(width / figure_dpi, height / figure_dpi), dpi=figure_dpi)

    display_image = overlay_polygon_fills(image, polygons, alpha=alpha, color_choices=color_choices)
    axis.imshow(display_image)

    if seg is not None:
        seg *= 0.9
        axis.imshow(seg)

    if crossfield is not None:
        plot_crossfield(axis, crossfield, crossfield_stride=1, alpha=0.5, width=0.1, add_scale=1.1, invert_y=False)

    plot_polygons(axis, polygons, polygon_probs=polygon_probs, draw_vertices=draw_vertices, linewidths=linewidths,
                  markersize=markersize, alpha=alpha, color_choices=color_choices)

    if corners is not None and len(corners):
        assert len(corners[0].shape) == 2
        for corner_array in corners:
            plt.plot(corner_array[:, 0], corner_array[:, 1], marker="o", linewidth=0, markersize=20, color="red")

    axis.autoscale(False)
    axis.axis('equal')
    axis.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins
    plt.savefig(out_filepath, transparent=False, dpi=dpi)
    plt.close()


def main():
    image = torch.zeros((2, 3, 512, 512)) + 0.5
    seg = torch.zeros((2, 2, 512, 512))
    seg[:, 0, 100:200, 100:200] = 1
    crossfield = torch.zeros((2, 4, 512, 512))
    # u_angle = np.random.random(10000) * np.pi
    # v_angle = np.random.random(10000) * np.pi
    u_angle = 0.25
    v_angle = u_angle + np.pi / 2
    u = np.cos(u_angle) + 1j * np.sin(u_angle)
    v = np.cos(v_angle) + 1j * np.sin(v_angle)
    c0 = np.power(u, 2) * np.power(v, 2)
    c2 = - (np.power(u, 2) + np.power(v, 2))
    # print("c0:")
    # print(np.abs(c0).min(), np.abs(c0).mean(), np.abs(c0).max())
    # print(c0.real.min(), c0.real.mean(), c0.real.mean())
    # print(c0.imag.min(), c0.imag.mean(), c0.imag.max())
    # print("c2:")
    # print(np.abs(c2).min(), np.abs(c2).mean(), np.abs(c2).max())
    # print(c2.real.min(), c2.real.mean(), c2.real.max())
    # print(c2.real.min(), c2.imag.mean(), c2.imag.max())

    crossfield[:, 0, :, :] = c0.real
    crossfield[:, 1, :, :] = c0.imag
    crossfield[:, 2, :, :] = c2.real
    crossfield[:, 3, :, :] = c2.imag

    image_seg_display = get_tensorboard_image_seg_display(image, seg, crossfield=crossfield)
    image_seg_display = image_seg_display.cpu().numpy().transpose(0, 2, 3, 1)
    skimage.io.imsave("image_seg_display.png", image_seg_display[0])


if __name__ == "__main__":
    main()
