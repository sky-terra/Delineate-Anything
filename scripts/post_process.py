from numpy.lib.stride_tricks import sliding_window_view
import shapely
import sys

import pandas as pd
import shapely.geometry as shg
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from pathlib import Path

from shapely.plotting import plot_polygon
from shapelysmooth import taubin_smooth
from simplification.cutil import simplify_coords_vwp,simplify_coords_vw
from sympy.integrals.manualintegrate import orthogonal_poly_rule

project_folder = Path(__file__).resolve().parents[1]
data_folder = project_folder / 'data'


def calculate_orthogonal_ratio_weighted(row: pd.Series, angle_tolerance: float = 1):
    # geom = row.geometry.simplify(5)
    xy = np.stack(row.geometry.exterior.coords.xy, -1)
    xy_simplified = simplify_coords_vwp(xy, 25)
    geom = shg.Polygon(xy_simplified)
    if geom.area < 300000:
        return 0
    """Calculate ratio of orthogonal edge lengths to total perimeter"""
    coords = list(geom.exterior.coords[:-1])  # Exclude closing point
    if len(coords) < 3:
        return 0

    orthogonal_length = 0
    total_perimeter = geom.length
    debug = row.id == 19124
    if debug:
        # plot_polygon(row.geometry, alpha=0.5)
        plot_polygon(geom, alpha=0.5)
        plt.show()
    for i in range(len(coords)):
        p1 = coords[i]
        p2 = coords[(i + 1) % len(coords)]

        # Calculate edge length
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        edge_length = np.sqrt(dx * dx + dy * dy)

        # Calculate bearing angle
        angle = np.degrees(np.arctan2(dy, dx)) % 180
        if debug:
            print(
                f'Edge {i}: {p1} -> {p2}, angle: {angle}, ratio: {edge_length / total_perimeter:.3f}')
        # Check alignment to cardinal/intercardinal directions
        for target_angle in [0, 90]:
            if abs(angle % 90 - target_angle) < angle_tolerance:
                # if debug:
                #     print('edge_length:', edge_length)

                if edge_length < 60:
                    continue
                elif edge_length / total_perimeter > 0.1:
                    return 1
                orthogonal_length += edge_length
                break
    ratio = orthogonal_length / total_perimeter if total_perimeter > 0 else 0
    if debug:
        # plot_polygon(row.geometry, alpha=0.5)
        print(orthogonal_length)
        print(total_perimeter)
        print(ratio)

    return ratio


def calculate_slopes(points1, points2, handle_vertical='inf'):
    """
    Calculate slopes with proper handling of edge cases.

    Parameters:
    points1, points2: array-like, shape (n_points, 2)
    handle_vertical: 'inf', 'nan', or 'large' for vertical lines
    """
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)

    dx = points2[:, 0] - points1[:, 0]
    dy = points2[:, 1] - points1[:, 1]

    # Initialize slopes array
    slopes = np.full(len(points1), np.nan)

    # Normal cases (non-vertical lines)
    valid_mask = dx != 0
    slopes[valid_mask] = dy[valid_mask] / dx[valid_mask]

    # Vertical lines (dx = 0, dy ≠ 0)
    vertical_mask = (dx == 0) & (dy != 0)
    if handle_vertical == 'inf':
        slopes[vertical_mask] = np.sign(dy[vertical_mask]) * np.inf

    return slopes


def calculate_orthogonal_ratio_weighted2(row: pd.Series, angle_tolerance: float = 1.0):
    debug = row.id == -37378
    geom = row.geometry
    # perimeter = geom.simplify(10, preserve_topology=True).length
    xy = np.stack(geom.exterior.coords.xy, -1)
    # res = np.concatenate([xy[:-1], xy[1:]], axis=-1)
    # print(res)
    # print("len(res):", len(res))
    p2, p1 = xy[1:], xy[:-1]
    lengths = np.linalg.norm(p2 - p1, axis=-1)

    score = lengths * (lengths > 100) * np.log(geom.area)
    if debug:
        print(score)
        plot_polygon(geom, alpha=0.5)
        plt.show()
    # return int(np.log(score.sum() or 1) * 100)
    return score.sum()

    # slopes = (p2[:, 1] - p1[:, 1]) / (p2[:, 0] - p1[:, 0])
    # angles_degrees = np.degrees(np.arctan(np.abs(calculate_slopes(p1, p2))))
    # print(angles_degrees)


def filter_by_bounding_box_touching(row: pd.Series):
    # xy = np.stack(row.geometry.exterior.coords.xy, -1)
    # xy_simplified = simplify_coords_vwp(xy, 25)
    # geom = shg.Polygon(xy_simplified)
    geom_envelope = shapely.envelope(row.geometry)
    threshold_length = geom_envelope.length
    result = row.geometry.exterior.intersection(geom_envelope.exterior)
    match result:
        case shg.MultiLineString() | shg.LineString():
            return result.length / threshold_length
        case shg.MultiPoint() | shg.Point():
            return 0
        case shg.GeometryCollection(geoms=geoms):
            return sum(
                [x.length / threshold_length for x in geoms if
                 isinstance(x, shg.LineString)])
        case _:
            raise NotImplementedError(result)


def windowed_mean_sliding_view(arr, window_size):
    windows = sliding_window_view(arr, window_shape=window_size)
    return np.count_nonzero(windows > 100, axis=-1)


def detect_right_triangles(row: pd.Series):
    geom = row.geometry
    xy = np.stack(geom.exterior.coords.xy, -1)
    p2, p1 = xy[1:], xy[:-1]
    lengths = np.linalg.norm(p2 - p1, axis=-1)
    window_lengths = np.concatenate([lengths, lengths[:4]])
    windows = sliding_window_view(window_lengths, window_shape=4)
    is_corner = (np.all(windows[:, 1:3] == 5, axis=1)
                 & (np.all(windows[:, [0, -1]] > 150, axis=1)))
    if not np.any(is_corner):
        return 0
    side_lengths = windows[is_corner][:, [0, -1]]
    areas = np.prod(side_lengths, axis=1) / 2
    score = areas.max() / geom.area
    # debug = row.id == -284
    # if debug:
    #     print(side_lengths)
    #     print(areas)
    #     print(geom.area)
    #     print(score)
    #     plot_polygon(geom, alpha=0.5)
    #     plt.show()
    return score


def detect_orthogonal_lines(row: pd.Series):
    geom = row.geometry
    debug = row.id == 13760
    xy = np.stack(geom.exterior.coords.xy, -1)
    p2, p1 = xy[1:], xy[:-1]
    lengths = np.linalg.norm(p2 - p1, axis=-1)
    if np.any(lengths > 1000):
        return 1
    perimeter = lengths.sum()
    # score = np.count_nonzero(lengths > 500)

    window_lengths = np.concatenate([lengths, lengths[:4]])
    windows = sliding_window_view(window_lengths, window_shape=4)
    score = (np.all(windows[:, 1:3] == 5, axis=1)
             & (np.all(windows[:, [0, -1]] > 300, axis=1))).sum()
    if score > 0:
        return score
    windows = sliding_window_view(window_lengths, window_shape=2)
    score = np.all(windows > 300, axis=1).sum()

    # two_consecutively_long_edges = (
    #         np.count_nonzero(windows > perimeter / 10, axis=-1) >= 2).sum()
    # if two_consecutively_long_edges and geom.area > 500_000:
    #     return 1
    # lengths = windowed_mean_sliding_view(lengths, 4) * 4
    # angles_degrees = np.degrees(np.arctan(np.abs(calculate_slopes(p1, p2))))
    # mask = (((angles_degrees == 0) & (lengths > 100))
    #         | ((angles_degrees == 90) & (lengths > 100)))
    # mask = lengths > 100
    # score = lengths[mask].sum() / perimeter
    # if debug:
    #     area0 = geom.area
    #     print('len(xy)', len(xy))
    #     xy = simplify_coords_vwp(xy, len(xy) / 4)
    #     print('len(xy)', len(xy))
    #
    #     geom = shg.Polygon(xy)
    #     area1 = geom.area
    #     print('area_reduction', (area0 - area1) / area0 * 100, '%')
    #     plot_polygon(geom, alpha=0.5)
    #     # print(lengths)
    #     # print(score)
    #     # print(windows.shape)
    #     # plot_polygon(geom, alpha=0.5)
    #     plt.show()
    return score


def simplify_geoms(row: pd.Series):
    debug = row.id == -17918
    geom = row.geometry

    # if debug:
    #     plot_polygon(geom, alpha=0.5, label='Original')
    #     plt.legend()
    #     plt.show()

    geom = geom.simplify(3, preserve_topology=True)
    # if debug:
    #     plot_polygon(geom, alpha=0.5, label='Simplified')
    #
    #     plt.legend()
    #     plt.show()
    geom = taubin_smooth(geom, steps=5)
    # if debug:
    #     plot_polygon(geom, alpha=0.5, label='Taubin smoothed')
    #
    #     plt.legend()
    #     plt.show()
    xy_orig = np.stack(geom.exterior.coords.xy, -1)

    xy = simplify_coords_vw(xy_orig, 300)
    try:
        geom = shg.Polygon(xy)
    except ValueError:
        print(xy_orig)
        print(xy)
        plot_polygon(geom, alpha=0.5)
        plt.show()
        return geom
    if debug:
        plot_polygon(geom, alpha=0.5, label='VWP')
        plt.legend()
        plt.show()
    # debug = row.id == 13760
    # if debug:
    #     area0 = geom.area
    #     # print('len(xy)', len(xy))
    #     # xy = simplify_coords_vwp(xy, len(xy) / 4)
    #     # print('len(xy)', len(xy))
    #
    #     # geom = shg.Polygon(xy)
    #     # area1 = geom.area
    #     # print('area_reduction', (area0 - area1) / area0 * 100, '%')
    #     plot_polygon(geom, alpha=0.5)
    #     # print(lengths)
    #     # print(score)
    #     # print(windows.shape)
    #     # plot_polygon(geom, alpha=0.5)
    #     plt.show()
    return geom


def remove_artifacts(data: gpd.GeoDataFrame):
    # data['score'] = 0
    data = data[data.area > 30000].copy()
    data['touches_bbox'] = data.apply(filter_by_bounding_box_touching, axis=1)
    data = data[(data['touches_bbox'] < 0.75) & (data.area < 4.5e6)].copy()
    # data['orthogonality'] = 0
    # mask = data.area > 1e6
    data['orthogonality'] = data.apply(detect_orthogonal_lines, axis=1)
    data = data[data.orthogonality == 0].copy()
    data['is_triangular'] = data.apply(detect_right_triangles, axis=1)
    data = data[data['is_triangular'] < 0.55].copy()

    # data['score'] = data['touches_bbox']
    # data.loc[data['touches_bbox'] > 0.25, 'score'] = 1

    # data['score'] = ((data['orthogonality'] < 0.95)
    #                  & (data['touches_bbox'] < 0.75))
    # # data = data[data['ortho_touch']].copy()
    # data.loc[data.area < 1_000_000, 'score'] = True
    # data.loc[data['orthogonality'] > 1, 'score'] = False
    # data.loc[data['touches_bbox'] > 0.35, 'score'] = False
    # data.loc[data.area < 300_000, 'score'] = True
    # data.loc[data['touches_bbox'] > 0.4, 'score'] = False
    data['geometry'] = data.apply(simplify_geoms, axis=1)
    data.area=data.area

    # data['ortho_weighted'] = data['orthogonality'] * np.sqrt(data.area)
    # data = data[data['ortho_w2eighted'] < 1000].copy()
    # data['score'] = 0

    # data['is_ok'] = data['orthogonality'] < 0.48
    # data.apply(detect_right_triangles, axis=1)
    # data['product_score'] = data['touches_bbox'] * 100*data['orthogonality'] * np.sqrt(data.area)
    # data['score'] *= np.log2(data.area)
    # data['score'] = data.apply(calculate_orthogonal_ratio_weighted2, axis=1)
    # data['score'] = np.clip(data['score'], 0, 3/storage/skyterra/kz/epsg_326380)
    return data


pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 40)


def main():
    path = data_folder / 'delineated/epsg_32639.GPKG'
    data = gpd.read_file(path)
    data = data.pipe(remove_artifacts)
    print(data)
    out_path = path.with_stem(path.stem + '_filtered_simple')
    out_path.unlink(missing_ok=True)
    data.to_file(out_path, driver='GPKG')
    pass


if __name__ == '__main__':
    main()
