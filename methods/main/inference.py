from ultralytics import YOLO
import os
import time
import logging
from tqdm import tqdm

from .utils import *
from .DataAnalyser import DataAnalyser
from .ExecutionPlanner import ExecutionPlanner
from .DataLoaderCached import DataLoaderCached
from . import PostprocHandler as PostprocHandlerLib
from .PostprocHandler import PostprocHandler
from .PolygonizationWorker import PolygonizationWorker

from osgeo import gdal, osr, ogr
import shutil
import math
import torch
from copy import deepcopy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_output_folder(folder_name):
    """Ensure the output folder exists."""
    if not os.path.exists(folder_name):
        logger.info(f"Creating output folder: {folder_name}")
        os.makedirs(folder_name)

def execute(model_paths, config, verbose):
    if verbose:
        logger.setLevel(logging.DEBUG)
        PostprocHandlerLib.logger.setLevel(logging.DEBUG)

    src_folder, temp_folder, output_path, keep_temp, mask_filepath = (
        config["execution_args"]["src_folder"],
        config["execution_args"]["temp_folder"],
        config["execution_args"]["output_path"],
        config["execution_args"]["keep_temp"],
        config["execution_args"]["mask_filepath"]
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    time_start = time.time()

    # Load models once
    logger.info("Loading the model(s)...")
    loaded_models = []
    for model_path in model_paths:
        model = YOLO(model_path).to(device)
        loaded_models.append(model)

    model_names = config["model"]
    models_map = {model_names[i]: loaded_models[i] for i in range(len(model_names))}

    create_output_folder(temp_folder)
    create_output_folder(os.path.dirname(output_path))

    all_tiffs = [os.path.join(src_folder, file) for file in os.listdir(src_folder) if (file.endswith(".tif") or file.endswith(".tiff"))]

    # Group TIFFs by CRS
    crs_groups = {}
    for tiff_path in all_tiffs:
        try:
            ds = gdal.Open(tiff_path)
            if ds is None:
                logger.warning(f"Could not open {tiff_path}. Skipping.")
                continue
            proj = ds.GetProjection()
            if proj not in crs_groups:
                crs_groups[proj] = []
            crs_groups[proj].append(tiff_path)
            ds = None
        except Exception as e:
            logger.error(f"Error reading CRS from {tiff_path}: {e}")

    output_basename, output_ext = os.path.splitext(output_path)
    group_index = 1

    if not crs_groups:
        logger.warning("No valid TIFF files found in the input directory.")
        return

    for crs, tiffs in crs_groups.items():
        logger.info(f"Processing group {group_index}/{len(crs_groups)}: {len(tiffs)} TIFFs with the same CRS.")

        analyser = DataAnalyser(tiffs, config["data_loader"]["bands"], config["super_resolution"])
        if not analyser.isCompatible():
            logger.error(f"Incompatible tiff files in a group. This should not happen. Skipping group.")
            continue

        logger.info("Estimating normalization bounds...")
        analyser.calcNormalizationBounds()

        current_config = deepcopy(config)
        current_config["data_loader"]["min"] = analyser.min
        current_config["data_loader"]["max"] = analyser.max

        planner = ExecutionPlanner(analyser, current_config["execution_planner"])

        lclu_mask_path = warp_lclu(mask_filepath, os.path.join(temp_folder, os.path.basename(src_folder) + f"_{group_index}.lclu.tif"),
                                   tiffs[0], analyser.total_bounds, [analyser.pixel_size_x, analyser.pixel_size_y],
                                   ["BIGTIFF=YES", "COMPRESS=ZSTD", "ZSTD_LEVEL=2", "TILED=YES", "NUM_THREADS=ALL_CPUS"])

        # Modify output path for each group
        current_output_path = f"{output_basename}_{group_index}{output_ext}" if len(crs_groups) > 1 else output_path

        gpkg_path, layer_name = create_geopackage_with_same_projection(
                current_output_path, current_config["polygonization_args"]["layer_name"], analyser.projection,
                override_if_exists=current_config["polygonization_args"]["override_if_exists"]
            )

        time_delineate_start = time.time()
        logger.info(f"Starting delineation for group {group_index}...")
        execute_delineation(models_map, planner, current_config["postprocess_limits"], current_config["passes"], current_config["data_loader"],
                            (gpkg_path, layer_name), lclu_mask_path, current_config["mask_info"], current_config, device)

        logger.info(f"Group {group_index} delineated in {time.time() - time_delineate_start:.2f} seconds.")

        postdelineation_merge((gpkg_path, layer_name), current_config["filtering_args"])

        group_index += 1

    if not keep_temp:
        folder_content = os.listdir(temp_folder)
        for file in folder_content:
            if os.path.basename(src_folder) in file and ".lclu.tif" in file:
                 os.remove(os.path.join(temp_folder, file))

        if not os.listdir(temp_folder):
            shutil.rmtree(temp_folder)
        logger.info("Temporary files have been deleted.")

    logger.info(f"Execution finished in {time.time() - time_start:.2f} seconds.")

def execute_delineation(models, planner, postproc_config, passes, dataloader_config, layer_info, lclu_path, lclu_config, full_config, device):
    gpkg = ogr.Open(layer_info[0], 1)
    out_layer = gpkg.GetLayerByName(layer_info[1])
    srs = out_layer.GetSpatialRef()

    postproc_handler = PostprocHandler(planner.region_size, postproc_config, srs.ExportToWkt(), full_config["filtering_args"])

    global_field_counter = 2
    field_counter_increment = 2

    counter_history_array = [{} for _ in passes]
    region_counter = 0

    total_dataloader_time = 0
    dataloader = None
    num_regions = planner.get_num_regions()
    with tqdm(total=num_regions, desc="Delineating", unit="region") as pbar_delineate:
        pbar_delineate.n = region_counter
        pbar_delineate.refresh()

        while planner.move_to_next_region():
            start_start = time.time()
            for pass_id in range(len(passes)):
                pass_args = passes[pass_id].copy()
                pass_args["delineation_config"]["region_offset"] = planner.current_region

                postproc_handler.set_postproc_config(pass_args["delineation_config"])

                tileSize = pass_args["tile_size"]
                tileStep = pass_args["tile_step"]
                batchSize = pass_args["batch_size"]

                plan = planner.get_plan(tileSize, tileStep)
                for entry in plan:
                    start = time.time()

                    is_compatible = dataloader is not None and dataloader.is_compatible(entry)
                    if dataloader is None or not is_compatible:
                        dataloader = DataLoaderCached(entry, dataloader_config, batchSize, lclu_path, lclu_config)
                        dt = time.time() - start
                        logger.debug(f"Data loader load data in: {dt} s.")
                        total_dataloader_time += dt
                    else:
                        dataloader.set_plan(entry)
                    
                    while True:
                        images_batch, nodata_batch, bounds_batch = dataloader.get_batch()

                        if images_batch is None:
                            break
                        model_results = []
                        for model_args in pass_args["model_args"]:
                            model = models[model_args["name"]]
                            prediction = model.predict(images_batch, conf=model_args["minimal_confidence"], half=model_args["use_half"], verbose=False)
                            model_results.append(prediction)

                        # push each batch item into queue
                        for i in range(len(images_batch)):
                            lbound = bounds_batch[i]
                            tile_key = tuple([lbound["filename"], lbound["infile"]])

                            counter_history = counter_history_array[pass_id]
                            id_counter = global_field_counter
                            if tile_key in counter_history:
                                id_counter = counter_history[tile_key]
                            else:
                                counter_history[tile_key] = id_counter
                                for result in model_results:
                                    if result[i].masks is not None:
                                        global_field_counter += field_counter_increment * len(result[i].masks)

                            args = ([results[i].cpu() for results in model_results], nodata_batch[i], bounds_batch[i], id_counter)
                            postproc_handler.put(args)
                        
                        if device == 'cuda':
                            torch.cuda.empty_cache()

                    postproc_handler.sync()

            end = time.time()
            logger.debug(f"Pass ended in: {end - start_start} s.")

            postproc_handler.map()
            postproc_handler.polygonize(planner.get_geotransform(), layer_info)
            postproc_handler.clear()

            region_counter += 1

            pbar_delineate.n = region_counter
            
            pbar_delineate.refresh()

    postproc_handler.dispose()
    logger.debug(f"Total time on creating dataloader: {total_dataloader_time} s.")

def postdelineation_merge(layer_info, filter_config):
    gpkg_path, layer_name = layer_info
    gpkg = ogr.Open(gpkg_path, 1)
    layer = gpkg.GetLayerByName(layer_name)

    MIN_AREA = filter_config["minimum_area_m2"]
    MIN_HOLE_AREA = filter_config["minimum_hole_area_m2"]

    try:
        layer.StartTransaction()

        # creating transform to equali-area projection
        src_srs = layer.GetSpatialRef()
        dst_src = osr.SpatialReference()
        dst_src.ImportFromEPSG(6933)

        transform = osr.CoordinateTransformation(src_srs, dst_src)

        # filtering polygons, and collect polygons what require merge
        field_parts = {}
        features_to_delete = []

        for feature in tqdm(layer, desc="Filtering", unit="poly"):
            fid = feature.GetFID()
            id = feature.GetField("id")
            if id < 0:
                continue

            orig_geom = feature.GetGeometryRef().Clone()

            if id in field_parts:
                field_parts[id].append(orig_geom)
            else:
                field_parts[id] = [orig_geom]

            features_to_delete.append(fid)
                
        # delete useless features
        for fid in tqdm(features_to_delete, desc="Deleting", unit="poly"):
            layer.DeleteFeature(fid)

        out_feat = ogr.Feature(layer.GetLayerDefn())
        # merge features and add them to the layer
        for id in tqdm(field_parts.keys(), desc="Merging", unit="poly"):
            cleaned_geoms = [g.Buffer(0) for g in field_parts[id] if g and not g.IsEmpty()]
            if not cleaned_geoms:
                logger.debug(f"Skipping id={id}: no valid geometries after cleaning")
                continue

            # Perform manual union (pairwise)
            merged = cleaned_geoms[0]
            for g in cleaned_geoms[1:]:
                merged = merged.Union(g)

            if merged is None or merged.IsEmpty():
                logger.debug(f"Skipping id={id}: union failed or returned empty")
                continue

            # Now decompose MultiPolygon into individual Polygon features
            geom_type = merged.GetGeometryType()
            if geom_type == ogr.wkbPolygon:
                parts = [merged]
            elif geom_type == ogr.wkbMultiPolygon:
                parts = [merged.GetGeometryRef(i).Clone() for i in range(merged.GetGeometryCount())]
            else:
                raise RuntimeError(f"Unexpected geometry type: {merged.GetGeometryName()}")

            # Create one feature per polygon part
            for part in parts:
                geom, area = PolygonizationWorker.remove_holes(part, MIN_HOLE_AREA, transform)
                if area < MIN_AREA:
                    continue

                out_feat.SetFID(-1)
                out_feat.SetGeometry(geom)
                out_feat.SetField("id", id)
                out_feat.SetField("area", float(area))
                layer.CreateFeature(out_feat)

        out_feat = None

        layer.CommitTransaction()
    except Exception as e:
        layer.RollbackTransaction()
        raise e

def warp_lclu(src, dst, sample_tiff, total_bounds, pixel_size, warp_options):
    temp_lclu_tiff_path = None

    mask_path = src
    if mask_path is not None:
        temp_lclu_tiff_path = dst

        if os.path.exists(dst):
            return temp_lclu_tiff_path

        sample_raster = gdal.Open(sample_tiff)
        target_proj = sample_raster.GetProjection()

        minx, miny, maxx, maxy = total_bounds

        cols = int(math.ceil((maxx - minx) / pixel_size[0]))
        rows = int(math.ceil((maxy - miny) / abs(pixel_size[1])))

        pbar = tqdm(total=100, desc="Warping LCLU", unit="%")

        def warping_progress_callback(complete, message, unknown):
            pbar.n = int(complete * 100)
            pbar.refresh()
            return 1

        # Perform the warp
        gdal.Warp(
            dst,
            mask_path,
            format='GTiff',
            dstSRS=target_proj,
            outputBounds=total_bounds,
            width=cols,
            height=rows,
            resampleAlg='nearest',
            creationOptions=warp_options,
            callback=warping_progress_callback,
        )

    return temp_lclu_tiff_path