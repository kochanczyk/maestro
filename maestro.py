#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,c-extension-no-member

# Licensed under the GPLv3: https://www.gnu.org/licenses/gpl-3.0.html
# Copyright (c) 2023-2024 Marek Kochańczyk, Paweł Nałęcz-Jawecki, Frederic Grabowski

'''
This module provides command-line utilities to manage and en masse process images generated
by Harmony, software featuring Operetta CLS, a high-throughput microplate imager.

Commands
--------

show
    inspect contents (intented uses are: show channels, show wells, show settings)

check
    search for single-pixel images and images devoid of respective entries in metadata

remaster
    apply flat-field correction, stitch corrected tiles, generate intensity-adjusted pseudo-colored
    multi-channel overlays and individual channels in grayscale (collectively termed "remixes"),
    and encode movies, all according to a given config file (for an example config file,
    see 'example_config.yaml')

trackerabilize
    create folders with ShuttleTracker-compatibile views of "remixes" folders obtained during
    remastering (using the above-described command 'remaster')
    [ShuttleTracker: https://pmbm.ippt.pan.pl/software/shuttletracker]

folderize
    move images to respective individual plate well-specific folders

archivize
    move images to individual well-specific folders & archive/compress the folders

Type 'python maestro.py COMMAND --help' to learn about COMMAND's arguments and options.
'''

import sys
import platform
import shutil
import string
import re
import random
from itertools import product
from collections import Counter
from time import sleep
from datetime import timedelta
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Literal
import subprocess as sp
from multiprocessing import Process, Queue
import tarfile

import psutil
import yaml
import click
import pandas as pd
import numpy as np
from tqdm import tqdm
import tifffile
import cv2

from operetta import OPERETTA_IMAGES_SUBFOLDER_NAME
from operetta import OPERETTA_IMAGES_INDEX_FILE_NAME
from operetta import OPERETTA_IMAGE_NAME_RE
from operetta import OPERETTA_IMAGE_NAME_EXTENSION
from operetta import OPERETTA_OBSERVABLE_COLOR_RE
from operetta import OPERETTA_EXTRA_TILE_OVERLAP
from operetta import WellLocation
from operetta import assemble_image_file_name
from operetta import get_image_files_paths
from operetta import determine_timepoints_count
from operetta import determine_fields_count
from operetta import determine_fields_layout
from operetta import field_number_to_xy
from operetta import determine_channels_count
from operetta import extract_plate_layout
from operetta import extract_wells_info
from operetta import extract_channels_info
from operetta import extract_image_acquisition_settings
from operetta import extract_time_interval
from operetta import extract_and_derive_images_layout

COLOR_COMPONENTS = {
    'red':     'R',
    'green':   'G',
    'blue':    'B',
    'cyan':    'GB',
    'magenta': 'RB',
    'yellow':  'RG',
    'gray':    'RGB',
    'grey':    'RGB',
}

WELL_PREFIX_NAME = 'well'

CORRTILES_FOLDER_BASE_NAME = 'Tiles'
STITCHES_FOLDER_BASE_NAME = 'Stitches'
REMIXES_FOLDER_BASE_NAME = 'Remixes'
MOVIES_FOLDER_BASE_NAME = 'Movies'
SHUTTLETRACKER_FOLDER_BASE_NAME = 'ST'

TIMESTAMP_SUFFIX = '--timestamped'

CONVERT_EXE_PATH = Path('/usr/bin/convert')
MOGRIFY_EXE_PATH = Path('/usr/bin/mogrify')

XVFB_EXE_PATH = Path('/usr/bin/xvfb-run')
XVFB_SERVER_NUMS = set(range(11, 91))

FIJI_EXE_PATH = Path('/opt/fiji/Fiji.app/ImageJ-linux64')
FIJI_SCRIPT_BASE_PATH = Path('/tmp/stitch-')

TEXT_ANNOTATIONS_DEFAULT_FONT_SIZE = 96

FFMPEG_EXE_PATH = Path('/usr/bin/ffmpeg')
FFMPEG_OPTS_SILENCE = '-nostdin -loglevel error -y'.split()
FFMPEG_OPTS_CODEC = '-vcodec libx264 -pix_fmt yuv420p -an'.split()  # -b:v 24M
FFMPEG_OPTS_FILTERS = ''.split()  # '-vf deshake'.split()  # -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2"
FFMPEG_OPTS_QUALITY = '-preset medium -crf 28 -tune fastdecode'.split()

MOVIE_DEFAULT_REALTIME_SPEEDUP = 3600  # 1m of live experiment => 1h of movie

CLICK_EXISTING_FILE_PATH_TYPE   = click.Path(exists=True, file_okay=True,  dir_okay=False)
CLICK_EXISTING_FOLDER_PATH_TYPE = click.Path(exists=True, file_okay=False, dir_okay=True)

TQDM_STYLE = {'bar_format': '{desc} {percentage:3.0f}% {postfix}'}

CONVERT_EXE_PATH_S, MOGRIFY_EXE_PATH_S, XVFB_EXE_PATH_S, \
FIJI_EXE_PATH_S, FIJI_SCRIPT_BASE_PATH_S, FFMPEG_EXE_PATH_S = \
map(lambda p: str(p.absolute()), [
CONVERT_EXE_PATH  , MOGRIFY_EXE_PATH,   XVFB_EXE_PATH  , \
FIJI_EXE_PATH  , FIJI_SCRIPT_BASE_PATH  , FFMPEG_EXE_PATH
])

assert CONVERT_EXE_PATH.exists()
assert MOGRIFY_EXE_PATH.exists()
assert FFMPEG_EXE_PATH.exists()
assert XVFB_EXE_PATH.exists()
assert FIJI_EXE_PATH.exists()


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)



def _group_paths_by(
    files_paths: List[Path],
    chunk_re: str,
) -> List[Tuple[str, List[Path]]]:

    assert '<groupby>' in chunk_re
    groupby_re = re.compile(chunk_re)

    grouped_files_paths: Dict[str, List[Path]] = {}
    for file_path in files_paths:
        groupby_search = groupby_re.search(file_path.name)
        assert groupby_search is not None
        by = groupby_search.group('groupby')
        if by not in grouped_files_paths:
            grouped_files_paths[by] = []
        grouped_files_paths[by].append(file_path)
        grouped_files_paths[by] = sorted(grouped_files_paths[by])

    return list(sorted(grouped_files_paths.items(), key=lambda _: _[0]))



def _prepare_flatfield_correction_images(
    channels_info: pd.DataFrame,
    image_shape: tuple,
) -> dict:

    assert image_shape[0]*image_shape[1] > 1, "A stub of an image encountered."

    ffc_profile_images = {}
    for channel_number in channels_info.index:

        flatfield_correction_params = channels_info.loc[channel_number]['Correction']

        if flatfield_correction_params['Character'] == 'Null':
            ffc_profile_image = np.ones(shape=image_shape)
        else:
            assert flatfield_correction_params['Character'] == 'NonFlat'

            background_profile = flatfield_correction_params['Profile']
            (width, height), (origin_x, origin_y), (scale_x, scale_y), coeffs_triangle = [
                background_profile[s] for s in ('Dims', 'Origin', 'Scale', 'Coefficients')
            ]

            yy, xx = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
            xx = (xx - origin_x) * scale_x
            yy = (yy - origin_y) * scale_y

            ffc_profile_image = np.zeros(shape=image_shape)
            for coeffs_row in coeffs_triangle:
                for j, coeff in enumerate(coeffs_row):
                    i = len(coeffs_row) - 1 - j
                    ffc_profile_image += coeff * xx**i * yy**j


        ffc_profile_images[channel_number] = ffc_profile_image

    return ffc_profile_images



def _flatfield_correct(
    orig_image_file_path: Path,
    corr_image_file_path: Path,
    profile_image: np.ndarray,
    image_shape: tuple,
    image_dtype: str,
) -> None:
    tiff16 = cv2.imread(str(orig_image_file_path.absolute()), cv2.IMREAD_UNCHANGED)
    if tiff16 is None:
        # Re-try ('permission denied' may happen occasionally through an overloaded SMB connection).
        sleep(2 + random.choice(range(4)))
        tiff16 = cv2.imread(str(orig_image_file_path.absolute()), cv2.IMREAD_UNCHANGED)
        if tiff16 is None:
            print(f"Error reading image file '{str(orig_image_file_path.absolute())}'!")
            tiff16 = np.zeros(shape=image_shape, dtype=image_dtype)
    assert tiff16.dtype == 'uint16', \
        "Depth of an image subjected to flat-field correction is other than 16 bit!"
    bit_depth = 16

    ffced_image = tiff16 / profile_image
    assert ffced_image.dtype == 'float64'
    ffced_image = np.clip(ffced_image, 0, 2**bit_depth - 1)
    cv2.imwrite(str(corr_image_file_path.absolute()), ffced_image.astype(np.uint16))



def correct_tiles(
    well_id: str,
    well_loc: WellLocation,
    channels_info: pd.DataFrame,
    images_layout: pd.DataFrame,
    well_orig_image_files_paths: list,
    tiles_folder_path: Path,
    force: bool = False,
    apply_flatfield_correction: bool = True,
    clean_up_single_channel_tiles: bool = True,
    compress_output_tif: bool = False,
) -> None:
    '''
    Result: multiple single-channel images converted to a single multi-page image file.

    Arguments
    ---------
    apply_flatfield_correction:
        If False, the outcome is just a renamed symlink.
    '''

    tiles_folder_path.mkdir(exist_ok=True, parents=True)

    if apply_flatfield_correction:

        for an_image_path in well_orig_image_files_paths:
            an_image_path_str = str(an_image_path.absolute())
            an_image = cv2.imread(an_image_path_str, cv2.IMREAD_UNCHANGED)
            image_shape, image_dtype = an_image.shape, an_image.dtype

            if image_shape[0]*image_shape[1] == 1:
                print(f"Warning: Image file '{an_image_path.name}' has just one pixel!")
                continue

            ffc_profile_images = _prepare_flatfield_correction_images(channels_info, image_shape)
            break

        disclosure = ''
    else:
        disclosure = 'non-'

    well_orig_img_paths_gby_tpoint = _group_paths_by(well_orig_image_files_paths,
                                                     r'sk(?P<groupby>[0-9]+)')
    for _tpoint, tpoint_well_orig_img_paths in tqdm(
        well_orig_img_paths_gby_tpoint,
        desc=f"[{well_id}] {CORRTILES_FOLDER_BASE_NAME}: {disclosure}correcting:",
        **TQDM_STYLE
    ):
        tpoint_well_orig_img_paths_gby_field = _group_paths_by(tpoint_well_orig_img_paths,
                                                               r'(?P<groupby>f[0-9]+)')
        for _well, field_tpoint_orig_img_paths in tpoint_well_orig_img_paths_gby_field:

            # effectively, the loop below goes over all field-channels
            field_tpoint_corr_images_paths = []
            for orig_image_file_path in field_tpoint_orig_img_paths:

                row, column, channel, timepoint = map(int, [
                    image_file_name_match.group(chunk_name)
                    for chunk_name in ('row', 'column', 'channel', 'timepoint')
                    if (image_file_name_match := OPERETTA_IMAGE_NAME_RE.search(
                                                             orig_image_file_path.name)) is not None
                ])
                assert well_loc == WellLocation(row=row, column=column)

                # Note: All indices (including: timepoint indices, channel indices) become 0-based.
                field_ix, field_iy = field_number_to_xy(orig_image_file_path.name, images_layout)
                corr_image_file_name = f"Img_t{timepoint - 1:04d}" \
                                       f"_x{field_ix}_y{field_iy}" \
                                       f"_c{channel - 1:1d}"       \
                                       ".tif"
                corr_image_file_path = tiles_folder_path / corr_image_file_name
                multichannel_corr_image_file_path = corr_image_file_path.parent / re.sub('_c[0-9]+',
                                                                           '', corr_image_file_name)

                if not force and multichannel_corr_image_file_path.exists() \
                             and multichannel_corr_image_file_path.stat().st_size > 0:
                    break

                if not force and corr_image_file_path.exists() \
                             and corr_image_file_path.stat().st_size > 0:
                    continue

                if apply_flatfield_correction:
                    _flatfield_correct(orig_image_file_path, corr_image_file_path,
                                       ffc_profile_images[channel], image_shape, image_dtype)
                else:
                    if platform.system().lower() == 'linux':
                        corr_image_file_path.symlink_to(orig_image_file_path.absolute())
                    else:
                        shutil.copy(orig_image_file_path, corr_image_file_path)
                field_tpoint_corr_images_paths.append(corr_image_file_path)

            if not force and multichannel_corr_image_file_path.exists() \
                         and multichannel_corr_image_file_path.stat().st_size > 0:
                continue

            # generate a multi-page tiff file
            src_files_paths_s = [str(f.absolute()) for f in field_tpoint_corr_images_paths]
            dst_file_path_s = str(multichannel_corr_image_file_path)
            compress_args = ['-compress', 'lzw'] if compress_output_tif else ['-compress', 'none']
            cmd = [CONVERT_EXE_PATH_S, *src_files_paths_s, *compress_args, dst_file_path_s]
            sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True)

            #  if symlinks to original images or ff-corrected images are no longer needed
            if clean_up_single_channel_tiles:
                for tile_image_path in field_tpoint_corr_images_paths:
                    tile_image_path.unlink()



def _generate_fiji_stitching_script(script_file_path, **kwargs) -> None:

    # Note: 'tile_overlap' is provided, so 'compute_overlap', 'subpixel_sampling' are not requested.
    script_template = string.Template('''
        setBatchMode(true);

        function leftPad(n, width) {
            s = "" + n;
            while (lengthOf(s) < width) { s = "0" + s; }
            return s;
        }

        ts = "t" + leftPad(${TIMEPOINT}, 4);

        fileNamePattern = "Img_" + ts + "_x{x}_y{y}.tif";

        run("Grid/Collection stitching", '''
         '''"type=[Filename defined position] '''
         '''order=[Defined by filename         ] '''
         '''grid_size_x=" + d2s(${N_FIELDS_X}, 0) + " '''
         '''grid_size_y=" + d2s(${N_FIELDS_Y}, 0) + " '''
         '''tile_overlap=${TILE_OVERLAP} '''
         '''first_file_index_x=0 '''
         '''first_file_index_y=0 '''
         '''directory=[${TILES_FOLDER}] '''
         '''file_names=" + fileNamePattern + " '''
         '''output_textfile_name=stitching.txt '''
         '''fusion_method=[Linear Blending] '''
         '''regression_threshold=0.4 '''
         '''max/avg_displacement_threshold=0.5 '''
         '''absolute_displacement_threshold=0.75 '''
         '''display_fusion '''
         '''computation_parameters=[Save computation time (but use more RAM)] '''
         '''image_output=[Fuse and display]"); '''
         '''
        selectWindow("Fused");

        if (${DOWNSCALE}) {
            run("Scale...", "x=0.5 y=0.5 interpolation=Bilinear average create title=FusedDown");
            if (isOpen("Fused")) { close("Fused"); }
            selectWindow("FusedDown");
        }

        saveAs("Tiff", "${STITCHES_FOLDER}/Img_" + ts + ".tif");  // uncompressed

        if (${DOWNSCALE}) {
            close("FusedDown");
        } else {
            close("Fused");
        }

        setBatchMode(false);
        run("Quit");
    ''')

    script = script_template.substitute(kwargs)

    with open(script_file_path, 'w', encoding='UTF-8') as script_file:
        print(script, file=script_file)



def stitch_tiles(
    well_id: str,
    n_fields_xy: tuple,
    tiles_folder_path: Path,
    stitches_folder_path: Path,
    tile_overlap: float,
    force: bool = False,
    downscale: bool = False,
    compress_output_tif: bool = True,  # non-compressed better zippable externally thou
    require_no_zero_pixels_in_output_image: bool = True,
    zero_pixels_in_output_image_retries_count: int = 10,
    same_zero_pixels_counts_consecutively_for_early_exit: int = 3,
) -> None:

    assert tiles_folder_path.exists(), \
        f"Tiles folder '{tiles_folder_path.absolute()}' does not exist!"
    assert tiles_folder_path.is_dir()

    stitches_folder_path.mkdir(exist_ok=True, parents=True)

    n_fields_x, n_fields_y = n_fields_xy
    tile_images_paths = get_image_files_paths(tiles_folder_path)
    tile_images_paths_grouped_by_timepoint = _group_paths_by(tile_images_paths,
                                                             r't(?P<groupby>[0-9]{4})')

    if n_fields_x == n_fields_y == 1:
        require_no_zero_pixels_in_output_image = False
        zero_pixels_in_output_image_retries_count = 0


    desc = (
        f"[{well_id}] {STITCHES_FOLDER_BASE_NAME}: "
        f"{'fijing:' if n_fields_x*n_fields_y>1 else 'copying:'}"
    )
    for timepoint, tile_images_paths in tqdm(tile_images_paths_grouped_by_timepoint, desc=desc,
                                             **TQDM_STYLE):

        dst_file_path = stitches_folder_path / f"Img_t{int(timepoint):04d}.tif"
        if not force and dst_file_path.exists() and dst_file_path.stat().st_size > 0:
            continue

        src_tiles_folder = tile_images_paths[0].parent

        if n_fields_x == n_fields_y == 1:

            src_file_path = src_tiles_folder / f"Img_t{int(timepoint):04d}_x0_y0.tif"
            assert src_file_path.exists()
            shutil.copy(src_file_path, dst_file_path)

        else:

            n_trials_remaining = zero_pixels_in_output_image_retries_count
            zero_pixel_counts = []
            while n_trials_remaining > 0:
                n_trials_remaining -= 1

                fiji_script_path_s = FIJI_SCRIPT_BASE_PATH_S \
                        + ''.join(random.choice(string.ascii_uppercase + string.digits)
                                for _ in range(12)) + '.ij1'
                substitutions = {
                    'TIMEPOINT': timepoint,
                    'N_FIELDS_X': n_fields_x,
                    'N_FIELDS_Y': n_fields_y,
                    'TILES_FOLDER': str(src_tiles_folder.absolute()),
                    'STITCHES_FOLDER': str(stitches_folder_path.absolute()),
                    'TILE_OVERLAP': tile_overlap + OPERETTA_EXTRA_TILE_OVERLAP,
                    'DOWNSCALE': str(downscale).lower(),
                }
                _generate_fiji_stitching_script(fiji_script_path_s, **substitutions)

                xvfb_server_nums_currently_in_use = {
                    int(arg[1:])
                    for process in psutil.process_iter()
                    for arg in process.cmdline()
                    if arg.startswith(':')
                    if process.name() == 'Xvfb'
                }
                xvfb_server_num = random.choice(
                    list(XVFB_SERVER_NUMS - xvfb_server_nums_currently_in_use)
                )

                try:
                    cmd = [
                        XVFB_EXE_PATH_S, '--auto-servernum', f"--server-num={xvfb_server_num}",
                        FIJI_EXE_PATH_S, '-macro', fiji_script_path_s
                    ]
                    sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True, timeout=3600)
                except sp.TimeoutExpired:
                    print('Fiji TIMED OUT while stitching!')
                    raise
                except:
                    with open(fiji_script_path_s, 'r', encoding='utf-8') as f:
                        print(40*'- ')
                        print(f.read())
                        print(40*'- ')
                    raise

                Path(fiji_script_path_s).unlink()

                if require_no_zero_pixels_in_output_image:
                    dst_file_path_s = str(dst_file_path.absolute())
                    reading_ok, dst_img = cv2.imreadmulti(dst_file_path_s,
                                                          flags=cv2.IMREAD_UNCHANGED)
                    assert reading_ok
                    assert len(dst_img) > 0
                    if (zero_pixel_count := sum(sum(dst_img).ravel() == 0)) > 0:
                        zero_pixel_counts.append(zero_pixel_count)
                        z = same_zero_pixels_counts_consecutively_for_early_exit
                        if len(zero_pixel_counts) >= z and len(set(zero_pixel_counts[-z:])) == 1:
                            break
                        continue  # retry

        if compress_output_tif:
            dst_file_path_s = str(dst_file_path.absolute())
            cmd = [CONVERT_EXE_PATH_S, dst_file_path_s, '-compress', 'zip', dst_file_path_s]
            sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True)



def _assemble_imagemagick_crop_args(
    image_files_paths: list,
    downscale: bool
) -> List[str]:

    min_width, min_height = 2*(sys.maxsize,)
    for image_file_path in image_files_paths:
#       width, height = cv2.imread(str(image_file_path.absolute()), cv2.IMREAD_UNCHANGED).shape
        height, width = tifffile.TiffFile(image_file_path).pages[0].shape[0:2]
        min_width  = min(width, min_width)
        min_height = min(height, min_height)

    if downscale:
        min_width //= 2
        min_height //= 2

    if min_width % 2:
        min_width -= 1
    if min_height % 2:
        min_height -= 1

    # just in case
    min_width -= 2
    min_height -= 2

    return ['-crop', f"{min_width}x{min_height}+0+0"]



def _assemble_suffix(info: pd.DataFrame) -> str:

    if len(info) > 1:
        suffix = "-".join(f"{entries['Observable']}_{color}" for color, entries in info.iterrows())
    elif len(info) == 1:
        observable = info.iloc[0]['Observable']
        suffix = f"{observable}"
    else:
        assert len(info) > 0

    return suffix



def _intensity_range_arithmetics(obs_norm: str, obs_name: str) -> Tuple[str, str]:
    # 'Normalization' means manual adjustment of the dynamic range of pixel intensities.

    if pd.isna(obs_norm):
        print(f"Warning: Missing normalization for observable '{obs_name}'! Assuming -0, *1.")
        return ('0', '1.')

    normalization_re1 = re.compile(r'-(?P<sub>[0-9]+)?'
                                   r'\ *,\ *'
                                   r'\*(?P<mul>(\d+(\.\d*)?)|(\.\d+))?')
    if normalization_re1_match := normalization_re1.match(obs_norm):
        return tuple(map(normalization_re1_match.group, ['sub', 'mul']))

    normalization_re2 = re.compile(r'((?P<begin>(\d+(\.\d*)?)|(\.\d+))%)?'
                                   r'\ *...\ *'
                                   r'((?P<end>(\d+(\.\d*)?)|(\.\d+))%)?')
    if normalization_re2_match := normalization_re2.match(obs_norm):
        begin, end = map(float, map(normalization_re2_match.group, ['begin', 'end']))
        return tuple(map(str, [int((begin/100.)*(2**16 - 1)), 100./(end - begin)]))

    print("Cannot parse the normalization for observable '{obs_name}' ('obs_norm'?)! "
          "Assuming -0, *1.")
    return ('0', '1.')



def _derive_channels_composition(mixing_info: pd.DataFrame) -> dict:

    # single_channel
    if len(mixing_info) == 1 and mixing_info.index[0] == 'gray':
        tiff_page, obs_norm = mixing_info[['TiffPage', 'Normalization']].iloc[0]
        sub, mul = _intensity_range_arithmetics(obs_norm, mixing_info['Observable'].iloc[0])
        return {'gray': (tiff_page, (sub, mul))}

    # multi_channel
    composition: Dict[str, List[tuple]] = {}
    for component in 'RGB':
        composition[component] = []
        for color, (tiff_page, obs_norm) in mixing_info[['TiffPage', 'Normalization']].iterrows():
            if component in COLOR_COMPONENTS[color]:
                sub, mul = _intensity_range_arithmetics(obs_norm, color)
                component_definition = (tiff_page, (sub, mul))
                composition[component].append(component_definition)
    return composition



def remix_channels(
    well_id: str,
    mixing_info: pd.DataFrame,
    stitches_folder_path: Path,
    well_remixes_folder_path: Path,
    downscale: bool = False,
    force: bool = False,
    wells_info_for_annotation: pd.DataFrame | None = None,
    is_part_of_timeseries: bool = False,
    annotation_font_size: int = TEXT_ANNOTATIONS_DEFAULT_FONT_SIZE,
) -> None:

    assert stitches_folder_path.exists() and stitches_folder_path.is_dir()

    well_remixes_folder_path.mkdir(exist_ok=True, parents=True)

    well_stitches_images_paths = get_image_files_paths(stitches_folder_path)
    well_stitches_images_paths_grouped_by_timepoint = _group_paths_by(well_stitches_images_paths,
                                                                      r't(?P<groupby>[0-9]{4})')
    crop_args = _assemble_imagemagick_crop_args(well_stitches_images_paths, downscale)
    suffix = _assemble_suffix(mixing_info)

    composition = _derive_channels_composition(mixing_info)
    if len(composition) == 1:
        grayscale = True
        composition_s = f"(gray) ≡ ({mixing_info.iloc[0]['Observable']})"
    else:
        grayscale = False
        composition_s = ', '.join(map(lambda s: '~' if s == '' else s, [
            '+'.join([
                mixing_info[ mixing_info['TiffPage'] == cc[0] ]['Observable'].values[0]
                for cc in composition[c]
            ])
            for c in 'RGB'
        ]))
        composition_s = f"(R, G, B) ≡ ({composition_s})"


    for timepoint, stitches_image_path in tqdm(
        well_stitches_images_paths_grouped_by_timepoint,
        desc=f"[{well_id}] {REMIXES_FOLDER_BASE_NAME}: mixing {composition_s}:",
        **TQDM_STYLE
    ):
        assert len(stitches_image_path) == 1
        stitches_image_path = stitches_image_path.pop()

        rgb_overlay_file_path = well_remixes_folder_path / f"Img_t{int(timepoint):04}--{suffix}.png"

        if not force and rgb_overlay_file_path.exists() and rgb_overlay_file_path.stat().st_size >0:
            continue

        source_image_path_s = str(stitches_image_path.absolute())
        target_image_path_s = str(rgb_overlay_file_path.absolute())

        cmd = [CONVERT_EXE_PATH_S]

        if grayscale:
            tiff_page, (sub, mul) = composition['gray']
            cmd.append(f"{source_image_path_s}[{tiff_page}]")
            cmd.extend(['-colorspace', 'gray'])
            cmd.extend(['-evaluate', 'subtract', sub])
            cmd.extend(['-evaluate', 'multiply', mul])
            cmd.extend(['-set', 'colorspace', 'gray'])

        else:
            cmd.extend(['-background', 'black'])
            used_component_channels = ''
            for component in 'RGB':
                n_source_channels = len(composition[component])
                if n_source_channels == 0:
                    continue
                assert n_source_channels < 3, \
                    f"""Too many source channels for the *{
                        {'R': 'red', 'G': 'green', 'B': 'blue'}[component]
                    }* component of the overlay!"""

                used_component_channels += component

                if n_source_channels == 2:
                    cmd.append('(')

                for tiff_page, (sub, mul) in composition[component]:
                    cmd.append('(')
                    cmd.extend(['-evaluate', 'subtract', sub])
                    cmd.extend(['-evaluate', 'multiply', mul])
                    cmd.append(f"{source_image_path_s}[{tiff_page}]")
                    cmd.append(')')

                if n_source_channels == 2:
                    cmd.extend(['-compose', 'plus', '-composite'])  # IM cannot compose >2 images.
                    cmd.append(')')

            cmd.extend(['-channel', used_component_channels])
            cmd.extend(['-colorspace', 'RGB'])
            cmd.append('-combine')

        cmd.extend(['-depth', '8'])
        cmd.extend(['-alpha', 'remove', '-alpha', 'off'])

        if downscale:
            cmd.extend(['-scale', '50%'])

        cmd.extend(crop_args)

        cmd.extend('-define png:compression-filter=0 '            # Empirically determined
                   '-define png:compression-level=9 '             # to give best compression
                   '-define png:compression-strategy=2'.split())  # ratio at a reasonable speed.
        cmd.append(target_image_path_s)

        sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True)


        if not wells_info_for_annotation is None:

            well_info = wells_info_for_annotation[
                wells_info_for_annotation['WellId'] == well_id
            ]

            text = (' `\n\n' if is_part_of_timeseries else '') + '\n'.join([
                f"{k.replace('WellId', 'WELL')}:  {well_info.iloc[0].to_dict()[k]}"
                for k in well_info.columns
            ])

            cmd = [
                'convert',
                '-size', '1800x1200+0+0', 'xc:none',
                '-font', 'Helvetica', '-pointsize', str(annotation_font_size),
                '-stroke', 'gray', '-strokewidth', '2',
                '-gravity', 'north-west',
                '-annotate', '0', text,
                '-background', 'none',
                '-shadow', '1800x0+0+0', '+repage',
                '-font', 'Helvetica', '-pointsize', str(annotation_font_size),
                '-stroke', 'none', '-fill', 'LightGray',
                '-gravity', 'north-west',
                '-annotate', '0', text,
                target_image_path_s, '+swap',
                '-gravity', 'north-west',
                '-geometry', f"+{annotation_font_size//4}+{annotation_font_size//4}",
                '-composite',
                target_image_path_s,
            ]
            sp.run(cmd, check=True)



def _annotate_remixes_with_timestamps(
    well_id,
    well_remixes_folder_path: Path,
    well_movies_folder_path: Path,
    delta_t_secs: float,
    obs_suffix: str,
    force: bool = False,
    annotation_font_size: int = TEXT_ANNOTATIONS_DEFAULT_FONT_SIZE,
) -> None:

    remixes_image_files_paths = well_remixes_folder_path.glob(f"Img_t*--{obs_suffix}.png")
    desc = f"[{well_id}] Movies: annotating mix '{obs_suffix}':"

    for ti, path in enumerate(tqdm(sorted(remixes_image_files_paths), **TQDM_STYLE, desc=desc)):

        time_min_i = int(ti*delta_t_secs/60)
        time_min_s = f"{time_min_i // 60:02d}h {time_min_i % 60:02d}m"
        p_annot = well_movies_folder_path / f"{path.stem}{TIMESTAMP_SUFFIX}{path.suffix}"
        if not force and p_annot.exists():
            continue

        cmd = [
            'convert',
            '-size', '360x90', 'xc:none',
            '-gravity', 'center',
            '-font', 'Helvetica', '-pointsize', str(annotation_font_size),
            '-stroke', 'gray', '-strokewidth', '2',
            '-annotate', '0', time_min_s,
            '-background', 'none',
            '-shadow', '360x5+0+0', '+repage',
            '-font', 'Helvetica', '-pointsize', str(annotation_font_size),
            '-stroke', 'none', '-fill', 'LightGray',
            '-annotate', '0', time_min_s,
            str(path.absolute()), '+swap',
            '-gravity', 'north-west',
            '-geometry', f"+{annotation_font_size//2}+{annotation_font_size//4}",
            '-composite',
            str(p_annot.absolute())
        ]
        sp.run(cmd, check=True)



def encode_movies(
    well_id: str,
    infos: List[pd.DataFrame],
    delta_t: timedelta,
    well_remixes_folder_path: Path,
    well_movies_folder_path: Path,
    realtime_speedup: int,
    force: bool = False,
    annotate_with_timestamp: bool = True,
    annotation_font_size: int = TEXT_ANNOTATIONS_DEFAULT_FONT_SIZE,
    force_annotation: bool = False,
    clean_up_images_annotated_with_timestamp: bool = True,
) -> None:

    assert delta_t is not None
    delta_t_secs = delta_t.total_seconds()
    movie_fps_s = f"{round(realtime_speedup)}/{round(delta_t_secs)}"  # rational
    movie_fps_i = round(realtime_speedup/delta_t_secs)  # used for frequency of keyint insertion
    movie_fps_i = max(movie_fps_i, 1)

    well_movies_folder_path.mkdir(exist_ok=True, parents=True)

    for obs_suffix in map(_assemble_suffix, infos):

        movie_file_name = f"{well_id}--{obs_suffix}.mp4"
        movie_file_path = well_movies_folder_path / movie_file_name

        if not force and movie_file_path.exists() and movie_file_path.stat().st_size > 48:
            print(f"[{well_id}] {MOVIES_FOLDER_BASE_NAME}: '{movie_file_name}' exists "
                  "(encoding skipped)")
            continue

        if annotate_with_timestamp:
            _annotate_remixes_with_timestamps(well_id, well_remixes_folder_path,
                                              well_movies_folder_path, delta_t_secs, obs_suffix,
                                              force=force_annotation,
                                              annotation_font_size=annotation_font_size)
            source_image_files_folder_path = well_movies_folder_path
            input_image_filename_pattern = f"Img_t%04d--{obs_suffix}{TIMESTAMP_SUFFIX}.png"
        else:
            source_image_files_folder_path = well_remixes_folder_path
            input_image_filename_pattern = f"Img_t%04d--{obs_suffix}.png"

        ffmpeg_opts_filters = FFMPEG_OPTS_FILTERS
        example_image_path_s = str(next(source_image_files_folder_path.glob('*.png')))
        example_image_shape = cv2.imread(example_image_path_s, cv2.IMREAD_UNCHANGED).shape
        if max(example_image_shape) > 8192:
            print("Warning: Images to be encoded are very large => halving movie width and height.")
            ffmpeg_opts_filters.extend('-vf scale=trunc((iw/2)/2)*2:trunc((ih/2)/2)*2'.split())

        cmd = [
            FFMPEG_EXE_PATH_S,
            *FFMPEG_OPTS_SILENCE,
            '-r', movie_fps_s,
            '-f', 'image2',
            '-i', str((source_image_files_folder_path / input_image_filename_pattern).absolute()),
            *ffmpeg_opts_filters,
            *FFMPEG_OPTS_CODEC,
            '-x264-params', f"keyint={3*movie_fps_i}:scenecut=0",
            *FFMPEG_OPTS_QUALITY,
            str(movie_file_path.absolute())
        ]
        print(f"[{well_id}] {MOVIES_FOLDER_BASE_NAME}: encoding '{movie_file_name}' ", end='... ',
              flush=True)
        sp.run(cmd, check=True)
        print('done')

        if annotate_with_timestamp and clean_up_images_annotated_with_timestamp:
            for path in well_movies_folder_path.glob(f"Img_t*--{obs_suffix}--timestamped.png"):
                path.unlink()



def _place_image_files_in_their_respective_well_folders(
    operetta_images_folder: Path,
    abolish_internal_compression: bool = True,
    tolerate_extra_files: bool = False,
    retain_original_files: bool = False,
) -> List[Path]:

    operetta_images_folder_path = Path(operetta_images_folder)
    assert operetta_images_folder_path.exists(), \
        f"Error: The source folder '{operetta_images_folder_path.absolute()}' does not exist!"

    if operetta_images_folder_path.resolve().name != OPERETTA_IMAGES_SUBFOLDER_NAME:
        print("Error: The argument OPERETTA_IMAGES_FOLDER_PATH "
              f"should point to a folder named '{OPERETTA_IMAGES_SUBFOLDER_NAME}'!")
        return []

    # the index file is not required currently, but lack thereof is perceived suspicious
    if not (operetta_images_folder_path / OPERETTA_IMAGES_INDEX_FILE_NAME).exists():
        print(f"Error: The folder does not contain '{OPERETTA_IMAGES_INDEX_FILE_NAME}'!")
        return []

    files_paths = [
        item for item in operetta_images_folder_path.iterdir()
        if item.is_file() and not item.is_symlink()
    ]

    image_files_paths = [
        file_path for file_path in files_paths
        if OPERETTA_IMAGE_NAME_RE.match(
            re.sub('.orig$', '', file_path.name)   # accepting both '*.tiff' and '*.tiff.orig'
        )
    ]

    # the folder is impure, refuse to continue just in case
    if len(files_paths) != len(image_files_paths) + 1:
        message_type, end_mark = ('Warning', '.') if tolerate_extra_files else ('Error', '!')
        print(f"{message_type}: The source folder '{operetta_images_folder_path.absolute()}' "
              f"appears to contain some extra files{end_mark}")
        if not tolerate_extra_files:
            return []


    # -- build {(row, col): [list of all images of the well in the (row, col)]} --------------------

    def extract_row_col_(path: Path) -> tuple:
        image_name_match = OPERETTA_IMAGE_NAME_RE.search(path.name)
        assert image_name_match is not None
        return tuple(map(image_name_match.group, ['row', 'column']))

    def assemble_well_folder_name_(row: str, col: str) -> str:
        return f"{WELL_PREFIX_NAME}-{row}{col}"

    wells_images: dict = {
        extract_row_col_(image_path): []
        for image_path in image_files_paths
    }

    for image_path in image_files_paths:
        row, col = extract_row_col_(image_path)
        wells_images[(row, col)].append(image_path)


    # -- copy/move image files ---------------------------------------------------------------------

    transfer_file = shutil.copy2 if retain_original_files else shutil.move

    for (row, col), image_files in sorted(wells_images.items()):

        dest_dir_path = operetta_images_folder_path / assemble_well_folder_name_(row, col)
        if dest_dir_path.exists():
            print(f"Warning: destination folder {dest_dir_path.absolute()} already exists!")
        dest_dir_path.mkdir(exist_ok=False, parents=False)

        desc = f"{OPERETTA_IMAGES_SUBFOLDER_NAME}/[{len(image_files)} images]" \
               f" ⤚({transfer_file.__name__})→ " \
               f"{OPERETTA_IMAGES_SUBFOLDER_NAME}/{dest_dir_path.name}:"
        for image_file in tqdm(image_files, desc=desc, **TQDM_STYLE):
            assert image_file.exists()
            transfer_file(image_file, dest_dir_path)

            if abolish_internal_compression:
                image_file_path = dest_dir_path / image_file
                assert image_file_path.exists()
                compress_args = ['-compress', 'none']
                image_file_path_s = str(image_file_path.absolute())
                cmd = [MOGRIFY_EXE_PATH_S, *compress_args, image_file_path_s]
                sp.run(cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL, check=True)

    wells_folders_paths = list(sorted([
        operetta_images_folder_path / assemble_well_folder_name_(row, col)
        for row, col in wells_images
    ]))

    return wells_folders_paths



def _extract_xml_metadata(
    operetta_export_folder_path: Path,
    results: Queue,
) -> None:

    print('Info: Extracting metadata ... ', end='', flush=True)

    plate_layout: dict = extract_plate_layout(operetta_export_folder_path)
    wells_info: pd.DataFrame = extract_wells_info(operetta_export_folder_path)
    channels_info = extract_channels_info(operetta_export_folder_path)
    images_layout = extract_and_derive_images_layout(operetta_export_folder_path)
    time_interval = extract_time_interval(operetta_export_folder_path)

    print('done', flush=True)

    results.put( (plate_layout, wells_info, channels_info, images_layout, time_interval) )



def select_best_focus_planes(
    operetta_images_folder_path: Path
) -> None:
    '''A remaster's "before" operation.'''

    def __best_focused_image(
        image_paths: List[Path],
        median_filter_kernel_size: int = 3,
        laplacian_kernel_size: int = 5
    ) -> Path:

        variances: Dict[Path, float] = {}

        for path in sorted(image_paths):
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            image_denoised = cv2.medianBlur(image, median_filter_kernel_size)
            laplacian = cv2.Laplacian(image_denoised, cv2.CV_32F, laplacian_kernel_size)
            variances[path] = laplacian.var()

        best_focus_image_path = pd.Series(variances).idxmax()
        return best_focus_image_path

    # the experiment folder
    operetta_export_folder_path = Path(operetta_images_folder_path)
    assert operetta_export_folder_path.exists()

    # the source images folder within the experiment folder
    original_images_folder_path = operetta_export_folder_path / OPERETTA_IMAGES_SUBFOLDER_NAME
    assert original_images_folder_path.exists()

    # the original images folder, renamed
    source_images_folder_name = f"{OPERETTA_IMAGES_SUBFOLDER_NAME}--original"
    source_images_folder_path = operetta_export_folder_path / source_images_folder_name

    # the selection folder
    destination_images_folder_name = f"{OPERETTA_IMAGES_SUBFOLDER_NAME}--selection"
    destination_images_folder_path = operetta_export_folder_path / destination_images_folder_name
    if destination_images_folder_path.exists():
        print(f"Note: Folder '{destination_images_folder_name}' exists (skipping plane selection)")
        return

    assert not source_images_folder_path.exists(), "The source image folder already exists."

    # rename the source folder, create the destination folder
    original_images_folder_path.rename(source_images_folder_path)
    destination_images_folder_path.mkdir(exist_ok=False)

    source_image_file_paths = [
        path
        for path in source_images_folder_path.iterdir()
        if path.name.endswith(OPERETTA_IMAGE_NAME_EXTENSION)
    ]

    chunk_indices: Dict[str, set] = {
        chunk_name: set()
        for chunk_name in 'row column field plane channel timepoint'.split()
    }
    for image_path in source_image_file_paths:
        image_file_name_match = OPERETTA_IMAGE_NAME_RE.search(image_path.name)
        assert image_file_name_match is not None
        for chunk_name in chunk_indices.keys():
            chunk_index = image_file_name_match.group(chunk_name)
            chunk_indices[chunk_name].add(chunk_index)
    for name, indices in chunk_indices.items():
        print(f"Info: {name.capitalize()} indices count: {len(indices)}")

    nonplane_indices_outer_product = list(product(*[
        indices for name, indices in chunk_indices.items()
        if name not in 'plane'
    ]))
    print(f"Info: Image Z-group count: {len(nonplane_indices_outer_product)}")

    z_group_chunk_names = tuple(name for name in chunk_indices.keys() if name != 'plane')
    tqdm_desc = 'Info: Selecting best-focused images:'
    for z_group_indices in tqdm(nonplane_indices_outer_product, desc=tqdm_desc, **TQDM_STYLE):
        z_group_named_indices = dict(zip(z_group_chunk_names, z_group_indices))
        z_group_image_file_names = [
            assemble_image_file_name(**(z_group_named_indices | {'plane': p}))
            for p in chunk_indices['plane']
        ]
        z_group_image_file_paths = [
            source_images_folder_path / name
            for name in z_group_image_file_names
            if (source_images_folder_path / name).exists()
        ]
        if z_group_image_file_paths:
            best_focused_source_image_path = __best_focused_image(z_group_image_file_paths)
            target_image_name = assemble_image_file_name(**(z_group_named_indices | {'plane':'01'}))
            target_image_path = destination_images_folder_path / target_image_name
            target_image_path.symlink_to(best_focused_source_image_path.absolute())
        else:
            print(f"Warning: group {z_group_named_indices} skipped")

    print(f"Info: Symlinking: '{OPERETTA_IMAGES_SUBFOLDER_NAME}'"
          f" --> '{destination_images_folder_path.name}' ... ", end='', flush=True)

    # Images--selection/Index.idx.xml --> ../Images--original/Index.idx.xml
    (destination_images_folder_path / OPERETTA_IMAGES_INDEX_FILE_NAME).symlink_to \
        (Path('..') / source_images_folder_path.name  / OPERETTA_IMAGES_INDEX_FILE_NAME)

    # Images --> Images--selection
    Path(destination_images_folder_path.parent / OPERETTA_IMAGES_SUBFOLDER_NAME).symlink_to\
        (destination_images_folder_path.name)

    print('done')



def rename_single_timepoint_stitches_by_well_id(
    output_folder: Path,
):
    '''A remaster's "after" operation.'''

    output_folder_path = Path.cwd() if output_folder is None else Path(output_folder)

    stitches_folders = {}
    for item in output_folder_path.iterdir():
        if not item.is_dir():
            continue
        folder = item
        if folder.name.endswith(STITCHES_FOLDER_BASE_NAME):
            stitches_folder = folder
            image_files = list(stitches_folder.glob('*.tif'))
            assert len(image_files) == 1, \
                f"Folder '{stitches_folder}' should contain exactly 1 image file."
            stitches_folders[stitches_folder] = image_files.pop()

    all_stitches_folder = output_folder_path / STITCHES_FOLDER_BASE_NAME
    all_stitches_folder.mkdir(exist_ok=False)
    for stitches_folder, image in stitches_folders.items():
        well_id = stitches_folder.name.replace(f"-{STITCHES_FOLDER_BASE_NAME}", "")
        symlink = all_stitches_folder / f"{well_id}.tif"
        symlink.symlink_to(image)
        print(f"{symlink.parent.name}/{symlink.name} --> {image.parent.name}/{image.name}")



def rename_single_timepoint_remixes_by_well_id(
    output_folder: Path,
):
    '''A remaster's "after" operation.'''

    output_folder_path = Path.cwd() if output_folder is None else Path(output_folder)

    remixes_folders = {}
    for item in output_folder_path.iterdir():
        if not item.is_dir():
            continue
        folder = item
        if folder.name.endswith(REMIXES_FOLDER_BASE_NAME):
            remixes_folder = folder
            image_files = list(remixes_folder.glob('*.png'))
            remixes_folders[remixes_folder] = image_files

    all_remixes_folder = output_folder_path / REMIXES_FOLDER_BASE_NAME
    all_remixes_folder.mkdir(exist_ok=False)
    for remixes_folder, image_files in remixes_folders.items():
        well_id = remixes_folder.name.replace(f"-{REMIXES_FOLDER_BASE_NAME}", "")
        for image_file in image_files:
            symlink_name = image_file.name.replace('Img_t0000', well_id)
            symlink = all_remixes_folder / symlink_name
            symlink.symlink_to(Path('..') / image_file.parent / image_file.name)
            print(f"{symlink.parent.name}/{symlink.name} --> "
                  f"{image_file.parent.name}/{image_file.name}")



def show_wells(
    operetta_export_folder_path: Path,
    simple: bool,
) -> None:

    wells_info = extract_wells_info(operetta_export_folder_path)
    wells_info_str= '\n' .join(wells_info['WellId']) if simple else \
        wells_info.to_string(max_cols=100)
    print(wells_info_str)



def show_channels(
    operetta_export_folder_path: Path,
    simple: bool,
) -> None:

    channels_info = extract_channels_info(operetta_export_folder_path)
    channels_info_str = '\n'.join(channels_info['Name']) if simple else \
        channels_info.to_string(max_colwidth=48)
    print(channels_info_str)



def show_settings(
    operetta_export_folder_path: Path,
    simple: bool,
) -> None:

    general_settings_info, channel_settings_info = \
        extract_image_acquisition_settings(operetta_export_folder_path)

    if not simple:
        channels_info = extract_channels_info(operetta_export_folder_path)
        channel_settings_info = channel_settings_info.join(channels_info['Name'])
        channel_settings_info = channel_settings_info[[
            'Name', 'Excitation [nm]', 'Emission [nm]', 'Exposure [s]'
        ]]

    general_settings_info_str = general_settings_info.to_string(max_colwidth=24, header=False)
    channel_settings_info_str = channel_settings_info.to_string(max_colwidth=24)
    print(general_settings_info_str)
    print(channel_settings_info_str)



# -------------------------------------- user interface --------------------------------------------

@click.group()
def commands():
    pass



@click.command()
@click.argument('operetta_export_folder', type=CLICK_EXISTING_FOLDER_PATH_TYPE)
@click.argument('config_file', type=CLICK_EXISTING_FILE_PATH_TYPE)
@click.argument('output_root_folder', type=click.Path(), default=Path.cwd())
def trackerabilize(
    operetta_export_folder: Path,
    config_file: Path,
    output_root_folder: Path,
) -> None:
    '''
    Create ShuttleTracker-compatible folders.

    Single-channel images from "remixes" (obtained by running command 'remaster')
    are symlinked to their destination paths.
    '''

    single_channel_remix_re = re.compile(r'Img_t[0-9]{4}--(?P<observable>[A-Za-z0-9]+)[^-_]*'
                                         r'\.(png|tiff?)')

    operetta_export_folder_path = Path(operetta_export_folder)
    output_root_folder_path = Path(output_root_folder)

    assert operetta_export_folder_path.exists()


    print("Info: Extracting metadata ... ", flush=True)

    # config file
    with open(Path(config_file), 'r', encoding='UTF-8') as config_file_io:
        config = yaml.load(config_file_io, Loader=yaml.FullLoader)

    # channels info
    if not config['Remixes'] or not config['Remixes']['multi_channel']:
        print("Error: No pseudocolors given in the config file!")
        print("Hint: Provide a dummy multi-channel remix to define pseudocolors for all channels.")
        return
    channels_info: pd.DataFrame = extract_channels_info(operetta_export_folder_path)
    channels_info['Observable'] = channels_info['Name'].map(config['Observables'])
    observables_colors: dict = {}
    for overlay_s in config['Remixes']['multi_channel']:
        observables_colors |= dict(
            map(obs_col_re_match.group, ['observable', 'color'])
            for oc in list(map(lambda s: s.strip(), overlay_s.split('+')))
            if (obs_col_re_match := OPERETTA_OBSERVABLE_COLOR_RE.search(oc)) is not None
        )
    channels_info['Color'] = channels_info['Observable'].map(observables_colors)
    colors = channels_info[~channels_info['Color'].isnull()].reset_index().set_index('Observable')

    # wells info
    wells_info: pd.DataFrame = extract_wells_info(operetta_export_folder_path)

    # time interval
    delta_t = extract_time_interval(operetta_export_folder_path)

    print("Info: Extracting metadata ... done ", flush=True)
    print("Info: Creating ShuttleTracker-viewable folders ... ", flush=True)

    processed_folders_count = 0
    for item in output_root_folder_path.iterdir():
        if not item.is_dir():
            continue
        folder = item
        if folder.name.endswith(REMIXES_FOLDER_BASE_NAME) and \
                folder.name != REMIXES_FOLDER_BASE_NAME:
            remixes_folder = folder

            # source image files
            image_files_paths = [
                path for path in Path(remixes_folder).iterdir()
                if path.is_file() and single_channel_remix_re.match(path.name) is not None
            ]

            # observables in source file names
            observables = dict(enumerate(set(
                single_channel_remix_match.group('observable')
                for path in image_files_paths
                if (single_channel_remix_match := single_channel_remix_re.search(path.name)) \
                        is not None
            )))
            for obs in observables.values():
                assert obs in colors.index, f"Observable '{obs}' has no assigned pseudocolor!"

            # create output folder
            well_id = remixes_folder.name.replace(f"-{REMIXES_FOLDER_BASE_NAME}", '')
            shuttletrackerable_folder = output_root_folder_path / \
                f"{well_id}-{SHUTTLETRACKER_FOLDER_BASE_NAME}"
            shuttletrackerable_folder.mkdir(exist_ok=False, parents=True)

            # make symlinks to image files
            for obs_i, obs in observables.items():
                obs_image_files_paths = [path for path in image_files_paths if obs in path.name]
                for source_image_path in obs_image_files_paths:
                    symlink_name = source_image_path.name.replace(f"--{obs}", f"_ch{str(obs_i)}")
                    symlink_path = Path(shuttletrackerable_folder) / symlink_name
                    symlink_path.symlink_to(source_image_path.absolute())

            # create a metadata file
            shuttletracker_metadata_path = Path(shuttletrackerable_folder) \
                / 'shuttletracker_metadata.txt'
            with open(shuttletracker_metadata_path, 'w', encoding='UTF-8') as st_file:
                for obs_i, obs in observables.items():
                    print(f"channel {obs_i} {obs} {colors.loc[obs]['Color']} 8", file=st_file)
                if delta_t is not None:
                    print(f"time_interval {delta_t.total_seconds()}", file=st_file)

            print(f"Info: Images from '{remixes_folder.name}' symlinked in "
                  f"'{shuttletrackerable_folder.name}'.")

            processed_folders_count += 1

    if processed_folders_count == len(wells_info):
        print("Info: Creating ShuttleTracker-viewable folders ... done")
    else:
        if processed_folders_count > 0:
            print("Info: Creating ShuttleTracker-viewable folders ... PARTLY done")
        else:
            print("Info: Creating ShuttleTracker-viewable folders ... NOT done")
        print(f"Warning: Processed folder count: {processed_folders_count}, "
              f"well count in metadata: {len(wells_info)}.")

commands.add_command(trackerabilize)



@click.command()
@click.argument('what', type=click.Choice(['wells', 'channels', 'settings']))
@click.argument('operetta_export_folder', type=CLICK_EXISTING_FOLDER_PATH_TYPE)
@click.option('--simple', type=bool, is_flag=True, default=False, show_default=True,
              help="print one (simplified) item per line")
def show(
    operetta_export_folder: Path,
    what: Literal['wells', 'channels'],
    simple: bool = False,
) -> None:
    '''
    Inspect contents (wells, channels, microscope settings).
    '''

    operetta_export_folder_path = Path(operetta_export_folder)

    if what == 'channels':
        show_channels(operetta_export_folder_path, simple)

    elif what == 'wells':
        show_wells(operetta_export_folder_path, simple)

    elif what == 'settings':
        show_settings(operetta_export_folder_path, simple)

    else:
        print(f"Error: Don't know how to show '{what}'.")

commands.add_command(show)



@click.command()
@click.argument('operetta_export_folder', type=CLICK_EXISTING_FOLDER_PATH_TYPE)
@click.argument('config_file', type=CLICK_EXISTING_FILE_PATH_TYPE)
@click.argument('output_folder', type=click.Path(exists=False), default=Path.cwd())
@click.option('--well', default=None,
              help='Process images from selected wells only.')
@click.option('--correct/--no-correct', default=True,
              help='(Do not) perform FF-correction stage.')
@click.option('--stitch/--no-stitch', default=True,
              help='(Do not) stitch tiles.')
@click.option('--remix/--no-remix', default=True,
              help='(Do not) prepare desired channel remixes.')
@click.option('--encode/--no-encode', default=True,
              help='(Do not) encode a movie.')
@click.option('--force-correct',  is_flag=True, default=False,
              help='Generate FF-corrected tiles even if exist.')
@click.option('--force-stitch',  is_flag=True, default=False,
              help='Stitch even when stitched images exist.')
@click.option('--force-remix',  is_flag=True, default=False,
              help='Remix even when remixed images exist.')
@click.option('--force-encode', is_flag=True, default=False,
              help='Encode even when a movie file exists.')
@click.option('--remove-intermediates', is_flag=True, default=False, show_default=True,
              help=f"Remove folders '{CORRTILES_FOLDER_BASE_NAME}', '{STITCHES_FOLDER_BASE_NAME}'.")
@click.option('--annotate-remixes-with-wells-info', is_flag=True, default=False, show_default=True,
              help="Use plate layout information to annotate remixed images.")
@click.option('--before', is_flag=False, default=None, help="Pre-organize input images. ",
              multiple=True, type=click.Choice([
                  select_best_focus_planes.__name__,
                  'fix_single_pixel_images',
              ]))
@click.option('--after', is_flag=False, default=None, help="Post-organize output files. ",
              multiple=True, type=click.Choice([
                  rename_single_timepoint_stitches_by_well_id.__name__,
                  rename_single_timepoint_remixes_by_well_id.__name__,
              ]))
def remaster(
    operetta_export_folder: Path,
    config_file: Path,
    output_folder: Path,
    well: str | None,
    correct: bool,
    stitch: bool,
    remix: bool,
    encode: bool,
    force_correct: bool,
    force_stitch: bool,
    force_remix: bool,
    force_encode: bool,
    remove_intermediates: bool,
    annotate_remixes_with_wells_info: bool,
    before: str | None,
    after: str | None,
) -> None:
    '''
    Apply the grand image processing pipeline.
    '''

    # input experiment folder
    operetta_export_folder_path = Path(operetta_export_folder)
    assert operetta_export_folder_path.exists()
    print(f"Info: Export folder: '{operetta_export_folder_path.absolute()}'")

    # image pre-processing
    if before:
        if 'fix_single_pixel_images' in before:
            _check(operetta_export_folder_path, fix_single_pixel_and_illegible_images=True)
        if select_best_focus_planes.__name__ in before:
            select_best_focus_planes(operetta_export_folder_path)

    # config file
    config_file_path = Path(config_file)
    with open(config_file_path, 'r', encoding='UTF-8') as config_file_io:
        config = yaml.load(config_file_io, Loader=yaml.FullLoader)
    for obligatory_section in ('Observables', 'Normalization', 'Stitching', 'Remixes'):
        assert obligatory_section in config, \
            f"Error: Section '{obligatory_section}' missing in the config file."

    # output folder
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(exist_ok=True, parents=True)
    print(f"Info: Output folder: '{output_folder_path.absolute()}'")

    # leaky behavior of lxml resolved when a subprocess ends
    results_queue: Queue = Queue()
    parse_process = Process(
        target=_extract_xml_metadata,
        args=(operetta_export_folder_path, results_queue)
    )
    parse_process.start()
    plate_layout, wells_info, channels_info, images_layout, time_interval = results_queue.get()
    parse_process.join()


    print(80*'-' + '\n' + f"Plate layout: {plate_layout['Rows']}x{plate_layout['Cols']}. ", end='')
    print(f" Wells used: {len(wells_info)}.", flush=True)

    with open(output_folder_path / 'WellsInfo.html', 'w', encoding='UTF-8') as wells_info_file:
        print(f"<html><body>{wells_info.to_html()}</body></html>", file=wells_info_file)
    with open(output_folder_path / 'WellsInfo.pkl', 'wb') as wells_info_archive:
        pickle.dump(wells_info, wells_info_archive)
    wells_info_str = wells_info.to_string(max_cols=100)
    print(80*'-' + '\n', wells_info_str, '\n' + 80*'-')

    channels_info['Observable'] = channels_info['Name'].map(config['Observables'])
    missing_observables = channels_info[ channels_info['Observable'].isna() ]['Name']
    assert len(missing_observables) == 0, \
            print(f"Error: Unaliased observable(s): {','.join(missing_observables.values)}\n"
                  f"{channels_info}")

    channels_info['Normalization'] = channels_info['Observable'].map(config['Normalization'])
    channels_info = channels_info[['Name', 'Observable', 'Correction', 'Normalization']]

    channels_info_to_print = channels_info.copy()
    channels_info_to_print['MeanBg'] = channels_info_to_print['Correction'].map(
        lambda c: round(c['Mean']) if c['Character'] == 'NonFlat' else pd.NA
    )
    channels_info_to_print['Correction'] = channels_info_to_print['Correction'].map(
        lambda c: c['Character']
    )
    channels_info_to_print_str = channels_info_to_print.to_string()
    print(channels_info_to_print_str, '\n' + 80*'-')

    if time_interval is not None:
        time_interval_secs = time_interval.total_seconds()
        print(f"Time interval: {time_interval_secs//60:.0f}m{time_interval_secs%60:0>2.0f}s.")

    # stitching settings
    stitching_tile_overlap = float(str(config['Stitching']['tile_overlap']).replace('%', '')) \
            if 'tile_overlap' in config['Stitching'] else 0.5
    stitching_downscale = config['Stitching']['downscale'] \
            if 'downscale' in config['Stitching'] else False

    # remixing settings
    remixing_downscale = config['Remixes']['downscale'] \
            if 'downscale' in config['Remixes'] else False

    # (optional) movie settings
    if 'Movies' in config and 'realtime_speedup' in config['Movies']:
        realtime_speedup = int(config['Movies']['realtime_speedup'].replace('x', ''))
    else:
        realtime_speedup = MOVIE_DEFAULT_REALTIME_SPEEDUP
        print("Info: Assumed the default movie real time speed-up: "
              f"{MOVIE_DEFAULT_REALTIME_SPEEDUP}x.")

    # (optional) text annotation settings
    if 'Annotations' in config and 'font_size' in config['Annotations']:
        annotation_font_size = int(config['Annotations']['font_size'].replace('pt', ''))
    else:
        annotation_font_size = TEXT_ANNOTATIONS_DEFAULT_FONT_SIZE
        print("Info: Assumed default text annotation font size: "
              f"{TEXT_ANNOTATIONS_DEFAULT_FONT_SIZE} pt.")


    # selection of wells
    if well is not None:
        is_re = len(set(well).intersection(set('*.[{?^'))) > 0
        if is_re:
            wells_re = re.compile(well.replace('*', r'.*'))
            wells_re_filter = lambda text: re.match(wells_re, text) is not None if text else False
            wells_info = wells_info[ wells_info['WellId'].apply(wells_re_filter) ]
        else:
            wells_info = wells_info[ wells_info['WellId'].isin(well.split(',')) ]
            assert len(wells_info) == len(well.split(',')), \
                "Non-existing (or duplicate) well selected: " \
                f"{','.join(list(sorted( set(well.split(',')) .difference (set(wells_info['WellId'])) )))}."
        assert len(wells_info) > 0, "Effectively empty well selection."


    # loop over all wells
    for _, well_info in wells_info.reset_index()[::+1].iterrows():

        # extract well location and well identifier
        well_row, well_column, well_id = well_info[['Row', 'Column', 'WellId']]
        well_loc = WellLocation(row=well_row, column=well_column)

        print(32*'-', f"[ Well: {well_id} ]", 32*'-')

        # extract or infer statistics of image files in the current well
        orig_images_folder_path = operetta_export_folder_path / OPERETTA_IMAGES_SUBFOLDER_NAME
        well_orig_image_files_paths = get_image_files_paths(orig_images_folder_path, well_loc)
        if correct:
            assert len(well_orig_image_files_paths) > 0, \
                f"No image files found in '{orig_images_folder_path.absolute()}' " \
                f"for {well_loc}."

        n_image_files = len(well_orig_image_files_paths)
        n_timepoints = determine_timepoints_count(well_orig_image_files_paths)
        n_fovs = determine_fields_count(well_orig_image_files_paths)
        n_fovs_x, n_fovs_y = determine_fields_layout(images_layout, well_loc)
        n_channels = determine_channels_count(well_orig_image_files_paths)

        print(f"[{well_id}] Number of timepoints: {n_timepoints if n_timepoints > 0 else '(?)'}.")
        print(f"[{well_id}] Number of channels: {n_channels if n_channels > 0 else '(?)'}.")
        print(f"[{well_id}] Number of fields per well: {n_fovs if n_fovs > 0 else '(?)'}"
              + (f" (layout: {n_fovs_x}x{n_fovs_y})." if n_fovs > 0 else "."))
        print(f"[{well_id}] Number of all images: {n_image_files if n_image_files > 0 else '(?)'}.")
        assert time_interval is not None if n_timepoints > 1 else True

        # assemble paths of input/output folders
        well_tiles_folder_path    = output_folder_path / f"{well_id}-{CORRTILES_FOLDER_BASE_NAME}"
        well_stitches_folder_path = output_folder_path / f"{well_id}-{STITCHES_FOLDER_BASE_NAME}"
        well_remixes_folder_path  = output_folder_path / f"{well_id}-{REMIXES_FOLDER_BASE_NAME}"
        well_movies_folder_path   = output_folder_path / f"{well_id}-{MOVIES_FOLDER_BASE_NAME}"


        # ---- Tiles ----------------------------------------------------------

        if correct:
            correct_tiles(
                well_id,
                well_loc,
                channels_info,
                images_layout,
                well_orig_image_files_paths,
                well_tiles_folder_path,
                force=force_correct
            )
        else:
            print(f"[{well_id}] (correcting skipped)")


        # ---- Stitches -------------------------------------------------------

        if stitch:
            stitch_tiles(
                well_id,
                (n_fovs_x, n_fovs_y),
                well_tiles_folder_path,
                well_stitches_folder_path,
                tile_overlap=stitching_tile_overlap,
                downscale=stitching_downscale,
                force=force_stitch
            )
            if remove_intermediates:
                shutil.rmtree(well_tiles_folder_path, ignore_errors=True)
                print(f"[{well_id}] Removed folder '{well_tiles_folder_path.name}'.")
        else:
            print(f"[{well_id}] (stitching skipped)")


        # ---- Remixes --------------------------------------------------------

        overlays_and_separates_infos = []  # for both remixing and encoding

        if remix or encode:

            # -- single-channel remixes (aka "separates")

            separates: list = config['Remixes']['single_channel'] \
                    if 'single_channel' in config['Remixes'] else []
            if separates is None:  # empty section exists
                separates = []

            for separate_obs in separates:

                mixing_info = channels_info.copy()[['Observable', 'Normalization']]
                assert separate_obs in mixing_info['Observable'].values, \
                        f"No such observable: '{separate_obs}'"
                mixing_info = mixing_info[ mixing_info['Observable'] == separate_obs ].reset_index()
                mixing_info.index = ['gray']
                mixing_info['TiffPage'] = mixing_info['ChannelNumber'] - 1
                assert mixing_info['TiffPage'].min() >= 0

                overlays_and_separates_infos.append(mixing_info)

                if remix:
                    remix_channels(
                        well_id,
                        mixing_info,
                        well_stitches_folder_path,
                        well_remixes_folder_path,
                        downscale=remixing_downscale,
                        force=force_remix,
                        wells_info_for_annotation = \
                            wells_info if annotate_remixes_with_wells_info else None,
                        is_part_of_timeseries=n_timepoints > 1,
                        annotation_font_size=annotation_font_size,
                    )


            # -- multi-channel remixes (aka "overlays")

            overlays: list | None = config['Remixes']['multi_channel'] \
                    if 'multi_channel' in config['Remixes'] else []
            if overlays is None:  # empty section exists
                overlays = []

            for overlay_s in overlays:

                mixing_info = channels_info.copy()[['Observable', 'Normalization']]

                # add observables' pseudocolors according to the config
                obs_to_color: Dict[str, str] = dict(
                    map(obs_color_re_match.group, ['observable', 'color'])
                    for oc in list(map(lambda s: s.strip(), overlay_s.split('+')))
                    if (obs_color_re_match := OPERETTA_OBSERVABLE_COLOR_RE.search(oc)) is not None
                )
                for obs, color in obs_to_color.items():
                    assert obs in channels_info['Observable'].values, \
                            f"Incorrect overlay observable: '{obs}'!"
                    assert color in COLOR_COMPONENTS, \
                            f"Unknown color: '{color}'!"
                mixing_info['Color'] = mixing_info['Observable'].map(obs_to_color)
                mixing_info = mixing_info[~mixing_info['Color'].isnull()].reset_index() \
                                                                         .set_index('Color')
                mixing_info['TiffPage'] = mixing_info['ChannelNumber'] - 1
                assert mixing_info['TiffPage'].min() >= 0

                overlays_and_separates_infos.append(mixing_info)

                if remix:
                    remix_channels(
                        well_id,
                        mixing_info,
                        well_stitches_folder_path,
                        well_remixes_folder_path,
                        downscale=remixing_downscale,
                        force=force_remix,
                        wells_info_for_annotation = \
                            wells_info if annotate_remixes_with_wells_info else None,
                        is_part_of_timeseries=n_timepoints > 1,
                        annotation_font_size=annotation_font_size,
                    )


            # -- optional cleanup

            if remove_intermediates:
                shutil.rmtree(well_stitches_folder_path, ignore_errors=True)
                print(f"[{well_id}] Removed folder '{well_stitches_folder_path.name}'.")

        else:
            print(f"[{well_id}] (remixing channels skipped)")


        # ---- Movies ---------------------------------------------------------

        if time_interval is not None:
            if encode:
                encode_movies(
                    well_id,
                    overlays_and_separates_infos,
                    time_interval,
                    well_remixes_folder_path,
                    well_movies_folder_path,
                    realtime_speedup=realtime_speedup,
                    force=force_encode,
                    annotation_font_size=annotation_font_size,
                )
            else:
                print(f"[{well_id}] (encoding of movies skipped)")

    try:
        shutil.copy(config_file_path, output_folder_path)
    except shutil.SameFileError:
        pass


    if after:
        if rename_single_timepoint_remixes_by_well_id.__name__ in after:
            rename_single_timepoint_remixes_by_well_id(output_folder_path)
        if rename_single_timepoint_stitches_by_well_id.__name__ in after:
            rename_single_timepoint_stitches_by_well_id(output_folder_path)

commands.add_command(remaster)



def _check(
    operetta_export_folder: Path,
    fix_single_pixel_and_illegible_images: bool = False,
) -> list:

    operetta_export_folder_path = Path(operetta_export_folder)
    assert operetta_export_folder_path.exists()

    images_folder_path = operetta_export_folder_path / OPERETTA_IMAGES_SUBFOLDER_NAME
    image_files_paths = get_image_files_paths(images_folder_path)
    print(f"Info: Image files count: {len(image_files_paths)}")

    images_layout = extract_and_derive_images_layout(operetta_export_folder_path)

    images_not_in_layout, image_shapes = [], {}
    for image_file_path in tqdm(image_files_paths, desc='Info: Reading images ...', **TQDM_STYLE):
        if image_file_path.name not in images_layout.index:
            images_not_in_layout.append(image_file_path)
        image = cv2.imread(str(image_file_path.absolute()), cv2.IMREAD_UNCHANGED)
        image_shapes[image_file_path] = image.shape if image is not None else (0, 0)

    if len(images_not_in_layout) > 0:
        n_absent = len(images_not_in_layout)
        print(f"Warning: There are {n_absent} image(s) not described in the layout metadata!")

    shape_counts = dict(Counter(image_shapes.values()))
    assert len(shape_counts), "Could not determine the (most prevalent) image shape."

    if len(shape_counts) > 1:
        shape_counts_s = ', '.join([
            f"{shape[0]}x{shape[1]}: {count} images"
            for shape, count in shape_counts.items()
        ])
        print(f"Warning: There are multiple image shapes: {shape_counts_s}")

        if fix_single_pixel_and_illegible_images:

            # search for the largest-area shape
            correct_shape = (0, 0)
            for shape in shape_counts:
                if shape[0]*shape[1] > correct_shape[0]*correct_shape[1]:
                    correct_shape = shape

            # draw white 'X' on black background
            white = int(2**16 - 1)
            empty_image = np.zeros(shape=correct_shape)
            cv2.line(empty_image, (0, 0), (correct_shape[0], correct_shape[1]), white, 2)
            cv2.line(empty_image, (correct_shape[0], 0), (0, correct_shape[1]), white, 2)
            empty_image_path = images_folder_path / 'empty.tiff'
            cv2.imwrite(str(empty_image_path.absolute()), empty_image.astype(np.uint16))

            for image_path, image_shape in tqdm(image_shapes.items(), desc='Info: Fixing ...',
                                                **TQDM_STYLE):
                if image_shape[0]*image_shape[1] in (0, 1):
                    image_path.rename(str(image_path.absolute()) + '.orig')
                    image_path.symlink_to(empty_image_path.name)



@click.command()
@click.argument('operetta_export_folder', type=CLICK_EXISTING_FOLDER_PATH_TYPE)
@click.option('--fix-single-pixel-and-illegible-images', is_flag=True, default=False,
    show_default=True)
def check(
    operetta_export_folder: Path,
    fix_single_pixel_and_illegible_images: bool = False,
) -> list:
    '''
    Search for:
    (1) single-pixel images (and optionally "fix" them),
    (2) images devoid of respective entries in metadata.
    '''

    return _check(operetta_export_folder, fix_single_pixel_and_illegible_images)

commands.add_command(check)



@click.command()
@click.argument('operetta_images_folder', type=CLICK_EXISTING_FOLDER_PATH_TYPE)
@click.option('--retain-original-files', is_flag=True, default=False, show_default=True)
def folderize(
    operetta_images_folder: Path,
    tolerate_extra_files: bool = False,
    retain_original_files: bool = False,
) -> list:
    '''
    Move images to their respective well-specific folders.
    '''

    return _place_image_files_in_their_respective_well_folders(
        operetta_images_folder=operetta_images_folder,
        tolerate_extra_files=tolerate_extra_files,
        retain_original_files=retain_original_files
    )

commands.add_command(folderize)



@click.command()
@click.argument('operetta_images_folder', type=CLICK_EXISTING_FOLDER_PATH_TYPE)
@click.option('--abolish-internal-compression', is_flag=True, default=True, show_default=True)
@click.option('--tolerate-extra-files', is_flag=True, default=False, show_default=True)
@click.option('--retain-original-files', is_flag=True, default=False, show_default=True)
@click.option('--retain-well-folders', is_flag=True, default=False, show_default=True)
def archivize(
    operetta_images_folder: Path,
    abolish_internal_compression: bool = True,
    tolerate_extra_files: bool = False,
    retain_original_files: bool = False,
    retain_well_folders: bool = False,
    archive_extension: str = '.tar.bz2',
    archive_root: str = '.', # extracted files will be placed in the current folder (that's desired)
):
    '''
    Create per-well image archives.

    This command moves or copies image files to individual well-specific folders and then
    compresses the folders.
    '''

    operetta_images_folder_path = Path(operetta_images_folder)

    wells_folders_paths = _place_image_files_in_their_respective_well_folders(
        operetta_images_folder=operetta_images_folder_path,
        abolish_internal_compression=abolish_internal_compression,
        tolerate_extra_files=tolerate_extra_files,
        retain_original_files=retain_original_files,
    )

    for well_folder_path in wells_folders_paths:

        assert well_folder_path.exists(), f"Non-existent folder: '{well_folder_path.absolute()}'!"

        archive_file_path = well_folder_path.with_suffix(archive_extension)

        try:
            print(f"Folder '{well_folder_path.name}' → '{archive_file_path.name}' ... ", end='',
                                                                                         flush=True)
            archive_mode = f"w:{archive_extension.replace('.tar.' ,'')}"
            with tarfile.open(archive_file_path, archive_mode) as tar:
                tar.add(well_folder_path, arcname=archive_root)
            print('done')
            success = True
        except Exception as e:
            print(f"Error: Could not create an archive of '{well_folder_path.name}'!")
            if archive_file_path.exists():
                archive_file_path.unlink()
            print(e, file=sys.stderr)
            success = False

        if success and not retain_well_folders:
            try:
                print(f"Folder '{well_folder_path.name}' → ∅  ... ", end='', flush=True)
                shutil.rmtree(well_folder_path)
                print('done')
            except Exception as e:
                print(f"Error: Could not remove folder '{well_folder_path.absolute()}'.")
                print(e, file=sys.stderr)

commands.add_command(archivize)



if __name__ == '__main__':
    commands()
