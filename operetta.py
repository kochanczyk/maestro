
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring,c-extension-no-member

# Licensed under the GPLv3: https://www.gnu.org/licenses/gpl-3.0.html
# Copyright (c) 2023-2024 Marek Kochańczyk, Paweł Nałęcz-Jawecki, Frederic Grabowski

'''
This module aggregates basic functions related to management and processing of image files generated
by Harmony, software featuring Operetta CLS, a high-throughput microplate imager.
'''

import re
import json
from datetime import datetime, timedelta
from functools import cache
from collections import namedtuple
from pathlib import Path
from typing import Tuple, List

from xml.etree import ElementTree
import pandas as pd


WellLocation = namedtuple('WellLocation', ['row', 'column'])
WellContents = namedtuple('WellContents', ['id', 'desc'])

OPERETTA_IMAGE_NAME_EXTENSION = '.tiff'
OPERETTA_IMAGE_NAME_RE = re.compile(
    ''.join([f"{letter}(?P<{meaning}>[0-9]{str('{2,}')})"
    for letter, meaning in [
        ('r', 'row'),
        ('c', 'column'),
        ('f', 'field'),
        ('p', 'plane')
    ]]) +
    '-'
    'ch(?P<channel>[0-9]+)'
    'sk(?P<timepoint>[0-9]+)'
    'fk[0-9]+fl[0-9]+' +
    OPERETTA_IMAGE_NAME_EXTENSION
)
OPERETTA_OBSERVABLE_COLOR_RE = re.compile('(?P<observable>[^{]+){(?P<color>[a-z]+)}')
OPERETTA_IMAGES_SUBFOLDER_NAME = 'Images'
OPERETTA_ASSAY_LAYOUT_SUBFOLDER_NAME = 'Assaylayout'
OPERETTA_IMAGES_INDEX_FILE_NAME = 'Index.idx.xml'
OPERETTA_EXTRA_TILE_OVERLAP = 0.55  # Even if in Harmony you set "3%", it will be ~3.6% ultimately.

def _chunk_count(re_group_name: str, image_files_paths: list) -> int:
    return len({
        matched_file_name.group(re_group_name)
        for path in image_files_paths
        if (matched_file_name := OPERETTA_IMAGE_NAME_RE.search(path.name)) is not None
    })

def determine_channels_count(image_files_paths: list) -> int:
    return _chunk_count('channel', image_files_paths)

def determine_planes_count(image_files_paths: list) -> int:
    return _chunk_count('plane', image_files_paths)

def determine_timepoints_count(image_files_paths: list) -> int:
    return _chunk_count('timepoint', image_files_paths)

def determine_fields_count(image_files_paths: list) -> int:
    return _chunk_count('field', image_files_paths)


def determine_fields_layout(
    images_layout: pd.DataFrame,
    well_loc: WellLocation,
) -> Tuple[int, int]:

    well_images_layout = images_layout[
          (images_layout['Row'] == well_loc.row)
        & (images_layout['Col'] == well_loc.column)
    ]

    # check field arrangement
    fields_span_x = well_images_layout['field_ix'].max() - well_images_layout['field_ix'].min() + 1
    fields_span_y = well_images_layout['field_iy'].max() - well_images_layout['field_iy'].min() + 1
    fields_expected_count = fields_span_x*fields_span_y
    fields_actual_count = len(set(map(tuple, well_images_layout[['field_ix', 'field_iy']].values)))
    if fields_expected_count != fields_actual_count:
        print(f"[{well_loc.row:02d}{well_loc.column:02d}] Warning:",
              'Some fields are missing or fields are not arranged in a rectangle!')

    n_fields_x = len(well_images_layout['field_ix'].unique())
    n_fields_y = len(well_images_layout['field_iy'].unique())
    return (n_fields_x, n_fields_y)


def assemble_image_file_name(
    row: str,
    column: str,
    field: str,
    plane: str,
    channel: str,
    timepoint: str,
    fk: str = '1',
    fl: str = '1'
) -> str:
    image_file_name = (
        f"r{row}c{column}f{field}p{plane}"
        "-"
        f"ch{channel}sk{timepoint}fk{fk}fl{fl}"
        f"{OPERETTA_IMAGE_NAME_EXTENSION}"
    )
    return image_file_name



@cache
def _read_xml(xml_file_path: Path) -> tuple:

    assert xml_file_path.exists(), f"The XML file '{xml_file_path.absolute()}' does not exist!"

    xml_root = ElementTree.parse(xml_file_path).getroot()

    namespaces_chunk = ElementTree.iterparse(xml_file_path, events=['start-ns'])
    namespaces = dict(node for _, node in namespaces_chunk)

    return (xml_root, namespaces)



def extract_plate_wells(operetta_export_folder_path: Path) -> dict:

    index_xml_path = operetta_export_folder_path / OPERETTA_IMAGES_SUBFOLDER_NAME \
                                                 / OPERETTA_IMAGES_INDEX_FILE_NAME
    xml, ns = _read_xml(index_xml_path)

    assert len(xml.findall('Plates', ns)) == 1
    assert len(xml.findall('Plates', ns).pop().findall('Plate', ns)) == 1
    plate_xml = xml.findall('Plates', ns).pop().findall('Plate', ns).pop()

    plate_wells_ids = []
    for well in plate_xml.findall('Well', ns):
        well_id = well.attrib['id']
        plate_wells_ids.append(well_id)

    assert len(xml.findall('Wells', ns)) == 1
    wells_xml = xml.findall('Wells', ns).pop()

    plate_wells = {}
    for well in wells_xml.findall('Well', ns):
        loc = WellLocation(row   =int(well.find('Row', ns).text),
                           column=int(well.find('Col', ns).text))
        plate_wells[loc] = well.find('id', ns).text

    return plate_wells



def extract_and_derive_images_layout(operetta_export_folder_path: Path) -> pd.DataFrame:

    index_xml_file_path = operetta_export_folder_path / OPERETTA_IMAGES_SUBFOLDER_NAME \
                                                      / OPERETTA_IMAGES_INDEX_FILE_NAME
    xml, ns = _read_xml(index_xml_file_path)

    assert len(xml.findall('Images', ns)) == 1
    images_xml, = xml.findall('Images', ns)

    position_names = ['PositionX', 'PositionY']

    # CHECK: What about PlaneID?
    column_names = ['URL', *position_names, 'Row', 'Col', 'TimepointID', 'ChannelID']

    # collect (relative) positions (in absolute units) and other info (used for grouping)
    images_info = []
    for image_xml in images_xml.findall('Image', ns):

        image_info = {}
        for name in column_names:
            assert len(image_xml.findall(name, ns)) == 1
            image_info[name] = image_xml.findall(name, ns).pop().text

        images_info.append(image_info)

    df = pd.DataFrame(images_info, columns=column_names)
    df = df.rename(columns={'URL': 'ImageFileName'}).set_index('ImageFileName')

    # take real-valued positions and compute their indices in a regular planar grid
    def assign_grid_position_(subdf: pd.DataFrame) -> pd.DataFrame:
        for p in position_names:
            to_order = {v: i
                        for i, v in enumerate(sorted(subdf[p].unique(),
                                                     key=lambda _: round(float(_), 6),
                                                     reverse='Y' in p))}
            subdf[f"field_i{p[-1].lower()}"] = subdf[p].map(to_order)
        return subdf

    df = pd.concat([
        assign_grid_position_(gdf)
        for _, gdf in df.groupby(['Row', 'Col', 'TimepointID'])  # all channels are expected
    ])                                                           # to have identical positions

    # final touches
    df = df.drop(columns=position_names)
    df_num = df.apply(pd.to_numeric)

    return df_num



def extract_time_interval(operetta_export_folder_path: Path) -> timedelta | None:

    timestamp_fmt = '%Y-%m-%dT%H:%M:%S.%f%z'
    coord_names = ('Row', 'Col', 'ChannelID', 'FieldID', 'PlaneID')

    index_xml_file_path = operetta_export_folder_path / OPERETTA_IMAGES_SUBFOLDER_NAME \
                                                      / OPERETTA_IMAGES_INDEX_FILE_NAME
    xml, ns = _read_xml(index_xml_file_path)

    t_start, t_return, visitee_coords = None, None, None
    for image_entry in xml.find('Images', ns):
        if t_start is None:
            visitee_coords = tuple(map(int, [image_entry.find(c, ns).text for c in coord_names]))
            t_start = datetime.strptime(image_entry.find('AbsTime', ns).text, timestamp_fmt)
        else:
            assert visitee_coords
            if tuple(map(int, [image_entry.find(c, ns).text for c in coord_names]))==visitee_coords:
                t_return = datetime.strptime(image_entry.find('AbsTime', ns).text, timestamp_fmt)
                break

    if t_start is None or t_return is None:
        return None

    return t_return - t_start



def extract_plate_layout(operetta_export_folder_path: Path) -> dict:

    layout_folder_path = operetta_export_folder_path / OPERETTA_ASSAY_LAYOUT_SUBFOLDER_NAME
    assert layout_folder_path.exists(), \
        f"Layout folder '{str(layout_folder_path.absolute())}' does not exist!"

    layout_file_paths = list(layout_folder_path.glob('*.xml'))
    assert len(layout_file_paths) == 1, \
        "Missing (or multiple) layout file(s)!"

    xml, ns = _read_xml(layout_file_paths.pop())

    plate_layout = {
        axis: int(xml.findall(f"Plate{axis}", ns).pop().text)
        for axis in ['Rows', 'Cols']
    }

    return plate_layout



def extract_wells_info(operetta_export_folder_path: Path) -> pd.DataFrame:

    layout_folder_path = operetta_export_folder_path / OPERETTA_ASSAY_LAYOUT_SUBFOLDER_NAME
    assert layout_folder_path.exists(), \
            f"Error: Layout folder '{str(layout_folder_path.absolute())}' does not exist!"

    layout_file_paths = list(layout_folder_path.glob('*.xml'))
    assert len(layout_file_paths) == 1

    xml, ns = _read_xml(layout_file_paths.pop())
    assert len(xml.findall("Layer", ns)) > 0

    plate_wells: dict = extract_plate_wells(operetta_export_folder_path)
    assert len(plate_wells) > 0

    wells_descriptions = {well_loc: WellContents(id=well_id, desc={})
                          for well_loc, well_id in plate_wells.items()}

    for well_loc, well_id in plate_wells.items():
        for layer in xml.findall("Layer", ns):
            layer_name = layer.find('Name', ns).text
            for well in layer.findall('Well', ns):
                r = int(well.find('Row', ns).text)
                c = int(well.find('Col', ns).text)
                if well_loc.row == r and well_loc.column == c:
                    v = well.find('Value', ns).text
                    wc = WellContents(id=well_id,
                                      desc=wells_descriptions[well_loc].desc | {layer_name: v})
                    wells_descriptions[well_loc] = wc

    fields = set(tuple(vs.desc.keys()) for vs in wells_descriptions.values())
    assert len(fields) == 1
    fieldz = list(fields.pop())

    df = pd.DataFrame({(loc.row, loc.column): [cont.id, *[cont.desc[f] for f in fieldz]]
                       for loc, cont in wells_descriptions.items()},
                      index = ['WellId'] + fieldz).T
    df.index.set_names(['Row', 'Column'], inplace=True)

    return df



def extract_channels_info(operetta_export_folder_path: Path) -> pd.DataFrame:

    images_index_file_path = operetta_export_folder_path \
                            /OPERETTA_IMAGES_SUBFOLDER_NAME \
                            /OPERETTA_IMAGES_INDEX_FILE_NAME
    xml, ns = _read_xml(images_index_file_path)

    channel_names, corrections = {}, {}
    for entry in xml.find('Maps', ns).find('Map', ns):
        channel_number = int(entry.attrib['ChannelID'])
        channel_json_text = entry.find('FlatfieldProfile', ns).text

        # fix (1 of 3, remove colon)
        channel_json_text = channel_json_text.replace('Acapella:', 'Acapella')

        # fix (2 of 3, remove spaces in channel name)
        if ch_name_match := re.search(r'(?<=ChannelName: )(?=.* *.*[^,])[A-Za-z0-9_ ]+',
                                      channel_json_text):
            ch_name = ch_name_match.group(0)
            channel_json_text = channel_json_text.replace(ch_name, ch_name.replace(' ', ''))

        # fix (3 of 3, add quotes)
        channel_json_text = re.sub(r'(?P<id>[A-Za-z]+[A-Za-z0-9_]*)', '"\\1"', channel_json_text)

        channel_json = json.loads(channel_json_text)
        channel_names[channel_number] = channel_json['ChannelName']
        corrections  [channel_number] = channel_json['Background']

        assert int(channel_json['Channel']) == channel_number

    channels_info_df = pd.Series(channel_names, name='Name').sort_index().to_frame()
    channels_info_df.index.rename('ChannelNumber', inplace=True)
    channels_info_df['Correction'] = channels_info_df.index.map(corrections)

    return channels_info_df



def get_image_files_paths(images_folder: Path, well_loc: WellLocation | None = None) -> List[Path]:

    assert images_folder.exists(), f"Images folder '{images_folder.absolute()}' does not exist!"

    paths = []
    for image_file_path in images_folder.iterdir():
        prefix_ok = True if well_loc is None else image_file_path.name.startswith(
                                                       f"r{well_loc.row:02d}c{well_loc.column:02d}")
        suffix_ok = '.tif' in image_file_path.suffix.lower()
        if prefix_ok and suffix_ok:
            assert image_file_path.is_file() or image_file_path.is_symlink()
            paths.append(image_file_path)

    return list(sorted(paths))



def field_number_to_xy(image_file_name: str, images_layout_info: pd.DataFrame) -> Tuple[int, int]:
    '''Returns zero-based FOV indices.'''

    if image_file_name in images_layout_info.index:
        image_layout_info = images_layout_info.loc[image_file_name]
    else:
        print(f"Warning: No layout metadata entry for image file '{image_file_name}'!")

        # try with a close alternative
        # (same row, column, field, position, channel -- but in the initial time point)
        sk_re = re.compile(r'^r\d{2}c\d{2}f\d{2}p\d{2}-ch\d+(sk\d+fk)\d+fl\d+\.tiff$')
        sk_re_match = sk_re.match(image_file_name)
        assert sk_re_match
        replacement_image_file_name = image_file_name.replace(sk_re_match.group(1), 'sk1fk')
        if replacement_image_file_name in images_layout_info.index:
            image_layout_info = images_layout_info.loc[replacement_image_file_name]
        else:
            # try with any alternative with the field index matching
            re_match = OPERETTA_IMAGE_NAME_RE.match(image_file_name)
            assert re_match
            field = re_match.group('field')
            for any_image_file_name in images_layout_info.index:
                any_re_match = OPERETTA_IMAGE_NAME_RE.match(any_image_file_name)
                assert any_re_match
                if any_re_match.group('field') == field:
                    if any_image_file_name in images_layout_info.index:
                        image_layout_info = images_layout_info.loc[any_image_file_name]
                        break

    return (
        image_layout_info['field_ix'],
        image_layout_info['field_iy']
    )



# useful for up to 5x5 (e.g., for 11x11, the central tile has index (5, 4), not (5, 5), psikus!)
def _field_number_to_xy_OBSOLETE(fov_index: int, n_fovs_x: int, n_fovs_y) -> Tuple[int, int]:

    if (n_fovs_x, n_fovs_y) == (1, 1):
        assert fov_index == 1
        return (0, 0)

    if (n_fovs_x, n_fovs_y) == (2, 2):
        if fov_index == 1:
            return (1, 1)
        if fov_index == 2:
            return (0, 0)
        if fov_index == 3:
            return (1, 0)
        return (0, 1)

    fov_i = fov_index - 1

    before_central = fov_i > n_fovs_x*n_fovs_y//2
    extra = 1 if before_central else 0
    if fov_i == 0:
        x, y = n_fovs_x//2, n_fovs_y//2
    else:
        y = (fov_i - 1 + extra) // n_fovs_x
        x = (fov_i - 1 + extra) %  n_fovs_x
        row_is_odd = y % 2
        if row_is_odd:
            x = n_fovs_x - 1 - x
        assert (x, y) != (n_fovs_x//2, n_fovs_y//2)

    return (x, y)
