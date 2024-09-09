Overview
========

Python module **maestro** provides command-line utilities to organize and process images acquired with a high-throughput microplate imager, Operetta CLS (from PerkinElmer).

Maestro works directly on image data plus metadata exported by Harmony software and is capable of:

* extracting and applying flat-field correction,
* generation of stitched images,
* linear scaling of the pixel intensity range with conversion to 8-bit depth,
* assembly of multi-channel pseudo-colored overlays annotated with plate well information,
* encoding timestamped movies,
* archiving exported images.

The script routinely helps researchers at the [Laboratory of Modeling in Biology and Medicine of IPPT PAN](https://pmbm.ippt.pan.pl) not to get overwhelmed by a barrage of pixels spewed out by the imager.



Installation
============

1. Create a dedicated python virtual environment and populate it with required modules listed in file `requirements.txt`.

2. Make sure that you have ImageJ/Fiji, ImageMagick, and ffmpeg installed. The module often works as a configurable glue-script that delegates tasks to these tools.



Workflow
========

## Definitions

export
: series of images (16-bit grayscale TIFFs) and associated metadata (XML files) dumped by Harmony

tile
: an image of a single field of view

stitch
: result of stitching all tiles of a well

remix
: user-defined composition (overlay) of pseudo-colored stitch channels (8-bit grayscale or 24-bit RGB PNGs)

movie
: time-lapse of remixes (an MP4 file)


## Conventions

Plate wells have their IDs built of a column index and a row index, both one-based. For example, well B5 would be named 0205. Such well IDs are used pervasively to name generated folders and files.


## Pipeline


### Outline

The general image processing pipeline is as follows:

> **Export** &rarr; **Tiles** &rarr; **Stitches** &rarr; **Remixes** &rarr; **Movies**.


To invoke maestro and execute the complete pipeline one should use its `remaster` command:
```
    python maestro.py remaster --correct --stitch --remix --encode  PATH_TO_EXPORT_FOLDER  PATH_TO_CONFIG_FILE  OPTIONAL_PATH_TO_OUTPUT_FOLDER
```

Image processing is controlled both by additional command-line options (described at specific pipeline steps) and by a config file (described further).



### Preliminary step: Data export

Data needs to be exported in Harmony. Go to Settings &rarr; Data Management &rarr; Export Data, and select "Measurements Incl. Associated Files".

Note: Harmony occasionally exports metadata that do not cover all images. This does not break the image processing pipeline.

Hint: Data transfer to external network locations turns out to be nearly twice faster when Windows Defender is instructed not to scan TIFF files.



### Image pre-processing (optional)

#### Problem addressed: Empty single-pixel images

When the imager encounters exceptionally dark fields (especially in areas at or out of well borders; marked in red in Harmony's preview during image acquisition), their corresponding images are "empty", that is, contain just one black pixel. It is often useful to visually discern the fields that were "empty", and were expanded to normal tile dimensions, from the ordinary low-signal tiles. To draw diagonal lines on "empty" upscaled tiles, add option `--before fix_single_pixel_and_illegible_images` to maestro's command `remaster`.


#### Problem addressed: Tilted focal plane

When you do not have the same focal plane (optimal distance in Z direction) for all fields in a well (which may happen when, e.g., you imaged coverslips mounted to well bottoms), it may be useful to acquire a Z-stack and then automatically select the best-focused slice. To this end, add option `--before select_best_focus_planes` to maestro's command `remaster`.

The optimal Z slice selection algorithm finds the Z, at which a laplacian-transformed image has maximum variance. Optimal Z is selected separately for each channel in each field (that is, XY position).



### Step 1: Export &rarr; Tiles (flat-field correction)

Option `--correct` (added by default, use `--no-correct` to disable).

The first step of the pipeline begins with the extraction of flat-field correction profiles from metadata (file Index.idx.xml contains bivariate polynomial coefficients, which appear to be derived by some auxiliary microscope software, Acapella). Then, fluorescence channel images are flat-field corrected and individual channels for a single field-of-view become frames of one multi-frame grayscale TIFF file. An original channel order is retained. The output file is named in a manner that encodes tile grid position indices and in this way is conformant with the standard ImageJ "Grid/Collection stitching" plugin.

If an output tile file exists, it is not re-generated and overwritten until the option `--force-correct` is passed.



### Step 2: Tiles &rarr; Stitches (stitching)

Option `--stitch` (added by default, use `--no-stitch` to disable).

Stitching of tiles is delegated to the standard ImageJ "Grid/Collection stitching" plugin. ImageJ is called in the background in a headless mode (additionally enforced by using Xvfb).

The value of the tile-over-tile overlap needs to be given in the config file, in section Stitching. The Harmony setting of '0%' is in reality about 0.5% but in the config file you are expected to provide the nominal value given in Harmony. Please bear in mind that the ideal overlap in bright-field may turn out to be suboptimal in fluorescence channels.

Stitches retain the channel order of tiles.

As stitches may be an alternative manner of long-term data storage, they are ZIP-compressed internally.

Stitches may be downsized to 50% of the original (exported) image size, meaning that both dimensions are halved. The optional downscaling is controlled with a True/False switch in the config file, in section Stitching.

If an output stitched image file exists, it is not re-generated and overwritten until the option `--force-stitch` is passed.

If tiles should be removed after assembling a stitched image, use option `--remove-intermediates` (note: setting this option also affects remixing, see below).



### Step 3: Stitches &rarr; Remixes (remixing)

Option `--remix` (added by default, use `--no-remix` to disable).

Generation of so-called remixes involves: (i) affine adjustment of the dynamic range of pixel intensities, (ii) conversion from 16-bit to 8-bit depth with a simple scaling of pixel intensities, (iii) pseudo-coloring of individual channel images.


Affine adjustments are given in normalization section of the config file, as a pair of numbers comprising a subtracted (background) intensity value and a multiplicative factor (usually > 1). The list and exact definition of overlaid channels and their respective pseudo-colors is provided in the config file in section Remixes. Available pseudo-colors are: gray, red, green, blue, cyan, magenta, yellow.

Well information provided in and extracted from the plate layout definition may be used to annotate remixes by burning in textual information in top-left image corner. To annotate remixes, add option `--annotate-remixes-with-wells-info`.

Remixes may be either grayscale images (8-bit PNGs) that show a single channel image or RGB images (24-bit PNGs) that combine channels in a user-defined manner using pseudcolors. Internally, PNGs have the following compression settings: filter 0, level 9, strategy 2, which based on exhaustive empirical measurements is expected to yield best lossless compression ratio in most cases.

Remixes may be optionally downsized to 50% of the size of stitched images, meaning that both dimensions are halved. The optional downscaling is controlled with a True/False switch in the config file. Note that when downscaling is enabled during both stitching and remixing, each pixel of remixes (and then movies) is an average of 4x4 pixels of the original image.

If stitched images should be removed after remixing, use option `--remove-intermediates` (note: setting this option also affects stitching, see above).



### Step 4: Remixes &rarr; Movies (encoding)

Option `--encode` (added by default, use `--no-encode` to disable).

In the case of time-lapse experiments, remixes of consecutive time points may be used as frames of a movie.

The frame rate is adjusted so that one hour of experiment real time is one second of the movie. This default can be changed in the config file (see the optional sections).

Hint: To watch movies with zoom, you may use [VLC](https://videolan.com) and then in its menu Tools &rarr; Adjustments and Effects &rarr; Video effects tick the option 'Interactive zoom'; use the bird-view in the top-left corner to move around.



### Post-processing

When only a single time-point has been imaged, and folders with stitches contain just single stitched image files and folders with remixes contain just handfuls of files, it may be reasonable to name all the image files using well ID. To achieve this, use these self-explanatory options: `--after rename_single_timepoint_stitches_by_well_id`, `--after rename_single_timepoint_remixes_by_well_id` (both these options may be used simultaneously).


## Configuration file
Image processing details are controlled by a YAML-formatted configuration file. The contents of an example config file:

```
Observables:
    DAPI:         dapi
    Brightfield:  bf
    Alexa488:     p65
    Alexa546:     polyic

Stitching:
    downscale:     False
    tile_overlap:  0%

Normalization:
    dapi:    -670, *1.1
    bf:     -6500, *1.9
    p65:    -2420, *2.9
    polyic: -5635, *7.8

Remixes:
    downscale:  False
    multi_channel:
        - p65{green} + polyic{magenta}
    single_channel:
        - dapi
        - bf
        - p65
        - polyic
```
(an immunostaining from 2024-04-04 by Juan Alfonso Redondo Marin).

The four sections above are obligatory. You may also include optional sections:
```
Annotations:
    font_size: 32pt  # default is 96pt

Movies:
    realtime_speedup: 60x  # one minute of the real time -> one second of the movie
```



Inspection
==========

Maestro may be used to get a glimpse of the exported data:

* list of used wells, together with plate layout info:
```
    python remaster.py show wells EXPORT_FOLDER_PATH
```

* list of channels:
```
    python remaster.py show channels EXPORT_FOLDER_PATH
```

* image acquisition settings in each channel:
```
    python remaster.py show settings EXPORT_FOLDER_PATH
```



Archives
=========

### Archiving

One may call:
```
    python maestro.py archivize PATH_TO_IMAGES_SUBFOLDER
```
to obtain BZ2-compressed archives of images, one archive per well.

By default, internal compression is abrogated (TIFF files exported by Harmony are internally compressed with LZW, which prevents a more efficient external compression).

Note on compression ratio: bzip2, although slow, was experimentally checked to give the best compression ratio, exceeding that of zip and even xz (at its "ultra" settings and extra-large dictionary), and is considered more suitable for long-term data storage than xz.


### Unarchiving

To extract all images, enter the Images folder containing per-well archives and type:
```
    find . -name 'well-*.tar.bz2' -exec tar xfj {} \;
```
The images are decompressed directly in the Images folder (not in any well subfolder), recovering the original flat layout of files in folder Images.



Limitations
===========

The module has never been used to process Z-stacks (except for a built-in capability to select best focused image in a Z stack).



Authors and license
===================

The source code, licensed under GNU GPLv3, was written by Marek Kochańczyk. Feedback and contributions from Paweł Nałęcz-Jawecki and Frederic Grabowski are gratefully acknowledged.
