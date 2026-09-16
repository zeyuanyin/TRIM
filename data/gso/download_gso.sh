#!/bin/bash
# Google Scanned Objects (public dataset), used for image-to-3D evaluation (Table 2).
# `gso_data_300.json` / `gso_data_300.txt` list the 300 objects used in the paper.
wget https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/google_scanned_objects.zip
unzip google_scanned_objects.zip -d google_scanned_objects
