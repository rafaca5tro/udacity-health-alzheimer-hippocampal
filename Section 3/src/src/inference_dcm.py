"""
Here we do inference on a DICOM volume, constructing the volume first, and then sending it to the
clinical archive

This code will do the following:
    1. Identify the series to run HippoCrop.AI algorithm on from a folder containing multiple studies
    2. Construct a NumPy volume from a set of DICOM files
    3. Run inference on the constructed volume
    4. Create report from the inference
    5. Call a shell script to push report to the storage archive
"""

import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from inference.UNetInferenceAgent import UNetInferenceAgent


def load_dicom_volume_as_numpy_from_list(dcmlist: list) -> tuple:
    """Loads a list of PyDicom objects as a Numpy array.
    Assumes that only one series is in the array.

    Arguments:
        dcmlist {list of PyDicom objects} -- DICOM instances to stack

    Returns:
        tuple of (3D volume as Numpy array, header of the 1st image)
    """
    slices = [np.flip(dcm.pixel_array).T for dcm in sorted(dcmlist, key=lambda dcm: dcm.InstanceNumber)]

    hdr = dcmlist[0]

    # We return header so that we can inspect metadata properly.
    # We zero-out Pixel Data since the users of this function are only interested in metadata
    hdr.PixelData = None
    return (np.stack(slices, 2), hdr)


def get_predicted_volumes(pred: np.ndarray) -> dict:
    """Gets volumes of two hippocampal structures from the predicted array.

    Arguments:
        pred {Numpy array} -- array with labels. Assuming 0 is bg, 1 is anterior, 2 is posterior

    Returns:
        A dictionary with respective volumes
    """
    volume_ant = int(np.count_nonzero(pred == 1))
    volume_post = int(np.count_nonzero(pred == 2))
    total_volume = volume_ant + volume_post

    return {"anterior": volume_ant, "posterior": volume_post, "total": total_volume}


def create_report(inference: dict, header, orig_vol: np.ndarray, pred_vol: np.ndarray):
    """Generates an image with inference report.

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """
    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.truetype("./src/assets/Roboto-Regular.ttf", size=40)
    main_font = ImageFont.truetype("./src/assets/Roboto-Regular.ttf", size=20)

    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4]

    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text((10, 90),
                         f"Patient ID: {header.PatientID}\n Study Description: {header.StudyDescription}\n Modality: {header.Modality}\n Hippocampal total volume: {inference['total']}\n Anterior volume: {inference['anterior']}\n Posterior volume: {inference['posterior']}",
                         (255, 255, 255), font=main_font)

    # Create a PIL image from array, with zero-division guard
    max_val = np.max(orig_vol)
    if max_val > 0:
        nd_img = np.flip((orig_vol[slice_nums[0], :, :] / max_val) * 0xff).T.astype(np.uint8)
    else:
        nd_img = np.zeros_like(orig_vol[slice_nums[0], :, :].T, dtype=np.uint8)

    pil_i = Image.fromarray(nd_img, mode="L").convert("RGBA").resize(nd_img.shape)
    pimg.paste(pil_i, box=(10, 280))

    return pimg


def save_report_as_dcm(header, report, path: str) -> None:
    """Writes the supplied image as a DICOM Secondary Capture file.

    Arguments:
        header {PyDicom Dataset} -- original DICOM file header
        report {PIL image} -- image representing the report
        path {str} -- where to save the report
    """
    out = pydicom.Dataset(header)

    out.file_meta = pydicom.Dataset()
    out.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    out.is_little_endian = True
    out.is_implicit_VR = False

    # We need to change class to Secondary Capture
    out.SOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    out.file_meta.MediaStorageSOPClassUID = out.SOPClassUID

    # Our report is a separate image series of one image
    out.SeriesInstanceUID = pydicom.uid.generate_uid()
    out.SOPInstanceUID = pydicom.uid.generate_uid()
    out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    out.Modality = "OT"  # Other
    out.SeriesDescription = "HippoVolume.AI"

    out.Rows = report.height
    out.Columns = report.width

    out.ImageType = r"DERIVED\PRIMARY\AXIAL"
    out.SamplesPerPixel = 3  # RGB image
    out.PhotometricInterpretation = "RGB"
    out.PlanarConfiguration = 0  # R1G1B1R2G2B2... encoding
    out.BitsAllocated = 8
    out.BitsStored = 8
    out.HighBit = 7
    out.PixelRepresentation = 0

    # Set time and date
    dt = datetime.date.today().strftime("%Y%m%d")
    tm = datetime.datetime.now().strftime("%H%M%S")
    out.StudyDate = dt
    out.StudyTime = tm
    out.SeriesDate = dt
    out.SeriesTime = tm

    out.ImagesInAcquisition = 1

    # We empty these since most viewers will then default to auto W/L
    out.WindowCenter = ""
    out.WindowWidth = ""

    # Data imprinted directly into image pixels is called "burned in annotation"
    out.BurnedInAnnotation = "YES"

    out.PixelData = report.tobytes()

    pydicom.filewriter.dcmwrite(path, out, write_like_original=False)


def get_series_for_inference(path: str) -> list:
    """Reads multiple series from one folder and picks the one
    to run inference on.

    Arguments:
        path {string} -- location of the DICOM files

    Returns:
        List of PyDicom objects representing the series
    """
    dicoms = []
    for root, subdirs, files in os.walk(path):
        for subdir in subdirs:
            dicoms.extend([pydicom.dcmread(os.path.join(path, subdir, f)) for f in os.listdir(os.path.join(path, subdir))])

    series_for_inference = [d for d in dicoms if d.SeriesDescription == "HippoCrop"]

    # Check if there are more than one series (using set comprehension)
    if len({f.SeriesInstanceUID for f in series_for_inference}) != 1:
        print("Error: can not figure out what series to run inference on")
        return []

    return series_for_inference


def os_command(command: str) -> None:
    """Execute a shell command."""
    sp = subprocess.run(["/bin/bash", "-c", command], check=False)


if __name__ == "__main__":
    # This code expects a single command line argument with link to the directory containing
    # routed studies
    if len(sys.argv) != 2:
        print("You should supply one command line argument pointing to the routing folder. Exiting.")
        sys.exit()

    # Find all subdirectories within the supplied directory. We assume that
    # one subdirectory contains a full study
    subdirs = [os.path.join(sys.argv[1], d) for d in os.listdir(sys.argv[1]) if
                os.path.isdir(os.path.join(sys.argv[1], d))]

    # Get the latest directory
    study_dir = sorted(subdirs, key=lambda dir: os.stat(dir).st_mtime, reverse=True)[0]

    print(f"Looking for series to run inference on in directory {study_dir}...")

    volume, header = load_dicom_volume_as_numpy_from_list(get_series_for_inference(study_dir))
    print(f"Found series of {volume.shape[2]} axial slices")

    print("HippoVolume.AI: Running inference...")
    inference_agent = UNetInferenceAgent(
        device="cpu",
        parameter_file_path=r"./out/model.pth")

    # Run inference
    pred_label = inference_agent.single_volume_inference_unpadded(np.array(volume))
    pred_volumes = get_predicted_volumes(pred_label)

    # Create and save the report
    print("Creating and pushing report...")
    report_save_path = r"./out/report.dcm"
    report_img = create_report(pred_volumes, header, volume, pred_label)
    save_report_as_dcm(header, report_img, report_save_path)

    # Send report to our storage archive
    os_command("storescu 127.0.0.1 4242 -v -aec HIPPOAI +r +sd ./out/report.dcm")

    # This line will remove the study dir if run as root user
    time.sleep(2)
    shutil.rmtree(study_dir, onerror=lambda f, p, e: print(f"Error deleting: {e[1]}"))

    print(f"Inference successful on {header['SOPInstanceUID'].value}, out: {pred_label.shape}",
          f"volume ant: {pred_volumes['anterior']}, ",
          f"volume post: {pred_volumes['posterior']}, total volume: {pred_volumes['total']}")
