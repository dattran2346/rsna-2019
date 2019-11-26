import argparse
import numpy as np
import os
import pandas as pd
import pydicom
from tqdm import tqdm


def get_metadata(image_dir):
    """
    Read DICOM files and extract relevant info.
    """
    labels = [
        'BitsAllocated', 'BitsStored', 'Columns', 'HighBit', 
        'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
        'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
        'ImagePositionPatient_0', 'ImagePositionPatient_1', 'ImagePositionPatient_2',
        'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation',
        'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope', 'Rows', 'SOPInstanceUID',
        'SamplesPerPixel', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID', 
        'WindowCenter', 'WindowWidth', 'Image',
    ]
    data = {l: [] for l in labels}
    for image in tqdm(os.listdir(image_dir)):
        data["Image"].append(image[:-4])
        ds = pydicom.dcmread(os.path.join(image_dir, image), force=True)
        for metadata in ds.dir():
            if metadata != "PixelData":
                metadata_values = getattr(ds, metadata)
                if type(metadata_values) == pydicom.multival.MultiValue and metadata not in ["WindowCenter", "WindowWidth"]:
                    for i, v in enumerate(metadata_values):
                        data[f"{metadata}_{i}"].append(v)
                else:
                    if type(metadata_values) == pydicom.multival.MultiValue and metadata in ["WindowCenter", "WindowWidth"]:
                        data[metadata].append(metadata_values[0])
                    else:
                        data[metadata].append(metadata_values)
    return pd.DataFrame(data).set_index("Image")


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/")
    parser.add_argument('--dicom_folder', type=str, default="dicom/stage_2_test_images/")
    parser.add_argument('--sample_sub_csv', type=str, default="stage_2_sample_submission.csv")
    parser.add_argument('--test_meta_csv', type=str, default="stage_2_test_metadata.csv")
    args = parser.parse_args()
    # Generate metadata dataframe
    test_metadata = get_metadata(
        os.path.join(args.data_dir, 
                     args.dicom_folder))
    dropped_cols = [
        'BitsAllocated', 'BitsStored',
        'Columns', 'HighBit', 
        'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 
        'ImageOrientationPatient_2', 'ImageOrientationPatient_3', 
        'ImageOrientationPatient_4', 'ImageOrientationPatient_5', 
        'ImagePositionPatient_0', 'ImagePositionPatient_1', 
        'Modality', 'PhotometricInterpretation', 'PixelRepresentation',
        'PixelSpacing_0', 'PixelSpacing_1', 
        'RescaleIntercept', 'RescaleSlope',
        'Rows', 'SamplesPerPixel', 
        'SOPInstanceUID', 'SeriesInstanceUID', 'StudyID',
    ]
    test_df = pd.read_csv(args.data_dir + args.sample_sub_csv) \
        .drop_duplicates()
    test_df['image'] = test_df["ID"].str.slice(stop=12)
    test = test_df["image"].drop_duplicates()
    test_metadata["image"] = test_metadata.index
    merged_test = pd.merge(test, test_metadata, how="inner", on="image")
    merged_test.drop(columns=dropped_cols, inplace=True)
    merged_test = merged_test.groupby(["StudyInstanceUID"]) \
        .apply(lambda x: x.sort_values(["ImagePositionPatient_2"], 
                                       ascending = True)) \
        .reset_index(drop=True)
    merged_test.to_csv(args.data_dir + args.test_meta_csv, 
        index=False)