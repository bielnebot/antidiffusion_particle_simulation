import copernicusmarine
import config


def download_main(fields_to_download):
    data_ids = {
        "currents": {
            "dataset_id": "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT15M-i",
            "variables": ["vo", "uo"],
            "directory": config.currents_files_directory
        },
        "waves": {
            "dataset_id": "cmems_mod_ibi_wav_anfc_0.05deg_PT1H-i",
            "variables": ["VSDX", "VSDY"],
            "directory": config.waves_files_directory
        }
    }

    for field_i in fields_to_download:
        copernicusmarine.subset(
            dataset_id=data_ids[field_i]["dataset_id"],
            variables=data_ids[field_i]["variables"],
            minimum_longitude=0,
            maximum_longitude=4,
            minimum_latitude=40,
            maximum_latitude=43,
            start_datetime="2022-02-1T00:00:00",
            end_datetime="2022-03-1T23:59:59",
            minimum_depth=0,
            maximum_depth=1,
            output_filename=f"CMEMS_Catalan_coast_{field_i}.nc",
            output_directory=config.data_base_path + data_ids[field_i]["directory"],
            credentials_file="C:/Users/Biel/.copernicusmarine/.copernicusmarine - credentials."
)

if __name__ == "__main__":
    copernicusmarine.login()
    download_main(fields_to_download=["currents"])