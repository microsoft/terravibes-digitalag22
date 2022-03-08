# TerraVibes Tech Session - 2022 Digital Ag Hackathon 

This repo contains code and tutorials for the TerraVibes Tech Session at the 2022 Digital Ag Hackathon by Cornell. TerraVibes is a research platform/tool for obtaining unique and deep insights from curated earth observation data and build models by combining them with other earth observation modalities.

## Azure VM Setup
Details on how to set up your vitual machine and download the available data from blob storage are provided in the `azure_tutorial.pdf` document.

## Downloading the data
You can download the data from blob storage using the links below. We recommend using [azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-ref-azcopy) (for more information check `azure_tutorial.pdf`). To get access, add the provided SAS token in the `<SAS-TOKEN>` placeholder in the commands below. Change the `<DATA-DIR>` placeholder below to the local directory of your preference.
1. To download the preprocessed NDVI data use following command:

```
azcopy copy "https://digitalag22.blob.core.windows.net/ndvi?<SAS-TOKEN>" <DATA-DIR> --recursive

```

2. To download the SpaceEye data use following command:

```
azcopy copy "https://digitalag22.blob.core.windows.net/spaceeye-data?<SAS-TOKEN>" <DATA-DIR> --recursive

```

We recommend downloading the data before going through the notebooks. 

## Creating the environment

We recommend using [conda](https://docs.conda.io/en/latest/) to manage the packages required on this project. The dependencies are defined in the file `environment.yaml`. A conda environment can be created using the command below:

```
conda env create -f environment.yaml
```

## About the notebooks

There are two notebooks included in this tutorial. The first notebook is an exploration of the provided SpaceEye and NDVI data, where we demonstrate how to load and visualize the provided multispectral images, as well as how to compute vegetation indices such as the Normalized Difference Vegeration Index (NDVI). NDVI is a vegetation index widely used for environmental impact assessment, agricultural evaluation, and land use change metrics. It evaluates vegetation by estimating the contrast between near infrared (which vegetation strongly reflects) and red light (which vegetation absorbs).

The second notebook demonstrates how to train a UNet to segment crops using NDVI timeseries from SpaceEye and CDL as ground-truth data. The network is trained on chips/patches of NDVI values over the whole year. Targets come from CDL at 30m resolution and are upsampled to 10m resolution via nearest neighbor interpolation. 

## Data format

We provide two years (2019 and 2020) of daily cloud-free Sentinel 2 images computed via SpaceEye and preprocessed NDVI values at a 10-day interval for a 10800km² area in Washington state. This data can be used with the provided code or for other activities in the hackathon.

NDVI data is provided as two tiff files (for 2019 and 2020, respectively), each with 37 bands. Each band provides NDVI values for one day (starting at Jan. 1st), with a 10-day interval between successive bands).

Each day of SpaceEye data is separated in a grid, with each cell being a separate tiff file. We provide the regions covered by each grid cell (and the total region of interest) in GeoJSON files with two different coordinate systems (CRSs): `grid_epsg_5070.geojson` (`EPSG:5070`, the same CRS as CDL rasters), or `grid_epsg_4326.geojson` (`EPSG:4326`, WGS84, lat-long). The grid is shown in the image below.
![spaceeye_grid](https://user-images.githubusercontent.com/4806997/157137974-d306ace0-83d1-4f61-a719-782a42ad2979.png)

You can read the files using geopandas
```
import geopandas as gpd
df = gpd.read_file("grid_epsg_4326.geojson").set_crs("epsg:4326")
```

## About SpaceEye
The tutorial showcases SpaceEye, which is a neural-network-based solution to recover pixels occluded by clouds in satellite images. SpaceEye leverages radio frequency (RF) signals in the ultra/super-high frequency band that penetrate clouds to help reconstruct the occluded regions in multispectral images. We introduce the first multi-modal multi-temporal cloud removal model that uses publicly available satellite observations and produces daily cloud-free images.
## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
