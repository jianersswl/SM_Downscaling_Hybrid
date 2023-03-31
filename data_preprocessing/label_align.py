import arcpy
from arcpy import env
from arcpy.sa import *
import numpy as np

# Set environment settings
env.workspace = "D:/1GRADUATED/paper/downscaling_data/Soil_moisture_downscale_czt"
inPointFeatures = "Babao.gdb/sites.shp"

DAYLIST = np.load('D:/1GRADUATED/paper/downscaling_data/Soil_moisture_downscale_czt/BATCH/day_list.npy')
for day in DAYLIST:
    print(day)

    # # Set local variables
    inRaster = "SMAP_Babao.tif"
    outPointFeatures = "BATCH/SHP/Extract2015{}_Babao_Sites.shp".format(day)

    # # Check out the ArcGIS Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # # Execute ExtractValuesToPoints
    ExtractValuesToPoints(inPointFeatures, inRaster, outPointFeatures,
                        "None", "VALUE_ONLY")

    # arcpy.AlterField_management(outPointFeatures, "RASTERVALU", "SMAPID", "SMAPID")
    # Set local variables
    inRasters = [["SMAP/SMAP_DAY/SMAP2015{}.tif".format(day), "SMAP"],
                  ["ATI/TIFFONLY/ATI/ATI2015{}.tif".format(day), "ATI"], 
                  ["ATI/TIFFONLY/ATIMEAN/ATIMean2015{}.tif".format(day), "ATIM"],
                  ["ATI/TIFFONLY/ATISD/ATISD2015{}.tif".format(day), "ATISD"]]
    # outPointFeatures = "points_extract.shp"

    # # Check out the ArcGIS Spatial Analyst extension license
    arcpy.CheckOutExtension("Spatial")

    # # Execute ExtractMultiValuesToPoints
    ExtractMultiValuesToPoints(outPointFeatures, inRasters, "None")

    outExcel = "BATCH/EXCEL/2015{}.xls".format(day)
    arcpy.TableToExcel_conversion(outPointFeatures, outExcel)
