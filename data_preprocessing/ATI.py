# -*- coding=utf-8 -*-
### 批量计算ATI
import arcpy
import math
import numpy
import os

def SolarCorrectionCoefficient(phi,nd):
    gamma=2*math.pi*(nd-1)/365.25
    delta=0.00691-0.3999912*math.cos(gamma)+0.070257*math.sin(gamma)-0.006758*math.cos(2*gamma)\
          +0.000907*math.sin(2*gamma)-0.002697*math.cos(3*gamma)+0.00148*math.sin(3*gamma)
    a=1-((math.tan(phi))**2)*((math.tan(delta))**2)
    b=math.acos(-math.tan(phi)*math.tan(delta))
    C=math.sin(phi)*math.sin(delta)*math.pow(a,0.5)\
      +math.cos(phi)*math.cos(delta)*b
    return C

def BroadbandAlbedo(a1,a2,a3,a4,a5,a7):
    a0=0.160*a1+0.291*a2+0.243*a3+0.116*a4\
        +0.112*a5+0.081*a7-0.0015
    return a0

def TRange(T1,T2,T3,T4):
    omega=7.292e-5
    t1=5400.0
    t2=37800.0
    t3=48600.0
    t4=81000.0
    a=(numpy.double(T1)-numpy.double(T3))*(math.cos(omega*t2)-math.cos(omega*t4))
    b=(numpy.double(T2)-numpy.double(T4))*(math.cos(omega*t1)-math.cos(omega*t3))
    c=(numpy.double(T2)-numpy.double(T4))*(math.sin(omega*t1)-math.sin(omega*t3))
    d=(numpy.double(T1)-numpy.double(T3))*(math.sin(omega*t2)-math.sin(omega*t4))
    xi=(a-b)/(c-d)
    Phi=math.atan(xi)+math.pi
    cos1=math.cos(omega*t1-Phi)
    cos2=math.cos(omega*t2-Phi)
    cos3=math.cos(omega*t3-Phi)
    cos4=math.cos(omega*t4-Phi)
    A=2*(4*(cos1*T1+cos2*T2+cos3*T3+cos4*T4)-(cos1+cos2+cos3+cos4)*(T1+T2+T3+T4))/(4*(cos1**2+cos2**2+cos3**2+cos4**2)-(cos1+cos2+cos3+cos4)**2)
    return A

def ATIFunc(LSTdir,REFdir,latFile,RefStr,ndStr,nd):
    #LSTdir="F:\\2019土壤水\\testData\\LST裁剪\\"
    #REFdir="F:\\2019土壤水\\testData\\反射率裁剪重采样\\"
    rasT1=arcpy.Raster(LSTdir+"\\MYD11A1_A2015"+ndStr+"_Night.tif")#T1 1.30 MYD night nodata 0
    rasT2=arcpy.Raster(LSTdir+"\\MOD11A1_A2015"+ndStr+"T_Day.tif")#T2 10.30 MOD day nodata 0
    rasT3=arcpy.Raster(LSTdir+"\\MYD11A1_A2015"+ndStr+"_Day.tif")#T3 13.30 MYD day nodata 0
    rasT4=arcpy.Raster(LSTdir+"\\MOD11A1_A2015"+ndStr+"_Night.tif")#T4 22.30 MOD night nodata 0
    REF=[] #nodata 0

    for i in range(7):
        if i !=5:
            filename=REFdir+"\\MOD09A1_A2015"+RefStr+"_b0"+str(i+1)+".tif"
            RefRasFile=arcpy.Raster(filename)
            REF.append(RefRasFile)
        else:
            continue
    #latFile="F:\\2019土壤水\\testData\\纬度.tif"
    rasLat=arcpy.Raster(latFile)# nodata 0
    #metadata
    lowerLeft = arcpy.Point(rasLat.extent.XMin,rasLat.extent.YMin)
    cellSize = rasLat.meanCellWidth
    #print "read OK!"
    LatArr=arcpy.RasterToNumPyArray(rasLat,nodata_to_value=0)/180.0*math.pi
    T1Arr=0.02*arcpy.RasterToNumPyArray(rasT1,nodata_to_value=0)
    T2Arr=0.02*arcpy.RasterToNumPyArray(rasT2,nodata_to_value=0)
    T3Arr=0.02*arcpy.RasterToNumPyArray(rasT3,nodata_to_value=0)
    T4Arr=0.02*arcpy.RasterToNumPyArray(rasT4,nodata_to_value=0)
    REFArr=[]
    for refFile in REF:
        REFArr.append(arcpy.RasterToNumPyArray(refFile,nodata_to_value=0)*0.0001)
    newRasArr=numpy.empty(LatArr.shape,LatArr.dtype)
    newRasArrA = numpy.empty(LatArr.shape, LatArr.dtype)
    #print "convert OK!"
    for i in range(len(LatArr)):
        #print "row"+str(i+1)
        for j in range(len(LatArr[i])):
            if LatArr[i][j]==0 or T1Arr[i][j]==0 or T2Arr[i][j]==0 or T3Arr[i][j]==0 or T4Arr[i][j]==0\
                    or REFArr[0][i][j]==0 or REFArr[1][i][j]==0 or REFArr[2][i][j]==0 or REFArr[3][i][j]==0 or REFArr[4][i][j]==0 or REFArr[5][i][j]==0:
                newRasArr[i][j]=65535
                newRasArrA[i][j]=65535
            else:
                C=SolarCorrectionCoefficient(LatArr[i][j],nd)
                a0=BroadbandAlbedo(REFArr[0][i][j],REFArr[1][i][j],REFArr[2][i][j],REFArr[3][i][j],REFArr[4][i][j],REFArr[5][i][j])
                A=TRange(T1Arr[i][j],T2Arr[i][j],T3Arr[i][j],T4Arr[i][j])
                ATIValue=C*(1-a0)/A
                newRasArr[i][j]=ATIValue
                newRasArrA[i][j]=A
    #print "calculate OK!"
    newRaster = arcpy.NumPyArrayToRaster(newRasArr,lowerLeft,cellSize,value_to_nodata=65535)
    newRasterA = arcpy.NumPyArrayToRaster(newRasArrA, lowerLeft, cellSize, value_to_nodata=65535)
    newRaster.save(u"F:\\2020soilMoisture\\MODIS\\ATIs\\ATI2015"+ndStr+".tif")
    newRasterA.save(u"F:\\2020soilMoisture\\MODIS\\温差\\A2015"+ndStr+".tif")
    #print "save OK!"

def getFiles(filepath,suffix):
    soundfile=[]

    pathdir = os.listdir(filepath)
    for s in pathdir:
        newdir = os.path.join(filepath, s)  # 将文件名加入到当前文件路径后面
        if os.path.isfile(newdir):  # 如果是文件
            if os.path.splitext(newdir)[1] == suffix:  # 如果文件是".pdb"后缀的
                soundfile.append(newdir)
    return soundfile

# land surface temperature
LSTdir=u"F:\\2020soilMoisture\\MODIS\\11A1"
# latitude
latFile=u"F:\\2020soilMoisture\\MODIS\\lat\\lat.tif"
#
REFdir=u'F:\\2020soilMoisture\\MODIS\\09A1resample_setnull'
REFFileList=getFiles(REFdir,'.tif')


for i in range(0,len(REFFileList),6):
    REFFile=REFFileList[i]
    a=REFFile.find("_A")+2
    RefStr=REFFile[a + 4:a + 7]
    if RefStr[0]=='0':
        Refday=eval(RefStr[1:3])
    else:
        Refday = eval(RefStr)
    for i in range(8):
        nd=Refday+i
        if nd<100:
            ndStr="0"+str(nd)
        else:
            ndStr=str(nd)
        if os.path.exists(LSTdir+"\\MYD11A1_A2015"+ndStr+"_Night.tif") and os.path.exists(
        LSTdir +"\\MOD11A1_A2015" + ndStr + "T_Day.tif")and os.path.exists(
        LSTdir + "\\MYD11A1_A2015" + ndStr + "_Day.tif") and os.path.exists(
        LSTdir + "\\MOD11A1_A2015" + ndStr + "_Night.tif"):
            print ndStr
            ATIFunc(LSTdir,REFdir,latFile,RefStr,ndStr,nd)












