
from functools import reduce
import os
import datetime as dt
from loguru import logger
import requests
import bs4
import shutil
import glob
import pandas as pd
import numpy as np

class DMSP(object):

    def __init__(
        self, date=dt.datetime(2017,9,10),
        sats = ["f17", "f18"],
        clear=False
    ):
        if clear: shutil.rmtree("database/sussi/")
        os.makedirs("database/sussi/", exist_ok=True)
        self.date = date
        self.sats = sats
        self.baseUrl = "http://ssusi.jhuapl.edu/"
        self.dataType = "sdr"
        return

    def download(self):
        inpDoY = "%02d"%self.date.timetuple().tm_yday
        for sat in self.sats:
            logger.info(f"Downlaod {sat}")
            outDir = f"database/sussi/{sat}/"
            os.makedirs(outDir, exist_ok=True)
            payload = { 
                "spc":sat,
                "type":self.dataType,
                "Doy":inpDoY,
                "year":str(self.date.year)
            }
            r = requests.get(self.baseUrl + "data_retriver/",\
                            params=payload, verify=False)
            soup = bs4.BeautifulSoup(r.text, "html.parser")
            divFList = soup.find("div", {"id": "filelist"})
            hrefList = divFList.find_all(href=True)
            urlList = [ self.baseUrl + hL["href"] for hL in hrefList ]
                
            for fUrl in urlList:
                # we only need data files which have .NC
                if ".NC" not in fUrl:
                    continue
                # If working with sdr data use only
                # sdr-disk files
                if self.dataType == "sdr":
                    if "SDR-DISK" not in fUrl:
                        continue
                fname = outDir + fUrl.split("/")[-1]
                if not os.path.exists(fname):
                    try:
                        logger.info(f"currently downloading--> {fUrl}")
                        rf = requests.get( fUrl, verify=False )
                        currFName = rf.url.split("/")[-1]
                        
                        with open( outDir + currFName, "wb" ) as ssusiData:
                            ssusiData.write( rf.content )
                    except:
                        pass
        return

    def read_DMSP(self, coords="geo"):
        logger.info("Reading SUSSI...")
        fname = f"database/sussi-{self.date.day}.csv"
        if os.path.exists(fname):
            self.dataset = pd.read_csv(
                fname,
                parse_dates=["date"]
            )
        else:
            from netCDF4 import Dataset
            from cdflib.epochs import CDFepoch

            self.dataset = pd.DataFrame()
            for sat in self.sats:
                logger.info(f"Reading {sat}")
                tag = f"database/sussi/{sat}/*.NC"
                files = glob.glob(tag)
                for i, f in enumerate(files):
                    ds = Dataset(f)
                    prpntLats = ds.variables["PIERCEPOINT_DAY_LATITUDE"][:]
                    prpntLons = ds.variables["PIERCEPOINT_DAY_LONGITUDE"][:]
                    prpntAlts = ds.variables["PIERCEPOINT_DAY_ALTITUDE"][:]
                    dskInt121 = ds.variables["DISK_INTENSITY_DAY"][:, :, 0]
                    dskInt130 = ds.variables["DISK_INTENSITY_DAY"][:, :, 1]
                    dskInt135 = ds.variables["DISK_INTENSITY_DAY"][:, :, 2]
                    dskIntLBHS = ds.variables["DISK_INTENSITY_DAY"][:, :, 3]
                    dskIntLBHL = ds.variables["DISK_INTENSITY_DAY"][:, :, 4]
                    dtList = [ dt.datetime(*e)
                        for e in CDFepoch.breakdown_epoch(ds.variables["TIME_EPOCH_DAY"][:] )
                    ]
                    
                    latColList = [ "glat." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                    lonColList = [ "glon." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                    d121ColList = [ "d121." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                    d130ColList = [ "d130." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                    d135ColList = [ "d135." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                    dLBHSColList = [ "dlbhs." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                    dLBHLColList = [ "dlbhl." + str(cNum+1) for cNum in range(prpntLats.shape[0]) ]
                    # # create dataframes with
                    dfLat = pd.DataFrame(prpntLats.T,columns=latColList, index=dtList)
                    dfLon = pd.DataFrame(prpntLons.T,columns=lonColList, index=dtList)
                    dfD121 = pd.DataFrame(dskInt121.T,columns=d121ColList, index=dtList)
                    dfD130 = pd.DataFrame(dskInt130.T,columns=d130ColList, index=dtList)
                    dfD135 = pd.DataFrame(dskInt135.T,columns=d135ColList, index=dtList)
                    dfDLBHS = pd.DataFrame(dskIntLBHS.T,columns=dLBHSColList, index=dtList)
                    dfDLBHL = pd.DataFrame(dskIntLBHL.T,columns=dLBHLColList, index=dtList)

                    ssusiDF = reduce(lambda left,right: pd.merge(left,right,\
                            left_index=True, right_index=True), [ dfLat, \
                            dfLon, dfD121, dfD130, dfD135, dfDLBHL, dfDLBHS ])
                    ssusiDF["orbitNum"] = ds.variables["ORBIT_DAY"][:]
                    ssusiDF["sat"] = sat
                    ssusiDF["shapeArr"] = prpntLats.shape[0]
                    # # reset index, we need datetime as a col
                    ssusiDF = ssusiDF.reset_index()
                    ssusiDF = ssusiDF.rename(columns = {"index":"date"})
                    outCols = ["date", "sat", "orbitNum", "shapeArr"] + latColList + lonColList + d121ColList + \
                                d130ColList + d135ColList + dLBHSColList + dLBHLColList
                    ssusiDF = ssusiDF[ outCols ]
                    self.dataset = pd.concat([
                        self.dataset,
                        ssusiDF
                    ])
            self.dataset.to_csv(fname, header=True)
        return

    def get_pole_times(self, hemi="north"):
        self.poleTimesDict = {}
        for sat_id in self.sats:
            ds = self.dataset[self.dataset.sat==sat_id]
            if "mlat.1" in ds.columns:
                latCols = [col for col in ds if col.startswith("mlat")]             
            else:
                latCols = [col for col in ds if col.startswith("glat")]
                selCols = [ "orbitNum", "date" ] + latCols
            cutOffLat = 85.
            if hemi == "north":
                poleLat = 90.
                evalStr = "(ds['{0}'] >" + str( int(cutOffLat) ) + ".)" #
            else:
                poleLat = -90.
                evalStr = "(ds['{0}'] <" + str( int(-1*cutOffLat) ) + ".)" #
            ds = ds[selCols][eval(" | ".join([\
                evalStr.format(col) 
                for col in latCols]))].reset_index(drop=True)
            for l in latCols:
                    ds[l] = abs(ds[l] - poleLat)
            ds["coLatSum"] = ds[latCols].sum(axis=1)
            # groupby orbit to get min colatsum
            poleTimesDF = ds[ ["orbitNum", "coLatSum"]\
                ].groupby( "orbitNum" ).min().reset_index()
            poleTimesDF = pd.merge( poleTimesDF, ds,\
                                on=["orbitNum", "coLatSum"] )
            # sometimes we get multiple rows for same orbit,
            # so we"ll just take the first row
            poleTimesDF = poleTimesDF.groupby("orbitNum").first()
            self.poleTimesDict[sat_id] = poleTimesDF["date"].values
        return self.poleTimesDict

    def filter_data_by_time(
        self, time, hemi="north", 
        timeDelta=20, filterLat=40.
    ):
        filteredDict = {}
        for sat_id in self.sats:
            ds = self.dataset[self.dataset.sat==sat_id]
            ds = ds.fillna(0.)
            timeMin = time - dt.timedelta(minutes=timeDelta)
            timeMax = time + dt.timedelta(minutes=timeDelta)
            ds = ds[ (ds["date"] >= timeMin) &\
                    (ds["date"] <= timeMax) ]
            if hemi == "north":
                evalStr = "(ds['{0}'] >" + str( int(filterLat) ) + ".)" #
                filterCol = [col for col in ds if col.startswith("glat")]
                if len(filterCol) > 0:
                    ds = ds[eval(" & ".join([\
                            evalStr.format(col) 
                            for col in filterCol]))].reset_index(drop=True)
                else:
                    filterCol = [col for col in ds if col.startswith("glat")]
                    ds = ds[eval(" & ".join([\
                            evalStr.format(col) 
                            for col in filterCol]))].reset_index(drop=True)
            else:
                evalStr = "(ds['{0}'] <" + str( int(-1*filterLat) ) + ".)" #
                filterCol = [col for col in ds if col.startswith('mlat')]
                if len(filterCol) > 0:
                    ds = ds[eval(" & ".join([\
                            evalStr.format(col) 
                            for col in filterCol]))].reset_index(drop=True)
                else:
                    filterCol = [col for col in ds if col.startswith('glat')]
                    ds = ds[eval(" & ".join([\
                            evalStr.format(col) 
                            for col in filterCol]))].reset_index(drop=True)
            if ds.shape[0] == 0:
                logger.info("********NO DATA FOUND FOR "  + str(sat_id) + ", CHECK FOR A " +\
                         "DIFFERENT TIME OR INCREASE TIMEDEL********")
            # Sort the DF by time, since orbits mess it up
            ds = ds.sort_values("date")
            filteredDict[sat_id] = ds
        return filteredDict


if __name__ == "__main__":
    d = DMSP()
    d.download()
    d.read_DMSP()