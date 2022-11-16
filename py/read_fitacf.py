"""
    This python module is used to read the dataset from fitacf/fitacf3 
    level dataset.
"""

import os
import pandas as pd
import pydarn 
import glob
import bz2
from loguru import logger
import datetime as dt
import numpy as np
from scipy import constants as C

from rad_fov import CalcFov

class Radar(object):

    def __init__(self, rad, dates=None, clean=False, type="fitacf3",):
        logger.info(f"Initialize radar: {rad}")
        self.rad = rad
        self.dates = dates
        self.clean = clean
        self.type = type
        self.__setup__()
        self.__fetch_data__()
        self.calculate_decay_rate()
        return
    
    def __setup__(self):
        logger.info(f"Setup radar: {self.rad}")
        self.files = glob.glob(f"database/{self.type}/*{self.rad}.*")
        self.files.sort()
        self.hdw = pydarn.read_hdw_file(self.rad)
        #self.fov = pydarn.Coords.GEOGRAPHIC(self.hdw.stid)
        fov = CalcFov(hdw=self.hdw)
        self.fov = (fov.latFull.T, fov.lonFull.T)
        logger.info(f"Files: {len(self.files)}")
        return

    def get_glatlon(self, row):
        from pysolar.solar import get_altitude
        bm, gate = row["bmnum"], row["slist"]
        row["gdlat"], row["glong"] = self.fov[0][gate, bm], self.fov[1][gate, bm]
        date = row["time"].to_pydatetime().replace(tzinfo=dt.timezone.utc)
        row["sza"] = 90.-get_altitude(row["gdlat"], row["glong"], date)
        return row

    def __fetch_data__(self):
        if self.clean: os.remove(f"database/{self.rad}.{self.type}.csv")
        if os.path.exists(f"database/{self.rad}.{self.type}.csv"):
            self.df = pd.read_csv(f"database/{self.rad}.{self.type}.csv", parse_dates=["time"])
        else:
            records = []
            for f in self.files:
                logger.info(f"Reading file: {f}")
                with bz2.open(f) as fp:
                    reader = pydarn.SuperDARNRead(fp.read(), True)
                    records += reader.read_fitacf()
            if len(records)>0:
                self.__tocsv__(records)
        self.df.tfreq = np.round(np.array(self.df.tfreq)/1e3, 1)
        return

    def __tocsv__(self, records):
        time, v, slist, p_l, frang, scan, beam,\
            w_l, gflg, elv, phi0, tfreq, rsep = (
            [], [], [],
            [], [], [],
            [], [], [],
            [], [], [],
            [],
        )
        for r in records:
            if "v" in r.keys():
                t = dt.datetime(
                    r["time.yr"], 
                    r["time.mo"],
                    r["time.dy"],
                    r["time.hr"],
                    r["time.mt"],
                    r["time.sc"],
                    r["time.us"],
                )
                time.extend([t]*len(r["v"]))
                tfreq.extend([r["tfreq"]]*len(r["v"]))
                rsep.extend([r["rsep"]]*len(r["v"]))
                frang.extend([r["frang"]]*len(r["v"]))
                scan.extend([r["scan"]]*len(r["v"]))
                beam.extend([r["bmnum"]]*len(r["v"]))
                v.extend(r["v"])
                gflg.extend(r["gflg"])
                slist.extend(r["slist"])
                p_l.extend(r["p_l"])
                w_l.extend(r["w_l"])
                if "elv" in r.keys(): elv.extend(r["elv"])
                if "phi0" in r.keys(): phi0.extend(r["phi0"])                
            
        self.df = pd.DataFrame()
        self.df["v"] = v
        self.df["gflg"] = gflg
        self.df["slist"] = slist
        self.df["bmnum"] = beam
        self.df["p_l"] = p_l
        self.df["w_l"] = w_l
        if len(elv) > 0: self.df["elv"] = elv
        if len(phi0) > 0: self.df["phi0"] = phi0
        self.df["time"] = time
        self.df["tfreq"] = tfreq
        self.df["scan"] = scan
        self.df["rsep"] = rsep
        self.df["frang"] = frang

        if self.dates:
            self.df = self.df[
                (self.df.time>=self.dates[0]) & 
                (self.df.time<=self.dates[1])
            ]
        self.df = self.df.apply(self.get_glatlon, axis=1)
        self.df.to_csv(f"database/{self.rad}.{self.type}.csv", index=False, header=True)
        return

    def recalculate_elv_angle(self, XOff=0, YOff=100, ZOff=0):
        return

    def calculate_decay_rate(self):
        logger.info(f"Calculate Decay")
        f, w_l = np.array(self.df.tfreq)*1e6, np.array(self.df.w_l)
        k = 2*np.pi*f/C.c
        self.df["tau_l"] = 1.e3/(k*w_l)
        return

if __name__ == "__main__":
    Radar("sas")
    Radar("kod")
    Radar("pgr")
    Radar("fhw")
    Radar("cvw")
    Radar("cve")