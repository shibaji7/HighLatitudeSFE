import datetime as dt
from loguru import logger
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter as GF
import glob

from read_fitacf import Radar
from smag import SuperMAG
from plot import RangeTimePlot, CartoBase, plot_FISM2,\
    grid_by_latlon_cell
from dmsp import DMSP

from scipy.interpolate import interp2d
from multiprocessing import Pool

def parse_TEC_dataset(start, hour_window = [15, 18], day=10):
    import h5py
    fname = f"database/los_201709{'%02d'%day}.002.h5"
    print(f"Start - {start}")
    dfname = f"database/TEC_LoS.{start}-{day}.csv"
    start, stop= start*1000000, ( ((start+1)*1000000) - 1)
    if not os.path.exists(dfname):
        dat = pd.read_hdf(
            fname,"/Data/Table Layout",
            columns=["hour", "min", "sec", "tec", "gdlat", "glon"],
            start=start, stop=stop
        )
        dat = dat[
            (dat.hour>=hour_window[0]) & (dat.hour<=hour_window[1])
        ]
        if len(dat) > 0:
            # Extract date
            dat["date"] = dat.apply(
                lambda row: dt.datetime(2017,9,day,row["hour"],row["min"],row["sec"]),
                axis=1
            )
            # Select required elements
            dat = dat[["date","tec", "gdlat", "glon"]]
            # Select 3-hour inteval
            dat = dat[
                (dat.date>=dt.datetime(2017,9,day,hour_window[0])) &
                (dat.date<=dt.datetime(2017,9,day,hour_window[1]))
            ]
            if len(dat) > 0: dat.to_csv(dfname, header=True, index=False)
    return

class Analysis(object):

    def __init__(self, rads, dates, base, dlon=5, dlat=5, range=[-150, -70, 20, 90]):
        self.rads = rads
        self.dates = dates
        self.base = base
        self.radars = {}
        self.dlon = dlon
        self.dlat = dlat
        self.range = range
        self.smag = SuperMAG.FetchSM(
            "database/", 
            dates
        )
        self.fradars = {}
        for r in rads: 
            rad_data = Radar(r, dates)
            self.radars[r] = rad_data
        self.dmsp = DMSP()
        self.dmsp.read_DMSP()
        self.pole_times = self.dmsp.get_pole_times()
        return

    def extract_TEC(self, dates, lon_range=[-149, -69]):
        dMinutes = int((dates[1]-dates[0]).total_seconds()/60.)
        self.dateList = [dates[0]+dt.timedelta(minutes=i) for i in range(dMinutes)]
        
        files = glob.glob(f"database/TEC_LoS.*-{dates[0].day}.csv")
        files.sort()
        o = pd.DataFrame()
        for f in files:
            o = pd.concat([o, pd.read_csv(f, parse_dates=["date"])])
        o = o.sort_values(by="date")
        self.XArray, self.YArray, self.ZArray = {}, {}, {}
        x, y = np.arange(lon_range[0], lon_range[1], self.dlon), np.arange(20, 80, self.dlat)
        for u in self.dateList:
            logger.info(f"Processing TEC: {u.strftime('%H.%M')}")
            du = o[o.date==u]
            X, Y, Z = grid_by_latlon_cell(du, x, y, self.dlon, self.dlat)
            self.XArray[u] = X
            self.YArray[u] = Y
            self.ZArray[u] = Z
        return
    
    def filter_radar(self, rad, tfrq=None, gflg=None):
        df = self.radars[rad].df.copy()
        self.filter_summ = ""
        df["gate"] = np.copy(df.slist)
        df.slist = (df.slist*df.rsep) + df.frang
        df["unique_tfreq"] = df.tfreq.apply(lambda x: int(x/0.5)*0.5)
        logger.info(f"Unique tfreq: {df.unique_tfreq.unique()}")
        if tfrq: 
            df = df[df.unique_tfreq==tfrq]
            self.filter_summ += r"$f_0$=%.1f MHz"%tfrq + "\n"
        else:
            self.filter_summ += r"$f_0$="
            for tf in df.unique_tfreq.unique():
                self.filter_summ += ("%.1f, "%tf)
            self.filter_summ = self.filter_summ[:-2]+" MHz \n"
        if gflg is not None: 
            df = df[df.gflg==gflg]
            self.filter_summ += r"IS/GS$\sim$%d"%gflg
        self.fradars[rad] = df.copy()
        return

    def plotRTI(self, rad, time=None):
        df = self.fradars[rad].copy()
        for b in df.bmnum.unique():
            o = df.copy()
            rti = RangeTimePlot(self.radars[rad].fov, 
                    3000, self.dates, "", num_subplots=1)
            ax = rti.addParamPlot(o, b, self.filter_summ, p_max=300, p_min=-300,
                xlabel="Time [UT]", ylabel="Slant Range [km]", zparam="v", label="Velocity [m/s]",
                add_gflg=True)
            if time: ax.axvline(time, ls="--", lw=1.5, color="k")
            rti.save(f"{self.base}{rad}-{'%02d'%b}~{o.unique_tfreq.iloc[0]}.png")
            rti.close()
        return

    def nearest(self, items, pivot):
        return min(items, key=lambda x: abs(x - pivot))

    def plotMaps(self, dates, dmsp_sat="f17"):
        pole_times = [
            dt.datetime.utcfromtimestamp(
                pt.astype(dt.datetime) * 1e-9
            )
            for pt in self.pole_times[dmsp_sat]
        ]
        dMinutes = int((dates[1]-dates[0]).total_seconds()/60.)
        dateList = [dates[0]+dt.timedelta(minutes=i) for i in range(dMinutes)]
        Z0TEC = self.ZArray[dateList[0]]
        btags = {
            dt.datetime(2017,9,6,11,50): 0,
            dt.datetime(2017,9,6,12,2): 1,
            dt.datetime(2017,9,6,12,25): 2,
            dt.datetime(2017,9,6,13): 3,
            dt.datetime(2017,9,6,13,30): 4,
            dt.datetime(2017,9,6,13,59): 5,
        }
        gbTag = True
        for ij, d in enumerate(dateList):
            btag = 0
            if d in btags.keys():
                btag = btags[d]
            pole_time = self.nearest(pole_times, d)
            logger.info(f"Pole time: {pole_time}")
            dmsp_df = self.dmsp.filter_data_by_time(pole_time)[dmsp_sat]
            cb = CartoBase(
                d, xPanels=1, yPanels=1,
                range=self.range,
                basetag=btag,
                ytitlehandle=0.95 if ij==0 else 0.88,
            )

            # Adding dTEC maps
            X, Y, Z = (
                self.XArray[d], 
                self.YArray[d], 
                self.ZArray[d]
            )
            x, y = X[0,:], Y[:,0]
            dtZ = (Z-Z0TEC)
            dxZ, dyZ = np.gradient(GF(Z.T, 0.01), x, y)
            #cb.add_dTEC(X, Y, dtZ.T)
            cb.add_TEC_gradient(X, Y, dxZ, dyZ, gbTag)
            # Overlay 
            cb.overlay_DMSP_data(dmsp_df, tag=gbTag)

            # Add Radar plots
            cb.add_radars(self.radars, draw_labels=gbTag)
            # Add Magnetometers
            cb.add_magnetometers(self.smag.sm_data, draw_labels=gbTag)
            
            cb.save(f"{self.base}{d.strftime('%H-%M')}.png")
            cb.close()
            Z0TEC = Z
            gbTag = False
        return

def analyze_201710_event():
    base = "figures/10Sep/"
    os.makedirs(base, exist_ok=True)
    anl = Analysis(
        rads=[],
        dates=[
            dt.datetime(2017,9,10,15,30),
            dt.datetime(2017,9,10,16,30) 
        ],
        base = base,
        range=[-180, -100, 20, 90]
    )
    dates = [
        dt.datetime(2017,9,10,15,50),
        dt.datetime(2017,9,10,16,50) 
    ]
    anl.extract_TEC(dates, lon_range=[-179, -99])
    anl.plotMaps(dates)
    return

def analyze_201706_event():
    base = "figures/06Sep/"
    os.makedirs(base, exist_ok=True)
    rads=["sas"]
    anl = Analysis(
        rads=rads,
        dates=[
            dt.datetime(2017,9,6,11),
            dt.datetime(2017,9,6,17)
        ],
        base = base,
    )
    for rad in rads:
        anl.filter_radar(rad)
        #anl.plotRTI(rad, dt.datetime(2017,9,6,12,2))

    dates = [
        dt.datetime(2017,9,6,11,50),
        dt.datetime(2017,9,6,14) 
    ]
    anl.extract_TEC(dates)
    anl.plotMaps(dates)
    return

def parse_omni():
    o = []
    with open("database/omni.csv", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = list(filter(None, line.replace("\n", "").split(" "))) 
            o.append({
                "date": dt.datetime(int(line[0]), 1, 1) +\
                    dt.timedelta(int(line[1])-1) + dt.timedelta(hours=int(line[2])) +\
                    dt.timedelta(minutes=int(line[3])),
                "Bx": float(line[4]), "By": float(line[5]), "Bz": float(line[6]),
                "V": float(line[7]), "AE": float(line[8]), "AL": float(line[9]),
                "AU": float(line[10]), "SYMD": float(line[11]), "SYMH": float(line[12]),
                "ASYD": float(line[13]), "ASYH": float(line[14])
            })
    o = pd.DataFrame.from_records(o)
    o = o.replace(99999.9, np.nan)
    return o

def plot_GDI(latitude):
    radars = {}
    dates=[
            dt.datetime(2017,9,6,11),
            dt.datetime(2017,9,6,17)
        ]
    rads, flist =["sas", "pgr", "kod"], [10.5, 10.5, 10.5]
    if latitude=="Middle": rads, flist =["fhw", "cve", "cvw"], [10.5, 10.5, 10.5]
    for r in rads: 
        rad_data = Radar(r, dates)
        radars[r] = rad_data
    tag = "a" if latitude=="High" else "b"
    name = f"({tag}) {latitude}-Latitude radars"
    rti = RangeTimePlot(None, 3000, dates, name, num_subplots=len(rads))
    th = 108. if latitude=="High" else 114.
    for i, r in enumerate(rads):
        xlabel = "Time [UT]" if i == len(rads)-1 else ""
        o = radars[r].df
        o["gate"] = np.copy(o.slist)
        o.slist = (o.slist*o.rsep) + o.frang
        o["unique_tfreq"] = o.tfreq.apply(lambda x: int(x/0.5)*0.5)
        summ = r"(%d) Rad: %s / Beam: 7"%(i+1, r)
        if latitude=="High": o = o[o.unique_tfreq==flist[i]]
        if flist[i]: 
            summ += "\n"
            summ += r"$f_0$=%.1f MHz"%flist[i]
        logger.info(f"Unique tfreq: {o.unique_tfreq.unique()}")
        rti.fov = radars[r].fov
        ax = rti.addParamPlot(o, 7, "", p_max=300, p_min=-300,
            xlabel=xlabel, ylabel="Slant Range [km]", zparam="v", label="Velocity [m/s]",
            add_gflg=False, sza_th=th)
        ax.text(0.05, 0.8, summ, ha="left", va="center", transform=ax.transAxes)
        ax.axvline(dt.datetime(2017,9,6,12,2), ls="--", lw=1.5, color="k")
    rti.save(f"figures/06Sep/GDI-{latitude}.png")
    rti.close()
    return

if __name__ == "__main__":
    # with Pool(4) as p:
    #     p.map(parse_TEC_dataset, np.arange(0,3000))
    #plot_FISM2(parse_omni())
    analyze_201706_event()
    #analyze_201710_event()
    #plot_GDI("High")
    #plot_GDI("Middle")