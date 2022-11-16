import matplotlib.pyplot as plt
plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans",
                                   "Lucida Grande", "Verdana"]

import matplotlib as mpl

import pandas as pd

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import datetime as dt
import numpy as np
from scipy.ndimage import gaussian_filter as GF

import cartopy.crs as ccrs
from cartopy.feature.nightshade import Nightshade
import datetime as dt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def grid_by_latlon_cell(
    q, xcell, ycell, dx, dy
):
    xparam, yparam, zparam = "glon", "gdlat", "tec"
    paramDF = q[ [xparam, yparam, zparam] ]
    X, Y  = np.meshgrid( xcell, ycell )
    Z = np.zeros_like(X)*np.nan
    for i, x in enumerate(xcell):
        for j, y in enumerate(ycell):
            df = paramDF[
                (paramDF[xparam]>=x) &
                (paramDF[xparam]<x+dx) &
                (paramDF[yparam]>=y) &
                (paramDF[yparam]<y+dy)
            ]
            if len(df) > 0: Z[j, i] = np.nanmean(df[zparam])
    Z = np.ma.masked_invalid(Z)
    return X,Y,Z

def get_gridded_parameters(q, xparam="time", yparam="slist", zparam="v", round=False):
    """
    Method converts scans to "beam" and "slist" or gate
    """
    plotParamDF = q[ [xparam, yparam, zparam] ]
    if round:
        plotParamDF[xparam] = np.array(plotParamDF[xparam]).astype(int)
        plotParamDF[yparam] = np.array(plotParamDF[yparam]).astype(int)
    plotParamDF = plotParamDF.groupby( [xparam, yparam] ).agg(np.nanmean).reset_index()
    plotParamDF = plotParamDF[ [xparam, yparam, zparam] ].pivot( xparam, yparam )
    x = plotParamDF.index.values
    y = plotParamDF.columns.levels[1].values
    X, Y  = np.meshgrid( x, y )
    # Mask the nan values! pcolormesh can't handle them well!
    Z = np.ma.masked_where(
            np.isnan(plotParamDF[zparam].values),
            plotParamDF[zparam].values)
    return X,Y,Z


class RangeTimePlot(object):
    """
    Create plots for IS/GS flags, velocity, and algorithm clusters.
    """
    def __init__(self, fov, nrang, unique_times, fig_title, num_subplots=3):
        self.fov = fov
        self.nrang = nrang
        self.unique_gates = np.linspace(1, nrang, nrang)
        self.unique_times = unique_times
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(8, 3*num_subplots), dpi=240) # Size for website
        plt.suptitle(fig_title, x=0.075, y=0.95, ha="left", fontweight="bold", fontsize=18)
        mpl.rcParams.update({"xtick.labelsize": 15, "ytick.labelsize":15, "font.size":15})
        return
    
    def addParamPlot(self, df, beam, title, p_max=100, p_min=-100, xlabel="Time UT",
             ylabel="Range gate", zparam="v", label="Velocity [m/s]", add_gflg=False, sza_th=108):
        ax = self._add_axis()
        df = df[df.bmnum==beam]
        X, Y, Z = get_gridded_parameters(df, xparam="time", yparam="slist", zparam=zparam)
        cmap = plt.cm.jet_r
        # cmap.set_bad("w", alpha=0.0)
        # Configure axes
        ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_major_locator(hours)
        minutes = mdates.MinuteLocator(byminute=range(0, 60, 30))
        ax.xaxis.set_minor_locator(minutes)
        ax.set_xlabel(xlabel, fontdict={"size":15, "fontweight": "bold"})
        ax.set_xlim(self.unique_times)
        ax.set_ylim([180, self.nrang])
        ax.set_ylabel(ylabel, fontdict={"size":15, "fontweight": "bold"})
        ax.set_title(title, loc="right", fontdict={"fontweight": "bold"})
        if add_gflg:
            Xg, Yg, Zg = get_gridded_parameters(df, xparam="time", yparam="slist", zparam="gflg")
            Zx = np.ma.masked_where(Zg==0, Zg)
            ax.pcolormesh(Xg, Yg, Zx.T, lw=0.01, edgecolors="None", cmap="gray",
                        vmax=2, vmin=0, shading="nearest")
            Z = np.ma.masked_where(Zg==1, Z)
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest")
        else:
            im = ax.pcolormesh(X, Y, Z.T, lw=0.01, edgecolors="None", cmap=cmap,
                        vmax=p_max, vmin=p_min, shading="nearest")
        self._add_colorbar(im, ax, cmap, label=label)
        self.overlay_sza(ax, df.time.unique(), beam, [0, np.max(df.gate)], 
                df.rsep.iloc[0], df.frang.iloc[0], sza_th)
        return ax

    def overlay_sza(self, ax, times, beam, gate_range, rsep, frang, th=108):
        R = 6378.1
        from pysolar.solar import get_altitude
        gates = np.arange(gate_range[0], gate_range[1])
        SZA = np.zeros((len(times), len(gates)))
        for i, d in enumerate(times):
            d = dt.datetime.utcfromtimestamp(d.astype(dt.datetime) * 1e-9).replace(tzinfo=dt.timezone.utc)
            for j, g in enumerate(gates):
                gdlat, glong = self.fov[0][g, beam],self.fov[1][g, beam]
                sza = 90.-get_altitude(gdlat, glong, d)
                if (sza > 85.) & (sza < 120.): sza += np.rad2deg(np.arccos(R/(R+300)))
                SZA[i,j] = sza
        ZA = np.zeros_like(SZA)
        ZA[SZA>th] = 1.
        ZA[SZA<=th] = 0.
        times, gates = np.meshgrid(times, frang + (rsep*gates))
        ax.pcolormesh(times.T, gates.T, ZA, lw=0.01, edgecolors="None", cmap="gray_r",
                        vmax=2, vmin=0, shading="nearest", alpha=0.3)
        return

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        ax.tick_params(axis="both", labelsize=15)
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")

    def close(self):
        self.fig.clf()
        plt.close()

    def _add_colorbar(self, im, ax, colormap, label=""):
        """
        Add a colorbar to the right of an axis.
        :param fig:
        :param ax:
        :param colormap:
        :param label:
        :return:
        """
        import matplotlib as mpl
        pos = ax.get_position()
        cpos = [pos.x1 + pos.width * 0.01, pos.y0 + pos.height*.1,
                0.01, pos.height * 0.8]                # this list defines (left, bottom, width, height
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(im, cax,
                   spacing="uniform",
                   orientation="vertical", 
                   cmap=colormap)
        cb2.set_label(label)
        return

class CartoBase(object):
    """
    This class holds cartobase code for the
    SD, SMag, and GPS TEC dataset.
    """

    def __init__(self, date, xPanels=1, yPanels=1, range=[-150, -70, 20, 90], basetag=0, ytitlehandle=0.95):
        self.date = date
        self.xPanels = xPanels
        self.yPanels = yPanels
        self.range = range
        self.basetag = basetag
        self._num_subplots_created = 0
        self.fig = plt.figure(figsize=(4.5*yPanels, 4.5*xPanels), dpi=240) # Size for website
        mpl.rcParams.update({"xtick.labelsize": 15, "ytick.labelsize":15, "font.size":15})
        self.ytitlehandle = ytitlehandle
        self.proj = {
            "to": ccrs.Orthographic(-110, 60),
            #"to": ccrs.NorthPolarStereo(-90),
            "from": ccrs.PlateCarree(),
        }
        return

    def _add_axis(self, draw_labels=True):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(
            self.xPanels, self.yPanels, 
            self._num_subplots_created,
            projection=self.proj["to"],
        )
        ax.tick_params(axis="both", labelsize=15)
        ax.add_feature(Nightshade(self.date, alpha=0.3))
        ax.set_global()
        ax.coastlines(color="k", alpha=0.5, lw=0.5)
        gl = ax.gridlines(crs=self.proj["from"], linewidth=0.3, 
            color="k", alpha=0.5, linestyle="--", draw_labels=draw_labels)
        gl.xlocator = mticker.FixedLocator(np.arange(-180,180,30))
        gl.ylocator = mticker.FixedLocator(np.arange(-90,90,20))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if self.range: ax.set_extent(self.range)
        tag = chr(96 + self._num_subplots_created + self.basetag)
        # ax.text(0.05, 1.05, , 
        #     ha="left", va="bottom", transform=ax.transAxes)
        plt.suptitle("(%s) "%tag+self.date.strftime("%d %b %Y, %H:%M UT"), 
            x=0.5, y=self.ytitlehandle, ha="center", va="bottom", fontweight="bold", fontsize=15)
        return ax

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def _fetch_axis(self, draw_labels=True):
        if not hasattr(self, "ax"): self.ax = self._add_axis(draw_labels)
        return

    def add_radars(self, radars, draw_labels=True):
        self._fetch_axis(draw_labels)
        for r in radars.keys():
            rad = radars[r]
            self.overlay_radar(rad)
            self.overlay_fov(rad)
            self.ovrlay_radar_data(rad, draw_labels)
        return

    def add_magnetometers(self, mags, draw_labels=True):
        self._fetch_axis(draw_labels)
        mags = mags[mags.tval==self.date]
        for i, mag in mags.iterrows():
            self.overlay_magnetometer(mag)
        self.ovrlay_magnetometer_data(mags, tag=draw_labels)
        return

    def overlay_magnetometer(
        self, mag, marker="D", zorder=2, 
        markerColor="r", markerSize=2
    ):
        lat, lon = mag.glat, mag.glon
        self.ax.scatter([lon], [lat], s=markerSize, marker=marker,
            color=markerColor, zorder=zorder, transform=self.proj["from"], lw=0.8, alpha=0.4)
        return

    def ovrlay_magnetometer_data(
        self, mags, scalef=300, lkey=100, tag=False
    ):
        lat, lon = np.array(mags.glat), np.array(mags.glon)
        # Extract H-component
        N, E = np.array(mags.N_geo), np.array(mags.E_geo)
        xyz = self.proj["to"].transform_points(self.proj["from"], lon, lat)
        x, y = xyz[:, 0], xyz[:, 1]
        # Add Quiver for H-Component
        self.ax.scatter(x, y, color="r", s=2)
        ql = self.ax.quiver(
            x,
            y,
            E,
            N,
            scale=scalef,
            headaxislength=0,
            linewidth=0.0,
            scale_units="inches",
            color="r"
        )
        if tag:
            self.ax.quiverkey(
                ql,
                0.95,
                1.05,
                lkey,
                str(lkey) + " nT",
                labelpos="N",
                transform=self.proj["from"],
                color="r",
                fontproperties={"size": 15},
            )
        return

    def overlay_radar(
        self, rad, marker="D", zorder=2, markerColor="k", 
        markerSize=2, fontSize="small", font_color="darkblue", xOffset=-5, 
        yOffset=-1.5, annotate=True,
    ):
        """ Adding the radar location """
        lat, lon = rad.hdw.geographic.lat, rad.hdw.geographic.lon
        self.ax.scatter([lon], [lat], s=markerSize, marker=marker,
            color=markerColor, zorder=zorder, transform=self.proj["from"], lw=0.8, alpha=0.4)
        nearby_rad = [["adw", "kod", "cve", "fhe", "wal", "gbr", "pyk", "aze", "sys"],
                    ["ade", "ksr", "cvw", "fhw", "bks", "sch", "sto", "azw", "sye"]]
        if annotate:
            rad = rad.rad
            if rad in nearby_rad[0]: xOff, ha = -5 if not xOffset else -xOffset, -2
            elif rad in nearby_rad[1]: xOff, ha = 5 if not xOffset else xOffset, -2
            else: xOff, ha = xOffset, -1
            x, y = self.proj["to"].transform_point(lon+xOff, lat+ha, src_crs=self.proj["from"])
            self.ax.text(x, y, rad.upper(), ha="center", va="center", transform=self.proj["to"],
                        fontdict={"color":font_color, "size":fontSize}, alpha=0.8)
        return

    def overlay_fov(
        self, rad, maxGate=110, rangeLimits=None, beamLimits=None,
        model="IS", fov_dir="front", fovColor=None, fovAlpha=0.2,
        fovObj=None, zorder=1, lineColor="k", lineWidth=0.5, ls="-"
    ):
        """ Overlay radar FoV """
        from numpy import transpose, ones, concatenate, vstack, shape
        hdw = rad.hdw
        sgate = 0
        egate = hdw.gates if not maxGate else maxGate
        ebeam = hdw.beams
        if beamLimits is not None: sbeam, ebeam = beamLimits[0], beamLimits[1]
        else: sbeam = 0
        latFull, lonFull = rad.fov[0].T, rad.fov[1].T
        xyz = self.proj["to"].transform_points(self.proj["from"], lonFull, latFull)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        contour_x = concatenate((x[sbeam, sgate:egate], x[sbeam:ebeam, egate],
                    x[ebeam, egate:sgate:-1],
                    x[ebeam:sbeam:-1, sgate]))
        contour_y = concatenate((y[sbeam, sgate:egate], y[sbeam:ebeam, egate],
                y[ebeam, egate:sgate:-1],
                y[ebeam:sbeam:-1, sgate]))
        self.ax.plot(contour_x, contour_y, color=lineColor, 
            zorder=zorder, linewidth=lineWidth, ls=ls, alpha=1.0)
        return

    def ovrlay_radar_data(self, rad, cbar=False):
        data = rad.df[
            (rad.df.time>=self.date) &
            (rad.df.time<self.date+dt.timedelta(minutes=2)) &
            (rad.df.slist<=110)
        ]
        kwargs = {"rad": rad}
        # add a function to create GLAT/GLON in Data
        data = data.apply(self.convert_to_latlon, axis=1, **kwargs)
        # Grid based on GLAT/GLON
        X, Y, Z = get_gridded_parameters(data, "glon", "glat", "v")
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        im = self.ax.scatter(
            x, y, c=Z.T,
            cmap="jet_r",
            vmin=-300,
            vmax=300,
            transform=self.proj["to"],
            alpha=0.6,
            s=1.2
        )
        if cbar: self._add_hcolorbar(im, label="Velocity [m/s]")
        return

    def convert_to_latlon(self, row, rad):
        row["glat"], row["glon"] = (
            rad.fov[0].T[row["bmnum"], row["slist"]],
            rad.fov[1].T[row["bmnum"], row["slist"]],
        )
        return row
    
    def add_dTEC(self, X, Y, Z):
        self.ax = self._add_axis(draw_labels=False)
        # Plot based on transcript
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        im = self.ax.pcolor(
            x, y, Z.T,
            cmap="jet",
            vmin=-3,
            vmax=3,
            transform=self.proj["to"],
            alpha=0.6
        )
        self._add_hcolorbar(im, label="dTEC [TECu]")
        return

    def add_TEC_gradient(self, X, Y, dxZ, dyZ, tag=False):
        self.ax = self._add_axis(draw_labels=tag)
        # Plot based on transcript
        xyz = self.proj["to"].transform_points(self.proj["from"], X, Y)
        x, y = xyz[:, :, 0], xyz[:, :, 1]
        self.ax.scatter(x, y, color="k", s=2)
        ql = self.ax.quiver(
            x,
            y,
            dxZ,
            dyZ,
            scale=1.,
            headaxislength=0,
            linewidth=0.1,
            scale_units="inches",
        )
        if tag:
            self.ax.quiverkey(
                ql,
                1.25,
                1.,
                0.3,
                r"$\nabla_{\lambda,\phi}TEC$:0.3",
                labelpos="N",
                transform=self.proj["from"],
                color="k",
                fontproperties={"size": 15},
            )
        return

    def _add_hcolorbar(self, im, colormap="jet_r", label=""):
        """Add a colorbar to the right of an axis."""
        pos = self.ax.get_position()
        cpos = [
            pos.x0 + 0.3 * pos.width,
            pos.y0 - 0.15 * pos.height,
            pos.width * 0.5,
            0.02,
        ]  # this list defines (left, bottom, width, height)
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(
            im,
            cax=cax,
            cmap=colormap,
            spacing="uniform",
            orientation="horizontal",
        )
        cb2.set_label(label)
        return

    def _add_colorbar(self, im, colormap="jet_r", label=""):
        """Add a colorbar to the right of an axis."""
        pos = self.ax.get_position()
        cpos = [
            pos.x1 + 0.15,
            pos.y0 + 0.2 * pos.height,
            0.02,
            pos.height * 0.5,
        ]  # this list defines (left, bottom, width, height)
        cax = self.fig.add_axes(cpos)
        cb2 = self.fig.colorbar(
            im,
            cax=cax,
            cmap=colormap,
            spacing="uniform",
            orientation="vertical",
        )
        cb2.set_label(label)
        return

    def overlay_DMSP_data(
        self, dmsp_df, plotType="dlbhs",
        autoScale=True, vmin=0., vmax=1000.,
        ssusiCmap="Greens", tag=False
    ):
        ds = dmsp_df.copy()
        ssusiDisk = ds[
            ds.columns[pd.Series(ds.columns).str.startswith(plotType)
        ]].values
        if autoScale:
            vmin, vmax = 0, (int(np.median(ssusiDisk/500.))+2)*500.
        lats = ds[
            ds.columns[pd.Series(ds.columns).str.startswith("glat")
        ]].values
        lons = ds[
            ds.columns[pd.Series(ds.columns).str.startswith("glon")
        ]].values
        ssusiDisk = GF(ssusiDisk, 1.2)
        xyz = self.proj["to"].transform_points(self.proj["from"], lons, lats)
        X, Y = xyz[:, :, 0], xyz[:, :, 1]
        im = self.ax.pcolor(
            X, Y, ssusiDisk,
            cmap=ssusiCmap,
            vmin=vmin,
            vmax=vmax,
            transform=self.proj["to"],
            alpha=0.4
        )
        if tag: self._add_colorbar(im, label="Rayleighs")
        return

def plot_FISM2(omni):
    # f201710 = pd.read_csv("database/fism_flare_hr-201710.csv")
    # f201710.time = f201710.time.apply(
    #     lambda x: dt.datetime(1970,1,1) + dt.timedelta(seconds=x)
    # )
    # f201710 = f201710[
    #     (f201710.time>=dt.datetime(2017, 9, 10, 16, 1)) & 
    #     (f201710.time<dt.datetime(2017, 9, 10, 16, 2))
    # ]
    f201706 = pd.read_csv("database/fism_flare_hr-201706.csv")
    f201706.time = f201706.time.apply(
        lambda x: dt.datetime(1970,1,1) + dt.timedelta(seconds=x)
    )
    f201706flr = f201706[
        (f201706.time>=dt.datetime(2017, 9, 6, 12, 2)) &
        (f201706.time<dt.datetime(2017, 9, 6, 12, 3))
    ]

    fig = plt.figure(figsize=(8, 3), dpi=200)
    ax = fig.add_subplot(111)
    ax.tick_params(axis="both", labelsize=15)
    #ax.semilogy(f201710.wavelength, f201710.irradiance, "r-", lw=0.8, label="X8.5, 10 September 2003")
    ax.semilogy(f201706flr.wavelength, f201706flr.irradiance, 
            "k-", lw=0.8, label="12:02 UT")
    ax.set_title("Flare Class X9.3 / 6 September 2017", ha="left", va="center", fontdict={"size":15})
    ax.legend(loc=1, prop={"size":15})
    ax.set_xlabel("Wavelength [nm]", fontdict={"size":15, "fontweight": "bold"})
    ax.set_xlim([0, 50])
    #ax.set_ylim([1e-6, 1e-1])
    ax.set_ylabel(r"Irradiance [$W/m^2/nm$]", fontdict={"size":15, "fontweight": "bold"})
    fig.savefig("figures/fism2.png", bbox_inches="tight")

    f201706 = pd.read_csv("database/2017-0.15.csv")
    f201706.time = f201706.time.apply(
        lambda x: dt.datetime(1970,1,1) + dt.timedelta(seconds=x)
    )
    print(f201706.head())
    fig = plt.figure(figsize=(8, 4*3), dpi=200)
    ax = fig.add_subplot(411)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=15)
    ax.semilogy(f201706.time, f201706.irradiance, 
            "k-", lw=0.8, label="12:02 UT")
    ax.set_title("Flare Class X9.3 / 6 September 2017", ha="left", va="center", fontdict={"size":15})
    ax.legend(loc=1, prop={"size":15})
    #ax.set_xlabel("Time [UT]", fontdict={"size":15, "fontweight": "bold"})
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.axvline(dt.datetime(2017,9,6,12,2), ls="--", color="r", lw=0.9)
    ax.set_ylabel(r"Irradiance [$W/m^2/nm$]", fontdict={"size":15, "fontweight": "bold"})
    ax.text(0.1,0.9,"(a)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    #fig.savefig("figures/fism2-0.15.png", bbox_inches="tight")

    #fig = plt.figure(figsize=(8, 9), dpi=200)
    ax = fig.add_subplot(412)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.Bz, "r.", ms=1, ls="None", label=r"$B_z$")
    ax.plot(omni.date, omni.By, "b.", ms=1, ls="None", label=r"$B_y$")
    ax.legend(loc=1, fontsize=15)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.set_ylim(-7, 7)
    ax.set_ylabel("IMF [nT]", fontdict={"size":15, "fontweight": "bold"})
    ax.text(0.1,0.9,"(b)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    ax.axvline(dt.datetime(2017,9,6,12,2), color="k", ls="--", lw=0.9)
    ax = fig.add_subplot(413)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.AU, "r-", lw=1, label="AU")
    ax.plot(omni.date, omni.AL, "b-", lw=1, label="AL")
    ax.plot(omni.date, omni.AE, "k-", lw=1, label="AE")
    ax.legend(loc=1, fontsize=15)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.set_ylabel("AE/AL/AU [nT]", fontdict={"size":15, "fontweight": "bold"})
    ax.text(0.1,0.9,"(c)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    ax.set_ylim(-500, 1000)
    ax.axvline(dt.datetime(2017,9,6,12,2), color="k", ls="--", lw=0.9)
    ax = fig.add_subplot(414)
    ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
    ax.tick_params(axis="both", labelsize=15)
    ax.plot(omni.date, omni.SYMH, "k", lw=1, label="SYM-H")
    ax.plot(omni.date, omni.ASYH, "r--", lw=1, label="ASY-H")
    ax.legend(loc=1, fontsize=15)
    ax.set_ylabel("A/SYM-H [nT]", fontdict={"size":15, "fontweight": "bold"})
    ax.set_ylim(-100, 100)
    ax.set_xlim([dt.datetime(2017,9,6), dt.datetime(2017,9,7)])
    ax.axvline(dt.datetime(2017,9,6,12,2), color="k", ls="--", lw=0.9)
    ax.set_xlabel("Time [UT]", fontdict={"size":15, "fontweight": "bold"})
    ax.text(0.1,0.9,"(d)",ha="left",va="center",transform=ax.transAxes, 
            fontdict={"size":15, "fontweight": "bold"})
    fig.savefig("figures/omni.png", bbox_inches="tight")
    return