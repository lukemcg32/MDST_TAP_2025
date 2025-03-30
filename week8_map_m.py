import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.widgets import Slider

proj = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(11, 6))
ax.set_extent([-125, -65, 24, 50], crs=ccrs.PlateCarree())

# TODO: Add map features (land, ocean, lakes, borders, states, coastline)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.LAND, linewidth=0.5, edgecolor='black', facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, linewidth=0.5, edgecolor='black', facecolor='lightblue')
ax.add_feature(cfeature.LAKES, linewidth=0.5, edgecolor='black', facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black')
ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black', facecolor='lightgray')


ax.set_title("Click on a location to predict severity", fontsize=14, pad=20)

plt.subplots_adjust(top=0.88, bottom=0.35)  
plt.show()