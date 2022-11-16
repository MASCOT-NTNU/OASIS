# Note for rasterio operating geotiff file.

`dataset = rasterio.open('file.tif')`

To get crs: `dataset.crs`
To get bounds: `dataset.bounds`
To get transform: `dataset.transform`
To get index: `dataset.indexes`
To get img: `img = dataset.read(i)`

To visualize: `show(image)`
