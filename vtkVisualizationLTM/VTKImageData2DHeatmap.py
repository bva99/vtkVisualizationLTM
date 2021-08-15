import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtkVisualizationLTM.vtkVisualizationBase import VTKImageData2DBaseClass
from matplotlib.cm import ScalarMappable
import warnings

class VTKImageData2DHeatmap(VTKImageData2DBaseClass):
    """Class for plotting heatmaps of a 2D domain using the
    vtkImageData type

    The vtkImageData type is the simplest form of VTK data type,
    defining a regular 2D grid of equal cells. It also referred to as
    uniform rectilinear grid.
    """

    def __init__(self, *,
                 background_color=None, window_size=(1280, 720), window_name="Mesh Window",  # window properties
                 origin=None, domain_size=None, spacing=None, nels=None,  # ImageData properties:
                 n_cbar_colors=256, cmap_str="viridis", scalar_values=None):

        super().__init__(background_color=background_color, window_size=window_size,
                         window_name=window_name, origin=origin, domain_size=domain_size,
                         spacing=spacing, nels=nels)

        # the colors of the colormap
        cm = ScalarMappable(None, cmap_str).get_cmap()._resample(n_cbar_colors)
        colors = cm(np.linspace(0., 1., n_cbar_colors))
        self.colors = (colors*255).astype(np.ubyte)
        self.colorsVTK = numpy_to_vtk(self.colors, deep=0, array_type=vtk.VTK_UNSIGNED_CHAR)
        self.colorsVTK.SetName("lookupTableColors")

        # the scalar_values to be shown
        if scalar_values is None:
            raise ValueError(f"`scalar_values` should be given.")
        if (not isinstance(scalar_values, np.ndarray)) and (scalar_values.ndim != 1) \
                and (scalar_values.shape[0] not in [self.tot_nels, self.tot_points]):
            raise TypeError(f"`scalar_values` should be a 1D numpy.ndarray of " +
                            f"size {self.tot_nels} or {self.tot_points}, " +
                            f"not of type {type(scalar_values)} of size " +
                            f"{scalar_values.shape[0]} and {scalar_values.ndim} dimensions.")
        if isinstance(scalar_values, np.ndarray) and (scalar_values.dtype != np.float64):
            print(f"WARNING: scalar_values should be composed of 64 bit floats.")
            scalar_values = scalar_values.astype(np.float64)
        self.scalar_values = scalar_values

        # define if cell data or point data are given
        if self.scalar_values.shape[0] == self.tot_points:
            self._point_or_cell = "point"
        else:
            self._point_or_cell = "cell"

        # setup the lookup table
        self.lookup_table = vtk.vtkLookupTable()
        self.lookup_table.SetTable(self.colorsVTK)
        self.lookup_table.SetRange(self.scalar_values.min(), self.scalar_values.max())

        # vtk scalar scalar_values
        self.scalar_values_vtk = numpy_to_vtk(self.scalar_values, deep=0, array_type=vtk.VTK_DOUBLE)
        self.scalar_values_vtk.SetName("heatmapValues")

        # associate the scalar scalar_values to the image data
        if self._point_or_cell == "point":
            self.image_data.GetPointData().SetScalars(self.scalar_values_vtk)
        else:
            self.image_data.GetCellData().SetScalars(self.scalar_values_vtk)

        # create a filter object
        self.image_dataPolyData = vtk.vtkImageDataGeometryFilter()
        self.image_dataPolyData.SetInputData(self.image_data)

        # create a mapper and an actor
        self.image_dataMapper = vtk.vtkPolyDataMapper()
        self.image_dataMapper.SetInputConnection(self.image_dataPolyData.GetOutputPort())
        if self._point_or_cell == "point":
            self.image_dataMapper.SetScalarModeToUsePointData()
        else:
            self.image_dataMapper.SetScalarModeToUseCellData()
        self.image_dataMapper.SetLookupTable(self.lookup_table)
        self.image_dataMapper.UseLookupTableScalarRangeOn()

        self.actor = vtk.vtkActor()  # an actor represents an object in rendered scene
        self.actor.SetMapper(self.image_dataMapper)
        # Note that `actor.GetProperty()` returns a pointer to a vtkProperty object,
        # which in my machine enables setting/getting OpenGL rendering properties for
        # this actor
        # actor.GetProperty().EdgeVisibilityOn()
        # actor.GetProperty().SetEdgeColor(0., 0., 0.)
        # actor.GetProperty().SetLineWidth(10.)

        # create the colorbar
        self.colorBar = vtk.vtkScalarBarActor()
        self.colorBar.SetLookupTable(self.lookup_table)
        self.colorBar.GetLabelTextProperty().ItalicOff()
        self.colorBar.GetLabelTextProperty().BoldOff()
        self.colorBar.GetLabelTextProperty().SetColor([0, 0, 0])
        self.colorBar.SetMaximumNumberOfColors(256)
        self.colorBar.SetLabelFormat("%-#6.3g")
        # self.colorBar.SetUnconstrainedFontSize(1)
        # self.colorBar.GetLabelTextProperty().SetFontSize(30)

        # renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.AddActor(self.colorBar)
        self.renderer.SetBackground(self.background_color)
        self.renderer.SetBackgroundAlpha(0.0)
        self.renderer.SetUseDepthPeeling(1)  # only works with a current GPU
        self.renderer.SetOcclusionRatio(0.8)  # 0. is better, but much slower

        # render window
        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.SetWindowName(self.window_name)
        self.renderWindow.SetSize(self.window_size)
        self.renderWindow.Render()

        # get the camera
        self.camera = self.renderer.GetActiveCamera()

        # utility for camera translation
        self.camera_transform_mat = vtk.vtkTransform()  # initialized to identity matrix

        # utilities for saving to png
        self.windowToImage = vtk.vtkWindowToImageFilter()
        self.windowToImage.SetInput(self.renderWindow)
        # # if VTK_MAJOR_VERSION >= 8 || VTK_MAJOR_VERSION == 8 && VTK_MINOR_VERSION >= 90
        # windowToImageFilter.SetScale(2) # image quality
        # # else
        # windowToImageFilter.SetMagnification(2) # image quality
        # # endif
        self.windowToImage.SetInputBufferTypeToRGBA()  # record the alpha (transparency) channel
        self.pngWriter = vtk.vtkPNGWriter()
        self.pngWriter.SetInputConnection(self.windowToImage.GetOutputPort())

        # utilities for saving to XML
        self.vtiWriter = vtk.vtkXMLImageDataWriter()
        self.vtiWriter.SetInputDataObject(self.image_data)

        # utilities for interaction
        self.renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        self.renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()
        self.renderWindowInteractor.SetRenderWindow(self.renderWindow)
        self.renderWindowInteractor.Initialize()

    def update(self, scalar_values):
        if (not isinstance(scalar_values, np.ndarray)) and (scalar_values.ndim != 1) \
                and (scalar_values.shape[0] not in [self.tot_nels, self.tot_points]):
            raise TypeError(f"`scalar_values` should be a 1D numpy.ndarray of " +
                            f"size {self.tot_nels} or {self.tot_points}, " +
                            f"not of type {type(scalar_values)} of size " +
                            f"{scalar_values.shape[0]} and {scalar_values.ndim} dimensions.")
        if isinstance(scalar_values, np.ndarray) and (scalar_values.dtype != np.float64):
            print(f"WARNING: scalar_values should be composed of 64 bit floats.")
            scalar_values = scalar_values.astype(np.float64)
        if (self._point_or_cell == "point") and (scalar_values.shape[0] == self.tot_nels):
            warnings.warn(f"Changing from showing a point data heatmap to a cell data heatmap " +
                          f"is inefficient. Create a different visualization object.", RuntimeWarning)
            self._point_or_cell = "cell"
        elif (self._point_or_cell == "cell") and (scalar_values.shape[0] == self.tot_points):
            warnings.warn(f"Changing from showing a cell data heatmap to a point data heatmap " +
                          f"is inefficient. Create a different visualization object.", RuntimeWarning)
            self._point_or_cell = "point"
        self.scalar_values = scalar_values
        self.lookup_table.SetRange(self.scalar_values.min(), self.scalar_values.max())
        self.scalar_values_vtk = numpy_to_vtk(self.scalar_values, deep=0, array_type=vtk.VTK_DOUBLE)
        self.scalar_values_vtk.SetName("heatmapValues")
        if self._point_or_cell == "point":
            self.image_data.GetPointData().SetScalars(self.scalar_values_vtk)
        else:
            self.image_data.GetCellData().SetScalars(self.scalar_values_vtk)
        self.renderWindowInteractor.Render()
        self.renderWindowInteractor.ProcessEvents()

    def __repr__(self):
        rep = super().__repr__()
        rep = rep + f"\nClass {type(self)} for viewing heatmaps."
        return rep






