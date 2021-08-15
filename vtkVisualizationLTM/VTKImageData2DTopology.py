import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtkVisualizationLTM.vtkVisualizationBase import VTKImageData2DBaseClass, _global_colors_list

class VTKImageData2DTopology(VTKImageData2DBaseClass):
    """Class for plotting the topology of a 2D domain using the
    vtkImageData type.

    The vtkImageData type is the simplest form of VTK data type,
    defining a regular 2D grid of equal cells. It also referred to as
    uniform rectilinear grid.

    Note that a "point" in VTK is equivalent to a Finite Element (FE) node
    and a "cell" in VTK is equivalent to a FE element.

    Attributes
    ----------
    background_color: list, default: [1., 1., 1.]
        List of three float values for the background color. Default is white.
    window_size: tuple, default: (1280, 720)
        Tuple of two integers containing the dimensions of the display window, in pixels.
    window_name: str, default: "Mesh Window"
        Name of the display window.
    origin: list, default: [0., 0., 0.]
        List of the coordinates of the origin of the mesh. The coordinates are from
        the view world. Index 0. is always set to 0.
    domain_size: list, default: [1., 1.]
        List of two floats, the dimensions of the design space.
    nels: list, default: [10, 10]
        List of two integers, the number of elements in the x and y directions, resp.
    spacing: list, optional
        List of three floats. `spacing[2]` is always 0. `spacing[0]` is the size
        of the cells in the x direction, and `spacing[1]` is the size of the
        cells in the y direction. The default is `spacing[i] = domain_size[i]/nels[i]`.
    number_of_vals: int, default: 2
        Number of different element color/opacity types, to be plotted.
    colors: list of lists
        List containing lists of RGB triplets, which should be floats between 0. and
        1. The number of colors should be equal to `number_of_vals`.
        If `number_of_vals <= 6`, then this values is optional and default colors
        are chosen.
    opacities: list, optional
        List containing the opacities of each type of element. The length of
        `opacities` should be equal to `number_of_vals`, corresponding to each
        list of `colors`. Defaults to all opaque colors (1.).
    values: np.ndarray, default: `np.zeros(number_of_vals, dtype=np.int64)`
        Numpy array of 64 bit integers containing the index numbers relating each
        element to a color. `values` should have a length equal to the total
        number of elements in the mesh.
    tot_nels: int
        Total number of elements.
    tot_points: int
        Total number of points.
    lookup_table: vtkLookupTable object
        A VTK lookup table. It is used by mapper objects to map scalar values (in
        this case, from `values`) to RGBA colors (in this case defined by `colors`
        and `opacities`).
    image_data: vtkImageData object
        This object is the VTK representation of the mesh.
    values_vtk: vtkArray object
        This object is the VTK array of `values`.
    image_dataPolyData: vtkPolyData object
        This object is a polygonal representation of `image_data`. This is what the
        GPU uses for rendering.
    image_dataMapper: vtkMapper object
        This object maps the polygonal data to the actor.
    actor: vtkActor object
        This object represents the polydata in a rendered scene.
    renderer: vtkRenderer object
        This object renders the actors to a scene.
    renderWindow: vtkRenderWindow object
        This object wraps several OS specific directives to generate a window.
    camera: vtkCamera object
        This object enables translations and rotations of the view camera.
    camera_transform_mat: vtkTransform object
        This object is a 4x4 matrix that allows for translations (implemented)
        and rotations (not implemented) of the view camera.
    windowToImage: vtkWindowToImageFilter object
        This object gets the image shown in a vtkWindow as input.
    pngWriter: vtkPNGWriter object
        This object enables saving of the rendered scene to a png file.
        For now only transparent backgrounds are implemented. But this might
        require current GPUs, so it might become optional later on.
    vtiWriter: vtkXMLImageDataWriter object
        This object writes `image_data` as a .vti file.
    renderWindowInteractor: vtkRenderWindowInteractor object
        This object allows for control of the render window.
    """

    def __init__(self, *,
                 background_color=None, window_size=(1280, 720), window_name="Mesh Window",  # window properties
                 origin=None, domain_size=None, spacing=None, nels=None,  # ImageData properties
                 number_of_vals=2, opacities=None, colors=None, values=None):  # values for topology visualization

        super().__init__(background_color=background_color, window_size=window_size,
                         window_name=window_name, origin=origin, domain_size=domain_size,
                         spacing=spacing, nels=nels)

        self.number_of_vals = number_of_vals

        if opacities is None:
            opacities = [1.] * number_of_vals
        self.opacities = opacities

        if (colors is None) and (number_of_vals < 7):
            colors = _global_colors_list[0:number_of_vals].copy()
        elif (colors is None) and (number_of_vals >= 7):
            raise ValueError(f"Since `number_of_vals` is greater than 7, a valid list of" +
                             f"character RGB triplets must be given.")
        self.colors = colors

        if values is None:
            values = np.zeros(self.tot_nels, dtype=np.int64)
        if (not isinstance(values, np.ndarray)) and (values.ndim != 1) and (values.shape[0] != self.tot_nels):
            raise TypeError(f"`values` should be a 1D numpy.ndarray of size {self.tot_nels}, " +
                            f"not of type {type(values)} of size {values.shape[0]} and {values.ndim} dimensions.")
        if isinstance(values, np.ndarray) and (values.dtype != np.int64):
            print(f"WARNING: values should be composed of 64 bit integers.")
            values = values.astype(np.int64)
        self.values = values

        # lookup table to associate colors with cell types
        self.lookup_table = vtk.vtkLookupTable()
        self.lookup_table.SetNumberOfTableValues(self.number_of_vals)
        for i in range(self.number_of_vals):
            self.lookup_table.SetTableValue(i, *self.colors[i], self.opacities[i])
        self.lookup_table.SetRange(0, self.number_of_vals-1)  # this is important to map the colors correctly
        self.lookup_table.GetTable().SetName("colorTableArray")

        # vtk array of values
        self.values_vtk = numpy_to_vtk(self.values, deep=0, array_type=vtk.VTK_LONG_LONG)
        self.values_vtk.SetName("colorClassificationArray")
        # CHANGED for mapper
        # self.values_vtk.SetLookupTable(self.lookup_table)  # use lut as the lookup table

        # associate values to cell data of the mesh
        self.image_data.GetCellData().SetScalars(self.values_vtk)
        self.image_data.GetFieldData().AddArray(self.lookup_table.GetTable())

        # create a polygonal dataset
        self.image_dataPolyData = vtk.vtkImageDataGeometryFilter()
        self.image_dataPolyData.SetInputData(self.image_data)

        # create the mapper
        self.image_dataMapper = vtk.vtkPolyDataMapper()
        self.image_dataMapper.SetInputConnection(self.image_dataPolyData.GetOutputPort())
        self.image_dataMapper.SetScalarModeToUseCellData()
        self.image_dataMapper.SetLookupTable(self.lookup_table)
        self.image_dataMapper.UseLookupTableScalarRangeOn()  # super important, for the table scaling to work

        # create the actor
        self.actor = vtk.vtkActor()  # an actor represents an object in a rendered scene
        self.actor.SetMapper(self.image_dataMapper)

        # visualization
        self.renderer = vtk.vtkRenderer()
        self.renderer.AddActor(self.actor)
        self.renderer.SetBackground(self.background_color)
        self.renderer.SetBackgroundAlpha(0.0)
        self.renderer.SetUseDepthPeeling(1)  # only works with a current GPU
        self.renderer.SetOcclusionRatio(0.8)  # 0. is better, but much slower

        self.renderWindow = vtk.vtkRenderWindow()
        self.renderWindow.AddRenderer(self.renderer)
        self.renderWindow.SetWindowName(self.window_name)
        self.renderWindow.SetSize(self.window_size)
        self.renderWindow.AlphaBitPlanesOn()  # enable usage of alpha channel
        self.renderWindow.ShowCursor()
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

    def update(self, values):
        """Updates the rendered scene

        Arguments
        ---------
        values: np.ndarray
            Check the class docstring.
        """
        if (not isinstance(values, np.ndarray)) and (values.ndim != 1) and (values.shape[0] != self.tot_nels):
            raise TypeError(f"`values` should be a 1D numpy.ndarray of size {self.tot_nels}, " +
                            f"not of type {type(values)} of size {values.shape[0]} and {values.ndim} dimensions.")
        if isinstance(values, np.ndarray) and (values.dtype != np.int64):
            print(f"WARNING: `values` should be composed of 64 bit integers.")
            values = values.astype(np.int64)
        self.values = values
        self.values_vtk = numpy_to_vtk(self.values, deep=0, array_type=vtk.VTK_LONG_LONG)
        self.values_vtk.SetName("colorClassificationArray")
        self.image_data.GetCellData().SetScalars(self.values_vtk)
        self.renderWindowInteractor.Render()
        self.renderWindowInteractor.ProcessEvents()

    def __repr__(self):
        rep = super().__repr__()
        rep = rep + f"\nClass {type(self)} for viewing mesh topologies."
        return rep

    @property
    def number_of_vals(self):
        return self._number_of_vals

    @number_of_vals.setter
    def number_of_vals(self, number_of_vals):
        if (not isinstance(number_of_vals, (int, float))) and (number_of_vals < 1):
            raise TypeError(f"`number_of_vals` should be an integer greater than 0, not {number_of_vals}.")
        self._number_of_vals = int(number_of_vals)

    @property
    def opacities(self):
        return self._opacities

    @opacities.setter
    def opacities(self, opacities):
        if len(opacities) != self.number_of_vals:
            raise ValueError(f"Length of `opacities` ({len(opacities)}) should be equal to" +
                             f"`number_of_vals` ({self.number_of_vals}).")
        for val in opacities:
            if (not isinstance(val, (int, float))) and ((val < 0.0) or (val > 1.0)):
                raise ValueError(f"Cell opacity values should be between 0. and 1.")
        self._opacities = list(opacities)

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, colors):
        if len(colors) != self.number_of_vals:
            raise ValueError(f"Length of `colors` ({len(colors)}) should be equal to" +
                             f"`number_of_vals` ({self.number_of_vals}).")
        for color in colors:
            for val in color:
                if (not isinstance(val, float)) and ((val < 0.) or (val > 1.)):
                    raise ValueError(f"Colors should be floats between 0 and 255.")
        self._colors = list(colors)
