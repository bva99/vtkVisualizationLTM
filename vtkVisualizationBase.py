import vtk

_global_colors_list = [[0.000, 0.447, 0.741],  # Matlab blue
                       [0.850, 0.325, 0.098],  # Matlab orange
                       [0.929, 0.694, 0.125],  # Matlab yellow
                       [0.494, 0.184, 0.556],  # Matlab purple
                       [0.466, 0.674, 0.188],  # Matlab green
                       [0.301, 0.745, 0.933],  # Matlab sky-blue
                       [0.635, 0.078, 0.184],  # Matlab terracotta-like color
                       [0.500, 0.500, 0.500]]  # Grey

class VTKImageData2DBaseClass:
    """Base class for plotting everything in 2D using the vtkImageData
    type.

    This class is an abstract class and returns an error if directly
    instantiated.
    """

    def __init__(self, *,
                 background_color=None, window_size=(1280, 720), window_name="Mesh Window",  # window properties
                 origin=None, domain_size=None, spacing=None, nels=None,  # ImageData properties
                 ):
        if type(self) == VTKImageData2DBaseClass:
            raise RuntimeError(f"VTKImageData2DBaseClass must be subclassed.")

        # Initialize a lookup_table attribute to None
        self.lookup_table = None

        # Window properties
        if background_color is None:
            background_color = [1, 1, 1]
        self.background_color = background_color
        self.window_size = window_size
        self.window_name = window_name

        # ImageData properties
        if origin is None:
            origin = [0., 0., 0.]
        self.origin = origin
        if domain_size is None:
            domain_size = [1., 1.]
        self.domain_size = domain_size
        if nels is None:
            nels = [10, 10]
        self.nels = nels
        if spacing is None:
            spacing = [self.domain_size[0]/self.nels[0],
                       self.domain_size[1]/self.nels[1],
                       0.0]
        self.spacing = spacing

        self.tot_nels = self.nels[0]*self.nels[1]

        self.tot_points = (self.nels[0] + 1)*(self.nels[1] + 1)

        # create the image data object
        self.image_data = vtk.vtkImageData()
        self.image_data.SetOrigin(self._origin)
        self.image_data.SetSpacing(*self._spacing)
        self.image_data.SetExtent(0, self.nels[0], 0, self.nels[1], 0, 0)

    def update(self, values):
        raise RuntimeError("This method should be reimplemented in the concrete class.")

    def zoom_camera(self, val):
        """
        Method to zoom camera.

        Sublcasses should have a `camera` attribute of the type `vtk.vtkCamera`
        and a `renderWindow` attribute of the type `vtk.vtkRenderWindow`.

        Arguments
        ---------
        val: float
            Value greater than 0. to zoom the `camera` in (`val>1.`) or out (`val<1.`).
        """
        self.camera.Zoom(val)
        self.renderWindow.Render()

    def translate_camera(self, translate_list):
        """
        Method to translate the camera.

        Sublcasses should have a `camera` attribute of the type `vtk.vtkCamera`,
        a `renderWindow` attribute of the type `vtk.vtkRenderWindow` and a
        `camera_transform_mat` attribute of the type `vtk.vtkTransform`.

        Arguments
        ---------
        translate_list: list
            List of 3 floats. How much to translate the camera in the x, y
            and z directions.
        """
        if (type(translate_list) != list) or (len(translate_list) != 3):
            raise ValueError(f"`translate_list` must be a list of size 3, not a " +
                             f"{type(translate_list)} of size {len(translate_list)}")
        self.camera_transform_mat.Translate(translate_list)
        self.camera.ApplyTransform(self.camera_transform_mat)
        self.renderWindow.Render()

    def show(self):
        """
        Method to start interaction with window.

        Sublcasses should have a `renderWindowInteractor` attribute of the type
        `vtk.vtkRenderWindowInteractor`.
        """
        self.renderWindowInteractor.Interact()

    def save_to_png(self, file_name="./mesh.png"):
        """
        Method to save a png image of the rendered window.

        Subclasses should have a `pngWriter` attribute of the type
        `vtk.vtkPNGWriter`.

        Arguments
        ---------
        file_name: str, default: "./mesh.vti"
            Name of the save file.
        """
        self.pngWriter.SetFileName(file_name)
        self.pngWriter.Write()
        self.windowToImage = vtk.vtkWindowToImageFilter()
        self.windowToImage.SetInput(self.renderWindow)
        self.windowToImage.SetInputBufferTypeToRGBA()  # record the alpha (transparency) channel
        self.pngWriter = vtk.vtkPNGWriter()
        self.pngWriter.SetInputConnection(self.windowToImage.GetOutputPort())

    def save_to_vtk(self, file_name="./mesh.vtk", binary=True):
        raise NotImplementedError(f"This method is not implemented yet. " +
                                  f"Use the method `.save_to_xml()` instead.")
        pass

    def save_to_xml(self, file_name="./mesh.vti", binary=True):
        """
        Method to write an XML image file of the ImageData.

        Sublcasses should have a `vtiWriter` attribute of the type
        `vtk.vtkXMLImageDataWriter`.

        Arguments
        ---------
        file_name: str, default: "./mesh.vti"
            Name of the save file.
        binary: bool, default: True
            Check whether to save as a binary (True) or an ASCII (False) file.
        """
        if binary:
            self.vtiWriter.SetDataModeToBinary()
        else:
            self.vtiWriter.SetDataModeToAscii()
        self.vtiWriter.SetFileName(file_name)
        self.vtiWriter.Write()

    def __repr__(self):
        rep = f"Topology VTK viewer object. Data in ImageData format.\n" + \
              f"Origin is {self.image_data.GetOrigin()}, " + \
              f"extent is {self.image_data.GetExtent()}, and \n" + \
              f"spacing is {self.image_data.GetSpacing()}"
        return rep

    @property
    def background_color(self):
        return self._background_color

    @background_color.setter
    def background_color(self, background_color):
        if len(background_color) != 3:
            raise ValueError(f"RGB background color should be a list of 3 floats between 0. and 1., " +
                             f"not {len(background_color)}")
        for val in background_color:
            if (not isinstance(val, (int, float))) and ((val < 0.0) or (val > 1.0)):
                raise ValueError(f"RGB background color values should be between 0. and 1.")
        self._background_color = list(background_color)

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, window_size):
        if len(window_size) != 2:
            raise ValueError(f"Window size should be a tuple of size 2, not size {len(window_size)}")
        self._window_size = tuple(window_size)

    @property
    def domain_size(self):
        return self._domain_size

    @domain_size.setter
    def domain_size(self, domain_size):
        if len(domain_size) != 2:
            raise ValueError(f"Domain size should be a list of size 2, not size {len(domain_size)}")
        self._domain_size = list(domain_size)

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origin):
        if len(origin) not in [2, 3]:
            raise ValueError(f"`origin` should be a list of 2 or 3 values, not {len(origin)}")
        if len(origin) == 2:
            origin = origin + [0.]
        origin[2] = 0.
        self._origin = origin

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, spacing):
        if len(spacing) not in [2, 3]:
            raise ValueError(f"`spacing` should be a list of 2 or 3 values, not {len(spacing)}")
        elif len(spacing) == 2:
            spacing = spacing + [0.]
        spacing[2] = 0.
        self._spacing = spacing

    @property
    def nels(self):
        return self._nels

    @nels.setter
    def nels(self, nels):
        if len(nels) != 2:
            raise ValueError(f"Number of elements should be a list of size 2, not {len(nels)}")
        for i in range(len(nels)):
            nels[i] = int(nels[i])
            if nels[i] < 0:
                raise ValueError(f"Number of elements should be an integer greater than 0, not {nels[i]}")
        self._nels = list(nels)


