# vtkVisualizationLTM

Package with basic visualization tools for rendering Finite Element related figures efficiently using VTK.

The package was developed for the needs of the [Laboratory of Topology Optimization and Multiphysics Analysis](https://www.fem.unicamp.br/~ltm/) (LTM) group.

## Description

The package contains high level classes that wrap several verbose VTK functionalities for general visualization needed in Topology Optimization and is much faster than `matplotlib` visualization functions such as `imshow` or `patches`, since it leverages GPU visualization capabilities of OpenGL availabe through the VTK package.

For now the package contains:
- A class for plotting and updating 2D topologies of uniform rectilinear rectangular meshes.
- A class for plotting and updating 2D heatmaps of uniform rectilinear rectangular meshes.

The classes all inherit from a 2D visualization base class that contains general methods such as:
- saving to png with transparent background
- saving to the old VTK format ".vtk"
- saving to the newer XML VTK format as ".vti"
- translating the camera
- zooming the camera in or out

Since all VTK objects required for visualization are attributes of the instanced objects, all VTK methods may be used directly, if one is well-versed in VTK.

## Getting started

### Dependencies
- Python
- Numpy
- VTK
- Matplotlib (for colormaps)
- A "current" GPU. Due to the way transparent backgrounds for PNG images are implemented. This may be changed in future versions.

### Installing
- Copy the vtkVisualizationLTM folder to your working directory.
- To run the test, copy the `/tests/test.py` file (just the file, **not** the `tests` folder) to your working directory.

### Running
- Be sure your file structure looks as follows:
```
.
├── ...
├── /your modules, scripts and packages/
├── ...
├── test.py
└── vtkVisualizationLTM
    ├── __init__.py
    ├── VTKVisualizationBase.py
    ├── VTKImageData2DTopology.py
    └── VTKImageData2DHeatmap.py
```
- Run the `test.py` file as a script.

## Help

All classes and methods in the implementation are well documented.

If unimplemented functionalities are required, it is possible to use the underlying VTK methods directly. Reading the [VTK Doxygen](https://vtk.org/doc/nightly/html/index.html) is recommended in order to understand what these functionalities are.

Further questions may be sent to `balmeida /at/ fem.unicamp.br`

## Authors

Breno Vincenzo de Almeida [bva99](https://github.com/bva99)

## Version history

- 0.1
  - Initial release.
  - Contains only classes for 2D visualization (topology and heatmaps).

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgments

- [VTK support](https://discourse.vtk.org/c/support/6)
- [VTK Doxygen](https://vtk.org/doc/nightly/html/index.html)
- Schroeder, Will; Martin, Ken; Lorensen, Bill (2006), [The Visualization Toolkit](https://vtk.org/documentation/) (4th ed.), Kitware, ISBN 978-1-930934-19-1
