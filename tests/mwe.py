import numpy as np
from vtkVisualizationLTM import VTKImageData2DTopology, VTKImageData2DHeatmap
import time

def genAlphas(lenX, lenY, ncellsx, ncellsy, trans):
    cell_centers = np.column_stack((np.tile((lenX/ncellsx)*np.arange(0, ncellsx) + lenX/2/ncellsx, ncellsy),
                                    np.repeat((lenY/ncellsy)*np.arange(0, ncellsy) + lenY/2/ncellsy, ncellsx)))
    linFun = lambda x: 1. - (0.5 / lenX) * x
    alphas = np.sin((cell_centers[:, 0] - trans)*2*np.pi / lenX*3)*5*linFun(cell_centers[:, 0]) * \
             0.2*np.cos((cell_centers[:, 1])*2*np.pi / lenY*6)
    return alphas

def topology(n_iter=100, sleep_time=0.01):
    nels = [20, 10]
    design_domain = [1., 0.5]
    types_of_els = 3
    n_tot_els = nels[0]*nels[1]
    values = np.zeros(n_tot_els, dtype=np.int64)
    values[1::3] = 1
    values[2::3] = 2

    imageObject = VTKImageData2DTopology(nels=nels, number_of_vals=types_of_els,
                                         values=values, domain_size=design_domain)
    imageObject.zoom_camera(1.66)

    for i in range(n_iter):
        if (i % 3) == 0:
            values = np.ones(n_tot_els, dtype=np.int64)*2
            values[1::3] = 0
            values[2::3] = 1
        elif (i % 3) == 1:
            values = np.ones(n_tot_els, dtype=np.int64)
            values[1::3] = 2
            values[2::3] = 0
        else:
            values = np.zeros(n_tot_els, dtype=np.int64)
            values[1::3] = 1
            values[2::3] = 2

        imageObject.update(values)
        time.sleep(sleep_time)

def heatmap(n_iter=100, sleep_time=0.01):
    nels = [100, 50]
    design_domain = [1., 0.5]
    n_tot_els = nels[0]*nels[1]
    values = genAlphas(*design_domain, *nels, 0.0)

    imageObject = VTKImageData2DHeatmap(nels=nels, domain_size=design_domain, cmap_str="viridis", scalar_values=values)
    imageObject.zoom_camera(1.33)
    imageObject.translate_camera([0.15, 0.0, 0.0])  # the amount is done by trial and error

    for i in range(n_iter):
        values = genAlphas(*design_domain, *nels, i/100.)

        imageObject.update(values)
        time.sleep(sleep_time)

    imageObject.save_to_png(file_name="./heatmap.png")


if __name__ == "__main__":
    # topology(n_iter=100, sleep_time=0.05)
    print(f"Finished topology animation")
    heatmap(n_iter=200, sleep_time=0.001)
    print(f"Finished heatmap animation")

